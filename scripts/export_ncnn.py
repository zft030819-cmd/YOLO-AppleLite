"""Portable native/unrolled FP32 NCNN export and training-only mixed INT8 calibration."""
import argparse,hashlib,json,os,shutil,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,choices=[320,416,512,640],required=True);p.add_argument('--ema',choices=['native','unroll'],required=True)
    p.add_argument('--pnnx',type=Path,required=True);p.add_argument('--mixed-int8',action='store_true');p.add_argument('--data-root',type=Path)
    p.add_argument('--ncnn2table',type=Path);p.add_argument('--ncnn2int8',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; use a fresh export directory')
    if not a.weights.is_file() or not a.pnnx.is_file():p.error('Checkpoint or PNNX executable missing')
    if a.mixed_int8 and (a.ema!='unroll' or not a.data_root or not a.ncnn2table or not a.ncnn2int8):p.error('Mixed INT8 requires unrolled EMA, data-root, ncnn2table and ncnn2int8')
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')];os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime'))
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from ema_export import ema_forward
    from raspberry_pi.inference import preprocess
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(0)
    out=a.output.resolve();out.mkdir(parents=True);build=out/'build';build.mkdir();checkpoint=build/'source.pt';shutil.copyfile(a.weights,checkpoint)
    model=YOLO(str(checkpoint));model.model.cpu().float().eval();x=torch.rand(1,3,a.size,a.size)
    with torch.no_grad():
        reference=model.model(x)[0].clone()
        if a.ema=='unroll':
            for module in model.model.modules():
                if type(module).__name__=='EMA':module.forward=types.MethodType(ema_forward,module)
        actual=model.model(x)[0];torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-3)
    fp=out/'fp32';fp.mkdir();script=Path(model.export(format='torchscript',imgsz=a.size,batch=1,device='cpu'))
    cmd=[str(a.pnnx.resolve()),str(script.resolve()),f'inputshape=[1,3,{a.size},{a.size}]','fp16=0',f'ncnnparam={fp/"model.ncnn.param"}',f'ncnnbin={fp/"model.ncnn.bin"}',f'ncnnpy={build/"model_ncnn.py"}']
    with (build/'pnnx.log').open('w') as log:subprocess.run(cmd,cwd=build,stdout=log,stderr=log,check=True)
    metadata={'names':model.names,'imgsz':[a.size,a.size],'stride':32,'task':'detect','batch':1};(fp/'metadata.yaml').write_text(json.dumps(metadata),encoding='utf8')
    report={'weights_sha256':sha(a.weights),'size':a.size,'ema':a.ema,'ema_parity_max_abs_error':float((actual-reference).abs().max()),'pnnx_sha256':sha(a.pnnx),'pnnx_command':cmd,'fp32':{f.name:sha(f) for f in fp.glob('model.ncnn.*')},'ncnn_numerical_parity_measured':False}
    if a.mixed_int8:
        records=json.loads((ROOT/'data/calibration_manifest.json').read_text());frozen=json.loads((ROOT/'data/split_manifest.json').read_text());train={r['image_sha256'] for r in frozen if r['split']=='train'}
        if len(records)!=512 or len({r['image_sha256'] for r in records})!=512 or not {r['image_sha256'] for r in records}<=train:raise ValueError('Calibration must use the fixed 512 training images')
        cal=build/'calibration';cal.mkdir();paths=[]
        for i,r in enumerate(records):
            src=(a.data_root/r['image']).resolve()
            if not src.is_relative_to(a.data_root.resolve()) or sha(src)!=r['image_sha256']:raise ValueError('Calibration input changed')
            tensor=preprocess(cv2.imread(str(src)),a.size)[0];bgr=(tensor.transpose(1,2,0)[:,:,::-1]*255).round().astype('uint8');dest=cal/f'{i:04d}.png'
            if not cv2.imwrite(str(dest),bgr):raise RuntimeError('Calibration write failed')
            paths.append(str(dest))
        listing=build/'calibration.txt';listing.write_text('\n'.join(paths)+'\n',encoding='utf8');table=build/'calibration.table'
        cmd=[str(a.ncnn2table.resolve()),str(fp/'model.ncnn.param'),str(fp/'model.ncnn.bin'),str(listing),str(table),'mean=[0,0,0]','norm=[0.003921568627,0.003921568627,0.003921568627]',f'shape=[{a.size},{a.size},3]','pixel=RGB','thread=4','method=kl']
        with (build/'calibration.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
        original=(fp/'model.ncnn.param').read_text().splitlines();allowed=set();excluded=[]
        for line in original[2:]:
            t=line.split()
            if t[0] not in ['Convolution','ConvolutionDepthWise']:continue
            params=dict(v.split('=',1) for v in t[4+int(t[2])+int(t[3]):])
            if t[0]=='Convolution' or params.get('7')==params.get('0'):allowed.add(t[1])
            else:excluded.append(t[1])
        mixed=build/'mixed.table';mixed.write_text('\n'.join(line for line in table.read_text().splitlines() if line.split()[0].removesuffix('_param_0') in allowed)+'\n')
        quant=out/'mixed_int8';quant.mkdir();cmd=[str(a.ncnn2int8.resolve()),str(fp/'model.ncnn.param'),str(fp/'model.ncnn.bin'),str(quant/'model.ncnn.param'),str(quant/'model.ncnn.bin'),str(mixed)]
        with (build/'quantization.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
        definitions={line.split()[1]:line for line in original[2:]};lines=(quant/'model.ncnn.param').read_text().splitlines()
        for i,line in enumerate(lines[2:],2):
            t=line.split()
            if t[0] not in ['Convolution','ConvolutionDepthWise']:lines[i]=definitions[t[1]]
        (quant/'model.ncnn.param').write_text('\n'.join(lines)+'\n');shutil.copyfile(fp/'metadata.yaml',quant/'metadata.yaml')
        report['quantization']={'calibration_images':512,'method':'KL','grouped_convolutions_retained_float':excluded,'ncnn2table_sha256':sha(a.ncnn2table),'ncnn2int8_sha256':sha(a.ncnn2int8),'sha256':{f.name:sha(f) for f in quant.glob('model.ncnn.*')}}
    (out/'export_report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(out/'export_report.json')
if __name__=='__main__':main()
