"""Re-evaluate archived NCNN models with square preprocessing and Ultralytics AP."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=['pair640','six_experiments'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='Smoke-test prefix; nonzero never counts as complete evaluation')
    a=p.parse_args()
    if a.limit<0:p.error('limit must be nonnegative')
    if a.output.exists():p.error('Output exists; historical records are never overwritten')
    sys.path[:0]=[str(ROOT),str(ROOT/'deployment'/a.experiment)];os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime'))
    import cv2,numpy as np,torch,yaml
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics
    from ultralytics.utils.ops import non_max_suppression
    from lab import Backend,preprocess
    cv2.setNumThreads(1);torch.set_num_threads(4)
    cfg=yaml.safe_load(a.data.read_text(encoding='utf8'));test=Path(cfg['test']);test=test if test.is_absolute() else a.data.parent/test
    paths=[Path(x) for x in test.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    paths=[x if x.is_absolute() else test.parent/x for x in paths]
    if len(paths)!=356:raise ValueError('Frozen test set must have 356 images')
    frozen=json.loads((ROOT/'data/split_manifest.json').read_text());expected={x['image_sha256'] for x in frozen if x['split']=='test'};label_by_image={x['image_sha256']:x['label_sha256'] for x in frozen if x['split']=='test'}
    if {hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}!=expected:raise ValueError('Test image identities differ')
    if a.limit:paths=paths[:a.limit]
    base=ROOT/'deployment'/a.experiment;config=json.loads((base/'models.json').read_text())['models'];results=[]
    for item in config:
        b=Backend(item,base,4);validator=object.__new__(DetectionValidator);validator.iouv=torch.linspace(.5,.95,10);stats=[]
        for path in paths:
            image=cv2.imread(str(path));h,w=image.shape[:2];x,ratio,left,top=preprocess(image,item['size'])
            parts=list(path.parts);pos=max(i for i,v in enumerate(parts) if v=='images');parts[pos]='labels';lp=Path(*parts).with_suffix('.txt')
            if hashlib.sha256(lp.read_bytes()).hexdigest()!=label_by_image[hashlib.sha256(path.read_bytes()).hexdigest()]:raise ValueError('Test label differs')
            labels=np.loadtxt(lp,dtype=np.float32,ndmin=2);gt=labels[:,1:].copy();gt[:,[0,2]]*=w*ratio;gt[:,[1,3]]*=h*ratio;gt[:,:2]-=gt[:,2:]/2;gt[:,2:]+=gt[:,:2];gt[:,[0,2]]+=left;gt[:,[1,3]]+=top
            raw=torch.from_numpy(b.run(x)[None]);pred=non_max_suppression(raw,conf_thres=.001,iou_thres=.7,max_det=300)[0]
            match=validator._process_batch(pred,torch.from_numpy(gt),torch.from_numpy(labels[:,0]));stats.append((match.numpy(),pred[:,4].numpy(),pred[:,5].numpy(),labels[:,0]))
        arrays=[np.concatenate(v,axis=0) for v in zip(*stats)];m=DetMetrics(names=dict(enumerate(b.names)));m.process(*arrays)
        results.append({'model':item['name'],'size':item['size'],'metrics':{k:float(v) for k,v in m.results_dict.items()},'model_hashes':b.hashes})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'protocol':'square NCNN / Ultralytics AP','images':len(paths),'complete':a.limit==0,'scope':'independent full test' if a.limit==0 else 'SMOKE TEST ONLY','models':results},indent=2),encoding='utf8');print(a.output)
if __name__=='__main__':main()
