"""Area-aware COCO evaluation on val, test, or an explicitly exploratory merge."""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime'))
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from area_metrics import coco_eval,summarize,digest


def files(value):
    result=[]
    for entry in value if isinstance(value,list) else [value]:
        path=Path(entry)
        if path.is_dir():
            result.extend(sorted(p.resolve() for p in path.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}))
        elif path.suffix=='.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    p=Path(line.strip())
                    result.append(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
        else:raise ValueError(f'Expected image directory or image-list TXT: {path}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test','merged'],default='test')
    parser.add_argument('--device',default='0')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if not args.weights.is_file():parser.error('Checkpoint does not exist')
    output=args.output or ROOT/'runs/eval'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    config=check_det_dataset(str(args.data.resolve()),autodownload=False)
    names=config['names'];names=dict(enumerate(names)) if isinstance(names,list) else names
    expected=['Brown_Spot','Frog_Eye_Leaf_Spot','Grey_Spot','Healthy','Mosaic','Powdery_Mildew','Rust','Scab']
    assert [names[i] for i in range(8)]==expected,'Class mapping must match the eight-class model'
    splits=['val','test'] if args.split=='merged' else [args.split]
    paths=[p for split in splits for p in files(config[split])]
    assert len(paths)==len(set(paths)),'Repeated images in evaluation lists'
    assert not set(paths)&set(files(config['train'])),'Evaluation images overlap the training list'
    data=dict(info={'description':'Scaled640 area evaluation'},images=[],annotations=[],
              categories=[dict(id=i,name=names[i]) for i in names])
    lookup={}
    for iid,path in enumerate(paths,1):
        with Image.open(path) as image:w,h=image.size
        gain=640/max(w,h)
        data['images'].append(dict(id=iid,width=w*gain,height=h*gain))
        lookup[str(path).casefold()]=(iid,gain)
        parts=list(path.parts)
        positions=[i for i,x in enumerate(parts) if x=='images']
        if not positions:raise ValueError('YOLO image paths must include an images directory')
        parts[positions[-1]]='labels';label=Path(*parts).with_suffix('.txt')
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            c,x,y,bw,bh=map(float,line.split());bw*=w*gain;bh*=h*gain
            data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=iid,category_id=int(c),
                bbox=[x*w*gain-bw/2,y*h*gain-bh/2,bw,bh],area=bw*bh,iscrowd=0))
    predictions=[]
    class Capture(DetectionValidator):
        def update_metrics(self,preds,batch):
            for i,pred in enumerate(preds):
                prepared=self._prepare_batch(i,batch)
                actual=self._prepare_pred(pred,prepared).detach().cpu().numpy()
                iid,gain=lookup[str(Path(batch['im_file'][i]).resolve()).casefold()]
                for x1,y1,x2,y2,score,cls in actual:
                    predictions.append(dict(image_id=iid,category_id=int(cls),score=float(score),
                        bbox=[float(x1*gain),float(y1*gain),float((x2-x1)*gain),float((y2-y1)*gain)]))
            super().update_metrics(preds,batch)
    model=YOLO(str(args.weights.resolve()))
    assert [model.names[i] for i in range(8)]==expected
    for split in splits:
        model.val(validator=Capture,data=str(args.data.resolve()),split=split,imgsz=640,batch=16,
                  device=args.device,workers=0,conf=.001,iou=.7,max_det=300,half=False,
                  plots=False,project=str(output),name=split,exist_ok=True)
        empty=output/split
        if empty.is_dir() and not any(empty.iterdir()):empty.rmdir()
    metrics=summarize(data,coco_eval(data,predictions),coco_eval(data,[p for p in predictions if p['score']>=.25]))
    report=dict(evaluation_scope='val+test exploratory' if args.split=='merged' else args.split,
                independent_test=args.split=='test',selection_set_included=args.split!='test',
                checkpoint_sha256=digest(args.weights),images=len(paths),metrics=metrics,
                settings=dict(imgsz=640,conf_ap=.001,conf_fixed=.25,nms_iou=.7,max_det=300,
                              ap_method='COCO 101-point',tiny_area='scaled640 area < 256'))
    for name,value in [('metrics.json',report),('predictions.json',predictions)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'tiny':metrics['tiny'],'small':metrics['small']},ensure_ascii=False))


if __name__=='__main__':main()
