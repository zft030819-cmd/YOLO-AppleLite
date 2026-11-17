"""Ultralytics AP for paper training tables; COCO area analysis remains evaluate.py."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--imgsz',type=int,default=640);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    if not a.weights.is_file() or not a.data.is_file():p.error('Weights/data not found')
    sys.path.insert(0,str(ROOT));os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime'))
    from ultralytics import YOLO
    m=YOLO(str(a.weights.resolve()));v=m.val(data=str(a.data.resolve()),split=a.split,imgsz=a.imgsz,batch=16,device=a.device,workers=0,conf=.001,iou=.7,max_det=300,half=False,plots=False,project=str(a.output.parent.resolve()),name=a.output.name,exist_ok=False)
    a.output.mkdir(parents=True,exist_ok=True)
    report={'metric_protocol':'Ultralytics 8.3.63 DetectionValidator','split':a.split,'imgsz':a.imgsz,'conf':.001,'iou':.7,'max_det':300,'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'metrics':{k:float(x) for k,x in v.results_dict.items()}}
    (a.output/'paper_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
