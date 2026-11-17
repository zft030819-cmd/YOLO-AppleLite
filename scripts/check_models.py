"""Check final model parsing, parameter counts and finite forward outputs."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime'))
import torch
from ultralytics import YOLO


def main():
    torch.set_num_threads(2)
    for name,expected in [('yolov12n',2521224),('applelite',927032)]:
        model=YOLO(str(ROOT/'configs/models'/f'{name}.yaml')).model.cpu().eval()
        count=sum(p.numel() for p in model.parameters())
        assert count==expected,(name,count,expected)
        with torch.no_grad():output=model(torch.zeros(1,3,640,640))[0]
        assert tuple(output.shape)==(1,12,8400)
        assert torch.isfinite(output).all()
        print(f'{name}: {count} parameters; output {tuple(output.shape)}; OK')


if __name__=='__main__':main()
