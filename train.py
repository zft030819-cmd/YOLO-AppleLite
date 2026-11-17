"""Portable training entry point for YOLOv12n and YOLO-AppleLite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', choices=['yolov12n', 'applelite'], default='applelite')
    parser.add_argument('--config', type=Path, help='Optional ablation architecture YAML')
    parser.add_argument('--loss', choices=['ciou', 'nwd', 'mpdiou'], default='ciou')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', type=Path, default=ROOT/'runs/train')
    parser.add_argument('--name', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('--data must point to an existing dataset YAML')
    config = args.config.resolve() if args.config else ROOT/'configs/models'/f'{args.model}.yaml'
    if not config.is_file():
        parser.error('Model architecture YAML does not exist')
    os.environ.update(USE_NWD=str(args.loss=='nwd'), BBOX_LOSS_TYPE='mpdiou' if args.loss=='mpdiou' else 'ciou',
                      NWD_WEIGHT='0.5' if args.loss=='nwd' else '0.0', NWD_CONSTANT='12.8')
    from ultralytics import YOLO
    settings = yaml.safe_load((ROOT/'configs/train.yaml').read_text(encoding='utf-8'))
    settings.update(data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch, device=args.device,
                    workers=args.workers, project=str(args.project.resolve()),
                    name=args.name or f'{args.model}_{args.loss}_{datetime.now():%Y%m%d_%H%M%S}')
    model = YOLO(str(config))
    parameters = sum(p.numel() for p in model.model.parameters())
    plan = dict(model=args.model, loss=args.loss, parameters=parameters, settings=settings,
                architecture_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                nwd_weight=.5 if args.loss=='nwd' else 0., nwd_constant_pixels=12.8)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    destination = args.project.resolve()/settings['name']
    if destination.exists():
        parser.error(f'Run directory already exists: {destination}')
    model.train(**settings)
    saved = Path(model.trainer.save_dir)
    (saved/'experiment.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
