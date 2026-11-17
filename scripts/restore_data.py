"""Restore exact final labels and frozen split lists; never modify downloaded originals."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(root,rel):
    rel=Path(rel)
    if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):raise ValueError('Unsafe relative path: '+str(rel))
    p=(root/rel).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path leaves root')
    return p

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--images-root',type=Path,required=True)
    a.add_argument('--output',type=Path)
    a.add_argument('--check-only',action='store_true')
    a.add_argument('--manifest',type=Path,default=ROOT/'data/split_manifest.json')
    a.add_argument('--labels',type=Path,default=ROOT/'data/processed_labels')
    args=a.parse_args()
    if not args.check_only and args.output is None:a.error('--output required unless --check-only')
    if args.output and args.output.exists():a.error('Output already exists; choose a new empty path')
    if not args.images_root.is_dir():a.error('Image root not found')
    if args.output and (args.output.resolve().is_relative_to(args.images_root.resolve()) or args.images_root.resolve().is_relative_to(args.output.resolve())):a.error('Source and output trees must be separate')
    records=json.loads(args.manifest.read_text(encoding='utf8'));byname=None;plan=[];groups=defaultdict(set);seen=set()
    for r in records:
        if r['split'] not in ['train','val','test']:raise ValueError('Invalid split')
        if r['image'] in seen:raise ValueError('Repeated image identity')
        seen.add(r['image']);groups[r['group']].add(r['split'])
        image=safe(args.images_root,r['image']);label=safe(args.labels,r['label'])
        if not image.is_file() or sha(image)!=r['image_sha256']:
            if byname is None:
                byname=defaultdict(list)
                for f in args.images_root.rglob('*'):
                    if f.is_file() and f.suffix.lower() in ['.jpg','.jpeg','.png','.webp','.bmp']:byname[f.name].append(f)
            matches=[f for f in byname[Path(r['image']).name] if sha(f)==r['image_sha256']]
            if not matches:raise ValueError('Matching original image not found: '+r['image'])
            image=matches[0]
        if not label.is_file() or sha(label)!=r['label_sha256']:raise ValueError('Final label differs: '+r['label'])
        plan.append((r,image,label))
    if any(len(v)>1 for v in groups.values()):raise ValueError('Source group crosses splits')
    summary={'images':dict(Counter(r['split'] for r,_,_ in plan)),'groups':len(groups),'sha256_checked':len(plan),'mode':'check_only' if args.check_only else 'restored'}
    if not args.check_only:
        out=args.output.resolve();out.mkdir(parents=True);lists=defaultdict(list)
        for r,image,label in plan:
            dest=safe(out,r['image']);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(image,dest)
            target=safe(out,r['label']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(label,target)
            lists[r['split']].append(str(dest))
        cfg={'path':str(out),'nc':8,'names':json.loads((ROOT/'data/classes.json').read_text())}
        for split,paths in lists.items():
            f=out/(split+'.txt');f.write_text('\n'.join(paths)+'\n',encoding='utf8');cfg[split]=str(f)
        (out/'data.yaml').write_text(json.dumps(cfg,indent=2),encoding='utf8')
        (out/'restoration_report.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
