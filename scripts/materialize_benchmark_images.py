"""Restore the fixed 80 test images without distributing image pixels in the release."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
    for group in ['pair640','six_experiments']:
        base=ROOT/'deployment'/group;rows=json.loads((base/'images.json').read_text());plan=[]
        for r in rows:
            src=(a.data_root/r['dataset_image']).resolve();dst=(base/r['file']).resolve()
            if not src.is_relative_to(a.data_root.resolve()) or not dst.is_relative_to(base.resolve()):raise ValueError('Unsafe path')
            if not src.is_file() or sha(src)!=r['sha256']:raise ValueError('Image mismatch: '+str(src))
            if dst.exists() and sha(dst)!=r['sha256']:raise ValueError('Refusing to overwrite changed benchmark image')
            plan.append((src,dst))
        for src,dst in plan:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        print(group,len(plan),'images restored')
if __name__=='__main__':main()
