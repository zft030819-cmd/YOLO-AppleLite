"""Verify public file integrity and absence of historical experimental results."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    forbidden_dirs={'results','validation','provenance','release_assets','private'}
    forbidden_keys={'metrics','val_metrics','per_class','backends','accuracy_reference','max_output_error','max_score_error','mean_inference_ms','p95_inference_ms','inference_fps'}
    def inspect(x,path):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in forbidden_keys:raise ValueError('Private result field in '+str(path)+': '+k)
                inspect(v,path)
        elif isinstance(x,list):
            for v in x:inspect(v,path)
    manifest=json.loads((ROOT/'release_manifest.json').read_text(encoding='utf8'))
    for r in manifest['files']:
        p=ROOT/r['path'];rel=p.relative_to(ROOT)
        if set(rel.parts)&forbidden_dirs or p.suffix.lower() in {'.pt','.pth','.csv','.npz','.jsonl'}:raise ValueError('Private artifact in public manifest: '+r['path'])
        if not p.is_file() or sha(p)!=r['sha256']:raise ValueError('File changed: '+r['path'])
        if p.suffix=='.json':inspect(json.loads(p.read_text(encoding='utf-8-sig')),rel)
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(rel))
    for group in ['pair640','six_experiments']:
        base=ROOT/'deployment'/group;provs=json.loads((base/'provenance.json').read_text())['models'];config=json.loads((base/'models.json').read_text())['models']
        for item,record in zip(config,provs):
            for name,digest in record.get('ncnn_sha256',record.get('sha256')).items():
                if sha(base/item['path']/name)!=digest:raise ValueError('Deployment model changed')
    print(json.dumps({'public_integrity_passed':True,'historical_results_included':False,'training_checkpoints_included':False,'verified_files':len(manifest['files'])},indent=2))
if __name__=='__main__':main()
