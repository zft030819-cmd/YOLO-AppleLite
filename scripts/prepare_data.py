"""Restore the fixed, group-separated lists from a user-supplied dataset copy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Original dataset root containing train/valid/test')
    parser.add_argument('--output',type=Path,default=ROOT/'prepared_data')
    args=parser.parse_args()
    dataset=args.root.resolve()
    entries=json.loads((ROOT/'data/split_manifest.json').read_text(encoding='utf-8'))
    lists={s:[] for s in ['train','val','test']}
    groups={s:set() for s in lists}
    for row in entries:
        image=dataset/row['image'];label=dataset/row['label']
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f'Missing image or label: {row["image"]}')
        if sha(image)!=row['image_sha256'] or sha(label)!=row['label_sha256']:
            raise ValueError(f'Dataset content differs: {row["image"]}')
        lists[row['split']].append(str(image))
        groups[row['split']].add(row['group'])
    assert not groups['train']&(groups['val']|groups['test'])
    assert not groups['val']&groups['test']
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    data={'path':str(dataset),'nc':8,'names':json.loads((ROOT/'data/classes.json').read_text())}
    for split,images in lists.items():
        path=output/f'{split}.txt'
        path.write_text('\n'.join(images)+'\n',encoding='utf-8')
        data[split]=str(path)
    (output/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(json.dumps({'data':str(output/'data.yaml'),'images':{s:len(x) for s,x in lists.items()}},ensure_ascii=False))


if __name__=='__main__':main()
