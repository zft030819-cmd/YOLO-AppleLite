import json
import hashlib
import contextlib
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def coco_eval(data, predictions):
    gt = COCO()
    gt.dataset = data
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.maxDets = [1, 10, 300]
        ev.params.areaRng = [[lo, float(np.nextafter(hi, -np.inf))] for _, lo, hi in AREAS]
        ev.params.areaRngLbl = [n for n, _, _ in AREAS]
        ev.evaluate()
        ev.accumulate()
    return ev

def avg_valid(array):
    valid = np.asarray(array)
    valid = valid[valid >= 0]
    return float(valid.mean()) if len(valid) else None

def counts(data, lo, hi, c=None):
    anns = [a for a in data['annotations'] if lo <= a['area'] < hi and (c is None or a['category_id'] == c)]
    return (len(anns), len(set((a['image_id'] for a in anns))))

def summarize(data, ev, fixed):
    result = {}
    p = ev.eval['precision']
    for a, (name, lo, hi) in enumerate(AREAS):
        n, images = counts(data, lo, hi)
        valid_classes = [c for c in range(8) if counts(data, lo, hi, c)[0]]
        per = []
        for c in valid_classes:
            pr = p[0, :, c, a, -1]
            ok = pr >= 0.9
            r90 = float(ev.params.recThrs[ok].max()) if ok.any() else 0.0
            per.append(dict(class_id=c, name=NAMES[c], targets=counts(data, lo, hi, c)[0], ap50=avg_valid(p[0, :, c, a, -1]), ap50_95=avg_valid(p[:, :, c, a, -1]), interpolated_recall_at_precision90_iou50=r90, recall_iou50_conf025=float(fixed.eval['recall'][0, c, a, -1])))
        imgs = [x for x in fixed.evalImgs if x is not None and x['aRng'] == fixed.params.areaRng[a]]
        tp = sum((int(((x['dtMatches'][0] > 0) & ~x['dtIgnore'][0]).sum()) for x in imgs))
        fp = sum((int(((x['dtMatches'][0] == 0) & ~x['dtIgnore'][0]).sum()) for x in imgs))
        result[name] = dict(targets=n, images=images, classes=len(valid_classes), ap50=avg_valid(p[0, :, :, a, -1]), ap50_95=avg_valid(p[:, :, :, a, -1]), macro_recall_at_precision90_iou50=float(np.mean([x['interpolated_recall_at_precision90_iou50'] for x in per])) if per else None, fixed_conf025_iou50=dict(tp=tp, fp=fp, fn=n - tp, micro_precision=tp / (tp + fp) if tp + fp else None, micro_recall=tp / n if n else None, macro_recall=avg_valid(fixed.eval['recall'][0, :, a, -1])), per_class=per)
    return result
NAMES = ['Brown_Spot', 'Frog_Eye_Leaf_Spot', 'Grey_Spot', 'Healthy', 'Mosaic', 'Powdery_Mildew', 'Rust', 'Scab']
AREAS = [('all', 0, 1000000000000.0), ('tiny', 0, 256), ('small', 0, 1024), ('medium', 1024, 9216), ('large', 9216, 1000000000000.0)]
