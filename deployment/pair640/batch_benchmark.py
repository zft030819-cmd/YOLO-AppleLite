"""Paired, alternating-order multi-image CPU benchmark for the selected models."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from lab import Backend, annotate, environment, preprocess, sha, stats

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; zero uses all images')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--cooldown', type=float, default=2)
    args = parser.parse_args()
    if min(args.rounds, args.threads) < 1 or args.warmup < 0 or args.limit < 0 or args.cooldown < 0:
        parser.error('Invalid settings')
    cv2.setNumThreads(1)
    config = json.loads((ROOT / 'models.json').read_text())['models']
    records = json.loads((ROOT / 'images.json').read_text(encoding='utf-8'))
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise RuntimeError('Empty image list')
    images = []
    for item in records:
        path = ROOT / item['file']
        if sha(path) != item['sha256']:
            raise RuntimeError(f'Image hash mismatch: {path}')
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(str(path))
        images.append(image)
    output = ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    report = dict(started=datetime.now().astimezone().isoformat(), settings=vars(args), image_count=len(images), images=records,
                  provenance=json.loads((ROOT / 'provenance.json').read_text(encoding='utf-8')), environment=environment(), rounds=[],
                  timing='Image decode preloaded; measured copy+preprocess, NCNN inference, NMS+drawing. Camera/disk/network/display excluded.')
    report['historical_accuracy_included'] = False
    samples = {item['name']: [] for item in config}
    round_means = {item['name']: [] for item in config}
    file_samples = []
    for round_index in range(args.rounds):
        order = list(reversed(config)) if (round_index + int(args.reverse)) % 2 else config
        for item in order:
            name = item['name']
            print(f'Round {round_index + 1}/{args.rounds}: {name}, {len(images)} images', flush=True)
            entry = dict(round=round_index + 1, model=name, environment_before=environment())
            backend = Backend(item, ROOT, args.threads)
            entry['model_sha256'] = backend.hashes
            for i in range(args.warmup):
                image = images[i % len(images)].copy()
                x, ratio, left, top = preprocess(image, backend.size)
                annotate(backend.run(x), backend, image, ratio, left, top, .25, .45)
            times = []
            for index, source in enumerate(images):
                start = time.perf_counter()
                image = source.copy()
                x, ratio, left, top = preprocess(image, backend.size)
                prepared = time.perf_counter()
                raw = backend.run(x)
                inferred = time.perf_counter()
                detections = annotate(raw, backend, image, ratio, left, top, .25, .45)
                finished = time.perf_counter()
                timing = [(prepared - start) * 1000, (inferred - prepared) * 1000, (finished - inferred) * 1000, (finished - start) * 1000]
                times.append(timing)
                file_samples.append([name, round_index + 1, records[index]['file'], *timing, len(detections)])
                if round_index == 0 and index % 10 == 0:
                    if not cv2.imwrite(str(output / f'{name}_{index:03d}.jpg'), image):
                        raise RuntimeError('Cannot save example detection image')
            samples[name].extend(times)
            entry['metrics'] = {key: stats(np.asarray(times)[:, index]) for index, key in enumerate(['preprocess', 'inference', 'postprocess_draw', 'processing'])}
            round_means[name].append(entry['metrics']['inference']['mean_ms'])
            entry['environment_after'] = environment()
            report['rounds'].append(entry)
            (output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'  {entry["metrics"]["inference"]["mean_ms"]:.2f} ms, {entry["metrics"]["inference"]["fps"]:.2f} inference FPS', flush=True)
            del backend
            time.sleep(args.cooldown)
    summary = []
    for item in config:
        values = np.asarray(samples[item['name']])
        inference, processing = stats(values[:, 1]), stats(values[:, 3])
        provenance = next(p for p in report['provenance']['models'] if p['id'] == item['name'])
        summary.append(dict(model=item['name'], backend='ncnn', precision='FP32', size=640, threads=args.threads,
                            parameters=provenance['parameters'], ncnn_bin_MB=provenance['ncnn_bytes'] / 1e6,
                            rounds=args.rounds, images_per_round=len(images), warmup_per_round=args.warmup,
                            mean_inference_ms=inference['mean_ms'], std_inference_ms=inference['std_ms'],
                            p95_inference_ms=inference['p95_ms'], inference_fps=inference['fps'],
                            mean_processing_ms=processing['mean_ms'], processing_fps=processing['fps'],
                            round_mean_std_ms=float(np.std(round_means[item['name']], ddof=1)) if args.rounds > 1 else 0))
    report['summary'] = summary
    (output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output / 'samples.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'round', 'image', 'preprocess_ms', 'inference_ms', 'postprocess_draw_ms', 'processing_ms', 'detections'])
        writer.writerows(file_samples)
    print(json.dumps(summary, indent=2), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
