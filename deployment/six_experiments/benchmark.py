"""Six-condition NCNN benchmark using a common fixed image set and rotating order."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from lab import Backend, annotate, environment, preprocess, sha, stats
from system_monitor import Monitor

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--cooldown', type=float, default=2)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-monitor', action='store_true')
    args = parser.parse_args()
    if min(args.rounds, args.threads) < 1 or min(args.warmup, args.limit, args.cooldown) < 0:
        parser.error('Invalid settings')
    cv2.setNumThreads(1)
    config = json.loads((ROOT / 'models.json').read_text())['models']
    if args.reverse:
        config.reverse()
    if len(config) != 6 or {(m['size'], m['precision']) for m in config} != {(s, p) for s in [640, 512, 416] for p in ['FP32', 'mixed_INT8']}:
        raise RuntimeError('Expected all six size/precision configurations')
    records = json.loads((ROOT / 'images.json').read_text(encoding='utf-8'))
    if args.limit:
        records = records[:args.limit]
    images = []
    for item in records:
        path = ROOT / item['file']
        if sha(path) != item['sha256']:
            raise RuntimeError(f'Image changed: {path}')
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(str(path))
        images.append(image)
    if not images:
        raise RuntimeError('No test images')
    output = ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    provenance_map = {m['id']: m for m in provenance['models']}
    start = time.perf_counter()
    context = dict(stage='starting', model='', round=0)
    monitor = Monitor(output / 'system_monitor.jsonl', start, context)
    report = dict(settings=vars(args), image_count=len(images), images=records, provenance=provenance, historical_accuracy_included=False,
                  formal_protocol=args.rounds >= 6 and args.limit == 0 and args.warmup >= 10,
                  environment=environment(), rounds=[], completed=False,
                  timing='Mean inference FPS=1000/mean latency; processing includes copy/preprocess/inference/NMS/draw. Disk, camera, network and display excluded. Linux process sampler at 1 Hz unless disabled.')
    write_json(output / 'results.json', report)
    samples = {m['name']: [] for m in config}
    round_means = {m['name']: [] for m in config}
    if not args.no_monitor:
        monitor.begin()
    try:
        with (output / 'samples.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['model', 'round', 'image', 'start_elapsed_s', 'preprocess_ms', 'inference_ms', 'postprocess_draw_ms', 'processing_ms', 'detections'])
            for round_index in range(args.rounds):
                offset = round_index % len(config)
                order = config[offset:] + config[:offset]
                for item in order:
                    name = item['name']
                    context.update(stage='load_and_warmup', model=name, round=round_index + 1)
                    print(f'Round {round_index + 1}/{args.rounds}: {name}, {len(images)} images', flush=True)
                    entry = dict(round=round_index + 1, model=name, environment_before=environment())
                    backend = Backend(item, ROOT, args.threads)
                    if backend.hashes != provenance_map[name]['sha256']:
                        raise RuntimeError(f'Model hash mismatch: {name}')
                    for index in range(args.warmup):
                        image = images[index % len(images)].copy()
                        x, ratio, left, top = preprocess(image, item['size'])
                        annotate(backend.run(x), backend, image, ratio, left, top, .25, .45)
                    context['stage'] = 'measuring'
                    current = []
                    for index, source in enumerate(images):
                        tick = time.perf_counter()
                        image = source.copy()
                        x, ratio, left, top = preprocess(image, item['size'])
                        prepared = time.perf_counter()
                        raw = backend.run(x)
                        inferred = time.perf_counter()
                        detections = annotate(raw, backend, image, ratio, left, top, .25, .45)
                        finished = time.perf_counter()
                        timing = [(prepared - tick) * 1000, (inferred - prepared) * 1000, (finished - inferred) * 1000, (finished - tick) * 1000]
                        current.append(timing)
                        writer.writerow([name, round_index + 1, records[index]['file'], tick - start, *timing, len(detections)])
                        if round_index == 0 and index % 10 == 0:
                            if not cv2.imwrite(str(output / f'{name}_{index:03d}.jpg'), image):
                                raise RuntimeError('Cannot save example')
                    file.flush()
                    samples[name].extend(current)
                    entry['metrics'] = {key: stats(np.asarray(current)[:, index]) for index, key in enumerate(['preprocess', 'inference', 'postprocess_draw', 'processing'])}
                    round_means[name].append(entry['metrics']['inference']['mean_ms'])
                    entry['environment_after'] = environment()
                    report['rounds'].append(entry)
                    write_json(output / 'results.json', report)
                    print(f'  Inference {entry["metrics"]["inference"]["mean_ms"]:.2f} ms / {entry["metrics"]["inference"]["fps"]:.2f} FPS', flush=True)
                    del backend
                    context['stage'] = 'cooldown'
                    time.sleep(args.cooldown)
        summary = []
        for item in config:
            name = item['name']
            values = np.asarray(samples[name])
            infer, total = stats(values[:, 1]), stats(values[:, 3])
            summary.append(dict(model=name, size=item['size'], precision=item['precision'], threads=args.threads,
                                rounds=args.rounds, images_per_round=len(images), warmup_per_round=args.warmup,
                                ncnn_bin_MB=provenance_map[name]['ncnn_bin_bytes'] / 1e6,
                                mean_inference_ms=infer['mean_ms'], std_inference_ms=infer['std_ms'], p95_inference_ms=infer['p95_ms'], inference_fps=infer['fps'],
                                mean_processing_ms=total['mean_ms'], processing_fps=total['fps'],
                                round_mean_std_ms=float(np.std(round_means[name], ddof=1)) if args.rounds > 1 else 0))
        report.update(summary=summary, completed=True)
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        monitor.close()
        write_json(output / 'results.json', report)
        print('Results:', output, flush=True)


if __name__ == '__main__':
    main()
