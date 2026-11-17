"""Repeatable single-image YOLO benchmarking for NCNN and ONNX CPU runtimes."""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    values = dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                  logical_cpus=os.cpu_count(), opencv=cv2.__version__, numpy=np.__version__)
    for package in ['ncnn', 'onnxruntime']:
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    for command in ['measure_temp', 'get_throttled', 'measure_clock arm']:
        try:
            values[command] = subprocess.check_output(['vcgencmd', *command.split()], timeout=3, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return values


def preprocess(image, size):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(image, (round(w * ratio), round(h * ratio)))
    left, top = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
    return tensor, ratio, left, top


class Backend:
    def __init__(self, item, base, threads):
        self.item = item
        self.size = int(item['size'])
        path = Path(os.path.expanduser(item['path']))
        self.path = path if path.is_absolute() else base / path
        self.names = item.get('names')
        self.kind = item['backend']
        if item.get('output_format') != 'yolo_xywh_scores':
            raise ValueError('Declare output_format=yolo_xywh_scores. End-to-end/NMS/YOLOv5 outputs require a separate decoder.')
        if self.kind == 'ncnn':
            import ncnn
            self.ncnn = ncnn
            self.net = ncnn.Net()
            self.net.opt.num_threads = threads
            self.net.opt.use_vulkan_compute = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False
            self.net.opt.use_int8_inference = True
            for loader, filename in [(self.net.load_param, 'model.ncnn.param'), (self.net.load_model, 'model.ncnn.bin')]:
                if not (self.path / filename).is_file() or loader(str(self.path / filename)):
                    raise RuntimeError(f'Failed to load {self.path / filename}')
            self.hashes = {name: sha(self.path / name) for name in ['model.ncnn.param', 'model.ncnn.bin']}
            if self.names is None and (self.path / 'metadata.yaml').exists():
                import yaml
                self.names = yaml.safe_load((self.path / 'metadata.yaml').read_text(encoding='utf-8')).get('names')
        elif self.kind == 'onnx':
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.net = ort.InferenceSession(str(self.path), sess_options=options, providers=['CPUExecutionProvider'])
            inputs = self.net.get_inputs()
            if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
                raise ValueError('Only a single float32 NCHW input is supported')
            shape = inputs[0].shape
            if len(shape) != 4 or any(isinstance(v, int) and v != expected for v, expected in zip(shape, [1, 3, self.size, self.size])):
                raise ValueError(f'Configured size does not match ONNX input {shape}')
            if self.names is None:
                text = self.net.get_modelmeta().custom_metadata_map.get('names')
                if text:
                    self.names = ast.literal_eval(text)
            self.hashes = {self.path.name: sha(self.path)}
        else:
            raise ValueError('backend must be ncnn or onnx')
        if isinstance(self.names, dict):
            self.names = [self.names.get(i, self.names.get(str(i))) for i in range(len(self.names))]
        if not self.names or any(not isinstance(name, str) for name in self.names):
            raise ValueError('Class names required in metadata or config names array')

    def run(self, tensor):
        if self.kind == 'ncnn':
            with self.net.create_extractor() as ex:
                if ex.input(self.item.get('input', 'in0'), self.ncnn.Mat(tensor).clone()):
                    raise RuntimeError('NCNN input failed')
                status, output = ex.extract(self.item.get('output', 'out0'))
                if status:
                    raise RuntimeError(f'NCNN extraction failed: {status}')
                raw = np.array(output, copy=True)
        else:
            raw = self.net.run([self.item.get('output') or self.net.get_outputs()[0].name], {self.net.get_inputs()[0].name: tensor[None]})[0]
        return raw


def annotate(raw, backend, image, ratio, left, top, confidence, iou):
    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    channels = 4 + len(backend.names)
    layout = backend.item.get('layout', 'channels_first')
    if raw.ndim != 2 or layout not in ['channels_first', 'channels_last']:
        raise ValueError(f'Unsupported output shape/layout: {raw.shape}, {layout}')
    rows = raw.T if layout == 'channels_first' else raw
    if rows.shape[1] != channels or not np.isfinite(rows).all():
        raise ValueError(f'Output must contain xywh plus {len(backend.names)} class scores; got {raw.shape}')
    categories, scores = rows[:, 4:].argmax(1), rows[:, 4:].max(1)
    mask = scores >= confidence
    rows, scores, categories = rows[mask], scores[mask], categories[mask]
    boxes = rows[:, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    keep = []
    for category in np.unique(categories):
        indices = np.where(categories == category)[0]
        selected = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(), confidence, iou)
        keep.extend(indices[np.asarray(selected, dtype=int).reshape(-1)].tolist())
    h, w = image.shape[:2]
    results = []
    for index in sorted(keep, key=lambda k: -scores[k])[:300]:
        x, y, bw, bh = boxes[index]
        box = [int(np.clip((x - left) / ratio, 0, w - 1)), int(np.clip((y - top) / ratio, 0, h - 1)),
               int(np.clip((x + bw - left) / ratio, 0, w - 1)), int(np.clip((y + bh - top) / ratio, 0, h - 1))]
        label, score = backend.names[int(categories[index])], float(scores[index])
        results.append(dict(label=label, confidence=score, box=box))
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 220, 0), 2)
        cv2.putText(image, f'{label} {score:.2f}', (box[0], max(18, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1)
    return results


def stats(values):
    return dict(mean_ms=float(np.mean(values)), std_ms=float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                fps=float(1000 / np.mean(values)))


def run_model(backend, original, args):
    def once():
        start = time.perf_counter()
        image = original.copy()
        tensor, ratio, left, top = preprocess(image, backend.size)
        prepared = time.perf_counter()
        raw = backend.run(tensor)
        inferred = time.perf_counter()
        detections = annotate(raw, backend, image, ratio, left, top, args.conf, args.iou)
        end = time.perf_counter()
        return image, detections, np.array([prepared - start, inferred - prepared, end - inferred, end - start]) * 1000
    for _ in range(args.warmup):
        once()
    times, rounds = [], []
    for number in range(args.repeat):
        current = []
        for _ in range(args.runs):
            image, detections, timing = once()
            times.append(timing)
            current.append(timing[1])
        rounds.append(float(np.mean(current)))
        print(f'  Round {number + 1}/{args.repeat}: inference {rounds[-1]:.2f} ms', flush=True)
    times = np.asarray(times)
    metrics = {name: stats(times[:, index]) for index, name in enumerate(['preprocess', 'inference', 'postprocess_draw', 'processing'])}
    metrics['round_inference_means_ms'] = rounds
    metrics['round_mean_std_ms'] = float(np.std(rounds, ddof=1)) if len(rounds) > 1 else 0
    return image, detections, metrics, times.tolist()


def report_html(root, results):
    rows, images = [], []
    for item in results:
        name = html.escape(item['name'])
        if item['status'] != 'ok':
            rows.append(f'<tr><td>{name}</td><td colspan="5">失败：{html.escape(item["error"])}</td></tr>')
            continue
        m = item['metrics']
        rows.append(f'<tr><td>{name}</td><td>{item["size"]}</td><td>{m["inference"]["mean_ms"]:.2f}</td><td>{m["inference"]["fps"]:.2f}</td><td>{m["processing"]["fps"]:.2f}</td><td>{m["inference"]["p95_ms"]:.2f}</td></tr>')
        images.append(f'<figure><figcaption>{name} · 本图 {len(item["detections"])} 个框</figcaption><img src="{item["image"]}" alt="{name} 检测结果"></figure>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单帧检测对比报告</title><style>body{font:15px/1.7 system-ui;margin:30px auto;padding:0 20px;max-width:1200px;color:#18291d;background:#f3f6f0}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #d8dfd5}img{width:100%;display:block}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}figure{margin:20px 0;background:white;padding:12px}small,p{color:#526357}</style><h1>单帧检测与测速对比</h1><p>重复同一张图片；帧率按 1000 / 平均延迟计算。不包含图像读取、保存、摄像头或显示，不代表实时视频帧率。处理 FPS 包含预处理、推理、后处理和画框；检测框数量不是准确率。</p><table><tr><th>模型</th><th>输入尺寸</th><th>推理 ms</th><th>推理 FPS</th><th>处理 FPS</th><th>推理 P95 ms</th></tr>' + ''.join(rows) + '</table><section>' + ''.join(images) + '</section><p>完整配置、输入及权重 SHA256、运行环境、逐次延迟和失败记录见 results.json；表格见 summary.csv。不同输入尺寸及运行后端需要在论文中明确标注。</p></html>'
    (root / 'report.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('models_640.json'))
    parser.add_argument('--image', type=Path, default=Path(__file__).with_name('sample.jpg'))
    parser.add_argument('--capture', action='store_true', help='Capture one CSI image and exit; then benchmark that saved image')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--allow-mixed-size', action='store_true')
    parser.add_argument('--reverse', action='store_true', help='Reverse model order for a second comparison session')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if min(args.runs, args.repeat, args.threads) < 1 or args.warmup < 0 or not 0 < args.conf < 1 or not 0 < args.iou < 1:
        parser.error('Invalid benchmark settings')
    if args.capture:
        if args.image.exists():
            parser.error('Capture target exists; choose a new --image path to preserve earlier experiments')
        from picamera2 import Picamera2
        camera = Picamera2(args.camera)
        try:
            camera.configure(camera.create_still_configuration(main={'size': (1280, 960), 'format': 'RGB888'}))
            camera.start()
            time.sleep(2)
            image = camera.capture_array('main')
            args.image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.image), image):
                raise RuntimeError('Failed to save camera image')
        finally:
            camera.close()
        print(f'Captured {args.image}. Run again without --capture to benchmark this image.')
        return
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    models = config['models']
    if args.reverse:
        models = list(reversed(models))
    if not models or len({item['name'] for item in models}) != len(models):
        parser.error('Provide at least one model; model names must be unique')
    if len({int(item['size']) for item in models}) > 1 and not args.allow_mixed_size:
        parser.error('Mixed input sizes rejected; compare at the same size or explicitly use --allow-mixed-size')
    original = cv2.imread(str(args.image))
    if original is None:
        parser.error(f'Cannot read image: {args.image}')
    output = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(started=datetime.now().astimezone().isoformat(), input=str(args.image.resolve()), image_sha256=sha(args.image),
                  configuration=config, settings={key: value for key, value in vars(args).items() if not isinstance(value, Path)},
                  environment_before=environment(), timing='Repeated single image. Disk IO/camera/display excluded. Processing includes copy, preprocess, inference, NMS, and drawing.', results=[])
    for index, item in enumerate(models):
        print(f'[{index + 1}/{len(models)}] {item["name"]}', flush=True)
        result = dict(name=item['name'], size=int(item['size']), backend=item['backend'], status='failed')
        try:
            load_start = time.perf_counter()
            backend = Backend(item, args.config.resolve().parent, args.threads)
            result['load_ms'] = (time.perf_counter() - load_start) * 1000
            result['model_sha256'] = backend.hashes
            result['environment_before'] = environment()
            image, detections, metrics, times = run_model(backend, original, args)
            filename = f'model_{index + 1:02d}.jpg'
            if not cv2.imwrite(str(output / filename), image):
                raise RuntimeError('Failed to save annotated image')
            result.update(status='ok', image=filename, detections=detections, metrics=metrics, samples_ms=times,
                          sample_columns=['preprocess', 'inference', 'postprocess_draw', 'processing'])
            result['environment_after'] = environment()
            print(f'  Inference {metrics["inference"]["fps"]:.2f} FPS; processing {metrics["processing"]["fps"]:.2f} FPS', flush=True)
            del backend
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            print('  FAILED:', result['error'], flush=True)
        report['results'].append(result)
        (output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'backend', 'size', 'threads', 'warmup', 'runs_per_round', 'rounds', 'inference_mean_ms', 'inference_std_ms', 'inference_median_ms', 'inference_p95_ms', 'inference_fps', 'processing_mean_ms', 'processing_fps', 'round_mean_std_ms', 'status', 'error'])
        for result in report['results']:
            base = [result['name'], result['backend'], result['size'], args.threads, args.warmup, args.runs, args.repeat]
            if result['status'] == 'ok':
                m = result['metrics']
                values = [m['inference'][key] for key in ['mean_ms', 'std_ms', 'median_ms', 'p95_ms', 'fps']] + [m['processing']['mean_ms'], m['processing']['fps'], m['round_mean_std_ms']]
                writer.writerow(base + [round(value, 4) for value in values] + ['ok', ''])
            else:
                writer.writerow(base + [''] * 8 + ['failed', result['error']])
    report_html(output, report['results'])
    print(f'Results saved: {output}', flush=True)
    if any(result['status'] != 'ok' for result in report['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
