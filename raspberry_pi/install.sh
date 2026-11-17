#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ "$(uname -m)" != 'aarch64' ]; then
    echo '请使用64位 Raspberry Pi OS（aarch64）。' >&2
    exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-numpy python3-opencv python3-pil python3-picamera2 rpicam-apps
python3 -m venv --system-site-packages .venv
PY_TAG=$(.venv/bin/python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
WHEEL=$(find wheels -maxdepth 1 -name "ncnn-*-${PY_TAG}-*aarch64*.whl" -print -quit 2>/dev/null || true)
if [ -n "$WHEEL" ]; then
    .venv/bin/python -m pip install --no-deps "$WHEEL"
else
    .venv/bin/python -m pip install --only-binary=:all: --no-deps 'ncnn==1.0.20261020'
fi
.venv/bin/python -c 'import ncnn, cv2, numpy; from PIL import Image; from picamera2 import Picamera2; print("NCNN:", ncnn.__version__); print("Cameras:", Picamera2.global_camera_info())'
echo '安装完成。运行 bash start.sh，然后打开 http://树莓派IP:8080'
