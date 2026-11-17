#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x '.venv/bin/python' ]; then
    echo '请先执行 bash install.sh' >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
exec .venv/bin/python server.py "$@"
