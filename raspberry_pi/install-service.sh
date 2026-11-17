#!/bin/sh
# Optional: user explicitly runs this script to enable background startup.
set -eu
APP_ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
RUN_USER=${SUDO_USER:-$(id -un)}
if [ ! -x "$APP_ROOT/.venv/bin/python" ]; then
    echo '请先运行 bash install.sh' >&2
    exit 1
fi
if [ "$RUN_USER" = 'root' ]; then
    echo '请以树莓派普通用户运行本脚本，脚本会自行调用sudo。' >&2
    exit 1
fi
case "$APP_ROOT" in *' '*) echo '开机服务请安装在不含空格的目录。' >&2; exit 1;; esac
sudo tee /etc/systemd/system/applelite-studio.service >/dev/null <<EOF
[Unit]
Description=AppleLite Studio leaf monitoring
After=network.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_ROOT
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=4
ExecStart=$APP_ROOT/.venv/bin/python $APP_ROOT/server.py --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now applelite-studio.service
echo '后台服务已启用。日志：journalctl -u applelite-studio -f'
