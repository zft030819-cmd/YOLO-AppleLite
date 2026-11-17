# AppleLite Studio（叶观）

本地 CPU 检测软件：CSI/USB 实时监测、单张上传、阈值和 320/640 输入切换、结果图片与 JSON 下载、最近检测记录、FPS 曲线。前端资源全部随代码提供。

```bash
bash install.sh
bash start.sh
```

64 位 Raspberry Pi OS；安装脚本使用系统 OpenCV、Picamera2、Pillow、NumPy，另安装 NCNN。训练环境和 PyTorch不用于树莓派运行。模型位于 `models/`，`models.json` 校验 SHA256。`wheels/` 可选放入匹配 Python ABI 的 ARM64 NCNN wheel；为空时联网安装。

打开 `http://127.0.0.1:8080`。启动后摄像头默认关闭，实时监测中选择 CSI 或 USB 后点击开始。USB 编号通常从0开始。上传 JPG/PNG/WebP，最多12MB、1600万像素；大图缩放后 JSON 框坐标对应处理图。结果存于 `results/`，不提交至 GitHub。

可选后台及开机启动：停止前台后执行 `bash install-service.sh`。查看 `systemctl status applelite-studio`；日志 `journalctl -u applelite-studio -f`。

查看摄像头：`rpicam-hello --list-cameras`；USB设备：`ls /dev/video*`。模型输入320优先速度，640优先细节，具体FPS需实机测试。界面 FPS 只表示软件测量的处理吞吐，不等同于包含采集、网络和显示的全链路性能。

应用无账号认证，供可信局域网使用。服务默认监听0.0.0.0，可用 `bash start.sh --host 127.0.0.1` 限定本机访问。首次依赖安装完成后，树莓派本机使用不需要联网。
