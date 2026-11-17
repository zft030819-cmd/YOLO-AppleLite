from ultralytics import YOLO

# 直接加载原版 YOLOv12n 的配置文件
model = YOLO('yolov12n.yaml')

# 打印模型信息
model.info()