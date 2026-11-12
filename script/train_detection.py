import torch
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(
    data="data_detection.yaml",
    epochs = 20,
    imgsz = 640,
    batch = 4,
    device = 0
)

model.export(format="onnx")
