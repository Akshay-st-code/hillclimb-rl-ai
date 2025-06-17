from ultralytics import YOLO
print("YOLOv8 is ready!")
from ultralytics import YOLO
import torch

print("YOLOv8 version:", YOLO.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
