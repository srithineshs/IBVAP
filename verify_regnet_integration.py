import os
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from ultralytics import YOLO
from src.models.regnet import register_regnet_modules
from src.detector import BorderObjectDetector
from src.config import DetectorConfig

print("[1] Registering RegNet modules...")
register_regnet_modules()

print("[2] Building YOLO model with RegNet-Y backbone (data/yolo_regnet.yaml)...")
model = YOLO("data/yolo_regnet.yaml")
print(f"[+] RegNet YOLO created! Total parameters: {sum(p.numel() for p in model.model.parameters()):,}")

print("[3] Testing forward pass with sample frame (480x480)...")
dummy_img = torch.randn(1, 3, 480, 480)
results = model.predict(dummy_img, verbose=False)
print(f"[+] Prediction successful! Output detections count: {len(results[0].boxes)}")

print("[4] Testing ONNX export for RegNet YOLO...")
onnx_path = model.export(format="onnx", imgsz=480, dynamic=False, opset=12)
print(f"[+] Successfully exported RegNet ONNX to: {onnx_path}")

print("[5] Testing BorderObjectDetector with ONNX...")
cfg = DetectorConfig(model_path=onnx_path, input_width=480, input_height=480)
detector = BorderObjectDetector(cfg)
import numpy as np
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
dets = detector.detect(dummy_frame)
print(f"[+] Detector execution passed successfully! Detected objects: {len(dets)}")
print("\n[SUCCESS] RegNet-Y backbone is fully functional & integrated with IBVAP pipeline!")
