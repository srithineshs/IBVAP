import os, sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from src.config import DetectorConfig
from src.detector import BorderObjectDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
if not cap.isOpened():
    print("Could not open webcam for test.")
    sys.exit(0)

# Grab 5 frames to let camera auto-adjust exposure
for _ in range(5):
    ret, frame = cap.read()

cap.release()

if ret and frame is not None:
    print(f"Captured test camera frame: {frame.shape}")
    
    # Test with fast ONNX
    cfg = DetectorConfig(model_path="weights/yolo_border_fast.onnx", confidence_threshold=0.35)
    detector = BorderObjectDetector(cfg)
    dets = detector.detect(frame)
    print(f"ONNX Detections count: {len(dets)}")
    for d in dets:
        print(f" -> {d.class_name} ({d.confidence:.2f}) at bbox {d.bbox_xyxy}")

    # Test with PyTorch fallback
    cfg_torch = DetectorConfig(model_path="yolov8s.pt", confidence_threshold=0.35)
    detector_torch = BorderObjectDetector(cfg_torch)
    dets_torch = detector_torch.detect(frame)
    print(f"PyTorch Detections count: {len(dets_torch)}")
    for d in dets_torch:
        print(f" -> {d.class_name} ({d.confidence:.2f}) at bbox {d.bbox_xyxy}")
