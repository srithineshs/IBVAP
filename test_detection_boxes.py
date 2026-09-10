import os, sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from src.config import DetectorConfig
from src.detector import BorderObjectDetector

# Test on a real image from dataset/images/train/
img_path = "dataset/images/train/border_train_000.jpg"
frame = cv2.imread(img_path)
if frame is None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

cfg = DetectorConfig(model_path="weights/yolo26_border.onnx", confidence_threshold=0.30)
detector = BorderObjectDetector(cfg)

dets = detector.detect(frame)
print(f"Test on {img_path}:")
print(f"Detected {len(dets)} objects:")
for d in dets:
    print(f" - {d.class_name} ({d.confidence:.2f}) at bbox {d.bbox_xyxy}")
