import cv2
import os
import numpy as np
import sys
sys.path.insert(0, ".")
from src.config import BorderPipelineConfig, DetectorConfig
from src.detector import BorderObjectDetector

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
url = "rtsp://admin:admin123@10.245.106.240:554/Streaming/Channels/102"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Could not open stream.")
    exit()

ret, frame = cap.read()
cap.release()

if ret and frame is not None:
    print(f"Captured camera frame shape: {frame.shape}")
    cv2.imwrite("scratch/cctv_frame.jpg", frame)
    
    # Test detector with different thresholds
    cfg = DetectorConfig()
    cfg.confidence_threshold = 0.20
    detector = BorderObjectDetector(cfg)
    
    # Run detection
    detections = detector.detect(frame)
    print(f"\nDetections at 0.20 conf ({len(detections)}):")
    for d in detections:
        print(f" - {d.class_name} (conf: {d.confidence:.3f}) at bbox {d.bbox_xyxy}")
        
    # Draw on frame
    annotated = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{d.class_name} {d.confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite("scratch/cctv_annotated.jpg", annotated)
    print("Saved annotated frame to scratch/cctv_annotated.jpg")
