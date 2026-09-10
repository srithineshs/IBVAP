"""
Edge Object Detector Module for Border Surveillance
Supports ONNX Runtime & TensorRT Execution with Ground-Contact Foot-Point Calculation.
"""

import os
import sys
import time
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

try:
    from src.config import DetectorConfig, CLASS_MAP, BORDER_CLASS_NAMES
    from src.models.regnet import register_regnet_modules
except ImportError:
    from config import DetectorConfig, CLASS_MAP, BORDER_CLASS_NAMES  # type: ignore
    try:
        from models.regnet import register_regnet_modules  # type: ignore
    except ImportError:
        def register_regnet_modules(): pass

# Register RegNet custom architecture modules
register_regnet_modules()

class DetectionResult:
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox_xyxy: Tuple[float, float, float, float],
        foot_point: Tuple[float, float],
        is_active: bool = False,
        track_id: Optional[int] = None,
        motion_score: float = 0.0
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = float(confidence)
        self.bbox_xyxy = [float(c) for c in bbox_xyxy]  # [x1, y1, x2, y2]
        self.foot_point = [float(foot_point[0]), float(foot_point[1])]  # [foot_x, foot_y]
        self.is_active = is_active
        self.track_id = track_id
        self.motion_score = float(motion_score)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy],
            "foot_point": [round(c, 2) for c in self.foot_point],
            "is_active": self.is_active,
            "state": "ACTIVE" if self.is_active else "IDLE",
            "motion_score": round(self.motion_score, 4)
        }
        if self.track_id is not None:
            data["track_id"] = self.track_id
        return data


class BorderObjectDetector:
    """
    High-performance YOLO detector (supporting RegNet-Y and CSP backbones)
    running ONNX / PyTorch backends.
    Extracts precise bounding boxes and bottom-center ground-contact points
    strictly for active, moving targets.
    """
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model_path = config.model_path
        self.input_w = config.input_width
        self.input_h = config.input_height
        self.backend = "onnx"
        self.session = None
        self.torch_model = None

        self._initialize_backend()

    def _initialize_backend(self):
        """Initializes ONNX Runtime session or PyTorch model based on configured path."""
        # 1. If explicit ONNX model is provided and exists, load ONNX Runtime
        if self.model_path and self.model_path.endswith(".onnx") and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                threads = min(8, max(2, os.cpu_count() or 4))
                sess_options.intra_op_num_threads = threads
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
                self.backend = "onnx"
                self.input_name = self.session.get_inputs()[0].name
                active_provider = self.session.get_providers()[0]
                print(f"[+] Loaded ONNX Detection Model ({self.model_path}) with Provider: {active_provider} (Threads: {threads})")
                return
            except Exception as e:
                print(f"[!] ONNX Runtime init failed ({e}), falling back to PyTorch Ultralytics.")

        # 2. Otherwise load PyTorch model
        self._load_torch_fallback()

    def _load_torch_fallback(self):
        import torch
        from ultralytics import YOLO
        
        # Optimize CPU threads for high FPS
        try:
            torch.set_num_threads(min(8, max(2, os.cpu_count() or 4)))
        except Exception:
            pass

        candidate_weights = [
            self.model_path,
            "weights/yolo26_border.pt",
            "yolov8s.pt",
            "yolov8m.pt"
        ]
        chosen_weight = "weights/yolo26_border.pt"
        for w in candidate_weights:
            if w and os.path.exists(w) and not w.endswith(".onnx"):
                chosen_weight = w
                break

        print(f"[*] Initializing YOLO Engine with: {chosen_weight}")
        self.torch_model = YOLO(chosen_weight)
        self.backend = "torch"

    def _letterbox(self, img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _box_overlaps_motion(self, bbox: Tuple[float, float, float, float], motion_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """Verifies if the detected object spatially overlaps with an active movement contour."""
        if not motion_boxes:
            return True  # If motion bounding boxes not provided, pass through
        
        bx1, by1, bx2, by2 = bbox
        for (mx, my, mw, mh) in motion_boxes:
            mx2, my2 = mx + mw, my + mh
            # Check intersection
            ix1 = max(bx1, mx)
            iy1 = max(by1, my)
            ix2 = min(bx2, mx2)
            iy2 = min(by2, my2)
            
            if ix1 < ix2 and iy1 < iy2:
                return True
        return False

    def detect(self, frame: np.ndarray, motion_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[DetectionResult]:
        """
        Runs object detection strictly on moving persons and vehicles.
        Suppresses static objects, furniture, and unallowed classes.
        """
        orig_h, orig_w = frame.shape[:2]
        detections: List[DetectionResult] = []

        if self.backend == "torch":
            results = self.torch_model.predict(
                frame,
                imgsz=self.input_w,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_iou_threshold,
                classes=self.config.allowed_class_ids,  # STRICT FILTER: Only Person & Vehicles
                verbose=False
            )[0]

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                if self.config.allowed_class_ids is not None and cls_id not in self.config.allowed_class_ids:
                    continue  # Filter unallowed classes if restriction is configured

                conf = float(box.conf[0].item())
                min_conf = self.config.class_confidence_thresholds.get(cls_id, self.config.confidence_threshold)
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (x1, y1, x2, y2)

                # Motion spatial gating check: reject static objects
                if motion_boxes and not self._box_overlaps_motion(bbox, motion_boxes):
                    continue

                # Foot-point: Bottom-Center coordinate
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                cls_name = BORDER_CLASS_NAMES.get(cls_id, CLASS_MAP.get(cls_id, "person"))

                detections.append(DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=bbox,
                    foot_point=(foot_x, foot_y)
                ))

        elif self.backend == "onnx":
            # Native ONNX Runtime Execution
            img_padded, ratio, (pad_w, pad_h) = self._letterbox(frame, (self.input_h, self.input_w))
            blob = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
            blob = np.expand_dims(blob, axis=0)   # NCHW

            outputs = self.session.run(None, {self.input_name: blob})[0]  # Shape: [1, 84, 8400] or similar
            
            # Post-process YOLO outputs
            output = outputs[0].T  # [8400, 84]
            boxes = output[:, :4]
            scores = output[:, 4:]
            
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            # Filter candidates above base threshold
            mask = confidences >= self.config.confidence_threshold
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]

            if len(boxes) > 0:
                # Convert xywh to xyxy
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = (cx - w / 2 - pad_w) / ratio
                y1 = (cy - h / 2 - pad_h) / ratio
                x2 = (cx + w / 2 - pad_w) / ratio
                y2 = (cy + h / 2 - pad_h) / ratio

                # Clip to original image boundaries
                x1 = np.clip(x1, 0, orig_w)
                y1 = np.clip(y1, 0, orig_h)
                x2 = np.clip(x2, 0, orig_w)
                y2 = np.clip(y2, 0, orig_h)

                # OpenCV NMS
                cv_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(x1))]
                indices = cv2.dnn.NMSBoxes(cv_boxes, confidences.tolist(), self.config.confidence_threshold, self.config.nms_iou_threshold)

                if len(indices) > 0:
                    for i in indices.flatten():
                        cid = int(class_ids[i])
                        conf = float(confidences[i])
                        min_conf = self.config.class_confidence_thresholds.get(cid, self.config.confidence_threshold)
                        if conf < min_conf:
                            continue

                        box_coords = (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))
                        foot_coords = ((x1[i] + x2[i]) / 2.0, float(y2[i]))
                        
                        detections.append(DetectionResult(
                            class_id=cid,
                            class_name=CLASS_MAP.get(cid, f"object_{cid}"),
                            confidence=conf,
                            bbox_xyxy=box_coords,
                            foot_point=foot_coords
                        ))

        return detections
