"""
Motion Gating & Preprocessing Module for Border Surveillance
Filters static background/vegetation movement and triggers heavy AI inference only upon confirmed motion.
"""

import sys
from pathlib import Path

# Add project root and src directory to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
from typing import Tuple, List, Optional, Any, Dict

try:
    from src.config import MotionGatingConfig, PreprocessingConfig
except ImportError:
    from config import MotionGatingConfig, PreprocessingConfig  # type: ignore

class BorderFramePreprocessor:
    """
    Applies adaptive contrast enhancement, CLAHE, and gamma correction for
    challenging low-light, IR night mode, and foggy border environments.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        # Precompute gamma lookup table for fast transformation
        inv_gamma = 1.0 / self.config.gamma_value
        self.gamma_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)
        ]).astype("uint8")

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Enhances input frame visibility based on configured low-light algorithms."""
        processed = frame.copy()

        # 1. Gamma Correction for Dark Border Scenes
        if self.config.enable_gamma_correction:
            processed = cv2.LUT(processed, self.gamma_table)

        # 2. CLAHE on Luminance Channel (LAB color space)
        if self.config.enable_clahe:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = self.clahe.apply(l_channel)
            enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
            processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return processed


class CentroidDisplacementTracker:
    """
    Tracks entity centroid coordinates across frames and calculates true physical displacement.
    Configured for HIGH-MOVEMENT sensitivity: minor hand/head shifts and micro-movements are ignored.
    """
    def __init__(self, displacement_threshold_px: float = 24.0, smoothing_window: int = 8):
        self.threshold = displacement_threshold_px
        self.window = smoothing_window
        self.history: List[Tuple[float, float]] = []

    def update(self, centroid: Optional[Tuple[float, float]]) -> bool:
        """
        Updates tracker with new centroid (x, y).
        Returns True ONLY if high, prominent physical movement occurs.
        """
        if centroid is None:
            self.history.clear()
            return False

        self.history.append(centroid)
        if len(self.history) > self.window:
            self.history.pop(0)

        if len(self.history) < 4:
            return False

        # Calculate max distance moved within the temporal window
        first_x, first_y = self.history[0]
        curr_x, curr_y = self.history[-1]
        displacement = np.sqrt((curr_x - first_x) ** 2 + (curr_y - first_y) ** 2)

        # True ONLY if displacement exceeds the high movement threshold (e.g. >= 24px)
        return displacement >= self.threshold


class BorderMotionGater:
    """
    Evaluates physical motion with strict high-movement filtering.
    """
    def __init__(self, config: MotionGatingConfig, roi_polygon_norm: List[List[float]], frame_shape: Tuple[int, int]):
        self.config = config
        self.height, self.width = frame_shape[:2]
        
        # Build binary ROI mask
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        poly_pts = np.array([
            [int(x * self.width), int(y * self.height)] for x, y in roi_polygon_norm
        ], dtype=np.int32)
        cv2.fillPoly(self.roi_mask, [poly_pts], 255)
        
        # Strict Background Subtractor (MOG2)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )
        
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.consecutive_triggers = 0
        self.centroid_tracker = CentroidDisplacementTracker(
            displacement_threshold_px=self.config.min_high_velocity_px
        )

    def check_motion(self, frame: np.ndarray, detected_centroid: Optional[Tuple[float, float]] = None) -> Tuple[bool, Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Evaluates physical motion strictly for HIGH / PROMINENT MOVEMENT:
        Filters out low-sensitivity noise, subtle postures, breathing, and minor twitches.
        """
        if not self.config.enabled:
            return True, None, []

        # 1. Heavy Gaussian Blur to eliminate all sensor noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        fg_mask = self.bg_subtractor.apply(blurred)
        
        # Restrict strictly within ROI polygon
        fg_mask = cv2.bitwise_and(fg_mask, self.roi_mask)
        
        # Morphological opening/closing to eliminate pixel flicker
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        fg_mask = cv2.dilate(fg_mask, self.morph_kernel, iterations=2)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_boxes = []
        has_high_motion = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Must exceed high motion area threshold (e.g. >= 6500px)
            if area >= self.config.min_motion_area_pixels:
                has_high_motion = True
                x, y, w, h = cv2.boundingRect(cnt)
                motion_boxes.append((x, y, w, h))

        # 2. Centroid displacement verification (verifies high speed/translation)
        is_moving_by_displacement = self.centroid_tracker.update(detected_centroid)

        if has_high_motion or is_moving_by_displacement:
            self.consecutive_triggers = min(self.config.consecutive_motion_frames + 1, self.consecutive_triggers + 1)
        else:
            # Instant reset when stationary (eliminates 2-4 second delay)
            self.consecutive_triggers = 0
            
        is_triggered = self.consecutive_triggers >= self.config.consecutive_motion_frames
        return is_triggered, fg_mask, motion_boxes

    def evaluate_box_motion(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        motion_mask: Optional[np.ndarray]
    ) -> Tuple[bool, float, int]:
        """
        Evaluates physical motion strictly within a specific bounding box [x1, y1, x2, y2].
        Returns:
            is_active (bool): True if motion inside the bbox exceeds motion threshold.
            motion_ratio (float): Ratio of motion pixels relative to bbox area.
            motion_pixels (int): Count of foreground motion pixels inside the bbox.
        """
        if motion_mask is None:
            return False, 0.0, 0

        h, w = motion_mask.shape[:2]
        x1 = max(0, min(int(bbox_xyxy[0]), w - 1))
        y1 = max(0, min(int(bbox_xyxy[1]), h - 1))
        x2 = max(0, min(int(bbox_xyxy[2]), w))
        y2 = max(0, min(int(bbox_xyxy[3]), h))

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        box_area = box_w * box_h

        # Extract foreground mask region for this box
        roi_mask = motion_mask[y1:y2, x1:x2]
        motion_pixels = int(cv2.countNonZero(roi_mask))
        motion_ratio = motion_pixels / float(box_area)

        min_pixels = getattr(self.config, 'per_object_motion_min_pixels', 150)
        min_ratio = getattr(self.config, 'per_object_motion_ratio', 0.02)

        is_active = (motion_pixels >= min_pixels) or (motion_ratio >= min_ratio and motion_pixels >= 60)
        return is_active, motion_ratio, motion_pixels


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class TrackedEntity:
    """Represents a tracked physical object across consecutive video frames."""
    def __init__(self, track_id: int, bbox_xyxy: Tuple[float, float, float, float], class_name: str):
        self.track_id = track_id
        self.bbox_xyxy = bbox_xyxy
        self.class_name = class_name
        self.centroid = ((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)
        self.centroid_history: List[Tuple[float, float]] = [self.centroid]
        self.active_frames: int = 0
        self.hold_frames: int = 0
        self.missed_frames: int = 0
        self.is_active: bool = False
        self.motion_score: float = 0.0

    def update_position(self, bbox_xyxy: Tuple[float, float, float, float]):
        self.bbox_xyxy = bbox_xyxy
        self.centroid = ((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)
        self.centroid_history.append(self.centroid)
        if len(self.centroid_history) > 8:
            self.centroid_history.pop(0)
        self.missed_frames = 0

    def get_displacement(self) -> float:
        if len(self.centroid_history) < 3:
            return 0.0
        first_x, first_y = self.centroid_history[0]
        curr_x, curr_y = self.centroid_history[-1]
        return float(np.sqrt((curr_x - first_x) ** 2 + (curr_y - first_y) ** 2))


class MultiObjectMotionTracker:
    """
    Per-object tracking & motion discriminator for border analytics.
    Tracks multiple entities independently and discriminates physical motion on a PER-OBJECT basis.
    Stationary targets remain IDLE even if another entity in the frame is active.
    """
    def __init__(self, config: MotionGatingConfig):
        self.config = config
        self.next_id: int = 1
        self.tracks: Dict[int, TrackedEntity] = {}
        self.max_missed_frames: int = 15

    def update(
        self,
        detections: List[Any],
        motion_gater: BorderMotionGater,
        motion_mask: Optional[np.ndarray]
    ) -> List[Any]:
        """
        Matches detections with existing tracks, evaluates localized motion
        in each bounding box, and sets is_active and track_id on each detection.
        """
        if not detections:
            to_delete = []
            for tid, trk in self.tracks.items():
                trk.missed_frames += 1
                if trk.missed_frames > self.max_missed_frames:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return detections

        det_indices = list(range(len(detections)))
        track_ids = list(self.tracks.keys())
        matched_pairs: List[Tuple[int, int]] = []  # (det_idx, track_id)

        if track_ids:
            # Build IoU matrix
            iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)
            for d_idx, det in enumerate(detections):
                for t_idx, tid in enumerate(track_ids):
                    iou_matrix[d_idx, t_idx] = compute_iou(det.bbox_xyxy, self.tracks[tid].bbox_xyxy)

            # Greedy matching by highest IoU
            matched_dets = set()
            matched_trks = set()

            while True:
                max_val = -1.0
                best_d, best_t = -1, -1
                for d in range(len(detections)):
                    if d in matched_dets:
                        continue
                    for t in range(len(track_ids)):
                        if t in matched_trks:
                            continue
                        if iou_matrix[d, t] > max_val:
                            max_val = iou_matrix[d, t]
                            best_d, best_t = d, t

                if max_val >= 0.25 and best_d >= 0 and best_t >= 0:
                    matched_pairs.append((best_d, track_ids[best_t]))
                    matched_dets.add(best_d)
                    matched_trks.add(best_t)
                else:
                    break

            # Match remaining detections by centroid proximity (< 70 pixels)
            for d in range(len(detections)):
                if d in matched_dets:
                    continue
                det_cx = (detections[d].bbox_xyxy[0] + detections[d].bbox_xyxy[2]) / 2.0
                det_cy = (detections[d].bbox_xyxy[1] + detections[d].bbox_xyxy[3]) / 2.0

                best_dist = 70.0
                best_t = -1
                for t in range(len(track_ids)):
                    if t in matched_trks:
                        continue
                    trk_cx, trk_cy = self.tracks[track_ids[t]].centroid
                    dist = float(np.sqrt((det_cx - trk_cx) ** 2 + (det_cy - trk_cy) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best_t = t

                if best_t >= 0:
                    matched_pairs.append((d, track_ids[best_t]))
                    matched_dets.add(d)
                    matched_trks.add(best_t)

            for t, tid in enumerate(track_ids):
                if t not in matched_trks:
                    self.tracks[tid].missed_frames += 1

        matched_det_indices = {p[0] for p in matched_pairs}

        # Create new tracks for unmatched detections
        for d in det_indices:
            if d not in matched_det_indices:
                new_track = TrackedEntity(
                    track_id=self.next_id,
                    bbox_xyxy=detections[d].bbox_xyxy,
                    class_name=detections[d].class_name
                )
                self.tracks[self.next_id] = new_track
                matched_pairs.append((d, self.next_id))
                self.next_id += 1

        # Expire old tracks
        to_delete = [tid for tid, trk in self.tracks.items() if trk.missed_frames > self.max_missed_frames]
        for tid in to_delete:
            del self.tracks[tid]

        # Evaluate motion PER OBJECT
        hold_window = getattr(self.config, 'object_motion_hold_frames', 12)
        vel_threshold = getattr(self.config, 'min_high_velocity_px', 24.0)

        for det_idx, tid in matched_pairs:
            det = detections[det_idx]
            trk = self.tracks[tid]
            trk.update_position(det.bbox_xyxy)

            # Localized motion analysis strictly inside this object bounding box
            box_motion_active, motion_ratio, motion_px = motion_gater.evaluate_box_motion(det.bbox_xyxy, motion_mask)
            disp = trk.get_displacement()

            # Active if localized motion exists inside bbox OR substantial physical displacement occurred
            instant_motion = box_motion_active or (disp >= vel_threshold)

            if instant_motion:
                trk.active_frames += 1
                trk.hold_frames = hold_window
            else:
                trk.active_frames = 0
                trk.hold_frames = max(0, trk.hold_frames - 1)

            # Temporal confirmation & hold window to prevent flickering
            if trk.active_frames >= self.config.consecutive_motion_frames or trk.hold_frames > 0:
                trk.is_active = True
            else:
                trk.is_active = False

            trk.motion_score = motion_ratio

            # Assign per-detection properties
            det.is_active = trk.is_active
            det.track_id = trk.track_id
            det.motion_score = motion_ratio

        return detections
