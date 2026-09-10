"""
Centralized Configuration for IBVAP Border Object & Movement Detection Engine
Uses standard library dataclasses (zero external dependency).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# Target Object Class Taxonomy (Full 80 COCO classes mapped to human-readable names)
CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Standard Border Class Mapping
BORDER_CLASS_NAMES = CLASS_MAP

# Color mapping for visualization (BGR)
CLASS_COLORS = {
    "person": (0, 0, 255),            # Red: Person Intruder (High Priority)
    "civilian_vehicle": (0, 165, 255), # Orange: Civilian Vehicle
    "heavy_vehicle": (0, 140, 255),    # Deep Orange: Heavy Truck/Bus
    "military_utility_vehicle": (0, 255, 255) # Yellow: Security Vehicle
}

@dataclass
class PreprocessingConfig:
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    enable_gamma_correction: bool = True
    gamma_value: float = 1.3  # Brightens dark/night border scenes

@dataclass
class MotionGatingConfig:
    enabled: bool = True
    history: int = 300
    var_threshold: float = 60.0              # Strict variance to ignore sensor noise and small lighting shifts
    detect_shadows: bool = False
    min_motion_area_pixels: int = 6500       # High threshold: requires significant physical movement (not minor twitches)
    min_high_velocity_px: float = 24.0       # Minimum centroid displacement (pixels) to classify as high movement
    dilation_kernel_size: int = 5
    consecutive_motion_frames: int = 2       # Must sustain high movement across frames
    per_object_motion_min_pixels: int = 150  # Minimum motion pixels in bbox to trigger individual active state
    per_object_motion_ratio: float = 0.02    # Minimum motion pixel ratio (2%) relative to bbox area
    object_motion_hold_frames: int = 12      # Hold ACTIVE state for ~12 frames to prevent flicker

@dataclass
class DetectorConfig:
    model_path: str = "weights/yolo_border_fast.onnx"  # Hardware-accelerated ONNX model (High FPS)
    input_width: int = 480                    # 480x480 yields 35-60+ FPS on CPU with crisp detection
    input_height: int = 480
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    allowed_class_ids: Optional[List[int]] = None  # None: Detect all supported object classes
    
    # Class-specific confidence overrides (empty by default to allow smooth detection)
    class_confidence_thresholds: Dict[int, float] = field(default_factory=dict)

@dataclass
class BorderPipelineConfig:
    camera_id: str = "BOP_NORTH_SECTOR_01"
    source: str = "0"  # 0 for webcam, RTSP URL, or video file path
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    motion_gating: MotionGatingConfig = field(default_factory=MotionGatingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    
    # Region of Interest (ROI) polygon normalized [0.0 - 1.0] coordinates: (x, y) - Full Frame
    roi_polygon: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0],  # Full 100% field of view
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    # Temporal confirmation window
    temporal_confirmation_frames: int = 3
