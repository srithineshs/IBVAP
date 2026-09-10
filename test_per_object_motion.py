import sys
from pathlib import Path
import numpy as np
import cv2

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config import MotionGatingConfig
from src.motion_gating import BorderMotionGater, MultiObjectMotionTracker
from src.detector import DetectionResult

def run_test():
    print("[*] Starting multi-target per-object motion verification test...")
    config = MotionGatingConfig(
        enabled=True,
        history=50,
        consecutive_motion_frames=2,
        per_object_motion_min_pixels=100,
        per_object_motion_ratio=0.02,
        object_motion_hold_frames=5
    )
    
    frame_h, frame_w = 480, 640
    roi = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    motion_gater = BorderMotionGater(config, roi, (frame_h, frame_w))
    tracker = MultiObjectMotionTracker(config)

    # Define 4 simulated persons across the frame
    # Person 1 (left), Person 2 (mid-left), Person 3 (mid-right), Person 4 (right)
    person_boxes = [
        (40.0, 100.0, 140.0, 360.0),   # P1: stationary
        (180.0, 100.0, 280.0, 360.0),  # P2: MOVING
        (340.0, 100.0, 440.0, 360.0),  # P3: stationary
        (500.0, 100.0, 600.0, 360.0),  # P4: stationary
    ]

    base_frame = np.full((frame_h, frame_w, 3), 120, dtype=np.uint8)

    # 1. Warm up background subtractor with stationary frames
    print("    Phase 1: Warming up background model (stationary state)...")
    for i in range(10):
        frame = base_frame.copy()
        is_motion, mask, _ = motion_gater.check_motion(frame)
        
        detections = [
            DetectionResult(0, "person", 0.9, box, ((box[0]+box[2])/2, box[3]))
            for box in person_boxes
        ]
        detections = tracker.update(detections, motion_gater, mask)

    # After warm-up, all 4 should be IDLE
    for idx, d in enumerate(detections):
        print(f"      Target #{d.track_id}: is_active={d.is_active}, state={d.to_dict()['state']}")
        assert not d.is_active, f"Target #{d.track_id} should be IDLE during warm-up!"

    print("[+] Phase 1 passed: All 4 targets are IDLE when nobody moves.")

    # 2. Simulate movement strictly on Person 2
    print("    Phase 2: Simulating motion on Person 2 ONLY (Person 1, 3, 4 remain still)...")
    for i in range(6):
        frame = base_frame.copy()
        # Add dynamic motion inside Person 2's bounding box
        noise = np.random.randint(0, 255, (200, 80, 3), dtype=np.uint8)
        frame[120:320, 190:270] = noise
        
        is_motion, mask, _ = motion_gater.check_motion(frame)
        
        detections = [
            DetectionResult(0, "person", 0.9, box, ((box[0]+box[2])/2, box[3]))
            for box in person_boxes
        ]
        detections = tracker.update(detections, motion_gater, mask)

    # Check states: Person 2 MUST be ACTIVE, Persons 1, 3, 4 MUST be IDLE!
    p1 = detections[0]
    p2 = detections[1]
    p3 = detections[2]
    p4 = detections[3]

    print(f"      Target 1 (Left):       state={p1.to_dict()['state']}, is_active={p1.is_active}, motion_score={p1.motion_score:.3f}")
    print(f"      Target 2 (Mid - MOVE): state={p2.to_dict()['state']}, is_active={p2.is_active}, motion_score={p2.motion_score:.3f}")
    print(f"      Target 3 (Mid-Right):  state={p3.to_dict()['state']}, is_active={p3.is_active}, motion_score={p3.motion_score:.3f}")
    print(f"      Target 4 (Right):      state={p4.to_dict()['state']}, is_active={p4.is_active}, motion_score={p4.motion_score:.3f}")

    assert p2.is_active, "Person 2 MUST be predicted as ACTIVE!"
    assert not p1.is_active, "Person 1 MUST remain IDLE!"
    assert not p3.is_active, "Person 3 MUST remain IDLE!"
    assert not p4.is_active, "Person 4 MUST remain IDLE!"

    print("[+] Phase 2 passed: Only Person 2 is ACTIVE. Persons 1, 3, and 4 stay IDLE!")
    print("[SUCCESS] All multi-target per-object motion tests passed!")

if __name__ == "__main__":
    run_test()
