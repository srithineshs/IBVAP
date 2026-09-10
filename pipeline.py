"""
IBVAP - Intelligent Border Video Analytics Platform
High-FPS Pipeline Execution for Object & Movement Detection in Border Areas
Optimized with Multi-threaded Ingestion, ONNX Runtime, and Motion Gated Inference.
"""

import os
import sys
import threading
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Add project root and src directory to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
# pyrefly: ignore [missing-import]
import numpy as np

# Enable global OpenCV CPU SIMD optimizations
cv2.setUseOptimized(True)

try:
    from src.config import BorderPipelineConfig, CLASS_COLORS
    from src.motion_gating import BorderFramePreprocessor, BorderMotionGater, MultiObjectMotionTracker
    from src.detector import BorderObjectDetector
except ImportError:
    from config import BorderPipelineConfig, CLASS_COLORS  # type: ignore
    from motion_gating import BorderFramePreprocessor, BorderMotionGater, MultiObjectMotionTracker  # type: ignore
    from detector import BorderObjectDetector  # type: ignore


class ThreadedVideoStream:
    """
    Dedicated background thread for zero-latency camera frame ingestion.
    Prevents cv2.VideoCapture.read() from stalling the main processing loop.
    """
    def __init__(self, source, width: int = 640, height: int = 480):
        self.source = source
        self.is_webcam = False
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
            self.is_webcam = True
        elif isinstance(source, int):
            self.is_webcam = True

        try:
            if self.is_webcam and sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            elif isinstance(self.source, str) and (self.source.startswith("rtsp://") or self.source.startswith("http://")):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)
        except Exception as e:
            print(f"[!] Warning during video capture open: {e}")
            self.cap = cv2.VideoCapture()

        if not self.cap.isOpened():
            print(f"[!] ERROR: Could not open video source: '{self.source}'")
            print(f"    Check network connection, IP address, credentials, and camera lock status.")

        if self.is_webcam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        if self.ret and self.frame is not None:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


class BorderAnalyticsPipeline:
    def __init__(self, config: BorderPipelineConfig):
        self.config = config
        self.preprocessor = BorderFramePreprocessor(config.preprocessing)
        self.detector = BorderObjectDetector(config.detector)
        self.motion_gater: Optional[BorderMotionGater] = None
        self.tracker = MultiObjectMotionTracker(config.motion_gating)
        
        # Frame and FPS metrics
        self.frame_count = 0
        self.fps_history = []
        
        # Output directory for event logging
        self.log_dir = Path("logs/detections")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_log_path = self.log_dir / f"{config.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def run(self, display: bool = True, save_video_path: Optional[str] = None):
        """Starts ultra-fast video stream ingestion and analytics loop."""
        print(f"\n[*] Connecting to High-Speed Border Video Feed: {self.config.source} (Camera: {self.config.camera_id})...")
        
        # Start Threaded Camera Reader for minimum latency
        stream = ThreadedVideoStream(self.config.source, width=640, height=480)
        
        if not stream.isOpened():
            print(f"[!] Error: Unable to connect to video source: {self.config.source}")
            return

        width = stream.width
        height = stream.height
        fps_source = stream.fps
        
        print(f"[+] Connected successfully. Resolution: {width}x{height} @ {fps_source:.1f} Native FPS")
        print(f"[*] Engine Backend: {self.detector.backend.upper()} | Model: {self.detector.model_path}")
        
        # Initialize Motion Gater
        self.motion_gater = BorderMotionGater(
            config=self.config.motion_gating,
            roi_polygon_norm=self.config.roi_polygon,
            frame_shape=(height, width)
        )

        # Video Writer (if saving output)
        video_writer = None
        if save_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(save_video_path, fourcc, fps_source, (width, height))
            print(f"[+] Recording stream to: {save_video_path}")

        poly_pts = np.array([
            [int(x * width), int(y * height)] for x, y in self.config.roi_polygon
        ], dtype=np.int32)

        consecutive_empty = 0
        try:
            while True:
                start_time = time.perf_counter()
                ret, frame = stream.read()
                
                if not ret or frame is None:
                    consecutive_empty += 1
                    if consecutive_empty > 50:
                        print("[!] Stream ended or camera disconnected.")
                        break
                    time.sleep(0.01)
                    continue
                
                consecutive_empty = 0
                self.frame_count += 1
                timestamp_iso = datetime.now().astimezone().isoformat()
                
                # 1. Fast Adaptive Enhancement
                enhanced_frame = self.preprocessor.enhance(frame)

                # 2. Fast Motion Check First (Takes ~1.0 ms)
                cached_centroid = getattr(self, 'cached_centroid', None)
                is_frame_motion, motion_mask, motion_boxes = self.motion_gater.check_motion(
                    enhanced_frame,
                    detected_centroid=cached_centroid
                )
                
                # 3. Motion-Guided High-Speed Inference
                # Run detection every 2nd frame when motion is detected, or every 6th frame in idle state
                should_run_detector = (
                    (is_frame_motion and self.frame_count % 2 == 0) or
                    (not is_frame_motion and self.frame_count % 6 == 0) or
                    not hasattr(self, 'cached_detections')
                )

                if should_run_detector:
                    raw_detections = self.detector.detect(enhanced_frame, motion_boxes=motion_boxes if is_frame_motion else None)
                    
                    # Filter detections inside ROI
                    detections = []
                    primary_centroid = None
                    for det in raw_detections:
                        foot_x, foot_y = det.foot_point
                        inside_roi = cv2.pointPolygonTest(poly_pts, (foot_x, foot_y), False) >= 0
                        if inside_roi:
                            detections.append(det)
                            if primary_centroid is None and det.class_name == "person":
                                x1, y1, x2, y2 = det.bbox_xyxy
                                primary_centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                                
                    self.cached_detections = detections
                    self.cached_centroid = primary_centroid
                else:
                    detections = getattr(self, 'cached_detections', [])

                # 3b. Per-Object Motion & Activity Tracking
                detections = self.tracker.update(detections, self.motion_gater, motion_mask)
                active_count = sum(1 for d in detections if getattr(d, 'is_active', False))
                total_targets = len(detections)
                has_active_targets = active_count > 0

                # 4. Structured Telemetry & Event Logging (Only on active targets)
                if detections and (has_active_targets or is_frame_motion):
                    event_payload = {
                        "camera_id": self.config.camera_id,
                        "timestamp": timestamp_iso,
                        "frame_id": self.frame_count,
                        "movement_detected": has_active_targets or is_frame_motion,
                        "active_target_count": active_count,
                        "total_target_count": total_targets,
                        "detections": [d.to_dict() for d in detections]
                    }
                    with open(self.json_log_path, "a") as f:
                        f.write(json.dumps(event_payload) + "\n")

                # Compute Processing Speed (FPS)
                elapsed = time.perf_counter() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 30.0
                self.fps_history.append(fps)
                if len(self.fps_history) > 20:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # 5. Fast Minimal HUD Rendering with Per-Object State Coloring
                annotated = frame.copy()

                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                    foot_x, foot_y = [int(v) for v in det.foot_point]
                    
                    is_active = getattr(det, 'is_active', False)
                    # Red for ACTIVE intruder/moving target, Green for IDLE stationary target
                    box_color = (0, 0, 255) if is_active else (0, 255, 0)
                    
                    # Target BBox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Target Label
                    state_tag = "ACTIVE" if is_active else "IDLE"
                    track_str = f"#{det.track_id} " if getattr(det, 'track_id', None) is not None else ""
                    label = f"{det.class_name.upper()} {track_str}[{state_tag}] {det.confidence:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (x1, y1 - 18), (x1 + lw + 4, y1), box_color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    # Ground Contact Point
                    cv2.circle(annotated, (foot_x, foot_y), 4, (0, 0, 255), -1)

                # Status Strip: Clean Minimal FPS, Movement Status & Target Counts
                status_text = "Movement: ACTIVE" if (has_active_targets or is_frame_motion) else "Movement: IDLE"
                status_color = (0, 0, 255) if (has_active_targets or is_frame_motion) else (0, 255, 0)
                
                status_banner = f"FPS: {avg_fps:.1f}  |  {status_text}  |  Active: {active_count}/{total_targets}"
                (bw, bh), _ = cv2.getTextSize(status_banner, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(annotated, (10, 10), (25 + bw, 42), (0, 0, 0), -1)
                cv2.putText(annotated, status_banner, 
                            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2)

                if video_writer:
                    video_writer.write(annotated)

                if display:
                    cv2.imshow("IBVAP - Border Surveillance Analytics", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[*] User interrupted stream.")
                        break

        finally:
            stream.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[+] Analytics pipeline shut down cleanly. Event logs stored at: {self.json_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IBVAP High-FPS Border Object & Movement Detection")
    parser.add_argument("--source", type=str, default=None, help="Direct video source (webcam ID, full RTSP URL, or video path)")
    parser.add_argument("--cam", "--channel", dest="channel", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="Camera/Channel Number (1, 2, 3, 4...)")
    parser.add_argument("--stream", type=str, default="sub", choices=["sub", "main", "1", "2"], help="Stream quality: 'sub' (fast, high FPS) or 'main' (1080p)")
    parser.add_argument("--ip", type=str, default="10.245.106.240", help="Camera/NVR IP Address (default: 10.245.106.240)")
    parser.add_argument("--user", type=str, default="admin", help="Camera username")
    parser.add_argument("--pwd", type=str, default="admin123", help="Camera password")
    parser.add_argument("--model", type=str, default="weights/yolo_border_fast.onnx", help="Path to ONNX/PyTorch model (default: weights/yolo_border_fast.onnx)")
    parser.add_argument("--cam-id", type=str, default=None, help="Camera Identifier (e.g. CAM_01, CAM_02)")
    parser.add_argument("--save-video", type=str, default=None, help="Path to save annotated MP4 video output")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    
    args = parser.parse_args()
    
    # Determine video source
    if args.source:
        source = args.source
    else:
        # Construct Hikvision RTSP channel format:
        # Channel 1 -> Main: 101, Sub: 102
        # Channel 2 -> Main: 201, Sub: 202
        # Channel 3 -> Main: 301, Sub: 302
        # Channel 4 -> Main: 401, Sub: 402
        stream_code = "02" if args.stream in ["sub", "2"] else "01"
        channel_code = f"{args.channel}{stream_code}"
        source = f"rtsp://{args.user}:{args.pwd}@{args.ip}:554/Streaming/Channels/{channel_code}"
    
    cam_id = args.cam_id or f"CAMERA_{args.channel:02d}"
    
    config = BorderPipelineConfig()
    config.source = source
    config.camera_id = cam_id
    config.detector.model_path = args.model
    
    pipeline = BorderAnalyticsPipeline(config)
    pipeline.run(display=not args.no_display, save_video_path=args.save_video)
