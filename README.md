# IBVAP: Border Surveillance Object & Movement Detection Engine

> **Targeted Solution for Sashastra Seema Bal (SSB), Ministry of Home Affairs (MHA)**
> AI-Based Intelligent Video Analytics Platform for Border Surveillance using Existing CCTV Infrastructure (Phase 2 Master Implementation).

---

## 🚀 Key Capabilities Implemented

1. **Border Taxonomy Detection**:
   - `person` (Standing, crouching, crawling, armed/backpacks)
   - `civilian_vehicle` (Cars, motorcycles, pickup trucks)
   - `heavy_vehicle` (Trucks, tractors, commercial freight)
   - `military_utility_vehicle` (Patrol jeeps, security vehicles)
   - `animal` (Livestock, wildlife filtering for false alarm suppression)
2. **Ground Contact (Foot-Point) Extraction**:
   - Computes precise bottom-center `(x_mid, y_bottom)` ground-contact coordinates for accurate border fence tripwire calculations.
3. **Motion Gating & ROI Optimization**:
   - Background subtraction (MOG2) inside a user-defined polygon ROI to gate the deep learning model, slashing edge GPU load by 60–80%.
4. **Low-Light & IR Contrast Enhancement**:
   - Built-in CLAHE and Gamma correction for night-time and foggy perimeter cameras.
5. **RegNet-Y Edge Backbone with SE Attention**:
   - Integrated Meta AI's **RegNet-Y** (Grouped Convolutions + Squeeze-and-Excitation attention) for enhanced low-contrast intruder detection and ultra-low FLOPs footprint on edge devices.
6. **Dual-Backend Acceleration**:
   - Supports high-throughput **ONNX Runtime (CUDA/TensorRT/CPU)** and **PyTorch/Ultralytics**.

---

## 📁 Directory Structure

```
Obj_detection/
├── data/
│   ├── border_surveillance.yaml    # Dataset paths and class mapping
│   └── yolo_regnet.yaml            # RegNet-Y YOLO model architecture
├── src/
│   ├── config.py                   # Centralized pipeline configurations & thresholds
│   ├── detector.py                 # ONNX/PyTorch detector & foot-point calculator
│   ├── motion_gating.py            # Motion gating & low-light preprocessor
│   ├── pipeline.py                 # Main video analytics runner with HUD overlay
│   ├── train.py                    # Model training & ONNX export script
│   └── models/
│       └── regnet.py               # Custom RegNet-Y stages & SE attention blocks
├── weights/                        # Trained ONNX / TensorRT / PyTorch model weights
├── logs/                           # Standardized JSON event logs
└── requirements.txt                # Python dependencies
```

---

## 🛠️ Quickstart Guide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Train Custom Border Model (RegNet-Y Backbone)

Place your annotated border dataset into the `dataset/` folder matching `data/border_surveillance.yaml`, then run:

```bash
# Train with RegNet-Y Backbone (Grouped Conv + Squeeze-and-Excitation):
python -m src.train --data data/border_surveillance.yaml --model data/yolo_regnet.yaml --epochs 150 --imgsz 640 --batch 16
```
*Trained weights will automatically be benchmarked and exported to `weights/yolo_regnet_border.onnx` and `weights/yolo_regnet_border.pt`.*

### 3. Run Real-Time Border Analytics Pipeline

**From a Live Webcam:**
```bash
python -m src.pipeline --source 0 --cam-id BOP_NORTH_01
```

**From an RTSP CCTV Stream:**
```bash
python -m src.pipeline --source "rtsp://admin:pass@192.168.1.100:554/h264Preview_01_main" --cam-id BOP_CHECKPOST_02
```

**From a Recorded Video File (Saving Output):**
```bash
python -m src.pipeline --source sample_border_video.mp4 --save-video output_annotated.mp4
```

---

## 📊 Standardized Event Telemetry Output (JSONL)

Every detection automatically records to `logs/detections/` matching the master C2 schema:

```json
{
  "camera_id": "BOP_NORTH_SECTOR_01",
  "timestamp": "2026-09-07T21:31:38.120Z",
  "frame_id": 14205,
  "motion_triggered": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.934,
      "bbox_xyxy": [450.2, 310.5, 510.8, 480.0],
      "foot_point": [480.5, 480.0]
    }
  ]
}
```
