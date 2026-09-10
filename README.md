# IBVAP: Intelligent Border Video Analytics Platform

> **Targeted Solution for Sashastra Seema Bal (SSB) & Ministry of Home Affairs (MHA)**
> AI-Driven Software-Defined Surveillance Platform transforming standard CCTV into an Intelligent Security Network.

---

## 🚀 Key Capabilities Implemented

1. **Layer 1: Ingestion & Decoding**:
   - RTSP IP cameras, local video files, USB webcams, and automated synthetic border simulation.
2. **Layer 2: Adaptive Environmental Preprocessing**:
   - 8-condition classifier (`DAY_NORMAL`, `DAY_OVERLIT`, `TWILIGHT`, `HAZY`, `NIGHT_IR`, `NIGHT_AMBIENT`, `NIGHT_DARK`, `RAIN`).
   - Adaptive enhancement: CLAHE, Gamma LUTs, Dehazing, and thermal BONE colormap rendering.
3. **Layer 3: Primary Detection & Tracking**:
   - **Quantized INT8 & FP16 ONNX detection** for maximum FPS on low-end CPUs.
   - Ground contact **(Foot-Point)** calculation `(x_mid, y_bottom)` for precise boundary tripwires.
   - **OC-SORT Multi-Object Tracker**: Kalman filtering, Observation-Centric Momentum (OCM), and Appearance Re-ID.
4. **Layer 4: Secondary Inference & Routing**:
   - **Branch 1 (Human)**:
     - Standalone SCRFD ONNX Face Detection (`scrfd_int8.onnx`).
     - 5-point landmark affine partial 2D alignment.
     - **Unified Multi-Task Model** (`ir_se50_int8.onnx` backbone):
       - 512-d Face Recognition embedding.
       - Head Pose Regressor (Yaw, Pitch, Roll in degrees).
       - Liveness / Anti-Spoofing Classifier (Score 0.0 - 1.0).
     - Quality estimation & dynamic adaptive cosine matching threshold.
     - Posture Anomaly Analyzer (Fallen person / crawling intruder detection).
   - **Branch 2 (Vehicle)**:
     - License plate candidate extraction & contrast preprocessing.
     - Alphanumeric OCR text reading.
     - Blacklisted plate verification, wrong-way detection & speed estimation.
   - **Branch 3 (Object)**:
     - Stationary luggage & unattended baggage detection.
   - **Spatial Intelligence**:
     - Multi-zone polygonal virtual fences: **Buffer Strip (Warning)** & **Zero-Line (Critical Breach)**.
5. **Layer 5: Standardized C2 Telemetry & Persistence**:
   - Master JSON telemetry schema (compliant with SSB Command & Control).
   - JSONL event logger and SQLite queryable database.
6. **Layer 6: Complex Event Processing (CEP) Threat Scoring**:
   - Multi-factor threat scoring ($0 - 100$) with real-time alert dispatch.
7. **Tactical Web Command & Control Dashboard**:
   - FastAPI + WebSocket backend with live video stream, real-time threat gauge, and active incident feed.

---

## 📁 Directory Layout

```
ibvap_integrated/
├── configs/
│   ├── pipeline_config.yaml         # Central system configuration
│   ├── zones_config.json            # Virtual fence polygons & tripwires
│   ├── watchlist_config.yaml        # FRS watchlist matching thresholds
│   └── vehicle_blacklist.json       # Blacklisted vehicle database
├── models/
│   ├── primary/                     # Detection weights (Pose_detection.onnx, yolo_border_fast.onnx)
│   ├── face/                        # Face weights (scrfd.onnx, ir_se50.pt, pose_head.onnx, liveness_head.onnx)
│   └── quantized/                   # Ultra-fast INT8 models (yolo_border_fast_int8.onnx, scrfd_int8.onnx, ir_se50_int8.onnx)
├── src/
│   ├── ingestion/                   # Stream manager & synthetic border stream
│   ├── preprocessing/               # 8-condition classifier & adaptive enhancer
│   ├── detection/                   # Primary detector & foot-point math
│   ├── tracking/                    # OC-SORT multi-object tracker
│   ├── secondary/
│   │   ├── face/                    # SCRFD, aligner, unified model, watchlist engine, posture
│   │   ├── vehicle/                 # Plate detector, OCR, and vehicle analytics
│   │   └── object/                  # Abandoned baggage detector
│   ├── spatial/                     # Multi-zone virtual fence engine
│   ├── cep/                         # Threat scoring engine & alert manager
│   ├── telemetry/                   # C2 JSON schema, JSONL logger, SQLite DB
│   ├── visualization/               # Tactical border surveillance HUD renderer
│   ├── quantization/                # INT8 dynamic quantization script
│   └── pipeline.py                  # Master End-to-End Pipeline Orchestrator
├── web/                             # FastAPI Command Center Web Dashboard
│   ├── server.py                    # Web server with MJPEG feed & WebSocket telemetry
│   └── static/                      # Tactical Dark Mode UI (HTML, CSS, JS)
├── scripts/
│   ├── enroll_watchlist.py          # Watchlist enrollment utility
│   ├── calibrate_zones.py           # Interactive polygon calibration GUI
│   ├── run_pipeline.py              # CLI runner for video, webcam, or RTSP
│   └── run_server.py                # Command Center Web Server launcher
└── tests/                           # Comprehensive test suite
```

---

## 🛠️ Quickstart Guide

### 1. Installation
```bash
cd ibvap_integrated
pip install -r requirements.txt
```

### 2. Enroll Suspect Watchlist
```bash
python scripts/enroll_watchlist.py --input data/watchlist_raw --out data/watchlist_db/embeddings.pkl
```

### 3. Run Pipeline (CLI / Native Desktop Window)
```bash
# Run on built-in synthetic border simulation:
python scripts/run_pipeline.py --source synthetic

# Run on live webcam:
python scripts/run_pipeline.py --source 0

# Run on RTSP IP Camera:
python scripts/run_pipeline.py --source "rtsp://admin:pass@192.168.1.100:554/stream"

# Run on video file and save output:
python scripts/run_pipeline.py --source sample_border.mp4 --save-video output_annotated.mp4
```

### 4. Launch Tactical Web Command Center
```bash
python scripts/run_server.py --port 8000
```
Open **`http://localhost:8000`** in your browser to view the real-time tactical dashboard.

### 5. Run Automated Verification Tests
```bash
pytest tests/ -v
```
