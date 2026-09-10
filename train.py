"""
Border Surveillance Model Training & Export Script
Trained for Person/Vehicle/Animal Detection with Low-Light and Adverse Weather Augmentations
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
def train_border_model(
    data_yaml: str = "data/border_surveillance.yaml",
    model_variant: str = "yolov8s.pt",  # Lightweight & optimized for high stable FPS
    epochs: int = 150,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",  # GPU 0 or 'cpu'
    project_name: str = "runs/train",
    experiment_name: str = "yolo26_border",
    export_onnx: bool = True
):
    print("=" * 70)
    print("       IBVAP - YOLO26 BORDER SURVEILLANCE OBJECT DETECTION TRAINING")
    print("=" * 70)
    print(f"[*] Dataset Config     : {data_yaml}")
    print(f"[*] Base Weights       : {model_variant}")
    print(f"[*] Epochs             : {epochs}")
    print(f"[*] Input Resolution   : {imgsz}x{imgsz}")
    print(f"[*] Batch Size         : {batch_size}")
    print(f"[*] Compute Device     : {device}")
    print("=" * 70)

    # 1. Load Pretrained Backbone
    model = YOLO(model_variant)

    # 2. Train with Border Domain-Specific Hyperparameters & Augmentations
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name=experiment_name,
        exist_ok=True,
        
        # Optimizer & Learning Rate
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        cos_lr=True,
        
        # Augmentations for Border Scenarios (Low-Light, Camouflage & Small Distance Detection)
        hsv_h=0.015,       # Slight hue jitter
        hsv_s=0.7,         # Saturation variation (handles night IR vs bright daylight)
        hsv_v=0.6,         # Value/Brightness variation (handles deep shadows / night)
        degrees=5.0,       # Slight rotation for tilted pole-mounted cameras
        translate=0.1,     # Object translation
        scale=0.6,         # Multi-scale zooming for distant entities (small pixel footprint)
        mosaic=1.0,        # Mosaic augmentation for complex multi-object scenes
        mixup=0.15,        # Mixup for overlapping targets/foliage camouflage
        copy_paste=0.1,    # Copy-paste for sparse person classes
        
        # Loss Gains
        box=7.5,           # High box loss weight for precise foot-point localization
        cls=0.5,           # Classification loss
        dfl=1.5,           # Distribution Focal Loss
        
        # Validation & Checkpoints
        val=True,
        save=True,
        save_period=10,
        patience=30        # Early stopping patience
    )

    print("\n[+] Training completed successfully.")
    
    # 3. Validate Model Performance
    print("\n[*] Running Validation Evaluation...")
    metrics = model.val()
    print(f"[+] mAP@50     : {metrics.box.map50:.4f}")
    print(f"[+] mAP@50-95  : {metrics.box.map:.4f}")

    # 4. Export to Edge Optimized Formats (ONNX / TensorRT)
    best_weights_path = Path(project_name) / experiment_name / "weights" / "best.pt"
    
    if export_onnx and best_weights_path.exists():
        print(f"\n[*] Exporting Best Weights ({best_weights_path}) to ONNX for Edge Ingestion...")
        onnx_model = YOLO(str(best_weights_path))
        exported_path = onnx_model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,   # Static shape for max TensorRT/ONNX Runtime optimization
            simplify=True,   # ONNX-simplifier
            opset=12
        )
        print(f"[+] Successfully exported ONNX model to: {exported_path}")
        
        # Save a copy to weights/
        os.makedirs("weights", exist_ok=True)
        import shutil
        target_onnx = "weights/yolo26_border.onnx"
        target_pt = "weights/yolo26_border.pt"
        shutil.copy(exported_path, target_onnx)
        shutil.copy(str(best_weights_path), target_pt)
        print(f"[+] YOLO26 model successfully saved to: {target_onnx} & {target_pt}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IBVAP Border Object Detector")
    parser.add_argument("--data", type=str, default="data/border_surveillance.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO base weights (e.g. yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    
    args = parser.parse_args()
    
    train_border_model(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    )
