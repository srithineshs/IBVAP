"""
Automated Border Surveillance Dataset Builder & Collector
Captures annotated images from webcam or video files to train custom YOLO26 models.
"""

import cv2
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO

def collect_training_samples(source: str = "0", max_samples: int = 200, class_id: int = 0):
    """
    Auto-captures video frames and generates YOLO format annotations
    for continuous model fine-tuning.
    """
    dataset_dir = Path("dataset")
    img_train_dir = dataset_dir / "images" / "train"
    lbl_train_dir = dataset_dir / "labels" / "train"
    img_val_dir = dataset_dir / "images" / "val"
    lbl_val_dir = dataset_dir / "labels" / "val"
    
    for d in [img_train_dir, lbl_train_dir, img_val_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing Auto-Collector for Dataset (Target Class: {class_id})...")
    
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[!] Unable to open source: {source}")
        return

    # Use pretrained base model for auto-pseudo-labeling
    model = YOLO("yolov8n.pt")
    count = 0

    print("[*] Press 'c' to capture a training frame, or 'q' to finish.")

    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        results = model.predict(frame, conf=0.4, classes=[class_id], verbose=False)[0]

        # Draw detected preview
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display_frame, f"Collected: {count}/{max_samples} | Press 'c' to save", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Dataset Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and len(results.boxes) > 0:
            count += 1
            # 80% train, 20% val
            is_val = (count % 5 == 0)
            target_img_dir = img_val_dir if is_val else img_train_dir
            target_lbl_dir = lbl_val_dir if is_val else lbl_train_dir

            img_name = f"border_sample_{count:04d}.jpg"
            txt_name = f"border_sample_{count:04d}.txt"

            cv2.imwrite(str(target_img_dir / img_name), frame)

            # Save normalized YOLO annotations
            h, w = frame.shape[:2]
            with open(target_lbl_dir / txt_name, "w") as f:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            print(f"[+] Saved sample {count}: {img_name}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Finished collection! Saved {count} samples to dataset/ folder.")
    print("[*] You can now run: python src/train.py to train your model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data for border model")
    parser.add_argument("--source", type=str, default="0", help="Webcam 0 or video path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--class-id", type=int, default=0, help="Target Class ID (0=Person)")
    args = parser.parse_args()

    collect_training_samples(source=args.source, max_samples=args.samples, class_id=args.class_id)
