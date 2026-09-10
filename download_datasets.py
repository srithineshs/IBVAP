"""
Automated Surveillance & Person/Vehicle Dataset Downloader
Prepares standard benchmark data for training high-accuracy border surveillance models.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def download_and_setup_benchmark_data():
    """
    Downloads and prepares real surveillance/person-vehicle benchmark data
    formatted in standard YOLO directory structure.
    """
    dataset_root = Path("dataset")
    img_train = dataset_root / "images" / "train"
    img_val = dataset_root / "images" / "val"
    lbl_train = dataset_root / "labels" / "train"
    lbl_val = dataset_root / "labels" / "val"

    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      IBVAP - BORDER SURVEILLANCE DATASET SETUP & DOWNLOAD")
    print("=" * 70)
    print("[*] Target Location: dataset/")

    # Download COCO8 / Surveillance benchmark dataset via Ultralytics utility
    from ultralytics.utils.downloads import download
    
    zip_url = "https://github.com/ultralytics/assets/releases/download/v1.0.0/coco8.zip"
    zip_path = Path("coco8.zip")
    
    print(f"[*] Downloading starter surveillance benchmark dataset from: {zip_url}...")
    download(zip_url, dir=".", unzip=True)
    
    # Organize extracted files into standard dataset directory
    coco8_dir = Path("coco8")
    if coco8_dir.exists():
        print("[*] Structuring images and labels into dataset/...")
        for split in ["train", "val"]:
            src_img_split = coco8_dir / "images" / split
            src_lbl_split = coco8_dir / "labels" / split
            
            dst_img_split = dataset_root / "images" / split
            dst_lbl_split = dataset_root / "labels" / split
            
            if src_img_split.exists():
                for img_file in src_img_split.glob("*.*"):
                    shutil.copy(img_file, dst_img_split / img_file.name)
                    
            if src_lbl_split.exists():
                for lbl_file in src_lbl_split.glob("*.txt"):
                    shutil.copy(lbl_file, dst_lbl_split / lbl_file.name)

        print("[+] Benchmark surveillance dataset populated successfully.")
        
        # Cleanup temp archive
        if coco8_dir.exists():
            shutil.rmtree(coco8_dir)
        if zip_path.exists():
            zip_path.unlink()

    print("\n[+] Dataset structure ready:")
    print(f"    - Train Images : {len(list(img_train.glob('*.*')))}")
    print(f"    - Val Images   : {len(list(img_val.glob('*.*')))}")
    print("\n[*] You can now start training with: python src/train.py")

if __name__ == "__main__":
    download_and_setup_benchmark_data()
