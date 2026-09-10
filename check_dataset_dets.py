from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")
# Test on a dummy person image or webcam frame or sample
img = cv2.imread("dataset/images/train/border_train_000.jpg")
results = model(img)
print("Detections on border_train_000 with yolov8s.pt:", len(results[0].boxes))

# Let's inspect the images in dataset/images/train/
import glob
imgs = glob.glob("dataset/images/train/*.jpg")
for p in imgs[:5]:
    r = model(p, verbose=False)[0]
    print(f"{p}: {len(r.boxes)} boxes detected, classes: {[int(c) for c in r.boxes.cls] if len(r.boxes)>0 else []}")
