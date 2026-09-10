from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for _ in range(5):
    ret, frame = cap.read()
cap.release()

if ret and frame is not None:
    model = YOLO("yolov8s.pt")
    r = model.predict(frame, verbose=False)[0]
    print(f"YOLO predict on camera frame: {len(r.boxes)} boxes found!")
    for b in r.boxes:
        cls_id = int(b.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(b.conf[0].item())
        print(f" -> Found: {cls_name} (ID: {cls_id}) with confidence {conf:.3f}, bbox: {b.xyxy[0].tolist()}")
