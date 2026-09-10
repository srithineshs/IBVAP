import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX session
session = ort.InferenceSession("weights/yolo_border_fast.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Capture a camera frame
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for _ in range(5):
    ret, frame = cap.read()
cap.release()

if ret and frame is not None:
    h, w = frame.shape[:2]
    # Let's resize to 480x480
    resized = cv2.resize(frame, (480, 480))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    
    outputs = session.run(None, {input_name: blob})[0]
    out = outputs[0].T # [4725, 84]
    boxes = out[:, :4]
    scores = out[:, 4:] # 80 classes
    
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    mask = max_scores >= 0.40
    print(f"Candidates above 0.40: {np.sum(mask)}")
    for cid, conf in zip(class_ids[mask], max_scores[mask]):
        print(f" - Class {cid} with conf {conf:.3f}")
