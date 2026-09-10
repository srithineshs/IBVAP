import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("weights/yolo_border_fast.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print("ONNX Model Input Shape:", input_shape)

frame = cv2.imread("scratch/cctv_frame.jpg")
h, w = frame.shape[:2]
print(f"Image shape: {w}x{h}")

# Resize to 480x480 (or 640x640)
for target_size in [480, 640]:
    resized = cv2.resize(frame, (target_size, target_size))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    
    outputs = session.run(None, {input_name: blob})[0]
    out = outputs[0].T
    boxes = out[:, :4]
    scores = out[:, 4:]
    
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    top_indices = np.argsort(max_scores)[::-1][:10]
    print(f"\nTop 10 raw scores at size {target_size}x{target_size}:")
    for idx in top_indices:
        print(f"  Class {class_ids[idx]} with score {max_scores[idx]:.4f}")
