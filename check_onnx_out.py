import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("weights/yolo26_border.onnx", providers=['CPUExecutionProvider'])
print("Inputs:", [i.name for i in session.get_inputs()], "Shape:", session.get_inputs()[0].shape)
print("Outputs:", [o.name for o in session.get_outputs()], "Shape:", session.get_outputs()[0].shape)

img = cv2.imread("dataset/images/train/border_train_000.jpg")
img_resized = cv2.resize(img, (480, 480))
blob = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

out = session.run(None, {session.get_inputs()[0].name: blob})[0]
print("Out shape:", out.shape)
print("Out min:", out.min(), "max:", out.max())

# YOLOv8 output shape is [1, 84, 4725]
# where 4 are bbox (cx, cy, w, h) and 80 are class scores
# Let's inspect scores:
output = out[0].T # [4725, 84]
boxes = output[:, :4]
scores = output[:, 4:]
print("Scores max:", scores.max(), "at index:", np.unravel_index(np.argmax(scores), scores.shape))
print("Class with max score:", np.argmax(scores, axis=1)[np.unravel_index(np.argmax(scores), scores.shape)[0]])
