import torch
import torch.nn as nn
import torchvision.models as tv_models
import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut
from ultralytics import YOLO

class RegNetBackboneStage(nn.Module):
    """
    RegNet Backbone Stage Wrapper supporting pretrained weights.
    stage: 0=stem, 1=block1 (P2), 2=block2 (P3), 3=block3 (P4), 4=block4 (P5)
    """
    def __init__(self, variant="regnet_y_800mf", stage=0, pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        model_fn = getattr(tv_models, variant, tv_models.regnet_y_800mf)
        try:
            full_model = model_fn(weights=weights)
        except Exception:
            full_model = model_fn(weights=None)
            
        if stage == 0:
            self.layer = full_model.stem
        elif stage == 1:
            self.layer = full_model.trunk_output.block1
        elif stage == 2:
            self.layer = full_model.trunk_output.block2
        elif stage == 3:
            self.layer = full_model.trunk_output.block3
        elif stage == 4:
            self.layer = full_model.trunk_output.block4
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def forward(self, x):
        return self.layer(x)

# Register to ultralytics
setattr(um, 'RegNetBackboneStage', RegNetBackboneStage)
setattr(ut, 'RegNetBackboneStage', RegNetBackboneStage)

# Let's test channel outputs for regnet_y_800mf
# stage 0: 3 -> 32
# stage 1: 32 -> 64
# stage 2: 64 -> 128 (P3)
# stage 3: 128 -> 320 (P4)
# stage 4: 320 -> 768 (P5)

yaml_content = """
# YOLO with RegNetY-800MF Backbone
nc: 5

backbone:
  # [from, repeats, module, args]
  - [-1, 1, RegNetBackboneStage, ["regnet_y_800mf", 0, False]] # 0-P1 (Stem: 32 ch)
  - [-1, 1, RegNetBackboneStage, ["regnet_y_800mf", 1, False]] # 1-P2 (Stage 1: 64 ch)
  - [-1, 1, RegNetBackboneStage, ["regnet_y_800mf", 2, False]] # 2-P3 (Stage 2: 128 ch)
  - [-1, 1, RegNetBackboneStage, ["regnet_y_800mf", 3, False]] # 3-P4 (Stage 3: 320 ch)
  - [-1, 1, RegNetBackboneStage, ["regnet_y_800mf", 4, False]] # 4-P5 (Stage 4: 768 ch)
  - [-1, 1, SPPF, [768, 5]]                                     # 5-P5 SPPF (768 ch)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]                 # 6 (from 5, 768 ch)
  - [[-1, 3], 1, Concat, [1]]                                  # 7 cat P4 (768 + 320 = 1088 ch)
  - [-1, 1, C2f, [320]]                                        # 8

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]                 # 9 (from 8, 320 ch)
  - [[-1, 2], 1, Concat, [1]]                                  # 10 cat P3 (320 + 128 = 448 ch)
  - [-1, 1, C2f, [128]]                                        # 11 (P3/8-small, 128 ch)

  - [-1, 1, Conv, [128, 3, 2]]                                 # 12 (stride 2 -> 128 ch)
  - [[-1, 8], 1, Concat, [1]]                                  # 13 cat head P4 (128 + 320 = 448 ch)
  - [-1, 1, C2f, [320]]                                        # 14 (P4/16-medium, 320 ch)

  - [-1, 1, Conv, [320, 3, 2]]                                 # 15 (stride 2 -> 320 ch)
  - [[-1, 5], 1, Concat, [1]]                                  # 16 cat head P5 (320 + 768 = 1088 ch)
  - [-1, 1, C2f, [768]]                                        # 17 (P5/32-large, 768 ch)

  - [[11, 14, 17], 1, Detect, [nc]]                            # 18 Detect(P3, P4, P5)
"""

with open("scratch/yolo_regnet_800mf.yaml", "w") as f:
    f.write(yaml_content)

print("[*] Instantiating YOLO with RegNetY-800MF...")
model = YOLO("scratch/yolo_regnet_800mf.yaml")
print("[+] YOLO-RegNet model loaded successfully!")

# Forward pass test
dummy = torch.randn(1, 3, 640, 640)
out = model.predict(dummy, verbose=False)
print(f"[+] Prediction successful! Output boxes: {len(out[0].boxes)}")

# ONNX export test
print("[*] Testing ONNX export for YOLO-RegNet...")
onnx_path = model.export(format="onnx", imgsz=640, dynamic=False, opset=12)
print(f"[+] Successfully exported ONNX model to: {onnx_path}")
