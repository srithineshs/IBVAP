import torch
import torchvision
from ultralytics import YOLO

# Let's inspect torchvision models for regnet
models = [m for m in dir(torchvision.models) if 'regnet' in m]
print("Available RegNet models in torchvision:", models)

# Let's inspect children of regnet_y_400mf
regnet = torchvision.models.regnet_y_400mf(weights=None)
print("RegNet structure:")
for idx, (name, child) in enumerate(regnet.named_children()):
    print(f"Child {idx} ({name}):", type(child))

# Test with a dummy input
x = torch.randn(1, 3, 480, 480)
# stem
stem_out = regnet.stem(x)
print("Stem out:", stem_out.shape) # [1, 32, 240, 240] (stride 2)
# trunk_output
s1 = regnet.trunk_output.block1(stem_out)
print("Block1 out (P2):", s1.shape) # [1, 48, 120, 120] (stride 4)
s2 = regnet.trunk_output.block2(s1)
print("Block2 out (P3):", s2.shape) # [1, 104, 60, 60] (stride 8)
s3 = regnet.trunk_output.block3(s2)
print("Block3 out (P4):", s3.shape) # [1, 208, 30, 30] (stride 16)
s4 = regnet.trunk_output.block4(s3)
print("Block4 out (P5):", s4.shape) # [1, 440, 15, 15] (stride 32)
