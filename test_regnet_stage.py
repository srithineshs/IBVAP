import torch
import torch.nn as nn
import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut
from ultralytics import YOLO

class SELayer(nn.Module):
    """Squeeze-and-Excitation attention layer."""
    def __init__(self, channel, reduction=4):
        super().__init__()
        reduced = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, reduced, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class RegNetBlock(nn.Module):
    """RegNet Y-Block with Grouped Conv & Squeeze-and-Excitation."""
    def __init__(self, c1, c2, stride=1, group_width=16, se=True):
        super().__init__()
        groups = max(1, c2 // max(1, group_width))
        if c2 % groups != 0:
            groups = 1
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.se = SELayer(c2, reduction=4) if se else nn.Identity()
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.act = nn.SiLU(inplace=True)
        
        if stride != 1 or c1 != c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        return self.act(out + res)

class RegNetStage(nn.Module):
    """RegNet Stage comprising multiple RegNet blocks with stride downsampling."""
    def __init__(self, c1, c2, depth=2, group_width=16, se=True):
        super().__init__()
        blocks = []
        # First block has stride 2 (downsamples spatial dimension)
        blocks.append(RegNetBlock(c1, c2, stride=2, group_width=group_width, se=se))
        # Subsequent blocks have stride 1
        for _ in range(1, max(1, depth)):
            blocks.append(RegNetBlock(c2, c2, stride=1, group_width=group_width, se=se))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

# Patch ultralytics parse_model to recognize RegNetStage as a base_module
orig_parse_model = ut.parse_model

def patched_parse_model(d, ch, verbose=True):
    # Register RegNetStage and RegNetBlock in globals and tasks
    tasks_mod = ut
    tasks_mod.RegNetStage = RegNetStage
    tasks_mod.RegNetBlock = RegNetBlock
    tasks_mod.SELayer = SELayer
    
    # We call orig_parse_model, but before that, let's ensure base_modules contains RegNetStage
    return custom_parse(d, ch, verbose)

def custom_parse(d, ch, verbose=True):
    # Intercept parse_model by updating base_modules dynamically
    import ast, contextlib
    from ultralytics.nn.tasks import (
        LOGGER, colorstr, make_divisible, Conv, ConvTranspose, GhostConv, Bottleneck,
        GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
        C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn,
        C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
        AIFI, HGStem, HGBlock, ResNetLayer, Concat, Detect, WorldDetect, YOLOEDetect,
        Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26,
        v10Detect, ImagePoolingAttn, RTDETRDecoder, CBLinear, CBFuse, TorchVision, Index
    )
    
    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    
    base_modules = frozenset({
        Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
        C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4,
        ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, torch.nn.ConvTranspose2d,
        DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
        RegNetStage, RegNetBlock
    })

    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3,
        C2fPSA, C2fCIB, C2PSA, A2C2f
    })

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        if isinstance(m, str):
            if "nn." in m:
                m = getattr(torch.nn, m[3:])
            elif m == "RegNetStage":
                m = RegNetStage
            elif m == "RegNetBlock":
                m = RegNetBlock
            elif m == "SELayer":
                m = SELayer
            else:
                m = getattr(ut, m, getattr(um, m, globals().get(m)))

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect, WorldDetect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            m.legacy = legacy
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)

ut.parse_model = custom_parse

yaml_content = """
# YOLO with RegNet-Y Backbone Architecture
nc: 5

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [32, 3, 2]]                  # 0-P1 (Stem, stride 2, 32 ch)
  - [-1, 1, RegNetStage, [64, 2, 16, True]]    # 1-P2 (Stage 1, stride 4, 64 ch, depth 2)
  - [-1, 1, RegNetStage, [128, 4, 16, True]]   # 2-P3 (Stage 2, stride 8, 128 ch, depth 4) -> to neck
  - [-1, 1, RegNetStage, [256, 6, 32, True]]   # 3-P4 (Stage 3, stride 16, 256 ch, depth 6) -> to neck
  - [-1, 1, RegNetStage, [512, 2, 64, True]]   # 4-P5 (Stage 4, stride 32, 512 ch, depth 2) -> to neck
  - [-1, 1, SPPF, [512, 5]]                    # 5-P5 SPPF (512 ch)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 6
  - [[-1, 3], 1, Concat, [1]]                  # 7 cat P4 (512 + 256 = 768)
  - [-1, 1, C2f, [256]]                        # 8

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 9
  - [[-1, 2], 1, Concat, [1]]                  # 10 cat P3 (256 + 128 = 384)
  - [-1, 1, C2f, [128]]                        # 11 (P3/8-small, 128 ch)

  - [-1, 1, Conv, [128, 3, 2]]                 # 12 (stride 2 -> 128 ch)
  - [[-1, 8], 1, Concat, [1]]                  # 13 cat head P4 (128 + 256 = 384)
  - [-1, 1, C2f, [256]]                        # 14 (P4/16-medium, 256 ch)

  - [-1, 1, Conv, [256, 3, 2]]                 # 15 (stride 2 -> 256 ch)
  - [[-1, 5], 1, Concat, [1]]                  # 16 cat head P5 (256 + 512 = 768)
  - [-1, 1, C2f, [512]]                        # 17 (P5/32-large, 512 ch)

  - [[11, 14, 17], 1, Detect, [nc]]            # 18 Detect(P3, P4, P5)
"""

with open("scratch/yolo_regnet_working.yaml", "w") as f:
    f.write(yaml_content)

print("[*] Instantiating YOLO with RegNet-Y Backbone...")
model = YOLO("scratch/yolo_regnet_working.yaml")
print("[+] YOLO-RegNet model loaded successfully!")

# Test Prediction
dummy = torch.randn(1, 3, 640, 640)
out = model.predict(dummy, verbose=False)
print(f"[+] Prediction successful! Output boxes count: {len(out[0].boxes)}")

# Test ONNX Export
print("[*] Exporting to ONNX...")
onnx_path = model.export(format="onnx", imgsz=640, dynamic=False, opset=12)
print(f"[+] ONNX export successful: {onnx_path}")
