"""
RegNet-Y Backbone Architecture and Custom Modules for YOLO Object Detection
Implements RegNet with Squeeze-and-Excitation (SE) Attention & Grouped Convolutions.
Designed for high-throughput edge inference on perimeter and border surveillance cameras.
"""

import sys
import copy
import torch
import torch.nn as nn
from pathlib import Path

# Add project root and src directory to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention Block.
    Recalibrates channel-wise feature responses to boost small/camouflaged target detection.
    """
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, reduced, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class RegNetBlock(nn.Module):
    """
    RegNet Residual Bottleneck Block with Grouped Convolution & Squeeze-and-Excitation.
    Matches the Regular Network Design Space (Meta AI / FAIR).
    """
    def __init__(self, c1: int, c2: int, stride: int = 1, group_width: int = 16, se: bool = True):
        super().__init__()
        # Ensure number of groups divides output channels evenly
        groups = max(1, c2 // max(1, group_width))
        if c2 % groups != 0:
            groups = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.se = SELayer(c2, reduction=4) if se else nn.Identity()
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.act = nn.SiLU(inplace=True)

        if stride != 1 or c1 != c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        return self.act(out + res)


class RegNetStage(nn.Module):
    """
    RegNet Stage consisting of depth D blocks.
    The first block performs spatial downsampling (stride 2), and subsequent blocks maintain resolution (stride 1).
    """
    def __init__(self, c1: int, c2: int, depth: int = 2, group_width: int = 16, se: bool = True):
        super().__init__()
        blocks = []
        # Downsampling block (stride 2)
        blocks.append(RegNetBlock(c1, c2, stride=2, group_width=group_width, se=se))
        # Resolution-preserving blocks (stride 1)
        for _ in range(1, max(1, int(depth))):
            blocks.append(RegNetBlock(c2, c2, stride=1, group_width=group_width, se=se))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


_REGISTRATION_DONE = False

def register_regnet_modules():
    """
    Hooks RegNet modules into Ultralytics model parser and task registries.
    Enables native training, prediction, validation, and ONNX/TensorRT export with YOLO.
    """
    global _REGISTRATION_DONE
    if _REGISTRATION_DONE:
        return

    try:
        import ultralytics.nn.modules as um
        import ultralytics.nn.tasks as ut
        import ast
        import contextlib

        # Register classes in module dictionaries
        for mod in (um, ut):
            setattr(mod, 'SELayer', SELayer)
            setattr(mod, 'RegNetBlock', RegNetBlock)
            setattr(mod, 'RegNetStage', RegNetStage)

        # Patch parse_model so RegNetStage is recognized as a base block computing output channels
        orig_parse_model = ut.parse_model

        def patched_parse_model(d, ch, verbose=True):
            from ultralytics.nn.tasks import (
                LOGGER, colorstr, make_divisible, Conv, ConvTranspose, GhostConv, Bottleneck,
                GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
                C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn,
                C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
                Concat, Detect, WorldDetect, YOLOEDetect, Segment, Segment26, YOLOESegment,
                YOLOESegment26, Pose, Pose26, OBB, OBB26
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

        ut.parse_model = patched_parse_model
        _REGISTRATION_DONE = True

    except Exception as e:
        print(f"[!] Warning: Could not register RegNet modules to Ultralytics: {e}")

