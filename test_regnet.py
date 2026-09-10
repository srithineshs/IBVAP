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

def register_regnet_modules():
    """Injects RegNet into ultralytics module registry and tasks."""
    setattr(um, 'SELayer', SELayer)
    setattr(um, 'RegNetBlock', RegNetBlock)
    setattr(um, 'RegNetStage', RegNetStage)
    setattr(ut, 'RegNetStage', RegNetStage)
    setattr(ut, 'RegNetBlock', RegNetBlock)
    setattr(ut, 'SELayer', SELayer)
    
    orig_parse_model = ut.parse_model

    def patched_parse_model(d, ch, verbose=True):
        import copy
        # Inject c1 into RegNetStage and RegNetBlock args before parsing if not already present
        # Alternatively, ensure parse_model handles them properly
        d_copy = copy.deepcopy(d)
        return orig_parse_model(d_copy, ch, verbose=verbose)

    # Let's check how parse_model handles base_modules
    # We can patch parse_model directly
    import types
    # Get the code of parse_model and update base_modules dynamically
    tasks_dict = ut.__dict__
    tasks_dict['RegNetStage'] = RegNetStage
    tasks_dict['RegNetBlock'] = RegNetBlock
    tasks_dict['SELayer'] = SELayer

register_regnet_modules()

# Let's test using custom parse_model wrapper
orig_parse_model = ut.parse_model

def custom_parse(d, ch, verbose=True):
    # Hook RegNetStage in parse_model
    from ultralytics.nn.tasks import LOGGER, colorstr, make_divisible
    import ast, contextlib
    
    # We can invoke original parse_model by ensuring RegNetStage is recognized
    # In original parse_model, if m in base_modules: c1, c2 = ch[f], args[0]
    # Otherwise: c2 = ch[f]
    # We can define a wrapper or patch parse_model
    pass

