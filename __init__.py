"""
Custom Deep Learning Backbones & Modules for IBVAP Border Surveillance
"""
from src.models.regnet import (
    SELayer,
    RegNetBlock,
    RegNetStage,
    register_regnet_modules
)

__all__ = [
    "SELayer",
    "RegNetBlock",
    "RegNetStage",
    "register_regnet_modules"
]
