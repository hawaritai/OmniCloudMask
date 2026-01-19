from __future__ import annotations
from enum import Enum

class AngleConvention(Enum):
    OPK = "omega_phi_kappa_deg"
    RPY = "roll_pitch_yaw_deg"
    YPR = "yaw_pitch_roll_deg"

class ColorSpace(Enum):
    UNKNOWN = "unknown"
    GRAY = "gray"
    RGB = "rgb"
    RGBA = "rgba"
    MULTISPECTRAL = "multispectral"
    PANCHROMATIC = "panchromatic"
