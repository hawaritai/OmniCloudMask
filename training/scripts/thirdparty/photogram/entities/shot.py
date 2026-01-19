from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .camera_io import CameraIO
from .camera_eo import CameraEO

@dataclass(frozen=True)
class Shot:
    image_id: str
    camera_id: str
    io: CameraIO
    eo: CameraEO
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
