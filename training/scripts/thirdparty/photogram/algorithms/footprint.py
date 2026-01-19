# photogram/algorithms/footprint.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
# from photogram.entities.camera_eo import CameraEO
# from photogram.entities.camera_io import CameraIO
from ..entities.camera_eo import CameraEO
from ..entities.camera_io import CameraIO

def _corner_pixels(w: int, h: int) -> list[tuple[float, float]]:
    """
    UL, UR, LR, LL in image space (pixel centers). Uses [0..w-1], [0..h-1].
    """
    return [
        (0.0,      0.0),        # UL
        (w - 1.0,  0.0),        # UR
        (w - 1.0,  h - 1.0),    # LR
        (0.0,      h - 1.0),    # LL
    ]

def _intersect_Z_plane(C_w: np.ndarray, v_w: np.ndarray, Z: float) -> np.ndarray | None:
    """
    Ray P(t) = C + t*v with plane Z = constant.
    Returns intersection point or None if parallel or behind the camera.
    """
    vz = v_w[2]
    if abs(vz) < 1e-12:
        return None
    t = (Z - C_w[2]) / vz
    if t <= 0:
        return None
    return C_w + t * v_w

def image_footprint_on_plane(io: CameraIO, eo: CameraEO, ground_z: float = 0.0) -> List[Tuple[float, float]]:
    """
    Intersect the 4 corner rays with a horizontal plane Z=ground_z.
    Returns list of (X,Y) in world CRS, in UL→UR→LR→LL order.
    """
    R_c2w = eo.rotation_c2w()
    C_w = eo.C

    w, h = io.image_width_px, io.image_height_px
    pts: list[tuple[float, float]] = []
    for (u, v) in _corner_pixels(w, h):
        d_c = io.ray_dir_camera(u, v)    # unit vector (camera frame)
        d_w = R_c2w @ d_c                # unit vector (world frame)
        P = _intersect_Z_plane(C_w, d_w, ground_z)
        if P is None:
            raise ValueError("Ray does not intersect Z=ground_z in front of the camera.")
        pts.append((float(P[0]), float(P[1])))
    return pts
