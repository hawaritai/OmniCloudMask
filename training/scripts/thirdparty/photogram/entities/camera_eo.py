# photogram/entities/camera_eo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
from .enums import AngleConvention  # expects AngleConvention.OPK etc.

@dataclass(frozen=True)
class OrientationAngles:
    a1: float; a2: float; a3: float
    convention: AngleConvention = AngleConvention.OPK

def _rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)

def _rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def _rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=float)

def _quat_xyzw_to_R(q: Tuple[float, float, float, float]) -> np.ndarray:
    # q = (x, y, z, w), normalized
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)]
    ], dtype=float)

def _angles_to_R_c2w(angles: OrientationAngles) -> np.ndarray:
    """
    Photogrammetric OPK (Ω, Φ, Κ) assumed when convention == OPK:
      R_c2w = Rz(κ) · Ry(φ) · Rx(ω)
    Angles are in degrees in the dataclass; convert to radians here.
    """
    a1, a2, a3 = map(math.radians, (angles.a1, angles.a2, angles.a3))
    if angles.convention == AngleConvention.OPK:
        ω, φ, κ = a1, a2, a3
        return _rot_z(κ) @ _rot_y(φ) @ _rot_x(ω)
    # You can add other conventions here if needed:
    # elif angles.convention == AngleConvention.KOP: ...
    # For now default to OPK behavior:
    ω, φ, κ = a1, a2, a3
    return _rot_z(κ) @ _rot_y(φ) @ _rot_x(ω)

@dataclass(frozen=True)
class CameraEO:
    X: float; Y: float; Z: float
    angles_deg: OrientationAngles
    crs: Optional[str] = None
    timestamp: Optional[float] = None
    R_c2w: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = None
    q_c2w_xyzw: Optional[tuple[float, float, float, float]] = None

    @property
    def C(self) -> np.ndarray:
        """Camera center in world coordinates."""
        return np.array([self.X, self.Y, self.Z], dtype=float)

    def rotation_c2w(self) -> np.ndarray:
        """
        Camera-to-world rotation matrix.
        Priority:
          1) R_c2w if provided
          2) q_c2w_xyzw if provided
          3) angles_deg (per convention)
        """
        if self.R_c2w is not None:
            return np.array(self.R_c2w, dtype=float)
        if self.q_c2w_xyzw is not None:
            q = self.q_c2w_xyzw
            # normalize quaternion in case it's not unit length
            norm = math.sqrt(sum(v*v for v in q))
            qn = tuple(v / norm for v in q) if norm > 0 else q
            return _quat_xyzw_to_R(qn)
        return _angles_to_R_c2w(self.angles_deg)
