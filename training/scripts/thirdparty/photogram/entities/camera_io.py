# photogram/entities/camera_io.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence
import numpy as np

# ---- Distortion container (kept simple; extend if needed) ----
@dataclass(frozen=True, slots=True)
class RadialTangentialDistortion:
    k1: float = 0.0; k2: float = 0.0; k3: float = 0.0
    p1: float = 0.0; p2: float = 0.0; k4: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraIO:
    # ---- Canonical (fully-resolved) fields ----
    focal_length_mm: float                      # f in millimetres
    focal_length_px: float                      # fx in pixels (primary axis)
    image_size_px: Tuple[int, int]              # (w, h) in pixels
    image_size_mm: Tuple[float, float]          # (w_mm, h_mm)
    pixel_size_um: Tuple[float, float]          # (sx_um, sy_um)
    principal_point_px: Tuple[float, float]     # (cx_px, cy_px)
    principal_point_mm: Tuple[float, float]     # (cx_mm, cy_mm)

    model_name: Optional[str] = None
    serial: Optional[str] = None
    # skew: float = 0.0                           # pixels
    # distortion: RadialTangentialDistortion = field(default_factory=RadialTangentialDistortion)

    # # Optional: store exact K if one was provided (for reproducibility)
    # K_pixels: Optional[
    #     tuple[
    #         tuple[float, float, float],
    #         tuple[float, float, float],
    #         tuple[float, float, float],
    #     ]
    # ] = None

    # ---------------- Derived helpers ----------------
    @property
    def sx_mm(self) -> float:
        """Pixel size along x in mm."""
        return float(self.pixel_size_um[0]) * 1e-3

    @property
    def sy_mm(self) -> float:
        """Pixel size along y in mm."""
        return float(self.pixel_size_um[1]) * 1e-3

    @property
    def fx_px(self) -> float:
        """Focal length along x in pixels (alias)."""
        return float(self.focal_length_px)

    @property
    def fy_px(self) -> float:
        """Focal length along y in pixels, derived from f(mm) & sy."""
        # keep fy consistent even if pixels are slightly anisotropic
        return (self.focal_length_mm / self.sy_mm) if self.sy_mm > 0 else self.focal_length_px

    @property
    def cx_px(self) -> float:
        """Principle point x in pixel."""
        return float(self.principal_point_px[0])

    @property
    def cy_px(self) -> float:
        """Principle point y in pixel."""
        return float(self.principal_point_px[1])

    @property
    def cx_mm(self) -> float:
        """Principle point x in mm."""
        return float(self.principal_point_mm[0])

    @property
    def cy_mm(self) -> float:
        """Principle point y in mm."""
        return float(self.principal_point_mm[1])

    @property
    def image_width_px(self) -> int:
        return int(self.image_size_px[0])

    @property
    def image_height_px(self) -> int:
        return int(self.image_size_px[1])

    # @property
    # def K(self) -> np.ndarray:
    #     """
    #     Intrinsic matrix in pixel units.
    #     Preference: if K_pixels was supplied, return it verbatim; else build from fields.
    #     """
    #     if self.K_pixels is not None:
    #         return np.array(self.K_pixels, dtype=float)
    #     return np.array(
    #         [[self.fx_px, self.skew, self.cx_px],
    #          [0.0,        self.fy_px, self.cy_px],
    #          [0.0,        0.0,        1.0]],
    #         dtype=float
    #     )

    def ray_dir_camera(self, u_px: float, v_px: float) -> np.ndarray:
        """
        Unit ray direction in the CAMERA frame for pixel (u,v) using a pinhole model.
        Conventions: camera x right, y up, z forward; image y grows down → flip sign.
        """
        x_mm = (u_px - self.cx_px) * self.sx_mm
        y_mm = -(v_px - self.cy_px) * self.sy_mm
        d = np.array([x_mm, y_mm, -self.focal_length_mm], dtype=float)
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    # ---------------- Flexible factory ----------------
    @classmethod
    def from_any(
        cls,
        *,
        image_size_px: Tuple[int, int],
        # You can provide either pixel size or sensor/image size in mm:
        pixel_size_um: Tuple[float, float] | None = None,
        image_size_mm: Tuple[float, float] | None = None,
        sensor_size_mm: Tuple[float, float] | None = None,  # alias of image_size_mm if full-frame
        # Focal length options (provide one of the two):
        focal_length_mm: float | None = None,
        focal_length_px: float | None = None,
        fx_pixels: float | None = None,    # accepted alias for focal_length_px
        fy_pixels: float | None = None,    # optional; if given we’ll store fx and compute f_mm from sx
        # Principal point:
        principal_point_px: Tuple[float, float] | None = None,  # same as cx_px
        principal_point_mm: Tuple[float, float] | None = None,  # same as cy_mm
        # Other:
        skew: float | None = None,
        model_name: Optional[str] = None,
        serial: Optional[str] = None,
        distortion: Optional[RadialTangentialDistortion] = None,
        K_pixels: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> "CameraIO":
        """
        Build a fully-resolved CameraIO from a variety of input shapes.

        Supply enough info to connect px↔mm:
          - Provide pixel_size_um, OR image_size_mm, OR sensor_size_mm (equivalent).
          - Provide focal_length_mm OR focal_length_px (or fx_pixels alias).
          - principal_point defaults to image center if not given (in px); mm will be derived.
        """
        w_px, h_px = int(image_size_px[0]), int(image_size_px[1])

        # Resolve pixel size (μm)
        if pixel_size_um is None:
            size_mm = image_size_mm or sensor_size_mm
            if size_mm is not None:
                sx_um = (float(size_mm[0]) * 1000.0) / w_px
                sy_um = (float(size_mm[1]) * 1000.0) / h_px
                pixel_size_um = (sx_um, sy_um)
            else:
                # Without pixel size (or image size mm), we cannot convert px↔mm
                raise ValueError("Provide pixel_size_um or image_size_mm/sensor_size_mm to relate px↔mm.")

        sx_mm = float(pixel_size_um[0]) * 1e-3
        sy_mm = float(pixel_size_um[1]) * 1e-3

        # Resolve focal lengths
        if focal_length_px is None:
            focal_length_px = fx_pixels  # accept alias
        if focal_length_mm is None and focal_length_px is None:
            raise ValueError("Provide either focal_length_mm or focal_length_px (fx_pixels).")

        if focal_length_mm is None:
            # derive mm from px using sx
            focal_length_mm = float(focal_length_px) * sx_mm

        if focal_length_px is None:
            # derive px from mm using sx
            focal_length_px = float(focal_length_mm) / sx_mm

        # Resolve principal point
        if principal_point_px is None and principal_point_mm is None:
            principal_point_px = (w_px / 2.0, h_px / 2.0)

        if principal_point_px is None and principal_point_mm is not None:
            # Convert mm → px
            principal_point_px = (float(principal_point_mm[0]) / sx_mm,
                                  float(principal_point_mm[1]) / sy_mm)

        if principal_point_mm is None and principal_point_px is not None:
            # Convert px → mm
            principal_point_mm = (float(principal_point_px[0]) * sx_mm,
                                  float(principal_point_px[1]) * sy_mm)

        # Image size in mm (if not given)
        if image_size_mm is None:
            image_size_mm = (w_px * sx_mm, h_px * sy_mm)

        # # Skew default
        # if skew is None:
        #     skew = 0.0

        # # Keep verbatim K if provided (validated as 3x3)
        # K_store = None
        # if K_pixels is not None:
        #     K = np.asarray(K_pixels, dtype=float)
        #     if K.shape != (3, 3):
        #         raise ValueError("K_pixels must be a 3x3 matrix.")
        #     K_store = tuple(map(tuple, K))

        #     # If fy is explicitly provided, it can differ from fx due to anisotropy.
        #     # We still store focal_length_px as fx; fy will be derived property-wise.

        return cls(
            focal_length_mm=float(focal_length_mm),
            focal_length_px=float(focal_length_px),
            image_size_px=(w_px, h_px),
            image_size_mm=(float(image_size_mm[0]), float(image_size_mm[1])),
            pixel_size_um=(float(pixel_size_um[0]), float(pixel_size_um[1])),
            principal_point_px=(float(principal_point_px[0]), float(principal_point_px[1])),
            principal_point_mm=(float(principal_point_mm[0]), float(principal_point_mm[1])),
            model_name=model_name,
            serial=serial,
            # skew=float(skew),
            # distortion=distortion or RadialTangentialDistortion(),
            # K_pixels=K_store,
        )
