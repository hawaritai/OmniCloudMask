# photogram/algorithms/overlap.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except Exception as _e:
    Polygon = None  # type: ignore

try:
    from pyproj import Transformer
except Exception:
    Transformer = None  # type: ignore

from ..entities.image_record import ImageRecord


# ---------- Small linear algebra helpers ----------

def _to_np_pts(pts: List[Tuple[float, float]]) -> np.ndarray:
    a = np.asarray(pts, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Points must be Nx2.")
    return a

def _homography_4pt(src4: List[Tuple[float, float]],
                    dst4: List[Tuple[float, float]]) -> np.ndarray:
    """
    DLT homography from 4 non-collinear point correspondences.
    src4: list of 4 (x,y)
    dst4: list of 4 (X,Y)
    Returns 3x3 H such that [X,Y,1]^T ~ H @ [x,y,1]^T
    """
    src = _to_np_pts(src4)
    dst = _to_np_pts(dst4)
    if src.shape[0] != 4 or dst.shape[0] != 4:
        raise ValueError("Need 4 src and 4 dst points.")
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([ x, y, 1, 0, 0, 0, -X*x, -X*y, -X ])
        A.append([ 0, 0, 0, x, y, 1, -Y*x, -Y*y, -Y ])
    A = np.asarray(A, dtype=float)
    # Solve Ah = 0 → SVD smallest singular vector
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H

def _apply_H(H: np.ndarray, pts: List[Tuple[float, float]]) -> np.ndarray:
    P = _to_np_pts(pts)
    ones = np.ones((P.shape[0], 1))
    Ph = np.hstack([P, ones])
    Qh = (H @ Ph.T).T
    Q = Qh[:, :2] / Qh[:, 2:3]
    return Q

def _poly_area(pts: List[Tuple[float, float]]) -> float:
    """Signed area; absolute value is polygon area."""
    x, y = np.asarray(pts).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------- CRS utilities ----------

def _transform_pts(pts: List[Tuple[float, float]], crs_from: str, crs_to: str) -> List[Tuple[float, float]]:
    if crs_from == crs_to:
        return pts
    if Transformer is None:
        raise ImportError("pyproj is required to transform between CRSs.")
    tr = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    xs, ys = zip(*pts)
    X, Y = tr.transform(xs, ys)
    return list(zip(map(float, X), map(float, Y)))


# ---------- Core: overlap from two ImageRecord ----------

def overlap_from_imagerecords(
    A: ImageRecord,
    B: ImageRecord,
    *,
    prefer_crs: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute overlap polygon between two images using their footprints.

    Expects at least:
      - A.geo_footprint_xy and B.geo_footprint_xy as 4-corner lists (UL,UR,LR,LL)
      - A.pix_footprint_uv and B.pix_footprint_uv as 4-corner lists (UL,UR,LR,LL)
      - A.footprint_crs and B.footprint_crs (e.g., "EPSG:28992")

    If footprints are missing, compute them upstream and set these fields
    (e.g., with your photogram.algorithms.footprint.image_footprint_on_plane()).

    Returns:
      {
        "crs": str,                         # CRS used for geo overlap
        "geo_overlap_xy": [(x,y), ...],     # geo intersection polygon (in 'crs')
        "area_m2": float,                   # area of geo overlap (if CRS is projected)
        "percent_A": float,                 # overlap area / A footprint area * 100
        "percent_B": float,                 # overlap area / B footprint area * 100
        "pix_overlap_A_uv": [(u,v), ...],   # overlap polygon mapped into A's pixel space
        "pix_overlap_B_uv": [(u,v), ...],   # overlap polygon mapped into B's pixel space
      }
    """
    # --- Validate inputs
    if A.geo_footprint_xy is None or B.geo_footprint_xy is None:
        raise ValueError("geo_footprint_xy missing. Compute footprints first.")
    if A.pix_footprint_uv is None or B.pix_footprint_uv is None:
        raise ValueError("pix_footprint_uv missing. Provide the 4 image-corner pixels.")
    if A.footprint_crs is None or B.footprint_crs is None:
        # Try using raster CRS if present
        A_crs = (A.footprint_crs or (A.geo.crs if A.geo else None))
        B_crs = (B.footprint_crs or (B.geo.crs if B.geo else None))
        if not A_crs or not B_crs:
            raise ValueError("CRS unknown for one/both footprints.")
        A.footprint_crs = A_crs
        B.footprint_crs = B_crs

    crsA = A.footprint_crs
    crsB = B.footprint_crs
    target_crs = prefer_crs or crsA or crsB
    if not target_crs:
        raise ValueError("Target CRS cannot be determined.")

    # --- Transform footprints into a common CRS
    A_xy = _transform_pts(A.geo_footprint_xy, crsA, target_crs) if crsA else A.geo_footprint_xy
    B_xy = _transform_pts(B.geo_footprint_xy, crsB, target_crs) if crsB else B.geo_footprint_xy

    # --- Build polygons (Shapely strongly recommended)
    if Polygon is None:
        raise ImportError("Shapely is required to compute polygon overlap. Please install shapely.")

    polyA = Polygon(A_xy)
    polyB = Polygon(B_xy)
    if not polyA.is_valid or not polyB.is_valid:
        polyA = polyA.buffer(0)
        polyB = polyB.buffer(0)

    inter = polyA.intersection(polyB)
    if inter.is_empty:
        return {
            "crs": target_crs,
            "geo_overlap_xy": [],
            "area_m2": 0.0,
            "percent_A": 0.0,
            "percent_B": 0.0,
            "pix_overlap_A_uv": [],
            "pix_overlap_B_uv": [],
        }

    # For MultiPolygon, dissolve to single boundary (or pick the largest)
    if inter.geom_type == "MultiPolygon":
        inter = max(inter.geoms, key=lambda g: g.area)

    geo_overlap_coords = list(map(lambda p: (float(p[0]), float(p[1])), inter.exterior.coords[:-1]))
    area_inter = float(inter.area)

    # --- Areas for percentages (in projected CRS units; if degrees, area isn't m²)
    areaA = float(polyA.area) if polyA.area is not None else 0.0
    areaB = float(polyB.area) if polyB.area is not None else 0.0
    percent_A = (area_inter / areaA * 100.0) if areaA > 0 else 0.0
    percent_B = (area_inter / areaB * 100.0) if areaB > 0 else 0.0

    # --- Map geo overlap polygon into each image's pixel space via homography
    # Build geo<->pixel H using the 4 corner correspondences:
    #   (pix UL,UR,LR,LL)  <->  (geo UL,UR,LR,LL)
    # For each image, compute H_pix_from_geo = H_pix_from_geo(geo->pix) and apply.
    H_A_geo2pix = np.linalg.inv(_homography_4pt(A.pix_footprint_uv, A.geo_footprint_xy))
    H_B_geo2pix = np.linalg.inv(_homography_4pt(B.pix_footprint_uv, B.geo_footprint_xy))

    pix_overlap_A = _apply_H(H_A_geo2pix, geo_overlap_coords).tolist()
    pix_overlap_B = _apply_H(H_B_geo2pix, geo_overlap_coords).tolist()

    # Optionally, clip to image bounds if desired (0..w-1, 0..h-1)

    return {
        "crs": target_crs,
        "geo_overlap_xy": geo_overlap_coords,
        "area_m2": area_inter,
        "percent_A": percent_A,
        "percent_B": percent_B,
        "pix_overlap_A_uv": [(float(u), float(v)) for (u, v) in pix_overlap_A],
        "pix_overlap_B_uv": [(float(u), float(v)) for (u, v) in pix_overlap_B],
    }
