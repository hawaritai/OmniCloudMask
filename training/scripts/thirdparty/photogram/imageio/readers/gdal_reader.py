from __future__ import annotations
from ..interfaces import ImageReader
from ..registry import register_reader
import numpy as np
from typing import Dict, Optional, Tuple, Any, Type, Callable



try:
    from osgeo import gdal, osr
except Exception:
    gdal = None
    osr = None  # type: ignore


@register_reader
class GDALReader(ImageReader):
    """Reader using GDAL Python bindings directly. Good fallback for exotic formats (e.g., .pix)."""
    geo_exts = {".tif", ".tiff", ".vrt", ".img", ".pix", ".ntf", ".hdf", ".grd"}

    @classmethod
    def can_handle(cls, path: str, ext: str) -> bool:
        if gdal is None:
            return False
        if ext in cls.geo_exts:
            return True
        # Probe
        try:
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds:
                ds = None
                return True
        except Exception:
            return False
        return False

    def read(self, path: str) -> np.ndarray:
        if gdal is None:
            raise RuntimeError("GDAL is not installed.")
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL could not open {path}")
        try:
            bands = ds.RasterCount
            arrs = []
            for b in range(1, bands + 1):
                rb = ds.GetRasterBand(b)
                arrs.append(rb.ReadAsArray())
            arr = np.stack(arrs, axis=-1) if bands > 1 else arrs[0]
            return arr
        finally:
            ds = None

    def read_region(self, path: str, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        if gdal is None:
            return super().read_region(path, xywh)
        x, y, w, h = xywh
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL could not open {path}")
        try:
            bands = ds.RasterCount
            arrs = []
            for b in range(1, bands + 1):
                rb = ds.GetRasterBand(b)
                arrs.append(rb.ReadAsArray(x, y, w, h))
            arr = np.stack(arrs, axis=-1) if bands > 1 else arrs[0]
            return arr
        finally:
            ds = None

    def metadata(self, path: str) -> Dict[str, Any]:
        if gdal is None:
            return {}
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            return {}
        try:
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            crs_wkt = proj if proj else None
            md = {
                "driver": ds.GetDriver().ShortName if ds.GetDriver() else None,
                "width": ds.RasterXSize,
                "height": ds.RasterYSize,
                "count": ds.RasterCount,
                "dtype": gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) if ds.RasterCount else None,
                "geotransform": tuple(gt) if gt else None,
                "projection_wkt": crs_wkt,
                "bounds": _bounds_from_gt(gt, ds.RasterXSize, ds.RasterYSize) if gt else None,
                "nodata": ds.GetRasterBand(1).GetNoDataValue() if ds.RasterCount else None,
            }
            return md
        finally:
            ds = None


def _bounds_from_gt(gt, width, height):
    """Compute bounds (minx, miny, maxx, maxy) from GDAL geotransform."""
    # gt: (x_min, x_res, x_rot, y_max, y_rot, -y_res) for north-up images
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + width * gt[1]
    miny = maxy + height * gt[5]
    return (minx, miny, maxx, maxy)
