from __future__ import annotations
from ..interfaces import ImageReader
from ..registry import register_reader
import numpy as np
from typing import Dict, Optional, Tuple, Any, Type, Callable


# Optional deps
try:
    import rasterio
    from rasterio.windows import Window
except Exception:
    rasterio = None
    Window = None  # type: ignore


@register_reader
class RasterioReader(ImageReader):
    """Reader using rasterio (GDAL-backed). Best for GeoTIFF and many raster formats."""
    geo_exts = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".pix"}  # extend as needed

    @classmethod
    def can_handle(cls, path: str, ext: str) -> bool:
        return rasterio is not None and (ext in cls.geo_exts or cls._rasterio_can_open(path))

    @staticmethod
    def _rasterio_can_open(path: str) -> bool:
        if rasterio is None:
            return False
        try:
            with rasterio.open(path) as _:
                return True
        except Exception:
            return False

    def read(self, path: str) -> np.ndarray:
        if rasterio is None:
            raise RuntimeError("rasterio is not installed.")
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            arr = np.moveaxis(arr, 0, -1) if arr.shape[0] > 1 else arr.squeeze(0)
            return arr

    def read_region(self, path: str, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        if rasterio is None or Window is None:
            return super().read_region(path, xywh)
        x, y, w, h = xywh
        with rasterio.open(path) as src:
            window = Window(col_off=x, row_off=y, width=w, height=h)
            arr = src.read(window=window)  # (bands, h, w)
            arr = np.moveaxis(arr, 0, -1) if arr.shape[0] > 1 else arr.squeeze(0)
            return arr

    def metadata(self, path: str) -> Dict[str, Any]:
        if rasterio is None:
            return {}
        with rasterio.open(path) as src:
            md = {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]) if src.count else None,
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform),
                "bounds": tuple(src.bounds),
                "nodata": src.nodata,
                "resolution": src.res,
            }
        return md
