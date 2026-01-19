from __future__ import annotations
from ..interfaces import ImageReader
from ..registry import register_reader
import numpy as np
from typing import Dict, Optional, Tuple, Any, Type, Callable


try:
    from PIL import Image, TiffImagePlugin
except Exception:
    Image = None  # type: ignore
    TiffImagePlugin = None  # type: ignore


@register_reader
class PillowReader(ImageReader):
    """Generic non-geo reader (PNG/JPG/TIFF/etc.) via Pillow."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

    @classmethod
    def can_handle(cls, path: str, ext: str) -> bool:
        if Image is None:
            return False
        # Prefer Pillow for web/bitmapy formats; allow TIFF if rasterio/GDAL absent
        if ext in cls.exts:
            if ext in {".tif", ".tiff"} and (rasterio is not None or gdal is not None):
                # Let geo-capable readers handle GeoTIFF first
                return False
            return True
        # Probe with Pillow as a fallback
        try:
            with Image.open(path) as _:
                return True
        except Exception:
            return False

    def read(self, path: str) -> np.ndarray:
        if Image is None:
            raise RuntimeError("Pillow (PIL) is not installed.")
        with Image.open(path) as im:
            im = im.convert("RGBA") if im.mode in ("P", "LA") else im.convert("RGB") if im.mode != "RGB" and im.mode != "L" else im
            arr = np.array(im)
            return arr

    def metadata(self, path: str) -> Dict[str, Any]:
        if Image is None:
            return {}
        with Image.open(path) as im:
            md = {
                "format": im.format,
                "mode": im.mode,
                "width": im.width,
                "height": im.height,
                "info": dict(im.info) if im.info else {},
            }
            # Basic EXIF if present
            try:
                exif = im.getexif()
                if exif:
                    md["exif"] = {TiffImagePlugin.TAGS.get(k, k): v for k, v in exif.items()} if TiffImagePlugin else dict(exif)
            except Exception:
                pass
            return md

