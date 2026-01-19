from __future__ import annotations
from ..interfaces import ImageReader
from ..registry import register_reader
import numpy as np
from typing import Dict, Optional, Tuple, Any, Type, Callable


# Optional deps
try:
    import rawpy
except Exception:
    rawpy = None


@register_reader
class RawpyReader(ImageReader):
    """Reader using rawpy for RAW image formats. Best for IIQ and other camera RAW files."""
    raw_exts = {".iiq", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".dng"}  # extend as needed

    @classmethod
    def can_handle(cls, path: str, ext: str) -> bool:
        return rawpy is not None and (ext.lower() in cls.raw_exts or cls._rawpy_can_open(path))

    @staticmethod
    def _rawpy_can_open(path: str) -> bool:
        if rawpy is None:
            return False
        try:
            with rawpy.imread(path) as _:
                return True
        except Exception:
            return False

    def read(self, path: str) -> np.ndarray:
        if rawpy is None:
            raise RuntimeError("rawpy is not installed.")
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                output_bps=8  # Change to 16 based on 16-bit output
            )
            return rgb

    def read_region(self, path: str, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        # rawpy doesn't support windowed reading, so we read full image and crop
        # This could be memory-intensive for very large RAW files
        if rawpy is None:
            return super().read_region(path, xywh)
        
        full_img = self.read(path)
        x, y, w, h = xywh
        
        # Ensure we don't go out of bounds
        img_h, img_w = full_img.shape[:2]
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        return full_img[y:y+h, x:x+w]

    def metadata(self, path: str) -> Dict[str, Any]:
        if rawpy is None:
            return {}

        try:
            with rawpy.imread(path) as raw:
                md = {
                    "raw_type": getattr(raw, "raw_type", None),
                    "color_desc": getattr(raw, "color_desc", None),
                    "num_colors": getattr(raw, "num_colors", None),
                    "raw_pattern": getattr(raw, "raw_pattern", None),
                    "sizes": {
                        "raw_width": getattr(raw.sizes, "raw_width", None),
                        "raw_height": getattr(raw.sizes, "raw_height", None),
                        "width": getattr(raw.sizes, "width", None),
                        "height": getattr(raw.sizes, "height", None),
                        "top_margin": getattr(raw.sizes, "top_margin", None),
                        "left_margin": getattr(raw.sizes, "left_margin", None),
                        "iwidth": getattr(raw.sizes, "iwidth", None),
                        "iheight": getattr(raw.sizes, "iheight", None),
                    } if hasattr(raw, "sizes") else None,
                    "color_matrix": getattr(raw, "color_matrix", None),
                    "rgb_xyz_matrix": getattr(raw, "rgb_xyz_matrix", None),
                    "camera_whitebalance": getattr(raw, "camera_whitebalance", None),
                    "daylight_whitebalance": getattr(raw, "daylight_whitebalance", None),
                    "black_level_per_channel": getattr(raw, "black_level_per_channel", None),
                    "white_level": getattr(raw, "white_level", None),
                }

                # Handle decoding/array conversion gracefully
                if isinstance(md["color_desc"], (bytes, bytearray)):
                    md["color_desc"] = md["color_desc"].decode("utf-8", errors="ignore")
                if hasattr(md["raw_pattern"], "tolist"):
                    md["raw_pattern"] = md["raw_pattern"].tolist()
                if hasattr(md["color_matrix"], "tolist"):
                    md["color_matrix"] = md["color_matrix"].tolist()
                if hasattr(md["rgb_xyz_matrix"], "tolist"):
                    md["rgb_xyz_matrix"] = md["rgb_xyz_matrix"].tolist()

        except Exception as e:
            md = {"error": f"Could not extract metadata: {str(e)}"}

        return md
