
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np



class ImageReader(ABC):
    """Strategy interface for reading images with a common API."""

    @classmethod
    @abstractmethod
    def can_handle(cls, path: str, ext: str) -> bool:
        """Return True if this reader is appropriate for path/extension."""
        ...

    @abstractmethod
    def read(self, path: str) -> np.ndarray:
        """Read the full image as a numpy array in (H, W, C) or (H, W) for single-band."""
        ...

    def read_region(self, path: str, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        """Optional: read a region (x, y, w, h). Default: fallback to full read + slice."""
        x, y, w, h = xywh
        arr = self.read(path)
        # If channels present, preserve them
        if arr.ndim == 3:
            return arr[y:y+h, x:x+w, :]
        return arr[y:y+h, x:x+w]

    @abstractmethod
    def metadata(self, path: str) -> Dict[str, Any]:
        """Return metadata dict (size, dtype, mode, georef if available, etc.)."""
        ...
