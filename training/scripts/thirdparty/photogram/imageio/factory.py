from __future__ import annotations
# # C:\projects\cameraPoseEstimation\generate_aligned_patches\imageio\factory.py
# from __future__ import annotations
# from .interfaces import ImageReader
# # if the factory needs the concrete readers (for registration), import them with package dots:
# from .readers import pillow_reader, rasterio_reader, gdal_reader  # noqa: F401
# from .registry import get_readers
# from typing import Dict, Optional, Tuple, Any, Type, Callable
from dataclasses import dataclass




import os
from typing import Dict, Any, Tuple
import numpy as np
from .interfaces import ImageReader
from .registry import get_readers



# _READERS: list[Type[ImageReader]] = []

# def register_reader(cls: Type[ImageReader]) -> Type[ImageReader]:
#     _READERS.append(cls)
#     return cls

@dataclass(frozen=True)
class ImageSource:
    path: str

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1].lower()

class ImageIOFactory:
    """Factory that selects an appropriate reader based on extension and library availability."""
    # def __init__(self, readers: Optional[list[Type[ImageReader]]] = None):
    #     self.readers = readers or list(_READERS)
    def __init__(self, readers=None):
        self.readers = readers or get_readers()

    def get_reader(self, path: str, preferred_reader: str | None = None) -> ImageReader:
        """
        Returns a reader instance for the given path.

        Args:
            path: Path to the image.
            preferred_reader: Optional name of the reader class to force selection.
                              Example: 'rawpyreader', 'rasterioreader'.
        """
        src = ImageSource(path)

        # List of candidate readers that can handle this file
        candidates = []
        for reader_cls in self.readers:
            try:
                if reader_cls.can_handle(src.path, src.ext):
                    candidates.append(reader_cls)
            except Exception:
                continue  # Skip readers with import/runtime issues

        if not candidates:
            raise RuntimeError(
                f"No suitable reader found for '{path}' (ext={src.ext}). "
                "Install optional deps (rasterio/GDAL/Pillow) or register a custom reader."
            )

        # If a preferred reader is specified, try to use it
        if preferred_reader:
            for reader_cls in candidates:
                if reader_cls.__name__.lower().startswith(preferred_reader.lower()):
                    return reader_cls()
            raise RuntimeError(
                f"Preferred reader '{preferred_reader}' cannot open '{path}'. "
                "Available readers: " + ", ".join(r.__name__ for r in candidates)
            )

        # Default: return the first candidate
        return candidates[0]()

    # Convenience helpers
    def read(self, path: str, preferred_reader: str | None = None) -> np.ndarray:
        return self.get_reader(path, preferred_reader).read(path)

    def metadata(self, path: str, preferred_reader: str | None = None) -> Dict[str, Any]:
        return self.get_reader(path, preferred_reader).metadata(path)

    def read_region(
        self, path: str, xywh: Tuple[int, int, int, int], preferred_reader: str | None = None
    ) -> np.ndarray:
        return self.get_reader(path, preferred_reader).read_region(path, xywh)
