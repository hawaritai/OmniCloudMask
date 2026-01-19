# photogram/entities/image_record.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pathlib as Path
from .enums import ColorSpace
from ..imageio.factory import ImageIOFactory, ImageSource

@dataclass(frozen=True)
class RasterGeoMeta:
    """Holds geospatial metadata for an image."""
    crs: Optional[str] = None  # e.g., "EPSG:28992"
    transform_affine6: Optional[Tuple[float, float, float, float, float, float]] = None  # Affine transform
    bounds_xyxy: Optional[Tuple[float, float, float, float]] = None  # Bounding box (x_min, y_min, x_max, y_max)
    nodata: Optional[float] = None  # No-data value
    gsd_xy: Optional[Tuple[float, float]] = None  # Ground sample distance (x, y)
    footprint_xy: Optional[List[Tuple[float, float]]] = None  # 4-corner footprint in world CRS (UL, UR, LR, LL)
    footprint_crs: Optional[str] = None  # CRS for footprint_xy (usually same as crs)

@dataclass
class ImageRecord:
    """Encapsulates image data and metadata, with lazy loading support."""
    image_id: str
    path: str
    width: int
    height: int
    bands: int
    dtype: str
    color_space: ColorSpace = ColorSpace.UNKNOWN
    geo: Optional[RasterGeoMeta] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    pix_footprint_uv: Optional[List[Tuple[float, float]]] = None  # Pixel coords (u,v) for 4 corners (UL, UR, LR, LL)
    _data: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # Cached rasterio image data
    _raw_data: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # Cached rawpy image data
    _metadata: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)  # Cached metadata from default reader
    _raw_metadata: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)  # Cached metadata from rawpy
    _io_factory: Optional[ImageIOFactory] = field(default=None, init=False, repr=False)  # Image reader factory

    def __post_init__(self) -> None:
        """Initialize the IO factory for reading image data and metadata."""
        self._io_factory = ImageIOFactory()

    @property
    def is_loaded(self) -> bool:
        """Return True if the image data has been successfully loaded."""
        return bool(self.path and (self._data is not None or self._raw_data is not None))


    def _load_data_and_metadata(self, load_data: bool = True) -> None:
        """Internal loader for data and metadata, handles .iiq vs others."""
        src = ImageSource(self.path)

        if src.ext.lower() == ".iiq":
            # Rawpy side
            self._raw_metadata = self._io_factory.metadata(
                self.path, preferred_reader="rawpyreader"
            )
            if load_data:
                self._raw_data = self._io_factory.read(
                    self.path, preferred_reader="rawpyreader"
                )

            # Rasterio side (processed)
            self._metadata = self._io_factory.metadata(
                self.path, preferred_reader="rasterioreader"
            )
            if load_data:
                self._data = self._io_factory.read(
                    self.path, preferred_reader="rasterioreader"
                )
        else:
            # Default (usually rasterio)
            self._metadata = self._io_factory.metadata(self.path)
            if load_data:
                self._data = self._io_factory.read(self.path)

        # Choose metadata source
        metadata_to_use = self._metadata or self._raw_metadata
        if metadata_to_use:
            self._update_attributes_from_metadata(metadata_to_use)


    def _update_attributes_from_metadata(self, metadata: dict) -> None:
        """Sync core attributes from metadata."""
        object.__setattr__(self, "width", metadata.get("width", self.width))
        object.__setattr__(self, "height", metadata.get("height", self.height))
        object.__setattr__(self, "bands", metadata.get("count", self.bands))
        object.__setattr__(self, "dtype", metadata.get("dtype", self.dtype))

        geo_data = {
            "crs": metadata.get("crs"),
            "transform_affine6": metadata.get("transform"),
            "bounds_xyxy": metadata.get("bounds"),
            "nodata": metadata.get("nodata"),
            "gsd_xy": metadata.get("resolution"),
            "footprint_crs": metadata.get("crs"),
        }
        self.geo = RasterGeoMeta(**{k: v for k, v in geo_data.items() if v is not None})


    @property
    def data(self) -> Optional[np.ndarray]:
        """Lazily load and cache image data (ensures metadata too)."""
        if self._data is None and self.path:
            try:
                self._load_data_and_metadata(load_data=True)

                if self._data is not None:
                    # Set shape-based attributes
                    object.__setattr__(self, "height", self._data.shape[0])
                    object.__setattr__(self, "width", self._data.shape[1])
                    object.__setattr__(self, "bands", self._data.shape[2] if self._data.ndim == 3 else 1)
                    object.__setattr__(self, "dtype", str(self._data.dtype))

                    # Infer color_space
                    inferred_color_space = (
                        ColorSpace.RGB if self.bands == 3 else
                        ColorSpace.GRAY if self.bands == 1 else
                        ColorSpace.UNKNOWN
                    )
                    object.__setattr__(self, "color_space", inferred_color_space)

            except Exception as e:
                raise RuntimeError(f"Failed to load image data from {self.path}: {e}")
        return self._data


    @property
    def raw_data(self) -> Optional[np.ndarray]:
        """Access raw data for .iiq files."""
        if self._raw_data is None and self.path and ImageSource(self.path).ext.lower() == ".iiq":
            try:
                self._load_data_and_metadata(load_data=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load raw image data from {self.path}: {e}")
        return self._raw_data


    def refresh_metadata(self) -> None:
        """Reload metadata without re-reading image pixels."""
        if not self.path:
            return
        try:
            self._load_data_and_metadata(load_data=False)
        except Exception as e:
            raise RuntimeError(f"Failed to refresh metadata for {self.path}: {e}")

    
    def read_region(self, xywh: Tuple[int, int, int, int]) -> np.ndarray:
        """Read a specific region of the image.

        Args:
            xywh: Tuple of (x, y, width, height) defining the region.

        Returns:
            numpy.ndarray: Image data for the specified region.
        """
        if not self.path:
            raise ValueError("Image path not specified.")
        return self._io_factory.read_region(self.path, xywh)


    def set_data(self, data: np.ndarray, is_raw: bool = False) -> None:
        """Manually set image data and update dimensions and metadata.

        Args:
            data: numpy.ndarray containing image data.
            is_raw: If True, set raw_data and raw_metadata (for .iiq files); otherwise, set data and metadata.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy.ndarray.")
        try:
            if is_raw:
                object.__setattr__(self, "_raw_data", data)
                # Attempt to fetch raw metadata if path is available
                if self.path:
                    object.__setattr__(self, "_raw_metadata", self._io_factory.metadata(self.path, preferred_reader='rawpyreader'))
            else:
                object.__setattr__(self, "_data", data)
                if self.path:
                    object.__setattr__(self, "_metadata", self._io_factory.metadata(self.path))
            
            # Update metadata from data or reader metadata
            metadata_to_use = self._metadata if not is_raw else self._raw_metadata
            object.__setattr__(self, "height", data.shape[0])
            object.__setattr__(self, "width", data.shape[1])
            object.__setattr__(self, "bands", data.shape[2] if data.ndim == 3 else 1)
            object.__setattr__(self, "dtype", str(data.dtype))
            if metadata_to_use:
                object.__setattr__(self, "height", metadata_to_use.get("height", self.height))
                object.__setattr__(self, "width", metadata_to_use.get("width", self.width))
                object.__setattr__(self, "bands", metadata_to_use.get("count", self.bands))
                object.__setattr__(self, "dtype", metadata_to_use.get("dtype", self.dtype))
            inferred_color_space = ColorSpace.RGB if self.bands == 3 else \
                                  ColorSpace.GRAY if self.bands == 1 else ColorSpace.UNKNOWN
            object.__setattr__(self, "color_space", inferred_color_space)
        except Exception as e:
            raise RuntimeError(f"Failed to set data or metadata for {self.path}: {e}")
        
    def release(self, clear_extra: bool = False) -> None:
        """Free heavy image buffers and metadata."""
        object.__setattr__(self, "_data", None)
        object.__setattr__(self, "_raw_data", None)
        object.__setattr__(self, "_metadata", None)
        object.__setattr__(self, "_raw_metadata", None)
        object.__setattr__(self, "_io_factory", None)  # drop reference to factory
        if clear_extra:
            self.extra.clear()  # optionally clear detector results

