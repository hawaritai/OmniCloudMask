# photogram/services/dataset_builder.py
from __future__ import annotations
import os, glob
from typing import Iterable, Optional, Union, Dict
from ..entities import PhotoDataset, ImageRecord, RasterGeoMeta, ColorSpace
from ..imageio.factory import ImageIOFactory
from ..imageio.parsers.camera_io_xml import parse_camera_io_xml
from ..imageio.parsers.camera_eo_csv import parse_camera_eo_csv

GlobLike = Union[str, Iterable[str]]

def _iter_globbed_paths(patterns: GlobLike) -> Iterable[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    seen: set[str] = set()
    for pat in patterns:
        for path in glob.iglob(pat, recursive=True):
            if os.path.isfile(path) and path not in seen:
                seen.add(path)
                yield os.path.abspath(path)

def dataset_builder(
    camera_id: str,
    io_file_path: str,
    eo_file_path: str,
    image_glob: GlobLike,
    *,
    read_metadata: bool = False,   # False = index only; True = read per-file metadata now
) -> PhotoDataset:
    """
    Build a dataset from one or more glob patterns (e.g., "C:/img/**/*.tif").
    - No pixels are loaded here.
    - Metadata can be deferred (read_metadata=False) and loaded later selectively.
    """
    ds = PhotoDataset()
    io = parse_camera_io_xml(io_file_path)
    ds.cameras[camera_id] = io

    eos = parse_camera_eo_csv(eo_file_path)
    io_factory = ImageIOFactory()

    for path in sorted(_iter_globbed_paths(image_glob)):
        image_id = os.path.splitext(os.path.basename(path))[0]

        width = height = bands = 0
        dtype = "unknown"
        geo = None

        if read_metadata:
            meta: Dict = io_factory.metadata(path)  # lightweight; still no pixel load
            width  = int(meta.get("width") or meta.get("WIDTH") or meta.get("RasterXSize") or 0)
            height = int(meta.get("height") or meta.get("HEIGHT") or meta.get("RasterYSize") or 0)
            bands  = int(meta.get("count") or meta.get("bands") or (3 if meta.get("mode") == "RGB" else 1))
            dtype  = str(meta.get("dtype") or meta.get("format") or "uint8")
            if "crs" in meta or "transform" in meta or "bounds" in meta:
                geo = RasterGeoMeta(
                    crs=meta.get("crs"),
                    transform_affine6=tuple(meta.get("transform")) if meta.get("transform") else None,
                    bounds_xyxy=tuple(meta.get("bounds")) if meta.get("bounds") else None,
                    nodata=meta.get("nodata"),
                )

        rec = ImageRecord(
            image_id=image_id,
            path=path,
            width=width,
            height=height,
            bands=bands,
            dtype=dtype,
            color_space=ColorSpace.RGB if bands == 3 else ColorSpace.UNKNOWN,
            geo=geo,
        )
        ds.images[image_id] = rec

        if image_id in eos:
            from ..entities import Shot
            ds.shots.append(Shot(image_id=image_id, camera_id=camera_id, io=io, eo=eos[image_id]))

    # Attach helpers for lazy operations
    ds._meta_loader = lambda img_id: _load_metadata_into_record(ds, img_id, io_factory)
    ds._open_image = lambda img_id: io_factory.open(ds.images[img_id].path)

    return ds

def _load_metadata_into_record(ds: PhotoDataset, image_id: str, io_factory: ImageIOFactory) -> None:
    rec = ds.images[image_id]
    meta = io_factory.metadata(rec.path)
    rec.width  = int(meta.get("width") or meta.get("WIDTH") or meta.get("RasterXSize") or 0)
    rec.height = int(meta.get("height") or meta.get("HEIGHT") or meta.get("RasterYSize") or 0)
    rec.bands  = int(meta.get("count") or meta.get("bands") or (3 if meta.get("mode") == "RGB" else 1))
    rec.dtype  = str(meta.get("dtype") or meta.get("format") or "uint8")
    if "crs" in meta or "transform" in meta or "bounds" in meta:
        rec.geo = RasterGeoMeta(
            crs=meta.get("crs"),
            transform_affine6=tuple(meta.get("transform")) if meta.get("transform") else None,
            bounds_xyxy=tuple(meta.get("bounds")) if meta.get("bounds") else None,
            nodata=meta.get("nodata"),
        )

# -------- public, clearer names than "hydrate" --------

def load_metadata(ds: PhotoDataset, image_ids: Optional[Iterable[str]] = None) -> None:
    """
    Populate width/height/bands/dtype/geo for selected images (or all if None).
    """
    if not hasattr(ds, "_meta_loader"):
        return
    ids = image_ids or list(ds.images.keys())
    for img_id in ids:
        ds._meta_loader(img_id)

def open_image(ds: PhotoDataset, image_id: str):
    """
    Open an image lazily. Use the returned reader to fetch tiles/windows.
    """
    return ds._open_image(image_id)
