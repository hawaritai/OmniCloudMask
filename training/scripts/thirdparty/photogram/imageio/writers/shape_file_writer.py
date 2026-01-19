from __future__ import annotations
import os
from typing import Iterable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import fiona
from fiona.crs import from_epsg
from shapely.geometry import Polygon, LinearRing, mapping
from shapely.validation import make_valid

Coord = Tuple[float, float]

def _ensure_closed(coords: List[Coord]) -> List[Coord]:
    if not coords:
        return coords
    if coords[0] != coords[-1]:
        return coords + [coords[0]]
    return coords

def _delete_shapefile_family(path: str) -> None:
    base, _ = os.path.splitext(path)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        f = base + ext
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

@dataclass
class ShapefileWriter:
    """
    Simple helper to create a Polygon shapefile and add features one-by-one.

    - Supports exterior + optional holes
    - Validates and auto-closes rings
    - Lets you attach attributes with each polygon
    - Works as a context manager (recommended)

    Parameters
    ----------
    path : str
        Output file path ending with .shp
    crs_epsg : int
        EPSG code (default 4326 WGS84)
    fields : Dict[str, str]
        Fiona schema for properties, e.g. {"id": "int", "name": "str:80"}
        If None, uses a sensible default.
    overwrite : bool
        If True, deletes existing shapefile family before writing.
    """
    path: str
    crs_epsg: int = 4326
    fields: Optional[Dict[str, str]] = None
    overwrite: bool = False

    _sink: Optional[fiona.Collection] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.path.lower().endswith(".shp"):
            raise ValueError("Output path must end with .shp")
        if self.overwrite:
            _delete_shapefile_family(self.path)
        if self.fields is None:
            # Default schema: id + name; extend as needed
            self.fields = {"id": "int", "name": "str:80"}

        schema = {
            "geometry": "Polygon",
            "properties": self.fields,
        }
        self._sink = fiona.open(
            self.path,
            mode="w",
            driver="ESRI Shapefile",
            schema=schema,
            crs=from_epsg(int(self.crs_epsg)),
            encoding="utf-8",
        )

    # --- Context manager support ---
    def __enter__(self) -> "ShapefileWriter":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --- API ---
    def add_polygon(
        self,
        exterior: Iterable[Coord],
        *,
        holes: Optional[List[Iterable[Coord]]] = None,
        properties: Optional[Dict[str, Any]] = None,
        auto_fix: bool = True,
    ) -> None:
        """
        Add a polygon feature.

        exterior: list of (x,y)
        holes: list of rings, each a list of (x,y)
        properties: dict with keys matching the schema fields
        auto_fix: if True, attempts to fix invalid polygons
        """
        if self._sink is None:
            raise RuntimeError("Writer is closed.")

        ext = list(exterior)
        if len(ext) < 3:
            raise ValueError("Exterior ring must have at least 3 distinct points.")
        ext = _ensure_closed(ext)

        holes_list: List[List[Coord]] = []
        if holes:
            for h in holes:
                ring = _ensure_closed(list(h))
                if len(ring) < 4:
                    # Need at least 4 to be a closed triangle (first=last)
                    raise ValueError("Each hole must have at least 4 coordinates (closed).")
                holes_list.append(ring)

        poly = Polygon(ext, holes_list)

        if auto_fix:
            if not poly.is_valid:
                poly = make_valid(poly)
            # make_valid may return a MultiPolygon; keep only polygon part(s)
            if poly.geom_type == "MultiPolygon":
                # choose largest area piece for simplicity
                poly = max(poly.geoms, key=lambda g: g.area)

        if poly.is_empty or poly.area == 0:
            raise ValueError("Polygon is empty or has zero area after validation.")

        # Attribute defaults (ensure every declared field exists)
        props = {k: None for k in self.fields or {}}
        if properties:
            for k, v in properties.items():
                if k not in props:
                    raise KeyError(f"Unknown field '{k}' in properties; schema={list(props.keys())}")
                props[k] = v

        self._sink.write({
            "geometry": mapping(poly),
            "properties": props,
        })

    def close(self):
        if self._sink is not None:
            try:
                self._sink.close()
            finally:
                self._sink = None
