"""The lat/lon bounds as a crop of the ABI fixed grid (ABI only; GLM is never cropped)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import LatLonBounds

_crop_rects: Dict[str, Tuple[float, float, float, float]] = {}  # per worker process


def geos_crs(projection: Dict[str, Any]) -> Any:
    """The ABI fixed grid (geostationary projection, metres) as a pyproj CRS."""
    from pyproj import CRS

    return CRS.from_proj4(
        f"+proj=geos +lon_0={projection['longitude_of_projection_origin']} +h={float(projection['perspective_point_height'])} "
        f"+a={projection['semi_major_axis']} +b={projection['semi_minor_axis']} "
        f"+sweep={projection['sweep_angle_axis']} +units=m +no_defs"
    )


def latlon_to_scan_angle(lat: np.ndarray, lon: np.ndarray, projection: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Lat/lon (deg) -> ABI fixed-grid scan angles (rad). Points not visible from the satellite -> inf."""
    from pyproj import CRS, Transformer

    h = float(projection["perspective_point_height"])
    tfm = Transformer.from_crs(CRS.from_epsg(4326), geos_crs(projection), always_xy=True)
    x_m, y_m = tfm.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    return np.asarray(x_m) / h, np.asarray(y_m) / h


def scan_angle_rect(bounds: LatLonBounds, projection: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Smallest (x_min, x_max, y_min, y_max) scan-angle rectangle containing the lat/lon box."""
    n = 200
    lats = np.linspace(bounds.lat_min, bounds.lat_max, n)
    lons = np.linspace(bounds.lon_min, bounds.lon_max, n)
    # The extremes of a smooth map over the box lie on its perimeter
    edge_lat = np.concatenate([lats, lats, np.full(n, bounds.lat_min), np.full(n, bounds.lat_max)])
    edge_lon = np.concatenate([np.full(n, bounds.lon_min), np.full(n, bounds.lon_max), lons, lons])
    x, y = latlon_to_scan_angle(edge_lat, edge_lon, projection)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        raise ValueError(f"bounds {bounds} are not visible from the satellite")
    return float(x[ok].min()), float(x[ok].max()), float(y[ok].min()), float(y[ok].max())


def crop_slice(centers: np.ndarray, lo: float, hi: float, name: str) -> slice:
    """Pixels whose cell overlaps [lo, hi]; centers are monotonic (x ascends, y descends)."""
    half = abs(float(centers[1] - centers[0])) / 2 if len(centers) > 1 else 0.0
    idx = np.nonzero((centers + half >= lo) & (centers - half <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"lat/lon bounds do not overlap the {name} grid")
    return slice(int(idx[0]), int(idx[-1]) + 1)


@dataclass
class DayCrop:
    """The configured bounds as one scan-angle rectangle, fixed by the first ABI scan decoded in the day."""

    bounds: LatLonBounds
    rect: Optional[Tuple[float, float, float, float]] = None

    def rect_for(self, projection: Dict[str, Any]) -> Tuple[float, float, float, float]:
        if self.rect is None:
            self.rect = scan_angle_rect(self.bounds, projection)
        return self.rect


def crop_window(bounds: LatLonBounds, projection: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> Tuple[int, int, int, int]:
    """(y0, y1, x0, x1) the main process will keep, computed the same way as there."""
    key = json.dumps([vars(bounds), projection], sort_keys=True)
    if key not in _crop_rects:
        _crop_rects[key] = scan_angle_rect(bounds, projection)
    x0, x1, y0, y1 = _crop_rects[key]
    ys, xs = crop_slice(y, y0, y1, "y"), crop_slice(x, x0, x1, "x")
    return ys.start, ys.stop, xs.start, xs.stop
