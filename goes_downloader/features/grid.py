"""The target grid (coarsened ABI fixed grid), regridding onto it, and the sun and satellite geometry of its cells."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zarr

from ..common import iter_days_utc
from ..crop import geos_crs, scan_angle_rect
from ..store import output_zarr_path_for_day
from .settings import ChannelSpec, FeatureConfig


@dataclass(frozen=True)
class TargetGrid:
    """
    Cells of coarsen x coarsen reference pixels on the ABI fixed grid. Positions are in reference pixel
    units u counted from the uncropped sector's first pixel centre: scan angle = origin + u * step.
    Cell (row, col) holds pixels [row * coarsen, (row + 1) * coarsen) in y, likewise in x; row0/col0 are
    the first cell kept.
    """

    reference_product: str
    reference_km: float
    resolution_km: float
    coarsen: int
    x_origin: float
    x_step: float
    y_origin: float
    y_step: float  # negative: rows run north to south
    row0: int
    col0: int
    ny: int
    nx: int
    projection: Dict[str, Any]

    def _centres(self, first: int, n: int) -> np.ndarray:
        return (np.arange(first, first + n) * self.coarsen + (self.coarsen - 1) / 2).astype(np.float64)

    def x(self) -> np.ndarray:
        return self.x_origin + self._centres(self.col0, self.nx) * self.x_step

    def y(self) -> np.ndarray:
        return self.y_origin + self._centres(self.row0, self.ny) * self.y_step

    def cells_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Flat cell index for scan angles (rad); -1 outside the grid or off the Earth."""
        col = np.floor(((x - self.x_origin) / self.x_step + 0.5) / self.coarsen) - self.col0
        row = np.floor(((y - self.y_origin) / self.y_step + 0.5) / self.coarsen) - self.row0
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        cells = np.full(np.shape(x), -1, dtype=np.int64)
        cells[ok] = row[ok].astype(np.int64) * self.nx + col[ok].astype(np.int64)
        return cells

    def same_cells(self, other: "TargetGrid") -> bool:
        """The same cells: shape and scan-angle centres (to well under a pixel)."""
        return (self.ny, self.nx) == (other.ny, other.nx) and bool(
            np.allclose(self.x(), other.x(), atol=1e-7) and np.allclose(self.y(), other.y(), atol=1e-7))


def grid_latlon(grid: TargetGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centre lat/lon (deg, float32); NaN for cells off the Earth's disk."""
    from pyproj import CRS, Transformer

    h = float(grid.projection["perspective_point_height"])
    tfm = Transformer.from_crs(geos_crs(grid.projection), CRS.from_epsg(4326), always_xy=True)
    xx, yy = np.meshgrid(grid.x() * h, grid.y() * h)
    lon, lat = tfm.transform(xx, yy)
    lat, lon = np.asarray(lat, dtype=np.float32), np.asarray(lon, dtype=np.float32)
    off_earth = ~(np.isfinite(lat) & np.isfinite(lon))
    lat[off_earth] = np.nan
    lon[off_earth] = np.nan
    return lat, lon


def glm_projection(projection: Dict[str, Any], ellipsoid_km: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    """
    The projection whose ellipsoid GLM lat/lon refer to. GLM navigates each event to the lightning
    ellipsoid (the Earth ellipsoid raised by ~14 km at the equator, ~6 km at the poles), so reading the
    lat/lon on that ellipsoid recovers GLM's view angle. The satellite stays where it is: h shrinks as a grows.
    """
    if ellipsoid_km is None:
        return projection
    eq, pol = ellipsoid_km[0] * 1000.0, ellipsoid_km[1] * 1000.0
    return {
        **projection,
        "semi_major_axis": float(projection["semi_major_axis"]) + eq,
        "semi_minor_axis": float(projection["semi_minor_axis"]) + pol,
        "perspective_point_height": float(projection["perspective_point_height"]) - eq,
    }


class SolarGeometry:
    """
    Sun position per cell with the algorithm of NOAA's solar calculator (after Meeus, Astronomical
    Algorithms): about 0.01 deg in declination and a few seconds in solar time, against a scan that can
    fall anywhere in its 5-min slot. Local solar time t (true solar time, 0-24 h) is given as sin/cos of
    2 pi t / 24 h, so it has no jump at midnight: noon is (0, -1), midnight (0, 1).
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        phi, lam = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
        self.sin_phi, self.cos_phi = np.sin(phi), np.cos(phi)
        self.sin_lam, self.cos_lam = np.sin(lam), np.cos(lam)

    @staticmethod
    def sun(t_ns: int) -> Tuple[float, float]:
        """(declination in rad, equation of time in minutes) at t (ns since 1970, UTC)."""
        T = (t_ns / 86400e9 + 2440587.5 - 2451545.0) / 36525  # Julian centuries since J2000
        rad = math.radians
        l0 = rad((280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360)  # mean longitude
        m = rad(357.52911 + T * (35999.05029 - 0.0001537 * T))  # mean anomaly
        e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)  # orbit eccentricity
        centre = rad(math.sin(m) * (1.914602 - T * (0.004817 + 0.000014 * T))
                     + math.sin(2 * m) * (0.019993 - 0.000101 * T) + math.sin(3 * m) * 0.000289)
        omega = rad(125.04 - 1934.136 * T)
        apparent_longitude = l0 + centre - rad(0.00569 + 0.00478 * math.sin(omega))
        mean_obliquity = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
        obliquity = rad(mean_obliquity + 0.00256 * math.cos(omega))
        decl = math.asin(math.sin(obliquity) * math.sin(apparent_longitude))
        y = math.tan(obliquity / 2) ** 2
        eqtime = 4 * math.degrees(y * math.sin(2 * l0) - 2 * e * math.sin(m) + 4 * e * y * math.sin(m) * math.cos(2 * l0)
                                  - 0.5 * y * y * math.sin(4 * l0) - 1.25 * e * e * math.sin(2 * m))
        return decl, eqtime

    def frames(self, t_ns: int) -> Dict[str, np.ndarray]:
        decl, eqtime = self.sun(t_ns)
        minutes = (t_ns % (86400 * 10**9)) / 60e9
        # True solar time as an angle: (minutes + eqtime) / 1440 turns at Greenwich, plus the longitude
        b = 2 * math.pi * (minutes + eqtime) / 1440
        day_cos = math.cos(b) * self.cos_lam - math.sin(b) * self.sin_lam
        day_sin = math.sin(b) * self.cos_lam + math.cos(b) * self.sin_lam
        # The hour angle is that angle minus pi, so cos(hour angle) = -day_cos
        cos_zenith = self.sin_phi * math.sin(decl) - self.cos_phi * math.cos(decl) * day_cos
        return {
            "cos_zenith": cos_zenith.astype(np.float32),
            "local_hour_sin": day_sin.astype(np.float32),
            "local_hour_cos": day_cos.astype(np.float32),
        }


def satellite_zenith_deg(lat: np.ndarray, lon: np.ndarray, projection: Dict[str, Any]) -> np.ndarray:
    """Angle between the local vertical and the direction to the satellite, on the ground ellipsoid (deg)."""
    a, b = float(projection["semi_major_axis"]), float(projection["semi_minor_axis"])
    r_sat = a + float(projection["perspective_point_height"])
    lon0 = math.radians(float(projection["longitude_of_projection_origin"]))
    phi, lam = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
    e2 = 1 - (b / a) ** 2
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    up = (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))  # ellipsoid normal
    point = (n * up[0], n * up[1], n * (1 - e2) * up[2])
    to_sat = (r_sat * math.cos(lon0) - point[0], r_sat * math.sin(lon0) - point[1], -point[2])
    cos_z = sum(v * u for v, u in zip(to_sat, up)) / np.sqrt(sum(v * v for v in to_sat))
    return np.degrees(np.arccos(np.clip(cos_z, -1.0, 1.0))).astype(np.float32)


def _nominal_km(step_rad: float, projection: Dict[str, Any]) -> float:
    """Sub-satellite pixel size, rounded to 0.5 km (56 urad -> 2 km, 280 urad -> 10 km)."""
    return round(step_rad * float(projection["perspective_point_height"]) / 500.0) / 2.0


def make_grid(cfg: FeatureConfig) -> TargetGrid:
    """Builds the target grid from the first raw day in the range that holds every product used."""
    products = list(dict.fromkeys(c.product for c in cfg.channels if c.kind == "product"))
    if cfg.reference_product is not None:
        products = [cfg.reference_product]

    groups: Dict[str, Any] = {}
    for day in iter_days_utc(cfg.start_date_utc, cfg.end_date_utc):
        path = output_zarr_path_for_day(cfg.raw.raw_output_root_dir, day)
        if not os.path.exists(os.path.join(path, ".zgroup")):
            continue
        root = zarr.open_consolidated(path, mode="r")
        groups = {k: root[f"abi/{k}"] for k in products if f"abi/{k}" in root and root[f"abi/{k}"].attrs.get("has_frames")}
        if len(groups) == len(products):
            break
    else:
        raise ValueError(f"no raw day between {cfg.start_date_utc} and {cfg.end_date_utc} holds all of {products}")

    steps = {k: abs(float(g["x_scan_angle_rad"][-1] - g["x_scan_angle_rad"][0])) / (g["x_scan_angle_rad"].shape[0] - 1) for k, g in groups.items()}
    ref = min(products, key=lambda k: steps[k])  # finest grid; ties keep config order
    g = groups[ref]
    projection = dict(g.attrs["goes_imager_projection"])
    for k, other in groups.items():
        lon0 = other.attrs["goes_imager_projection"]["longitude_of_projection_origin"]
        if lon0 != projection["longitude_of_projection_origin"]:
            raise ValueError(f"{k} is on a different projection (lon_0 {lon0}) than {ref}")

    ref_km = _nominal_km(steps[ref], projection)
    coarsen_f = cfg.resolution_km / ref_km
    coarsen = int(round(coarsen_f))
    if coarsen < 1 or abs(coarsen_f - coarsen) > 1e-6:
        raise ValueError(f"features.grid.resolution_km must be a multiple of {ref_km:g} km, the {ref} grid")

    x, y = g["x_scan_angle_rad"][:], g["y_scan_angle_rad"][:]
    x_step, y_step = float(x[-1] - x[0]) / (len(x) - 1), float(y[-1] - y[0]) / (len(y) - 1)
    crop_x0, crop_y0 = int(g.attrs["crop_x0"]), int(g.attrs["crop_y0"])
    x_origin, y_origin = float(x[0]) - crop_x0 * x_step, float(y[0]) - crop_y0 * y_step

    # Cells lying entirely inside the reference product's stored pixels ...
    col_lo, col_hi = -(-crop_x0 // coarsen), (crop_x0 + len(x)) // coarsen
    row_lo, row_hi = -(-crop_y0 // coarsen), (crop_y0 + len(y)) // coarsen
    # ... that overlap the bounds
    if cfg.bounds is not None:
        x_min, x_max, y_min, y_max = scan_angle_rect(cfg.bounds, projection)
        ux = sorted([(x_min - x_origin) / x_step, (x_max - x_origin) / x_step])
        uy = sorted([(y_min - y_origin) / y_step, (y_max - y_origin) / y_step])
        col_lo, col_hi = max(col_lo, math.floor((ux[0] + 0.5) / coarsen)), min(col_hi, math.floor((ux[1] + 0.5) / coarsen) + 1)
        row_lo, row_hi = max(row_lo, math.floor((uy[0] + 0.5) / coarsen)), min(row_hi, math.floor((uy[1] + 0.5) / coarsen) + 1)
    if col_hi <= col_lo or row_hi <= row_lo:
        raise ValueError("the feature bounds do not overlap the raw data")

    return TargetGrid(
        reference_product=ref, reference_km=ref_km, resolution_km=cfg.resolution_km, coarsen=coarsen,
        x_origin=x_origin, x_step=x_step, y_origin=y_origin, y_step=y_step,
        row0=row_lo, col0=col_lo, ny=row_hi - row_lo, nx=col_hi - col_lo, projection=projection,
    )


# =============================================================================
# Regridding source products onto the grid
# =============================================================================

@dataclass
class AxisMap:
    """
    How one source axis becomes the grid axis. reduce: cell i aggregates source pixels
    [index[i], index[i] + counts[i]) of the slice lo:hi. take: cell i copies source pixel index[i]
    (counts[i] is 0 where the cell centre is outside the source).
    """

    take: bool
    index: np.ndarray
    counts: np.ndarray
    lo: int
    hi: int
    n_reduce: int = 0  # reduce: cells up to the last one with source pixels (reduceat can't end a segment early)
    block: int = 0  # reduce: pixels per cell when every cell has the same number, which allows a fast reshape


def axis_map(src_u: np.ndarray, first_cell: int, n_cells: int, coarsen: int) -> AxisMap:
    """src_u: source pixel centres in reference pixel units, increasing."""
    step = float(src_u[-1] - src_u[0]) / (len(src_u) - 1) if len(src_u) > 1 else 1.0
    if step <= coarsen * (1 + 1e-6):
        cell = np.floor((src_u + 0.5) / coarsen).astype(np.int64) - first_cell
        targets = np.arange(n_cells)
        starts = np.searchsorted(cell, targets, "left")
        counts = np.searchsorted(cell, targets, "right") - starts
        lo, hi = int(starts[0]), int(starts[-1] + counts[-1])
        # reduceat needs indices inside the slice; empty cells are masked through counts == 0
        index = np.clip(starts - lo, 0, max(hi - lo - 1, 0))
        nonempty = np.nonzero(counts)[0]
        n_reduce = int(nonempty[-1]) + 1 if nonempty.size else 1
        k = int(counts[0])
        block = k if k > 0 and (counts == k).all() else 0
        return AxisMap(False, index, counts, lo, max(hi, lo + 1), n_reduce, block)
    centre = np.arange(first_cell, first_cell + n_cells) * coarsen + (coarsen - 1) / 2
    idx = np.floor((centre - src_u[0]) / step + 0.5 - 1e-6).astype(np.int64)  # ties go to the lower pixel
    ok = (idx >= 0) & (idx < len(src_u))
    if not ok.any():
        return AxisMap(True, np.zeros(n_cells, np.int64), np.zeros(n_cells, np.int64), 0, 1)
    lo, hi = int(idx[ok].min()), int(idx[ok].max()) + 1
    return AxisMap(True, np.clip(idx - lo, 0, hi - lo - 1), ok.astype(np.int64), lo, hi)


def reduce2d(a: np.ndarray, ymap: AxisMap, xmap: AxisMap, ufunc: Any) -> np.ndarray:
    for axis, m in ((0, ymap), (1, xmap)):
        if m.take:
            a = np.take(a, m.index, axis=axis)
            continue
        if m.block:  # equal blocks: combining every block-th slice is much faster than reduceat's short segments
            idx = [slice(None)] * a.ndim
            idx[axis] = slice(0, None, m.block)
            out = a[tuple(idx)].copy()
            for i in range(1, m.block):
                idx[axis] = slice(i, None, m.block)
                ufunc(out, a[tuple(idx)], out=out)
            a = out
            continue
        a = ufunc.reduceat(a, m.index[: m.n_reduce], axis=axis)
        if m.n_reduce < len(m.index):  # trailing cells without source pixels, masked later
            pad = [(0, 0)] * a.ndim
            pad[axis] = (0, len(m.index) - m.n_reduce)
            a = np.pad(a, pad)
    return a


class SourceFrame:
    """One decoded source frame (for one set of valid DQF values) with the pieces its channels share."""

    def __init__(self, valid: np.ndarray, phys: np.ndarray, ymap: AxisMap, xmap: AxisMap):
        self.valid, self.phys, self.ymap, self.xmap = valid, phys, ymap, xmap
        self.total = ymap.counts[:, None] * xmap.counts[None, :]
        self._n: Optional[np.ndarray] = None
        self._zeroed: Optional[np.ndarray] = None

    def reduce(self, a: np.ndarray, ufunc: Any) -> np.ndarray:
        return reduce2d(a, self.ymap, self.xmap, ufunc)

    @property
    def n(self) -> np.ndarray:
        """Valid source pixels per cell."""
        if self._n is None:
            self._n = self.reduce(self.valid.astype(np.float32), np.add)
        return self._n

    @property
    def zeroed(self) -> np.ndarray:
        """Physical values with invalid pixels set to 0, for sums."""
        if self._zeroed is None:
            self._zeroed = np.where(self.valid, self.phys, np.float32(0))
        return self._zeroed


def matches_any(phys: np.ndarray, values: Any) -> np.ndarray:
    """
    Where float32 physical values equal any of the (float64) config values. Exact equality would miss
    values such as 0.1 that float32 can't hold, so allow a relative 1e-5, far below any spacing of
    category values.
    """
    hit = np.zeros(phys.shape, dtype=bool)
    for v in values:
        hit |= np.abs(phys - np.float32(v)) <= np.float32(1e-5 * max(1.0, abs(v)))
    return hit


def aggregate(ch: ChannelSpec, src: SourceFrame) -> np.ndarray:
    """One product channel on the grid."""
    n = src.n
    with np.errstate(invalid="ignore", divide="ignore"):
        if ch.aggregation in ("min", "max"):
            out = src.reduce(np.where(src.valid, src.phys, np.float32(np.nan)), np.fmin if ch.aggregation == "min" else np.fmax)
        elif ch.aggregation == "fraction":
            out = src.reduce((src.valid & matches_any(src.phys, ch.values)).astype(np.float32), np.add) / n
        elif ch.aggregation == "mean":
            out = src.reduce(src.zeroed, np.add) / n
        else:
            x = src.zeroed.astype(np.float64)
            mean = src.reduce(x, np.add) / n
            out = np.sqrt(np.maximum(src.reduce(x * x, np.add) / n - mean * mean, 0.0))
        out = out.astype(np.float32, copy=False)
        out[(src.total == 0) | (n == 0) | (n < ch.min_valid_fraction * src.total)] = np.nan
    return out
