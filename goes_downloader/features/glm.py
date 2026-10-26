"""GLM flashes on the grid: flash centroids, flash footprints from events (extent), coverage, label windows."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common import GLM_WINDOW_SECONDS, NAT_NS, SLOT_SECONDS, SLOTS_PER_DAY
from ..crop import latlon_to_scan_angle
from .grid import TargetGrid, glm_projection
from .raw import RawTimeline
from .settings import SLOT_MINUTES, FeatureConfig

SLOT_NS = SLOT_SECONDS * 1_000_000_000
MINUTE_NS = 60 * 1_000_000_000
DAY_NS = SLOTS_PER_DAY * SLOT_NS
GLM_WINDOWS_PER_SLOT = SLOT_SECONDS // GLM_WINDOW_SECONDS
# A flash's first event lies within its file's 20 s, or up to ~2 s before it (it began in the
# previous file's window), so the files that can hold flashes of [t0, t1) start in [t0 - 20 s, t1 + 20 s)
FILE_MARGIN_NS = GLM_WINDOW_SECONDS * 1_000_000_000
EVENT_CHUNK = 5_000_000  # events mapped at a time (~1 GB peak); a day has ~50 M
COVER_BATCH = 500_000  # pixels spread onto cells at a time; each makes up to 25 (pixel, cell) pairs on a 2 km grid
PIXEL_CORNERS_PATH = os.path.join(os.path.dirname(__file__), "glm_pixel_corners.npz")
# How footprints are made, recorded in every store: stores built another way hold different extents and labels
FOOTPRINT = "glmtools pixel corners (G16 table), area-weighted, each flash at most once per cell, density rounded half up"
# Events closer than this are one pixel of one flash, as in glmtools (get_lutevents): under half the
# smallest pixel (152 urad), and well over the ~6 urad jitter between repeats of one pixel
PIXEL_ID_RAD = 28e-6


def _cat(parts: List[np.ndarray], dtype: Any) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)


class PixelCorners:
    """
    The GLM pixel at any view angle, from glmtools' table of pixel-corner offsets on the fixed grid
    (glm_pixel_corners.npz; its provenance and licence are in glm_pixel_corners_LICENSE.txt). The table is
    GOES-16's; GOES-19 has the same instrument at the same longitude, and its event spacing matches the
    table (README, "Flash footprints"). Each pixel is taken as the rectangle through the means of its
    corners: the table's quadrilaterals tilt by under 0.02, far less than a cell.
    """

    def __init__(self, path: str = PIXEL_CORNERS_PATH):
        with np.load(path) as f:
            self.x, self.y = f["x_urad"] * 1e-6, f["y_urad"] * 1e-6
            c = f["corners_urad"].astype(np.float64) * 1e-6  # [y, x, (nw, sw, se, ne), (x, y)]
        self.sides = np.stack([(c[..., 0, 0] + c[..., 1, 0]) / 2, (c[..., 2, 0] + c[..., 3, 0]) / 2,   # west, east
                               (c[..., 1, 1] + c[..., 2, 1]) / 2, (c[..., 0, 1] + c[..., 3, 1]) / 2])  # south, north

    def rectangles(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scan angles (rad) of pixel centres -> their (west, east, south, north) edges (rad), bilinear in the table."""
        i = np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2)
        j = np.clip(np.searchsorted(self.y, y) - 1, 0, len(self.y) - 2)
        tx = np.clip((x - self.x[i]) / (self.x[i + 1] - self.x[i]), 0.0, 1.0)  # the edge value beyond the table
        ty = np.clip((y - self.y[j]) / (self.y[j + 1] - self.y[j]), 0.0, 1.0)
        s = self.sides
        off = (s[:, j, i] * (1 - tx) * (1 - ty) + s[:, j, i + 1] * tx * (1 - ty)
               + s[:, j + 1, i] * (1 - tx) * ty + s[:, j + 1, i + 1] * tx * ty)
        return x + off[0], x + off[1], y + off[2], y + off[3]


def pixel_cover(grid: TargetGrid, flash: np.ndarray, minute: np.ndarray, west: np.ndarray, east: np.ndarray,
                south: np.ndarray, north: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flashes' pixels spread onto `grid` by area, as glmtools grids flash-extent density: (minute, cell,
    cover) per minute and cell with any cover, sorted by minute then cell. `cover` sums over the minute's
    flashes the share of the cell under each flash's pixels, at most 1 per flash: one flash covers a cell
    once even where two of its pixels overlap. Inputs are one entry per distinct (flash, pixel); `minute`
    is the flash's.
    """
    ncells = grid.ny * grid.nx
    order = np.argsort(minute, kind="stable")
    flash, minute, west, east, south, north = (a[order] for a in (flash, minute, west, east, south, north))
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    a = 0
    while a < len(minute):
        # Batches end on a minute boundary, so no (minute, cell) is split between two of them
        b = min(a + COVER_BATCH, len(minute))
        b = int(np.searchsorted(minute, minute[b - 1], "right"))
        uw, vn = grid.fractional(west[a:b], north[a:b])
        ue, vs = grid.fractional(east[a:b], south[a:b])
        u0, u1, v0, v1 = np.minimum(uw, ue), np.maximum(uw, ue), np.minimum(vn, vs), np.maximum(vn, vs)
        fu, fv = np.floor(u0), np.floor(v0)
        keys, weights = [], []
        for i in range(int((np.floor(u1) - fu).max()) + 1):
            col = fu + i
            du = np.minimum(u1, col + 1) - np.maximum(u0, col)
            for j in range(int((np.floor(v1) - fv).max()) + 1):
                row = fv + j
                w = du * (np.minimum(v1, row + 1) - np.maximum(v0, row))
                ok = (du > 0) & (w > 0) & (col >= 0) & (col < grid.nx) & (row >= 0) & (row < grid.ny)
                keys.append(flash[a:b][ok] * ncells + (row[ok] * grid.nx + col[ok]).astype(np.int64))
                weights.append(w[ok])
        key, w = np.concatenate(keys), np.concatenate(weights)
        if key.size:
            per_flash, inv = np.unique(key, return_inverse=True)
            cover = np.minimum(np.bincount(inv, w), 1.0)
            flashes, first = np.unique(flash[a:b], return_index=True)
            m = minute[a:b][first][np.searchsorted(flashes, per_flash // ncells)]
            per_minute, inv = np.unique(m * ncells + per_flash % ncells, return_inverse=True)
            out.append((per_minute // ncells, per_minute % ncells, np.bincount(inv, cover).astype(np.float32)))
        a = b
    if not out:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return tuple(np.concatenate(p) for p in zip(*out))  # type: ignore[return-value]


def round_density(cover: np.ndarray) -> np.ndarray:
    """Summed cover -> flashes, rounded half up as glmtools rounds flash-extent density (the epsilon absorbs float sums)."""
    return np.floor(cover + 0.5 + 1e-9).astype(np.int64)


class GlmTimeline:
    """
    Quality-filtered GLM data on the grid for global slots [s0, s1):
    - centroids: one (slot, cell) per flash, with its energy and area;
    - extent: each flash's footprint, the cells under its pixels weighted by area (`pixel_cover`), per
      minute; the basis of a flash-extent density, flashes per cell, computed as glmtools does. Also on
      `fine`, a finer grid nesting in `grid`, when a label needs it. Needs glm/events;
    - coverage: the fraction of 20 s GLM windows with data, per slot.
    A flash counts at the minute (and slot) of its first event and belongs to the day store of its file:
    a flash that starts a second or two before midnight is in the next day's first file. So the files that
    can hold flashes of the range are read, whichever day holds them, and flashes are selected by their
    own time.
    """

    def __init__(self, cfg: FeatureConfig, grid: TargetGrid, raw: RawTimeline, s0: int, s1: int,
                 fine: Optional[TargetGrid] = None):
        self.s0, self.s1, self.ncells = s0, s1, grid.ny * grid.nx
        self.m0 = s0 * SLOT_MINUTES  # minutes since 1970 at s0
        self.coverage = np.zeros(s1 - s0, dtype=np.float64)
        proj = glm_projection(grid.projection, cfg.glm_ellipsoid_km)
        slots, cells, energy, area = [], [], [], []
        pixels: List[Tuple[np.ndarray, ...]] = []
        n_before = 0  # flashes of earlier days, so a flash's number is unique across the days read
        f0, f1 = s0 * SLOT_NS - FILE_MARGIN_NS, s1 * SLOT_NS + FILE_MARGIN_NS  # file start times to read
        for d in range(f0 // DAY_NS, (f1 - 1) // DAY_NS + 1):
            root = raw.day(d).root
            if root is None or "glm/coverage" not in root:
                continue
            per_slot = root["glm/coverage/covered"][:].reshape(SLOTS_PER_DAY, GLM_WINDOWS_PER_SLOT).mean(axis=1)
            a, b = max(s0, d * SLOTS_PER_DAY), min(s1, (d + 1) * SLOTS_PER_DAY)
            if b > a:
                self.coverage[a - s0 : b - s0] = per_slot[a - d * SLOTS_PER_DAY : b - d * SLOTS_PER_DAY]

            f = root["glm/flashes"]
            t = f["time_first_ns"][:]
            keep = (t != NAT_NS) & (t >= s0 * SLOT_NS) & (t < s1 * SLOT_NS)
            if cfg.glm_quality_flags is not None:
                keep &= np.isin(f["quality_flag"][:], cfg.glm_quality_flags)
            if not keep.any():
                n_before += len(t)
                continue
            flash_slot = t // SLOT_NS
            x, y = latlon_to_scan_angle(f["lat"][:][keep], f["lon"][:][keep], proj)
            c = grid.cells_of(np.asarray(x), np.asarray(y))
            inside = c >= 0
            slots.append(flash_slot[keep][inside])
            cells.append(c[inside])
            energy.append(f["energy_j"][:][keep][inside].astype(np.float64) * 1e15)
            area.append(f["area_m2"][:][keep][inside].astype(np.float64) / 1e6)

            # Only the events of the files that can hold the range's flashes (events are in file order)
            files = root["glm/files"]
            start = files["start_ns"][:]
            first_event = np.concatenate(([0], np.cumsum(files["n_events"][:], dtype=np.int64)))
            sel = np.flatnonzero((start >= f0) & (start < f1))
            if sel.size:
                e0, e1 = int(first_event[sel[0]]), int(first_event[sel[-1] + 1])
                fi, px, py = self._flash_pixels(root["glm/events"], e0, e1, keep, grid, proj)
                pixels.append((fi + n_before, t[fi] // MINUTE_NS - self.m0, px, py))
            n_before += len(t)

        slot = _cat(slots, np.int64)
        order = np.argsort(slot, kind="stable")
        self.slot = slot[order]
        self.cell = _cat(cells, np.int64)[order]
        self.weights = {"count": None, "energy_fj": _cat(energy, np.float64)[order], "area_km2": _cat(area, np.float64)[order]}

        flash, minute, px, py = (_cat([p[k] for p in pixels], dt) for k, dt in enumerate((np.int64, np.int64, np.float64, np.float64)))
        sides = PixelCorners().rectangles(px, py)
        self.extent = {False: pixel_cover(grid, flash, minute, *sides)}
        self.fine_ncells = 0
        if fine is not None:
            self.extent[True] = pixel_cover(fine, flash, minute, *sides)
            self.fine_ncells = fine.ny * fine.nx

    @staticmethod
    def _flash_pixels(events: Any, e0: int, e1: int, keep: np.ndarray, grid: TargetGrid,
                      proj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (flash, x, y): every distinct pixel of each kept flash among events [e0, e1) that can reach the
        grid, at the scan angles of its first event. A flash lights the same pixel many times; that is one
        pixel of its footprint. Orphan events (flash_index -1: their group or flash can't be found in
        NOAA's file, a few hundred a day) belong to no flash and are skipped.
        """
        keys, xs, ys = [], [], []
        for a in range(e0, e1, EVENT_CHUNK):
            b = min(a + EVENT_CHUNK, e1)
            fi = events["flash_index"][a:b]
            sel = (fi >= 0) & keep[np.maximum(fi, 0)]
            if not sel.any():
                continue
            x, y = latlon_to_scan_angle(events["lat"][a:b][sel], events["lon"][a:b][sel], proj)
            x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
            col, row = grid.fractional(x, y)
            # A pixel is at most ~1.1 cells of the coarsest grid across, so one centred 2 cells outside can't reach it
            with np.errstate(invalid="ignore"):
                near = (col > -2) & (col < grid.nx + 2) & (row > -2) & (row < grid.ny + 2)
            pixel = (np.rint(x[near] / PIXEL_ID_RAD).astype(np.int64) + 2**13) * 2**14 + np.rint(y[near] / PIXEL_ID_RAD).astype(np.int64) + 2**13
            key, first = np.unique(fi[sel][near].astype(np.int64) * 2**28 + pixel, return_index=True)
            keys.append(key)
            xs.append(x[near][first])
            ys.append(y[near][first])
        if not keys:
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
        # A flash's events all come from its file, but a file can straddle a chunk boundary
        key, first = np.unique(np.concatenate(keys), return_index=True)
        return key // 2**28, np.concatenate(xs)[first], np.concatenate(ys)[first]

    @staticmethod
    def _range(slot: np.ndarray, a: int, b: int) -> slice:
        return slice(int(np.searchsorted(slot, a, "left")), int(np.searchsorted(slot, b, "left")))

    def sums(self, a: int, b: int, stat: str = "count") -> np.ndarray:
        """
        Flat per-cell totals over slots [a, b): flashes (count), their energy or area, or the flashes
        covering the cell weighted by the share they cover (extent: the flash-extent density before
        rounding, so it adds up exactly over any window).
        """
        if stat == "extent":
            minute, cell, cover = self.extent[False]
            r = self._range(minute, (a - self.s0) * SLOT_MINUTES, (b - self.s0) * SLOT_MINUTES)
            return np.bincount(cell[r], weights=cover[r], minlength=self.ncells)
        r = self._range(self.slot, a, b)
        w = self.weights[stat]
        return np.bincount(self.cell[r], weights=None if w is None else w[r], minlength=self.ncells)

    def extent_peak(self, a: int, b: int, minutes: int, fine: bool = False) -> np.ndarray:
        """
        Flat per-cell flash-extent density over slots [a, b) taken in windows of `minutes` counted from a,
        each rounded as glmtools rounds it: the largest window's. With one window, the window's density
        (what flash_extent stores and extent labels count).
        """
        minute, cell, cover = self.extent[fine]
        lo, hi = (a - self.s0) * SLOT_MINUTES, (b - self.s0) * SLOT_MINUTES
        r = self._range(minute, lo, hi)
        n = self.fine_ncells if fine else self.ncells
        windows, inv = np.unique((minute[r] - lo) // minutes * n + cell[r], return_inverse=True)
        density = round_density(np.bincount(inv, cover[r]))
        out = np.zeros(n, np.int64)
        np.maximum.at(out, windows % n, density)
        return out

    def coverage_mean(self, a: int, b: int) -> float:
        a, b = max(a, self.s0), min(b, self.s1)
        return float(self.coverage[a - self.s0 : b - self.s0].sum()) / max(1, b - a) if b > a else 0.0

    def n_flashes(self, a: int, b: int) -> int:
        r = self._range(self.slot, a, b)
        return r.stop - r.start


def disk_offsets(radius: float) -> List[Tuple[int, int]]:
    r = int(math.floor(radius))
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= radius * radius + 1e-9]


def neighbourhood_sum(a: np.ndarray, offsets: List[Tuple[int, int]]) -> np.ndarray:
    """Sum of `a` over the cells at the given offsets (zero beyond the edges)."""
    if offsets == [(0, 0)]:
        return a
    r = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    padded = np.pad(a, r)
    ny, nx = a.shape
    out = np.zeros_like(a)
    for dy, dx in offsets:
        out += padded[r + dy : r + dy + ny, r + dx : r + dx + nx]
    return out
