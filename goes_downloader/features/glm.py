"""GLM flashes on the grid: flash centroids, flash footprints from events (extent), coverage, label windows."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from ..common import GLM_WINDOW_SECONDS, NAT_NS, SLOT_SECONDS, SLOTS_PER_DAY
from ..crop import latlon_to_scan_angle
from .grid import TargetGrid, glm_projection
from .raw import RawTimeline
from .settings import FeatureConfig

SLOT_NS = SLOT_SECONDS * 1_000_000_000
DAY_NS = SLOTS_PER_DAY * SLOT_NS
GLM_WINDOWS_PER_SLOT = SLOT_SECONDS // GLM_WINDOW_SECONDS
# A flash's first event lies within its file's 20 s, or up to ~2 s before it (it began in the
# previous file's window), so the files that can hold flashes of [t0, t1) start in [t0 - 20 s, t1 + 20 s)
FILE_MARGIN_NS = GLM_WINDOW_SECONDS * 1_000_000_000
EVENT_CHUNK = 5_000_000  # events mapped at a time (~1 GB peak); a day has ~50 M


def _cat(parts: List[np.ndarray], dtype: Any) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)


class GlmTimeline:
    """
    Quality-filtered GLM data on the grid for global slots [s0, s1):
    - centroids: one (slot, cell) per flash, with its energy and area;
    - extent: one (slot, cell) per flash and cell that any of its events fell in, i.e. the flash's
      footprint, the basis of a flash-extent density (flashes per cell). Needs glm/events;
    - coverage: the fraction of 20 s GLM windows with data, per slot.
    A flash counts at the slot of its first event and belongs to the day store of its file: a flash
    that starts a second or two before midnight is in the next day's first file. So the files that can
    hold flashes of the range are read, whichever day holds them, and flashes are selected by their
    own time.
    """

    def __init__(self, cfg: FeatureConfig, grid: TargetGrid, raw: RawTimeline, s0: int, s1: int):
        self.s0, self.s1, self.ncells = s0, s1, grid.ny * grid.nx
        self.coverage = np.zeros(s1 - s0, dtype=np.float64)
        proj = glm_projection(grid.projection, cfg.glm_ellipsoid_km)
        slots, cells, energy, area = [], [], [], []
        ext_slots, ext_cells = [], []
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
            if not sel.size:
                continue
            e0, e1 = int(first_event[sel[0]]), int(first_event[sel[-1] + 1])
            s_pairs, c_pairs = self._extent_pairs(root["glm/events"], e0, e1, keep, flash_slot, grid, proj)
            ext_slots.append(s_pairs)
            ext_cells.append(c_pairs)

        slot = _cat(slots, np.int64)
        order = np.argsort(slot, kind="stable")
        self.slot = slot[order]
        self.cell = _cat(cells, np.int64)[order]
        self.weights = {"count": None, "energy_fj": _cat(energy, np.float64)[order], "area_km2": _cat(area, np.float64)[order]}
        ext_slot = _cat(ext_slots, np.int64)
        order = np.argsort(ext_slot, kind="stable")
        self.ext_slot = ext_slot[order]
        self.ext_cell = _cat(ext_cells, np.int64)[order]

    def _extent_pairs(self, events: Any, e0: int, e1: int, keep: np.ndarray, flash_slot: np.ndarray, grid: TargetGrid,
                      proj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        (slot, cell) for every distinct (kept flash, cell touched by one of its events), from events
        [e0, e1). Orphan events (flash_index -1: their group or flash can't be found in NOAA's file, a
        few hundred a day) belong to no flash and are skipped.
        """
        keys = []
        for a in range(e0, e1, EVENT_CHUNK):
            b = min(a + EVENT_CHUNK, e1)
            fi = events["flash_index"][a:b]
            sel = (fi >= 0) & keep[np.maximum(fi, 0)]
            if not sel.any():
                continue
            x, y = latlon_to_scan_angle(events["lat"][a:b][sel], events["lon"][a:b][sel], proj)
            c = grid.cells_of(np.asarray(x), np.asarray(y))
            inside = c >= 0
            keys.append(np.unique(fi[sel][inside].astype(np.int64) * self.ncells + c[inside]))
        if not keys:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        # A flash's events all come from its file, but a file can straddle a chunk boundary
        k = np.unique(np.concatenate(keys))
        return flash_slot[k // self.ncells], k % self.ncells

    @staticmethod
    def _range(slot: np.ndarray, a: int, b: int) -> slice:
        return slice(int(np.searchsorted(slot, a, "left")), int(np.searchsorted(slot, b, "left")))

    def sums(self, a: int, b: int, stat: str = "count") -> np.ndarray:
        """Flat per-cell totals over slots [a, b): flashes (count), their energy or area, or flashes touching the cell (extent)."""
        if stat == "extent":
            r = self._range(self.ext_slot, a, b)
            return np.bincount(self.ext_cell[r], minlength=self.ncells)
        r = self._range(self.slot, a, b)
        w = self.weights[stat]
        return np.bincount(self.cell[r], weights=None if w is None else w[r], minlength=self.ncells)

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
