"""
MRMS radar on the feature grid, from the MRMS day stores (layout in goes_downloader/mrms.py).

Placement: radar sees the ground under a storm, ABI and GLM see it from the satellite, and a storm top
10 km up appears 6-20 km (2-5 cells) further from the sub-satellite point than the radar column under it.
GLM reports each flash at about the ground position under it, so each MRMS pixel's lat/lon goes through
the very mapping GLM's do (read on features.glm.lightning_ellipsoid_km): a flash over a radar column and
the column land on the same cell, where the satellite saw the flash, and so do the labels. With null both
stay on the ground. Placing the pixel instead where the satellite sees the point at the ellipsoid's height
above it, which differs by 4-8 km because that ellipsoid is flatter than the Earth, put the flashes
0.4-1.2 rows south of the radar cores (README, "Channels").

Time: slot s takes the latest file decoded before its issue time, (s + 1) * 5 min, if that file is less
than MAX_AGE_NS older; otherwise the slot is missing.

Values: -999 (outside radar coverage) is missing, -99 (covered, no echo) takes the channel's no_echo.
A cell aggregates the pixels placed in it, and is missing if fewer than min_valid_fraction of them are
in radar coverage.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zarr

from ..crop import latlon_to_scan_angle
from ..store import output_zarr_path_for_day
from .glm import DAY_NS, MINUTE_NS, SLOT_NS
from .grid import TargetGrid, glm_projection
from .raw import day_of_index
from .settings import ChannelSpec, FeatureConfig

# The CONUS mosaics come every 2 minutes, so a newest file older than this means a gap in MRMS's data
MAX_AGE_NS = 10 * MINUTE_NS
OUTSIDE_COVERAGE, NO_ECHO = -999.0, -99.0  # MRMS's markers in the reflectivity products
MAP_ROWS = 256  # MRMS rows mapped onto the grid at a time (~1.5 M pixels)


class PixelCells:
    """Where each pixel of an MRMS grid lands on the target grid: its pixels inside, sorted by cell."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, grid: TargetGrid, projection: Dict[str, Any]):
        cell = np.empty((len(lat), len(lon)), np.int64)
        for r in range(0, len(lat), MAP_ROWS):
            la, lo = np.meshgrid(lat[r : r + MAP_ROWS], lon, indexing="ij")
            x, y = latlon_to_scan_angle(la, lo, projection)
            cell[r : r + MAP_ROWS] = grid.cells_of(np.asarray(x), np.asarray(y))
        cell = cell.ravel()
        inside = np.flatnonzero(cell >= 0)
        self.order = inside[np.argsort(cell[inside], kind="stable")]  # flat pixel indices, cell by cell
        # Each cell with pixels, where its run of them starts in `order`, and how many it has
        self.cells, self.starts, self.total = np.unique(cell[self.order], return_index=True, return_counts=True)
        self.shape = (grid.ny, grid.nx)

    def aggregate(self, ch: ChannelSpec, phys: np.ndarray, half_step: float) -> np.ndarray:
        """One channel on the grid from a file's physical values in `order`; half_step tells the markers apart."""
        out = np.full(self.shape[0] * self.shape[1], np.nan, np.float32)
        if not len(self.starts):
            return out.reshape(self.shape)
        covered = np.abs(phys - np.float32(OUTSIDE_COVERAGE)) > half_step
        v = np.where(np.abs(phys - np.float32(NO_ECHO)) <= half_step, np.float32(ch.no_echo), phys)
        n = np.add.reduceat(covered, self.starts, dtype=np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            if ch.aggregation == "mean":
                agg = np.add.reduceat(np.where(covered, v, np.float32(0)), self.starts) / n
            else:
                agg = (np.fmax if ch.aggregation == "max" else np.fmin).reduceat(np.where(covered, v, np.float32(np.nan)), self.starts)
        agg = agg.astype(np.float32, copy=False)
        agg[(n == 0) | (n < ch.min_valid_fraction * self.total)] = np.nan
        out[self.cells] = agg
        return out.reshape(self.shape)


class MrmsFiles:
    """One product's decoded files in one day store, by time."""

    def __init__(self, group: Any):
        present = np.flatnonzero(group["present"][:])
        self.time = group["time"][:][present]
        self.index = present
        self.scale_factor = group["scale_factor"][:][present]
        self.add_offset = group["add_offset"][:][present]
        self.value = group["value"]
        self.lat, self.lon = group["lat"][:], group["lon"][:]


class MrmsDay:
    def __init__(self, path: str):
        self.root = zarr.open_consolidated(path, mode="r") if os.path.exists(os.path.join(path, "zarr.json")) else None
        self.complete = bool(self.root.attrs.get("complete", False)) if self.root is not None else False
        self.products: Dict[str, Optional[MrmsFiles]] = {}

    def product(self, key: str) -> Optional[MrmsFiles]:
        if key not in self.products:
            g = self.root[key] if self.root is not None and key in self.root else None
            self.products[key] = MrmsFiles(g) if g is not None and g.attrs.get("has_frames") else None
        return self.products[key]


class MrmsTimeline:
    """MRMS day stores by day index (days since 1970), opened when first needed, and their pixels' cells."""

    def __init__(self, cfg: FeatureConfig, grid: TargetGrid):
        self.root_dir, self.grid = cfg.mrms_root_dir, grid
        self.projection = glm_projection(grid.projection, cfg.glm_ellipsoid_km)  # GLM's, so radar lands with the flashes
        self.days: Dict[int, MrmsDay] = {}
        self.maps: Dict[Tuple[Any, ...], PixelCells] = {}

    def day(self, index: int) -> MrmsDay:
        if index not in self.days:
            self.days[index] = MrmsDay(output_zarr_path_for_day(self.root_dir, day_of_index(index)))
        return self.days[index]

    def latest(self, key: str, issue_ns: int) -> Optional[Tuple[MrmsFiles, int]]:
        """The newest file of `key` decoded before issue_ns, as (files, position), if it is recent enough."""
        for d in range((issue_ns - 1) // DAY_NS, (issue_ns - MAX_AGE_NS) // DAY_NS - 1, -1):
            files = self.day(d).product(key)
            if files is None:
                continue
            i = int(np.searchsorted(files.time, issue_ns, "left")) - 1
            if i >= 0:  # an earlier day's files are older still
                return (files, i) if files.time[i] > issue_ns - MAX_AGE_NS else None
        return None

    def cells_for(self, files: MrmsFiles) -> PixelCells:
        k = (float(files.lat[0]), float(files.lat[-1]), len(files.lat), float(files.lon[0]), float(files.lon[-1]), len(files.lon))
        if k not in self.maps:
            self.maps[k] = PixelCells(files.lat, files.lon, self.grid, self.projection)
        return self.maps[k]

    def frame(self, ch: ChannelSpec, s: int, per_slot: Dict[Any, Any]) -> np.ndarray:
        """One MRMS channel at slot s; per_slot shares the file read between the slot's channels."""
        key = ("mrms", ch.mrms)
        if key not in per_slot:
            per_slot[key] = None
            found = self.latest(ch.mrms, (s + 1) * SLOT_NS)
            if found is not None:
                files, i = found
                cells = self.cells_for(files)
                phys = np.multiply(files.value[int(files.index[i])].ravel()[cells.order], np.float32(files.scale_factor[i]), dtype=np.float32)
                phys += np.float32(files.add_offset[i])
                per_slot[key] = (cells, phys, float(files.scale_factor[i]) / 2)
        if per_slot[key] is None:
            return np.full((self.grid.ny, self.grid.nx), np.nan, dtype=np.float32)
        cells, phys, half_step = per_slot[key]
        return cells.aggregate(ch, phys, half_step)
