"""Reading the raw day stores: ABI frames by global slot (5-min slots since 1970), across day boundaries."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..common import SLOTS_PER_DAY
from ..store import output_zarr_path_for_day
from .grid import AxisMap, TargetGrid, axis_map
from .settings import FeatureConfig

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def raw_day_path(cfg: FeatureConfig, day: datetime) -> str:
    return output_zarr_path_for_day(cfg.raw.raw_output_root_dir, day)


def day_of_index(index: int) -> datetime:
    return EPOCH + timedelta(days=index)


class RawProduct:
    """One ABI product in one raw day store, with its maps onto the target grid."""

    def __init__(self, group: Any, grid: TargetGrid):
        self.present = group["present"][:]
        self.scale_factor = group["scale_factor"][:]
        self.add_offset = group["add_offset"][:]
        self.fill_value = int(group.attrs["fill_value"])
        self.value = group["value"]
        self.dqf = group["dqf"]
        self.xmap: AxisMap = axis_map((group["x_scan_angle_rad"][:] - grid.x_origin) / grid.x_step, grid.col0, grid.nx, grid.coarsen)
        self.ymap: AxisMap = axis_map((group["y_scan_angle_rad"][:] - grid.y_origin) / grid.y_step, grid.row0, grid.ny, grid.coarsen)

    def read(self, slot: int) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        if not self.present[slot]:
            return None
        ys, xs = slice(self.ymap.lo, self.ymap.hi), slice(self.xmap.lo, self.xmap.hi)
        return self.value[slot, ys, xs], self.dqf[slot, ys, xs], float(self.scale_factor[slot]), float(self.add_offset[slot])


class RawDay:
    def __init__(self, path: str, grid: TargetGrid):
        import zarr

        self.grid = grid
        self.root = zarr.open_consolidated(path, mode="r") if os.path.exists(os.path.join(path, ".zgroup")) else None
        self.complete = bool(self.root.attrs.get("complete", False)) if self.root is not None else False
        self.products: Dict[str, Optional[RawProduct]] = {}

    def product(self, key: str) -> Optional[RawProduct]:
        if key not in self.products:
            g = self.root[f"abi/{key}"] if self.root is not None and f"abi/{key}" in self.root else None
            self.products[key] = RawProduct(g, self.grid) if g is not None and g.attrs.get("has_frames") else None
        return self.products[key]


class RawTimeline:
    """Raw day stores by day index (days since 1970), opened when first needed."""

    def __init__(self, cfg: FeatureConfig, grid: TargetGrid):
        self.cfg, self.grid = cfg, grid
        self.days: Dict[int, RawDay] = {}

    def day(self, index: int) -> RawDay:
        if index not in self.days:
            self.days[index] = RawDay(raw_day_path(self.cfg, day_of_index(index)), self.grid)
        return self.days[index]

    def read(self, product: str, s: int) -> Optional[Tuple[RawProduct, np.ndarray, np.ndarray, float, float]]:
        d, local = divmod(s, SLOTS_PER_DAY)
        p = self.day(d).product(product)
        r = p.read(local) if p is not None else None
        return None if r is None else (p, *r)
