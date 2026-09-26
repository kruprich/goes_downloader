"""
build_features.py

Raw GOES day stores -> CNN input features + lightning labels on one grid, one Zarr store per UTC day
===================================================================================================

Reads the raw stores written by download_raw.py and, for every 5-min ABI slot, puts the configured
channels on one regular grid, bins GLM flashes onto the same grid and derives lightning labels for the
configured forecast windows. Everything is set by the "features" section of config.json (see README.md).

Grid: the ABI fixed grid (scan angles, the projection every ABI product is delivered on), coarsened by
an integer factor from the finest product used. Each cell is a block of coarsen x coarsen reference
pixels counted from the uncropped CONUS sector's first pixel, so cells nest exactly in the 2 km and
10 km ABI grids (a 10 km pixel is 5x5 2 km pixels). Products at least as fine as the cell are
aggregated over the source pixels whose centres fall in it (mean/min/max/std/fraction); coarser
products take the source pixel containing the cell centre. GLM flashes go to the cell containing their
view angle, recovered with the lightning ellipsoid GLM navigates to, so they line up with the cloud tops
ABI sees from the same satellite (the ground ellipsoid would move them by ~2 km over CONUS).

Time: slot t holds the ABI scan starting in [t, t+5 min). The issue time is the end of the slot
(t + 5 min): every input ends there and every label window starts there.

Layout of <features.output_root_dir>/YYYY/YYYYDDD.zarr
  time[time]                           datetime64 slot start, every time_step_minutes
  channel[channel], label[label]       names in config order (channels with "output": false excluded)
  features[time, channel, y, x]        float16 (or float32); NaN = missing. Chunked (1, all channels, chunk_yx)
  labels[time, label, y, x]            uint8 1 = lightning in the window, 0 = none, 255 = unknown (GLM covered
                                       less than glm.min_coverage of the window)
  flash_count[time, y, x]              uint16 flashes (quality-filtered) whose first event is in
                                       [issue - time_step, issue), so other labels can be derived later
  glm_coverage[time]                   float32 fraction of the flash_count window covered by GLM files
  channel_valid_fraction[time, channel] float32 fraction of cells that are not NaN
  x[x], y[y]                           float64 cell-centre scan angles (rad) on the ABI fixed grid
  lat[y, x], lon[y, x]                 float32 cell centres (deg)

Root attrs:
  complete = true only if the day's raw store and every neighbouring raw day that was read (lookback for
  lags and gap filling, lookahead for labels) are complete. A neighbour outside the raw config's date range
  never will be, so it counts as unavailable instead: its slots are NaN / unknown. Incomplete days are
  rebuilt from scratch on the next run.
  channel_stats / label_stats: per-day sums for normalisation and class balance (combine with --stats).
  grid, channels, labels, glm, time_step_minutes: the settings the store was built with.

Usage:
  python build_features.py --config config.json                  # build every day in the range
  python build_features.py --config config.json --start 2025-10-16 --end 2026-05-31 --stats
                                                                  # normalisation stats over those days
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import zarr

import download_raw as dr


SLOT_NS = dr.SLOT_SECONDS * 1_000_000_000
SLOT_MINUTES = dr.SLOT_SECONDS // 60
SLOTS_PER_DAY = dr.SLOTS_PER_DAY
GLM_WINDOWS_PER_SLOT = dr.SLOT_SECONDS // dr.GLM_WINDOW_SECONDS
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LABEL_UNKNOWN = 255
FLOAT16_MAX = float(np.finfo(np.float16).max)

AGGREGATIONS = ("mean", "min", "max", "std", "fraction")
GLM_STATS = ("count", "energy_fj", "area_km2")


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: str  # "product", "glm" or "expr"
    output: bool = True
    description: str = ""
    # kind == "product"
    product: Optional[str] = None
    aggregation: str = "mean"
    values: Tuple[float, ...] = ()
    valid_dqf: Optional[Tuple[int, ...]] = (0,)
    min_valid_fraction: float = 0.5
    fill_gaps_slots: int = 0
    # kind == "glm"
    glm_stat: Optional[str] = None
    # kind == "expr"
    expr: Optional[str] = None


@dataclass(frozen=True)
class LabelSpec:
    name: str
    start_slots: int
    end_slots: int
    min_flashes: int
    radius_cells: float
    description: str = ""


@dataclass(frozen=True)
class FeatureConfig:
    raw: dr.RawConfig
    output_root_dir: str
    start_date_utc: str
    end_date_utc: str
    overwrite_existing: bool
    max_workers: int
    resolution_km: float
    bounds: Optional[dr.LatLonBounds]
    reference_product: Optional[str]
    time_step_slots: int
    dtype: str
    chunk_yx: Tuple[int, int]
    glm_quality_flags: Optional[Tuple[int, ...]]
    glm_ellipsoid_km: Optional[Tuple[float, float]]  # (equatorial, polar) heights above the ground ellipsoid
    glm_min_coverage: float
    channels: Tuple[ChannelSpec, ...]
    labels: Tuple[LabelSpec, ...]

    @property
    def uses_glm(self) -> bool:
        return self.raw.glm_product is not None


CHANNEL_KEYS = {
    "product": {"product", "aggregation", "values", "valid_dqf", "min_valid_fraction", "fill_gaps_minutes"},
    "glm": {"glm"},
    "expr": {"expr"},
}


def _minutes_to_slots(v: Any, what: str, positive: bool = False) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v % SLOT_MINUTES or (positive and v == 0):
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"{what} must be a {kind} multiple of {SLOT_MINUTES} minutes, got {v!r}")
    return int(v) // SLOT_MINUTES


def _fraction(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        raise ValueError(f"{what} must be a number between 0 and 1, got {v!r}")
    return float(v)


def _int_list(v: Any, what: str) -> Tuple[int, ...]:
    if not isinstance(v, list) or not v or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in v):
        raise ValueError(f"{what} must be a non-empty list of non-negative integers or null, got {v!r}")
    return tuple(v)


def _parse_channel(i: int, c: Any, abi_keys: List[str], has_glm: bool) -> ChannelSpec:
    where = f"features.channels[{i}]"
    if not isinstance(c, dict):
        raise ValueError(f"{where} must be an object")
    name = c.get("name")
    if not isinstance(name, str) or not name.isidentifier() or name in FUNCTION_NAMES:
        raise ValueError(f"{where}.name must be an identifier (letters, digits, _) that is not a function name, got {name!r}")
    where = f"channel {name!r}"
    kinds = [k for k in CHANNEL_KEYS if k in c]
    if len(kinds) != 1:
        raise ValueError(f"{where} needs exactly one of \"product\", \"glm\" or \"expr\"")
    kind = kinds[0]
    unknown = set(c) - {"name", "output", "description"} - CHANNEL_KEYS[kind]
    if unknown:
        raise ValueError(f"{where}: unknown or misplaced keys {sorted(unknown)} for a {kind} channel")
    common = dict(name=name, kind=kind, output=bool(c.get("output", True)), description=str(c.get("description", "")))

    if kind == "product":
        product = c["product"]
        if product not in abi_keys:
            raise ValueError(f"{where}: product {product!r} is not in the download list (downloaded ABI products: {abi_keys})")
        aggregation = c.get("aggregation", "mean")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"{where}: aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        values = c.get("values")
        if (aggregation == "fraction") != (values is not None):
            raise ValueError(f"{where}: \"values\" (the categories to count) goes with aggregation \"fraction\" only")
        if values is not None and (not isinstance(values, list) or not values or not all(isinstance(v, (int, float)) for v in values)):
            raise ValueError(f"{where}: values must be a non-empty list of numbers")
        valid_dqf = c.get("valid_dqf", [0])
        return ChannelSpec(
            **common,
            product=product,
            aggregation=aggregation,
            values=tuple(float(v) for v in values or ()),
            valid_dqf=None if valid_dqf is None else _int_list(valid_dqf, f"{where}: valid_dqf"),
            min_valid_fraction=_fraction(c.get("min_valid_fraction", 0.5), f"{where}: min_valid_fraction"),
            fill_gaps_slots=_minutes_to_slots(c.get("fill_gaps_minutes", 0), f"{where}: fill_gaps_minutes"),
        )
    if kind == "glm":
        if not has_glm:
            raise ValueError(f"{where}: GLM channels need \"glm\" in the download list")
        if c["glm"] not in GLM_STATS:
            raise ValueError(f"{where}: glm must be one of {GLM_STATS}, got {c['glm']!r}")
        return ChannelSpec(**common, glm_stat=c["glm"])
    if not isinstance(c["expr"], str):
        raise ValueError(f"{where}: expr must be a string")
    parse_expression(c["expr"], where)
    return ChannelSpec(**common, expr=c["expr"])


def _parse_label(i: int, l: Any) -> LabelSpec:
    where = f"features.labels[{i}]"
    if not isinstance(l, dict):
        raise ValueError(f"{where} must be an object")
    unknown = set(l) - {"name", "description", "start_minutes", "end_minutes", "min_flashes", "radius_cells"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    name = l.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}.name must be a non-empty string")
    where = f"label {name!r}"
    start = _minutes_to_slots(l.get("start_minutes", 0), f"{where}: start_minutes")
    end = _minutes_to_slots(l.get("end_minutes"), f"{where}: end_minutes", positive=True)
    if end <= start:
        raise ValueError(f"{where}: end_minutes must be greater than start_minutes")
    min_flashes = l.get("min_flashes", 1)
    if not isinstance(min_flashes, int) or isinstance(min_flashes, bool) or min_flashes < 1:
        raise ValueError(f"{where}: min_flashes must be a positive integer")
    radius = l.get("radius_cells", 0)
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius < 0:
        raise ValueError(f"{where}: radius_cells must be a non-negative number")
    return LabelSpec(name, start, end, min_flashes, float(radius), str(l.get("description", "")))


def load_feature_config(config_path: str) -> FeatureConfig:
    """Reads the "features" section of the shared config.json; the raw settings come from download_raw."""
    raw = dr.load_config(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        section = json.load(f).get("features")
    if not isinstance(section, dict):
        raise ValueError(f"{config_path} has no \"features\" object")

    abi_keys = [p.key for p in raw.abi_products]
    has_glm = raw.glm_product is not None
    channels_raw = dr._require(section, "channels", list)
    channels = tuple(_parse_channel(i, c, abi_keys, has_glm) for i, c in enumerate(channels_raw))
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise ValueError(f"channel names must be unique, got {names}")
    if not any(c.output for c in channels):
        raise ValueError("at least one channel must have output true")
    if not any(c.kind == "product" for c in channels):
        raise ValueError("at least one channel must read an ABI product (it defines the grid)")

    labels = tuple(_parse_label(i, l) for i, l in enumerate(section.get("labels", [])))
    if labels and not has_glm:
        raise ValueError("labels need \"glm\" in the download list")
    if len({l.name for l in labels}) != len(labels):
        raise ValueError("label names must be unique")

    grid = section.get("grid", {})
    resolution_km = grid.get("resolution_km", 4)
    if isinstance(resolution_km, bool) or not isinstance(resolution_km, (int, float)) or resolution_km <= 0:
        raise ValueError("features.grid.resolution_km must be a positive number")
    reference_product = grid.get("reference_product")
    if reference_product is not None and reference_product not in abi_keys:
        raise ValueError(f"features.grid.reference_product {reference_product!r} is not a downloaded ABI product")
    # "bounds" absent -> the download bounds; null -> the whole area the raw stores hold
    bounds = raw.bounds
    if "bounds" in grid:
        b = grid["bounds"]
        bounds = None if b is None else dr.LatLonBounds(**{k: float(dr._require(b, k, (int, float))) for k in ("lat_min", "lat_max", "lon_min", "lon_max")})

    step = _minutes_to_slots(section.get("time_step_minutes", SLOT_MINUTES), "features.time_step_minutes", positive=True)
    if SLOTS_PER_DAY % step:
        raise ValueError("features.time_step_minutes must divide a day (5, 10, 15, 20, 30, 60, ...)")

    dtype = section.get("dtype", "float16")
    if dtype not in ("float16", "float32"):
        raise ValueError("features.dtype must be \"float16\" or \"float32\"")
    chunk_yx = section.get("chunk_yx", [256, 256])
    if not (isinstance(chunk_yx, list) and len(chunk_yx) == 2 and all(isinstance(v, int) and v > 0 for v in chunk_yx)):
        raise ValueError("features.chunk_yx must be two positive integers")

    glm = section.get("glm", {})
    flags = glm.get("quality_flags")
    ellipsoid = glm.get("lightning_ellipsoid_km", {"equatorial": 14.0, "polar": 6.0})
    if ellipsoid is not None and not (isinstance(ellipsoid, dict) and set(ellipsoid) == {"equatorial", "polar"}):
        raise ValueError("features.glm.lightning_ellipsoid_km must be {\"equatorial\": km, \"polar\": km} or null")

    start = section.get("start_date_utc") or raw.start_date_utc
    end = section.get("end_date_utc") or raw.end_date_utc
    datetime.strptime(start, "%Y-%m-%d")
    datetime.strptime(end, "%Y-%m-%d")

    cfg = FeatureConfig(
        raw=raw,
        output_root_dir=str(section.get("output_root_dir", "./goes_features")),
        start_date_utc=start,
        end_date_utc=end,
        overwrite_existing=bool(section.get("overwrite_existing", False)),
        max_workers=max(1, int(section.get("max_workers", 4))),
        resolution_km=float(resolution_km),
        bounds=bounds,
        reference_product=reference_product,
        time_step_slots=step,
        dtype=dtype,
        chunk_yx=(chunk_yx[0], chunk_yx[1]),
        glm_quality_flags=None if flags is None else _int_list(flags, "features.glm.quality_flags"),
        glm_ellipsoid_km=None if ellipsoid is None else (float(ellipsoid["equatorial"]), float(ellipsoid["polar"])),
        glm_min_coverage=_fraction(glm.get("min_coverage", 0.95), "features.glm.min_coverage"),
        channels=channels,
        labels=labels,
    )
    make_plan(cfg)  # checks expression references and cycles
    return cfg


# =============================================================================
# Expressions (derived channels)
# =============================================================================

# f(x, minutes): x at an earlier slot, or reduced over the slots of the last `minutes` (current one included)
TIME_FUNCTIONS = ("lag", "sum_past", "mean_past", "min_past", "max_past")
ELEMENTWISE_FUNCTIONS = {
    "abs": (1, np.abs),
    "sqrt": (1, np.sqrt),
    "log1p": (1, np.log1p),
    "exp": (1, np.exp),
    "minimum": (2, np.minimum),
    "maximum": (2, np.maximum),
}
FUNCTION_NAMES = set(TIME_FUNCTIONS) | set(ELEMENTWISE_FUNCTIONS) | {"clip"}
BINARY_OPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
PAST_REDUCERS = {"sum_past": np.sum, "mean_past": np.mean, "min_past": np.min, "max_past": np.max}


def _number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _number(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    return None


def _time_function_slots(node: ast.Call) -> int:
    return int(_number(node.args[1])) // SLOT_MINUTES


def parse_expression(text: str, where: str) -> ast.Expression:
    """Parses a channel expression, allowing only arithmetic on channel names, numbers and FUNCTION_NAMES."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"{where}: cannot parse {text!r}: {e.msg}") from None

    def check(node: ast.AST) -> None:
        if _number(node) is not None or isinstance(node, ast.Name):
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return check(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
            check(node.left)
            check(node.right)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            f, args = node.func.id, node.args
            if f in TIME_FUNCTIONS:
                minutes = _number(args[1]) if len(args) == 2 else None
                if minutes is None or minutes <= 0 or minutes % SLOT_MINUTES:
                    raise ValueError(f"{where}: {f}(x, minutes) needs minutes as a positive multiple of {SLOT_MINUTES}")
                return check(args[0])
            if f == "clip":
                if len(args) != 3 or _number(args[1]) is None or _number(args[2]) is None:
                    raise ValueError(f"{where}: clip(x, low, high) needs numbers for low and high")
                return check(args[0])
            if f in ELEMENTWISE_FUNCTIONS:
                if len(args) != ELEMENTWISE_FUNCTIONS[f][0]:
                    raise ValueError(f"{where}: {f} takes {ELEMENTWISE_FUNCTIONS[f][0]} argument(s)")
                for a in args:
                    check(a)
                return
            raise ValueError(f"{where}: unknown function {f!r}; available: {sorted(FUNCTION_NAMES)}")
        raise ValueError(f"{where}: unsupported syntax {ast.unparse(node)!r} (use + - * / **, numbers, channel names and functions)")

    check(tree.body)
    return tree


def expression_refs(node: ast.AST, offset: int = 0) -> Iterator[Tuple[str, int]]:
    """(channel name, slots back) for every channel the expression reads."""
    if isinstance(node, ast.Expression):
        yield from expression_refs(node.body, offset)
    elif isinstance(node, ast.Name):
        yield node.id, offset
    elif isinstance(node, ast.Call):
        f = node.func.id
        if f in TIME_FUNCTIONS:
            n = _time_function_slots(node)
            yield from expression_refs(node.args[0], offset + (n if f == "lag" else n - 1))
        else:
            for a in node.args:
                yield from expression_refs(a, offset)
    else:
        for child in ast.iter_child_nodes(node):
            yield from expression_refs(child, offset)


@dataclass
class Plan:
    order: List[ChannelSpec]  # every channel after the channels it reads
    exprs: Dict[str, ast.Expression]
    depth: Dict[str, int]  # oldest slot (back from the current one) at which each channel is read
    lookback: int  # slots computed before the day so the first output slot has full history
    lookahead: int  # GLM slots after the last issue time that the labels need


def make_plan(cfg: FeatureConfig) -> Plan:
    by_name = {c.name: c for c in cfg.channels}
    exprs = {c.name: parse_expression(c.expr, f"channel {c.name!r}") for c in cfg.channels if c.kind == "expr"}
    refs = {name: list(expression_refs(tree)) for name, tree in exprs.items()}
    for name, rs in refs.items():
        missing = sorted({r for r, _ in rs if r not in by_name})
        if missing:
            raise ValueError(f"channel {name!r} refers to unknown channels {missing}")

    order: List[ChannelSpec] = []
    state: Dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(name: str, path: List[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f"channel expressions form a cycle: {' -> '.join(path + [name])}")
        state[name] = 1
        for r, _ in refs.get(name, []):
            visit(r, path + [name])
        state[name] = 2
        order.append(by_name[name])

    for c in cfg.channels:
        visit(c.name, [])

    depth = {c.name: 0 for c in cfg.channels}
    for rs in refs.values():
        for r, off in rs:
            depth[r] = max(depth[r], off)

    need: Dict[str, int] = {}
    for c in order:  # dependencies first, so need[r] is known
        if c.kind == "product":
            need[c.name] = c.fill_gaps_slots
        elif c.kind == "glm":
            need[c.name] = 0
        else:
            need[c.name] = max((off + need[r] for r, off in refs[c.name]), default=0)

    lookahead = max((l.end_slots for l in cfg.labels), default=0)
    return Plan(order, exprs, depth, max(need.values()), lookahead)


class Evaluator:
    """Evaluates a parsed expression at slot s from the channel frames cached so far."""

    def __init__(self, cache: Dict[str, Dict[int, np.ndarray]], shape: Tuple[int, int]):
        self.cache = cache
        self.missing = np.full(shape, np.nan, dtype=np.float32)
        self.missing.setflags(write=False)

    def __call__(self, node: ast.AST, s: int) -> Any:
        v = _number(node)
        if v is not None:
            return np.float32(v)
        if isinstance(node, ast.Expression):
            return self(node.body, s)
        if isinstance(node, ast.Name):
            # Slots before the computed range only occur while warming up; those outputs are never written
            return self.cache[node.id].get(s, self.missing)
        if isinstance(node, ast.UnaryOp):
            x = self(node.operand, s)
            return -x if isinstance(node.op, ast.USub) else x
        if isinstance(node, ast.BinOp):
            return BINARY_OPS[type(node.op)](self(node.left, s), self(node.right, s))
        f, args = node.func.id, node.args
        if f == "lag":
            return self(args[0], s - _time_function_slots(node))
        if f in PAST_REDUCERS:
            frames = [self(args[0], s - i) for i in range(_time_function_slots(node))]
            return PAST_REDUCERS[f](np.stack(np.broadcast_arrays(*frames)), axis=0)  # NaN in any slot -> NaN
        if f == "clip":
            return np.clip(self(args[0], s), _number(args[1]), _number(args[2]))
        return ELEMENTWISE_FUNCTIONS[f][1](*(self(a, s) for a in args))


# =============================================================================
# Grid
# =============================================================================

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


def _geos_crs(projection: Dict[str, Any]) -> Any:
    from pyproj import CRS

    return CRS.from_proj4(
        f"+proj=geos +lon_0={projection['longitude_of_projection_origin']} +h={projection['perspective_point_height']} "
        f"+a={projection['semi_major_axis']} +b={projection['semi_minor_axis']} "
        f"+sweep={projection['sweep_angle_axis']} +units=m +no_defs"
    )


def grid_latlon(grid: TargetGrid) -> Tuple[np.ndarray, np.ndarray]:
    from pyproj import CRS, Transformer

    h = float(grid.projection["perspective_point_height"])
    tfm = Transformer.from_crs(_geos_crs(grid.projection), CRS.from_epsg(4326), always_xy=True)
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


def _nominal_km(step_rad: float, projection: Dict[str, Any]) -> float:
    """Sub-satellite pixel size, rounded to 0.5 km (56 urad -> 2 km, 280 urad -> 10 km)."""
    return round(step_rad * float(projection["perspective_point_height"]) / 500.0) / 2.0


def raw_day_path(cfg: FeatureConfig, day: datetime) -> str:
    return dr.output_zarr_path_for_day(cfg.raw.raw_output_root_dir, day)


def make_grid(cfg: FeatureConfig) -> TargetGrid:
    """Builds the target grid from the first raw day in the range that holds every product used."""
    products = list(dict.fromkeys(c.product for c in cfg.channels if c.kind == "product"))
    if cfg.reference_product is not None:
        products = [cfg.reference_product]

    groups: Dict[str, Any] = {}
    for day in dr.iter_days_utc(cfg.start_date_utc, cfg.end_date_utc):
        path = raw_day_path(cfg, day)
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
        x_min, x_max, y_min, y_max = dr.scan_angle_rect(cfg.bounds, projection)
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
# Regridding
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


def aggregate(ch: ChannelSpec, src: SourceFrame) -> np.ndarray:
    """One product channel on the grid."""
    n = src.n
    with np.errstate(invalid="ignore", divide="ignore"):
        if ch.aggregation in ("min", "max"):
            out = src.reduce(np.where(src.valid, src.phys, np.float32(np.nan)), np.fmin if ch.aggregation == "min" else np.fmax)
        elif ch.aggregation == "fraction":
            out = src.reduce((src.valid & np.isin(src.phys, ch.values)).astype(np.float32), np.add) / n
        elif ch.aggregation == "mean":
            out = src.reduce(src.zeroed, np.add) / n
        else:
            x = src.zeroed.astype(np.float64)
            mean = src.reduce(x, np.add) / n
            out = np.sqrt(np.maximum(src.reduce(x * x, np.add) / n - mean * mean, 0.0))
        out = out.astype(np.float32, copy=False)
        out[(src.total == 0) | (n == 0) | (n < ch.min_valid_fraction * src.total)] = np.nan
    return out


class RawProduct:
    """One ABI product in one raw day store, with its maps onto the target grid."""

    def __init__(self, group: Any, grid: TargetGrid):
        self.present = group["present"][:]
        self.scale_factor = group["scale_factor"][:]
        self.add_offset = group["add_offset"][:]
        self.fill_value = int(group.attrs["fill_value"])
        self.value = group["value"]
        self.dqf = group["dqf"]
        self.xmap = axis_map((group["x_scan_angle_rad"][:] - grid.x_origin) / grid.x_step, grid.col0, grid.nx, grid.coarsen)
        self.ymap = axis_map((group["y_scan_angle_rad"][:] - grid.y_origin) / grid.y_step, grid.row0, grid.ny, grid.coarsen)

    def read(self, slot: int) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        if not self.present[slot]:
            return None
        ys, xs = slice(self.ymap.lo, self.ymap.hi), slice(self.xmap.lo, self.xmap.hi)
        return self.value[slot, ys, xs], self.dqf[slot, ys, xs], float(self.scale_factor[slot]), float(self.add_offset[slot])


class RawDay:
    def __init__(self, path: str, grid: TargetGrid):
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
            self.days[index] = RawDay(raw_day_path(self.cfg, EPOCH + timedelta(days=index)), self.grid)
        return self.days[index]

    def read(self, product: str, s: int) -> Optional[Tuple[RawProduct, np.ndarray, np.ndarray, float, float]]:
        d, local = divmod(s, SLOTS_PER_DAY)
        p = self.day(d).product(product)
        r = p.read(local) if p is not None else None
        return None if r is None else (p, *r)


# =============================================================================
# GLM
# =============================================================================

class GlmTimeline:
    """Quality-filtered flashes on the grid and GLM coverage per slot, for global slots [s0, s1)."""

    def __init__(self, cfg: FeatureConfig, grid: TargetGrid, raw: RawTimeline, s0: int, s1: int):
        self.s0, self.s1, self.ncells = s0, s1, grid.ny * grid.nx
        self.coverage = np.zeros(s1 - s0, dtype=np.float64)
        proj = glm_projection(grid.projection, cfg.glm_ellipsoid_km)
        slots, cells, energy, area = [], [], [], []
        for d in range(s0 // SLOTS_PER_DAY, (s1 - 1) // SLOTS_PER_DAY + 1):
            root = raw.day(d).root
            if root is None or "glm/coverage" not in root:
                continue
            per_slot = root["glm/coverage/covered"][:].reshape(SLOTS_PER_DAY, GLM_WINDOWS_PER_SLOT).mean(axis=1)
            a, b = max(s0, d * SLOTS_PER_DAY), min(s1, (d + 1) * SLOTS_PER_DAY)
            self.coverage[a - s0 : b - s0] = per_slot[a - d * SLOTS_PER_DAY : b - d * SLOTS_PER_DAY]

            f = root["glm/flashes"]
            t = f["time_first_ns"][:]
            keep = (t != dr.NAT_NS) & (t >= s0 * SLOT_NS) & (t < s1 * SLOT_NS)
            if cfg.glm_quality_flags is not None:
                keep &= np.isin(f["quality_flag"][:], cfg.glm_quality_flags)
            lat, lon = f["lat"][:][keep], f["lon"][:][keep]
            x, y = dr.latlon_to_scan_angle(lat, lon, proj)
            c = grid.cells_of(np.asarray(x), np.asarray(y))
            inside = c >= 0
            slots.append(t[keep][inside] // SLOT_NS)
            cells.append(c[inside])
            energy.append(f["energy_j"][:][keep][inside].astype(np.float64) * 1e15)
            area.append(f["area_m2"][:][keep][inside].astype(np.float64) / 1e6)

        def cat(parts: List[np.ndarray], dtype: Any) -> np.ndarray:
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)

        slot = cat(slots, np.int64)
        order = np.argsort(slot, kind="stable")
        self.slot = slot[order]
        self.cell = cat(cells, np.int64)[order]
        self.weights = {"count": None, "energy_fj": cat(energy, np.float64)[order], "area_km2": cat(area, np.float64)[order]}

    def _range(self, a: int, b: int) -> slice:
        return slice(int(np.searchsorted(self.slot, a, "left")), int(np.searchsorted(self.slot, b, "left")))

    def sums(self, a: int, b: int, stat: str = "count") -> np.ndarray:
        """Flat per-cell totals over slots [a, b)."""
        r = self._range(a, b)
        w = self.weights[stat]
        return np.bincount(self.cell[r], weights=None if w is None else w[r], minlength=self.ncells)

    def coverage_mean(self, a: int, b: int) -> float:
        a, b = max(a, self.s0), min(b, self.s1)
        return float(self.coverage[a - self.s0 : b - self.s0].sum()) / max(1, b - a) if b > a else 0.0

    def n_flashes(self, a: int, b: int) -> int:
        r = self._range(a, b)
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


# =============================================================================
# Per-day build
# =============================================================================

def features_day_path(cfg: FeatureConfig, day: datetime) -> str:
    return dr.output_zarr_path_for_day(cfg.output_root_dir, day)


class ChannelStats:
    """Sums over every stored value of one channel, combined across days by --stats."""

    def __init__(self) -> None:
        self.count, self.sum, self.sum_sq = 0, 0.0, 0.0
        self.min, self.max = math.inf, -math.inf

    def add(self, v: np.ndarray) -> None:
        v = v[np.isfinite(v)].astype(np.float64)
        if v.size:
            self.count += int(v.size)
            self.sum += float(v.sum())
            self.sum_sq += float(np.dot(v, v))
            self.min, self.max = min(self.min, float(v.min())), max(self.max, float(v.max()))

    def as_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "sum": self.sum, "sum_sq": self.sum_sq,
                "min": self.min if self.count else None, "max": self.max if self.count else None}


def channel_attrs(c: ChannelSpec) -> Dict[str, Any]:
    """A channel's definition in config.json terms."""
    if c.kind == "product":
        d: Dict[str, Any] = {"name": c.name, "product": c.product, "aggregation": c.aggregation}
        if c.aggregation == "fraction":
            d["values"] = list(c.values)
        d.update(valid_dqf=None if c.valid_dqf is None else list(c.valid_dqf), min_valid_fraction=c.min_valid_fraction,
                 fill_gaps_minutes=c.fill_gaps_slots * SLOT_MINUTES)
    elif c.kind == "glm":
        d = {"name": c.name, "glm": c.glm_stat}
    else:
        d = {"name": c.name, "expr": c.expr}
    return {**d, "description": c.description} if c.description else d


def label_attrs(l: LabelSpec) -> Dict[str, Any]:
    d = {"name": l.name, "start_minutes": l.start_slots * SLOT_MINUTES, "end_minutes": l.end_slots * SLOT_MINUTES,
         "min_flashes": l.min_flashes, "radius_cells": l.radius_cells}
    return {**d, "description": l.description} if l.description else d


def build_day(cfg: FeatureConfig, grid: TargetGrid, day_iso: str) -> Dict[str, Any]:
    """Builds one day's feature store. Runs in a worker process."""
    t_start = time.monotonic()
    day = datetime.strptime(day_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_path = features_day_path(cfg, day)
    raw = RawTimeline(cfg, grid)
    s_day = (day - EPOCH).days * SLOTS_PER_DAY
    if raw.day(s_day // SLOTS_PER_DAY).root is None:
        return {"day": day_iso, "status": "no raw store", "complete": False}

    plan = make_plan(cfg)
    step = cfg.time_step_slots
    shape = (grid.ny, grid.nx)
    s0 = s_day - plan.lookback
    out_slots = list(range(s_day, s_day + SLOTS_PER_DAY, step))
    outputs = [c for c in cfg.channels if c.output]

    glm: Optional[GlmTimeline] = None
    if cfg.uses_glm:
        # Inputs from the lookback, flash_count windows ending at each issue time, label windows after it
        glm_s0 = min(s0, s_day + 1 - step)
        glm_s1 = out_slots[-1] + 1 + plan.lookahead
        glm = GlmTimeline(cfg, grid, raw, glm_s0, max(glm_s1, s_day + SLOTS_PER_DAY))

    # --- store (built at *.tmp and moved into place when done)
    tmp_path = out_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(zarr.DirectoryStore(tmp_path), mode="w")
    n_t, cy, cx = len(out_slots), min(cfg.chunk_yx[0], grid.ny), min(cfg.chunk_yx[1], grid.nx)

    def frames(name: str, lead: List[int], dtype: Any, fill: Any, dims: List[str]) -> Any:
        a = root.create_dataset(name, shape=(n_t, *lead, *shape), chunks=(1, *lead, cy, cx), dtype=dtype,
                                fill_value=fill, compressor=dr.COMPRESSOR, overwrite=True)
        a.attrs.update({"_ARRAY_DIMENSIONS": dims, "coordinates": "lat lon"})
        return a

    features = frames("features", [len(outputs)], cfg.dtype, np.nan, ["time", "channel", "y", "x"])
    labels_arr = frames("labels", [len(cfg.labels)], np.uint8, LABEL_UNKNOWN, ["time", "label", "y", "x"]) if cfg.labels else None
    # Every slot is written, so no fill value: zarr's default of 0 would make xarray read 0 flashes as missing
    flash_arr = frames("flash_count", [], np.uint16, None, ["time", "y", "x"]) if glm is not None else None

    # --- evaluate every channel slot by slot, from the lookback to the end of the day
    cache: Dict[str, Dict[int, np.ndarray]] = {c.name: {} for c in cfg.channels}
    evaluate = Evaluator(cache, shape)
    gap_state: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    dqf_luts: Dict[Tuple[Any, Any], np.ndarray] = {}
    stats = {c.name: ChannelStats() for c in outputs}
    label_counts = {l.name: [0, 0] for l in cfg.labels}  # [n_known, n_positive]
    label_offsets = {l.name: disk_offsets(l.radius_cells) for l in cfg.labels}
    valid_fraction = np.zeros((n_t, len(outputs)), dtype=np.float32)
    glm_coverage = np.zeros(n_t, dtype=np.float32)
    n_overflow = {c.name: 0 for c in outputs}
    n_flashes = 0

    def product_frame(ch: ChannelSpec, s: int, per_slot: Dict[Any, Any]) -> np.ndarray:
        if ch.product not in per_slot:
            per_slot[ch.product] = raw.read(ch.product, s)
        r = per_slot[ch.product]
        if r is None:
            return np.full(shape, np.nan, dtype=np.float32)
        product, value, dqf, sf, ao = r
        key = (ch.product, ch.valid_dqf)
        if key not in per_slot:
            valid = value != product.fill_value
            if ch.valid_dqf is not None:
                lut_key = (dqf.dtype.str, ch.valid_dqf)
                if lut_key not in dqf_luts:
                    lut = np.zeros(int(np.iinfo(dqf.dtype).max) + 1, dtype=bool)
                    lut[[v for v in ch.valid_dqf if v < lut.size]] = True
                    dqf_luts[lut_key] = lut
                valid &= dqf_luts[lut_key][dqf]
            phys = np.multiply(value, np.float32(sf), dtype=np.float32)
            phys += np.float32(ao)
            per_slot[key] = SourceFrame(valid, phys, product.ymap, product.xmap)
        return aggregate(ch, per_slot[key])

    def fill_gaps(ch: ChannelSpec, s: int, frame: np.ndarray) -> np.ndarray:
        """Carries each cell's last valid value forward for up to fill_gaps_slots slots."""
        if ch.name not in gap_state:
            gap_state[ch.name] = (np.full(shape, np.nan, np.float32), np.full(shape, np.iinfo(np.int64).min // 2, np.int64))
        last, last_slot = gap_state[ch.name]
        ok = np.isfinite(frame)
        last[ok] = frame[ok]
        last_slot[ok] = s
        return np.where(s - last_slot <= ch.fill_gaps_slots, last, np.float32(np.nan))

    for s in range(s0, s_day + SLOTS_PER_DAY):
        for name, depth in plan.depth.items():
            cache[name].pop(s - 1 - depth, None)  # no longer reachable from slot s
        per_slot: Dict[Any, Any] = {}
        for ch in plan.order:
            if ch.kind == "product":
                frame = product_frame(ch, s, per_slot)
                if ch.fill_gaps_slots:
                    frame = fill_gaps(ch, s, frame)
            elif ch.kind == "glm":
                if glm.coverage_mean(s, s + 1) < cfg.glm_min_coverage:
                    frame = np.full(shape, np.nan, dtype=np.float32)
                else:
                    frame = glm.sums(s, s + 1, ch.glm_stat).reshape(shape).astype(np.float32)
            else:
                with np.errstate(all="ignore"):
                    # A copy: the result can be another channel's cached frame or the read-only NaN frame
                    frame = np.array(np.broadcast_to(evaluate(plan.exprs[ch.name], s), shape), dtype=np.float32)
                frame[np.isinf(frame)] = np.nan
            cache[ch.name][s] = frame

        if s < s_day or (s - s_day) % step:
            continue
        k = (s - s_day) // step
        stack = np.stack([cache[c.name][s] for c in outputs])
        if cfg.dtype == "float16":
            too_big = np.abs(stack) > FLOAT16_MAX
            if too_big.any():
                for i, c in enumerate(outputs):
                    n_overflow[c.name] += int(too_big[i].sum())
                stack[too_big] = np.nan
        stored = stack.astype(cfg.dtype)
        features[k] = stored
        for i, c in enumerate(outputs):
            stats[c.name].add(stored[i])
            valid_fraction[k, i] = float(np.isfinite(stored[i]).mean())

        if glm is not None:
            a = s + 1 - step
            counts = glm.sums(a, s + 1).reshape(shape)
            flash_arr[k] = np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16)
            glm_coverage[k] = glm.coverage_mean(a, s + 1)
            n_flashes += glm.n_flashes(a, s + 1)
        if labels_arr is not None:
            out = np.empty((len(cfg.labels), *shape), dtype=np.uint8)
            for i, l in enumerate(cfg.labels):
                a, b = s + 1 + l.start_slots, s + 1 + l.end_slots
                if glm.coverage_mean(a, b) < cfg.glm_min_coverage:
                    out[i] = LABEL_UNKNOWN
                    continue
                hits = neighbourhood_sum(glm.sums(a, b).reshape(shape), label_offsets[l.name]) >= l.min_flashes
                out[i] = hits
                label_counts[l.name][0] += hits.size
                label_counts[l.name][1] += int(hits.sum())
            labels_arr[k] = out

    # --- coordinates, per-time arrays, attrs
    slot_ns = np.array(out_slots, dtype=np.int64) * SLOT_NS
    dr._write_array(root, "time", slot_ns, ["time"], **dr.TIME_ATTRS, long_name="ABI slot start; issue time is 5 minutes later")
    dr._write_array(root, "channel", np.array([c.name for c in outputs]), ["channel"])
    dr._write_array(root, "x", grid.x(), ["x"], units="rad", long_name="ABI fixed grid E/W scan angle")
    dr._write_array(root, "y", grid.y(), ["y"], units="rad", long_name="ABI fixed grid N/S scan angle")
    lat, lon = grid_latlon(grid)
    dr._write_array(root, "lat", lat, ["y", "x"], units="degrees_north")
    dr._write_array(root, "lon", lon, ["y", "x"], units="degrees_east")
    dr._write_array(root, "channel_valid_fraction", valid_fraction, ["time", "channel"])
    if cfg.labels:
        dr._write_array(root, "label", np.array([l.name for l in cfg.labels]), ["label"])
    if glm is not None:
        dr._write_array(root, "glm_coverage", glm_coverage, ["time"])

    raw_days = {}
    for d, rd in sorted(raw.days.items()):
        raw_days[(EPOCH + timedelta(days=d)).date().isoformat()] = "missing" if rd.root is None else ("complete" if rd.complete else "incomplete")
    raw_start = datetime.strptime(cfg.raw.start_date_utc, "%Y-%m-%d").date()
    raw_end = datetime.strptime(cfg.raw.end_date_utc, "%Y-%m-%d").date()
    # Days outside the download range will never arrive, so they don't hold the day back
    blocking = [d for d, st in raw_days.items() if st != "complete" and raw_start <= datetime.strptime(d, "%Y-%m-%d").date() <= raw_end]
    complete = raw_days[day_iso] == "complete" and not blocking

    root.attrs.update(
        {
            "day_utc": day_iso,
            "complete": complete,
            "raw_days": raw_days,
            "grid": asdict(grid),
            "time_step_minutes": step * SLOT_MINUTES,
            "issue_offset_minutes": SLOT_MINUTES,
            "channels": [channel_attrs(c) for c in outputs],
            "helper_channels": [channel_attrs(c) for c in cfg.channels if not c.output],
            "labels": [label_attrs(l) for l in cfg.labels],
            "glm": {"quality_flags": cfg.glm_quality_flags, "lightning_ellipsoid_km": cfg.glm_ellipsoid_km, "min_coverage": cfg.glm_min_coverage},
            "channel_stats": {name: st.as_dict() for name, st in stats.items()},
            "label_stats": {name: {"n_known": n, "n_positive": p} for name, (n, p) in label_counts.items()},
            "n_cells_per_time": grid.ny * grid.nx,
            "n_flashes": n_flashes,
            "float16_overflow": {k: v for k, v in n_overflow.items() if v},
            "built_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    zarr.consolidate_metadata(root.store)
    shutil.rmtree(out_path, ignore_errors=True)
    shutil.move(tmp_path, out_path)

    return {
        "day": day_iso,
        "status": "OK" if complete else "INCOMPLETE (raw data: " + ", ".join(f"{d} {st}" for d, st in raw_days.items() if st != "complete") + ")",
        "complete": complete,
        "seconds": time.monotonic() - t_start,
        "bytes": dr.directory_size(out_path),
        "n_flashes": n_flashes,
        "overflow": {k: v for k, v in n_overflow.items() if v},
    }


# =============================================================================
# Runner
# =============================================================================

def run_build(cfg: FeatureConfig) -> None:
    days = [d.date().isoformat() for d in dr.iter_days_utc(cfg.start_date_utc, cfg.end_date_utc)]
    grid = make_grid(cfg)
    plan = make_plan(cfg)
    outputs = [c.name for c in cfg.channels if c.output]
    lat, lon = grid_latlon(grid)
    dr.log(
        f"grid: {grid.ny} x {grid.nx} cells of {grid.resolution_km:g} km (nominal, {grid.coarsen}x{grid.coarsen} {grid.reference_product} pixels), "
        f"lat {np.nanmin(lat):.1f}..{np.nanmax(lat):.1f}, lon {np.nanmin(lon):.1f}..{np.nanmax(lon):.1f}"
    )
    per_day = len(range(0, SLOTS_PER_DAY, cfg.time_step_slots)) * len(outputs) * grid.ny * grid.nx * np.dtype(cfg.dtype).itemsize
    dr.log(
        f"{len(outputs)} channels ({', '.join(outputs)}), {len(cfg.labels)} labels, every {cfg.time_step_slots * SLOT_MINUTES} min; "
        f"lookback {plan.lookback * SLOT_MINUTES} min; features {per_day / 1e9:.1f} GB/day before compression"
    )

    todo = []
    for d in days:
        path = features_day_path(cfg, datetime.strptime(d, "%Y-%m-%d"))
        if not cfg.overwrite_existing and dr.zarr_store_is_complete(path):
            continue
        todo.append(d)
    dr.log(f"{len(days) - len(todo)} of {len(days)} days already complete; building {len(todo)}")

    ok, failed, total_bytes, t0 = [], [], 0, time.monotonic()
    workers = min(cfg.max_workers, max(1, len(todo)))
    # Workers start fresh (spawn) so no zarr/numpy thread state is inherited from this process
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(build_day, cfg, grid, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                dr.log(f"!! {d} failed: {e!r}")
                failed.append(d)
                continue
            (ok if r["complete"] else failed).append(d)
            total_bytes += r.get("bytes", 0)
            elapsed = time.monotonic() - t0
            eta = dr._format_duration(elapsed / i * (len(todo) - i))
            extra = f"  float16 overflow set to NaN: {r['overflow']}" if r.get("overflow") else ""
            if "seconds" in r:
                dr.log(f"-> {d} {r['status']} in {r['seconds']:.0f}s, {r['bytes'] / 1e6:.0f} MB, {r['n_flashes']} flashes  [{i}/{len(todo)}, ETA {eta}]{extra}")
            else:
                dr.log(f"-> {d} {r['status']}  [{i}/{len(todo)}]")

    print("\n" + "=" * 72)
    print(f"Days built:      {len(ok) + len(failed)} of {len(todo)} ({len(ok)} complete)")
    print(f"Days incomplete: {len(failed)}" + (f"  {sorted(failed)}" if failed else ""))
    print(f"Saved to disk:   {total_bytes / 1e9:.2f} GB in {dr._format_duration(time.monotonic() - t0)}")
    print(f"Output root:     {os.path.abspath(cfg.output_root_dir)}")
    print("=" * 72)


def run_stats(cfg: FeatureConfig) -> None:
    """Combines the per-day sums into mean/std per channel and positive rates per label."""
    sums: Dict[str, Dict[str, float]] = {}
    labels: Dict[str, List[int]] = {}
    used, incomplete, missing, cells = [], [], [], 0
    for day in dr.iter_days_utc(cfg.start_date_utc, cfg.end_date_utc):
        path = features_day_path(cfg, day)
        if not os.path.exists(os.path.join(path, ".zgroup")):
            missing.append(day.date().isoformat())
            continue
        attrs = zarr.open_consolidated(path, mode="r").attrs
        (used if attrs.get("complete") else incomplete).append(day.date().isoformat())
        cells += attrs["n_cells_per_time"] * len(range(0, SLOTS_PER_DAY, attrs["time_step_minutes"] // SLOT_MINUTES))
        for name, st in attrs["channel_stats"].items():
            acc = sums.setdefault(name, {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": math.inf, "max": -math.inf})
            acc["count"] += st["count"]
            acc["sum"] += st["sum"]
            acc["sum_sq"] += st["sum_sq"]
            if st["count"]:
                acc["min"], acc["max"] = min(acc["min"], st["min"]), max(acc["max"], st["max"])
        for name, st in attrs["label_stats"].items():
            acc_l = labels.setdefault(name, [0, 0])
            acc_l[0] += st["n_known"]
            acc_l[1] += st["n_positive"]

    if not used and not incomplete:
        raise SystemExit(f"no feature stores between {cfg.start_date_utc} and {cfg.end_date_utc}")
    channels = {}
    for name, a in sums.items():
        n = a["count"]
        mean = a["sum"] / n if n else None
        std = math.sqrt(max(a["sum_sq"] / n - mean * mean, 0.0)) if n else None
        channels[name] = {"mean": mean, "std": std, "min": a["min"] if n else None, "max": a["max"] if n else None,
                          "valid_fraction": n / cells if cells else None}
    result = {
        "start_date_utc": cfg.start_date_utc,
        "end_date_utc": cfg.end_date_utc,
        "days_used": len(used) + len(incomplete),
        "days_incomplete": incomplete,
        "days_missing": missing,
        "channels": channels,
        "labels": {name: {"positive_fraction": p / n if n else None, "known_fraction": n / cells if cells else None}
                   for name, (n, p) in labels.items()},
    }
    out = os.path.join(cfg.output_root_dir, f"stats_{cfg.start_date_utc}_{cfg.end_date_utc}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"{'channel':24s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s} {'valid':>6s}")
    for name, c in channels.items():
        if c["mean"] is None:
            print(f"{name:24s} {'(no data)':>12s}")
            continue
        print(f"{name:24s} {c['mean']:12.4g} {c['std']:12.4g} {c['min']:12.4g} {c['max']:12.4g} {100 * c['valid_fraction']:5.1f}%")
    for name, l in result["labels"].items():
        pos = "n/a" if l["positive_fraction"] is None else f"{100 * l['positive_fraction']:.3f}%"
        print(f"label {name}: positive {pos} of known cells, known {100 * (l['known_fraction'] or 0):.1f}%")
    print(f"{result['days_used']} days ({len(incomplete)} incomplete, {len(missing)} missing) -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build CNN features and lightning labels from the raw GOES day stores.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    p.add_argument("--start", help="First UTC day (YYYY-MM-DD); overrides features.start_date_utc")
    p.add_argument("--end", help="Last UTC day (YYYY-MM-DD); overrides features.end_date_utc")
    p.add_argument("--workers", type=int, help="Days built in parallel; overrides features.max_workers")
    p.add_argument("--overwrite", action="store_true", help="Rebuild days that are already complete")
    p.add_argument("--stats", action="store_true", help="Combine per-day stats over the date range instead of building")
    args = p.parse_args()

    cfg = load_feature_config(args.config)
    overrides: Dict[str, Any] = {}
    for key, value in (("start_date_utc", args.start), ("end_date_utc", args.end)):
        if value is not None:
            datetime.strptime(value, "%Y-%m-%d")
            overrides[key] = value
    if args.workers is not None:
        overrides["max_workers"] = max(1, args.workers)
    if args.overwrite:
        overrides["overwrite_existing"] = True
    if overrides:
        cfg = replace(cfg, **overrides)
    if args.stats:
        run_stats(cfg)
    else:
        run_build(cfg)


if __name__ == "__main__":
    main()
