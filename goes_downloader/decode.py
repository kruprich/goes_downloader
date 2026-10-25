"""
Decoding NetCDF files into packed arrays (ABI) and flash/event tables (GLM), and the worker pool
that runs it. The decoders run in worker processes.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import re
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import NAT_NS
from .progress import log

SECONDS_SINCE_PATTERN = re.compile(r"seconds since (\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


def _squeeze_to_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.squeeze(np.asarray(a))
    if a.ndim != 2:
        raise ValueError(f"{name} not 2D after squeeze: shape={a.shape}")
    return a


def _as_unsigned(raw: np.ndarray, var: Any, target: Any, name: str) -> np.ndarray:
    """Reinterprets NetCDF _Unsigned integers and narrows to `target` without changing any value."""
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name}: expected packed integers, got {raw.dtype}")
    if str(getattr(var, "_Unsigned", "false")).lower() == "true" and np.issubdtype(raw.dtype, np.signedinteger):
        raw = raw.view(raw.dtype.str.replace("i", "u"))
    info = np.iinfo(target)
    if raw.size and (raw.min() < info.min or raw.max() > info.max):
        raise ValueError(f"{name}: values do not fit in {np.dtype(target)}")
    return raw.astype(target, copy=False)


def quantize_packed(value: np.ndarray, fill_value: Optional[int], scale_factor: float, precision: Optional[float]) -> Tuple[np.ndarray, float]:
    """
    Rounds packed values to the nearest multiple of 2**n, the largest power of two whose physical
    step (scale_factor * 2**n) is <= precision, with ties to even so the rounding adds no bias.
    Zeroing low bits lets the compressor shrink the data; physical = value * scale_factor +
    add_offset still holds. Returns (values, physical step).
    """
    step = abs(scale_factor)
    if precision is None or step <= 0 or precision < 2 * step:
        return value, step
    n = int(np.floor(np.log2(precision / step)))
    top = (fill_value - 1) if fill_value is not None else np.iinfo(np.uint16).max
    rounded = np.round(value.astype(np.float64) / (1 << n)) * (1 << n)  # exact: integers / powers of two
    rounded = np.minimum(rounded, (top >> n) << n).astype(np.uint16)
    if fill_value is not None:
        rounded[value == fill_value] = fill_value
    return rounded, step * (1 << n)


def _embed(part: np.ndarray, shape: Tuple[int, ...], ys: slice, xs: slice, fill: int) -> np.ndarray:
    full = np.full(shape, fill, dtype=part.dtype)
    full[ys, xs] = part
    return full


def _decode_abi_variable(
    ds: Any, data_variable: str, dqf_variable: str, precision: Optional[float], window: Optional[Tuple[int, int, int, int]]
) -> Dict[str, Any]:
    v = ds.variables[data_variable]
    q = ds.variables[dqf_variable]
    v.set_auto_maskandscale(False)
    q.set_auto_maskandscale(False)

    if window is None:
        value_raw, dqf_raw = v[:], q[:]
    else:
        if v.ndim != 2 or q.ndim != 2:
            raise ValueError(f"{data_variable}/{dqf_variable}: byte-range downloads need 2D variables")
        ys, xs = slice(window[0], window[1]), slice(window[2], window[3])
        value_raw, dqf_raw = v[ys, xs], q[ys, xs]
    value = _as_unsigned(_squeeze_to_2d(value_raw, data_variable), v, np.uint16, data_variable)
    dqf_raw = _squeeze_to_2d(dqf_raw, dqf_variable)
    dqf_dtype = np.uint8 if dqf_raw.dtype.itemsize == 1 else np.uint16
    dqf = _as_unsigned(dqf_raw, q, dqf_dtype, dqf_variable)

    fill = getattr(v, "_FillValue", None)
    fill_value = None
    if fill is not None:
        fill_value = int(_as_unsigned(np.asarray([fill], dtype=v.dtype), v, np.uint16, "_FillValue")[0])
    dqf_fill = getattr(q, "_FillValue", None)
    dqf_fill_value = (
        int(_as_unsigned(np.asarray([dqf_fill], dtype=q.dtype), q, dqf_dtype, "DQF _FillValue")[0])
        if dqf_fill is not None
        else int(np.iinfo(dqf_dtype).max)
    )

    scale_factor = float(getattr(v, "scale_factor", 1.0))
    value, quantization_step = quantize_packed(value, fill_value, scale_factor, precision)
    if window is not None:
        # Only the window was downloaded; the main process crops the rest of the frame away
        value = _embed(value, v.shape, ys, xs, fill_value if fill_value is not None else int(np.iinfo(np.uint16).max))
        dqf = _embed(dqf, q.shape, ys, xs, dqf_fill_value)
    if dqf.shape != value.shape:
        raise ValueError(f"{dqf_variable} shape {dqf.shape} != {data_variable} shape {value.shape}")
    return {
        "value": value,
        "dqf": dqf,
        "dqf_fill_value": dqf_fill_value,
        "scale_factor": scale_factor,
        "add_offset": float(getattr(v, "add_offset", 0.0)),
        "quantization_step": quantization_step,
        "fill_value": fill_value,
        "units": str(getattr(v, "units", "")),
    }


def abi_grid(ds: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(x, y, projection attributes) of an open ABI file."""
    # x/y are scaled int16 in the file; netCDF4 unpacks them to radians by default
    x = np.asarray(ds.variables["x"][:], dtype=np.float64).reshape(-1)
    y = np.asarray(ds.variables["y"][:], dtype=np.float64).reshape(-1)
    proj = ds.variables["goes_imager_projection"]
    return x, y, {k: _jsonable(proj.getncattr(k)) for k in proj.ncattrs()}


def decode_abi_file(
    payload: bytes, variables: List[Tuple[str, str, Optional[float]]], window: Optional[Tuple[int, int, int, int]] = None
) -> List[Dict[str, Any]]:
    """
    Decodes each (data_variable, dqf_variable, precision) from one file, so products that read
    different variables from the same file (e.g. the DSI indices) share one download. With a window
    (y0, y1, x0, x1) only those pixels are read and the rest of each frame is fill.
    """
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        x, y, projection = abi_grid(ds)
        return [
            {**_decode_abi_variable(ds, data_variable, dqf_variable, precision, window), "x": x, "y": y, "projection": projection}
            for data_variable, dqf_variable, precision in variables
        ]


def _offset_seconds_to_ns(var: Any) -> np.ndarray:
    m = SECONDS_SINCE_PATTERN.search(str(getattr(var, "units", "")))
    if not m:
        raise ValueError(f"{var.name}: unsupported time units {getattr(var, 'units', None)!r}")
    base_ns = np.datetime64(f"{m.group(1)}T{m.group(2)}", "ns").astype(np.int64)
    seconds = np.ma.filled(np.ma.asarray(var[:], dtype=np.float64), np.nan).reshape(-1)
    out = np.full(seconds.shape, NAT_NS, dtype=np.int64)
    ok = np.isfinite(seconds)
    out[ok] = base_ns + np.round(seconds[ok] * 1e9).astype(np.int64)
    return out


# GLM file variables. decode_glm_file reads only GLM_READ_VARIABLES, and byte-range downloads fetch only those.
GLM_FLASH_VARIABLES = {  # flash column -> variable
    "time_first_ns": "flash_time_offset_of_first_event",
    "time_last_ns": "flash_time_offset_of_last_event",
    "lat": "flash_lat",
    "lon": "flash_lon",
    "area_m2": "flash_area",
    "energy_j": "flash_energy",
    "quality_flag": "flash_quality_flag",
}
GLM_EVENT_VARIABLES = ("event_lat", "event_lon")
# Each event's flash: event -> parent group -> parent flash
GLM_LINK_VARIABLES = ("event_parent_group_id", "group_id", "group_parent_flash_id", "flash_id")
GLM_READ_VARIABLES = [*GLM_FLASH_VARIABLES.values(), *GLM_EVENT_VARIABLES, *GLM_LINK_VARIABLES]


def _parent_rows(parent_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """
    Row in `ids` of each parent id, or -1 where the parent can't be found in the file. NOAA's files
    have such orphans, which are kept rather than failing the file:
    - parents missing from the file (e.g. 64 of 2410 events on 2025-04-22 16:08:20; some files
      have events but no groups at all);
    - ids used twice, since flash ids are 16-bit and wrap around in very active files (6 of
      16,065 flash ids on 2026-08-23 16:53:40), so their children can't tell which is theirs.
    """
    if len(parent_ids) == 0 or len(ids) == 0:
        return np.full(len(parent_ids), -1, dtype=np.int32)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    first = np.searchsorted(sorted_ids, parent_ids, side="left")
    unique = np.searchsorted(sorted_ids, parent_ids, side="right") - first == 1
    return np.where(unique, order[np.minimum(first, len(ids) - 1)], -1).astype(np.int32)


def _through(rows: np.ndarray, parent_of_row: np.ndarray) -> np.ndarray:
    """parent_of_row[rows], with -1 where rows is -1 (also when parent_of_row is empty, as in files
    whose events all lack their groups)."""
    out = np.full(len(rows), -1, dtype=np.int32)
    ok = rows >= 0
    out[ok] = parent_of_row[rows[ok]]
    return out


def decode_glm_file(payload: bytes) -> Dict[str, np.ndarray]:
    """
    The flash columns, plus event_lat, event_lon and event_flash: the row of the event's flash in this
    file, or -1 for an orphan (its group, or its group's flash, is not in the file).
    """
    import netCDF4 as nc

    def floats(var: Any) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(var[:], dtype=np.float32), np.nan).reshape(-1)

    def ids(var: Any) -> np.ndarray:
        return np.ma.getdata(var[:]).astype(np.int64).reshape(-1)

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        var = {name: ds.variables[name] for name in GLM_READ_VARIABLES}
        flash = {column: var[name] for column, name in GLM_FLASH_VARIABLES.items()}
        quality = np.ma.filled(np.ma.asarray(flash["quality_flag"][:], dtype=np.int16), 255)
        event_group = _parent_rows(ids(var["event_parent_group_id"]), ids(var["group_id"]))
        group_flash = _parent_rows(ids(var["group_parent_flash_id"]), ids(var["flash_id"]))
        return {
            "time_first_ns": _offset_seconds_to_ns(flash["time_first_ns"]),
            "time_last_ns": _offset_seconds_to_ns(flash["time_last_ns"]),
            "lat": floats(flash["lat"]),
            "lon": floats(flash["lon"]),
            "area_m2": floats(flash["area_m2"]),
            "energy_j": floats(flash["energy_j"]),
            "quality_flag": quality.astype(np.uint8).reshape(-1),
            "event_lat": floats(var["event_lat"]),
            "event_lon": floats(var["event_lon"]),
            "event_flash": _through(event_group, group_flash),
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _ignore_ctrl_c() -> None:
    """Worker initializer: Ctrl-C reaches every process in the terminal's group, but only the main
    process should react; it stops the run and shuts the pool down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class DecodePool:
    """
    Process pool for decoding (and, with byte-range downloads, for fetching too). A worker process
    that dies (out of memory, a crash on a corrupt file) breaks the whole ProcessPoolExecutor, so the
    first task to notice replaces it. The files that were in the pool at that moment fail, which
    leaves their days incomplete for the next run to redo.
    """

    def __init__(self, workers: int):
        self.workers = workers
        self.pool = self._new_pool()

    def _new_pool(self) -> ProcessPoolExecutor:
        # Workers start as fresh processes on every platform (the macOS default). Linux would otherwise
        # fork this process while its HTTP and Zarr-writer threads run, which can deadlock a worker.
        return ProcessPoolExecutor(
            max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_ignore_ctrl_c
        )

    async def run(self, fn: Any, *args: Any) -> Any:
        pool = self.pool
        try:
            return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)
        except BrokenProcessPool:
            if self.pool is pool:
                log("!! a decode process died; starting a new process pool")
                self.pool = self._new_pool()
                # Before Python 3.12 the broken pool's feeder thread can block forever writing a queued
                # file to the dead workers, which then hangs the program at exit (CPython gh-94777).
                # Closing the read end of that pipe unblocks it, as newer Pythons do themselves.
                try:
                    pool._call_queue._reader.close()
                except Exception:
                    pass
                pool.shutdown(wait=False, cancel_futures=True)
            raise

    def shutdown(self) -> None:
        # Queued files are dropped (after Ctrl-C there can be many); files being decoded finish first
        self.pool.shutdown(cancel_futures=True)
