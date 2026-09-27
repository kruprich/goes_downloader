"""
download_raw.py

GOES ABI (CONUS L2) + GLM (LCFA) -> per-day raw Zarr store, no aggregation
==========================================================================

Mirrors the source data into one Zarr store per UTC day so that binning, regridding
and quality filtering can all happen later in a feature-building step.

Layout of <raw_output_root_dir>/YYYY/YYYYDDD.zarr

Every array names its dimensions and each day's metadata is consolidated, so xarray can open the
stores (see "Reading with xarray" below).

ABI: one group per configured product, one slot per 5-min CONUS scan (288 per day).
Each product keeps its source grid (CMIP bands 1, 3, 5: 1 km and band 2: 0.5 km in their own files, every
other band and every band of the all-band MCMIP file: 2 km, 1500x2500; ACM, ACTP: 2 km; ACHA, TPW, DSI: 10 km,
300x500).
  abi/<key>/time[slot]               datetime64 start of each 5-min slot (the xarray "time" coordinate)
  abi/<key>/value[slot, y, x]        uint16   packed values as in the file, rounded to the product's
                                              configured "precision" (physical units) if one is set
                                              physical = value * scale_factor[slot] + add_offset[slot]
                                              value == attrs["fill_value"] means no data (for files without a
                                              _FillValue the store uses 65535)
  abi/<key>/dqf[slot, y, x]          uint8    data quality flags as in the file (uint16 for products with
                                              16-bit flags; attrs["dqf_fill_value"] = fill)
  abi/<key>/scale_factor[slot]       float64
  abi/<key>/add_offset[slot]         float64
  abi/<key>/quantization_step[slot]  float64  physical step between stored values after rounding
  abi/<key>/scan_start_ns[slot]      int64    scan start/end from the filename, ns since 1970; NaT (int64 min)
                                              if no scan. xarray decodes these to datetime64
  abi/<key>/scan_end_ns[slot]        int64
  abi/<key>/present[slot]            uint8    1 if the scan was downloaded and decoded
  abi/<key>/x_scan_angle_rad[x]      float64
  abi/<key>/y_scan_angle_rad[y]      float64

GLM (only if "glm" is in the config's "download" list): every flash and event from every 20 s file whose
start falls in the day, all quality flags kept.
  glm/files/{start_ns,end_ns,ok,n_flashes,n_events}[file]    one row per file NOAA published for the day
  glm/flashes/{time_first_ns,time_last_ns,lat,lon,area_m2,energy_j,quality_flag,file_index}[flash]
                                       file_index is the row in the same day's glm/files; it is not
                                       unique once several days are combined
  glm/events/{lat,lon,flash_index}[event]   every event (one lit sensor pixel at one moment), in file
                                       order; flash_index is the row of its flash in the same day's
                                       glm/flashes. Events give each flash's footprint, which a
                                       flash-extent density needs
  glm/coverage/time[window]            datetime64 start of each 20 s window (4320 per day)
  glm/coverage/covered[window]         uint8  1 if a GLM file for the window was published and decoded
  Windows with covered=0 are gaps (NOAA published no file, or it failed): no flashes there does NOT
  mean no lightning.

Optional "bounds" crop (lat/lon box in config), ABI only:
  The box is turned into the smallest scan-angle rectangle that contains it, shared by all ABI
  products. Each ABI product keeps the pixels whose cell overlaps that rectangle (attrs crop_y0/crop_x0
  give the offset into the uncropped grid; x/y_scan_angle_rad are the cropped coordinates). GLM is not
  cropped: every flash and event is kept (~240 MB/day), and the feature step selects what its grid needs.
  Days already complete are not re-cropped when the bounds change; delete them or set
  overwrite_existing.

Root attrs:
  complete = true only if every listing request succeeded, every file was downloaded and decoded,
  and the day ended at least 24 h ago (NOAA publishes some files late). Incomplete days are rebuilt
  from scratch on the next run.

Reading with xarray (one product or GLM table across several days):
  xr.open_mfdataset(day_paths, engine="zarr", group="abi/mcmip_c13", combine="by_coords")
  xr.open_mfdataset(day_paths, engine="zarr", group="glm/flashes", combine="nested", concat_dim="flash")
  xarray turns value/dqf fill values into NaN (floats); pass mask_and_scale=False for the stored integers
  and apply scale_factor/add_offset yourself.

Config:
  python download_raw.py --config config.json
  abi_products / glm_product define every product; "download" lists the keys to fetch.
  Products that read different variables from the same files (e.g. the dsi_* indices) share one
  download per scan, and each bucket prefix is listed once per day.
  download_method "byte_ranges" fetches only the HDF5 metadata and the chunks that are decoded
  (ABI chunks inside the crop) instead of whole files; the stored data is the same. The mcmip_*
  products read single bands from the ~59 MB all-band files and are only practical this way.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import http.client
import io
import json
import mmap
import multiprocessing
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import zarr


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class ProductSpec:
    key: str
    gcs_product_prefix: str
    filename_must_contain: str
    data_variable: Optional[str] = None
    dqf_variable: Optional[str] = None
    # Round stored values to this step (physical units, e.g. K); None keeps full precision
    precision: Optional[float] = None


@dataclass(frozen=True)
class LatLonBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class RawConfig:
    gcs_bucket: str
    start_date_utc: str
    end_date_utc: str
    raw_output_root_dir: str
    overwrite_existing: bool

    # Total open connections (listing + downloads, all days); every request goes to one host
    max_concurrent_http_requests: int
    http_timeout_seconds: int
    http_retries: int

    max_decode_processes: int
    max_days_in_flight: int
    # Caps how many files are between "download started" and "written to Zarr" per day.
    max_inflight_payloads: int

    # "whole_files": the main process downloads each file and decode processes decode it.
    # "byte_ranges": worker processes fetch only the HDF5 metadata and the chunks that are read
    # (inside the ABI crop), then decode, so each worker handles one file at a time.
    download_method: str
    byte_range_processes: int

    # Only the products named in the config's "download" list
    abi_products: List[ProductSpec]
    glm_product: Optional[ProductSpec]

    # None keeps the full ABI sector and every GLM flash
    bounds: Optional[LatLonBounds]


def _require(d: Dict[str, Any], k: str, typ: Any):
    if k not in d:
        raise ValueError(f"Config missing required key: {k!r}")
    v = d[k]
    if typ is not None and not isinstance(v, typ):
        raise ValueError(f"Config key {k!r} must be {typ}, got {type(v)}")
    return v


CMIP_BAND_PATTERN = re.compile(r"CMIP\w*-M\d+C(\d{2})")  # single-band files: ...-CMIPC-M6C13
MCMIP_BAND_VARIABLE = re.compile(r"CMI_C(\d{2})")  # the all-band MCMIP files hold one CMI_Cnn per band


def is_cmip_ir_band(product: Dict[str, Any]) -> bool:
    """CMIP bands 7-16 hold brightness temperature (K); bands 1-6 are reflectance (0-1)."""
    m = CMIP_BAND_PATTERN.search(str(product.get("filename_must_contain", ""))) or MCMIP_BAND_VARIABLE.fullmatch(
        str(product.get("data_variable", ""))
    )
    return bool(m) and int(m.group(1)) >= 7


def load_config(config_path: str) -> RawConfig:
    """Reads the shared config.json; keys only used by the feature step are ignored."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # One precision (K) for every infrared CMIP band; a product's own "precision" overrides it
    cmip_precision = raw.get("cmip_precision")
    if cmip_precision is not None and (not isinstance(cmip_precision, (int, float)) or cmip_precision <= 0):
        raise ValueError("cmip_precision must be a positive number or null")

    defined_abi: List[ProductSpec] = []
    for i, p in enumerate(_require(raw, "abi_products", list)):
        if not isinstance(p, dict):
            raise ValueError(f"abi_products[{i}] must be an object")
        precision = p["precision"] if "precision" in p else (cmip_precision if is_cmip_ir_band(p) else None)
        if precision is not None and (not isinstance(precision, (int, float)) or precision <= 0):
            raise ValueError(f"abi_products[{i}].precision must be a positive number")
        defined_abi.append(
            ProductSpec(
                key=_require(p, "key", str),
                gcs_product_prefix=_require(p, "gcs_product_prefix", str),
                filename_must_contain=_require(p, "filename_must_contain", str),
                data_variable=_require(p, "data_variable", str),
                dqf_variable=_require(p, "dqf_variable", str),
                precision=float(precision) if precision is not None else None,
            )
        )
    glm_raw = _require(raw, "glm_product", dict)
    defined_glm = ProductSpec(
        key=_require(glm_raw, "key", str),
        gcs_product_prefix=_require(glm_raw, "gcs_product_prefix", str),
        filename_must_contain=_require(glm_raw, "filename_must_contain", str),
    )

    keys = [p.key for p in defined_abi] + [defined_glm.key]
    if len(set(keys)) != len(keys):
        raise ValueError(f"product keys must be unique across abi_products and glm_product, got {keys}")

    download = _require(raw, "download", list)
    unknown = [k for k in download if k not in keys]
    if unknown or not download:
        raise ValueError(f"download must list product keys from {keys}; unknown: {unknown}")
    abi_products = [p for p in defined_abi if p.key in download]
    glm_product = defined_glm if defined_glm.key in download else None

    bounds: Optional[LatLonBounds] = None
    bounds_raw = raw.get("bounds")
    if bounds_raw is not None:
        if not isinstance(bounds_raw, dict):
            raise ValueError("bounds must be an object with lat_min, lat_max, lon_min, lon_max")
        bounds = LatLonBounds(**{k: float(_require(bounds_raw, k, (int, float))) for k in ("lat_min", "lat_max", "lon_min", "lon_max")})
        if not (-90 <= bounds.lat_min < bounds.lat_max <= 90 and -180 <= bounds.lon_min < bounds.lon_max <= 180):
            raise ValueError(f"bounds out of range or min >= max: {bounds}")

    start_date_utc = _require(raw, "start_date_utc", str)
    end_date_utc = _require(raw, "end_date_utc", str)
    datetime.strptime(start_date_utc, "%Y-%m-%d")
    datetime.strptime(end_date_utc, "%Y-%m-%d")

    download_method = raw.get("download_method", "whole_files")
    if download_method not in ("whole_files", "byte_ranges"):
        raise ValueError(f"download_method must be 'whole_files' or 'byte_ranges', got {download_method!r}")

    max_concurrent_http_requests = int(raw.get("max_concurrent_http_requests", 32))
    byte_range_processes = int(raw.get("byte_range_processes", 48))
    # A file waiting for a byte-range worker holds no data, so let every worker have one per day
    default_inflight = byte_range_processes if download_method == "byte_ranges" else max_concurrent_http_requests
    return RawConfig(
        gcs_bucket=_require(raw, "gcs_bucket", str),
        start_date_utc=start_date_utc,
        end_date_utc=end_date_utc,
        raw_output_root_dir=str(raw.get("raw_output_root_dir", "./goes_raw")),
        overwrite_existing=bool(raw.get("overwrite_existing", False)),
        max_concurrent_http_requests=max_concurrent_http_requests,
        http_timeout_seconds=int(raw.get("http_timeout_seconds", 60)),
        http_retries=max(1, int(raw.get("http_retries", 5))),
        # Decoding takes ~35 s of CPU per day of data, so a few processes keep up with the download
        max_decode_processes=int(raw.get("max_decode_processes", 4)),
        max_days_in_flight=int(raw.get("max_days_in_flight", 2)),
        max_inflight_payloads=int(raw.get("max_inflight_payloads", default_inflight)),
        download_method=download_method,
        byte_range_processes=byte_range_processes,
        abi_products=abi_products,
        glm_product=glm_product,
        bounds=bounds,
    )


# =============================================================================
# Constants + helpers
# =============================================================================

SLOTS_PER_DAY = 288
SLOT_SECONDS = 300
GLM_WINDOW_SECONDS = 20
GLM_WINDOWS_PER_DAY = 86400 // GLM_WINDOW_SECONDS
# A day is only marked complete once it ended this long ago; NOAA publishes some files late
MIN_AGE_FOR_COMPLETE = timedelta(hours=24)
NAT_NS = np.iinfo(np.int64).min  # int64 view of numpy NaT
# CF attributes that make xarray decode int64 nanosecond arrays to datetime64 (int64 min -> NaT)
TIME_ATTRS = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}
LIST_HOURS = range(24)
MAX_ERRORS_LOGGED_PER_DAY = 5

GOES_FILENAME_TIME_PATTERN = re.compile(r"_([sec])(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)")
SECONDS_SINCE_PATTERN = re.compile(r"seconds since (\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)")

RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if _PROGRESS is not None:
        _PROGRESS.clear()
    print(f"[{ts}] {msg}", flush=True)
    if _PROGRESS is not None:
        _PROGRESS.restore()


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class Progress:
    """
    One-line progress bar with download speed and ETA. In a terminal it is redrawn in place and log
    lines print above it; when output goes to a file it prints a plain progress line once a minute.
    Each day's download size is known exactly once it is listed; days not listed yet are estimated
    from the average listed day. The bar and ETA count source file sizes; the GB downloaded and MB/s
    shown are what actually went over the network, which is less with byte-range downloads.
    """

    WIDTH = 30
    SPEED_WINDOW_SECONDS = 30.0
    PLAIN_EVERY_SECONDS = 60.0

    def __init__(self, total_days: int):
        self.total_days = total_days
        self.days_done = 0
        self.day_total: Dict[str, int] = {}  # source file bytes, per day in progress
        self.day_done: Dict[str, int] = {}
        self.listed_day_bytes: List[int] = []
        self.bytes_done = 0  # source file bytes of the files handled so far
        self.bytes_downloaded = 0
        self.bytes_saved = 0  # on-disk size of the day stores written this run, added as each day finishes
        self.started = time.monotonic()
        self.samples: deque = deque([(self.started, 0, 0)])
        self.tty = sys.stdout.isatty()
        self.text = ""
        self.shown = False
        self.last_plain = self.started

    def start_day(self, day: str, total_bytes: int) -> None:
        self.day_total[day] = total_bytes
        self.day_done[day] = 0
        self.listed_day_bytes.append(total_bytes)

    def add_bytes(self, day: str, n: int, downloaded: Optional[int] = None) -> None:
        """n: source file bytes handled; downloaded: bytes fetched for them (default n)."""
        if day in self.day_done:
            self.day_done[day] += n
        self.bytes_done += n
        self.bytes_downloaded += n if downloaded is None else downloaded

    def add_saved(self, n: int) -> None:
        self.bytes_saved += n

    def finish_day(self, day: str) -> None:
        self.day_total.pop(day, None)
        self.day_done.pop(day, None)
        self.days_done += 1

    def _speeds(self) -> Tuple[float, float]:
        """(source file bytes/s, downloaded bytes/s) over the last SPEED_WINDOW_SECONDS."""
        now = time.monotonic()
        self.samples.append((now, self.bytes_done, self.bytes_downloaded))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.SPEED_WINDOW_SECONDS:
            self.samples.popleft()
        t0, b0, d0 = self.samples[0]
        if now <= t0:
            return 0.0, 0.0
        return (self.bytes_done - b0) / (now - t0), (self.bytes_downloaded - d0) / (now - t0)

    def _remaining_bytes(self) -> Optional[float]:
        in_progress = sum(max(0, self.day_total[d] - self.day_done[d]) for d in self.day_total)
        not_started = self.total_days - self.days_done - len(self.day_total)
        if not_started <= 0:
            return float(in_progress)
        if not self.listed_day_bytes:
            return None
        return in_progress + not_started * sum(self.listed_day_bytes) / len(self.listed_day_bytes)

    def _line(self) -> str:
        speed, download_speed = self._speeds()
        partial = sum(min(1.0, self.day_done[d] / self.day_total[d]) for d in self.day_total if self.day_total[d])
        frac = min(1.0, (self.days_done + partial) / max(1, self.total_days))
        filled = int(round(frac * self.WIDTH))
        remaining = self._remaining_bytes()
        eta = _format_duration(remaining / speed) if remaining is not None and speed > 0 else "--"
        return (
            f"[{'█' * filled}{'░' * (self.WIDTH - filled)}] {100 * frac:5.1f}%  "
            f"{self.days_done}/{self.total_days} days  {self.bytes_downloaded / 1e9:.1f} GB downloaded  "
            f"{self.bytes_saved / 1e9:.1f} GB saved  {download_speed / 1e6:5.1f} MB/s  ETA {eta}"
        )

    def draw(self) -> None:
        self.text = self._line()
        if self.tty:
            self.restore()
        elif time.monotonic() - self.last_plain >= self.PLAIN_EVERY_SECONDS:
            self.last_plain = time.monotonic()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] progress {self.text}", flush=True)

    def restore(self) -> None:
        if self.tty and self.text:
            sys.stdout.write("\r\x1b[2K" + self.text)
            sys.stdout.flush()
            self.shown = True

    def clear(self) -> None:
        if self.tty and self.shown:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            self.shown = False

    def close(self) -> None:
        """Leaves the final progress line on screen and moves to the next line."""
        self.draw()
        if self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] progress {self.text}", flush=True)
        self.shown = False


_PROGRESS: Optional[Progress] = None


def to_ns(t: datetime) -> int:
    return int(np.datetime64(t.replace(tzinfo=None), "ns").astype(np.int64))


def iter_days_utc(start_date: str, end_date: str):
    cur = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def infer_satellite_label_from_bucket(bucket: str) -> str:
    m = re.search(r"goes-(\d+)", bucket)
    return f"GOES-{int(m.group(1))}" if m else "UNKNOWN"


def output_zarr_path_for_day(root_dir: str, day: datetime) -> str:
    return os.path.join(root_dir, str(day.year), f"{day.year}{day.timetuple().tm_yday:03d}.zarr")


def directory_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)


def zarr_store_is_complete(path: str) -> bool:
    if not os.path.exists(os.path.join(path, ".zgroup")):
        return False
    try:
        return bool(zarr.open_group(path, mode="r").attrs.get("complete", False))
    except Exception:
        return False


@dataclass(frozen=True)
class RemoteFile:
    blob: str
    start: datetime
    end: datetime
    created: datetime
    size: int = 0


def parse_remote_file(blob: str, size: int = 0) -> Optional[RemoteFile]:
    times: Dict[str, datetime] = {}
    for m in GOES_FILENAME_TIME_PATTERN.finditer(os.path.basename(blob)):
        kind, y, j, hh, mm, ss, tenth = m.groups()
        times[kind] = datetime(int(y), 1, 1, tzinfo=timezone.utc) + timedelta(
            days=int(j) - 1, hours=int(hh), minutes=int(mm), seconds=int(ss), milliseconds=100 * int(tenth)
        )
    if set(times) != {"s", "e", "c"}:
        return None
    return RemoteFile(blob=blob, start=times["s"], end=times["e"], created=times["c"], size=size)


# =============================================================================
# HTTP (retries on transient errors, raises when out of attempts)
# =============================================================================

def _backoff_seconds(attempt: int) -> float:
    return min(30.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random())


async def _backoff(attempt: int) -> None:
    await asyncio.sleep(_backoff_seconds(attempt))


async def http_get(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, str]] = None,
    *,
    as_json: bool,
) -> Any:
    timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_seconds)
    last_error = "no attempts"
    for attempt in range(1, cfg.http_retries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as r:
                if r.status == 200:
                    return await (r.json() if as_json else r.read())
                last_error = f"HTTP {r.status}"
                if r.status not in RETRYABLE_HTTP_STATUS:
                    break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = repr(e)
        if attempt < cfg.http_retries:
            await _backoff(attempt)
    raise RuntimeError(f"GET {url} failed: {last_error}")


async def gcs_list_object_names(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    prefix: str,
) -> Dict[str, int]:
    """Object name -> size in bytes for everything under the prefix."""
    url = f"https://storage.googleapis.com/storage/v1/b/{cfg.gcs_bucket}/o"
    params: Dict[str, str] = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
    objects: Dict[str, int] = {}
    while True:
        js = await http_get(cfg, session, url, params, as_json=True)
        objects.update((it["name"], int(it.get("size", 0))) for it in js.get("items", []) if it.get("name"))
        page_token = js.get("nextPageToken")
        if not page_token:
            return objects
        params["pageToken"] = page_token


async def list_day_names(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    gcs_product_prefix: str,
    day: datetime,
) -> Tuple[Dict[str, int], int]:
    """Returns (name -> size for every object under the day's 24 hourly prefixes, n_failed_prefixes)."""
    jday = day.timetuple().tm_yday
    prefixes = [f"{gcs_product_prefix}/{day.year}/{jday:03d}/{hh:02d}/" for hh in LIST_HOURS]
    results = await asyncio.gather(
        *(gcs_list_object_names(cfg, session, p) for p in prefixes), return_exceptions=True
    )

    objects: Dict[str, int] = {}
    n_failed = 0
    for prefix, res in zip(prefixes, results):
        if isinstance(res, BaseException):
            n_failed += 1
            log(f"!! listing failed for {prefix}: {res!r}")
            continue
        objects.update(res)
    return objects, n_failed


def select_files(objects: Dict[str, int], filename_must_contain: str) -> List[RemoteFile]:
    """
    Files whose name contains the pattern, deduplicated by scan start time (keeping the newest
    creation time, i.e. reprocessed files) and sorted by start time.
    """
    by_start: Dict[datetime, RemoteFile] = {}
    for name, size in objects.items():
        if filename_must_contain not in name:
            continue
        rf = parse_remote_file(name, size)
        if rf is None:
            continue
        prev = by_start.get(rf.start)
        if prev is None or rf.created > prev.created:
            by_start[rf.start] = rf
    return sorted(by_start.values(), key=lambda rf: rf.start)


# =============================================================================
# Decoding (runs in worker processes)
# =============================================================================

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


def _abi_grid(ds: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
        x, y, projection = _abi_grid(ds)
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


def _parent_rows(parent_ids: np.ndarray, ids: np.ndarray, what: str) -> np.ndarray:
    """Row in `ids` of each parent id; every id must be unique and every parent present."""
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"{what}: duplicate ids")
    if len(parent_ids) == 0:
        return np.zeros(0, dtype=np.int32)
    pos = np.minimum(np.searchsorted(sorted_ids, parent_ids), max(len(ids) - 1, 0))
    if len(ids) == 0 or np.any(sorted_ids[pos] != parent_ids):
        raise ValueError(f"{what}: parent id missing from the file")
    return order[pos].astype(np.int32)


def decode_glm_file(payload: bytes) -> Dict[str, np.ndarray]:
    """The flash columns, plus event_lat, event_lon and event_flash (row of the event's flash in this file)."""
    import netCDF4 as nc

    def floats(var: Any) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(var[:], dtype=np.float32), np.nan).reshape(-1)

    def ids(var: Any) -> np.ndarray:
        return np.ma.getdata(var[:]).astype(np.int64).reshape(-1)

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        var = {name: ds.variables[name] for name in GLM_READ_VARIABLES}
        flash = {column: var[name] for column, name in GLM_FLASH_VARIABLES.items()}
        quality = np.ma.filled(np.ma.asarray(flash["quality_flag"][:], dtype=np.int16), 255)
        event_group = _parent_rows(ids(var["event_parent_group_id"]), ids(var["group_id"]), "event_parent_group_id")
        group_flash = _parent_rows(ids(var["group_parent_flash_id"]), ids(var["flash_id"]), "group_parent_flash_id")
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
            "event_flash": group_flash[event_group],
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


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
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))

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
        self.pool.shutdown()


# =============================================================================
# Lat/lon bounds crop (ABI only)
# =============================================================================


def latlon_to_scan_angle(lat: np.ndarray, lon: np.ndarray, projection: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Lat/lon (deg) -> ABI fixed-grid scan angles (rad). Points not visible from the satellite -> inf."""
    from pyproj import CRS, Transformer

    h = float(projection["perspective_point_height"])
    geos = CRS.from_proj4(
        f"+proj=geos +lon_0={projection['longitude_of_projection_origin']} +h={h} "
        f"+a={projection['semi_major_axis']} +b={projection['semi_minor_axis']} "
        f"+sweep={projection['sweep_angle_axis']} +units=m +no_defs"
    )
    tfm = Transformer.from_crs(CRS.from_epsg(4326), geos, always_xy=True)
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


# =============================================================================
# Byte-range downloads (download_method "byte_ranges"; runs in worker processes)
# =============================================================================
#
# A worker reads the file's HDF5 metadata through h5py, which fetches it piece by piece, then
# fetches only the chunks of the variables that are decoded (for ABI, only chunks overlapping the
# crop). Everything lands in a file-sized buffer whose other bytes stay zero, and the same NetCDF
# decoders as for whole files read that buffer.

GCS_HOST = "storage.googleapis.com"
# The first request fetches both ends of the file, where netCDF-C writes most of the metadata
RANGE_HEAD_BYTES = 80 * 1024
RANGE_TAIL_BYTES = 64 * 1024
# Metadata elsewhere is fetched in aligned blocks of this size when h5py asks for it
RANGE_BLOCK_BYTES = 16 * 1024
# Needed ranges closer together than this are fetched in one request
RANGE_MERGE_GAP_BYTES = 16 * 1024
# Requests one worker makes at once for one file's chunks
RANGE_THREADS_PER_PROCESS = 4

_range_connections = threading.local()  # one keep-alive HTTPS connection per thread
_range_threads: Optional[ThreadPoolExecutor] = None
_crop_rects: Dict[str, Tuple[float, float, float, float]] = {}


def _get_range(path: str, start: int, end: int, timeout: int, retries: int) -> bytes:
    """Bytes [start, end) of a bucket object, over this thread's keep-alive connection; retries transient errors."""
    last_error = "no attempts"
    for attempt in range(1, retries + 1):
        conn = getattr(_range_connections, "conn", None)
        if conn is None:
            conn = _range_connections.conn = http.client.HTTPSConnection(GCS_HOST, timeout=timeout)
        try:
            conn.request("GET", path, headers={"Range": f"bytes={start}-{end - 1}"})
            r = conn.getresponse()
            body = r.read()
            if r.status == 206 and len(body) == end - start:
                return body
            last_error = f"HTTP {r.status} with {len(body)} bytes"
            if r.status != 206 and r.status not in RETRYABLE_HTTP_STATUS:
                break
        except (OSError, http.client.HTTPException) as e:
            last_error = repr(e)
            conn.close()
            _range_connections.conn = None
        if attempt < retries:
            time.sleep(_backoff_seconds(attempt))
    raise RuntimeError(f"GET {path} bytes {start}-{end - 1} failed: {last_error}")


def _range_executor() -> ThreadPoolExecutor:
    global _range_threads
    if _range_threads is None:
        _range_threads = ThreadPoolExecutor(RANGE_THREADS_PER_PROCESS)
    return _range_threads


class _RangeFile(io.RawIOBase):
    """
    A remote file for h5py and the NetCDF decoders: a file-sized buffer holding the byte ranges fetched
    so far, with everything else zero. Reads fetch missing bytes in aligned blocks. The buffer is an
    anonymous memory map, so only fetched pages use memory; call release() when done.
    """

    def __init__(self, path: str, size: int, timeout: int, retries: int):
        if size <= 0:
            raise ValueError(f"{path}: size unknown")
        self.path, self.size, self.timeout, self.retries = path, size, timeout, retries
        self.buf = mmap.mmap(-1, size)
        # Fetched ranges [starts[i], ends[i]), sorted, neither overlapping nor touching
        self.starts: List[int] = []
        self.ends: List[int] = []
        self.pos = 0
        self.downloaded = 0

    def release(self) -> None:
        self.buf.close()

    def _missing(self, start: int, end: int) -> List[Tuple[int, int]]:
        """The parts of [start, end) not fetched yet."""
        pos, end = max(0, start), min(self.size, end)
        out = []
        i = max(0, bisect.bisect_right(self.starts, pos) - 1)
        while pos < end and i < len(self.starts) and self.starts[i] < end:
            if self.ends[i] > pos:
                if self.starts[i] > pos:
                    out.append((pos, self.starts[i]))
                pos = self.ends[i]
            i += 1
        if pos < end:
            out.append((pos, end))
        return out

    def _add(self, start: int, end: int) -> None:
        i = bisect.bisect_left(self.starts, start)
        self.starts.insert(i, start)
        self.ends.insert(i, end)
        starts: List[int] = []
        ends: List[int] = []
        for s, e in zip(self.starts, self.ends):
            if ends and s <= ends[-1]:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        self.starts, self.ends = starts, ends

    def fetch(self, ranges: List[Tuple[int, int]], merge_gap: int = RANGE_MERGE_GAP_BYTES) -> None:
        """Fetches whatever is missing of the [start, end) ranges, merging close ones, in parallel."""
        spans: List[List[int]] = []
        for start, end in sorted(part for r in ranges for part in self._missing(*r)):
            if spans and start - spans[-1][1] < merge_gap:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
        if not spans:
            return

        def get(span: List[int]) -> bytes:
            return _get_range(self.path, span[0], span[1], self.timeout, self.retries)

        bodies = [get(spans[0])] if len(spans) == 1 else list(_range_executor().map(get, spans))
        for (start, end), body in zip(spans, bodies):
            self.buf[start:end] = body
            self._add(start, end)
            self.downloaded += len(body)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if self._missing(self.pos, self.pos + n):
            block = RANGE_BLOCK_BYTES
            self.fetch([(self.pos // block * block, -(-(self.pos + n) // block) * block)], merge_gap=block)
        b[:n] = self.buf[self.pos:self.pos + n]
        self.pos += n
        return n


def _open_remote(
    path: str, size: int, names: List[str], timeout: int, retries: int
) -> Tuple[_RangeFile, Dict[str, Tuple[Optional[Tuple[int, ...]], List[Tuple[Tuple[int, ...], int, int]]]]]:
    """
    Fetches a remote file's metadata and locates the stored chunks of `names`. Returns the file and,
    per name, (chunk shape or None, [(first element index, byte offset, byte size)]).
    """
    import h5py

    rf = _RangeFile(path, size, timeout, retries)
    rf.fetch([(0, RANGE_HEAD_BYTES), (size - RANGE_TAIL_BYTES, size)])
    with h5py.File(rf, "r") as h:
        # netCDF-C reads every object header and attribute when it opens a file, so fetch them all
        def touch(obj: Any) -> None:
            for name in obj.attrs:
                try:
                    obj.attrs[name]
                except Exception:
                    pass  # h5py cannot convert a few attribute types; their bytes are fetched with the header

        touch(h)
        h.visititems(lambda _, obj: touch(obj))
        chunks = {}
        for name in names:
            d = h[name]
            if d.chunks is None:
                offset = d.id.get_offset()  # None for compact data, which is inside the fetched header
                stored = [] if offset is None else [((0,) * d.ndim, offset, d.id.get_storage_size())]
            else:
                stored = [
                    (tuple(c.chunk_offset), c.byte_offset, c.size)
                    for c in (d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks()))
                ]
            chunks[name] = (d.chunks, stored)
    return rf, chunks


def _crop_window(bounds: LatLonBounds, projection: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> Tuple[int, int, int, int]:
    """(y0, y1, x0, x1) the main process will keep, computed the same way as there."""
    key = json.dumps([vars(bounds), projection], sort_keys=True)
    if key not in _crop_rects:
        _crop_rects[key] = scan_angle_rect(bounds, projection)
    x0, x1, y0, y1 = _crop_rects[key]
    ys, xs = crop_slice(y, y0, y1, "y"), crop_slice(x, x0, x1, "x")
    return ys.start, ys.stop, xs.start, xs.stop


def ranged_decode_abi_file(
    path: str, size: int, variables: List[Tuple[str, str, Optional[float]]], bounds: Optional[LatLonBounds], timeout: int, retries: int
) -> Tuple[List[Dict[str, Any]], int]:
    """decode_abi_file on a remote file, fetching only metadata, x/y and the chunks inside the crop. Returns (decoded, bytes downloaded)."""
    import netCDF4 as nc

    data_names = sorted({name for dv, qv, _ in variables for name in (dv, qv)})
    rf, chunks = _open_remote(path, size, ["x", "y", *data_names], timeout, retries)
    try:
        grid = [(offset, offset + n) for name in ("x", "y") for _, offset, n in chunks[name][1]]
        window = None
        if bounds is not None:
            rf.fetch(grid)
            with nc.Dataset("inmem", mode="r", memory=rf.buf) as ds:
                x, y, projection = _abi_grid(ds)
            window = _crop_window(bounds, projection, x, y)
        needed = list(grid)
        for name in data_names:
            chunk_shape, stored = chunks[name]
            for first, offset, n in stored:
                if window is not None and chunk_shape is not None:
                    (cy, cx), (chy, chx) = first[-2:], chunk_shape[-2:]
                    if not (cy < window[1] and cy + chy > window[0] and cx < window[3] and cx + chx > window[2]):
                        continue
                needed.append((offset, offset + n))
        rf.fetch(needed)
        decoded = decode_abi_file(rf.buf, variables, window)
    finally:
        rf.release()
    for dec in decoded:
        dec["window"] = window
    return decoded, rf.downloaded


def ranged_decode_glm_file(path: str, size: int, timeout: int, retries: int) -> Tuple[Dict[str, np.ndarray], int]:
    """decode_glm_file on a remote file, fetching only metadata and GLM_READ_VARIABLES. Returns (decoded, bytes downloaded)."""
    rf, chunks = _open_remote(path, size, GLM_READ_VARIABLES, timeout, retries)
    try:
        rf.fetch([(offset, offset + n) for name in GLM_READ_VARIABLES for _, offset, n in chunks[name][1]])
        decoded = decode_glm_file(rf.buf)
    finally:
        rf.release()
    return decoded, rf.downloaded


# =============================================================================
# Zarr
# =============================================================================

COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)


def _write_array(group: Any, name: str, data: np.ndarray, dims: List[str], **attrs: Any) -> Any:
    """
    Writes a fully populated array with its dimension names for xarray. fill_value=None matters:
    zarr's default fill of 0 would make xarray read every legitimate 0 as missing.
    """
    array = group.create_dataset(name, data=data, fill_value=None, overwrite=True, **attrs.pop("zarr", {}))
    array.attrs.update({"_ARRAY_DIMENSIONS": dims, **attrs})
    return array

GLM_FLASH_COLUMNS = {
    "time_first_ns": np.int64,
    "time_last_ns": np.int64,
    "lat": np.float32,
    "lon": np.float32,
    "area_m2": np.float32,
    "energy_j": np.float32,
    "quality_flag": np.uint8,
    "file_index": np.int32,
}
GLM_EVENT_COLUMNS = {"lat": np.float32, "lon": np.float32, "flash_index": np.int32}


@dataclass
class AbiProductState:
    """Per-slot metadata kept in memory; frames go straight to Zarr as each scan decodes."""

    spec: ProductSpec
    crop: Optional[DayCrop] = None
    value_array: Any = None
    dqf_array: Any = None
    full_shape: Optional[Tuple[int, int]] = None
    ys: slice = slice(None)
    xs: slice = slice(None)
    fill_value: Optional[int] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    scale_factor: np.ndarray = field(default_factory=lambda: np.full(SLOTS_PER_DAY, np.nan))
    add_offset: np.ndarray = field(default_factory=lambda: np.full(SLOTS_PER_DAY, np.nan))
    quantization_step: np.ndarray = field(default_factory=lambda: np.full(SLOTS_PER_DAY, np.nan))
    scan_start_ns: np.ndarray = field(default_factory=lambda: np.full(SLOTS_PER_DAY, NAT_NS, dtype=np.int64))
    scan_end_ns: np.ndarray = field(default_factory=lambda: np.full(SLOTS_PER_DAY, NAT_NS, dtype=np.int64))
    present: np.ndarray = field(default_factory=lambda: np.zeros(SLOTS_PER_DAY, dtype=np.uint8))

    def accept_first_or_validate(self, root: Any, dec: Dict[str, Any]) -> None:
        """Creates the frame arrays from the first decoded scan; later scans must match its grid."""
        full_shape = tuple(dec["value"].shape)
        if self.full_shape is None:
            if self.crop is not None:
                x0, x1, y0, y1 = self.crop.rect_for(dec["projection"])
                self.ys = crop_slice(dec["y"], y0, y1, f"{self.spec.key} y")
                self.xs = crop_slice(dec["x"], x0, x1, f"{self.spec.key} x")
            x, y = dec["x"][self.xs], dec["y"][self.ys]
            shape = (len(y), len(x))
            # Unwritten slots read back as this value; files without a _FillValue get 65535
            storage_fill = dec["fill_value"] if dec["fill_value"] is not None else int(np.iinfo(np.uint16).max)
            g = root.require_group(f"abi/{self.spec.key}")
            g.attrs.update(
                {
                    "data_variable": self.spec.data_variable,
                    "dqf_variable": self.spec.dqf_variable,
                    "units": dec["units"],
                    "fill_value": storage_fill,
                    "dqf_fill_value": dec["dqf_fill_value"],
                    "precision_requested": self.spec.precision,
                    "goes_imager_projection": dec["projection"],
                    "source_filename_must_contain": self.spec.filename_must_contain,
                    "uncropped_shape": list(full_shape),
                    "crop_y0": self.ys.start or 0,
                    "crop_x0": self.xs.start or 0,
                }
            )
            common = dict(overwrite=True, compressor=COMPRESSOR, chunks=(1, shape[0], shape[1]))
            self.value_array = g.create_dataset("value", shape=(SLOTS_PER_DAY, *shape), dtype="u2", fill_value=storage_fill, **common)
            self.dqf_array = g.create_dataset(
                "dqf", shape=(SLOTS_PER_DAY, *shape), dtype=dec["dqf"].dtype, fill_value=dec["dqf_fill_value"], **common
            )
            for array in (self.value_array, self.dqf_array):
                array.attrs.update({"_ARRAY_DIMENSIONS": ["time", "y", "x"], "coordinates": "x_scan_angle_rad y_scan_angle_rad"})
            _write_array(g, "x_scan_angle_rad", x.astype(np.float64), ["x"], units="rad")
            _write_array(g, "y_scan_angle_rad", y.astype(np.float64), ["y"], units="rad")
            self.full_shape, self.fill_value, self.x, self.y = full_shape, dec["fill_value"], dec["x"], dec["y"]
            return

        if full_shape != self.full_shape:
            raise ValueError(f"{self.spec.key}: grid {full_shape} != first scan's grid {self.full_shape}")
        if dec["dqf"].dtype != self.dqf_array.dtype:
            raise ValueError(f"{self.spec.key}: DQF dtype {dec['dqf'].dtype} != first scan's {self.dqf_array.dtype}")
        if dec["fill_value"] != self.fill_value:
            raise ValueError(f"{self.spec.key}: fill value {dec['fill_value']} != {self.fill_value}")
        if not (np.allclose(dec["x"], self.x, atol=1e-7) and np.allclose(dec["y"], self.y, atol=1e-7)):
            raise ValueError(f"{self.spec.key}: scan-angle coordinates differ from the first scan")

    def check_window(self, dec: Dict[str, Any]) -> None:
        """With byte-range downloads, the pixels kept must be exactly the ones the worker downloaded."""
        window = dec.get("window")
        if window is not None and window != (self.ys.start, self.ys.stop, self.xs.start, self.xs.stop):
            raise ValueError(f"{self.spec.key}: downloaded window {window} != crop {(self.ys, self.xs)}")

    def record_scan(self, slot: int, dec: Dict[str, Any], f: RemoteFile) -> None:
        self.scale_factor[slot] = dec["scale_factor"]
        self.add_offset[slot] = dec["add_offset"]
        self.quantization_step[slot] = dec["quantization_step"]
        self.scan_start_ns[slot] = to_ns(f.start)
        self.scan_end_ns[slot] = to_ns(f.end)
        self.present[slot] = 1

    def write_slot_arrays(self, root: Any, slot_start_ns: np.ndarray) -> None:
        g = root.require_group(f"abi/{self.spec.key}")
        _write_array(g, "time", slot_start_ns, ["time"], **TIME_ATTRS)
        _write_array(g, "scan_start_ns", self.scan_start_ns, ["time"], **TIME_ATTRS)
        _write_array(g, "scan_end_ns", self.scan_end_ns, ["time"], **TIME_ATTRS)
        for name, arr in [
            ("scale_factor", self.scale_factor),
            ("add_offset", self.add_offset),
            ("quantization_step", self.quantization_step),
            ("present", self.present),
        ]:
            _write_array(g, name, arr, ["time"])
        g.attrs["n_present"] = int(self.present.sum())
        g.attrs["has_frames"] = self.full_shape is not None


def write_abi_frame(state: AbiProductState, slot: int, dec: Dict[str, Any]) -> None:
    # One chunk per slot, so concurrent writes to different slots never touch the same file.
    state.value_array[slot] = dec["value"][state.ys, state.xs]
    state.dqf_array[slot] = dec["dqf"][state.ys, state.xs]


def write_glm_tables(
    root: Any, day_start: datetime, files: List[RemoteFile], file_ok: np.ndarray, tables: Dict[int, Dict[str, np.ndarray]]
) -> Tuple[int, int, int]:
    """Writes the file, flash, event and coverage tables; returns (n_flashes, n_events, n_windows_covered)."""
    g = root.require_group("glm")

    # Coverage per 20 s window, so outages where NOAA published no file show up as gaps too
    covered = np.zeros(GLM_WINDOWS_PER_DAY, dtype=np.uint8)
    for f, ok in zip(files, file_ok):
        if ok:
            covered[int((f.start - day_start).total_seconds() // GLM_WINDOW_SECONDS)] = 1
    cg = g.require_group("coverage")
    window_start_ns = to_ns(day_start) + np.arange(GLM_WINDOWS_PER_DAY, dtype=np.int64) * GLM_WINDOW_SECONDS * 1_000_000_000
    _write_array(cg, "time", window_start_ns, ["time"], **TIME_ATTRS)
    _write_array(cg, "covered", covered, ["time"])

    n_flashes = np.array([len(tables[i]["lat"]) if i in tables else 0 for i in range(len(files))], dtype=np.int32)
    n_events = np.array([len(tables[i]["event_lat"]) if i in tables else 0 for i in range(len(files))], dtype=np.int32)
    fg = g.require_group("files")
    _write_array(fg, "start_ns", np.array([to_ns(f.start) for f in files], dtype=np.int64), ["file"], **TIME_ATTRS)
    _write_array(fg, "end_ns", np.array([to_ns(f.end) for f in files], dtype=np.int64), ["file"], **TIME_ATTRS)
    _write_array(fg, "ok", file_ok.astype(np.uint8), ["file"])
    _write_array(fg, "n_flashes", n_flashes, ["file"])
    _write_array(fg, "n_events", n_events, ["file"])

    def write_column(group: Any, name: str, arr: np.ndarray, dim: str, **attrs: Any) -> None:
        _write_array(group, name, arr, [dim], **attrs, zarr=dict(chunks=(max(1, min(len(arr), 1 << 20)),), compressor=COMPRESSOR))

    # Flashes, sorted by the time of their first event
    file_indices = sorted(tables)
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in GLM_FLASH_COLUMNS}
    for i in file_indices:
        t = tables[i]
        for k in GLM_FLASH_COLUMNS:
            cols[k].append(np.full(len(t["lat"]), i, dtype=np.int32) if k == "file_index" else t[k])

    flash_group = g.require_group("flashes")
    merged = {k: np.concatenate(v).astype(GLM_FLASH_COLUMNS[k]) if v else np.zeros(0, GLM_FLASH_COLUMNS[k]) for k, v in cols.items()}
    order = np.argsort(merged["time_first_ns"], kind="stable")
    for k, arr in merged.items():
        write_column(flash_group, k, arr[order], "flash", **(TIME_ATTRS if k.startswith("time_") else {}))

    # Events in file order; flash_index is the row of the event's flash in glm/flashes
    sorted_row = np.empty(len(order), dtype=np.int32)
    sorted_row[order] = np.arange(len(order), dtype=np.int32)
    first_row = np.concatenate(([0], np.cumsum([len(tables[i]["lat"]) for i in file_indices]))).astype(np.int64)
    event_group = g.require_group("events")
    for k, dtype in GLM_EVENT_COLUMNS.items():
        if k == "flash_index":
            parts = [sorted_row[first_row[j] + tables[i]["event_flash"]] for j, i in enumerate(file_indices)]
        else:
            parts = [tables[i][f"event_{k}"] for i in file_indices]
        write_column(event_group, k, np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype), "event")
    return int(n_flashes.sum()), int(n_events.sum()), int(covered.sum())


# =============================================================================
# Per-day processing
# =============================================================================

async def process_single_day(
    cfg: RawConfig,
    day: datetime,
    session: aiohttp.ClientSession,
    decode_pool: DecodePool,
) -> bool:
    day_label = f"{day.date().isoformat()} ({day.year}{day.timetuple().tm_yday:03d})"
    out_path = output_zarr_path_for_day(cfg.raw_output_root_dir, day)
    if not cfg.overwrite_existing and zarr_store_is_complete(out_path):
        log(f"SKIP {day_label} already complete")
        return True

    day_start = day
    day_end = day + timedelta(days=1)

    # --- listing: each bucket prefix once, shared by every product that reads from it
    log(f"=== {day_label} listing ===")
    specs = cfg.abi_products + ([cfg.glm_product] if cfg.glm_product is not None else [])
    prefixes = sorted({spec.gcs_product_prefix for spec in specs})
    listings = await asyncio.gather(*(list_day_names(cfg, session, p, day) for p in prefixes))
    objects_by_prefix = {p: objects for p, (objects, _) in zip(prefixes, listings)}
    n_listing_errors = sum(n for _, n in listings)

    def day_files(spec: ProductSpec) -> List[RemoteFile]:
        files = select_files(objects_by_prefix[spec.gcs_product_prefix], spec.filename_must_contain)
        return [f for f in files if day_start <= f.start < day_end]

    glm_files = day_files(cfg.glm_product) if cfg.glm_product is not None else []

    # --- ABI slot assignment: slot = 5-min window containing the scan start. Products that read
    # the same files (e.g. the DSI indices) are grouped so each file is downloaded once.
    crop = DayCrop(cfg.bounds) if cfg.bounds is not None else None
    abi_states = {spec.key: AbiProductState(spec, crop) for spec in cfg.abi_products}
    same_files: Dict[Tuple[str, str], List[AbiProductState]] = {}
    for spec in cfg.abi_products:
        same_files.setdefault((spec.gcs_product_prefix, spec.filename_must_contain), []).append(abi_states[spec.key])

    abi_items: List[Tuple[List[AbiProductState], int, RemoteFile]] = []
    n_slot_collisions = 0
    for states in same_files.values():
        taken: Dict[int, RemoteFile] = {}
        for f in day_files(states[0].spec):
            slot = int((f.start - day_start).total_seconds() // SLOT_SECONDS)
            if slot in taken:
                n_slot_collisions += 1
                keys = "/".join(st.spec.key for st in states)
                log(f"!! {keys}: two scans in slot {slot}; keeping {os.path.basename(taken[slot].blob)}")
                continue
            taken[slot] = f
            abi_items.append((states, slot, f))

    log(
        f"{day_label}: ABI scans "
        + ", ".join(f"{k}={sum(1 for states, _, _ in abi_items for st in states if st.spec.key == k)}" for k in abi_states)
        + f"; ABI files={len(abi_items)}; GLM files={len(glm_files)}; listing errors={n_listing_errors}"
    )
    progress_key = day.date().isoformat()
    if _PROGRESS is not None:
        _PROGRESS.start_day(progress_key, sum(f.size for _, _, f in abi_items) + sum(f.size for f in glm_files))

    # --- store (built at *.tmp and moved into place when done)
    tmp_path = out_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(zarr.DirectoryStore(tmp_path), mode="w")
    root.attrs.update(
        {
            "day_utc": day.date().isoformat(),
            "year": day.year,
            "julian_day": day.timetuple().tm_yday,
            "slots_per_day": SLOTS_PER_DAY,
            "slot_seconds": SLOT_SECONDS,
            "slot_rule": "ABI scan goes to slot floor((scan_start - day_start) / slot_seconds)",
            "time_encoding": "int64 ns since 1970-01-01 UTC; int64 min = NaT",
            "source_gcs_bucket": cfg.gcs_bucket,
            "source_satellite_label": infer_satellite_label_from_bucket(cfg.gcs_bucket),
            "abi_product_keys": [p.key for p in cfg.abi_products],
            "glm_included": cfg.glm_product is not None,
            "bounds": vars(cfg.bounds) if cfg.bounds is not None else None,
            "complete": False,
        }
    )

    # --- download + decode + write, bounded by max_inflight_payloads
    inflight = asyncio.Semaphore(max(1, cfg.max_inflight_payloads))
    glm_ok = np.zeros(len(glm_files), dtype=bool)
    glm_tables: Dict[int, Dict[str, np.ndarray]] = {}
    errors: List[str] = []

    def record_error(blob: str, e: BaseException) -> None:
        errors.append(blob)
        if len(errors) <= MAX_ERRORS_LOGGED_PER_DAY:
            log(f"!! {os.path.basename(blob)}: {e!r}")

    byte_ranges = cfg.download_method == "byte_ranges"
    http_args = (cfg.http_timeout_seconds, cfg.http_retries)

    async def download(blob: str) -> bytes:
        payload = await http_get(cfg, session, f"https://{GCS_HOST}/{cfg.gcs_bucket}/{blob}", as_json=False)
        if _PROGRESS is not None:
            _PROGRESS.add_bytes(progress_key, len(payload))
        return payload

    async def run_ranged(f: RemoteFile, fn: Any, *args: Any) -> Any:
        decoded, n_downloaded = await decode_pool.run(fn, f"/{cfg.gcs_bucket}/{f.blob}", f.size, *args, *http_args)
        if _PROGRESS is not None:
            _PROGRESS.add_bytes(progress_key, f.size, n_downloaded)
        return decoded

    async def run_abi(states: List[AbiProductState], slot: int, f: RemoteFile) -> None:
        async with inflight:
            try:
                variables = [(st.spec.data_variable, st.spec.dqf_variable, st.spec.precision) for st in states]
                if byte_ranges:
                    decoded = await run_ranged(f, ranged_decode_abi_file, variables, cfg.bounds)
                else:
                    payload = await download(f.blob)
                    decoded = await decode_pool.run(decode_abi_file, payload, variables)
                    del payload
                # Validate every product before writing any, so a bad file leaves no half-written slot
                for state, dec in zip(states, decoded):
                    state.accept_first_or_validate(root, dec)
                    state.check_window(dec)
                for state, dec in zip(states, decoded):
                    await asyncio.to_thread(write_abi_frame, state, slot, dec)
                    state.record_scan(slot, dec, f)
            except Exception as e:
                record_error(f.blob, e)

    async def run_glm(i: int, f: RemoteFile) -> None:
        async with inflight:
            try:
                if byte_ranges:
                    glm_tables[i] = await run_ranged(f, ranged_decode_glm_file)
                else:
                    glm_tables[i] = await decode_pool.run(decode_glm_file, await download(f.blob))
                glm_ok[i] = True
            except Exception as e:
                record_error(f.blob, e)

    # Files start in scan-time order, so small GLM files (latency-bound with byte ranges) overlap the
    # larger ABI files (bandwidth-bound) instead of all running after them
    jobs = [(f.start, run_abi(states, slot, f)) for states, slot, f in abi_items]
    jobs += [(f.start, run_glm(i, f)) for i, f in enumerate(glm_files)]
    jobs.sort(key=lambda job: job[0])
    await asyncio.gather(*(job for _, job in jobs))

    # --- per-slot metadata, GLM tables, completion
    slot_start_ns = to_ns(day_start) + np.arange(SLOTS_PER_DAY, dtype=np.int64) * SLOT_SECONDS * 1_000_000_000
    for state in abi_states.values():
        state.write_slot_arrays(root, slot_start_ns)

    if crop is not None and crop.rect is not None:
        root.attrs["crop_scan_angle_rect"] = list(crop.rect)

    n_flashes, n_events, n_windows_covered = 0, 0, 0
    if cfg.glm_product is not None:
        n_flashes, n_events, n_windows_covered = write_glm_tables(root, day_start, glm_files, glm_ok, glm_tables)

    too_recent = datetime.now(timezone.utc) < day_end + MIN_AGE_FOR_COMPLETE
    complete = n_listing_errors == 0 and not errors and not too_recent
    root.attrs.update(
        {
            "n_listing_errors": n_listing_errors,
            "n_file_errors": len(errors),
            "n_abi_slot_collisions": n_slot_collisions,
            "n_glm_files": len(glm_files),
            "n_glm_flashes": n_flashes,
            "n_glm_events": n_events,
            "n_glm_windows_missing": GLM_WINDOWS_PER_DAY - n_windows_covered if cfg.glm_product is not None else None,
            "too_recent": too_recent,
            "complete": complete,
            "completed_utc": datetime.now(timezone.utc).isoformat() if complete else None,
        }
    )

    # One metadata file per day, so opening many days with xarray is fast
    zarr.consolidate_metadata(root.store)

    shutil.rmtree(out_path, ignore_errors=True)
    shutil.move(tmp_path, out_path)
    if _PROGRESS is not None:
        _PROGRESS.add_saved(await asyncio.to_thread(directory_size, out_path))

    if complete:
        status = "OK"
    elif too_recent and n_listing_errors == 0 and not errors:
        status = "INCOMPLETE (ended less than 24 h ago; NOAA may still publish files, rerun later)"
    else:
        status = f"INCOMPLETE (listing errors={n_listing_errors}, file errors={len(errors)}; rerun to retry)"
    gaps = f"  GLM windows missing={GLM_WINDOWS_PER_DAY - n_windows_covered}" if cfg.glm_product is not None else ""
    log(f"-> {day_label} {status}: {out_path}  flashes={n_flashes} events={n_events}{gaps}")
    return complete


# =============================================================================
# Runner
# =============================================================================

async def run_all_days(cfg: RawConfig) -> None:
    days = list(iter_days_utc(cfg.start_date_utc, cfg.end_date_utc))
    os.makedirs(cfg.raw_output_root_dir, exist_ok=True)
    if cfg.download_method == "whole_files" and any("MCMIP" in p.gcs_product_prefix for p in cfg.abi_products):
        log("!! whole_files downloads each all-band MCMIP file in full (~59 MB per scan); byte_ranges fetches only the bands used")

    connector = aiohttp.TCPConnector(
        limit=cfg.max_concurrent_http_requests,
        ttl_dns_cache=300,
    )
    day_queue: asyncio.Queue[datetime] = asyncio.Queue()
    for d in days:
        day_queue.put_nowait(d)

    ok: List[str] = []
    failed: List[str] = []

    global _PROGRESS
    progress = _PROGRESS = Progress(len(days))

    async def redraw() -> None:
        while True:
            progress.draw()
            await asyncio.sleep(1.0)

    redraw_task = asyncio.create_task(redraw())
    async with aiohttp.ClientSession(connector=connector) as session:
        byte_ranges = cfg.download_method == "byte_ranges"
        pool = DecodePool(cfg.byte_range_processes if byte_ranges else cfg.max_decode_processes)
        try:
            async def worker() -> None:
                while True:
                    try:
                        day = day_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    label = day.date().isoformat()
                    try:
                        good = await process_single_day(cfg, day, session, pool)
                    except Exception as e:
                        log(f"!! {label} failed: {e!r}")
                        good = False
                    finally:
                        progress.finish_day(label)
                    (ok if good else failed).append(label)

            await asyncio.gather(*(worker() for _ in range(max(1, cfg.max_days_in_flight))))
        finally:
            pool.shutdown()
            redraw_task.cancel()
            progress.close()
            _PROGRESS = None

    elapsed = time.monotonic() - progress.started
    print("\n" + "=" * 72)
    print(f"Days attempted:  {len(days)}")
    print(f"Days complete:   {len(ok)}")
    print(f"Days incomplete: {len(failed)}" + (f"  {sorted(failed)}" if failed else ""))
    downloaded = progress.bytes_downloaded
    print(f"Downloaded:      {downloaded / 1e9:.1f} GB in {_format_duration(elapsed)}"
          f" ({downloaded / 1e6 / max(elapsed, 1e-9):.0f} MB/s average)")
    if cfg.download_method == "byte_ranges":
        print(f"Source files:    {progress.bytes_done / 1e9:.1f} GB ({100 * downloaded / max(progress.bytes_done, 1):.0f}% downloaded)")
    print(f"Saved to disk:   {progress.bytes_saved / 1e9:.1f} GB"
          + (f" ({100 * progress.bytes_saved / downloaded:.0f}% of downloaded)" if downloaded else ""))
    print(f"Output root:     {os.path.abspath(cfg.raw_output_root_dir)}")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Download GOES ABI + GLM into per-day raw Zarr stores.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    args = p.parse_args()
    asyncio.run(run_all_days(load_config(args.config)))


if __name__ == "__main__":
    main()
