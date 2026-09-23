"""
generate_dataset.py

GOES ABI (CONUS L2) + GLM (LCFA) -> CONFIGURABLE bin aggregation (N x 5-min bins)
================================================================================

This script produces a per-day Zarr store containing:

ABI outputs (dynamic products from config):
  abi/product_value[time_bin, abi_product, y, x]             float32 (NaN where no valid samples)
  abi/valid_pixel_fraction[time_bin, abi_product, y, x]      float32 = valid_pixels / expected_pixels
  abi/bin_has_decoded_file[time_bin, abi_product]            uint8 (0/1)

GLM outputs:
  glm/flash_count[time_bin, y, x]                            float32 (sum of rasterized flash counts across files)
  glm/bin_has_decoded_file[time_bin]                         uint8 (0/1)

Data-quality / accounting outputs:
Per ABI product AND bin AND output pixel:
  validation/abi_expected_pixels[time_bin, abi_product, y, x]        int32
  validation/abi_valid_pixels[time_bin, abi_product, y, x]           int32   (DQF==0 raw samples summed into coarse blocks)
  validation/abi_valid_pixel_fraction[time_bin, abi_product, y, x]   float32 = valid_pixels / expected_pixels
  validation/abi_actual_pixels[time_bin, abi_product, y, x]          int32   (decoded_ok_files * f*f), constant per pixel
  validation/abi_actual_pixel_fraction[time_bin, abi_product, y, x]  float32 = valid_pixels / actual_pixels (0 if actual_pixels==0)

Per ABI product AND bin (file accounting):
  validation/abi_expected_files[time_bin, abi_product]               int32
  validation/abi_actual_valid_files[time_bin, abi_product]           int32   (decoded_ok files)
  validation/abi_actual_over_expected_files[time_bin, abi_product]   float32 = actual_valid / expected
  validation/abi_expected_over_actual_files[time_bin, abi_product]   float32 = expected / actual_valid (inf if 0)

Per GLM bin (file accounting):
  validation/glm_expected_files[time_bin]                            int32
  validation/glm_actual_valid_files[time_bin]                        int32
  validation/glm_actual_over_expected_files[time_bin]                float32
  validation/glm_expected_over_actual_files[time_bin]                float32

Strict bin assignment:
  file goes to bin k iff file_start >= bin_start AND file_end < bin_end

ABI aggregation rule:
For each bin and ABI product:
  - Decode each qualifying file to raw contribution:
      raw_sum_valid = value where DQF==0 else 0
      raw_count_valid = 1 where DQF==0 else 0
  - Sum raw_sum_valid and raw_count_valid across files in the bin.
  - Raw mean per pixel = raw_sum / raw_count where raw_count>0 else NaN.
  - Coarsen ONCE by mean ignoring NaNs.

valid_pixel_fraction definition:
  expected_pixels = expected_files_per_bin * (f*f)
  valid_pixels    = number of valid (DQF==0) raw samples contributing to that output pixel (summed into coarse blocks)
  valid_pixel_fraction = valid_pixels / expected_pixels

GLM aggregation:
  - Rasterize flash counts to ABI coarse grid for each GLM file; sum across files in bin.

Config:
  python generate_dataset.py --config path/to/config.json

Deps:
  pip install aiohttp netCDF4 numpy zarr pyproj
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import zarr
from concurrent.futures import ProcessPoolExecutor


# =============================================================================
# Config models + loading
# =============================================================================

@dataclass(frozen=True)
class RemoteProductSpec:
    key: str
    gcs_product_prefix: str
    filename_must_contain: str
    data_variable: Optional[str] = None
    dqf_variable: Optional[str] = None


@dataclass(frozen=True)
class PipelineConfig:
    # Data source
    gcs_bucket: str

    # Date range (inclusive, UTC)
    start_date_utc: str
    end_date_utc: str

    # Bin aggregation: number of 5-min bins per output bin
    number_of_5min_bins_per_bin: int

    # Coarsen factor f (block size fxf); 1 means no coarsen
    coarsen_factor: int

    # Listing + download concurrency
    max_concurrent_http_requests: int
    http_timeout_seconds: int
    http_retries: int
    tcp_limit_per_host: int
    max_concurrent_list_requests: int

    # CPU decode workers
    max_decode_processes: int

    # Concurrency across days + writer window
    max_days_in_flight: int
    time_bin_write_window: int

    # Output
    output_root_dir: str
    overwrite_existing: bool

    # Zarr chunking
    zarr_time_chunk: int
    zarr_chunk_y: int
    zarr_chunk_x: int

    # Warnings
    warn_if_write_slower_than_seconds: float

    # Products
    abi_products: List[RemoteProductSpec]
    glm_product: RemoteProductSpec

    # Expectations per 5-min bin
    expected_abi_files_per_5min: int
    expected_glm_files_per_5min: int

    # NEW: decoupled pipeline backpressure
    # Caps how many downloaded payloads are allowed to exist "in memory and not yet fully decoded".
    max_inflight_payloads: int


def _require(d: Dict[str, Any], k: str, typ: Any):
    if k not in d:
        raise ValueError(f"Config missing required key: {k!r}")
    v = d[k]
    if typ is not None and not isinstance(v, typ):
        raise ValueError(f"Config key {k!r} must be {typ}, got {type(v)}")
    return v


def load_config(config_path: str) -> PipelineConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Top-level required
    gcs_bucket = _require(raw, "gcs_bucket", str)
    start_date_utc = _require(raw, "start_date_utc", str)
    end_date_utc = _require(raw, "end_date_utc", str)

    number_of_5min_bins_per_bin = int(_require(raw, "number_of_5min_bins_per_bin", int))
    coarsen_factor = int(raw.get("coarsen_factor", 1) or 1)
    if coarsen_factor < 1:
        raise ValueError("coarsen_factor must be >= 1")

    # Network/concurrency defaults
    max_concurrent_http_requests = int(raw.get("max_concurrent_http_requests", 32))
    http_timeout_seconds = int(raw.get("http_timeout_seconds", 60))
    http_retries = int(raw.get("http_retries", 3))
    tcp_limit_per_host = int(raw.get("tcp_limit_per_host", 96))
    max_concurrent_list_requests = int(raw.get("max_concurrent_list_requests", 48))

    # CPU / pipeline
    max_decode_processes = int(raw.get("max_decode_processes", 12))
    max_days_in_flight = int(raw.get("max_days_in_flight", 4))
    time_bin_write_window = int(raw.get("time_bin_write_window", 24))

    # Output
    output_root_dir = str(raw.get("output_root_dir", "./goes_abi_glm"))
    overwrite_existing = bool(raw.get("overwrite_existing", False))

    # Zarr chunking
    zarr_time_chunk = int(raw.get("zarr_time_chunk", 16))
    zarr_chunk_y = int(raw.get("zarr_chunk_y", 256))
    zarr_chunk_x = int(raw.get("zarr_chunk_x", 256))

    warn_if_write_slower_than_seconds = float(raw.get("warn_if_write_slower_than_seconds", 20.0))

    # Expectations
    expected_abi_files_per_5min = int(raw.get("expected_abi_files_per_5min", 1))
    expected_glm_files_per_5min = int(raw.get("expected_glm_files_per_5min", 15))
    if expected_abi_files_per_5min < 0 or expected_glm_files_per_5min < 0:
        raise ValueError("expected_*_files_per_5min must be >= 0")

    # NEW: inflight payload cap (default: keep small to avoid memory spikes)
    max_inflight_payloads = int(raw.get("max_inflight_payloads", max(8, 2 * max_decode_processes)))

    # ABI products (dynamic)
    abi_products_raw = _require(raw, "abi_products", list)
    abi_products: List[RemoteProductSpec] = []
    for i, p in enumerate(abi_products_raw):
        if not isinstance(p, dict):
            raise ValueError(f"abi_products[{i}] must be an object")
        key = _require(p, "key", str)
        gcs_product_prefix = _require(p, "gcs_product_prefix", str)
        filename_must_contain = _require(p, "filename_must_contain", str)
        data_variable = p.get("data_variable")
        dqf_variable = p.get("dqf_variable")
        if data_variable is None or dqf_variable is None:
            raise ValueError(f"abi_products[{i}] must include data_variable and dqf_variable")
        abi_products.append(
            RemoteProductSpec(
                key=key,
                gcs_product_prefix=gcs_product_prefix,
                filename_must_contain=filename_must_contain,
                data_variable=str(data_variable),
                dqf_variable=str(dqf_variable),
            )
        )

    if not abi_products:
        raise ValueError("Config must include at least one ABI product in abi_products")

    # GLM product
    glm_raw = _require(raw, "glm_product", dict)
    glm_product = RemoteProductSpec(
        key=str(_require(glm_raw, "key", str)),
        gcs_product_prefix=str(_require(glm_raw, "gcs_product_prefix", str)),
        filename_must_contain=str(_require(glm_raw, "filename_must_contain", str)),
        data_variable=None,
        dqf_variable=None,
    )

    # Basic validation for date strings
    datetime.strptime(start_date_utc, "%Y-%m-%d")
    datetime.strptime(end_date_utc, "%Y-%m-%d")

    return PipelineConfig(
        gcs_bucket=gcs_bucket,
        start_date_utc=start_date_utc,
        end_date_utc=end_date_utc,
        number_of_5min_bins_per_bin=number_of_5min_bins_per_bin,
        coarsen_factor=coarsen_factor,
        max_concurrent_http_requests=max_concurrent_http_requests,
        http_timeout_seconds=http_timeout_seconds,
        http_retries=http_retries,
        tcp_limit_per_host=tcp_limit_per_host,
        max_concurrent_list_requests=max_concurrent_list_requests,
        max_decode_processes=max_decode_processes,
        max_days_in_flight=max_days_in_flight,
        time_bin_write_window=time_bin_write_window,
        output_root_dir=output_root_dir,
        overwrite_existing=overwrite_existing,
        zarr_time_chunk=zarr_time_chunk,
        zarr_chunk_y=zarr_chunk_y,
        zarr_chunk_x=zarr_chunk_x,
        warn_if_write_slower_than_seconds=warn_if_write_slower_than_seconds,
        abi_products=abi_products,
        glm_product=glm_product,
        expected_abi_files_per_5min=expected_abi_files_per_5min,
        expected_glm_files_per_5min=expected_glm_files_per_5min,
        max_inflight_payloads=max_inflight_payloads,
    )


# =============================================================================
# Constants + helpers
# =============================================================================

BASE_5MIN_BINS_PER_DAY = 288
BASE_BIN_SECONDS = 300  # 5 minutes

GOES_FILENAME_START_TIME_PATTERN = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)?")
GOES_FILENAME_END_TIME_PATTERN = re.compile(r"_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)?")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_directory_tree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def zarr_store_exists(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".zgroup"))


def zarr_store_is_complete(path: str) -> bool:
    if not zarr_store_exists(path):
        return False
    try:
        root = zarr.open_group(path, mode="r")
        return bool(root.attrs.get("complete", False))
    except Exception:
        return False


def mark_zarr_store_complete(root) -> None:
    root.attrs["complete"] = True
    root.attrs["completed_utc"] = datetime.now(timezone.utc).isoformat()


def output_zarr_path_for_day(output_root: str, year: int, julian_day: int) -> str:
    d = os.path.join(output_root, "merged", str(year))
    ensure_dir(d)
    return os.path.join(d, f"{year}{julian_day:03d}.zarr")


def infer_satellite_label_from_bucket(bucket: str) -> str:
    m = re.search(r"goes-(\d+)", bucket)
    if not m:
        return "UNKNOWN"
    return f"GOES-{int(m.group(1))}"


def bins_per_day(cfg: PipelineConfig) -> int:
    k = int(cfg.number_of_5min_bins_per_bin)
    if k < 1:
        raise ValueError("number_of_5min_bins_per_bin must be >= 1")
    if BASE_5MIN_BINS_PER_DAY % k != 0:
        raise ValueError(
            f"number_of_5min_bins_per_bin={k} must divide {BASE_5MIN_BINS_PER_DAY} exactly "
            f"(valid examples: 1,2,3,4,6,8,9,12,16,18,24,32,36,48,72,96,144,288)"
        )
    return BASE_5MIN_BINS_PER_DAY // k


def bin_seconds(cfg: PipelineConfig) -> int:
    return BASE_BIN_SECONDS * int(cfg.number_of_5min_bins_per_bin)


def expected_abi_files_per_bin(cfg: PipelineConfig) -> int:
    return int(cfg.number_of_5min_bins_per_bin) * int(cfg.expected_abi_files_per_5min)


def expected_glm_files_per_bin(cfg: PipelineConfig) -> int:
    return int(cfg.number_of_5min_bins_per_bin) * int(cfg.expected_glm_files_per_5min)


def iter_days_utc(start_date: str, end_date: str):
    a = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    b = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)


def utc_day_start(year: int, julian_day: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=julian_day - 1)


def build_bin_start_times_ns(cfg: PipelineConfig, year: int, julian_day: int) -> np.ndarray:
    n_bins = bins_per_day(cfg)
    d0 = utc_day_start(year, julian_day)
    base = np.datetime64(d0.replace(tzinfo=None), "ns")
    return base + (np.arange(n_bins, dtype=np.int64) * np.timedelta64(bin_seconds(cfg), "s"))


def _datetime_utc_from_parts(
    year: int, jday: int, hh: int, mm: int, ss: int, tenth: Optional[int]
) -> datetime:
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=jday - 1, hours=hh, minutes=mm, seconds=ss
    )
    if tenth is not None:
        base += timedelta(milliseconds=100 * int(tenth))
    return base


def parse_filename_scan_start_time_utc(blob_name: str) -> Optional[datetime]:
    m = GOES_FILENAME_START_TIME_PATTERN.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss, tenth = m.groups()
    return _datetime_utc_from_parts(int(y), int(j), int(hh), int(mm), int(ss), int(tenth) if tenth else None)


def parse_filename_scan_end_time_utc(blob_name: str) -> Optional[datetime]:
    m = GOES_FILENAME_END_TIME_PATTERN.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss, tenth = m.groups()
    return _datetime_utc_from_parts(int(y), int(j), int(hh), int(mm), int(ss), int(tenth) if tenth else None)


# =============================================================================
# HTTP listing
# =============================================================================

async def _gcs_list_object_names_http(
    cfg: PipelineConfig,
    session: aiohttp.ClientSession,
    bucket: str,
    prefix: str,
    list_request_semaphore: asyncio.Semaphore,
) -> List[str]:
    out: List[str] = []
    page_token: Optional[str] = None

    base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_seconds)
    params: Dict[str, str] = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}

    while True:
        if page_token:
            params["pageToken"] = page_token
        else:
            params.pop("pageToken", None)

        async with list_request_semaphore:
            try:
                async with session.get(base_url, params=params, timeout=timeout) as r:
                    if r.status != 200:
                        return out
                    js = await r.json()
            except Exception:
                return out

        for it in js.get("items", []):
            n = it.get("name")
            if n:
                out.append(n)

        page_token = js.get("nextPageToken")
        if not page_token:
            break

    return out


async def list_day_objects_for_product(
    cfg: PipelineConfig,
    session: aiohttp.ClientSession,
    product_spec: RemoteProductSpec,
    year: int,
    julian_day: int,
    list_request_semaphore: asyncio.Semaphore,
) -> List[Tuple[datetime, datetime, str]]:
    prefixes = [f"{product_spec.gcs_product_prefix}/{year}/{julian_day:03d}/{hh:02d}/" for hh in range(24)]
    tasks = [
        asyncio.create_task(_gcs_list_object_names_http(cfg, session, cfg.gcs_bucket, pfx, list_request_semaphore))
        for pfx in prefixes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    names: List[str] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        names.extend(res)

    out: List[Tuple[datetime, datetime, str]] = []
    for n in names:
        if product_spec.filename_must_contain not in n:
            continue
        ts = parse_filename_scan_start_time_utc(n)
        te = parse_filename_scan_end_time_utc(n)
        if ts is None or te is None:
            continue
        out.append((ts, te, n))

    out.sort(key=lambda x: x[0])
    return out


# =============================================================================
# Strict bin assignment
# =============================================================================

def _bin_index_for_time(cfg: PipelineConfig, day_start_utc: datetime, file_start: datetime) -> int:
    return int(((file_start - day_start_utc).total_seconds()) // float(bin_seconds(cfg)))


def _bin_start_end(cfg: PipelineConfig, day_start_utc: datetime, bin_index: int) -> Tuple[datetime, datetime]:
    t0 = day_start_utc + timedelta(seconds=bin_seconds(cfg) * bin_index)
    t1 = t0 + timedelta(seconds=bin_seconds(cfg))
    return t0, t1


def group_files_into_bins_strict(
    cfg: PipelineConfig,
    day_start_utc: datetime,
    file_time_triples: List[Tuple[datetime, datetime, str]],
) -> List[List[str]]:
    n_bins = bins_per_day(cfg)
    blobs_for_bin: List[List[str]] = [[] for _ in range(n_bins)]

    for file_start, file_end, blob in file_time_triples:
        bi = _bin_index_for_time(cfg, day_start_utc, file_start)
        if bi < 0 or bi >= n_bins:
            continue
        t0, t1 = _bin_start_end(cfg, day_start_utc, bi)
        if file_start >= t0 and file_end < t1:
            blobs_for_bin[bi].append(blob)

    return blobs_for_bin


# =============================================================================
# Downloads
# =============================================================================

async def download_gcs_object_bytes(cfg: PipelineConfig, session: aiohttp.ClientSession, blob_name: str) -> Optional[bytes]:
    url = f"https://storage.googleapis.com/{cfg.gcs_bucket}/{blob_name}"
    timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_seconds)

    for attempt in range(1, cfg.http_retries + 1):
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    return await r.read()
        except Exception:
            pass
        await asyncio.sleep(min(0.25 * attempt, 1.0))

    return None


# =============================================================================
# Array helpers
# =============================================================================

def pad_or_crop_to_shape_2d(a: np.ndarray, target_y: int, target_x: int, *, fill: float) -> np.ndarray:
    if a.shape == (target_y, target_x):
        return a
    out = np.full((target_y, target_x), fill, dtype=a.dtype)
    yy = min(target_y, a.shape[0])
    xx = min(target_x, a.shape[1])
    out[:yy, :xx] = a[:yy, :xx]
    return out


def squeeze_to_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.squeeze(np.asarray(a))
    if a.ndim != 2:
        raise ValueError(f"{name} not 2D after squeeze: shape={a.shape}")
    return a


def squeeze_to_1d(v: np.ndarray, name: str) -> np.ndarray:
    v = np.squeeze(np.asarray(v))
    if v.ndim != 1:
        raise ValueError(f"{name} not 1D after squeeze: shape={v.shape}")
    return v


# =============================================================================
# ABI projection probe + coarsen vectors
# =============================================================================

def read_goes_imager_projection_attributes(payload: bytes) -> Dict[str, float | str]:
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        if "goes_imager_projection" not in ds.variables:
            raise KeyError("ABI file missing goes_imager_projection variable")
        p = ds.variables["goes_imager_projection"]

        def _need(attr_name: str):
            if not hasattr(p, attr_name):
                raise KeyError(f"goes_imager_projection missing attribute {attr_name!r}")
            return getattr(p, attr_name)

        return {
            "projection_origin_longitude_deg": float(_need("longitude_of_projection_origin")),
            "perspective_point_height_m": float(_need("perspective_point_height")),
            "semi_major_axis_m": float(_need("semi_major_axis")),
            "semi_minor_axis_m": float(_need("semi_minor_axis")),
            "sweep_angle_axis": str(_need("sweep_angle_axis")),
        }


def probe_abi_reference_grid_and_projection(payload: bytes, data_variable: str, dqf_variable: str):
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        data = squeeze_to_2d(ds.variables[data_variable][:], f"probe:{data_variable}").astype(np.float32, copy=False)

        dqf = np.squeeze(np.asarray(ds.variables[dqf_variable][:]))
        if dqf.ndim == 0:
            dqf = np.full(data.shape, int(dqf.item()), dtype=np.int16)
        else:
            dqf = squeeze_to_2d(dqf, f"probe:{dqf_variable}").astype(np.int16, copy=False)

        if dqf.shape != data.shape:
            raise ValueError(f"probe dqf shape {dqf.shape} != data shape {data.shape}")

        x_scan_angle_rad = squeeze_to_1d(ds.variables["x"][:], "probe:x").astype(np.float64, copy=False)
        y_scan_angle_rad = squeeze_to_1d(ds.variables["y"][:], "probe:y").astype(np.float64, copy=False)

    raw_y, raw_x = int(data.shape[0]), int(data.shape[1])
    geos_projection = read_goes_imager_projection_attributes(payload)
    return raw_y, raw_x, x_scan_angle_rad, y_scan_angle_rad, geos_projection


def build_coarsened_scan_angle_vectors(x_raw: np.ndarray, y_raw: np.ndarray, f: int):
    if f == 1:
        return x_raw.astype(np.float64, copy=False), y_raw.astype(np.float64, copy=False)

    def trim(v: np.ndarray):
        n2 = (v.shape[0] // f) * f
        return v[:n2]

    x = trim(x_raw)
    y = trim(y_raw)
    x_c = x.reshape(x.shape[0] // f, f).mean(axis=1)
    y_c = y.reshape(y.shape[0] // f, f).mean(axis=1)
    return x_c.astype(np.float64), y_c.astype(np.float64)


# =============================================================================
# ABI decode (raw contrib) + coarsen
# =============================================================================

def _read_abi_raw_data_and_valid_mask(
    payload: bytes,
    data_variable: str,
    dqf_variable: str,
    reference_raw_y: int,
    reference_raw_x: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (data_raw, valid_mask_raw) both shaped (reference_raw_y, reference_raw_x).
    valid_mask_raw True where DQF==0.
    """
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        data = squeeze_to_2d(ds.variables[data_variable][:], data_variable).astype(np.float32, copy=False)

        dqf = np.squeeze(np.asarray(ds.variables[dqf_variable][:]))
        if dqf.ndim == 0:
            dqf = np.full(data.shape, int(dqf.item()), dtype=np.int16)
        else:
            dqf = squeeze_to_2d(dqf, dqf_variable).astype(np.int16, copy=False)

        if dqf.shape != data.shape:
            if dqf.size == 1:
                dqf = np.full(data.shape, int(dqf.reshape(-1)[0]), dtype=np.int16)
            else:
                raise ValueError(f"{dqf_variable} shape {dqf.shape} != {data_variable} shape {data.shape}")

    data = pad_or_crop_to_shape_2d(data, reference_raw_y, reference_raw_x, fill=np.nan)
    dqf = pad_or_crop_to_shape_2d(dqf, reference_raw_y, reference_raw_x, fill=1).astype(np.int16, copy=False)
    valid = (dqf == 0)
    return data, valid


def decode_abi_file_to_raw_contrib(
    payload: bytes,
    data_variable: str,
    dqf_variable: str,
    reference_raw_y: int,
    reference_raw_x: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      raw_sum_valid: float32 (raw_y, raw_x)  (value where valid else 0)
      raw_count_valid: int16 (raw_y, raw_x) (1 where valid else 0)
    """
    data, valid = _read_abi_raw_data_and_valid_mask(payload, data_variable, dqf_variable, reference_raw_y, reference_raw_x)
    v = np.where(valid, data, 0.0).astype(np.float32, copy=False)
    c = valid.astype(np.int16, copy=False)
    return v, c


def _decode_abi_worker(payload: bytes, data_variable: str, dqf_variable: str, raw_y: int, raw_x: int):
    return decode_abi_file_to_raw_contrib(payload, data_variable, dqf_variable, raw_y, raw_x)


def _coarsen_mean_ignore_nan(raw_mean: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return raw_mean.astype(np.float32, copy=False)

    y2 = (raw_mean.shape[0] // f) * f
    x2 = (raw_mean.shape[1] // f) * f
    a = raw_mean[:y2, :x2].astype(np.float32, copy=False)

    finite = np.isfinite(a)
    cnt = finite.reshape(y2 // f, f, x2 // f, f).sum(axis=(1, 3)).astype(np.float32, copy=False)

    a0 = np.where(finite, a, 0.0).astype(np.float32, copy=False)
    s = a0.reshape(y2 // f, f, x2 // f, f).sum(axis=(1, 3)).astype(np.float32, copy=False)

    denom = np.maximum(cnt, 1.0)
    mean = (s / denom).astype(np.float32, copy=False)
    mean = np.where(cnt > 0, mean, np.nan).astype(np.float32, copy=False)
    return mean


def _coarsen_sum_int(raw_count: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return raw_count
    y2 = (raw_count.shape[0] // f) * f
    x2 = (raw_count.shape[1] // f) * f
    a = raw_count[:y2, :x2]
    return a.reshape(y2 // f, f, x2 // f, f).sum(axis=(1, 3))


# =============================================================================
# GLM -> ABI grid flash_count
# =============================================================================

def rasterize_glm_flash_counts_to_abi_grid(
    payload: bytes,
    x_scan_angle_rad_coarse: np.ndarray,
    y_scan_angle_rad_coarse: np.ndarray,
    geos_projection: Dict[str, float | str],
) -> np.ndarray:
    import netCDF4 as nc
    from pyproj import CRS, Transformer

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        if ("flash_lat" not in ds.variables) or ("flash_lon" not in ds.variables) or ("flash_quality_flag" not in ds.variables):
            return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

        lat = np.asarray(ds.variables["flash_lat"][:], dtype=np.float64).reshape(-1)
        lon = np.asarray(ds.variables["flash_lon"][:], dtype=np.float64).reshape(-1)
        q = np.asarray(ds.variables["flash_quality_flag"][:], dtype=np.int16).reshape(-1)

    if lat.size == 0:
        return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

    good = (q == 0) & np.isfinite(lat) & np.isfinite(lon)
    if not np.any(good):
        return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

    lat = lat[good]
    lon = lon[good]

    lon0_deg = float(geos_projection["projection_origin_longitude_deg"])
    h_m = float(geos_projection["perspective_point_height_m"])
    a_m = float(geos_projection["semi_major_axis_m"])
    b_m = float(geos_projection["semi_minor_axis_m"])
    sweep = str(geos_projection["sweep_angle_axis"])

    geos = CRS.from_proj4(
        f"+proj=geos +lon_0={lon0_deg} +h={h_m} +a={a_m} +b={b_m} +sweep={sweep} +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    tfm = Transformer.from_crs(wgs84, geos, always_xy=True)

    x_m, y_m = tfm.transform(lon, lat)
    x_rad = np.asarray(x_m, dtype=np.float64) / float(h_m)
    y_rad = np.asarray(y_m, dtype=np.float64) / float(h_m)

    xmin, xmax = float(np.min(x_scan_angle_rad_coarse)), float(np.max(x_scan_angle_rad_coarse))
    ymin, ymax = float(np.min(y_scan_angle_rad_coarse)), float(np.max(y_scan_angle_rad_coarse))
    in_bounds = (x_rad >= xmin) & (x_rad <= xmax) & (y_rad >= ymin) & (y_rad <= ymax)
    if not np.any(in_bounds):
        return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

    x_rad = x_rad[in_bounds]
    y_rad = y_rad[in_bounds]

    x_vec = x_scan_angle_rad_coarse
    x_ascending = x_vec[0] < x_vec[-1]
    if not x_ascending:
        x_vec = x_vec[::-1]

    y_vec = y_scan_angle_rad_coarse
    y_ascending = y_vec[0] < y_vec[-1]
    if not y_ascending:
        y_vec = y_vec[::-1]

    ix = np.searchsorted(x_vec, x_rad, side="left")
    iy = np.searchsorted(y_vec, y_rad, side="left")
    ix = np.clip(ix, 1, x_vec.shape[0] - 1)
    iy = np.clip(iy, 1, y_vec.shape[0] - 1)

    x0 = x_vec[ix - 1]
    x1 = x_vec[ix]
    ix = np.where(np.abs(x_rad - x0) <= np.abs(x_rad - x1), ix - 1, ix)

    y0v = y_vec[iy - 1]
    y1v = y_vec[iy]
    iy = np.where(np.abs(y_rad - y0v) <= np.abs(y_rad - y1v), iy - 1, iy)

    if not x_ascending:
        ix = (x_scan_angle_rad_coarse.shape[0] - 1) - ix
    if not y_ascending:
        iy = (y_scan_angle_rad_coarse.shape[0] - 1) - iy

    out = np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)
    np.add.at(out, (iy, ix), 1.0)
    return out


def _decode_glm_worker(payload: bytes, x_scan_angle_rad: np.ndarray, y_scan_angle_rad: np.ndarray, geos_projection: Dict[str, float | str]):
    return rasterize_glm_flash_counts_to_abi_grid(payload, x_scan_angle_rad, y_scan_angle_rad, geos_projection)


# =============================================================================
# Zarr init + writer
# =============================================================================

def init_daily_zarr_store_atomic(
    cfg: PipelineConfig,
    out_zarr_path: str,
    num_abi_products: int,
    abi_product_keys: List[str],
    grid_y: int,
    grid_x: int,
    year: int,
    julian_day: int,
    bin_start_times_ns: np.ndarray,
    geos_projection: Dict[str, float | str],
) -> Any:
    n_bins = bins_per_day(cfg)

    if zarr_store_exists(out_zarr_path) and not cfg.overwrite_existing:
        return zarr.open_group(out_zarr_path, mode="r+")

    tmp_path = out_zarr_path + ".tmp"
    if os.path.isdir(tmp_path):
        remove_directory_tree(tmp_path)

    if zarr_store_exists(out_zarr_path) and cfg.overwrite_existing:
        remove_directory_tree(out_zarr_path)

    compressor = zarr.Blosc(cname="zstd", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE)

    store = zarr.DirectoryStore(tmp_path)
    root = zarr.group(store=store, overwrite=True)

    root.attrs.update(
        {
            "schema_version": "v6_true_pipeline_decoupled_download_decode",
            "year": year,
            "julian_day": julian_day,
            "coarsen_factor_effective": int(cfg.coarsen_factor),
            "number_of_5min_bins_per_bin": int(cfg.number_of_5min_bins_per_bin),
            "bin_seconds": int(bin_seconds(cfg)),
            "num_time_bins_per_day": int(n_bins),
            "abi_bin_assignment_rule": "STRICT [t0,t1): file_start>=t0 and file_end<t1; include ALL qualifying files",
            "abi_aggregation_rule": "RAW grid mean across files ignoring DQF!=0; then coarsen mean ignoring NaN",
            "abi_valid_pixel_fraction_definition": "valid_pixels / expected_pixels where expected_pixels = expected_files_per_bin * (f*f)",
            "glm_bin_assignment_rule": "STRICT [t0,t1): file_start>=t0 and file_end<t1; include all qualifying files",
            "missing_abi_behavior": "product_value=NaN, valid_pixel_fraction=0, bin_has_decoded_file=0",
            "missing_glm_behavior": "flash_count=0, bin_has_decoded_file=0",
            "expected_abi_files_per_5min": int(cfg.expected_abi_files_per_5min),
            "expected_glm_files_per_5min": int(cfg.expected_glm_files_per_5min),
            "expected_abi_files_per_bin": int(expected_abi_files_per_bin(cfg)),
            "expected_glm_files_per_bin": int(expected_glm_files_per_bin(cfg)),
            "zarr_compressor": "blosc:zstd:clevel1:bitshuffle",
            "grid_shape_yx": [int(grid_y), int(grid_x)],
            "zarr_chunking": {
                "time_chunk": int(cfg.zarr_time_chunk),
                "chunk_y": int(cfg.zarr_chunk_y),
                "chunk_x": int(cfg.zarr_chunk_x),
            },
            "pipeline_backpressure": {
                "max_inflight_payloads": int(cfg.max_inflight_payloads),
            },
            "abi_products": [
                {
                    "key": p.key,
                    "gcs_product_prefix": p.gcs_product_prefix,
                    "filename_must_contain": p.filename_must_contain,
                    "data_variable": p.data_variable,
                    "dqf_variable": p.dqf_variable,
                }
                for p in cfg.abi_products
            ],
            "glm_product": {
                "key": cfg.glm_product.key,
                "gcs_product_prefix": cfg.glm_product.gcs_product_prefix,
                "filename_must_contain": cfg.glm_product.filename_must_contain,
            },
            "source_gcs_bucket": cfg.gcs_bucket,
            "source_satellite_label": infer_satellite_label_from_bucket(cfg.gcs_bucket),
            "goes_geos_projection": geos_projection,
            "complete": False,
            "completed_utc": None,
        }
    )

    # Coordinates / labels
    root.create_dataset("time/bin_start_ns", shape=(n_bins,), chunks=(min(n_bins, 1024),), dtype="i8", overwrite=True)[:] = (
        bin_start_times_ns.astype("datetime64[ns]").astype(np.int64)
    )
    root.create_dataset("time/bin_center_ns", shape=(n_bins,), chunks=(min(n_bins, 1024),), dtype="i8", overwrite=True)[:] = (
        (bin_start_times_ns + np.timedelta64(bin_seconds(cfg) // 2, "s")).astype("datetime64[ns]").astype(np.int64)
    )

    root.create_dataset("abi/product_key", shape=(num_abi_products,), chunks=(num_abi_products,), dtype="U64", overwrite=True)[:] = np.array(
        abi_product_keys, dtype="U64"
    )

    root.create_dataset("grid/x_scan_angle_rad", shape=(grid_x,), chunks=(min(grid_x, 4096),), dtype="f8", overwrite=True)
    root.create_dataset("grid/y_scan_angle_rad", shape=(grid_y,), chunks=(min(grid_y, 4096),), dtype="f8", overwrite=True)

    # ABI core outputs
    root.create_dataset(
        "abi/product_value",
        shape=(n_bins, num_abi_products, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), 1, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/valid_pixel_fraction",
        shape=(n_bins, num_abi_products, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), 1, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/bin_has_decoded_file",
        shape=(n_bins, num_abi_products),
        chunks=(n_bins, num_abi_products),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    # GLM core outputs
    root.create_dataset(
        "glm/flash_count",
        shape=(n_bins, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/bin_has_decoded_file",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    # Validation: ABI per-pixel accounting
    for name, dtype, fill in [
        ("validation/abi_expected_pixels", "i4", 0),
        ("validation/abi_valid_pixels", "i4", 0),
        ("validation/abi_actual_pixels", "i4", 0),
    ]:
        root.create_dataset(
            name,
            shape=(n_bins, num_abi_products, grid_y, grid_x),
            chunks=(min(cfg.zarr_time_chunk, n_bins), 1, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
            dtype=dtype,
            overwrite=True,
            fill_value=fill,
            compressor=compressor,
        )

    for name, dtype, fill in [
        ("validation/abi_valid_pixel_fraction", "f4", 0.0),
        ("validation/abi_actual_pixel_fraction", "f4", 0.0),
    ]:
        root.create_dataset(
            name,
            shape=(n_bins, num_abi_products, grid_y, grid_x),
            chunks=(min(cfg.zarr_time_chunk, n_bins), 1, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
            dtype=dtype,
            overwrite=True,
            fill_value=fill,
            compressor=compressor,
        )

    # Validation: ABI per-bin file accounting
    for name, dtype, fill in [
        ("validation/abi_expected_files", "i4", 0),
        ("validation/abi_actual_valid_files", "i4", 0),
    ]:
        root.create_dataset(
            name,
            shape=(n_bins, num_abi_products),
            chunks=(n_bins, num_abi_products),
            dtype=dtype,
            overwrite=True,
            fill_value=fill,
            compressor=compressor,
        )

    for name, dtype, fill in [
        ("validation/abi_actual_over_expected_files", "f4", 0.0),
        ("validation/abi_expected_over_actual_files", "f4", 0.0),
    ]:
        root.create_dataset(
            name,
            shape=(n_bins, num_abi_products),
            chunks=(n_bins, num_abi_products),
            dtype=dtype,
            overwrite=True,
            fill_value=fill,
            compressor=compressor,
        )

    # Validation: GLM per-bin file accounting
    for name, dtype, fill in [
        ("validation/glm_expected_files", "i4", 0),
        ("validation/glm_actual_valid_files", "i4", 0),
    ]:
        root.create_dataset(
            name,
            shape=(n_bins,),
            chunks=(n_bins,),
            dtype=dtype,
            overwrite=True,
            fill_value=fill,
            compressor=compressor,
        )

    for name, dtype, fill in [
        ("validation/glm_actual_over_expected_files", "f4", 0.0),
        ("validation/glm_expected_over_actual_files", "f4", 0.0),
    ]:
        root.create_dataset(
            name,
            shape=(n_bins,),
            chunks=(n_bins,),
            dtype=dtype,
            overwrite=True,
            fill_value=fill,
            compressor=compressor,
        )

    # Publish atomically
    if os.path.isdir(out_zarr_path):
        remove_directory_tree(out_zarr_path)
    shutil.move(tmp_path, out_zarr_path)
    return zarr.open_group(out_zarr_path, mode="r+")


def _write_time_window_to_zarr(
    root: Any,
    window_start_bin: int,
    abi_value_window: np.ndarray,
    abi_valid_pixel_fraction_window: np.ndarray,
    abi_bin_present_window: np.ndarray,
    glm_flash_count_window: np.ndarray,
    glm_bin_present_window: np.ndarray,
    abi_expected_pixels_window: np.ndarray,
    abi_valid_pixels_window: np.ndarray,
    abi_expected_pixel_fraction_window: np.ndarray,
    abi_actual_pixels_window: np.ndarray,
    abi_actual_pixel_fraction_window: np.ndarray,
    abi_expected_files_window: np.ndarray,
    abi_actual_valid_files_window: np.ndarray,
    abi_actual_over_expected_files_window: np.ndarray,
    abi_expected_over_actual_files_window: np.ndarray,
    glm_expected_files_window: np.ndarray,
    glm_actual_valid_files_window: np.ndarray,
    glm_actual_over_expected_files_window: np.ndarray,
    glm_expected_over_actual_files_window: np.ndarray,
) -> None:
    window_end_bin = window_start_bin + abi_value_window.shape[0]

    # ABI
    root["abi/product_value"][window_start_bin:window_end_bin, :, :, :] = abi_value_window
    root["abi/valid_pixel_fraction"][window_start_bin:window_end_bin, :, :, :] = abi_valid_pixel_fraction_window
    root["abi/bin_has_decoded_file"][window_start_bin:window_end_bin, :] = abi_bin_present_window

    # GLM
    root["glm/flash_count"][window_start_bin:window_end_bin, :, :] = glm_flash_count_window
    root["glm/bin_has_decoded_file"][window_start_bin:window_end_bin] = glm_bin_present_window

    # Validation: ABI per-pixel
    root["validation/abi_expected_pixels"][window_start_bin:window_end_bin, :, :, :] = abi_expected_pixels_window
    root["validation/abi_valid_pixels"][window_start_bin:window_end_bin, :, :, :] = abi_valid_pixels_window
    root["validation/abi_valid_pixel_fraction"][window_start_bin:window_end_bin, :, :, :] = abi_expected_pixel_fraction_window
    root["validation/abi_actual_pixels"][window_start_bin:window_end_bin, :, :, :] = abi_actual_pixels_window
    root["validation/abi_actual_pixel_fraction"][window_start_bin:window_end_bin, :, :, :] = abi_actual_pixel_fraction_window

    # Validation: ABI per-bin files
    root["validation/abi_expected_files"][window_start_bin:window_end_bin, :] = abi_expected_files_window
    root["validation/abi_actual_valid_files"][window_start_bin:window_end_bin, :] = abi_actual_valid_files_window
    root["validation/abi_actual_over_expected_files"][window_start_bin:window_end_bin, :] = abi_actual_over_expected_files_window
    root["validation/abi_expected_over_actual_files"][window_start_bin:window_end_bin, :] = abi_expected_over_actual_files_window

    # Validation: GLM per-bin files
    root["validation/glm_expected_files"][window_start_bin:window_end_bin] = glm_expected_files_window
    root["validation/glm_actual_valid_files"][window_start_bin:window_end_bin] = glm_actual_valid_files_window
    root["validation/glm_actual_over_expected_files"][window_start_bin:window_end_bin] = glm_actual_over_expected_files_window
    root["validation/glm_expected_over_actual_files"][window_start_bin:window_end_bin] = glm_expected_over_actual_files_window


# =============================================================================
# True pipeline: decoupled download vs decode (bounded inflight payloads)
# =============================================================================

@dataclass(frozen=True)
class WorkItem:
    kind: str                 # "ABI" or "GLM"
    bin_index: int
    product_key: str          # ABI: spec.key ; GLM: "__GLM__"
    blob: str
    data_variable: Optional[str] = None
    dqf_variable: Optional[str] = None


@dataclass(frozen=True)
class DecodedResult:
    kind: str
    bin_index: int
    product_key: str
    ok: bool
    # ABI:
    raw_sum: Optional[np.ndarray] = None
    raw_count: Optional[np.ndarray] = None
    # GLM:
    glm_counts: Optional[np.ndarray] = None


async def run_true_pipeline_for_items(
    cfg: PipelineConfig,
    session: aiohttp.ClientSession,
    http_semaphore: asyncio.Semaphore,
    decode_pool: ProcessPoolExecutor,
    items: List[WorkItem],
    *,
    # decode inputs:
    raw_y: int,
    raw_x: int,
    x_scan_angle_rad: np.ndarray,
    y_scan_angle_rad: np.ndarray,
    geos_projection: Dict[str, float | str],
    # handler:
    handle: Any,
) -> None:
    """
    Decoupled pipeline:
      - download workers fetch bytes and push payloads to decoder queue
      - decoder workers run ProcessPool decode and emit DecodedResult
      - main consumer handles exactly len(items) results (including download fails)

    Backpressure:
      - inflight_payload_sem caps how many payloads can exist concurrently awaiting/inside decode
    """
    if not items:
        return

    loop = asyncio.get_running_loop()

    work_q: asyncio.Queue[Optional[WorkItem]] = asyncio.Queue()
    decode_q: asyncio.Queue[Optional[Tuple[WorkItem, bytes]]] = asyncio.Queue()
    result_q: asyncio.Queue[DecodedResult] = asyncio.Queue()

    inflight_payload_sem = asyncio.Semaphore(max(1, int(cfg.max_inflight_payloads)))

    # Heuristic worker counts
    n_dl = max(1, min(int(cfg.max_concurrent_http_requests), 64))
    n_dec = max(1, min(int(cfg.max_decode_processes), 64))

    for it in items:
        work_q.put_nowait(it)
    for _ in range(n_dl):
        work_q.put_nowait(None)

    async def downloader_worker() -> None:
        while True:
            it = await work_q.get()
            try:
                if it is None:
                    return

                # Backpressure BEFORE downloading: prevents many big payloads sitting in memory
                await inflight_payload_sem.acquire()

                async with http_semaphore:
                    payload = await download_gcs_object_bytes(cfg, session, it.blob)

                if payload is None:
                    # download failed -> emit result, release inflight
                    inflight_payload_sem.release()
                    await result_q.put(DecodedResult(kind=it.kind, bin_index=it.bin_index, product_key=it.product_key, ok=False))
                    continue

                # push to decode queue (payload kept alive until decode finishes)
                await decode_q.put((it, payload))

            finally:
                work_q.task_done()

    async def decoder_worker() -> None:
        while True:
            pair = await decode_q.get()
            try:
                if pair is None:
                    return
                it, payload = pair

                try:
                    if it.kind == "ABI":
                        rs, rc = await loop.run_in_executor(
                            decode_pool,
                            _decode_abi_worker,
                            payload,
                            str(it.data_variable),
                            str(it.dqf_variable),
                            raw_y,
                            raw_x,
                        )
                        await result_q.put(
                            DecodedResult(kind="ABI", bin_index=it.bin_index, product_key=it.product_key, ok=True, raw_sum=rs, raw_count=rc)
                        )
                    else:
                        counts = await loop.run_in_executor(
                            decode_pool,
                            _decode_glm_worker,
                            payload,
                            x_scan_angle_rad,
                            y_scan_angle_rad,
                            geos_projection,
                        )
                        await result_q.put(
                            DecodedResult(kind="GLM", bin_index=it.bin_index, product_key=it.product_key, ok=True, glm_counts=counts)
                        )
                except Exception:
                    await result_q.put(DecodedResult(kind=it.kind, bin_index=it.bin_index, product_key=it.product_key, ok=False))
                finally:
                    # payload lifetime ends here
                    inflight_payload_sem.release()

            finally:
                decode_q.task_done()

    dl_tasks = [asyncio.create_task(downloader_worker()) for _ in range(n_dl)]
    dec_tasks = [asyncio.create_task(decoder_worker()) for _ in range(n_dec)]

    try:
        # Consumer: exactly one result per input item
        for _ in range(len(items)):
            res = await result_q.get()
            try:
                handle(res)
            finally:
                result_q.task_done()
    finally:
        # Stop decode workers (best-effort). We may still have decode_q items; but consumer completion implies all
        # items produced results, so decode_q should be drained.
        for _ in range(n_dec):
            try:
                decode_q.put_nowait(None)
            except Exception:
                pass

        await asyncio.gather(*dl_tasks, return_exceptions=True)
        await asyncio.gather(*dec_tasks, return_exceptions=True)


# =============================================================================
# Per-day processing
# =============================================================================

async def process_single_day(
    cfg: PipelineConfig,
    day_utc: datetime,
    session: aiohttp.ClientSession,
    http_semaphore: asyncio.Semaphore,
    decode_pool: ProcessPoolExecutor,
) -> bool:
    n_bins = bins_per_day(cfg)
    year = day_utc.year
    julian_day = int(day_utc.strftime("%j"))
    out_zarr_path = output_zarr_path_for_day(cfg.output_root_dir, year, julian_day)

    if not cfg.overwrite_existing and zarr_store_is_complete(out_zarr_path):
        log(f"SKIP {day_utc.date().isoformat()} ({year}{julian_day:03d}) already merged (complete=true)")
        return True

    if (not cfg.overwrite_existing) and zarr_store_exists(out_zarr_path) and (not zarr_store_is_complete(out_zarr_path)):
        log(f"RETRY {day_utc.date().isoformat()} ({year}{julian_day:03d}) partial zarr; rebuilding")
        remove_directory_tree(out_zarr_path)

    f = int(cfg.coarsen_factor)
    exp_abi_files_bin = int(expected_abi_files_per_bin(cfg))
    exp_glm_files_bin = int(expected_glm_files_per_bin(cfg))

    abi_product_keys = [p.key for p in cfg.abi_products]
    num_abi_products = len(cfg.abi_products)
    p_index = {k: i for i, k in enumerate(abi_product_keys)}

    log(
        f"=== {day_utc.date().isoformat()} ({year}{julian_day:03d}) "
        f"bins={n_bins} bin_sec={bin_seconds(cfg)} coarsen_f={f} abi_products={num_abi_products} ==="
    )

    day_start = utc_day_start(year, julian_day)
    bin_starts_ns = build_bin_start_times_ns(cfg, year, julian_day)

    # Listing
    list_semaphore = asyncio.Semaphore(cfg.max_concurrent_list_requests)
    log("listing objects (async)...")
    t_list0 = time.perf_counter()

    abi_list_tasks = [
        asyncio.create_task(list_day_objects_for_product(cfg, session, spec, year, julian_day, list_semaphore))
        for spec in cfg.abi_products
    ]
    glm_list_task = asyncio.create_task(list_day_objects_for_product(cfg, session, cfg.glm_product, year, julian_day, list_semaphore))

    abi_file_triples_per_product = await asyncio.gather(*abi_list_tasks)
    glm_file_triples = await glm_list_task
    log(f"listing done in {time.perf_counter() - t_list0:.2f}s")

    # Strict bin assignment
    abi_blobs_in_bin_by_product: Dict[str, List[List[str]]] = {}
    for spec, triples in zip(cfg.abi_products, abi_file_triples_per_product):
        triples = triples or []
        abi_blobs_in_bin_by_product[spec.key] = (
            group_files_into_bins_strict(cfg, day_start, triples) if triples else [[] for _ in range(n_bins)]
        )

    glm_triples = glm_file_triples or []
    glm_blobs_in_bin = group_files_into_bins_strict(cfg, day_start, glm_triples) if glm_triples else [[] for _ in range(n_bins)]

    # Probe ABI reference (use first downloadable ABI file across any product/bin)
    log("probing ABI reference...")
    probe_payload: Optional[bytes] = None
    probe_spec: Optional[RemoteProductSpec] = None
    probe_blob: Optional[str] = None

    for spec in cfg.abi_products:
        for bin_list in abi_blobs_in_bin_by_product[spec.key]:
            if bin_list:
                probe_blob = bin_list[0]
                probe_spec = spec
                async with http_semaphore:
                    probe_payload = await download_gcs_object_bytes(cfg, session, probe_blob)
                if probe_payload:
                    break
        if probe_payload:
            break

    if probe_payload is None or probe_spec is None:
        log("!! no ABI probe downloadable; skipping day")
        return False

    try:
        raw_y, raw_x, x_raw, y_raw, geos_projection = probe_abi_reference_grid_and_projection(
            probe_payload,
            str(probe_spec.data_variable),
            str(probe_spec.dqf_variable),
        )
        x_scan_angle_rad, y_scan_angle_rad = build_coarsened_scan_angle_vectors(x_raw, y_raw, f)
        grid_y = int(y_scan_angle_rad.shape[0])
        grid_x = int(x_scan_angle_rad.shape[0])
        log(f"probe OK: raw={raw_y}x{raw_x} -> coarse={grid_y}x{grid_x} ({os.path.basename(probe_blob or '')})")
    except Exception as e:
        log(f"!! probe failed: {repr(e)}")
        return False

    # Init Zarr
    try:
        root = init_daily_zarr_store_atomic(
            cfg=cfg,
            out_zarr_path=out_zarr_path,
            num_abi_products=num_abi_products,
            abi_product_keys=abi_product_keys,
            grid_y=grid_y,
            grid_x=grid_x,
            year=year,
            julian_day=julian_day,
            bin_start_times_ns=bin_starts_ns,
            geos_projection=geos_projection,
        )
    except Exception as e:
        log(f"!! init zarr failed: {repr(e)}")
        return False

    try:
        root["grid/x_scan_angle_rad"][:] = x_scan_angle_rad.astype(np.float64, copy=False)
        root["grid/y_scan_angle_rad"][:] = y_scan_angle_rad.astype(np.float64, copy=False)
    except Exception:
        pass

    # Expected pixels per output pixel per bin for ABI: expected_files_per_bin * f*f
    expected_pixels_per_output_pixel = int(exp_abi_files_bin * (f * f))

    # Process in windows (NOTE: per-bin raw accumulators only -> drastically lower memory)
    for window_start in range(0, n_bins, cfg.time_bin_write_window):
        window_end = min(n_bins, window_start + cfg.time_bin_write_window)
        window_len = window_end - window_start

        # ABI outputs for this window (coarse grid only)
        abi_value_window = np.full((window_len, num_abi_products, grid_y, grid_x), np.nan, dtype=np.float32)
        abi_present_win = np.zeros((window_len, num_abi_products), dtype=np.uint8)

        # Validation ABI per-pixel
        abi_expected_pixels_win = np.zeros((window_len, num_abi_products, grid_y, grid_x), dtype=np.int32)
        abi_valid_pixels_win = np.zeros((window_len, num_abi_products, grid_y, grid_x), dtype=np.int32)
        abi_expected_frac_win = np.zeros((window_len, num_abi_products, grid_y, grid_x), dtype=np.float32)
        abi_actual_pixels_win = np.zeros((window_len, num_abi_products, grid_y, grid_x), dtype=np.int32)
        abi_actual_frac_win = np.zeros((window_len, num_abi_products, grid_y, grid_x), dtype=np.float32)

        # Validation ABI per-bin files
        abi_expected_files_win = np.full((window_len, num_abi_products), exp_abi_files_bin, dtype=np.int32)
        abi_actual_valid_files_win = np.zeros((window_len, num_abi_products), dtype=np.int32)
        abi_actual_over_expected_files_win = np.zeros((window_len, num_abi_products), dtype=np.float32)
        abi_expected_over_actual_files_win = np.zeros((window_len, num_abi_products), dtype=np.float32)

        # GLM outputs for this window
        glm_flash_count_sum = np.zeros((window_len, grid_y, grid_x), dtype=np.float32)
        glm_bin_present = np.zeros((window_len,), dtype=np.uint8)

        # Validation GLM per-bin files
        glm_expected_files_win = np.full((window_len,), exp_glm_files_bin, dtype=np.int32)
        glm_actual_valid_files_win = np.zeros((window_len,), dtype=np.int32)
        glm_actual_over_expected_files_win = np.zeros((window_len,), dtype=np.float32)
        glm_expected_over_actual_files_win = np.zeros((window_len,), dtype=np.float32)

        # Fill expected per-pixel constants now
        if expected_pixels_per_output_pixel > 0:
            abi_expected_pixels_win[:, :, :, :] = expected_pixels_per_output_pixel

        # Process each bin in the window using a true pipeline for that bin's files
        for bin_index in range(window_start, window_end):
            wi = bin_index - window_start

            # Per-bin raw accumulators (only for THIS bin) to avoid huge memory use.
            raw_sum_by_prod: Dict[str, Optional[np.ndarray]] = {p.key: None for p in cfg.abi_products}
            raw_count_by_prod: Dict[str, Optional[np.ndarray]] = {p.key: None for p in cfg.abi_products}
            ok_files_by_prod: Dict[str, int] = {p.key: 0 for p in cfg.abi_products}

            glm_ok_files = 0
            glm_sum = np.zeros((grid_y, grid_x), dtype=np.float32)

            # Build items for this bin
            items: List[WorkItem] = []

            for spec in cfg.abi_products:
                for blob in abi_blobs_in_bin_by_product[spec.key][bin_index]:
                    items.append(
                        WorkItem(
                            kind="ABI",
                            bin_index=bin_index,
                            product_key=spec.key,
                            blob=blob,
                            data_variable=str(spec.data_variable),
                            dqf_variable=str(spec.dqf_variable),
                        )
                    )

            for blob in glm_blobs_in_bin[bin_index]:
                items.append(
                    WorkItem(
                        kind="GLM",
                        bin_index=bin_index,
                        product_key="__GLM__",
                        blob=blob,
                    )
                )

            def handle_result(res: DecodedResult) -> None:
                nonlocal glm_ok_files, glm_sum
                if not res.ok:
                    return

                if res.kind == "ABI":
                    pk = res.product_key
                    if (res.raw_sum is None) or (res.raw_count is None):
                        return

                    if raw_sum_by_prod[pk] is None:
                        raw_sum_by_prod[pk] = res.raw_sum.astype(np.float32, copy=False)
                        raw_count_by_prod[pk] = res.raw_count.astype(np.int32, copy=False)
                    else:
                        raw_sum_by_prod[pk] = raw_sum_by_prod[pk] + res.raw_sum.astype(np.float32, copy=False)
                        raw_count_by_prod[pk] = raw_count_by_prod[pk] + res.raw_count.astype(np.int32, copy=False)

                    ok_files_by_prod[pk] += 1
                    abi_present_win[wi, p_index[pk]] = np.uint8(1)

                else:
                    if res.glm_counts is None:
                        return
                    glm_sum += res.glm_counts.astype(np.float32, copy=False)
                    glm_ok_files += 1
                    glm_bin_present[wi] = np.uint8(1)

            # Run decoupled pipeline for this bin's items
            await run_true_pipeline_for_items(
                cfg=cfg,
                session=session,
                http_semaphore=http_semaphore,
                decode_pool=decode_pool,
                items=items,
                raw_y=raw_y,
                raw_x=raw_x,
                x_scan_angle_rad=x_scan_angle_rad,
                y_scan_angle_rad=y_scan_angle_rad,
                geos_projection=geos_projection,
                handle=handle_result,
            )

            # GLM finalize for this bin
            glm_flash_count_sum[wi] = glm_sum
            glm_actual_valid_files_win[wi] = int(glm_ok_files)
            exp_g = float(exp_glm_files_bin)
            act_g = float(glm_ok_files)
            glm_actual_over_expected_files_win[wi] = (act_g / exp_g) if exp_g > 0 else 0.0
            glm_expected_over_actual_files_win[wi] = (exp_g / act_g) if act_g > 0 else (float("inf") if exp_g > 0 else 0.0)

            # ABI finalize per product for this bin
            for spec in cfg.abi_products:
                pk = spec.key
                pj = p_index[pk]
                ok_files = int(ok_files_by_prod[pk])

                # file accounting
                abi_actual_valid_files_win[wi, pj] = ok_files
                exp_f = float(exp_abi_files_bin)
                act_f = float(ok_files)
                abi_actual_over_expected_files_win[wi, pj] = (act_f / exp_f) if exp_f > 0 else 0.0
                abi_expected_over_actual_files_win[wi, pj] = (exp_f / act_f) if act_f > 0 else (float("inf") if exp_f > 0 else 0.0)

                # per-pixel constants
                actual_pixels_per_output_pixel = int(ok_files * (f * f))
                if actual_pixels_per_output_pixel > 0:
                    abi_actual_pixels_win[wi, pj, :, :] = actual_pixels_per_output_pixel

                if ok_files <= 0:
                    # nothing decoded -> keep defaults
                    continue

                rs = raw_sum_by_prod[pk]
                rc = raw_count_by_prod[pk]
                if rs is None or rc is None:
                    continue

                # raw mean per pixel across files, ignoring invalid samples
                raw_mean = np.full((raw_y, raw_x), np.nan, dtype=np.float32)
                m = rc > 0
                raw_mean[m] = (rs[m] / rc[m]).astype(np.float32, copy=False)

                # Coarsened value
                coarse_val = _coarsen_mean_ignore_nan(raw_mean, f)

                # valid_pixels per output pixel = sum(raw_count) in block
                coarse_valid_pixels = _coarsen_sum_int(rc.astype(np.int32, copy=False), f).astype(np.int32, copy=False)

                # Pad/crop (should match)
                coarse_val = pad_or_crop_to_shape_2d(coarse_val.astype(np.float32, copy=False), grid_y, grid_x, fill=np.nan)
                coarse_valid_pixels = pad_or_crop_to_shape_2d(coarse_valid_pixels.astype(np.int32, copy=False), grid_y, grid_x, fill=0)

                # expected fraction: valid_pixels / expected_pixels
                if expected_pixels_per_output_pixel > 0:
                    coarse_vf_expected = (coarse_valid_pixels.astype(np.float32) / float(expected_pixels_per_output_pixel)).astype(np.float32, copy=False)
                else:
                    coarse_vf_expected = np.zeros((grid_y, grid_x), dtype=np.float32)

                # actual fraction: valid_pixels / actual_pixels (decoded_ok_files*f*f)
                if actual_pixels_per_output_pixel > 0:
                    coarse_vf_actual = (coarse_valid_pixels.astype(np.float32) / float(actual_pixels_per_output_pixel)).astype(np.float32, copy=False)
                else:
                    coarse_vf_actual = np.zeros((grid_y, grid_x), dtype=np.float32)

                abi_value_window[wi, pj] = coarse_val
                abi_valid_pixels_win[wi, pj] = coarse_valid_pixels
                abi_expected_frac_win[wi, pj] = coarse_vf_expected
                abi_actual_frac_win[wi, pj] = coarse_vf_actual

        # Export ABI valid_pixel_fraction dataset to match requested definition
        abi_valid_pixel_fraction_window = abi_expected_frac_win

        # Write window
        t_write0 = time.perf_counter()
        await asyncio.to_thread(
            _write_time_window_to_zarr,
            root,
            window_start,
            abi_value_window,
            abi_valid_pixel_fraction_window,
            abi_present_win,
            glm_flash_count_sum,
            glm_bin_present,
            abi_expected_pixels_win,
            abi_valid_pixels_win,
            abi_expected_frac_win,
            abi_actual_pixels_win,
            abi_actual_frac_win,
            abi_expected_files_win,
            abi_actual_valid_files_win,
            abi_actual_over_expected_files_win,
            abi_expected_over_actual_files_win,
            glm_expected_files_win,
            glm_actual_valid_files_win,
            glm_actual_over_expected_files_win,
            glm_expected_over_actual_files_win,
        )
        write_seconds = time.perf_counter() - t_write0
        if write_seconds >= cfg.warn_if_write_slower_than_seconds:
            log(f"writer: WARNING slow write bins {window_start}:{window_end}  {write_seconds:.2f}s")

        log(f"{day_utc.date().isoformat()} wrote bins {window_start:03d}-{window_end-1:03d} / {n_bins-1:03d}")

    log(f"-> wrote merged: {out_zarr_path}  bins={n_bins}  grid={grid_y}x{grid_x}  abi_products={num_abi_products}")

    try:
        mark_zarr_store_complete(root)
    except Exception as e:
        log(f"!! WARNING: merged but failed to mark complete: {repr(e)}")
        return False

    return True


# =============================================================================
# Runner
# =============================================================================

def print_config(cfg: PipelineConfig) -> None:
    as_dict = {
        "gcs_bucket": cfg.gcs_bucket,
        "start_date_utc": cfg.start_date_utc,
        "end_date_utc": cfg.end_date_utc,
        "number_of_5min_bins_per_bin": cfg.number_of_5min_bins_per_bin,
        "coarsen_factor": cfg.coarsen_factor,
        "max_concurrent_http_requests": cfg.max_concurrent_http_requests,
        "http_timeout_seconds": cfg.http_timeout_seconds,
        "http_retries": cfg.http_retries,
        "tcp_limit_per_host": cfg.tcp_limit_per_host,
        "max_decode_processes": cfg.max_decode_processes,
        "max_days_in_flight": cfg.max_days_in_flight,
        "time_bin_write_window": cfg.time_bin_write_window,
        "output_root_dir": cfg.output_root_dir,
        "overwrite_existing": cfg.overwrite_existing,
        "zarr_time_chunk": cfg.zarr_time_chunk,
        "zarr_chunk_y": cfg.zarr_chunk_y,
        "zarr_chunk_x": cfg.zarr_chunk_x,
        "max_concurrent_list_requests": cfg.max_concurrent_list_requests,
        "warn_if_write_slower_than_seconds": cfg.warn_if_write_slower_than_seconds,
        "expected_abi_files_per_5min": cfg.expected_abi_files_per_5min,
        "expected_glm_files_per_5min": cfg.expected_glm_files_per_5min,
        "max_inflight_payloads": cfg.max_inflight_payloads,
        "abi_products": [
            {
                "key": p.key,
                "gcs_product_prefix": p.gcs_product_prefix,
                "filename_must_contain": p.filename_must_contain,
                "data_variable": p.data_variable,
                "dqf_variable": p.dqf_variable,
            }
            for p in cfg.abi_products
        ],
        "glm_product": {
            "key": cfg.glm_product.key,
            "gcs_product_prefix": cfg.glm_product.gcs_product_prefix,
            "filename_must_contain": cfg.glm_product.filename_must_contain,
        },
    }
    print("-" * 72)
    print("CONFIG")
    print("-" * 72)
    print(json.dumps(as_dict, indent=2))
    print("-" * 72)


async def run_all_days(cfg: PipelineConfig) -> None:
    ensure_dir(cfg.output_root_dir)
    print_config(cfg)

    # Validate bin config early
    _ = bins_per_day(cfg)
    _ = bin_seconds(cfg)

    days = list(iter_days_utc(cfg.start_date_utc, cfg.end_date_utc))
    total = len(days)

    connector = aiohttp.TCPConnector(
        limit=cfg.max_concurrent_http_requests,
        limit_per_host=cfg.tcp_limit_per_host,
        ttl_dns_cache=300,
    )
    http_semaphore = asyncio.Semaphore(cfg.max_concurrent_http_requests)

    q: asyncio.Queue[Optional[datetime]] = asyncio.Queue()
    for d in days:
        await q.put(d)
    for _ in range(cfg.max_days_in_flight):
        await q.put(None)

    ok = 0
    fail = 0
    counter_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        with ProcessPoolExecutor(max_workers=cfg.max_decode_processes) as pool:

            async def worker(worker_id: int):
                nonlocal ok, fail
                while True:
                    day = await q.get()
                    try:
                        if day is None:
                            return
                        good = await process_single_day(cfg, day, session, http_semaphore, pool)
                        async with counter_lock:
                            if good:
                                ok += 1
                            else:
                                fail += 1
                    finally:
                        q.task_done()

            tasks = [asyncio.create_task(worker(i)) for i in range(cfg.max_days_in_flight)]
            await q.join()
            await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Days attempted: {total}")
    print(f"Days merged OK: {ok}")
    print(f"Days failed:    {fail}")
    print(f"Output root:    {os.path.abspath(cfg.output_root_dir)}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import netCDF4  # noqa: F401
        import pyproj   # noqa: F401
    except Exception:
        print("Missing deps. Install:")
        print("  pip install aiohttp netCDF4 numpy zarr pyproj")
        raise SystemExit(1)

    cfg = load_config(args.config)
    asyncio.run(run_all_days(cfg))


if __name__ == "__main__":
    main()
