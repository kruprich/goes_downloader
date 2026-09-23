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
Each product keeps its native grid (CMIP bands and ACTP: 2 km, 1500x2500; ACHA/TPW: 10 km, 300x500).
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

GLM (only if "glm" is in the config's "download" list): every flash from every 20 s file whose start falls in the day, all quality flags kept.
  glm/files/{start_ns,end_ns,ok,n_flashes}[file]
  glm/flashes/{time_first_ns,time_last_ns,lat,lon,area_m2,energy_j,quality_flag,file_index}[flash]
  Files with ok=0 are coverage gaps: no flashes there does NOT mean no lightning.

Optional "bounds" crop (lat/lon box in config), ABI only:
  The box is turned into the smallest scan-angle rectangle that contains it, shared by all ABI
  products. Each ABI product keeps the pixels whose cell overlaps that rectangle (attrs crop_y0/crop_x0
  give the offset into the uncropped grid; x/y_scan_angle_rad are the cropped coordinates). GLM is not
  cropped: every flash is kept (~36 MB/day), and the feature step selects the ones its grid needs.
  Days already complete are not re-cropped when the bounds change; delete them or set
  overwrite_existing.

Root attrs:
  complete = true only if every listing request succeeded and every file was downloaded and
  decoded. Incomplete days are rebuilt from scratch on the next run.

Reading with xarray (one product or GLM table across several days):
  xr.open_mfdataset(day_paths, engine="zarr", group="abi/cmip_c13", combine="by_coords")
  xr.open_mfdataset(day_paths, engine="zarr", group="glm/flashes", combine="nested", concat_dim="flash")
  xarray turns value/dqf fill values into NaN (floats); pass mask_and_scale=False for the stored integers
  and apply scale_factor/add_offset yourself.

Config:
  python download_raw.py --config config.json
  abi_products / glm_product define every product; "download" lists the keys to fetch.
  Products that read different variables from the same files (e.g. the dsi_* indices) share one
  download per scan, and each bucket prefix is listed once per day.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
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


CMIP_BAND_PATTERN = re.compile(r"CMIP\w*-M\d+C(\d{2})")


def is_cmip_ir_band(product: Dict[str, Any]) -> bool:
    """CMIP bands 7-16 hold brightness temperature (K); bands 1-6 are reflectance (0-1)."""
    m = CMIP_BAND_PATTERN.search(str(product.get("filename_must_contain", "")))
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

    max_concurrent_http_requests = int(raw.get("max_concurrent_http_requests", 32))
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
        max_inflight_payloads=int(raw.get("max_inflight_payloads", max_concurrent_http_requests)),
        abi_products=abi_products,
        glm_product=glm_product,
        bounds=bounds,
    )


# =============================================================================
# Constants + helpers
# =============================================================================

SLOTS_PER_DAY = 288
SLOT_SECONDS = 300
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
    print(f"[{ts}] {msg}", flush=True)


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


def parse_remote_file(blob: str) -> Optional[RemoteFile]:
    times: Dict[str, datetime] = {}
    for m in GOES_FILENAME_TIME_PATTERN.finditer(os.path.basename(blob)):
        kind, y, j, hh, mm, ss, tenth = m.groups()
        times[kind] = datetime(int(y), 1, 1, tzinfo=timezone.utc) + timedelta(
            days=int(j) - 1, hours=int(hh), minutes=int(mm), seconds=int(ss), milliseconds=100 * int(tenth)
        )
    if set(times) != {"s", "e", "c"}:
        return None
    return RemoteFile(blob=blob, start=times["s"], end=times["e"], created=times["c"])


# =============================================================================
# HTTP (retries on transient errors, raises when out of attempts)
# =============================================================================

async def _backoff(attempt: int) -> None:
    await asyncio.sleep(min(30.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random()))


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
) -> List[str]:
    url = f"https://storage.googleapis.com/storage/v1/b/{cfg.gcs_bucket}/o"
    params: Dict[str, str] = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}
    names: List[str] = []
    while True:
        js = await http_get(cfg, session, url, params, as_json=True)
        names.extend(it["name"] for it in js.get("items", []) if it.get("name"))
        page_token = js.get("nextPageToken")
        if not page_token:
            return names
        params["pageToken"] = page_token


async def list_day_names(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    gcs_product_prefix: str,
    day: datetime,
) -> Tuple[List[str], int]:
    """Returns (every object name under the day's 24 hourly prefixes, n_failed_prefixes)."""
    jday = day.timetuple().tm_yday
    prefixes = [f"{gcs_product_prefix}/{day.year}/{jday:03d}/{hh:02d}/" for hh in LIST_HOURS]
    results = await asyncio.gather(
        *(gcs_list_object_names(cfg, session, p) for p in prefixes), return_exceptions=True
    )

    names: List[str] = []
    n_failed = 0
    for prefix, res in zip(prefixes, results):
        if isinstance(res, BaseException):
            n_failed += 1
            log(f"!! listing failed for {prefix}: {res!r}")
            continue
        names.extend(res)
    return names, n_failed


def select_files(names: List[str], filename_must_contain: str) -> List[RemoteFile]:
    """
    Files whose name contains the pattern, deduplicated by scan start time (keeping the newest
    creation time, i.e. reprocessed files) and sorted by start time.
    """
    by_start: Dict[datetime, RemoteFile] = {}
    for name in names:
        if filename_must_contain not in name:
            continue
        rf = parse_remote_file(name)
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
    step (scale_factor * 2**n) is <= precision. Zeroing low bits lets the compressor shrink the data;
    physical = value * scale_factor + add_offset still holds. Returns (values, physical step).
    """
    step = abs(scale_factor)
    if precision is None or step <= 0 or precision < 2 * step:
        return value, step
    n = int(np.floor(np.log2(precision / step)))
    top = (fill_value - 1) if fill_value is not None else np.iinfo(np.uint16).max
    rounded = ((value.astype(np.int32) + (1 << (n - 1))) >> n) << n
    rounded = np.minimum(rounded, (top >> n) << n).astype(np.uint16)
    if fill_value is not None:
        rounded[value == fill_value] = fill_value
    return rounded, step * (1 << n)


def _decode_abi_variable(ds: Any, data_variable: str, dqf_variable: str, precision: Optional[float]) -> Dict[str, Any]:
    v = ds.variables[data_variable]
    q = ds.variables[dqf_variable]
    v.set_auto_maskandscale(False)
    q.set_auto_maskandscale(False)

    value = _as_unsigned(_squeeze_to_2d(v[:], data_variable), v, np.uint16, data_variable)
    dqf_raw = _squeeze_to_2d(q[:], dqf_variable)
    dqf_dtype = np.uint8 if dqf_raw.dtype.itemsize == 1 else np.uint16
    dqf = _as_unsigned(dqf_raw, q, dqf_dtype, dqf_variable)
    if dqf.shape != value.shape:
        raise ValueError(f"{dqf_variable} shape {dqf.shape} != {data_variable} shape {value.shape}")

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


def decode_abi_file(payload: bytes, variables: List[Tuple[str, str, Optional[float]]]) -> List[Dict[str, Any]]:
    """
    Decodes each (data_variable, dqf_variable, precision) from one file, so products that read
    different variables from the same file (e.g. the DSI indices) share one download.
    """
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        # x/y are scaled int16 in the file; netCDF4 unpacks them to radians by default
        x = np.asarray(ds.variables["x"][:], dtype=np.float64).reshape(-1)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64).reshape(-1)
        proj = ds.variables["goes_imager_projection"]
        projection = {k: _jsonable(proj.getncattr(k)) for k in proj.ncattrs()}
        return [
            {**_decode_abi_variable(ds, data_variable, dqf_variable, precision), "x": x, "y": y, "projection": projection}
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


def decode_glm_file(payload: bytes) -> Dict[str, np.ndarray]:
    import netCDF4 as nc

    def floats(ds: Any, name: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(ds.variables[name][:], dtype=np.float32), np.nan).reshape(-1)

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        quality = np.ma.filled(np.ma.asarray(ds.variables["flash_quality_flag"][:], dtype=np.int16), 255)
        return {
            "time_first_ns": _offset_seconds_to_ns(ds.variables["flash_time_offset_of_first_event"]),
            "time_last_ns": _offset_seconds_to_ns(ds.variables["flash_time_offset_of_last_event"]),
            "lat": floats(ds, "flash_lat"),
            "lon": floats(ds, "flash_lon"),
            "area_m2": floats(ds, "flash_area"),
            "energy_j": floats(ds, "flash_energy"),
            "quality_flag": quality.astype(np.uint8).reshape(-1),
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


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


def write_glm_tables(root: Any, files: List[RemoteFile], file_ok: np.ndarray, tables: Dict[int, Dict[str, np.ndarray]]) -> int:
    g = root.require_group("glm")
    n_flashes = np.array([len(tables[i]["lat"]) if i in tables else 0 for i in range(len(files))], dtype=np.int32)
    fg = g.require_group("files")
    _write_array(fg, "start_ns", np.array([to_ns(f.start) for f in files], dtype=np.int64), ["file"], **TIME_ATTRS)
    _write_array(fg, "end_ns", np.array([to_ns(f.end) for f in files], dtype=np.int64), ["file"], **TIME_ATTRS)
    _write_array(fg, "ok", file_ok.astype(np.uint8), ["file"])
    _write_array(fg, "n_flashes", n_flashes, ["file"])

    cols: Dict[str, List[np.ndarray]] = {k: [] for k in GLM_FLASH_COLUMNS}
    for i in sorted(tables):
        t = tables[i]
        for k in GLM_FLASH_COLUMNS:
            cols[k].append(np.full(len(t["lat"]), i, dtype=np.int32) if k == "file_index" else t[k])

    flash_group = g.require_group("flashes")
    merged = {k: np.concatenate(v).astype(GLM_FLASH_COLUMNS[k]) if v else np.zeros(0, GLM_FLASH_COLUMNS[k]) for k, v in cols.items()}
    order = np.argsort(merged["time_first_ns"], kind="stable")
    for k, arr in merged.items():
        arr = arr[order]
        _write_array(
            flash_group, k, arr, ["flash"], **(TIME_ATTRS if k.startswith("time_") else {}),
            zarr=dict(chunks=(max(1, min(len(arr), 1 << 20)),), compressor=COMPRESSOR),
        )
    return int(n_flashes.sum())


# =============================================================================
# Per-day processing
# =============================================================================

async def process_single_day(
    cfg: RawConfig,
    day: datetime,
    session: aiohttp.ClientSession,
    decode_pool: ProcessPoolExecutor,
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
    names_by_prefix = {p: names for p, (names, _) in zip(prefixes, listings)}
    n_listing_errors = sum(n for _, n in listings)

    def day_files(spec: ProductSpec) -> List[RemoteFile]:
        files = select_files(names_by_prefix[spec.gcs_product_prefix], spec.filename_must_contain)
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
    loop = asyncio.get_running_loop()
    inflight = asyncio.Semaphore(max(1, cfg.max_inflight_payloads))
    glm_ok = np.zeros(len(glm_files), dtype=bool)
    glm_tables: Dict[int, Dict[str, np.ndarray]] = {}
    errors: List[str] = []

    def record_error(blob: str, e: BaseException) -> None:
        errors.append(blob)
        if len(errors) <= MAX_ERRORS_LOGGED_PER_DAY:
            log(f"!! {os.path.basename(blob)}: {e!r}")

    async def download(blob: str) -> bytes:
        return await http_get(cfg, session, f"https://storage.googleapis.com/{cfg.gcs_bucket}/{blob}", as_json=False)

    async def run_abi(states: List[AbiProductState], slot: int, f: RemoteFile) -> None:
        async with inflight:
            try:
                payload = await download(f.blob)
                variables = [(st.spec.data_variable, st.spec.dqf_variable, st.spec.precision) for st in states]
                decoded = await loop.run_in_executor(decode_pool, decode_abi_file, payload, variables)
                del payload
                # Validate every product before writing any, so a bad file leaves no half-written slot
                for state, dec in zip(states, decoded):
                    state.accept_first_or_validate(root, dec)
                for state, dec in zip(states, decoded):
                    await asyncio.to_thread(write_abi_frame, state, slot, dec)
                    state.record_scan(slot, dec, f)
            except Exception as e:
                record_error(f.blob, e)

    async def run_glm(i: int, f: RemoteFile) -> None:
        async with inflight:
            try:
                payload = await download(f.blob)
                glm_tables[i] = await loop.run_in_executor(decode_pool, decode_glm_file, payload)
                glm_ok[i] = True
            except Exception as e:
                record_error(f.blob, e)

    await asyncio.gather(
        *(run_abi(states, slot, f) for states, slot, f in abi_items),
        *(run_glm(i, f) for i, f in enumerate(glm_files)),
    )

    # --- per-slot metadata, GLM tables, completion
    slot_start_ns = to_ns(day_start) + np.arange(SLOTS_PER_DAY, dtype=np.int64) * SLOT_SECONDS * 1_000_000_000
    for state in abi_states.values():
        state.write_slot_arrays(root, slot_start_ns)

    if crop is not None and crop.rect is not None:
        root.attrs["crop_scan_angle_rect"] = list(crop.rect)

    n_flashes = write_glm_tables(root, glm_files, glm_ok, glm_tables) if cfg.glm_product is not None else 0

    complete = n_listing_errors == 0 and not errors
    root.attrs.update(
        {
            "n_listing_errors": n_listing_errors,
            "n_file_errors": len(errors),
            "n_abi_slot_collisions": n_slot_collisions,
            "n_glm_files": len(glm_files),
            "n_glm_flashes": n_flashes,
            "complete": complete,
            "completed_utc": datetime.now(timezone.utc).isoformat() if complete else None,
        }
    )

    # One metadata file per day, so opening many days with xarray is fast
    zarr.consolidate_metadata(root.store)

    shutil.rmtree(out_path, ignore_errors=True)
    shutil.move(tmp_path, out_path)

    status = "OK" if complete else f"INCOMPLETE (listing errors={n_listing_errors}, file errors={len(errors)}; rerun to retry)"
    log(f"-> {day_label} {status}: {out_path}  flashes={n_flashes}")
    return complete


# =============================================================================
# Runner
# =============================================================================

async def run_all_days(cfg: RawConfig) -> None:
    days = list(iter_days_utc(cfg.start_date_utc, cfg.end_date_utc))
    os.makedirs(cfg.raw_output_root_dir, exist_ok=True)

    connector = aiohttp.TCPConnector(
        limit=cfg.max_concurrent_http_requests,
        ttl_dns_cache=300,
    )
    day_queue: asyncio.Queue[datetime] = asyncio.Queue()
    for d in days:
        day_queue.put_nowait(d)

    ok: List[str] = []
    failed: List[str] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        with ProcessPoolExecutor(max_workers=cfg.max_decode_processes) as pool:

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
                    (ok if good else failed).append(label)

            await asyncio.gather(*(worker() for _ in range(max(1, cfg.max_days_in_flight))))

    print("\n" + "=" * 72)
    print(f"Days attempted:  {len(days)}")
    print(f"Days complete:   {len(ok)}")
    print(f"Days incomplete: {len(failed)}" + (f"  {sorted(failed)}" if failed else ""))
    print(f"Output root:     {os.path.abspath(cfg.raw_output_root_dir)}")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Download GOES ABI + GLM into per-day raw Zarr stores.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    args = p.parse_args()
    asyncio.run(run_all_days(load_config(args.config)))


if __name__ == "__main__":
    main()
