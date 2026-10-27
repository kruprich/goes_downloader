"""The JSON config: every defined product, the ones to download, and the run settings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .progress import log


@dataclass(frozen=True)
class ProductSpec:
    key: str
    product_prefix: str
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
    # The public bucket as a URL: "s3://noaa-goes19" (AWS) or "gs://gcp-public-data-goes-19" (Google Cloud)
    bucket: str
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


BUCKET_PATTERN = re.compile(r"(s3|gs)://([a-z0-9][a-z0-9._-]*[a-z0-9])/?")
CMIP_BAND_PATTERN = re.compile(r"CMIP\w*-M\d+C(\d{2})")  # single-band files: ...-CMIPC-M6C13
MCMIP_BAND_VARIABLE = re.compile(r"CMI_C(\d{2})")  # the all-band MCMIP files hold one CMI_Cnn per band


def is_cmip_ir_band(product: Dict[str, Any]) -> bool:
    """CMIP bands 7-16 hold brightness temperature (K); bands 1-6 are reflectance (0-1)."""
    m = CMIP_BAND_PATTERN.search(str(product.get("filename_must_contain", ""))) or MCMIP_BAND_VARIABLE.fullmatch(
        str(product.get("data_variable", ""))
    )
    return bool(m) and int(m.group(1)) >= 7


# Every key the downloader or the feature step reads. Others are reported (they are usually typos, which would
# silently fall back to a default); keys starting with "_" are skipped silently, e.g. "_comment".
TOP_LEVEL_KEYS = {
    "bucket", "start_date_utc", "end_date_utc", "raw_output_root_dir", "overwrite_existing",
    "download_method", "byte_range_processes", "max_concurrent_http_requests", "http_timeout_seconds",
    "http_retries", "max_decode_processes", "max_days_in_flight", "max_inflight_payloads",
    "bounds", "cmip_precision", "download", "abi_products", "glm_product",
    "features",  # read by the feature step (goes_downloader/features)
    "mrms",  # read by download_mrms.py (goes_downloader/mrms.py)
}
ABI_PRODUCT_KEYS = {"key", "description", "product_prefix", "filename_must_contain", "data_variable", "dqf_variable", "precision"}
GLM_PRODUCT_KEYS = {"key", "description", "product_prefix", "filename_must_contain"}
BOUNDS_KEYS = {"lat_min", "lat_max", "lon_min", "lon_max"}


def unknown_keys(raw: Dict[str, Any]) -> List[str]:
    """Keys in a loaded config that the downloader doesn't read, as dotted paths."""
    def check(d: Any, known: set, where: str) -> List[str]:
        if not isinstance(d, dict):
            return []
        return [f"{where}{k}" for k in d if k not in known and not str(k).startswith("_")]

    out = check(raw, TOP_LEVEL_KEYS, "")
    for i, p in enumerate(raw.get("abi_products") or []):
        name = p.get("key", i) if isinstance(p, dict) else i
        out += check(p, ABI_PRODUCT_KEYS, f"abi_products[{name}].")
    out += check(raw.get("glm_product"), GLM_PRODUCT_KEYS, "glm_product.")
    out += check(raw.get("bounds"), BOUNDS_KEYS, "bounds.")
    return out


def load_config(config_path: str) -> RawConfig:
    """Reads config.json; logs any key it doesn't use (see TOP_LEVEL_KEYS)."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    unknown = unknown_keys(raw)
    if unknown:
        log(f"!! {config_path}: ignoring keys the downloader doesn't use (typos?): {', '.join(unknown)}")

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
                product_prefix=_require(p, "product_prefix", str),
                filename_must_contain=_require(p, "filename_must_contain", str),
                data_variable=_require(p, "data_variable", str),
                dqf_variable=_require(p, "dqf_variable", str),
                precision=float(precision) if precision is not None else None,
            )
        )
    glm_raw = _require(raw, "glm_product", dict)
    defined_glm = ProductSpec(
        key=_require(glm_raw, "key", str),
        product_prefix=_require(glm_raw, "product_prefix", str),
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

    bucket = BUCKET_PATTERN.fullmatch(_require(raw, "bucket", str))
    if bucket is None:
        raise ValueError(f"bucket must be s3://<name> (AWS) or gs://<name> (Google Cloud), got {raw['bucket']!r}")

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
        bucket=f"{bucket.group(1)}://{bucket.group(2)}",
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
