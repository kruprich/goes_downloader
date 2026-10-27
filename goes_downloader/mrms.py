"""
MRMS (NOAA's Multi-Radar Multi-Sensor radar mosaics) from the public AWS bucket, one raw Zarr store per
UTC day, beside the GOES stores and in the same spirit: every file, its values packed as published, cropped
to the config's "bounds" and nothing else.

  python download_mrms.py --config config.json

The config's "mrms" section lists the products; the dates, "bounds", "overwrite_existing" and HTTP
settings are the GOES download's. A product is a folder of the bucket holding one folder per day
(YYYYMMDD) of gzipped GRIB2 files, one field each, every 2 minutes for the CONUS mosaics, e.g.
CONUS/Reflectivity_-10C_00.50 (reflectivity at the -10 C isotherm, LightningCast v2's radar input).

Layout of <mrms.output_root_dir>/YYYY/YYYYDDD.zarr (Zarr format 3, like the GOES stores), one group per product:
  <key>/time[file]             datetime64 the file's time (from its name), ns since 1970
  <key>/value[file, y, x]      uint16  packed values as in the file's PNG image; rows run north to south
                                       physical = value * scale_factor[file] + add_offset[file]
                                       65535 (attrs fill_value; the array's fill value and _FillValue
                                       attribute) = file not downloaded or decoded (present 0)
  <key>/scale_factor[file]     float64 2**E / 10**D of the file's GRIB2 packing
  <key>/add_offset[file]       float64 R / 10**D
  <key>/present[file]          uint8   1 if the file was downloaded and decoded
  <key>/lat[y], <key>/lon[x]   float64 pixel centres (deg; lon in -180..180)
  MRMS marks its own gaps in the values: for the reflectivity products -999 is outside radar coverage and
  -99 is covered with no echo.
Group attrs: prefix, description, n_files (listed), n_present, has_frames (without any file decoded,
value doesn't exist), fill_value, source_shape and crop_y0/crop_x0 (where the stored pixels start in the
file's grid: those whose 0.01 deg cell overlaps the bounds).
Root attrs: day_utc, complete (every listing and file succeeded and the day ended at least 24 h ago;
incomplete days are rebuilt from scratch on the next run), too_recent, completed_utc, n_listing_errors,
n_file_errors, source_bucket, product_keys, bounds.

Decoding runs in threads: gzip, Pillow's PNG decoder and zstd release the GIL, so no process pool is
needed (README: MRMS).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from PIL import Image
from zarr.codecs import ZstdCodec

from . import remote
from .common import TIME_ATTRS, iter_days_utc, to_ns
from .config import BUCKET_PATTERN, LatLonBounds, RawConfig, load_config
from .day import MIN_AGE_FOR_COMPLETE, MAX_ERRORS_LOGGED_PER_DAY
from .progress import format_duration, log
from .store import CHUNK_KEYS, _write_array, consolidate, directory_size, output_zarr_path_for_day, zarr_store_is_complete

MRMS_KEYS = {"bucket", "output_root_dir", "threads", "products"}
MRMS_PRODUCT_KEYS = {"key", "prefix", "description"}
FILE_TIME = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")
FILL_VALUE = int(np.iinfo(np.uint16).max)
# A frame is mostly two sentinels. Plain zstd 9 stores it 9% smaller than Blosc zstd 3 with byte shuffle and
# decodes twice as fast; Blosc zstd 3 with bitshuffle stored 52% more than with byte shuffle, pcodec 19% more
COMPRESSOR = ZstdCodec(level=9)


@dataclass(frozen=True)
class MrmsProduct:
    key: str
    prefix: str  # the bucket folder holding one folder per day, e.g. CONUS/Reflectivity_-10C_00.50
    description: str = ""


@dataclass(frozen=True)
class MrmsConfig:
    raw: RawConfig  # dates, bounds, overwrite_existing and HTTP settings; its bucket is MRMS's
    output_root_dir: str
    threads: int  # decoding and writing; each thread holds a ~50 MB decoded field
    products: Tuple[MrmsProduct, ...]


def load_mrms_config(config_path: str) -> MrmsConfig:
    """The config's "mrms" section, with the GOES settings it shares."""
    raw = load_config(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        section = json.load(f).get("mrms")
    if not isinstance(section, dict):
        raise ValueError(f"{config_path} has no \"mrms\" object")
    unknown = sorted(k for k in section if k not in MRMS_KEYS and not str(k).startswith("_"))
    if unknown:
        raise ValueError(f"mrms: unknown keys {unknown}")
    bucket = BUCKET_PATTERN.fullmatch(str(section.get("bucket", "s3://noaa-mrms-pds")))
    if bucket is None:
        raise ValueError(f"mrms.bucket must be s3://<name> or gs://<name>, got {section['bucket']!r}")
    products = []
    for i, p in enumerate(section.get("products") or []):
        if not isinstance(p, dict) or not isinstance(p.get("key"), str) or not isinstance(p.get("prefix"), str):
            raise ValueError(f"mrms.products[{i}] needs a \"key\" and a \"prefix\" (e.g. CONUS/Reflectivity_-10C_00.50)")
        extra = sorted(k for k in p if k not in MRMS_PRODUCT_KEYS and not str(k).startswith("_"))
        if extra:
            raise ValueError(f"mrms.products[{i}]: unknown keys {extra}")
        products.append(MrmsProduct(p["key"], p["prefix"].strip("/"), str(p.get("description", ""))))
    keys = [p.key for p in products]
    if not products or len(set(keys)) != len(keys):
        raise ValueError(f"mrms.products must list products with unique keys, got {keys}")
    threads = section.get("threads", 8)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("mrms.threads must be a positive integer")
    return MrmsConfig(
        raw=replace(raw, bucket=f"{bucket.group(1)}://{bucket.group(2)}"),
        output_root_dir=str(section.get("output_root_dir", "./mrms_raw")),
        threads=threads,
        products=tuple(products),
    )


# =============================================================================
# GRIB2
# =============================================================================

def _signed(b: bytes) -> int:
    """GRIB2's signed integers: the top bit is the sign, the rest the magnitude."""
    v = int.from_bytes(b, "big")
    top = 1 << (8 * len(b) - 1)
    return -(v - top) if v & top else v


def decode_mrms(payload: bytes) -> Dict[str, Any]:
    """
    One gzipped MRMS GRIB2 file -> {value [lat, lon] uint16 as packed, scale_factor, add_offset, lat, lon}.
    MRMS files hold one field on a regular lat/lon grid (template 3.0, rows north to south) packed as a
    16-bit PNG image (template 5.41) without a bitmap; anything else is refused rather than misread.
    """
    b = gzip.decompress(payload)
    if b[:4] != b"GRIB" or b[7] != 2:
        raise ValueError("not a GRIB2 file")
    if int.from_bytes(b[8:16], "big") != len(b):
        raise ValueError("expected exactly one GRIB2 message")
    sections: Dict[int, bytes] = {}
    p = 16
    while b[p : p + 4] != b"7777":
        n, number = int.from_bytes(b[p : p + 4], "big"), b[p + 4]
        if number in sections or n < 5:
            raise ValueError(f"unexpected GRIB2 section {number} at byte {p}: one field per file expected")
        sections[number] = b[p : p + n]
        p += n
    s3, s5, s6, s7 = sections[3], sections[5], sections[6], sections[7]
    if int.from_bytes(s3[12:14], "big") != 0 or s3[71] != 0:
        raise ValueError("expected a lat/lon grid (template 3.0) scanned west to east, north to south")
    if int.from_bytes(s5[9:11], "big") != 41 or s6[5] != 255:
        raise ValueError("expected PNG packing (template 5.41) without a bitmap")
    ni, nj = int.from_bytes(s3[30:34], "big"), int.from_bytes(s3[34:38], "big")
    lat1, lon1 = _signed(s3[46:50]) * 1e-6, int.from_bytes(s3[50:54], "big") * 1e-6
    di, dj = int.from_bytes(s3[63:67], "big") * 1e-6, int.from_bytes(s3[67:71], "big") * 1e-6
    reference = struct.unpack(">f", s5[11:15])[0]
    e, d = _signed(s5[15:17]), _signed(s5[17:19])
    value = np.asarray(Image.open(io.BytesIO(s7[5:])))
    if value.shape != (nj, ni) or value.dtype != np.uint16:
        raise ValueError(f"PNG image is {value.dtype} {value.shape}, expected uint16 {(nj, ni)}")
    return {
        "value": value,
        "scale_factor": 2.0**e / 10.0**d,
        "add_offset": reference / 10.0**d,
        "lat": lat1 - dj * np.arange(nj),
        "lon": (lon1 + di * np.arange(ni) + 180.0) % 360.0 - 180.0,
    }


def crop_window(lat: np.ndarray, lon: np.ndarray, bounds: Optional[LatLonBounds]) -> Tuple[slice, slice]:
    """Rows and columns of the pixels whose cell overlaps the bounds (every pixel without bounds)."""
    if bounds is None:
        return slice(0, len(lat)), slice(0, len(lon))
    half_lat, half_lon = abs(lat[1] - lat[0]) / 2, abs(lon[1] - lon[0]) / 2
    rows = np.flatnonzero((lat + half_lat > bounds.lat_min) & (lat - half_lat < bounds.lat_max))
    cols = np.flatnonzero((lon + half_lon > bounds.lon_min) & (lon - half_lon < bounds.lon_max))
    if not rows.size or not cols.size:
        raise ValueError("the bounds don't overlap the MRMS grid")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


# =============================================================================
# One day
# =============================================================================

def day_files(objects: Dict[str, int], day: datetime) -> List[Tuple[datetime, str]]:
    """(time, object name) of the day's files, by time; hours outside remote.LIST_HOURS are left out (tests)."""
    files = []
    for name in objects:
        m = FILE_TIME.search(name)
        if m is None:
            continue
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if day <= t < day + timedelta(days=1) and t.hour in remote.LIST_HOURS:
            files.append((t, name))
    return sorted(files)


@dataclass
class ProductDay:
    """One product's files on one day; the frame array is created from the first file decoded."""

    product: MrmsProduct
    files: List[Tuple[datetime, str]]
    value: Any = None
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    ys: slice = slice(None)
    xs: slice = slice(None)
    scale_factor: np.ndarray = field(init=False)
    add_offset: np.ndarray = field(init=False)
    present: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.files)
        self.scale_factor, self.add_offset = np.full(n, np.nan), np.full(n, np.nan)
        self.present = np.zeros(n, np.uint8)

    def accept(self, root: Any, dec: Dict[str, Any], bounds: Optional[LatLonBounds]) -> None:
        """Creates the frame array from the first file decoded; later files must be on its grid."""
        if self.value is not None:
            if not (np.array_equal(dec["lat"], self.lat) and np.array_equal(dec["lon"], self.lon)):
                raise ValueError(f"{self.product.key}: grid differs from the day's first file")
            return
        self.lat, self.lon = dec["lat"], dec["lon"]
        self.ys, self.xs = crop_window(self.lat, self.lon, bounds)
        shape = (self.ys.stop - self.ys.start, self.xs.stop - self.xs.start)
        g = root.require_group(self.product.key)
        g.attrs.update({"source_shape": list(dec["value"].shape), "crop_y0": self.ys.start, "crop_x0": self.xs.start})
        # The fill is also the _FillValue attribute: xarray masks that, not a Zarr v3 array's fill value
        self.value = g.create_array("value", shape=(len(self.files), *shape), chunks=(1, *shape), chunk_key_encoding=CHUNK_KEYS,
                                    dtype="u2", fill_value=FILL_VALUE, compressors=COMPRESSOR, dimension_names=["time", "y", "x"],
                                    attributes={"_FillValue": FILL_VALUE, "coordinates": "lat lon"}, overwrite=True)

    def write(self, i: int, dec: Dict[str, Any]) -> None:
        # One chunk per file, so concurrent writes never touch the same chunk
        self.value[i] = dec["value"][self.ys, self.xs]
        self.scale_factor[i], self.add_offset[i] = dec["scale_factor"], dec["add_offset"]
        self.present[i] = 1

    def write_file_arrays(self, root: Any) -> None:
        g = root.require_group(self.product.key)
        _write_array(g, "time", np.array([to_ns(t) for t, _ in self.files], np.int64), ["time"], **TIME_ATTRS)
        _write_array(g, "scale_factor", self.scale_factor, ["time"])
        _write_array(g, "add_offset", self.add_offset, ["time"])
        _write_array(g, "present", self.present, ["time"])
        if self.value is not None:
            _write_array(g, "lat", self.lat[self.ys], ["y"], units="degrees_north")
            _write_array(g, "lon", self.lon[self.xs], ["x"], units="degrees_east")
        g.attrs.update({"prefix": self.product.prefix, "description": self.product.description, "fill_value": FILL_VALUE,
                        "n_files": len(self.files), "n_present": int(self.present.sum()), "has_frames": self.value is not None})


async def process_mrms_day(cfg: MrmsConfig, day: datetime, session: Any) -> Tuple[bool, int]:
    """Builds one day's store; returns (complete, bytes downloaded)."""
    label = day.date().isoformat()
    out_path = output_zarr_path_for_day(cfg.output_root_dir, day)
    if not cfg.raw.overwrite_existing and zarr_store_is_complete(out_path):
        log(f"SKIP {label} already complete")
        return True, 0

    listings = await asyncio.gather(
        *(remote.list_object_names(cfg.raw, session, f"{p.prefix}/{day:%Y%m%d}/") for p in cfg.products), return_exceptions=True
    )
    states, n_listing_errors = [], 0
    for p, objects in zip(cfg.products, listings):
        if isinstance(objects, BaseException):
            n_listing_errors += 1
            log(f"!! listing failed for {p.prefix}/{day:%Y%m%d}/: {objects!r}")
            objects = {}
        states.append(ProductDay(p, day_files(objects, day)))

    tmp_path = out_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(tmp_path, mode="w", zarr_format=3)
    root.attrs.update({"day_utc": label, "source_bucket": cfg.raw.bucket, "product_keys": [p.key for p in cfg.products],
                       "bounds": vars(cfg.raw.bounds) if cfg.raw.bounds is not None else None, "complete": False})

    # Downloads run ahead of the decoding threads by at most as many files again
    inflight = asyncio.Semaphore(2 * cfg.threads)
    errors: List[str] = []
    downloaded = 0

    async def run(state: ProductDay, i: int, name: str) -> None:
        nonlocal downloaded
        async with inflight:
            try:
                payload = await remote.http_get(cfg.raw, session, remote.object_url(cfg.raw.bucket, name), as_json=False)
                downloaded += len(payload)
                dec = await asyncio.to_thread(decode_mrms, payload)
                del payload
                state.accept(root, dec, cfg.raw.bounds)  # in the event loop, so the first file creates the array once
                await asyncio.to_thread(state.write, i, dec)
            except Exception as e:
                errors.append(name)
                if len(errors) <= MAX_ERRORS_LOGGED_PER_DAY:
                    log(f"!! {os.path.basename(name)}: {e!r}")

    await asyncio.gather(*(run(st, i, name) for st in states for i, (_, name) in enumerate(st.files)))

    for st in states:
        st.write_file_arrays(root)
    too_recent = datetime.now(timezone.utc) < day + timedelta(days=1) + MIN_AGE_FOR_COMPLETE
    complete = n_listing_errors == 0 and not errors and not too_recent
    root.attrs.update({"n_listing_errors": n_listing_errors, "n_file_errors": len(errors), "too_recent": too_recent,
                       "complete": complete, "completed_utc": datetime.now(timezone.utc).isoformat() if complete else None})
    consolidate(root)
    shutil.rmtree(out_path, ignore_errors=True)
    shutil.move(tmp_path, out_path)

    counts = ", ".join(f"{st.product.key} {int(st.present.sum())}/{len(st.files)} files" for st in states)
    status = "OK" if complete else (
        "INCOMPLETE (ended less than 24 h ago; rerun later)" if too_recent and n_listing_errors == 0 and not errors
        else f"INCOMPLETE (listing errors={n_listing_errors}, file errors={len(errors)}; rerun to retry)")
    log(f"-> {label} {status}: {counts}, {downloaded / 1e9:.2f} GB downloaded, {directory_size(out_path) / 1e9:.2f} GB stored")
    return complete, downloaded


async def run_all_days(cfg: MrmsConfig) -> None:
    import aiohttp

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(cfg.threads))
    days = list(iter_days_utc(cfg.raw.start_date_utc, cfg.raw.end_date_utc))
    t0, ok, failed, total = time.monotonic(), [], [], 0
    connector = aiohttp.TCPConnector(limit=cfg.raw.max_concurrent_http_requests, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        for day in days:
            try:
                good, n = await process_mrms_day(cfg, day, session)
            except Exception as e:
                log(f"!! {day.date().isoformat()} failed: {e!r}")
                good, n = False, 0
            (ok if good else failed).append(day.date().isoformat())
            total += n

    print("\n" + "=" * 72)
    print(f"Days complete:   {len(ok)} of {len(days)}")
    print(f"Days incomplete: {len(failed)}" + (f"  {failed}" if failed else ""))
    print(f"Downloaded:      {total / 1e9:.1f} GB in {format_duration(time.monotonic() - t0)}")
    print(f"Output root:     {os.path.abspath(cfg.output_root_dir)}")
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Download MRMS radar products from AWS into per-day raw Zarr stores.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file (its \"mrms\" section)")
    args = p.parse_args(argv)
    cfg = load_mrms_config(args.config)
    try:
        asyncio.run(run_all_days(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted. Finished days are kept; run the same command again to continue.", file=sys.stderr)
        sys.exit(130)
