"""
The per-day raw Zarr store: its layout, described here, and the code that writes it.

Each store mirrors one UTC day of GOES ABI (CONUS L2) and GLM (LCFA) data without aggregation, so
binning, regridding and quality filtering can all happen later, in the feature step.

Layout of <raw_output_root_dir>/YYYY/YYYYDDD.zarr

Zarr format 3. Every array names its dimensions and each day's metadata is consolidated into its root
zarr.json, so xarray can open the stores (see "Reading with xarray" below). ABI frames and GLM columns are
encoded with pcodec (8-bit ones with Blosc zstd), so reading them needs numcodecs' pcodec package. Chunk
keys use "." (abi/<key>/value/c.<slot>.0.0), so each array's chunks are files in its own directory.

ABI: one group per configured product, one slot per 5-min CONUS scan (288 per day).
Each product keeps its source grid (CMIP bands 1, 3, 5: 1 km and band 2: 0.5 km in their own files, every
other band and every band of the all-band MCMIP file: 2 km, 1500x2500; ACM, ACTP: 2 km; ACHA, TPW, DSI: 10 km,
300x500).
  abi/<key>/time[slot]               datetime64 start of each 5-min slot (the xarray "time" coordinate)
  abi/<key>/value[slot, y, x]        uint16   packed values as in the file, rounded to the product's
                                              configured "precision" (physical units) if one is set
                                              physical = value * scale_factor[slot] + add_offset[slot]
                                              value == attrs["fill_value"] means no data (for files without a
                                              _FillValue the store uses 65535); it is the array's fill value
                                              and its _FillValue attribute
  abi/<key>/dqf[slot, y, x]          uint8    data quality flags as in the file (uint16 for products with
                                              16-bit flags; attrs["dqf_fill_value"] = fill, likewise)
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
  glm/files/{start_ns,end_ns,ok,n_flashes,n_events,n_orphan_events}[file]    one row per file NOAA published
  glm/flashes/{time_first_ns,time_last_ns,lat,lon,area_m2,energy_j,quality_flag,file_index}[flash]
                                       file_index is the row in the same day's glm/files; it is not
                                       unique once several days are combined
  glm/events/{lat,lon,flash_index}[event]   every event (one lit sensor pixel at one moment), in file
                                       order; flash_index is the row of its flash in the same day's
                                       glm/flashes, or -1 for an orphan event whose group or flash
                                       can't be found in its file (missing, or an id used twice). Events give each
                                       flash's footprint, which a flash-extent density needs
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
  too_recent             the day ended less than 24 h before the run (so complete is false)
  completed_utc          when a complete store was written (null otherwise)
  n_listing_errors, n_file_errors, n_abi_slot_collisions (extra scans in an already filled slot)
  n_glm_files, n_glm_flashes, n_glm_events, n_glm_windows_missing (null without GLM)
  n_glm_orphan_events    events stored with flash_index -1 (their group or flash isn't in the file)
  day_utc, year, julian_day, slots_per_day, slot_seconds, slot_rule, time_encoding
  source_bucket (s3:// or gs:// URL), source_satellite_label, abi_product_keys, glm_included
  bounds                 the configured lat/lon box (null for none); crop_scan_angle_rect is the
                         scan-angle rectangle (x_min, x_max, y_min, y_max) computed from it

ABI group attrs (abi/<key>):
  n_present, has_frames  scans stored, and whether any were (without any, value/dqf don't exist)
  fill_value, dqf_fill_value, units, data_variable, dqf_variable, source_filename_must_contain
  precision_requested    the product's rounding step in physical units (null: full precision)
  goes_imager_projection the file's projection attributes
  uncropped_shape, crop_y0, crop_x0   the source grid and where the stored crop starts in it

Reading with xarray (one product or GLM table across several days):
  xr.open_mfdataset(day_paths, engine="zarr", group="abi/mcmip_c13", combine="by_coords")
  xr.open_mfdataset(day_paths, engine="zarr", group="glm/flashes", combine="nested", concat_dim="flash")
  xarray turns value/dqf fill values into NaN (floats); pass mask_and_scale=False for the stored integers
  and apply scale_factor/add_offset yourself.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import PCodec

from .common import GLM_WINDOW_SECONDS, GLM_WINDOWS_PER_DAY, NAT_NS, SLOTS_PER_DAY, TIME_ATTRS, to_ns
from .config import ProductSpec
from .crop import DayCrop, crop_slice
from .remote import RemoteFile


def output_zarr_path_for_day(root_dir: str, day: datetime) -> str:
    return os.path.join(root_dir, str(day.year), f"{day.year}{day.timetuple().tm_yday:03d}.zarr")


def directory_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)


def zarr_store_is_complete(path: str) -> bool:
    if not os.path.exists(os.path.join(path, "zarr.json")):
        return False
    try:
        return bool(zarr.open_group(path, mode="r").attrs.get("complete", False))
    except Exception:
        return False


def consolidate(group: Any) -> None:
    """
    Copies every array's and group's metadata into the root zarr.json, so opening many days is fast.
    zarr warns that consolidated metadata isn't in the Zarr v3 specification yet: zarr-python reads it,
    and marks it so that other readers may ignore it and read each node's zarr.json.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Consolidated metadata", UserWarning)
        zarr.consolidate_metadata(group.store)


COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
# Chunk files c.<slot>.0.0 in the array's directory. zarr 3's default "/" makes c/<slot>/0/0, two 4 KB
# directories per frame chunk: ~28 GB over the configured range of raw, MRMS and feature days
CHUNK_KEYS = {"name": "default", "separator": "."}


def frame_codecs(dtype: Any) -> Dict[str, Any]:
    """
    create_array options for ABI frames and GLM columns. pcodec stores a raw day 18% smaller than Blosc
    zstd with bitshuffle for the same CPU (README: Size and speed), but has no 8-bit types: those (DQF,
    quality_flag) keep Blosc.
    """
    if np.dtype(dtype).itemsize == 1:
        return {"compressors": COMPRESSOR}
    return {"serializer": PCodec(level=8), "compressors": None}


def _write_array(group: Any, name: str, data: np.ndarray, dims: List[str], **attrs: Any) -> Any:
    """
    Writes a fully populated array with its dimension names for xarray. Its fill value is zarr's
    default (0), which xarray ignores in Zarr v3, so legitimate zeros aren't read as missing.
    """
    options = attrs.pop("zarr", {})
    return group.create_array(name, data=data, dimension_names=dims, attributes=attrs, chunk_key_encoding=CHUNK_KEYS,
                              overwrite=True, **options)


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
            common = dict(overwrite=True, chunks=(1, shape[0], shape[1]), chunk_key_encoding=CHUNK_KEYS, dimension_names=["time", "y", "x"])
            coordinates = "x_scan_angle_rad y_scan_angle_rad"
            # The fills are also the _FillValue attributes: xarray masks those, not a Zarr v3 array's fill value
            self.value_array = g.create_array(
                "value", shape=(SLOTS_PER_DAY, *shape), dtype="u2", fill_value=storage_fill,
                attributes={"_FillValue": storage_fill, "coordinates": coordinates}, **frame_codecs("u2"), **common
            )
            self.dqf_array = g.create_array(
                "dqf", shape=(SLOTS_PER_DAY, *shape), dtype=dec["dqf"].dtype, fill_value=dec["dqf_fill_value"],
                attributes={"_FillValue": dec["dqf_fill_value"], "coordinates": coordinates},
                **frame_codecs(dec["dqf"].dtype), **common
            )
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
    n_orphans = np.array([int((tables[i]["event_flash"] < 0).sum()) if i in tables else 0 for i in range(len(files))], dtype=np.int32)
    fg = g.require_group("files")
    _write_array(fg, "start_ns", np.array([to_ns(f.start) for f in files], dtype=np.int64), ["file"], **TIME_ATTRS)
    _write_array(fg, "end_ns", np.array([to_ns(f.end) for f in files], dtype=np.int64), ["file"], **TIME_ATTRS)
    _write_array(fg, "ok", file_ok.astype(np.uint8), ["file"])
    _write_array(fg, "n_flashes", n_flashes, ["file"])
    _write_array(fg, "n_events", n_events, ["file"])
    _write_array(fg, "n_orphan_events", n_orphans, ["file"])

    def write_column(group: Any, name: str, arr: np.ndarray, dim: str, **attrs: Any) -> None:
        _write_array(group, name, arr, [dim], **attrs, zarr=dict(chunks=(max(1, min(len(arr), 1 << 20)),), **frame_codecs(arr.dtype)))

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

    # Events in file order; flash_index is the row of the event's flash in glm/flashes, -1 for orphans
    sorted_row = np.empty(len(order), dtype=np.int32)
    sorted_row[order] = np.arange(len(order), dtype=np.int32)
    first_row = np.concatenate(([0], np.cumsum([len(tables[i]["lat"]) for i in file_indices]))).astype(np.int64)
    event_group = g.require_group("events")
    for k, dtype in GLM_EVENT_COLUMNS.items():
        if k == "flash_index":
            parts = []
            for j, i in enumerate(file_indices):
                local = tables[i]["event_flash"]
                rows = np.full(len(local), -1, dtype=np.int32)  # orphans stay -1
                rows[local >= 0] = sorted_row[first_row[j] + local[local >= 0]]
                parts.append(rows)
        else:
            parts = [tables[i][f"event_{k}"] for i in file_indices]
        write_column(event_group, k, np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype), "event")
    root.attrs["n_glm_orphan_events"] = int(n_orphans.sum())
    return int(n_flashes.sum()), int(n_events.sum()), int(covered.sum())
