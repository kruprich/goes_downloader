"""One UTC day: list, download, decode, validate and write its store."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np
import zarr

from . import progress
from .common import GLM_WINDOWS_PER_DAY, SLOT_SECONDS, SLOTS_PER_DAY, to_ns
from .config import ProductSpec, RawConfig
from .crop import DayCrop
from .decode import DecodePool, decode_abi_file, decode_glm_file
from .progress import log
from .ranges import ranged_decode_abi_file, ranged_decode_glm_file
from .remote import RemoteFile, http_get, list_day_names, object_url, select_files
from .store import (
    AbiProductState,
    directory_size,
    output_zarr_path_for_day,
    write_abi_frame,
    write_glm_tables,
    zarr_store_is_complete,
)

# A day is only marked complete once it ended this long ago; NOAA publishes some files late
MIN_AGE_FOR_COMPLETE = timedelta(hours=24)
MAX_ERRORS_LOGGED_PER_DAY = 5


def infer_satellite_label_from_bucket(bucket: str) -> str:
    m = re.search(r"goes-?(\d+)", bucket)  # noaa-goes19, gcp-public-data-goes-19
    return f"GOES-{int(m.group(1))}" if m else "UNKNOWN"


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
    prefixes = sorted({spec.product_prefix for spec in specs})
    listings = await asyncio.gather(*(list_day_names(cfg, session, p, day) for p in prefixes))
    objects_by_prefix = {p: objects for p, (objects, _) in zip(prefixes, listings)}
    n_listing_errors = sum(n for _, n in listings)

    def day_files(spec: ProductSpec) -> List[RemoteFile]:
        files = select_files(objects_by_prefix[spec.product_prefix], spec.filename_must_contain)
        return [f for f in files if day_start <= f.start < day_end]

    glm_files = day_files(cfg.glm_product) if cfg.glm_product is not None else []

    # --- ABI slot assignment: slot = 5-min window containing the scan start. Products that read
    # the same files (e.g. the DSI indices) are grouped so each file is downloaded once.
    crop = DayCrop(cfg.bounds) if cfg.bounds is not None else None
    abi_states = {spec.key: AbiProductState(spec, crop) for spec in cfg.abi_products}
    same_files: Dict[Tuple[str, str], List[AbiProductState]] = {}
    for spec in cfg.abi_products:
        same_files.setdefault((spec.product_prefix, spec.filename_must_contain), []).append(abi_states[spec.key])

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
    if progress.current is not None:
        progress.current.start_day(progress_key, sum(f.size for _, _, f in abi_items) + sum(f.size for f in glm_files))

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
            "source_bucket": cfg.bucket,
            "source_satellite_label": infer_satellite_label_from_bucket(cfg.bucket),
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
        payload = await http_get(cfg, session, object_url(cfg.bucket, blob), as_json=False)
        if progress.current is not None:
            progress.current.add_bytes(progress_key, len(payload))
        return payload

    async def run_ranged(f: RemoteFile, fn: Any, *args: Any) -> Any:
        decoded, n_downloaded = await decode_pool.run(fn, object_url(cfg.bucket, f.blob), f.size, *args, *http_args)
        if progress.current is not None:
            progress.current.add_bytes(progress_key, f.size, n_downloaded)
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
    if progress.current is not None:
        progress.current.add_saved(await asyncio.to_thread(directory_size, out_path))

    if complete:
        status = "OK"
    elif too_recent and n_listing_errors == 0 and not errors:
        status = "INCOMPLETE (ended less than 24 h ago; NOAA may still publish files, rerun later)"
    else:
        status = f"INCOMPLETE (listing errors={n_listing_errors}, file errors={len(errors)}; rerun to retry)"
    gaps = f"  GLM windows missing={GLM_WINDOWS_PER_DAY - n_windows_covered}" if cfg.glm_product is not None else ""
    log(f"-> {day_label} {status}: {out_path}  flashes={n_flashes} events={n_events}{gaps}")
    return complete
