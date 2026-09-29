"""Building the per-day feature stores (build_day, in worker processes), the run over a date range, and --stats."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common import SLOTS_PER_DAY, TIME_ATTRS, iter_days_utc
from ..progress import format_duration, log
from .expressions import Evaluator, make_plan
from .glm import SLOT_NS, GlmTimeline, disk_offsets, neighbourhood_sum
from .grid import SourceFrame, SolarGeometry, TargetGrid, aggregate, grid_latlon, make_grid
from .raw import EPOCH, RawTimeline, day_of_index
from .settings import SLOT_MINUTES, ChannelSpec, FeatureConfig, LabelSpec, geography_store_path, load_feature_config

LABEL_UNKNOWN = 255
FLOAT16_MAX = float(np.finfo(np.float16).max)


def features_day_path(cfg: FeatureConfig, day: datetime) -> str:
    return os.path.join(cfg.output_root_dir, str(day.year), f"{day.year}{day.timetuple().tm_yday:03d}.zarr")


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
    elif c.kind == "solar":
        d = {"name": c.name, "solar": c.solar}
    else:
        d = {"name": c.name, "expr": c.expr}
    return {**d, "description": c.description} if c.description else d


def label_attrs(l: LabelSpec) -> Dict[str, Any]:
    d = {"name": l.name, "start_minutes": l.start_slots * SLOT_MINUTES, "end_minutes": l.end_slots * SLOT_MINUTES,
         "min_flashes": l.min_flashes, "radius_cells": l.radius_cells, "source": l.source}
    return {**d, "description": l.description} if l.description else d


def build_day(cfg: FeatureConfig, grid: TargetGrid, day_iso: str) -> Dict[str, Any]:
    """Builds one day's feature store. Runs in a worker process."""
    import zarr

    from ..store import _write_array, directory_size

    t_start = time.monotonic()
    day = datetime.strptime(day_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_path = features_day_path(cfg, day)
    raw = RawTimeline(cfg, grid)
    s_day = (day - EPOCH).days * SLOTS_PER_DAY
    if raw.day(s_day // SLOTS_PER_DAY).root is None:
        return {"day": day_iso, "status": "no raw store", "complete": False}

    plan = make_plan(cfg.channels, [l.end_slots for l in cfg.labels])
    step = cfg.time_step_slots
    shape = (grid.ny, grid.nx)
    lat, lon = grid_latlon(grid)
    solar = SolarGeometry(lat, lon) if any(c.kind == "solar" for c in cfg.channels) else None
    s0 = s_day - plan.lookback
    out_slots = list(range(s_day, s_day + SLOTS_PER_DAY, step))
    outputs = [c for c in cfg.channels if c.output]

    glm: Optional[GlmTimeline] = None
    if cfg.uses_glm:
        # Inputs from the lookback, flash windows ending at each issue time, label windows after it
        glm_s0 = min(s0, s_day + 1 - step)
        glm_s1 = out_slots[-1] + 1 + plan.lookahead
        glm = GlmTimeline(cfg, grid, raw, glm_s0, max(glm_s1, s_day + SLOTS_PER_DAY), extent=cfg.uses_events)

    # --- store (built at *.tmp and moved into place when done)
    tmp_path = out_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(zarr.DirectoryStore(tmp_path), mode="w")
    n_t = len(out_slots)
    cy, cx = (grid.ny, grid.nx) if cfg.chunk_yx is None else (min(cfg.chunk_yx[0], grid.ny), min(cfg.chunk_yx[1], grid.nx))
    codec, level, shuffle = cfg.compression
    compressor = zarr.Blosc(cname=codec, clevel=level, shuffle={"byte": zarr.Blosc.SHUFFLE, "bit": zarr.Blosc.BITSHUFFLE, "none": zarr.Blosc.NOSHUFFLE}[shuffle])

    def frames(name: str, lead: List[int], dtype: Any, fill: Any, dims: List[str]) -> Any:
        a = root.create_dataset(name, shape=(n_t, *lead, *shape), chunks=(1, *lead, cy, cx), dtype=dtype,
                                fill_value=fill, compressor=compressor, overwrite=True)
        a.attrs.update({"_ARRAY_DIMENSIONS": dims, "coordinates": "lat lon"})
        return a

    features = frames("features", [len(outputs)], cfg.dtype, np.nan, ["time", "channel", "y", "x"])
    labels_arr = frames("labels", [len(cfg.labels)], np.uint8, LABEL_UNKNOWN, ["time", "label", "y", "x"]) if cfg.labels else None
    # Every slot is written, so no fill value: zarr's default of 0 would make xarray read 0 flashes as missing
    flash_arr = frames("flash_count", [], np.uint16, None, ["time", "y", "x"]) if glm is not None else None
    extent_arr = frames("flash_extent", [], np.uint16, None, ["time", "y", "x"]) if glm is not None and glm.has_extent else None

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
    u16 = np.iinfo(np.uint16).max

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
            elif ch.kind == "solar":
                if "solar" not in per_slot:  # at the middle of the slot, close to the scan time
                    per_slot["solar"] = solar.frames(s * SLOT_NS + SLOT_NS // 2)
                frame = per_slot["solar"][ch.solar]
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
            flash_arr[k] = np.minimum(glm.sums(a, s + 1).reshape(shape), u16).astype(np.uint16)
            if extent_arr is not None:
                extent_arr[k] = np.minimum(glm.sums(a, s + 1, "extent").reshape(shape), u16).astype(np.uint16)
            glm_coverage[k] = glm.coverage_mean(a, s + 1)
            n_flashes += glm.n_flashes(a, s + 1)
        if labels_arr is not None:
            out = np.empty((len(cfg.labels), *shape), dtype=np.uint8)
            for i, l in enumerate(cfg.labels):
                a, b = s + 1 + l.start_slots, s + 1 + l.end_slots
                if glm.coverage_mean(a, b) < cfg.glm_min_coverage:
                    out[i] = LABEL_UNKNOWN
                    continue
                counts = glm.sums(a, b, "extent" if l.source == "extent" else "count").reshape(shape)
                hits = neighbourhood_sum(counts, label_offsets[l.name]) >= l.min_flashes
                out[i] = hits
                label_counts[l.name][0] += hits.size
                label_counts[l.name][1] += int(hits.sum())
            labels_arr[k] = out

    # --- coordinates, per-time arrays, attrs
    slot_ns = np.array(out_slots, dtype=np.int64) * SLOT_NS
    _write_array(root, "time", slot_ns, ["time"], **TIME_ATTRS, long_name="ABI slot start; issue time is 5 minutes later")
    _write_array(root, "channel", np.array([c.name for c in outputs]), ["channel"])
    _write_array(root, "x", grid.x(), ["x"], units="rad", long_name="ABI fixed grid E/W scan angle")
    _write_array(root, "y", grid.y(), ["y"], units="rad", long_name="ABI fixed grid N/S scan angle")
    _write_array(root, "lat", lat, ["y", "x"], units="degrees_north")
    _write_array(root, "lon", lon, ["y", "x"], units="degrees_east")
    _write_array(root, "channel_valid_fraction", valid_fraction, ["time", "channel"])
    if cfg.labels:
        _write_array(root, "label", np.array([l.name for l in cfg.labels]), ["label"])
    if glm is not None:
        _write_array(root, "glm_coverage", glm_coverage, ["time"])

    raw_days = {}
    for d, rd in sorted(raw.days.items()):
        raw_days[day_of_index(d).date().isoformat()] = "missing" if rd.root is None else ("complete" if rd.complete else "incomplete")
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
            "glm": {"quality_flags": cfg.glm_quality_flags, "lightning_ellipsoid_km": cfg.glm_ellipsoid_km,
                    "min_coverage": cfg.glm_min_coverage, "extent": bool(glm is not None and glm.has_extent)},
            "geography_path": cfg.geography_path,
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
        "bytes": directory_size(out_path),
        "n_flashes": n_flashes,
        "overflow": {k: v for k, v in n_overflow.items() if v},
    }


def check_geography(cfg: FeatureConfig, grid: TargetGrid) -> None:
    """The geography store, if there is one, must be on the same grid as the features."""
    from .geography import geography_grid

    path = geography_store_path(cfg)
    geo = geography_grid(path)
    if geo is None:
        log(f"no geography store at {path} yet; run build_geography.py once (it doesn't block building features)")
    elif not geo.same_cells(grid):
        raise ValueError(f"{path} was built on a different grid ({geo.ny} x {geo.nx}) than these features "
                         f"({grid.ny} x {grid.nx}); rebuild it with build_geography.py --overwrite")


def run_build(cfg: FeatureConfig) -> None:
    from ..store import zarr_store_is_complete

    days = [d.date().isoformat() for d in iter_days_utc(cfg.start_date_utc, cfg.end_date_utc)]
    grid = make_grid(cfg)
    check_geography(cfg, grid)
    plan = make_plan(cfg.channels, [l.end_slots for l in cfg.labels])
    outputs = [c.name for c in cfg.channels if c.output]
    lat, lon = grid_latlon(grid)
    log(f"grid: {grid.ny} x {grid.nx} cells of {grid.resolution_km:g} km (nominal, {grid.coarsen}x{grid.coarsen} {grid.reference_product} pixels), "
        f"lat {np.nanmin(lat):.1f}..{np.nanmax(lat):.1f}, lon {np.nanmin(lon):.1f}..{np.nanmax(lon):.1f}")
    per_day = len(range(0, SLOTS_PER_DAY, cfg.time_step_slots)) * len(outputs) * grid.ny * grid.nx * np.dtype(cfg.dtype).itemsize
    log(f"{len(outputs)} channels ({', '.join(outputs)}), {len(cfg.labels)} labels, every {cfg.time_step_slots * SLOT_MINUTES} min; "
        f"lookback {plan.lookback * SLOT_MINUTES} min; features {per_day / 1e9:.1f} GB/day before compression")

    todo = [d for d in days if cfg.overwrite_existing or not zarr_store_is_complete(features_day_path(cfg, datetime.strptime(d, "%Y-%m-%d")))]
    log(f"{len(days) - len(todo)} of {len(days)} days already complete; building {len(todo)}")

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
                log(f"!! {d} failed: {e!r}")
                failed.append(d)
                continue
            (ok if r["complete"] else failed).append(d)
            total_bytes += r.get("bytes", 0)
            eta = format_duration((time.monotonic() - t0) / i * (len(todo) - i))
            extra = f"  float16 overflow set to NaN: {r['overflow']}" if r.get("overflow") else ""
            if "seconds" in r:
                log(f"-> {d} {r['status']} in {r['seconds']:.0f}s, {r['bytes'] / 1e6:.0f} MB, {r['n_flashes']} flashes  [{i}/{len(todo)}, ETA {eta}]{extra}")
            else:
                log(f"-> {d} {r['status']}  [{i}/{len(todo)}]")

    print("\n" + "=" * 72)
    print(f"Days built:      {len(ok) + len(failed)} of {len(todo)} ({len(ok)} complete)")
    print(f"Days incomplete: {len(failed)}" + (f"  {sorted(failed)}" if failed else ""))
    print(f"Saved to disk:   {total_bytes / 1e9:.2f} GB in {format_duration(time.monotonic() - t0)}")
    print(f"Output root:     {os.path.abspath(cfg.output_root_dir)}")
    print("=" * 72)


def run_stats(cfg: FeatureConfig) -> None:
    """Combines the per-day sums into mean/std per channel and positive rates per label."""
    import zarr

    sums: Dict[str, Dict[str, float]] = {}
    labels: Dict[str, List[int]] = {}
    used, incomplete, missing, cells = [], [], [], 0
    for day in iter_days_utc(cfg.start_date_utc, cfg.end_date_utc):
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


def features_main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build CNN features and lightning labels from the raw GOES day stores.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    p.add_argument("--start", help="First UTC day (YYYY-MM-DD); overrides features.start_date_utc")
    p.add_argument("--end", help="Last UTC day (YYYY-MM-DD); overrides features.end_date_utc")
    p.add_argument("--workers", type=int, help="Days built in parallel; overrides features.max_workers")
    p.add_argument("--overwrite", action="store_true", help="Rebuild days that are already complete")
    p.add_argument("--stats", action="store_true", help="Combine per-day stats over the date range instead of building")
    args = p.parse_args(argv)

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


def geography_main(argv: Optional[List[str]] = None) -> None:
    from .geography import build_geography, geography_grid

    p = argparse.ArgumentParser(description="Put the geography fields (terrain, land/water, viewing angle) on the feature grid, once.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    p.add_argument("--overwrite", action="store_true", help="Rebuild the geography store even if it matches the grid")
    args = p.parse_args(argv)

    cfg = load_feature_config(args.config)
    grid = make_grid(cfg)
    path = geography_store_path(cfg)
    existing = geography_grid(path)
    if existing is not None and not args.overwrite:
        if existing.same_cells(grid):
            log(f"{path} already exists on this grid; nothing to do (--overwrite rebuilds it)")
            return
        raise SystemExit(f"{path} exists on a different grid; rebuild it with --overwrite")
    t0 = time.monotonic()
    out = build_geography(cfg, grid)
    import zarr

    g = zarr.open_consolidated(out, mode="r")
    summary = ", ".join(f"{n} {st['min']:.4g}..{st['max']:.4g}" for n, st in g.attrs["stats"].items() if st["count"])
    log(f"wrote {out} ({grid.ny} x {grid.nx}) in {format_duration(time.monotonic() - t0)}: {summary}")
