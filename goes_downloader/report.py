"""
Data-availability report over the day stores of a config's date range: which days are complete,
and how many ABI scans and GLM windows each has. Reads only each store's consolidated metadata.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, Iterator, List, Optional

import zarr

from .common import GLM_WINDOWS_PER_DAY, SLOTS_PER_DAY, iter_days_utc
from .config import RawConfig, load_config
from .store import output_zarr_path_for_day

# A day needs attention when it is not complete or has less than this share of the data
MIN_ABI_SCAN_SHARE = 0.98  # more than ~6 of 288 scans (30 min) missing for some product
MIN_GLM_COVERAGE = 0.99  # more than ~43 of 4320 windows (14 min) without GLM data


def day_rows(cfg: RawConfig) -> Iterator[Dict[str, Any]]:
    """One row per day in the range; scans_<key> per ABI product the store holds."""
    for day in iter_days_utc(cfg.start_date_utc, cfg.end_date_utc):
        path = output_zarr_path_for_day(cfg.raw_output_root_dir, day)
        row: Dict[str, Any] = {"date": day.date().isoformat()}
        if not os.path.exists(os.path.join(path, ".zmetadata")):
            # A .tmp store is a day that was being built when a run stopped
            row["status"] = "interrupted" if os.path.exists(path + ".tmp") else "missing"
            yield row
            continue
        root = zarr.open_consolidated(path, mode="r")
        a = root.attrs
        row["status"] = "complete" if a.get("complete") else ("too recent" if a.get("too_recent") else "incomplete")
        row["listing_errors"] = a.get("n_listing_errors")
        row["file_errors"] = a.get("n_file_errors")
        for key in a.get("abi_product_keys", []):
            row[f"scans_{key}"] = root[f"abi/{key}"].attrs.get("n_present", 0)
        if a.get("glm_included"):
            row["glm_windows_missing"] = a.get("n_glm_windows_missing")
            row["glm_flashes"] = a.get("n_glm_flashes")
            row["glm_events"] = a.get("n_glm_events")
        yield row


def problems(row: Dict[str, Any]) -> List[str]:
    """Why a day needs attention; empty if it doesn't."""
    if row["status"] != "complete":
        return [row["status"]]
    out = []
    scans = {key[6:]: n for key, n in row.items() if key.startswith("scans_")}
    low = {key: n for key, n in scans.items() if n < MIN_ABI_SCAN_SHARE * SLOTS_PER_DAY}
    if len(low) > 1 and len(low) == len(scans) and len(set(low.values())) == 1:
        out.append(f"every ABI product {next(iter(low.values()))}/{SLOTS_PER_DAY} scans")  # e.g. an outage
    else:
        out += [f"{key} {n}/{SLOTS_PER_DAY} scans" for key, n in low.items()]
    missing = row.get("glm_windows_missing")
    if missing is not None and missing > (1 - MIN_GLM_COVERAGE) * GLM_WINDOWS_PER_DAY:
        out.append(f"GLM {100 * missing / GLM_WINDOWS_PER_DAY:.1f}% of windows missing")
    return out


def summary(rows: List[Dict[str, Any]]) -> List[str]:
    stored = [r for r in rows if "listing_errors" in r]
    statuses = {s: sum(1 for r in rows if r["status"] == s) for s in ("complete", "incomplete", "too recent", "interrupted", "missing")}
    lines = [
        f"Days {rows[0]['date']} to {rows[-1]['date']}: {len(rows)} in range, "
        + ", ".join(f"{n} {s}" for s, n in statuses.items() if n)
    ]
    if stored:
        keys = sorted({k for r in stored for k in r if k.startswith("scans_")})
        shares = []
        for k in keys:
            have = [r[k] for r in stored if k in r]
            shares.append(f"{k[6:]} {100 * sum(have) / (SLOTS_PER_DAY * len(have)):.2f}%")
        lines.append("ABI scans present: " + "  ".join(shares))
        glm = [r for r in stored if r.get("glm_windows_missing") is not None]
        if glm:
            covered = 1 - sum(r["glm_windows_missing"] for r in glm) / (GLM_WINDOWS_PER_DAY * len(glm))
            flashes = sum(r.get("glm_flashes") or 0 for r in glm)
            events = sum(r.get("glm_events") or 0 for r in glm)
            lines.append(f"GLM: {100 * covered:.2f}% of 20 s windows covered, {flashes:,} flashes, {events:,} events")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Report which days and how much data the raw stores hold.")
    p.add_argument("--config", default="config.json", help="Path to JSON config file")
    p.add_argument("--all", action="store_true", help="List every day, not only those needing attention")
    p.add_argument("--csv", help="Also write every day's numbers to this CSV file")
    args = p.parse_args()
    print_report(load_config(args.config), show_all=args.all, csv_path=args.csv)


def print_report(cfg: RawConfig, show_all: bool = False, csv_path: Optional[str] = None) -> None:
    """The summary and the days needing attention (every day with show_all); optionally a CSV."""
    rows = list(day_rows(cfg))
    print("\n".join(summary(rows)))
    listed = [(r, problems(r)) for r in rows]
    if not show_all:
        listed = [(r, why) for r, why in listed if why]
    if listed:
        title = "Every day" if show_all else (
            f"Days needing attention (not complete, an ABI product under {MIN_ABI_SCAN_SHARE:.0%} of its scans, "
            f"or GLM under {MIN_GLM_COVERAGE:.0%} coverage)"
        )
        print(f"\n{title}:")
        for r, why in listed:
            errors = f"  file errors {r['file_errors']}" if r.get("file_errors") else ""
            print(f"  {r['date']}  {'; '.join(why) or 'ok'}{errors}")
    else:
        print("\nNo days need attention.")

    if csv_path:
        columns: List[str] = []
        for r in rows:
            columns += [k for k in r if k not in columns]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} days to {csv_path}")
