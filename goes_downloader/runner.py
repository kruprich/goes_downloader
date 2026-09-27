"""The whole run: every day in the configured range, a few at a time, and the summary."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from typing import List

import aiohttp

from . import progress as progress_module
from .common import iter_days_utc
from .config import RawConfig, load_config
from .day import process_single_day
from .decode import DecodePool
from .progress import Progress, format_duration, log
from .report import print_report


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

    progress = progress_module.current = Progress(len(days))

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
                    # Not reached when Ctrl-C cancels the day, so it doesn't count as done
                    progress.finish_day(label)
                    (ok if good else failed).append(label)

            await asyncio.gather(*(worker() for _ in range(max(1, cfg.max_days_in_flight))))
        finally:
            pool.shutdown()
            redraw_task.cancel()
            progress.close()
            progress_module.current = None

    elapsed = time.monotonic() - progress.started
    print("\n" + "=" * 72)
    print(f"Days attempted:  {len(days)}")
    print(f"Days complete:   {len(ok)}")
    print(f"Days incomplete: {len(failed)}" + (f"  {sorted(failed)}" if failed else ""))
    downloaded = progress.bytes_downloaded
    print(f"Downloaded:      {downloaded / 1e9:.1f} GB in {format_duration(elapsed)}"
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
    p.add_argument("--no-report", action="store_true", help="Don't print the data report when the run finishes")
    args = p.parse_args()
    cfg = load_config(args.config)
    try:
        asyncio.run(run_all_days(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted. Finished days are kept; run the same command again to continue.", file=sys.stderr)
        sys.exit(130)
    if not args.no_report:
        # The whole date range, including days finished by earlier runs (same as report.py)
        print(f"\nData report for {cfg.start_date_utc} to {cfg.end_date_utc} (also: python report.py --config ...):")
        print_report(cfg)
