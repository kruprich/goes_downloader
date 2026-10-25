"""
Real files from the configured public bucket (run with `pytest --network`; downloads roughly 0.5 GB),
and the same listing from the other cloud. 2025-10-27 18 UTC is the reference hour.
"""

import asyncio
import dataclasses
import json
import os
import urllib.request
from datetime import datetime, timezone

import aiohttp
import pytest
import zarr

from conftest import REPO, same
from goes_downloader import ranges, remote, runner
from goes_downloader.config import load_config
from goes_downloader.crop import crop_slice, scan_angle_rect
from goes_downloader.decode import decode_abi_file, decode_glm_file

pytestmark = pytest.mark.network

DAY = datetime(2025, 10, 27, tzinfo=timezone.utc)
HOUR = 18


async def list_hour(cfg, prefix, must_contain):
    remote.LIST_HOURS = [HOUR]
    try:
        async with aiohttp.ClientSession() as session:
            objects, failed = await remote.list_day_names(cfg, session, prefix, DAY)
    finally:
        remote.LIST_HOURS = range(24)
    assert failed == 0
    return remote.select_files(objects, must_contain)


def test_real_files_decode_the_same_from_byte_ranges(junk_fill):
    """For every configured source file type: two files, whole vs byte ranges with junk in unfetched bytes."""
    cfg = load_config(os.path.join(REPO, "config.json"))
    groups = {}
    for p in cfg.abi_products:
        groups.setdefault((p.product_prefix, p.filename_must_contain), []).append(p)
    jobs = [(prefix, must, [(p.data_variable, p.dqf_variable, p.precision) for p in products]) for (prefix, must), products in groups.items()]
    jobs.append((cfg.glm_product.product_prefix, cfg.glm_product.filename_must_contain, None))
    for prefix, must, variables in jobs:
        for f in asyncio.run(list_hour(cfg, prefix, must))[:2]:
            url = remote.object_url(cfg.bucket, f.blob)
            whole_bytes = urllib.request.urlopen(url).read()
            if variables is None:
                whole = decode_glm_file(whole_bytes)
                partial, downloaded = ranges.ranged_decode_glm_file(url, f.size, 60, 5)
                assert all(same(whole[k], partial[k]) for k in whole), f.blob
            else:
                whole = decode_abi_file(whole_bytes, variables)
                partial, downloaded = ranges.ranged_decode_abi_file(url, f.size, variables, cfg.bounds, 60, 5)
                x0, x1, y0, y1 = scan_angle_rect(cfg.bounds, whole[0]["projection"])
                ys, xs = crop_slice(whole[0]["y"], y0, y1, "y"), crop_slice(whole[0]["x"], x0, x1, "x")
                assert partial[0]["window"] == (ys.start, ys.stop, xs.start, xs.stop), f.blob
                for w, p in zip(whole, partial):
                    assert same(w["value"][ys, xs], p["value"][ys, xs]) and same(w["dqf"][ys, xs], p["dqf"][ys, xs]), f.blob
                    assert all(w[k] == p[k] for k in ("scale_factor", "add_offset", "fill_value", "dqf_fill_value", "units")), f.blob
            assert downloaded <= f.size


def test_both_clouds_list_the_same_files(monkeypatch):
    """AWS and Google Cloud hold the same objects for the reference hour; small pages exercise the paging."""
    cfg = load_config(os.path.join(REPO, "config.json"))
    monkeypatch.setattr(remote, "LIST_PAGE_SIZE", 50)
    listings = {}
    for bucket in ("s3://noaa-goes19", "gs://gcp-public-data-goes-19"):
        cfg = dataclasses.replace(cfg, bucket=bucket)
        listings[bucket] = {
            prefix: {f.blob: f.size for f in asyncio.run(list_hour(cfg, prefix, ""))}
            for prefix in sorted({p.product_prefix for p in cfg.abi_products} | {cfg.glm_product.product_prefix})
        }
    aws, gcs = listings.values()
    assert aws == gcs
    assert len(aws["GLM-L2-LCFA"]) == 180 and len(aws["ABI-L2-MCMIPC"]) == 12


def test_hour_is_the_same_with_whole_files_and_byte_ranges(tmp_path, monkeypatch):
    """The whole pipeline, both download methods, on a small product list: identical stores."""
    raw = json.load(open(os.path.join(REPO, "config.json")))
    raw.update(start_date_utc="2025-10-27", end_date_utc="2025-10-27", overwrite_existing=True,
               download=["cmip_c13", "dsi_cape", "dsi_li", "dsi_ki", "glm"])
    monkeypatch.setattr(remote, "LIST_HOURS", [HOUR])
    stores = {}
    for method in ("whole_files", "byte_ranges"):
        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps({**raw, "download_method": method, "raw_output_root_dir": str(tmp_path / method)}))
        asyncio.run(runner.run_all_days(load_config(str(path))))
        stores[method] = zarr.open_consolidated(str(tmp_path / method / "2025" / "2025300.zarr"), mode="r")
    a, b = stores["whole_files"], stores["byte_ranges"]
    assert a.attrs["complete"] and b.attrs["complete"] and a.attrs["n_glm_events"] > 0

    names = []
    a.visitvalues(lambda v: names.append(v.path) if isinstance(v, zarr.Array) else None)
    assert len(names) > 30
    for name in names:
        assert same(a[name][:], b[name][:]), name
