"""The data-availability report, on small stores that carry only the attributes it reads."""

import json
import os
from datetime import datetime, timezone

import zarr

from goes_downloader.config import load_config
from goes_downloader.report import day_rows, problems, summary
from goes_downloader.store import output_zarr_path_for_day

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_store(root_dir, day, complete=True, scans=288, glm_missing=0):
    g = zarr.open_group(output_zarr_path_for_day(root_dir, day), mode="w")
    g.attrs.update(
        complete=complete, too_recent=False, n_listing_errors=0, n_file_errors=0 if complete else 3,
        abi_product_keys=["mcmip_c13"], glm_included=True, n_glm_windows_missing=glm_missing,
        n_glm_flashes=1000, n_glm_events=60000,
    )
    g.require_group("abi/mcmip_c13").attrs["n_present"] = scans
    zarr.consolidate_metadata(g.store)


def test_report_statuses_and_problems(tmp_path):
    root = str(tmp_path / "raw")
    day = lambda d: datetime(2025, 10, d, tzinfo=timezone.utc)  # noqa: E731
    make_store(root, day(1))
    make_store(root, day(2), complete=False)
    make_store(root, day(3), scans=250)  # 38 scans missing
    make_store(root, day(4), glm_missing=500)  # 11.6% of the GLM windows
    os.makedirs(output_zarr_path_for_day(root, day(5)) + ".tmp")  # a run stopped while building it
    # day 6 was never written

    cfg = json.load(open(os.path.join(REPO, "config.json")))
    cfg.update(start_date_utc="2025-10-01", end_date_utc="2025-10-06", raw_output_root_dir=root)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    rows = list(day_rows(load_config(str(tmp_path / "config.json"))))

    assert [r["status"] for r in rows] == ["complete", "incomplete", "complete", "complete", "interrupted", "missing"]
    assert [problems(r) for r in rows] == [
        [],
        ["incomplete"],
        ["mcmip_c13 250/288 scans"],
        ["GLM 11.6% of windows missing"],
        ["interrupted"],
        ["missing"],
    ]
    text = "\n".join(summary(rows))
    assert "6 in range, 3 complete, 1 incomplete, 1 interrupted, 1 missing" in text
    assert "4,000 flashes, 240,000 events" in text
