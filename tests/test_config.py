"""Config loading, the crop helpers, and the entry script."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from goes_downloader.config import load_config, unknown_keys
from goes_downloader.crop import crop_slice

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_config(tmp_path, **changes):
    cfg = json.load(open(os.path.join(REPO, "config.json")))
    cfg.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_repo_config_loads():
    cfg = load_config(os.path.join(REPO, "config.json"))
    assert cfg.abi_products and cfg.glm_product is not None
    assert unknown_keys(json.load(open(os.path.join(REPO, "config.json")))) == []


def test_unknown_keys_are_reported(tmp_path, capsys):
    raw = json.load(open(os.path.join(REPO, "config.json")))
    raw["byte_range_process"] = 32  # typo for byte_range_processes
    raw["_comment"] = "keys starting with _ are notes"
    raw["bounds"]["lat_mn"] = 20.0
    raw["abi_products"][0]["precison"] = 0.5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    cfg = load_config(str(path))  # still loads
    out = capsys.readouterr().out
    for key in ("byte_range_process", "bounds.lat_mn", "abi_products[cmip_c01].precison"):
        assert key in out
    assert "_comment" not in out
    assert cfg.byte_range_processes == 48  # the typo fell back to the default, hence the warning


def test_infrared_bands_get_cmip_precision_in_both_band_sets(tmp_path):
    raw = json.load(open(os.path.join(REPO, "config.json")))
    everything = [p["key"] for p in raw["abi_products"]] + [raw["glm_product"]["key"]]
    cfg = load_config(write_config(tmp_path, download=everything, cmip_precision=0.25))
    precision = {p.key: p.precision for p in cfg.abi_products}
    for band in range(1, 17):
        expected = 0.25 if band >= 7 else None
        assert precision[f"cmip_c{band:02d}"] == expected
        assert precision[f"mcmip_c{band:02d}"] == expected
    assert precision["acha"] is None and precision["dsi_cape"] is None


def test_mcmip_products_read_one_band_each_from_the_same_file():
    raw = json.load(open(os.path.join(REPO, "config.json")))
    mcmip = [p for p in raw["abi_products"] if p["key"].startswith("mcmip_")]
    assert len(mcmip) == 16
    assert {(p["product_prefix"], p["filename_must_contain"]) for p in mcmip} == {("ABI-L2-MCMIPC", "ABI-L2-MCMIPC-M6")}
    assert [p["data_variable"] for p in mcmip] == [f"CMI_C{b:02d}" for b in range(1, 17)]


def test_bad_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="download_method"):
        load_config(write_config(tmp_path, download_method="torrent"))
    with pytest.raises(ValueError, match="unknown"):
        load_config(write_config(tmp_path, download=["cmip_c99"]))


def test_byte_range_workers_each_get_a_file_in_flight(tmp_path):
    cfg = load_config(write_config(tmp_path, download_method="byte_ranges", byte_range_processes=40))
    assert cfg.max_inflight_payloads == 40
    cfg = load_config(write_config(tmp_path, download_method="whole_files", max_concurrent_http_requests=24))
    assert cfg.max_inflight_payloads == 24


def test_crop_slice_keeps_cells_overlapping_the_range():
    x = np.arange(10) * 1.0  # ascending, cells 1 wide
    assert crop_slice(x, 2.6, 5.2, "x") == slice(3, 6)  # the cell of 2 ends at 2.5, the cell of 6 starts at 5.5
    y = x[::-1]  # descending, as ABI y is
    assert crop_slice(y, 2.6, 5.2, "y") == slice(4, 7)
    with pytest.raises(ValueError):
        crop_slice(x, 20.0, 30.0, "x")


def test_entry_script_imports_nothing_outside_its_main_guard():
    """Spawned workers re-run download_raw.py; they must not load aiohttp or zarr through it."""
    code = (
        "import runpy, sys; runpy.run_path('download_raw.py', run_name='__mp_main__'); "
        "print(sorted(m for m in ('aiohttp', 'zarr', 'goes_downloader') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
