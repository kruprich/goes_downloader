"""Decoding: rounding packed values, and linking GLM events to their flashes."""

import numpy as np
import pytest

from goes_downloader.decode import _parent_rows, decode_glm_file, quantize_packed


def test_quantize_rounds_to_powers_of_two_within_precision():
    rng = np.random.default_rng(0)
    fill = 65535
    value = rng.integers(0, 4096, 100_000).astype(np.uint16)
    value[:100] = fill
    scale = 0.0617  # band 13 (K per count)
    rounded, step = quantize_packed(value, fill, scale, precision=0.25)
    assert step == pytest.approx(4 * scale) and step <= 0.25  # 2**2 counts is the largest step <= 0.25 K
    assert (rounded[:100] == fill).all()
    data = rounded[100:].astype(np.int64)
    assert (data % 4 == 0).all() and data.max() < fill
    error = data - value[100:].astype(np.int64)
    assert np.abs(error).max() <= 2
    assert abs(error.mean()) < 0.02  # ties go to even, so no bias


@pytest.mark.parametrize("precision", [None, 0.1])  # none requested, or finer than two counts
def test_quantize_leaves_values_alone(precision):
    value = np.arange(1000, dtype=np.uint16)
    rounded, step = quantize_packed(value, None, 0.0617, precision)
    assert rounded is value and step == pytest.approx(0.0617)


def test_parent_rows():
    ids = np.array([30, 10, 20])
    assert _parent_rows(np.array([20, 30, 30, 10]), ids, "x").tolist() == [2, 0, 0, 1]
    assert _parent_rows(np.array([], dtype=np.int64), ids, "x").tolist() == []
    with pytest.raises(ValueError, match="missing"):
        _parent_rows(np.array([40]), ids, "x")
    with pytest.raises(ValueError, match="missing"):
        _parent_rows(np.array([1]), np.array([], dtype=np.int64), "x")
    with pytest.raises(ValueError, match="duplicate"):
        _parent_rows(np.array([10]), np.array([10, 10]), "x")


def test_glm_events_link_to_their_flash(glm_file):
    data, event_flash = glm_file
    decoded = decode_glm_file(data)
    assert decoded["event_flash"].tolist() == event_flash.tolist()
    assert len(decoded["event_lat"]) == len(decoded["event_lon"]) == len(event_flash)
    n_flash = len(decoded["lat"])
    assert all(len(decoded[k]) == n_flash for k in ("time_first_ns", "time_last_ns", "lon", "area_m2", "energy_j", "quality_flag"))
    # Physical values come from the packed ones (lat/lon in degrees, times in 2025)
    assert np.nanmin(decoded["event_lat"]) >= -66.6 and np.nanmax(decoded["event_lat"]) <= 0.0
    assert (decoded["time_first_ns"] > np.datetime64("2025-10-27T17:59", "ns").astype(np.int64)).all()


def test_glm_file_with_a_missing_parent_is_an_error(tmp_path):
    import netCDF4 as nc
    from conftest import make_glm_file

    path = tmp_path / "glm.nc"
    make_glm_file(path)
    with nc.Dataset(path, "a") as ds:
        v = ds["group_parent_flash_id"]
        v.set_auto_maskandscale(False)
        v[0] = 32000  # no such flash
    with pytest.raises(ValueError, match="group_parent_flash_id"):
        decode_glm_file(path.read_bytes())
