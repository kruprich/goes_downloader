"""Writing the stores: ABI frames and their fill values; the GLM tables (flash order, event links, per-file counts, coverage)."""

import os
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import zarr

from conftest import chunk_dirs
from goes_downloader.common import GLM_WINDOWS_PER_DAY, SLOT_SECONDS, SLOTS_PER_DAY, to_ns
from goes_downloader.config import ProductSpec
from goes_downloader.decode import decode_abi_file
from goes_downloader.remote import RemoteFile
from goes_downloader.store import AbiProductState, consolidate, write_abi_frame, write_glm_tables, zarr_store_is_complete

DAY = datetime(2025, 10, 27, tzinfo=timezone.utc)


def codecs(array):
    return [c["name"] for c in array.metadata.to_dict()["codecs"]]


def glm_table(first_seconds, lats, event_flash):
    """One decoded file's tables; each flash is identified by its latitude."""
    n = len(lats)
    return {
        "time_first_ns": np.array([to_ns(DAY + timedelta(seconds=s)) for s in first_seconds], dtype=np.int64),
        "time_last_ns": np.array([to_ns(DAY + timedelta(seconds=s + 0.2)) for s in first_seconds], dtype=np.int64),
        "lat": np.array(lats, dtype=np.float32),
        "lon": np.zeros(n, np.float32),
        "area_m2": np.ones(n, np.float32),
        "energy_j": np.ones(n, np.float32),
        "quality_flag": np.zeros(n, np.uint8),
        "event_lat": np.array([lats[i] if i >= 0 else -99.0 for i in event_flash], dtype=np.float32),  # on its flash
        "event_lon": np.zeros(len(event_flash), np.float32),
        "event_flash": np.array(event_flash, dtype=np.int32),
    }


def test_events_point_at_their_flash_after_sorting(tmp_path):
    files = [
        RemoteFile(blob=f"f{i}", start=DAY + timedelta(seconds=20 * i), end=DAY + timedelta(seconds=20 * i + 20),
                   created=DAY, size=1)
        for i in range(4)
    ]
    # File 1 failed; flashes are out of time order within and across files
    tables = {
        0: glm_table([15.0, 2.0, 9.0], [10.0, 11.0, 12.0], [0, 0, 2, 1, 2, -1]),  # the last is an orphan
        2: glm_table([41.0, 40.5], [20.0, 21.0], [1, 1, 0]),
        3: glm_table([], [], [-1, -1]),  # last file: no flashes, only orphan events
    }
    root = zarr.open_group(str(tmp_path / "day.zarr"), mode="w")
    n_flashes, n_events, n_covered = write_glm_tables(root, DAY, files, np.array([True, False, True, True]), tables)
    assert (n_flashes, n_events, n_covered) == (5, 11, 3)

    flashes, events = root["glm/flashes"], root["glm/events"]
    times = flashes["time_first_ns"][:]
    assert (np.diff(times) >= 0).all()
    assert flashes["lat"][:].tolist() == [11.0, 12.0, 10.0, 21.0, 20.0]
    assert flashes["file_index"][:].tolist() == [0, 0, 0, 2, 2]
    # Every event still points at its own flash (same latitude) after the flashes were sorted
    index = events["flash_index"][:]
    assert index.tolist().count(-1) == 3 and events["lat"][:][index == -1].tolist() == [-99.0] * 3
    assert (flashes["lat"][:][index[index >= 0]] == events["lat"][:][index >= 0]).all()
    assert root["glm/files/n_flashes"][:].tolist() == [3, 0, 2, 0]
    assert root["glm/files/n_events"][:].tolist() == [6, 0, 3, 2]
    assert root["glm/files/n_orphan_events"][:].tolist() == [1, 0, 0, 2] and root.attrs["n_glm_orphan_events"] == 3
    covered = root["glm/coverage/covered"][:]
    assert len(covered) == GLM_WINDOWS_PER_DAY and covered[:4].tolist() == [1, 0, 1, 1] and covered.sum() == 3
    # pcodec for the columns, except the 8-bit one it can't encode
    for name in ("flashes/time_first_ns", "flashes/lat", "events/lat", "events/flash_index"):
        assert codecs(root[f"glm/{name}"]) == ["numcodecs.pcodec"], name
    assert codecs(root["glm/flashes/quality_flag"]) == ["bytes", "blosc"]


def test_a_day_without_glm_data_writes_empty_tables(tmp_path):
    root = zarr.open_group(str(tmp_path / "day.zarr"), mode="w", zarr_format=3)
    assert write_glm_tables(root, DAY, [], np.zeros(0, dtype=bool), {}) == (0, 0, 0)
    assert root["glm/flashes/lat"].shape == (0,) and root["glm/events/flash_index"][:].tolist() == []
    assert root["glm/coverage/covered"][:].sum() == 0


def test_abi_frames_keep_their_fill_for_zarr_and_xarray(tmp_path, abi_bytes):
    """
    Zarr v3: a slot without a scan reads as the fill, which is also the _FillValue attribute xarray masks
    with (it ignores the array's fill value); arrays written in full have no _FillValue.
    """
    path = str(tmp_path / "day.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=3)
    (dec,) = decode_abi_file(abi_bytes, [("CMI_C13", "DQF_C13", None)])
    state = AbiProductState(ProductSpec("c13", "ABI-L2-MCMIPC", "MCMIPC", "CMI_C13", "DQF_C13"))
    state.accept_first_or_validate(root, dec)
    write_abi_frame(state, 5, dec)
    state.record_scan(5, dec, RemoteFile(blob="f", start=DAY + timedelta(minutes=25), end=DAY + timedelta(minutes=28), created=DAY))
    state.write_slot_arrays(root, to_ns(DAY) + np.arange(SLOTS_PER_DAY, dtype=np.int64) * SLOT_SECONDS * 10**9)
    root.attrs["complete"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # zarr's warning that consolidated metadata isn't in the v3 spec is silenced
        consolidate(root)

    assert zarr_store_is_complete(path)
    g = zarr.open_consolidated(path, mode="r")["abi/c13"]
    value, dqf = g["value"], g["dqf"]
    assert codecs(value) == ["numcodecs.pcodec"] and codecs(dqf) == ["bytes", "blosc"]  # pcodec has no 8-bit types
    assert value.metadata.zarr_format == 3 and value.metadata.dimension_names == ("time", "y", "x")
    assert value.fill_value == value.attrs["_FillValue"] == g.attrs["fill_value"] == 65535
    assert dqf.fill_value == dqf.attrs["_FillValue"] == g.attrs["dqf_fill_value"] == 255
    np.testing.assert_array_equal(value[5], dec["value"])
    np.testing.assert_array_equal(dqf[5], dec["dqf"])
    assert (value[4] == 65535).all() and (dqf[4] == 255).all()
    for name in ("time", "present", "scale_factor", "scan_start_ns"):
        assert g[name].metadata.dimension_names == ("time",) and "_FillValue" not in g[name].attrs
    assert g["present"][:].tolist() == [0] * 5 + [1] + [0] * (SLOTS_PER_DAY - 6) and g.attrs["n_present"] == 1
    # Chunks are files in their array's directory (c.5.0.0), not zarr 3's default c/5/0/0 directories
    assert os.path.isfile(os.path.join(path, "abi/c13/value/c.5.0.0")) and chunk_dirs(path) == []
