"""
Shared fixtures. The offline tests build small NetCDF files shaped like the GOES ones and serve them
to the byte-range code through a fake remote, so they run in seconds without the network. Tests
marked `network` download real files and only run with `pytest --network`.
"""

import os
import sys

import netCDF4 as nc
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from goes_downloader import ranges  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", help="also run tests that download real GOES files")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="downloads real files; run with --network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


# GOES-East fixed grid, as in the real files
PROJECTION = {
    "grid_mapping_name": "geostationary",
    "perspective_point_height": 35786023.0,
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "inverse_flattening": 298.2572221,
    "latitude_of_projection_origin": 0.0,
    "longitude_of_projection_origin": -75.0,
    "sweep_angle_axis": "x",
}
# (band, scale_factor, add_offset): reflectance band 2, infrared band 13, and band 8 that tests never read
ABI_BANDS = [("C02", 0.00031, 0.0), ("C13", 0.0617, 89.62), ("C08", 0.0400, 138.0)]


def make_abi_file(path, seed=0):
    """A small MCMIPC-like file: x/y over the CONUS sector, chunked and compressed bands with DQF."""
    rng = np.random.default_rng(seed)
    ny, nx = 120, 160
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        for name, n, offset, step in (("x", nx, -0.101332, 0.000874), ("y", ny, 0.128212, -0.000700)):
            v = ds.createVariable(name, "i2", (name,))
            v.scale_factor, v.add_offset, v.units = step, offset, "rad"
            v.set_auto_maskandscale(False)
            v[:] = np.arange(n, dtype=np.int16)
        proj = ds.createVariable("goes_imager_projection", "i4")
        proj.setncatts(PROJECTION)
        for band, scale, offset in ABI_BANDS:
            v = ds.createVariable(f"CMI_{band}", "i2", ("y", "x"), zlib=True, chunksizes=(40, 40), fill_value=np.int16(-1))
            v.setncatts({"_Unsigned": "true", "scale_factor": scale, "add_offset": offset, "units": "K"})
            raw = rng.integers(0, 4000, (ny, nx)).astype(np.int16)
            raw[rng.random((ny, nx)) < 0.02] = -1
            v.set_auto_maskandscale(False)
            v[:] = raw
            q = ds.createVariable(f"DQF_{band}", "i1", ("y", "x"), zlib=True, chunksizes=(40, 40), fill_value=np.int8(-1))
            q.setncatts({"_Unsigned": "true"})
            q.set_auto_maskandscale(False)
            q[:] = rng.integers(0, 4, (ny, nx)).astype(np.int8)
    return open(path, "rb").read()


def make_glm_file(path, seed=0):
    """
    A small LCFA-like file. Ids are shuffled so rows and ids differ, and every flash has a group and
    every group an event. Returns (bytes, event_flash): the flash row of each event.
    """
    rng = np.random.default_rng(seed)
    n_flash, n_group, n_event = 40, 300, 900
    flash_id = rng.permutation(np.arange(1000, 1000 + n_flash)).astype(np.int16)
    group_id = rng.permutation(np.arange(50000, 50000 + n_group)).astype(np.int32)
    group_flash = np.concatenate([np.arange(n_flash), rng.integers(0, n_flash, n_group - n_flash)])
    event_group = np.concatenate([np.arange(n_group), rng.integers(0, n_group, n_event - n_group)])
    rng.shuffle(event_group)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        for dim, n in (("number_of_events", n_event), ("number_of_groups", n_group), ("number_of_flashes", n_flash)):
            ds.createDimension(dim, n)

        def var(name, dtype, dim, data, **attrs):
            v = ds.createVariable(name, dtype, (dim,), chunksizes=(min(64, len(data)),))
            v.setncatts(attrs)
            v.set_auto_maskandscale(False)
            v[:] = data

        ev, gr, fl = "number_of_events", "number_of_groups", "number_of_flashes"
        seconds = "seconds since 2025-10-27 18:00:00.000"
        var("event_id", "i4", ev, np.arange(n_event))
        var("event_time_offset", "i2", ev, rng.integers(0, 20000, n_event), scale_factor=0.000381, add_offset=-5.0, units=seconds)
        var("event_lat", "i2", ev, rng.integers(0, 30000, n_event), _Unsigned="true", scale_factor=0.00203128, add_offset=-66.56)
        var("event_lon", "i2", ev, rng.integers(0, 30000, n_event), _Unsigned="true", scale_factor=0.00203128, add_offset=-141.56)
        var("event_energy", "i2", ev, rng.integers(0, 30000, n_event), scale_factor=1.9e-17, add_offset=2.8e-16)
        var("event_parent_group_id", "i4", ev, group_id[event_group])
        var("group_id", "i4", gr, group_id)
        var("group_lat", "f4", gr, rng.uniform(-50, 50, n_group))
        var("group_parent_flash_id", "i2", gr, flash_id[group_flash])
        var("flash_id", "i2", fl, flash_id)
        var("flash_time_offset_of_first_event", "i2", fl, rng.integers(0, 20000, n_flash), scale_factor=0.000381, add_offset=-5.0, units=seconds)
        var("flash_time_offset_of_last_event", "i2", fl, rng.integers(0, 20000, n_flash), scale_factor=0.000381, add_offset=-5.0, units=seconds)
        var("flash_lat", "f4", fl, rng.uniform(-50, 50, n_flash))
        var("flash_lon", "f4", fl, rng.uniform(-150, -20, n_flash))
        var("flash_area", "i2", fl, rng.integers(0, 30000, n_flash), _Unsigned="true", scale_factor=152.6, add_offset=0.0)
        var("flash_energy", "i2", fl, rng.integers(0, 30000, n_flash), _Unsigned="true", scale_factor=1.5e-15, add_offset=0.0)
        var("flash_quality_flag", "i2", fl, rng.integers(0, 2, n_flash), _FillValue=np.int16(-1))
    return open(path, "rb").read(), group_flash[event_group].astype(np.int32)


@pytest.fixture
def abi_bytes(tmp_path):
    return make_abi_file(tmp_path / "abi.nc")


@pytest.fixture
def glm_file(tmp_path):
    return make_glm_file(tmp_path / "glm.nc")


class FakeRemote:
    def __init__(self):
        self.files = {}
        self.requests = []  # (path, start, end) of every request

    def get_range(self, path, start, end, timeout, retries):
        self.requests.append((path, start, end))
        return self.files[path][start:end]


@pytest.fixture
def junk_fill(monkeypatch):
    """Byte-range buffers start filled with random bytes instead of zeros, so decoding anything
    that was not fetched changes the result (in this process only, not in pool workers)."""
    init = ranges._RangeFile.__init__

    def init_with_junk(self, path, size, timeout, retries):
        init(self, path, size, timeout, retries)
        self.buf[:] = os.urandom(size)

    monkeypatch.setattr(ranges._RangeFile, "__init__", init_with_junk)


@pytest.fixture
def remote(monkeypatch, junk_fill):
    """
    A fake bucket for the byte-range code (remote.files[path] = bytes), with junk_fill. Fetch sizes
    are scaled down to the small test files, so most metadata is fetched on demand.
    """
    fake = FakeRemote()
    for name, size in (("RANGE_HEAD_BYTES", 4096), ("RANGE_TAIL_BYTES", 4096), ("RANGE_BLOCK_BYTES", 1024), ("RANGE_MERGE_GAP_BYTES", 512)):
        monkeypatch.setattr(ranges, name, size)
    monkeypatch.setattr(ranges, "_get_range", fake.get_range)
    return fake
