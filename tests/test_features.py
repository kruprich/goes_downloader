"""
The feature step on two small synthetic raw days: regridding, derived channels across midnight, GLM
centroids and footprints, labels, completeness, geography and the sun/satellite geometry.
"""

import json
import os
import shutil
from dataclasses import replace
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pytest
import zarr

from conftest import PROJECTION, REPO
from goes_downloader.common import SLOTS_PER_DAY
from goes_downloader.crop import latlon_to_scan_angle
from goes_downloader.features.build import build_day, check_geography
from goes_downloader.features.geography import EARTH_RADIUS_KM, build_geography, rasterize_rings, regrid_source
from goes_downloader.features.grid import SolarGeometry, grid_latlon, make_grid, satellite_zenith_deg
from goes_downloader.features.settings import GeographySpec, load_feature_config
from goes_downloader.store import consolidate

X0, DX, Y0, DY = -0.101332, 5.6e-5, 0.128212, -5.6e-5  # uncropped CONUS 2 km pixel centres (rad)
CROP_Y0, CROP_X0, H, W = 600, 1200, 30, 40  # 2 km pixels kept; the 10 km product covers the same area
DAY1, DAY2 = "2025-06-01", "2025-06-02"
NS_DAY = 86400 * 10**9
T1 = int(np.datetime64(DAY1, "ns").astype(np.int64))
MIN = 60 * 10**9


def p2_value(slot: int, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return (slot * 7 + yy * 3 + xx) % 1000


def write_raw_day(root_dir: str, day: str, missing_slot: int = -1):
    """ABI part of a raw store: a 2 km product with bad DQF and fill pixels, and a 10 km product."""
    d = datetime.strptime(day, "%Y-%m-%d")
    path = os.path.join(root_dir, str(d.year), f"{d.year}{d.timetuple().tm_yday:03d}.zarr")
    root = zarr.open_group(path, mode="w")
    base = (d - datetime(1970, 1, 1)).days * SLOTS_PER_DAY
    yy, xx = np.meshgrid(CROP_Y0 + np.arange(H), CROP_X0 + np.arange(W), indexing="ij")
    v2 = np.stack([p2_value(base + s, yy, xx) for s in range(SLOTS_PER_DAY)]).astype(np.uint16)
    q2 = np.zeros_like(v2, dtype=np.uint8)
    for s in range(SLOTS_PER_DAY):
        q2[s][(yy + xx + s) % 11 == 0] = 2  # not in valid_dqf
        fill = (yy * xx + s) % 13 == 0
        v2[s][fill], q2[s][fill] = 65535, 255
    y10, x10 = np.meshgrid(CROP_Y0 // 5 + np.arange(H // 5), CROP_X0 // 5 + np.arange(W // 5), indexing="ij")
    v10 = np.stack([(y10 * 100 + x10 + s) % 5000 for s in range(SLOTS_PER_DAY)]).astype(np.uint16)
    present = np.ones(SLOTS_PER_DAY, np.uint8)
    if missing_slot >= 0:
        present[missing_slot] = 0
        v2[missing_slot], v10[missing_slot] = 65535, 65535
    for key, values, dqf, factor, (sf, ao) in (("p2", v2, q2, 1, (0.1, 200.0)), ("p10", v10, np.zeros_like(v10, np.uint8), 5, (1.0, 0.0))):
        g = root.require_group(f"abi/{key}")
        n, h, w = values.shape
        g.create_array("value", data=values, chunks=(1, h, w))
        g.create_array("dqf", data=dqf, chunks=(1, h, w))
        g.create_array("scale_factor", data=np.full(n, sf))
        g.create_array("add_offset", data=np.full(n, ao))
        g.create_array("present", data=present)
        # A 10 km pixel is 5x5 2 km pixels: its centre is the third 2 km centre
        g.create_array("x_scan_angle_rad", data=X0 + (factor - 1) / 2 * DX + (CROP_X0 // factor + np.arange(w)) * factor * DX)
        g.create_array("y_scan_angle_rad", data=Y0 + (factor - 1) / 2 * DY + (CROP_Y0 // factor + np.arange(h)) * factor * DY)
        g.attrs.update(fill_value=65535, dqf_fill_value=255, crop_y0=CROP_Y0 // factor, crop_x0=CROP_X0 // factor, has_frames=True,
                       goes_imager_projection=PROJECTION, uncropped_shape=[1500 // factor, 2500 // factor])
    root.attrs.update(day_utc=day, complete=True)
    consolidate(root)
    return path


def write_glm(path: str, flashes, covered: np.ndarray, orphans=()):
    """
    flashes: (time_ns, centroid (lat, lon), [event (lat, lon), ...]), in time order; orphans: events
    (lat, lon) whose flash isn't in their file, stored with flash_index -1 after the first flash's events.
    """
    root = zarr.open_group(path, mode="a")
    g = root.require_group("glm")
    f = g.require_group("flashes")
    f.create_array("time_first_ns", data=np.array([t for t, _, _ in flashes], np.int64))
    f.create_array("lat", data=np.array([c[0] for _, c, _ in flashes], np.float32))
    f.create_array("lon", data=np.array([c[1] for _, c, _ in flashes], np.float32))
    f.create_array("energy_j", data=np.full(len(flashes), 1e-13, np.float32))
    f.create_array("area_m2", data=np.full(len(flashes), 1e8, np.float32))
    f.create_array("quality_flag", data=np.zeros(len(flashes), np.uint8))
    e = g.require_group("events")
    ev = [(la, lo, i) for i, (_, _, events) in enumerate(flashes) for la, lo in events]
    n0 = len(flashes[0][2]) if flashes else 0
    ev[n0:n0] = [(la, lo, -1) for la, lo in orphans]
    e.create_array("lat", data=np.array([x[0] for x in ev], np.float32))
    e.create_array("lon", data=np.array([x[1] for x in ev], np.float32))
    e.create_array("flash_index", data=np.array([x[2] for x in ev], np.int32))
    # One file per flash, starting at its 20 s window, but never before the store's day: a flash that
    # starts just before midnight is in the next day's first file, as in NOAA's data
    day_start = int(np.datetime64(root.attrs["day_utc"], "ns").astype(np.int64))
    w = 20 * 10**9
    fl = g.require_group("files")
    fl.create_array("start_ns", data=np.array([max(day_start, t // w * w) for t, _, _ in flashes], np.int64))
    fl.create_array("n_events", data=np.array([len(evs) + (len(orphans) if i == 0 else 0) for i, (_, _, evs) in enumerate(flashes)], np.int32))
    g.require_group("coverage").create_array("covered", data=covered.astype(np.uint8))
    consolidate(root)


def config(tmp_path, end=DAY2, **features):
    raw = {
        "bucket": "s3://test", "start_date_utc": DAY1, "end_date_utc": end, "raw_output_root_dir": str(tmp_path / "raw"),
        "download": ["p2", "p10", "glm"],
        "abi_products": [
            {"key": "p2", "product_prefix": "A", "filename_must_contain": "A-M6", "data_variable": "V", "dqf_variable": "DQF"},
            {"key": "p10", "product_prefix": "B", "filename_must_contain": "B-M6", "data_variable": "V", "dqf_variable": "DQF"},
        ],
        "glm_product": {"key": "glm", "product_prefix": "G", "filename_must_contain": "G"},
        "features": {
            "output_root_dir": str(tmp_path / "features"),
            "start_date_utc": DAY1, "end_date_utc": DAY2,
            "grid": {"resolution_km": 4},
            "time_step_minutes": 15,
            "dtype": "float32",
            "glm": {"lightning_ellipsoid_km": None, "min_coverage": 0.95},
            "channels": [
                {"name": "a", "product": "p2"},
                {"name": "a_min", "product": "p2", "aggregation": "min"},
                {"name": "b", "product": "p10"},
                {"name": "cz", "solar": "cos_zenith", "output": False},
                {"name": "a_day", "expr": "mask(a, cz)"},
                {"name": "a_change", "expr": "a - lag(a, 15)"},
                {"name": "fed", "glm": "extent", "output": False},
                {"name": "fed15", "expr": "sum_past(fed, 15)"},
            ],
            "labels": [
                {"name": "ext30", "end_minutes": 30, "radius_cells": 1, "source": "extent"},
                {"name": "cen15", "end_minutes": 15, "radius_cells": 0, "source": "centroid"},
                # LightningCast's: from the scan's slot, on the 2 km pixels, peak of 1-minute densities
                {"name": "lc60", "start_minutes": -5, "end_minutes": 55, "resolution_km": 2, "fed_minutes": 1},
            ],
            "geography": {"path": "geography.zarr", "cache_dir": str(tmp_path / "static"), "fields": [
                {"name": "sat_zenith", "geometry": "satellite_zenith"},
                {"name": "height", "file": str(tmp_path / "dem.nc"), "variable": "z", "clip": [0, None]},
            ]},
            **features,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return str(path)


@pytest.fixture(scope="module")
def raw_days(tmp_path_factory):
    """
    Two raw days with GLM, written once for the module (writing them is most of the setup time);
    flashes are placed on known cells. Tests must not write into them.
    """
    shared = tmp_path_factory.mktemp("shared")
    write_raw_day(str(shared / "raw"), DAY1, missing_slot=99)  # the 08:15 output slot
    p2 = write_raw_day(str(shared / "raw"), DAY2)
    p1 = os.path.join(str(shared / "raw"), "2025", "2025152.zarr")
    grid = make_grid(load_feature_config(config(shared)))
    lat, lon = grid_latlon(grid)
    at = lambda r, c: (float(lat[r, c]), float(lon[r, c]))  # noqa: E731
    # Events 1/8 cell east of a cell centre: GLM's pixel here is 2.005 x 1.62 cells (glmtools' table), so a
    # pixel covers its cell, 0.627 of the next one east and 0.377 west, and 0.31 of the rows either side;
    # on the 2 km grid every partial share is likewise at least 0.12 from the 0.5 that rounding turns on
    px = lambda r, c: scan_latlon(grid, grid.x()[c] + grid.coarsen * grid.x_step / 8, grid.y()[r])  # noqa: E731
    flashes1 = [
        (T1 + 7 * MIN, at(5, 5), [px(5, 5)] * 3 + [px(5, 7)] * 2),  # A: pixels at (5,5) and (5,7), two GLM pixels apart
        (T1 + 12 * MIN, at(10, 10), [px(5, 7), px(10, 10)]),        # B: pixels at (5,7) and (10,10)
    ]
    flashes2 = [
        (T1 + NS_DAY - 10**9, at(7, 7), [px(7, 7)]),                # D: starts before midnight, in day 2's first file
        (T1 + NS_DAY + 3 * MIN, at(3, 3), [px(3, 3)]),              # C: just after midnight
    ]
    cov2 = np.ones(4320)
    cov2[6 * 180 : 7 * 180] = 0  # no GLM data 06:00-07:00 on day 2
    write_glm(p1, flashes1, np.ones(4320), orphans=[at(12, 15)])  # read as the last flash (B) if -1 were an index
    write_glm(p2, flashes2, cov2)
    return shared / "raw"


@pytest.fixture
def built(tmp_path, raw_days):
    """The feature config and its grid, with tmp_path/raw linked to the shared raw days."""
    (tmp_path / "raw").symlink_to(raw_days)
    cfg = load_feature_config(config(tmp_path))
    return cfg, make_grid(cfg)


def cells(mask):
    return {tuple(int(v) for v in rc) for rc in np.argwhere(mask)}


def scan_latlon(grid, x, y):
    """The (lat, lon) GLM would report for scan angles (x, y) on the grid's ellipsoid."""
    from pyproj import CRS, Transformer

    from goes_downloader.crop import geos_crs

    h = float(grid.projection["perspective_point_height"])
    lon, lat = Transformer.from_crs(geos_crs(grid.projection), CRS.from_epsg(4326), always_xy=True).transform(x * h, y * h)
    return float(lat), float(lon)


def plus(points):
    """Cells within 1 cell (the radius-1 disk) of any of `points`."""
    return {(r + dr, c + dc) for r, c in points for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))}


def test_grid_is_blocks_of_the_finest_product(built):
    cfg, grid = built
    assert (grid.reference_product, grid.coarsen, grid.row0, grid.col0, grid.ny, grid.nx) == ("p2", 2, 300, 600, 15, 20)
    assert np.allclose(grid.x()[0], X0 + (1200 + 0.5) * DX) and np.allclose(grid.y()[0], Y0 + (600 + 0.5) * DY)


def test_channels_match_the_raw_pixels(built, tmp_path):
    cfg, grid = built
    assert build_day(cfg, grid, DAY1)["complete"]
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025152.zarr"), mode="r")
    chan = list(F["channel"][:])
    assert chan == ["a", "a_min", "b", "a_day", "a_change", "fed15"]
    t = F["time"][:]
    assert t[0] == T1 and len(t) == 96 and (np.diff(t) == 15 * MIN).all()  # slot starts, UTC
    base = (datetime(2025, 6, 1) - datetime(1970, 1, 1)).days * SLOTS_PER_DAY
    yy, xx = np.meshgrid(CROP_Y0 + np.arange(H), CROP_X0 + np.arange(W), indexing="ij")
    for k in (1, 40):
        s = 3 * k
        v = p2_value(base + s, yy, xx).astype(float)
        ok = ((yy + xx + s) % 11 != 0) & ((yy * xx + s) % 13 != 0)
        phys = np.where(ok, 200 + 0.1 * v, np.nan).reshape(15, 2, 20, 2).transpose(0, 2, 1, 3).reshape(15, 20, 4)
        n = np.isfinite(phys).sum(-1)
        with np.errstate(all="ignore"):
            np.testing.assert_allclose(F["features"][k, 0], np.where(n >= 2, np.nanmean(phys, -1), np.nan), rtol=1e-6)
            np.testing.assert_allclose(F["features"][k, 1], np.where(n >= 2, np.nanmin(phys, -1), np.nan), rtol=1e-6)
        # 4 km cells take the 10 km pixel under their centre (a centre on a pixel edge goes to the lower pixel)
        y10 = np.floor((np.arange(300, 315) * 2 + 1) / 5 - 1e-9).astype(int)
        x10 = np.floor((np.arange(600, 620) * 2 + 1) / 5 - 1e-9).astype(int)
        np.testing.assert_array_equal(F["features"][k, 2], ((y10[:, None] * 100 + x10[None, :] + s) % 5000).astype(float))
    # A missing scan (slot 99 = output 33) is missing in every ABI channel, and so is the lag that reads it
    assert np.isnan(F["features"][33, :3]).all() and np.isnan(F["features"][34, 4]).all()
    assert np.isfinite(F["features"][34, 0]).any()


def test_frames_are_stored_whole_with_the_configured_compression(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY1)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025152.zarr"), mode="r")
    for name in ("features", "labels", "flash_count", "flash_extent"):
        a = F[name]
        (c,) = a.compressors
        assert a.chunks == (1, *a.shape[1:]) and (c.cname, c.clevel, c.shuffle) == ("lz4hc", 5, "shuffle")
    # xarray masks only the _FillValue attribute in Zarr v3: unknown labels, never 0 flashes (NaN marks itself)
    assert F["labels"].fill_value == F["labels"].attrs["_FillValue"] == 255
    assert all("_FillValue" not in F[name].attrs for name in ("features", "flash_count", "flash_extent", "time", "lat"))
    assert F["features"].metadata.dimension_names == ("time", "channel", "y", "x") and F["lat"].metadata.dimension_names == ("y", "x")
    other = load_feature_config(config(tmp_path, chunk_yx=[8, 8], compression={"codec": "zstd", "level": 3, "shuffle": "bit"},
                                       output_root_dir=str(tmp_path / "other")))
    build_day(other, grid, DAY1)
    G = zarr.open_consolidated(str(tmp_path / "other/2025/2025152.zarr"), mode="r")
    (c,) = G["features"].compressors
    assert G["features"].chunks[2:] == (8, 8) and (c.cname, c.shuffle) == ("zstd", "bitshuffle")
    np.testing.assert_array_equal(G["features"][:], F["features"][:])


def test_lag_reads_the_previous_day_and_mask_follows_the_sun(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY2)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025153.zarr"), mode="r")
    a0 = F["features"][0, 0]
    # a_change at 00:00 subtracts the scan of 23:45 the day before
    cfg1 = replace(cfg, start_date_utc=DAY1, end_date_utc=DAY1)
    build_day(cfg1, grid, DAY1)
    F1 = zarr.open_consolidated(str(tmp_path / "features/2025/2025152.zarr"), mode="r")
    np.testing.assert_allclose(F["features"][0, 4], a0 - F1["features"][95, 0], rtol=1e-6, atol=1e-6)
    lat, lon = grid_latlon(grid)
    for k in (0, 60):
        cz = SolarGeometry(lat, lon).frames(int(F["time"][k]) + 150 * 10**9)["cos_zenith"]
        np.testing.assert_array_equal(np.isnan(F["features"][k, 3]), (cz <= 0) | np.isnan(F["features"][k, 0]))


def test_flash_centroids_footprints_and_labels(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY1)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025152.zarr"), mode="r")
    # 00:15 slot, issue 00:20: flashes of [00:05, 00:20) are A and B
    assert cells(F["flash_count"][1] > 0) == {(5, 5), (10, 10)}
    ext = F["flash_extent"][1]
    # Row 5: A covers 0.377 of (5,4), (5,5) to (5,7) wholly -- (5,6) by both its pixels, but once -- and 0.627
    # of (5,8); B adds (5,7) and 0.627 of (5,8) through the pixel they share, and 0.377 of (5,6). Rows 4 and 6
    # get 0.31 of (4,7) from each, 0.62 together: a share no single flash rounds up to. A's three events at
    # (5,5) are one pixel. The orphan event at (12, 15) is in no footprint.
    assert cells(ext > 0) == {(5, 5), (5, 6), (5, 7), (5, 8), (4, 7), (6, 7), (10, 10), (10, 11)}
    assert (ext[5, 5], ext[5, 6], ext[5, 7], ext[5, 8], ext[4, 7]) == (1, 1, 2, 1, 1)
    assert F["flash_extent"][:].sum() == 9
    # fed15 is the same window's density before rounding
    np.testing.assert_array_equal(np.floor(F["features"][1, 5] + 0.5), ext.astype(float))
    assert abs(F["features"][1, 5][5, 8] - 2 * 0.627) < 0.01 and abs(F["features"][1, 5][4, 7] - 0.62) < 0.01
    # Issue 00:05: extent within 1 cell in the next 30 min; centroids in the next 15 min
    assert cells(F["labels"][0, 0] == 1) == plus(cells(ext > 0))
    assert cells(F["labels"][0, 1] == 1) == {(5, 5), (10, 10)}
    # Issue 23:50: the window runs into day 2, which holds D (23:59:59) and C (00:03)
    assert cells(F["labels"][95, 0] == 1) == plus({(7, 7), (7, 8), (3, 3), (3, 4)})
    assert cells(F["labels"][95, 1] == 1) == {(7, 7), (3, 3)}  # [23:50, 00:05): D and C


def test_lightningcast_label_is_the_peak_one_minute_density_on_the_2km_pixels(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY1)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025152.zarr"), mode="r")
    assert list(F["label_fine"][:]) == ["lc60"] and list(F["label"][:]) == ["ext30", "cen15"]
    assert F["labels_fine"].shape == (96, 1, 2 * grid.ny, 2 * grid.nx) and F.attrs["fine_grid"]["coarsen"] == 1
    assert F["labels_fine"].fill_value == F["labels_fine"].attrs["_FillValue"] == 255
    assert F["labels_fine"].metadata.dimension_names == ("time", "label_fine", "y_fine", "x_fine")
    np.testing.assert_allclose(F["x_fine"][:].reshape(-1, 2).mean(axis=1), F["x"][:])  # 2 x 2 in each 4 km cell
    # Issue 00:05, window [00:00, 01:00) from the scan's slot: A (00:07) and B (00:12) in different minutes.
    # On 2 km pixels a GLM pixel covers 4.01 x 3.24: whole columns, 0.754 of the one west, 0.254 east, 0.62 of
    # the rows above and below
    a = {(r, c) for r in (10, 11) for c in range(9, 17)} | {(r, c) for r in (9, 12) for c in range(10, 17)}
    b = ({(r, c) for r in (10, 11) for c in range(13, 17)} | {(r, c) for r in (9, 12) for c in range(14, 17)}
         | {(r, c) for r in (20, 21) for c in range(19, 23)} | {(r, c) for r in (19, 22) for c in range(20, 23)})
    lc = F["labels_fine"][:, 0]
    assert cells(lc[0] == 1) == a | b
    # (10, 17) has 0.254 from each flash: 0.51 over the hour, but each minute's density rounds to 0
    assert lc[0][10, 17] == 0
    assert not (lc[1] == 1).any()  # issue 00:20, window [00:15, 01:15): after both flashes


def test_pixel_cover_spreads_pixels_by_area_once_per_flash():
    from goes_downloader.features.glm import pixel_cover, round_density
    from goes_downloader.features.grid import TargetGrid

    # Cell (r, c) spans x in [c - 0.5, c + 0.5) and y in (-(r + 0.5), -(r - 0.5)]
    g = TargetGrid("p", 1.0, 1.0, 1, 0.0, 1.0, 0.0, -1.0, 0, 0, 4, 5, {})
    rect = lambda c0, c1, r0, r1: (c0 - 0.5, c1 - 0.5, -(r1 - 0.5), -(r0 - 0.5))  # noqa: E731  cells [c0, c1) x [r0, r1)
    pixels = [  # (flash, minute, rectangle)
        (0, 0, rect(1, 2.5, 1, 2)),     # (1,1) wholly, half of (1,2)
        (0, 0, rect(2, 3, 1, 2)),       # the same flash covers (1,2) again: still once
        (1, 0, rect(2.75, 3.5, 1, 2)),  # another flash in the same minute: a quarter of (1,2), half of (1,3)
        (2, 3, rect(0, 1, 0.5, 1.5)),   # minute 3: half of (0,0) and of (1,0)
    ]
    flash, minute = np.array([p[0] for p in pixels]), np.array([p[1] for p in pixels])
    west, east, south, north = (np.array([p[2][k] for p in pixels]) for k in range(4))
    m, cell, cover = pixel_cover(g, flash, minute, west, east, south, north)
    got = {(int(mm), divmod(int(c), 5)): round(float(v), 6) for mm, c, v in zip(m, cell, cover)}
    assert got == {(0, (1, 1)): 1.0, (0, (1, 2)): 1.25, (0, (1, 3)): 0.5, (3, (0, 0)): 0.5, (3, (1, 0)): 0.5}
    assert list(m) == sorted(m)
    np.testing.assert_array_equal(round_density(np.array([0.49, 0.5, 1.25, 1.5])), [0, 1, 1, 2])


def test_pixel_corners_follow_glmtools_table():
    from goes_downloader.features.glm import PixelCorners

    w, e, s, n = PixelCorners().rectangles(np.array([0.0, 0.1]), np.array([0.0, 0.1]))
    assert abs((e - w)[0] - 224e-6) < 1e-6 and abs((n - s)[0] - 224e-6) < 1e-6  # 8 km at nadir
    assert (e - w)[1] < 210e-6 and (n - s)[1] < 210e-6  # smaller in angle towards the edge
    assert abs(w[0] + e[0]) < 1e-9 and abs(s[0] + n[0]) < 1e-9


def test_glm_gaps_make_labels_unknown(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY2)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025153.zarr"), mode="r")
    labels = F["labels"][:]
    assert (labels[23, 0] == 255).all()  # issue 05:50: 06:00-06:20 of its 30 min are missing
    assert (labels[22, 0] == 255).all() and (labels[21, 0] != 255).all()  # 05:35 misses 5 of 30 min; 05:20 none
    assert np.isnan(F["features"][24, 5]).all()  # fed15 during the gap is missing, not zero
    fine = F["labels_fine"][:, 0]
    assert (fine[21] == 255).all() and (fine[20] != 255).all()  # [05:15, 06:15) misses 15 of 60 min; [05:00, 06:00) none


def test_stats_count_each_label_on_its_own_grid(built, tmp_path):
    from goes_downloader.features.build import run_stats

    cfg, grid = built
    build_day(cfg, grid, DAY2)  # with its GLM gap, so some labels are unknown
    run_stats(replace(cfg, start_date_utc=DAY2, end_date_utc=DAY2))
    stats = json.loads((tmp_path / f"features/stats_{DAY2}_{DAY2}.json").read_text())["labels"]
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025153.zarr"), mode="r")
    for name, labels in (("ext30", F["labels"][:, 0]), ("lc60", F["labels_fine"][:, 0])):  # 4 km and 2 km cells
        known = labels != 255
        assert abs(stats[name]["known_fraction"] - known.mean()) < 1e-12
        assert abs(stats[name]["positive_fraction"] - (labels == 1).sum() / known.sum()) < 1e-12


def test_label_resolutions_must_nest_in_the_grid(built, tmp_path):
    from goes_downloader.features.build import fine_label_grid

    cfg, grid = built
    for res in (3, 8):
        bad = load_feature_config(config(tmp_path, labels=[{"name": "l", "end_minutes": 30, "resolution_km": res}]))
        with pytest.raises(ValueError, match="do not nest"):
            fine_label_grid(bad, grid)
    two = load_feature_config(config(tmp_path, labels=[{"name": "l", "end_minutes": 30, "resolution_km": 2},
                                                       {"name": "m", "end_minutes": 30, "resolution_km": 1}]))
    with pytest.raises(ValueError, match="one resolution besides"):
        fine_label_grid(two, grid)
    assert fine_label_grid(load_feature_config(config(tmp_path, labels=[{"name": "l", "end_minutes": 30, "resolution_km": 4}])), grid) is None


def test_neighbour_days_decide_completeness(built, tmp_path):
    cfg, grid = built
    assert build_day(cfg, grid, DAY2)["complete"]  # day 3 is outside the download range: never coming
    later = replace(cfg, raw=replace(cfg.raw, end_date_utc="2025-06-03"))
    r = build_day(later, grid, DAY2)
    assert not r["complete"] and "2025-06-03 missing" in r["status"]


def test_geography_is_built_once_on_the_same_grid(built, tmp_path):
    cfg, grid = built
    lat, lon = grid_latlon(grid)
    la = np.arange(np.nanmin(lat) - 1, np.nanmax(lat) + 1, 0.01)
    lo = np.arange(np.nanmin(lon) - 1, np.nanmax(lon) + 1, 0.01)
    with nc.Dataset(str(tmp_path / "dem.nc"), "w") as ds:
        ds.createDimension("lat", len(la))
        ds.createDimension("lon", len(lo))
        ds.createVariable("lat", "f8", ("lat",))[:] = la
        ds.createVariable("lon", "f8", ("lon",))[:] = lo
        ds.createVariable("z", "f4", ("lat", "lon"))[:] = np.full((len(la), len(lo)), 250.0)
    path = build_geography(cfg, grid)
    G = zarr.open_consolidated(path, mode="r")
    assert G.attrs["fields"] == ["sat_zenith", "height"]
    np.testing.assert_allclose(G["height"][:], 250.0)
    np.testing.assert_allclose(G["sat_zenith"][:], satellite_zenith_deg(lat, lon, PROJECTION))
    check_geography(cfg, grid)
    other = replace(cfg, resolution_km=8.0)
    with pytest.raises(ValueError, match="different grid"):
        check_geography(other, make_grid(other))


def test_geography_regridding_aggregates_fine_and_interpolates_coarse_sources(built):
    cfg, grid = built
    lat, lon = grid_latlon(grid)
    plane = lambda la, lo: 3.0 * la - 2.0 * lo  # noqa: E731  bilinear interpolation reproduces it exactly
    box = (np.nanmin(lat) - 1, np.nanmax(lat) + 1, np.nanmin(lon) - 1, np.nanmax(lon) + 1)
    coarse_la = np.arange(box[0], box[1], 0.5)
    coarse_lo = np.arange(box[2], box[3], 0.5)
    values = plane(*np.meshgrid(coarse_la, coarse_lo, indexing="ij")).astype(np.float32)
    (got,) = regrid_source([GeographySpec("p", variable="v")], coarse_la, coarse_lo, values, grid, lat, lon)
    np.testing.assert_allclose(got, plane(lat.astype(float), lon.astype(float)), atol=1e-3)
    with pytest.raises(ValueError, match="coarser than the grid"):
        regrid_source([GeographySpec("s", variable="v", aggregation="std")], coarse_la, coarse_lo, values, grid, lat, lon)
    fine_la, fine_lo = np.arange(box[0], box[1], 0.005), np.arange(box[2], box[3], 0.005)
    fine = np.where(np.meshgrid(fine_la, fine_lo, indexing="ij")[0] > float(np.nanmean(lat)), 1.0, 0.0).astype(np.float32)
    mean, std = regrid_source([GeographySpec("m", variable="v"), GeographySpec("s", variable="v", aggregation="std")],
                              fine_la, fine_lo, fine, grid, lat, lon)
    assert set(np.unique(mean[:2])) <= {1.0} and set(np.unique(mean[-2:])) <= {0.0}
    assert ((0 < mean) & (mean < 1)).any() and (std[(mean == 0) | (mean == 1)] == 0).all()


def test_rings_fill_by_the_even_odd_rule():
    centres = np.arange(10) + 0.5
    square = np.array([[1, 1], [8, 1], [8, 8], [1, 8], [1, 1]], float)  # [lon, lat]
    hole = np.array([[3, 3], [3, 5], [5, 5], [5, 3], [3, 3]], float)
    diamond = np.array([[9.5, 0.5], [9.9, 1.5], [9.5, 2.5], [9.1, 1.5], [9.5, 0.5]])  # vertices on pixel centres
    got = rasterize_rings([square, hole, diamond], centres, centres)
    want = np.zeros((10, 10), np.uint8)
    want[1:8, 1:8] = 1  # centres 1.5..7.5
    want[3:5, 3:5] = 0
    want[1, 9] = 1  # the diamond holds the centre (9.5, 1.5) only
    np.testing.assert_array_equal(got, want)


def test_borders_fields_rasterize_countries_and_measure_distance(built, tmp_path):
    cfg, grid = built
    lat, lon = grid_latlon(grid)
    d_lat, d_lon = float(np.mean(lat[:-1] - lat[1:])), float(np.mean(lon[:, 1:] - lon[:, :-1]))

    def box(r0, r1, c0, c1):
        """A closed lat/lon ring around cells rows r0..r1, cols c0..c1 (edges half a cell outside their centres)."""
        n, s_ = float(lat[r0, (c0 + c1) // 2]) + d_lat / 2, float(lat[r1, (c0 + c1) // 2]) - d_lat / 2
        w, e = float(lon[(r0 + r1) // 2, c0]) - d_lon / 2, float(lon[(r0 + r1) // 2, c1]) + d_lon / 2
        return [[w, s_], [e, s_], [e, n], [w, n], [w, s_]]

    def around(r, c, half):
        la, lo = float(lat[r, c]), float(lon[r, c])
        return [[lo - half * d_lon, la - half * d_lat], [lo - half * d_lon, la + half * d_lat], [lo + half * d_lon, la + half * d_lat],
                [lo + half * d_lon, la - half * d_lat], [lo - half * d_lon, la - half * d_lat]]

    near = [float(np.nanmin(lat)), float(np.nanmax(lat)), float(np.nanmin(lon)), float(np.nanmax(lon))]
    far_lat = near[1] + 5.0  # a second polygon of AAA whose only vertices are outside "within"
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"ADM0_A3": "AAA"}, "geometry": {"type": "MultiPolygon", "coordinates": [
            [box(2, 8, 3, 11), around(5, 7, 0.9)],  # with a lake over cell (5, 7)
            [[[near[2], far_lat], [near[3], far_lat], [near[3], far_lat + 1], [near[2], far_lat + 1], [near[2], far_lat]]]]}},
        {"type": "Feature", "properties": {"ADM0_A3": "AAA"}, "geometry": {"type": "Polygon", "coordinates": [box(11, 13, 15, 18)]}},
        {"type": "Feature", "properties": {"ADM0_A3": "BBB"}, "geometry": {"type": "Polygon", "coordinates": [box(10, 14, 0, 5)]}},
    ]}
    (tmp_path / "borders.geojson").write_text(json.dumps(geojson))
    src = str(tmp_path / "borders.geojson")
    within = [float(lat[9, 0]), near[1] + 0.01, near[2] - 1, near[3] + 1]  # AAA polygons reaching north of row 9 only
    fields = [
        {"name": "aaa", "borders": "fraction", "countries": ["AAA"], "within": within, "file": src},
        {"name": "aaa_km", "borders": "distance_km", "countries": ["AAA"], "within": within, "file": src},
        {"name": "land", "borders": "fraction", "file": src},
    ]
    cfg = load_feature_config(config(tmp_path, geography={"path": "geo.zarr", "cache_dir": str(tmp_path / "static"), "fields": fields}))
    G = zarr.open_consolidated(build_geography(cfg, make_grid(cfg)), mode="r")
    aaa, km, land = G["aaa"][:], G["aaa_km"][:], G["land"][:]
    assert G.attrs["definitions"][0]["countries"] == ["AAA"] and G.attrs["definitions"][2]["countries"] is None
    inside = np.zeros(aaa.shape, bool)
    inside[3:8, 4:11] = True  # a cell in from the box's edge
    lake = np.zeros(aaa.shape, bool)
    lake[4:7, 6:9] = True
    np.testing.assert_array_equal(aaa[inside & ~lake], 1.0)
    assert aaa[5, 7] == 0 and land[5, 7] == 0  # the lake is a hole
    assert (aaa[11:14, 15:19] == 0).all() and (land[12, 16:18] == 1).all()  # left out by "within", still land
    assert (aaa[11:15, 0:6] == 0).all() and (land[11:14, 1:5] == 1).all()  # BBB
    assert aaa[0, 19] == 0 and land[0, 19] == 0
    # Distance: 0 on AAA, else the great circle to the nearest cell at least half on it
    member = aaa >= 0.5
    phi, lam = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
    for r, c in [(0, 19), (5, 7), (12, 16), (14, 0), (9, 12)]:
        if member[r, c]:
            assert km[r, c] == 0
            continue
        hav = (np.sin((phi[member] - phi[r, c]) / 2) ** 2
               + np.cos(phi[r, c]) * np.cos(phi[member]) * np.sin((lam[member] - lam[r, c]) / 2) ** 2)
        np.testing.assert_allclose(km[r, c], (2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))).min(), rtol=1e-5)
    np.testing.assert_array_equal(km[member], 0.0)


@pytest.mark.parametrize("when, lat, lon, cos_zenith", [
    # astropy (apparent sun, no refraction)
    ("2026-06-21T18:00:00", 40.0, -100.0, 0.94681),
    ("2025-12-21T15:30:00", 30.0, -85.0, 0.47445),
    ("2025-04-10T12:00:00", 45.0, -120.0, -0.25306),
])
def test_sun_position(when, lat, lon, cos_zenith):
    t = int(np.datetime64(when, "ns").astype(np.int64))
    got = SolarGeometry(np.array([lat]), np.array([lon])).frames(t)
    assert abs(float(got["cos_zenith"][0]) - cos_zenith) < 5e-4
    # local solar time from sin/cos: about 06:40 at -100 deg for 18:00 UTC near the June solstice
    hours = (np.degrees(np.arctan2(got["local_hour_sin"], got["local_hour_cos"]))[0] % 360) / 15
    assert abs(hours - ((int(when[11:13]) + int(when[14:16]) / 60 + lon / 15) % 24)) < 0.3


def test_satellite_zenith():
    assert satellite_zenith_deg(np.array([0.0]), np.array([-75.0]), PROJECTION)[0] < 1e-3
    assert abs(satellite_zenith_deg(np.array([39.74]), np.array([-104.99]), PROJECTION)[0] - 55.39) < 0.1


def test_glm_mapping_round_trips_through_the_grid(built):
    cfg, grid = built
    lat, lon = grid_latlon(grid)
    x, y = latlon_to_scan_angle(lat, lon, grid.projection)
    np.testing.assert_array_equal(grid.cells_of(x, y).reshape(grid.ny, grid.nx), np.arange(grid.ny * grid.nx).reshape(grid.ny, grid.nx))


@pytest.mark.parametrize("change, match", [
    ({"channels": [{"name": "a", "product": "p2"}, {"name": "x", "expr": "mask(a)"}]}, "mask takes 2"),
    ({"channels": [{"name": "a", "product": "p9"}]}, "not in the download list"),
    ({"channels": [{"name": "a", "product": "p2"}, {"name": "x", "expr": "y + 1"}, {"name": "y", "expr": "lag(x, 5)"}]}, "cycle"),
    ({"channels": [{"name": "a", "product": "p2", "aggregaton": "min"}]}, "unknown keys"),
    ({"labels": [{"name": "l", "end_minutes": 30, "source": "density"}]}, "source must be"),
    ({"labels": [{"name": "l", "start_minutes": -10, "end_minutes": 30}]}, "scan's own slot"),
    ({"labels": [{"name": "l", "end_minutes": 30, "fed_minutes": 7}]}, "dividing the 30-minute window"),
    ({"labels": [{"name": "l", "end_minutes": 30, "fed_minutes": 1, "source": "centroid"}]}, "needs source"),
    ({"grid": {"resolution_km": 4, "bound": None}}, "unknown keys"),
    ({"geography": {"fields": [{"name": "d", "borders": "distance_km", "file": "b.geojson"}]}}, "distance_km needs countries"),
    ({"compression": {"codec": "snappy"}}, "codec must be one of"),
    ({"compression": {"level": 12}}, "level must be"),
    ({"chunk_yx": [256]}, "chunk_yx must be"),
    ({"geography": {"fields": [{"name": "d", "borders": "area", "file": "b.geojson"}]}}, "borders must be one of"),
    ({"geography": {"fields": [{"name": "d", "borders": "fraction", "file": "b.geojson", "within": [50, 20, 0, 1]}]}}, "within must be"),
])
def test_bad_feature_settings_are_rejected(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        load_feature_config(config(tmp_path, **change))


def test_repo_features_config_loads():
    cfg = load_feature_config(os.path.join(REPO, "config.json"))
    assert [c.name for c in cfg.channels if c.output][:2] == ["vis_064", "nir_16"]


def test_config_without_features_section_is_rejected(tmp_path):
    raw = json.load(open(config(tmp_path)))
    del raw["features"]
    (tmp_path / "plain.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no \"features\""):
        load_feature_config(str(tmp_path / "plain.json"))


def test_day_without_raw_store_is_reported(built):
    cfg, grid = built
    assert build_day(cfg, grid, "2025-06-05") == {"day": "2025-06-05", "status": "no raw store", "complete": False}


def test_flash_just_before_midnight_counts_without_labels(built, tmp_path):
    """With no labels the GLM range ends at midnight; D (23:59:59) is in day 2's first file and must count."""
    cfg, grid = built
    nolabels = load_feature_config(config(tmp_path, labels=[], time_step_minutes=5, output_root_dir=str(tmp_path / "nolabels")))
    build_day(nolabels, grid, DAY1)
    F = zarr.open_consolidated(str(tmp_path / "nolabels/2025/2025152.zarr"), mode="r")
    assert cells(F["flash_count"][287] > 0) == {(7, 7)} and cells(F["flash_extent"][287] > 0) == {(7, 7), (7, 8)}


def test_complete_day_is_rebuilt_when_its_missing_neighbour_arrives_or_it_cannot_be_read(built, tmp_path):
    from goes_downloader.features.build import features_day_path, raw_neighbour_arrived

    cfg, grid = built
    assert build_day(cfg, grid, DAY2)["complete"]  # day 3 is outside the download range
    path = features_day_path(cfg, datetime.strptime(DAY2, "%Y-%m-%d"))
    assert not raw_neighbour_arrived(cfg, path)
    raw = tmp_path / "raw"
    shared = raw.resolve()
    raw.unlink()
    shutil.copytree(shared, raw)  # a private copy: the shared raw days must stay as they are
    p3 = write_raw_day(str(raw), "2025-06-03")  # the range was extended and day 3 downloaded
    write_glm(p3, [], np.ones(4320))
    assert raw_neighbour_arrived(cfg, path)
    shutil.rmtree(p3)
    assert not raw_neighbour_arrived(cfg, path)
    meta_path = os.path.join(path, "zarr.json")
    with open(meta_path) as f:
        meta = json.load(f)
    del meta["consolidated_metadata"]  # e.g. a store left behind by an older version
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    assert raw_neighbour_arrived(cfg, path)


def test_rebuilding_a_day_replaces_the_store_without_leaving_one_behind(built):
    from goes_downloader.features.build import features_day_path

    cfg, grid = built
    path = features_day_path(cfg, datetime.strptime(DAY2, "%Y-%m-%d"))
    for _ in range(2):  # the second build publishes over the first one's store
        assert build_day(cfg, grid, DAY2)["complete"]
    assert zarr.open_consolidated(path, mode="r").attrs["complete"]
    assert not os.path.exists(path + ".old") and not os.path.exists(path + ".tmp")


def test_only_channels_read_at_earlier_slots_are_computed_every_slot(built):
    from goes_downloader.features.expressions import make_plan

    cfg, _ = built
    plan = make_plan(cfg.channels, [l.end_slots for l in cfg.labels])
    assert plan.every_slot == {"a", "fed"}  # lag(a, 15) and sum_past(fed, 15); the rest only at output slots


def test_fraction_matches_values_float32_cannot_hold_exactly():
    from goes_downloader.features.grid import matches_any

    phys = np.array([3, 1, 2], dtype=np.float32) * np.float32(0.1)  # 0.3 is 0.30000001 in float32
    assert not np.isin(phys, [0.3]).any()
    assert matches_any(phys, [0.3, 0.1]).tolist() == [True, True, False]


def test_ctrl_c_during_a_feature_build_exits_cleanly(monkeypatch, tmp_path, capsys):
    from goes_downloader.features import build

    def interrupted(cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(build, "run_build", interrupted)
    with pytest.raises(SystemExit) as stop:
        build.features_main(["--config", config(tmp_path)])
    assert stop.value.code == 130 and "Interrupted" in capsys.readouterr().err
