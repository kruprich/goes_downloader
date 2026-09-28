"""
The feature step on two small synthetic raw days: regridding, derived channels across midnight, GLM
centroids and footprints, labels, completeness, geography and the sun/satellite geometry.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone

import netCDF4 as nc
import numpy as np
import pytest
import zarr

from goes_downloader.common import SLOTS_PER_DAY
from goes_downloader.crop import latlon_to_scan_angle
from goes_downloader.features.build import build_day, check_geography
from goes_downloader.features.geography import EARTH_RADIUS_KM, build_geography, rasterize_rings, regrid_source
from goes_downloader.features.grid import SolarGeometry, grid_latlon, make_grid, satellite_zenith_deg
from goes_downloader.features.settings import GeographySpec, load_feature_config

PROJECTION = {  # GOES-East fixed grid, as in the real files
    "grid_mapping_name": "geostationary", "perspective_point_height": 35786023.0, "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414, "inverse_flattening": 298.2572221, "latitude_of_projection_origin": 0.0,
    "longitude_of_projection_origin": -75.0, "sweep_angle_axis": "x",
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    root = zarr.open_group(zarr.DirectoryStore(path), mode="w")
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
        g.array("value", values, chunks=(1, h, w))
        g.array("dqf", dqf, chunks=(1, h, w))
        g.array("scale_factor", np.full(n, sf))
        g.array("add_offset", np.full(n, ao))
        g.array("present", present)
        # A 10 km pixel is 5x5 2 km pixels: its centre is the third 2 km centre
        g.array("x_scan_angle_rad", X0 + (factor - 1) / 2 * DX + (CROP_X0 // factor + np.arange(w)) * factor * DX)
        g.array("y_scan_angle_rad", Y0 + (factor - 1) / 2 * DY + (CROP_Y0 // factor + np.arange(h)) * factor * DY)
        g.attrs.update(fill_value=65535, dqf_fill_value=255, crop_y0=CROP_Y0 // factor, crop_x0=CROP_X0 // factor, has_frames=True,
                       goes_imager_projection=PROJECTION, uncropped_shape=[1500 // factor, 2500 // factor])
    root.attrs.update(day_utc=day, complete=True)
    zarr.consolidate_metadata(zarr.DirectoryStore(path))
    return path


def write_glm(path: str, flashes, covered: np.ndarray, orphans=()):
    """
    flashes: (time_ns, centroid (lat, lon), [event (lat, lon), ...]), in time order; orphans: events
    (lat, lon) whose flash isn't in their file, stored with flash_index -1 after the first flash's events.
    """
    root = zarr.open_group(path, mode="a")
    g = root.require_group("glm")
    f = g.require_group("flashes")
    f.array("time_first_ns", np.array([t for t, _, _ in flashes], np.int64))
    f.array("lat", np.array([c[0] for _, c, _ in flashes], np.float32))
    f.array("lon", np.array([c[1] for _, c, _ in flashes], np.float32))
    f.array("energy_j", np.full(len(flashes), 1e-13, np.float32))
    f.array("area_m2", np.full(len(flashes), 1e8, np.float32))
    f.array("quality_flag", np.zeros(len(flashes), np.uint8))
    e = g.require_group("events")
    ev = [(la, lo, i) for i, (_, _, events) in enumerate(flashes) for la, lo in events]
    n0 = len(flashes[0][2]) if flashes else 0
    ev[n0:n0] = [(la, lo, -1) for la, lo in orphans]
    e.array("lat", np.array([x[0] for x in ev], np.float32))
    e.array("lon", np.array([x[1] for x in ev], np.float32))
    e.array("flash_index", np.array([x[2] for x in ev], np.int32))
    g.require_group("coverage").array("covered", covered.astype(np.uint8))
    zarr.consolidate_metadata(zarr.DirectoryStore(path))


def config(tmp_path, end=DAY2, **features):
    raw = {
        "gcs_bucket": "test", "start_date_utc": DAY1, "end_date_utc": end, "raw_output_root_dir": str(tmp_path / "raw"),
        "download": ["p2", "p10", "glm"],
        "abi_products": [
            {"key": "p2", "gcs_product_prefix": "A", "filename_must_contain": "A-M6", "data_variable": "V", "dqf_variable": "DQF"},
            {"key": "p10", "gcs_product_prefix": "B", "filename_must_contain": "B-M6", "data_variable": "V", "dqf_variable": "DQF"},
        ],
        "glm_product": {"key": "glm", "gcs_product_prefix": "G", "filename_must_contain": "G"},
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


@pytest.fixture
def built(tmp_path):
    """Two raw days with GLM, the feature config and its grid; flashes are placed on known cells."""
    write_raw_day(str(tmp_path / "raw"), DAY1, missing_slot=99)  # the 08:15 output slot
    p2 = write_raw_day(str(tmp_path / "raw"), DAY2)
    p1 = os.path.join(str(tmp_path / "raw"), "2025", "2025152.zarr")
    cfg = load_feature_config(config(tmp_path))
    grid = make_grid(cfg)
    lat, lon = grid_latlon(grid)
    at = lambda r, c: (float(lat[r, c]), float(lon[r, c]))  # noqa: E731
    flashes1 = [
        (T1 + 7 * MIN, at(5, 5), [at(5, 5)] * 3 + [at(5, 6)] * 2),  # A: footprint (5,5) and (5,6)
        (T1 + 12 * MIN, at(10, 10), [at(5, 6), at(10, 10)]),        # B: footprint (5,6) and (10,10)
    ]
    flashes2 = [
        (T1 + NS_DAY - 10**9, at(7, 7), [at(7, 7)]),                # D: starts before midnight, in day 2's first file
        (T1 + NS_DAY + 3 * MIN, at(3, 3), [at(3, 3)]),              # C: just after midnight
    ]
    cov2 = np.ones(4320)
    cov2[6 * 180 : 7 * 180] = 0  # no GLM data 06:00-07:00 on day 2
    write_glm(p1, flashes1, np.ones(4320), orphans=[at(12, 15)])  # read as the last flash (B) if -1 were an index
    write_glm(p2, flashes2, cov2)
    return cfg, grid


def cells(mask):
    return {tuple(int(v) for v in rc) for rc in np.argwhere(mask)}


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
        assert a.chunks == (1, *a.shape[1:]) and (a.compressor.cname, a.compressor.clevel, a.compressor.shuffle) == ("lz4hc", 5, 1)
    other = load_feature_config(config(tmp_path, chunk_yx=[8, 8], compression={"codec": "zstd", "level": 3, "shuffle": "bit"},
                                       output_root_dir=str(tmp_path / "other")))
    build_day(other, grid, DAY1)
    G = zarr.open_consolidated(str(tmp_path / "other/2025/2025152.zarr"), mode="r")
    assert G["features"].chunks[2:] == (8, 8) and (G["features"].compressor.cname, G["features"].compressor.shuffle) == ("zstd", 2)
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
    assert cells(ext > 0) == {(5, 5), (5, 6), (10, 10)} and ext[5, 6] == 2 and ext[5, 5] == 1  # A's 3 events count once
    assert F["flash_extent"][:].sum() == 4  # the orphan event at (12, 15) is in no footprint
    np.testing.assert_array_equal(F["features"][1, 5], ext.astype(float))  # fed15 = the same window
    # Issue 00:05: extent within 1 cell in the next 30 min; centroids in the next 15 min
    disk = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}
    want = {(r + dr, c + dc) for r, c in {(5, 5), (5, 6), (10, 10)} for dr, dc in disk}
    assert cells(F["labels"][0, 0] == 1) == want
    assert cells(F["labels"][0, 1] == 1) == {(5, 5), (10, 10)}
    # Issue 23:50: the window runs into day 2, which holds D (23:59:59) and C (00:03)
    assert cells(F["labels"][95, 0] == 1) == {(r + dr, c + dc) for r, c in {(7, 7), (3, 3)} for dr, dc in disk}
    assert cells(F["labels"][95, 1] == 1) == {(7, 7), (3, 3)}  # [23:50, 00:05): D and C


def test_glm_gaps_make_labels_unknown(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY2)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025153.zarr"), mode="r")
    labels = F["labels"][:]
    assert (labels[23, 0] == 255).all()  # issue 05:50: 06:00-06:20 of its 30 min are missing
    assert (labels[22, 0] == 255).all() and (labels[21, 0] != 255).all()  # 05:35 misses 5 of 30 min; 05:20 none
    assert np.isnan(F["features"][24, 5]).all()  # fed15 during the gap is missing, not zero


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


@pytest.mark.parametrize("script", ["build_features.py", "build_geography.py"])
def test_feature_entry_scripts_import_nothing_outside_their_main_guard(script):
    code = (f"import runpy, sys; runpy.run_path({script!r}, run_name='__mp_main__'); "
            "print(sorted(m for m in ('zarr', 'goes_downloader') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_config_without_features_section_is_rejected(tmp_path):
    raw = json.load(open(config(tmp_path)))
    del raw["features"]
    (tmp_path / "plain.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no \"features\""):
        load_feature_config(str(tmp_path / "plain.json"))


def test_day_without_raw_store_is_reported(built):
    cfg, grid = built
    assert build_day(cfg, grid, "2025-06-05") == {"day": "2025-06-05", "status": "no raw store", "complete": False}


def test_time_zone_of_slots(built, tmp_path):
    cfg, grid = built
    build_day(cfg, grid, DAY1)
    F = zarr.open_consolidated(str(tmp_path / "features/2025/2025152.zarr"), mode="r")
    t = F["time"][:]
    assert t[0] == T1 and np.all(np.diff(t) == 15 * MIN) and len(t) == 96
    assert datetime.fromtimestamp(t[0] / 1e9, timezone.utc).hour == 0
