"""MRMS: GRIB2 decoding of synthetic files, the crop, a day against a fake bucket, and the config."""

import asyncio
import gzip
import io
import json
import struct
from datetime import datetime, timezone

import numpy as np
import pytest
import zarr
from PIL import Image

from goes_downloader import mrms, remote
from goes_downloader.config import LatLonBounds

DAY = datetime(2025, 6, 1, tzinfo=timezone.utc)
NI, NJ = 40, 30  # a 0.01 deg grid from 51.0 N, 100.0 W (260.0 E), rows north to south


def signed(v: int, n: int) -> bytes:
    """GRIB2's sign-and-magnitude integers."""
    return ((1 << (8 * n - 1)) | -v if v < 0 else v).to_bytes(n, "big")


def grib2(values: np.ndarray, reference=-9990.0, e=0, d=1, lat1=51.0, lon1=260.0, packing=41) -> bytes:
    """A gzipped GRIB2 file laid out as MRMS's: lat/lon grid (3.0), PNG packing (5.41), no bitmap."""
    nj, ni = values.shape
    png = io.BytesIO()
    Image.fromarray(values.astype("<u2")).save(png, format="PNG")  # a 16-bit greyscale PNG, as MRMS's
    s3 = bytearray(72)
    s3[6:10], s3[12:14] = (ni * nj).to_bytes(4, "big"), (0).to_bytes(2, "big")
    s3[30:34], s3[34:38] = ni.to_bytes(4, "big"), nj.to_bytes(4, "big")
    s3[46:50], s3[50:54] = signed(round(lat1 * 1e6), 4), round(lon1 * 1e6).to_bytes(4, "big")
    s3[63:67], s3[67:71] = (10000).to_bytes(4, "big"), (10000).to_bytes(4, "big")
    s5 = bytearray(21)
    s5[5:9], s5[9:11], s5[11:15] = (ni * nj).to_bytes(4, "big"), packing.to_bytes(2, "big"), struct.pack(">f", reference)
    s5[15:17], s5[17:19], s5[19] = signed(e, 2), signed(d, 2), 16
    s6 = bytearray(6)
    s6[5] = 255
    sections = [bytearray(21), s3, bytearray(34), s5, s6, bytearray(5) + png.getvalue()]
    for number, s in zip((1, 3, 4, 5, 6, 7), sections):
        s[0:4], s[4] = len(s).to_bytes(4, "big"), number
    body = b"".join(sections) + b"7777"
    return gzip.compress(b"GRIB\x00\x00\xd1\x02" + (16 + len(body)).to_bytes(8, "big") + body)


def field(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 12000, (NJ, NI)).astype(np.uint16)


def test_grib2_png_fields_decode_to_the_packed_values():
    values = field(0)
    dec = mrms.decode_mrms(grib2(values, reference=-9990.0, e=-1, d=1))
    np.testing.assert_array_equal(dec["value"], values)
    assert dec["scale_factor"] == 0.05 and dec["add_offset"] == -999.0  # 2**-1 / 10, -9990 / 10
    np.testing.assert_allclose(dec["lat"], 51.0 - 0.01 * np.arange(NJ))
    np.testing.assert_allclose(dec["lon"], -100.0 + 0.01 * np.arange(NI))
    assert mrms.decode_mrms(grib2(values, lat1=-10.0))["lat"][0] == -10.0  # a negative latitude: sign bit


def test_other_packings_are_refused():
    with pytest.raises(ValueError, match="PNG packing"):
        mrms.decode_mrms(grib2(field(0), packing=0))
    with pytest.raises(ValueError, match="not a GRIB2"):
        mrms.decode_mrms(gzip.compress(b"<Error>NoSuchKey</Error>"))


def test_crop_keeps_pixels_whose_cell_overlaps_the_bounds():
    lat, lon = 51.0 - 0.01 * np.arange(NJ), -100.0 + 0.01 * np.arange(NI)
    ys, xs = mrms.crop_window(lat, lon, LatLonBounds(50.80, 50.90, -99.90, -99.80))
    assert (ys, xs) == (slice(10, 21), slice(10, 21))
    assert mrms.crop_window(lat, lon, None) == (slice(0, NJ), slice(0, NI))
    with pytest.raises(ValueError, match="overlap"):
        mrms.crop_window(lat, lon, LatLonBounds(10, 20, -99.9, -99.8))


def write_config(tmp_path, **mrms_section):
    cfg = {
        "bucket": "s3://test", "start_date_utc": "2025-06-01", "end_date_utc": "2025-06-01",
        "raw_output_root_dir": str(tmp_path / "raw"), "download": ["p"],
        "abi_products": [{"key": "p", "product_prefix": "A", "filename_must_contain": "A", "data_variable": "V", "dqf_variable": "Q"}],
        "glm_product": {"key": "glm", "product_prefix": "G", "filename_must_contain": "G"},
        "bounds": {"lat_min": 50.80, "lat_max": 50.90, "lon_min": -99.90, "lon_max": -99.80},
        "mrms": {"output_root_dir": str(tmp_path / "mrms"), "threads": 2,
                 "products": [{"key": "refl", "prefix": "CONUS/Refl", "description": "test"}], **mrms_section},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def fake_bucket(monkeypatch, files, failing=()):
    """The bucket as a name -> payload dict; downloads of `failing` names fail; returns the names downloaded."""
    fetched = []

    async def list_object_names(cfg, session, prefix):
        return {name: len(data) for name, data in files.items() if name.startswith(prefix)}

    async def http_get(cfg, session, url, params=None, *, as_json):
        name = url.split(".amazonaws.com/", 1)[1]
        fetched.append(name)
        if name in failing:
            raise RuntimeError(f"GET {url} failed: HTTP 503")
        return files[name]

    monkeypatch.setattr(remote, "list_object_names", list_object_names)
    monkeypatch.setattr(remote, "http_get", http_get)
    return fetched


def test_a_day_stores_every_file_cropped_and_is_rebuilt_until_complete(tmp_path, monkeypatch):
    cfg = mrms.load_mrms_config(write_config(tmp_path))
    name = lambda hhmmss, day="20250601": f"CONUS/Refl/{day}/MRMS_Refl_{day}-{hhmmss}.grib2.gz"  # noqa: E731
    files = {name("000037"): grib2(field(1)), name("000238"): grib2(field(2)), name("120039"): grib2(field(3)),
             name("000037", "20250602"): grib2(field(4))}  # the next day's folder: not listed for this day

    fetched = fake_bucket(monkeypatch, files, failing={name("000238")})
    complete, _ = asyncio.run(mrms.process_mrms_day(cfg, DAY, None))
    root = zarr.open_consolidated(str(tmp_path / "mrms/2025/2025152.zarr"), mode="r")
    g = root["refl"]
    assert not complete and root.attrs["n_file_errors"] == 1 and not root.attrs["complete"]
    assert list(g["present"][:]) == [1, 0, 1] and g.attrs["n_present"] == 2 and g.attrs["n_files"] == 3
    assert list(g["time"][:].astype("datetime64[ns]").astype("datetime64[s]").astype(str)) ==["2025-06-01T00:00:37", "2025-06-01T00:02:38", "2025-06-01T12:00:39"]
    np.testing.assert_array_equal(g["value"][0], field(1)[10:21, 10:21])
    np.testing.assert_array_equal(g["value"][2], field(3)[10:21, 10:21])
    assert (g["value"][1] == mrms.FILL_VALUE).all() and np.isnan(g["scale_factor"][1])
    assert (g.attrs["crop_y0"], g.attrs["crop_x0"], g.attrs["source_shape"]) == (10, 10, [NJ, NI])
    np.testing.assert_allclose(g["lat"][:], 50.9 - 0.01 * np.arange(11))
    assert len(fetched) == 3

    # The next run rebuilds the incomplete day; once complete, a run downloads nothing
    fetched = fake_bucket(monkeypatch, files)
    assert asyncio.run(mrms.process_mrms_day(cfg, DAY, None))[0]
    assert list(zarr.open_consolidated(str(tmp_path / "mrms/2025/2025152.zarr"), mode="r")["refl/present"][:]) == [1, 1, 1]
    fetched = fake_bucket(monkeypatch, files)
    assert asyncio.run(mrms.process_mrms_day(cfg, DAY, None)) == (True, 0) and fetched == []


def test_a_file_on_another_grid_is_an_error(tmp_path, monkeypatch):
    cfg = mrms.load_mrms_config(write_config(tmp_path))
    files = {"CONUS/Refl/20250601/MRMS_Refl_20250601-000037.grib2.gz": grib2(field(1)),
             "CONUS/Refl/20250601/MRMS_Refl_20250601-000238.grib2.gz": grib2(field(2), lat1=51.01)}
    fake_bucket(monkeypatch, files)
    complete, _ = asyncio.run(mrms.process_mrms_day(cfg, DAY, None))
    root = zarr.open_consolidated(str(tmp_path / "mrms/2025/2025152.zarr"), mode="r")
    assert not complete and root.attrs["n_file_errors"] == 1 and root["refl"].attrs["n_present"] == 1


def test_mrms_settings_are_checked(tmp_path):
    cfg = mrms.load_mrms_config(write_config(tmp_path))
    assert cfg.raw.bucket == "s3://noaa-mrms-pds" and cfg.threads == 2 and cfg.products[0].prefix == "CONUS/Refl"
    assert cfg.raw.bounds == LatLonBounds(50.80, 50.90, -99.90, -99.80)
    for change, match in (({"thread": 2}, "unknown keys"), ({"products": []}, "unique keys"),
                          ({"products": [{"key": "a"}]}, "prefix"), ({"threads": 0}, "positive integer")):
        with pytest.raises(ValueError, match=match):
            mrms.load_mrms_config(write_config(tmp_path, **change))


def test_repo_mrms_config_loads():
    from conftest import REPO

    cfg = mrms.load_mrms_config(f"{REPO}/config.json")
    assert cfg.raw.bucket == "s3://noaa-mrms-pds" and [p.key for p in cfg.products] == ["refl_m10c"]
