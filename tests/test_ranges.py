"""Byte-range downloads: the fetched-range bookkeeping, and decoding from partial files."""

import random
import threading

import numpy as np
import pytest

from conftest import same
from goes_downloader import ranges
from goes_downloader.config import LatLonBounds
from goes_downloader.crop import crop_slice, scan_angle_rect
from goes_downloader.decode import decode_abi_file, decode_glm_file

CONUS = LatLonBounds(lat_min=24.0, lat_max=50.0, lon_min=-125.0, lon_max=-66.0)
ABI_VARIABLES = [("CMI_C02", "DQF_C02", None), ("CMI_C13", "DQF_C13", 0.25)]


def test_range_bookkeeping_random(monkeypatch):
    """Against a one-flag-per-byte model: requested bytes are fetched once and land in the right place."""
    pattern = (np.arange(6000) * 7 + 3).astype(np.uint8).tobytes()
    monkeypatch.setattr(ranges, "_get_range", lambda path, start, end, timeout, retries: pattern[start:end])
    rng = random.Random(1)
    for _ in range(500):
        size = rng.randint(1, 5000)
        rf = ranges._RangeFile("/x", size, 1, 1)
        have = np.zeros(size, bool)
        for _ in range(rng.randint(1, 8)):
            requested = [(rng.randint(-50, size + 50), rng.randint(-50, size + 50)) for _ in range(rng.randint(1, 4))]
            requested = [(a, b) for a, b in requested if a < b]
            rf.fetch(requested, merge_gap=rng.choice([1, 16, 300]))
            for a, b in requested:
                have[min(max(a, 0), size):min(max(b, 0), size)] = True
            covered = np.zeros(size, bool)
            for s, e in zip(rf.starts, rf.ends):
                covered[s:e] = True
            assert covered[have].all()
            assert all(rf.ends[i] < rf.starts[i + 1] for i in range(len(rf.starts) - 1))
            buf = np.frombuffer(bytes(rf.buf), np.uint8)
            assert (buf[covered] == np.frombuffer(pattern[:size], np.uint8)[covered]).all()
            assert (buf[~covered] == 0).all()
            a, b = sorted(rng.sample(range(-10, size + 10), 2))
            missing = np.zeros(size, bool)
            for s, e in rf._missing(a, b):
                missing[s:e] = True
            wanted = np.zeros(size, bool)
            wanted[min(max(a, 0), size):min(max(b, 0), size)] = True
            assert (missing == (wanted & ~covered)).all()
        rf.release()


@pytest.mark.parametrize("bounds", [CONUS, None])
def test_ranged_abi_matches_whole_file(abi_bytes, remote, bounds):
    remote.files["/b/abi.nc"] = abi_bytes
    whole = decode_abi_file(abi_bytes, ABI_VARIABLES)
    partial, downloaded = ranges.ranged_decode_abi_file("/b/abi.nc", len(abi_bytes), ABI_VARIABLES, bounds, 1, 1)

    if bounds is None:
        assert partial[0]["window"] is None
        ys = xs = slice(None)
    else:
        # The window is exactly the crop the main process computes from the same file
        x0, x1, y0, y1 = scan_angle_rect(bounds, whole[0]["projection"])
        ys, xs = crop_slice(whole[0]["y"], y0, y1, "y"), crop_slice(whole[0]["x"], x0, x1, "x")
        assert partial[0]["window"] == (ys.start, ys.stop, xs.start, xs.stop)
        assert 0 < ys.stop - ys.start < len(whole[0]["y"]) and 0 < xs.stop - xs.start < len(whole[0]["x"])
    for w, p in zip(whole, partial):
        for key in ("value", "dqf"):
            assert same(w[key][ys, xs], p[key][ys, xs]), key
        for key in ("x", "y"):
            assert same(w[key], p[key])
        for key in ("scale_factor", "add_offset", "quantization_step", "fill_value", "dqf_fill_value", "units", "projection"):
            assert w[key] == p[key], key
    # Unneeded chunks are not fetched: band 8 (never read), and with bounds the chunks outside the window.
    # Metadata comes in aligned blocks that can spill into a neighbouring chunk, so allow a little.
    requested = [(start, end) for _, start, end in remote.requests]
    assert downloaded == sum(end - start for start, end in requested) < len(abi_bytes)
    fetched = np.zeros(len(abi_bytes), bool)
    for start, end in requested:
        fetched[start:end] = True
    window = partial[0]["window"]
    unneeded = fetched_unneeded = 0
    for name, (cy, cx), (chy, chx), offset, size in chunk_map(abi_bytes):
        outside = window is not None and not (cy < window[1] and cy + chy > window[0] and cx < window[3] and cx + chx > window[2])
        if name.endswith("C08") or outside:
            unneeded += size
            fetched_unneeded += int(fetched[offset:offset + size].sum())
    assert unneeded > 0 and fetched_unneeded < 0.25 * unneeded


def chunk_map(data):
    """(variable, first element, chunk shape, byte offset, byte size) of every stored 2-D chunk."""
    import io
    import h5py

    out = []
    with h5py.File(io.BytesIO(data), "r") as h:
        for name, d in h.items():
            if isinstance(d, h5py.Dataset) and d.ndim == 2 and d.chunks:
                for i in range(d.id.get_num_chunks()):
                    c = d.id.get_chunk_info(i)
                    out.append((name, c.chunk_offset, d.chunks, c.byte_offset, c.size))
    return out


def test_ranged_glm_matches_whole_file(glm_file, remote):
    data, _ = glm_file
    remote.files["/b/glm.nc"] = data
    whole = decode_glm_file(data)
    partial, downloaded = ranges.ranged_decode_glm_file("/b/glm.nc", len(data), 1, 1)
    assert whole.keys() == partial.keys()
    for key in whole:
        assert same(whole[key], partial[key]), key
    assert downloaded < len(data)


class FakeConnection:
    """http.client.HTTPSConnection stand-in: serves byte ranges of one pattern and records every request."""
    opened = []

    def __init__(self, host, timeout):
        self.host, self.requests = host, []
        FakeConnection.opened.append(self)

    def request(self, method, path, headers):
        self.requests.append((method, path, headers["Range"]))
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.body = bytes(range(256))[start:end + 1]

    def getresponse(self):
        body = self.body

        class Response:
            status = 206

            def read(self):
                return body
        return Response()

    def close(self):
        pass


def test_get_range_requests_the_object_from_its_host(monkeypatch):
    """The worker splits the object URL into host and path, and keeps one connection per host."""
    monkeypatch.setattr(ranges.http.client, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(ranges, "_range_connections", threading.local())
    FakeConnection.opened = []
    s3 = "https://noaa-goes19.s3.amazonaws.com/GLM-L2-LCFA/2025/300/18/f.nc"
    gcs = "https://storage.googleapis.com/gcp-public-data-goes-19/GLM-L2-LCFA/2025/300/18/f.nc"
    assert ranges._get_range(s3, 10, 20, 1, 1) == bytes(range(10, 20))
    assert ranges._get_range(s3, 0, 4, 1, 1) == bytes(range(4))
    assert ranges._get_range(gcs, 5, 6, 1, 1) == bytes([5])
    assert [(c.host, c.requests) for c in FakeConnection.opened] == [
        ("noaa-goes19.s3.amazonaws.com", [("GET", "/GLM-L2-LCFA/2025/300/18/f.nc", "bytes=10-19"),
                                          ("GET", "/GLM-L2-LCFA/2025/300/18/f.nc", "bytes=0-3")]),
        ("storage.googleapis.com", [("GET", "/gcp-public-data-goes-19/GLM-L2-LCFA/2025/300/18/f.nc", "bytes=5-5")]),
    ]
