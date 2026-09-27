"""
Byte-range downloads (download_method "byte_ranges"; runs in worker processes).

A worker reads the file's HDF5 metadata through h5py, which fetches it piece by piece, then
fetches only the chunks of the variables that are decoded (for ABI, only chunks overlapping the
crop). Everything lands in a file-sized buffer whose other bytes stay zero, and the same NetCDF
decoders as for whole files read that buffer.
"""

from __future__ import annotations

import bisect
import http.client
import io
import mmap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import GCS_HOST, RETRYABLE_HTTP_STATUS, backoff_seconds
from .config import LatLonBounds
from .crop import crop_window
from .decode import GLM_READ_VARIABLES, abi_grid, decode_abi_file, decode_glm_file

# The first request fetches both ends of the file, where netCDF-C writes most of the metadata
RANGE_HEAD_BYTES = 80 * 1024
RANGE_TAIL_BYTES = 64 * 1024
# Metadata elsewhere is fetched in aligned blocks of this size when h5py asks for it
RANGE_BLOCK_BYTES = 16 * 1024
# Needed ranges closer together than this are fetched in one request
RANGE_MERGE_GAP_BYTES = 16 * 1024
# Requests one worker makes at once for one file's chunks
RANGE_THREADS_PER_PROCESS = 4

_range_connections = threading.local()  # one keep-alive HTTPS connection per thread
_range_threads: Optional[ThreadPoolExecutor] = None


def _get_range(path: str, start: int, end: int, timeout: int, retries: int) -> bytes:
    """Bytes [start, end) of a bucket object, over this thread's keep-alive connection; retries transient errors."""
    last_error = "no attempts"
    for attempt in range(1, retries + 1):
        conn = getattr(_range_connections, "conn", None)
        if conn is None:
            conn = _range_connections.conn = http.client.HTTPSConnection(GCS_HOST, timeout=timeout)
        try:
            conn.request("GET", path, headers={"Range": f"bytes={start}-{end - 1}"})
            r = conn.getresponse()
            body = r.read()
            if r.status == 206 and len(body) == end - start:
                return body
            last_error = f"HTTP {r.status} with {len(body)} bytes"
            if r.status != 206 and r.status not in RETRYABLE_HTTP_STATUS:
                break
        except (OSError, http.client.HTTPException) as e:
            last_error = repr(e)
            conn.close()
            _range_connections.conn = None
        if attempt < retries:
            time.sleep(backoff_seconds(attempt))
    raise RuntimeError(f"GET {path} bytes {start}-{end - 1} failed: {last_error}")


def _range_executor() -> ThreadPoolExecutor:
    global _range_threads
    if _range_threads is None:
        _range_threads = ThreadPoolExecutor(RANGE_THREADS_PER_PROCESS)
    return _range_threads


class _RangeFile(io.RawIOBase):
    """
    A remote file for h5py and the NetCDF decoders: a file-sized buffer holding the byte ranges fetched
    so far, with everything else zero. Reads fetch missing bytes in aligned blocks. The buffer is an
    anonymous memory map, so only fetched pages use memory; call release() when done.
    """

    def __init__(self, path: str, size: int, timeout: int, retries: int):
        if size <= 0:
            raise ValueError(f"{path}: size unknown")
        self.path, self.size, self.timeout, self.retries = path, size, timeout, retries
        self.buf = mmap.mmap(-1, size)
        # Fetched ranges [starts[i], ends[i]), sorted, neither overlapping nor touching
        self.starts: List[int] = []
        self.ends: List[int] = []
        self.pos = 0
        self.downloaded = 0

    def release(self) -> None:
        self.buf.close()

    def _missing(self, start: int, end: int) -> List[Tuple[int, int]]:
        """The parts of [start, end) not fetched yet."""
        pos, end = max(0, start), min(self.size, end)
        out = []
        i = max(0, bisect.bisect_right(self.starts, pos) - 1)
        while pos < end and i < len(self.starts) and self.starts[i] < end:
            if self.ends[i] > pos:
                if self.starts[i] > pos:
                    out.append((pos, self.starts[i]))
                pos = self.ends[i]
            i += 1
        if pos < end:
            out.append((pos, end))
        return out

    def _add(self, start: int, end: int) -> None:
        i = bisect.bisect_left(self.starts, start)
        self.starts.insert(i, start)
        self.ends.insert(i, end)
        starts: List[int] = []
        ends: List[int] = []
        for s, e in zip(self.starts, self.ends):
            if ends and s <= ends[-1]:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        self.starts, self.ends = starts, ends

    def fetch(self, ranges: List[Tuple[int, int]], merge_gap: Optional[int] = None) -> None:
        """Fetches whatever is missing of the [start, end) ranges, merging close ones, in parallel."""
        if merge_gap is None:
            merge_gap = RANGE_MERGE_GAP_BYTES
        spans: List[List[int]] = []
        for start, end in sorted(part for r in ranges for part in self._missing(*r)):
            if spans and start - spans[-1][1] < merge_gap:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
        if not spans:
            return

        def get(span: List[int]) -> bytes:
            return _get_range(self.path, span[0], span[1], self.timeout, self.retries)

        bodies = [get(spans[0])] if len(spans) == 1 else list(_range_executor().map(get, spans))
        for (start, end), body in zip(spans, bodies):
            self.buf[start:end] = body
            self._add(start, end)
            self.downloaded += len(body)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if self._missing(self.pos, self.pos + n):
            block = RANGE_BLOCK_BYTES
            self.fetch([(self.pos // block * block, -(-(self.pos + n) // block) * block)], merge_gap=block)
        b[:n] = self.buf[self.pos:self.pos + n]
        self.pos += n
        return n


def _open_remote(
    path: str, size: int, names: List[str], timeout: int, retries: int
) -> Tuple[_RangeFile, Dict[str, Tuple[Optional[Tuple[int, ...]], List[Tuple[Tuple[int, ...], int, int]]]]]:
    """
    Fetches a remote file's metadata and locates the stored chunks of `names`. Returns the file and,
    per name, (chunk shape or None, [(first element index, byte offset, byte size)]).
    """
    import h5py

    rf = _RangeFile(path, size, timeout, retries)
    rf.fetch([(0, RANGE_HEAD_BYTES), (size - RANGE_TAIL_BYTES, size)])
    with h5py.File(rf, "r") as h:
        # netCDF-C reads every object header and attribute when it opens a file, so fetch them all
        def touch(obj: Any) -> None:
            for name in obj.attrs:
                try:
                    obj.attrs[name]
                except Exception:
                    pass  # h5py cannot convert a few attribute types; their bytes are fetched with the header

        touch(h)
        h.visititems(lambda _, obj: touch(obj))
        chunks = {}
        for name in names:
            d = h[name]
            if d.chunks is None:
                offset = d.id.get_offset()  # None for compact data, which is inside the fetched header
                stored = [] if offset is None else [((0,) * d.ndim, offset, d.id.get_storage_size())]
            else:
                stored = [
                    (tuple(c.chunk_offset), c.byte_offset, c.size)
                    for c in (d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks()))
                ]
            chunks[name] = (d.chunks, stored)
    return rf, chunks



def ranged_decode_abi_file(
    path: str, size: int, variables: List[Tuple[str, str, Optional[float]]], bounds: Optional[LatLonBounds], timeout: int, retries: int
) -> Tuple[List[Dict[str, Any]], int]:
    """decode_abi_file on a remote file, fetching only metadata, x/y and the chunks inside the crop. Returns (decoded, bytes downloaded)."""
    import netCDF4 as nc

    data_names = sorted({name for dv, qv, _ in variables for name in (dv, qv)})
    rf, chunks = _open_remote(path, size, ["x", "y", *data_names], timeout, retries)
    try:
        grid = [(offset, offset + n) for name in ("x", "y") for _, offset, n in chunks[name][1]]
        window = None
        if bounds is not None:
            rf.fetch(grid)
            with nc.Dataset("inmem", mode="r", memory=rf.buf) as ds:
                x, y, projection = abi_grid(ds)
            window = crop_window(bounds, projection, x, y)
        needed = list(grid)
        for name in data_names:
            chunk_shape, stored = chunks[name]
            for first, offset, n in stored:
                if window is not None and chunk_shape is not None:
                    (cy, cx), (chy, chx) = first[-2:], chunk_shape[-2:]
                    if not (cy < window[1] and cy + chy > window[0] and cx < window[3] and cx + chx > window[2]):
                        continue
                needed.append((offset, offset + n))
        rf.fetch(needed)
        decoded = decode_abi_file(rf.buf, variables, window)
    finally:
        rf.release()
    for dec in decoded:
        dec["window"] = window
    return decoded, rf.downloaded


def ranged_decode_glm_file(path: str, size: int, timeout: int, retries: int) -> Tuple[Dict[str, np.ndarray], int]:
    """decode_glm_file on a remote file, fetching only metadata and GLM_READ_VARIABLES. Returns (decoded, bytes downloaded)."""
    rf, chunks = _open_remote(path, size, GLM_READ_VARIABLES, timeout, retries)
    try:
        rf.fetch([(offset, offset + n) for name in GLM_READ_VARIABLES for _, offset, n in chunks[name][1]])
        decoded = decode_glm_file(rf.buf)
    finally:
        rf.release()
    return decoded, rf.downloaded
