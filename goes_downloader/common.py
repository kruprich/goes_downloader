"""Constants and small helpers shared by the modules, including the worker processes."""

from __future__ import annotations

import os

import random
from datetime import datetime, timedelta, timezone

import numpy as np

SLOTS_PER_DAY = 288
SLOT_SECONDS = 300
GLM_WINDOW_SECONDS = 20
GLM_WINDOWS_PER_DAY = 86400 // GLM_WINDOW_SECONDS
NAT_NS = np.iinfo(np.int64).min  # int64 view of numpy NaT
# CF attributes that make xarray decode int64 nanosecond arrays to datetime64 (int64 min -> NaT)
TIME_ATTRS = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}

RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def to_ns(t: datetime) -> int:
    return int(np.datetime64(t.replace(tzinfo=None), "ns").astype(np.int64))


def backoff_seconds(attempt: int) -> float:
    return min(30.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random())


def iter_days_utc(start_date: str, end_date: str):
    cur = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


_CODE_VERSION = None


def code_version() -> dict:
    """
    This checkout's commit and whether it had uncommitted changes, for the stores' attributes, so a model
    trained on them can name the build (goes_lightning_model's provenance.data_version). Unknown (None)
    outside a git checkout. Asked of git once per process.
    """
    global _CODE_VERSION
    if _CODE_VERSION is None:
        import subprocess

        here = os.path.dirname(os.path.abspath(__file__))

        def git(*args):
            try:
                r = subprocess.run(["git", "-C", here, *args], capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                return None
            return r.stdout.strip() if r.returncode == 0 else None

        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no") if commit else None
        _CODE_VERSION = {"commit": commit, "dirty": bool(status) if commit else None}
    return _CODE_VERSION
