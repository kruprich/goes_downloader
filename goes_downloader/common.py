"""Constants and small helpers shared by the modules, including the worker processes."""

from __future__ import annotations

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
