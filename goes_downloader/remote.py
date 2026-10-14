"""
The bucket (AWS S3 or Google Cloud Storage): file names, listing a day's files, and whole-file
downloads (main process). NOAA publishes the same files under the same names on both.
"""

from __future__ import annotations

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .common import RETRYABLE_HTTP_STATUS, backoff_seconds
from .config import RawConfig
from .progress import log

if TYPE_CHECKING:
    import aiohttp

# Hours listed per day; tests set this to a few hours to run a short download
LIST_HOURS = range(24)
# Objects per listing request (both clouds allow at most 1000); tests lower it to page through an hour
LIST_PAGE_SIZE = 1000

S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"

GOES_FILENAME_TIME_PATTERN = re.compile(r"_([sec])(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)")


@dataclass(frozen=True)
class RemoteFile:
    blob: str
    start: datetime
    end: datetime
    created: datetime
    size: int = 0


def parse_remote_file(blob: str, size: int = 0) -> Optional[RemoteFile]:
    times: Dict[str, datetime] = {}
    for m in GOES_FILENAME_TIME_PATTERN.finditer(os.path.basename(blob)):
        kind, y, j, hh, mm, ss, tenth = m.groups()
        times[kind] = datetime(int(y), 1, 1, tzinfo=timezone.utc) + timedelta(
            days=int(j) - 1, hours=int(hh), minutes=int(mm), seconds=int(ss), milliseconds=100 * int(tenth)
        )
    if set(times) != {"s", "e", "c"}:
        return None
    return RemoteFile(blob=blob, start=times["s"], end=times["e"], created=times["c"], size=size)


def object_url(bucket: str, blob: str) -> str:
    """HTTPS URL of an object in an s3:// or gs:// bucket (the anonymous, public endpoints)."""
    cloud, name = bucket.split("://")
    if cloud == "s3":
        return f"https://{name}.s3.amazonaws.com/{blob}"
    return f"https://storage.googleapis.com/{name}/{blob}"


async def _backoff(attempt: int) -> None:
    await asyncio.sleep(backoff_seconds(attempt))


async def http_get(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, str]] = None,
    *,
    as_json: bool,
) -> Any:
    import aiohttp  # only the main process downloads with aiohttp

    timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_seconds)
    last_error = "no attempts"
    for attempt in range(1, cfg.http_retries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as r:
                if r.status == 200:
                    return await (r.json() if as_json else r.read())
                last_error = f"HTTP {r.status}"
                if r.status not in RETRYABLE_HTTP_STATUS:
                    break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = repr(e)
        if attempt < cfg.http_retries:
            await _backoff(attempt)
    raise RuntimeError(f"GET {url} failed: {last_error}")


async def list_object_names(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    prefix: str,
) -> Dict[str, int]:
    """Object name -> size in bytes for everything under the prefix."""
    cloud, name = cfg.bucket.split("://")
    objects: Dict[str, int] = {}
    if cloud == "s3":  # ListObjectsV2, which answers in XML
        url = object_url(cfg.bucket, "")
        params: Dict[str, str] = {"list-type": "2", "prefix": prefix, "max-keys": str(LIST_PAGE_SIZE)}
        while True:
            page = ET.fromstring(await http_get(cfg, session, url, params, as_json=False))
            for it in page.iter(f"{S3_NAMESPACE}Contents"):
                key = it.findtext(f"{S3_NAMESPACE}Key")
                if key:
                    objects[key] = int(it.findtext(f"{S3_NAMESPACE}Size", "0"))
            token = page.findtext(f"{S3_NAMESPACE}NextContinuationToken")
            if not token:
                break
            params["continuation-token"] = token
    else:  # the JSON API
        url = f"https://storage.googleapis.com/storage/v1/b/{name}/o"
        params = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": str(LIST_PAGE_SIZE)}
        while True:
            js = await http_get(cfg, session, url, params, as_json=True)
            objects.update((it["name"], int(it.get("size", 0))) for it in js.get("items", []) if it.get("name"))
            token = js.get("nextPageToken")
            if not token:
                break
            params["pageToken"] = token
    return objects


async def list_day_names(
    cfg: RawConfig,
    session: aiohttp.ClientSession,
    product_prefix: str,
    day: datetime,
) -> Tuple[Dict[str, int], int]:
    """Returns (name -> size for every object under the day's 24 hourly prefixes, n_failed_prefixes)."""
    jday = day.timetuple().tm_yday
    prefixes = [f"{product_prefix}/{day.year}/{jday:03d}/{hh:02d}/" for hh in LIST_HOURS]
    results = await asyncio.gather(
        *(list_object_names(cfg, session, p) for p in prefixes), return_exceptions=True
    )

    objects: Dict[str, int] = {}
    n_failed = 0
    for prefix, res in zip(prefixes, results):
        if isinstance(res, BaseException):
            n_failed += 1
            log(f"!! listing failed for {prefix}: {res!r}")
            continue
        objects.update(res)
    return objects, n_failed


def select_files(objects: Dict[str, int], filename_must_contain: str) -> List[RemoteFile]:
    """
    Files whose name contains the pattern, deduplicated by scan start time (keeping the newest
    creation time, i.e. reprocessed files) and sorted by start time.
    """
    by_start: Dict[datetime, RemoteFile] = {}
    for name, size in objects.items():
        if filename_must_contain not in name:
            continue
        rf = parse_remote_file(name, size)
        if rf is None:
            continue
        prev = by_start.get(rf.start)
        if prev is None or rf.created > prev.created:
            by_start[rf.start] = rf
    return sorted(by_start.values(), key=lambda rf: rf.start)
