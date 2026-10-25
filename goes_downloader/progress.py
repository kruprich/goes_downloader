"""Log lines and the one-line progress display."""

from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if current is not None:
        current.clear()
    print(f"[{ts}] {msg}", flush=True)
    if current is not None:
        current.restore()


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class Progress:
    """
    One-line progress bar with download speed and ETA. In a terminal it is redrawn in place and log
    lines print above it; when output goes to a file it prints a plain progress line once a minute.
    Each day's download size is known exactly once it is listed; days not listed yet are estimated
    from the average listed day. The bar and ETA count source file sizes; the GB downloaded and MB/s
    shown are what actually went over the network, which is less with byte-range downloads.
    """

    WIDTH = 30
    SPEED_WINDOW_SECONDS = 30.0
    PLAIN_EVERY_SECONDS = 60.0

    def __init__(self, total_days: int):
        self.total_days = total_days
        self.days_done = 0
        self.day_total: Dict[str, int] = {}  # source file bytes, per day in progress
        self.day_done: Dict[str, int] = {}
        self.listed_day_bytes: List[int] = []
        self.bytes_done = 0  # source file bytes of the files handled so far
        self.bytes_downloaded = 0
        self.bytes_saved = 0  # on-disk size of the day stores written this run, added as each day finishes
        self.started = time.monotonic()
        self.samples: deque = deque([(self.started, 0, 0)])
        self.tty = sys.stdout.isatty()
        self.text = ""
        self.shown = False
        self.last_plain = self.started

    def start_day(self, day: str, total_bytes: int) -> None:
        self.day_total[day] = total_bytes
        self.day_done[day] = 0
        self.listed_day_bytes.append(total_bytes)

    def add_bytes(self, day: str, n: int, downloaded: Optional[int] = None) -> None:
        """n: source file bytes handled; downloaded: bytes fetched for them (default n)."""
        if day in self.day_done:
            self.day_done[day] += n
        self.bytes_done += n
        self.bytes_downloaded += n if downloaded is None else downloaded

    def add_saved(self, n: int) -> None:
        self.bytes_saved += n

    def finish_day(self, day: str) -> None:
        self.day_total.pop(day, None)
        self.day_done.pop(day, None)
        self.days_done += 1

    def _speeds(self) -> Tuple[float, float]:
        """(source file bytes/s, downloaded bytes/s) over the last SPEED_WINDOW_SECONDS."""
        now = time.monotonic()
        self.samples.append((now, self.bytes_done, self.bytes_downloaded))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.SPEED_WINDOW_SECONDS:
            self.samples.popleft()
        t0, b0, d0 = self.samples[0]
        if now <= t0:
            return 0.0, 0.0
        return (self.bytes_done - b0) / (now - t0), (self.bytes_downloaded - d0) / (now - t0)

    def _remaining_bytes(self) -> Optional[float]:
        in_progress = sum(max(0, self.day_total[d] - self.day_done[d]) for d in self.day_total)
        not_started = self.total_days - self.days_done - len(self.day_total)
        if not_started <= 0:
            return float(in_progress)
        if not self.listed_day_bytes:
            return None
        return in_progress + not_started * sum(self.listed_day_bytes) / len(self.listed_day_bytes)

    def _line(self) -> str:
        speed, download_speed = self._speeds()
        partial = sum(min(1.0, self.day_done[d] / self.day_total[d]) for d in self.day_total if self.day_total[d])
        frac = min(1.0, (self.days_done + partial) / max(1, self.total_days))
        filled = int(round(frac * self.WIDTH))
        remaining = self._remaining_bytes()
        eta = format_duration(remaining / speed) if remaining is not None and speed > 0 else "--"
        return (
            f"[{'█' * filled}{'░' * (self.WIDTH - filled)}] {100 * frac:5.1f}%  "
            f"{self.days_done}/{self.total_days} days  {self.bytes_downloaded / 1e9:.1f} GB downloaded  "
            f"{self.bytes_saved / 1e9:.1f} GB saved  {download_speed / 1e6:5.1f} MB/s  ETA {eta}"
        )

    def draw(self) -> None:
        self.text = self._line()
        if self.tty:
            self.restore()
        elif time.monotonic() - self.last_plain >= self.PLAIN_EVERY_SECONDS:
            self.last_plain = time.monotonic()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] progress {self.text}", flush=True)

    def restore(self) -> None:
        if self.tty and self.text:
            sys.stdout.write("\r\x1b[2K" + self.text)
            sys.stdout.flush()
            self.shown = True

    def clear(self) -> None:
        if self.tty and self.shown:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            self.shown = False

    def close(self) -> None:
        """Leaves the final progress line on screen and moves to the next line."""
        self.draw()
        if self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] progress {self.text}", flush=True)
        self.shown = False


# The run's progress display while one is active; log() keeps its line below the log output
current: Optional[Progress] = None
