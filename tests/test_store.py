"""Writing the GLM tables: flash order, event links, per-file counts and coverage."""

from datetime import datetime, timedelta, timezone

import numpy as np
import zarr

from goes_downloader.common import GLM_WINDOWS_PER_DAY, to_ns
from goes_downloader.remote import RemoteFile
from goes_downloader.store import write_glm_tables

DAY = datetime(2025, 10, 27, tzinfo=timezone.utc)


def glm_table(first_seconds, lats, event_flash):
    """One decoded file's tables; each flash is identified by its latitude."""
    n = len(lats)
    return {
        "time_first_ns": np.array([to_ns(DAY + timedelta(seconds=s)) for s in first_seconds], dtype=np.int64),
        "time_last_ns": np.array([to_ns(DAY + timedelta(seconds=s + 0.2)) for s in first_seconds], dtype=np.int64),
        "lat": np.array(lats, dtype=np.float32),
        "lon": np.zeros(n, np.float32),
        "area_m2": np.ones(n, np.float32),
        "energy_j": np.ones(n, np.float32),
        "quality_flag": np.zeros(n, np.uint8),
        "event_lat": np.array([lats[i] if i >= 0 else -99.0 for i in event_flash], dtype=np.float32),  # on its flash
        "event_lon": np.zeros(len(event_flash), np.float32),
        "event_flash": np.array(event_flash, dtype=np.int32),
    }


def test_events_point_at_their_flash_after_sorting(tmp_path):
    files = [
        RemoteFile(blob=f"f{i}", start=DAY + timedelta(seconds=20 * i), end=DAY + timedelta(seconds=20 * i + 20),
                   created=DAY, size=1)
        for i in range(4)
    ]
    # File 1 failed; flashes are out of time order within and across files
    tables = {
        0: glm_table([15.0, 2.0, 9.0], [10.0, 11.0, 12.0], [0, 0, 2, 1, 2, -1]),  # the last is an orphan
        2: glm_table([41.0, 40.5], [20.0, 21.0], [1, 1, 0]),
        3: glm_table([], [], [-1, -1]),  # last file: no flashes, only orphan events
    }
    root = zarr.open_group(str(tmp_path / "day.zarr"), mode="w")
    n_flashes, n_events, n_covered = write_glm_tables(root, DAY, files, np.array([True, False, True, True]), tables)
    assert (n_flashes, n_events, n_covered) == (5, 11, 3)

    flashes, events = root["glm/flashes"], root["glm/events"]
    times = flashes["time_first_ns"][:]
    assert (np.diff(times) >= 0).all()
    assert flashes["lat"][:].tolist() == [11.0, 12.0, 10.0, 21.0, 20.0]
    assert flashes["file_index"][:].tolist() == [0, 0, 0, 2, 2]
    # Every event still points at its own flash (same latitude) after the flashes were sorted
    index = events["flash_index"][:]
    assert index.tolist().count(-1) == 3 and events["lat"][:][index == -1].tolist() == [-99.0] * 3
    assert (flashes["lat"][:][index[index >= 0]] == events["lat"][:][index >= 0]).all()
    assert root["glm/files/n_flashes"][:].tolist() == [3, 0, 2, 0]
    assert root["glm/files/n_events"][:].tolist() == [6, 0, 3, 2]
    assert root["glm/files/n_orphan_events"][:].tolist() == [1, 0, 0, 2] and root.attrs["n_glm_orphan_events"] == 3
    covered = root["glm/coverage/covered"][:]
    assert len(covered) == GLM_WINDOWS_PER_DAY and covered[:4].tolist() == [1, 0, 1, 1] and covered.sum() == 3
