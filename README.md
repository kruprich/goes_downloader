# goes_downloader

Downloads GOES-19 ABI (CONUS L2) and GLM (LCFA) data from the public GCS bucket
`gcp-public-data-goes-19` into one raw Zarr store per UTC day, for building a
lightning-prediction training dataset. No time binning or spatial aggregation happens
here; that belongs in a separate feature-building step.

For each UTC day it:

1. Lists ABI and GLM files over the GCS JSON API, retrying transient errors.
2. ABI: stores every 5-min scan on the product's grid, cropped to `bounds`, as the file's packed
   integers plus its DQF flags. The current config stores bands 2, 5, 13 and 15 at 2 km (read
   from the all-band MCMIPC files) and ACHA cloud height, TPW and the DSI stability indices at 10 km.
3. GLM: stores every flash (time, lat/lon, area, energy, quality flag) and every event (lat/lon
   and the flash it belongs to), plus a coverage flag for every 20-second window, so outages
   (including files NOAA never published) show up as gaps rather than as "no lightning".
4. Writes `<raw_output_root_dir>/YYYY/YYYYDDD.zarr` and marks it `complete` only if every listing and file
   succeeded and the day ended at least 24 hours ago, since NOAA publishes some files late.
   Incomplete days are rebuilt on the next run.

The full store layout, including every attribute, is described in the docstring at the top of
`goes_downloader/store.py`.

## Files

| Path | What it is |
|---|---|
| `download_raw.py` | Runs the download (`python download_raw.py --config config.json`) |
| `report.py` | Shows which days the stores hold and how complete they are |
| `config.json` | Date range, output folder, products to download and all settings |
| `goes_downloader/` | The code; `store.py` documents the store layout |
| `tests/` | The test suite (see [Tests](#tests)) |
| `pyproject.toml` | The package, supported Python versions, dependencies and test settings |
| `requirements.lock` | The exact dependency versions it was tested with |
| `.github/workflows/tests.yml` | Runs the offline tests on GitHub for every push |

## Setup

Use Python 3.12 or 3.13 (both tested). The tested versions in `requirements.lock` need 3.12 or
newer, and Python 3.14 does not work: the `zarr<3` pin needs `numcodecs<0.16`, which has no 3.14
wheel. Installing on another version stops with "requires a different Python".

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate   # or: python3.13 -m venv .venv
uv pip install -e . -c requirements.lock                   # or: pip install -e . -c requirements.lock
```

`pyproject.toml` lists the direct dependencies; `-c requirements.lock` holds them to the exact
versions the downloader was tested with. `-e` installs the package in place, so edits to the code
take effect without reinstalling.

What a run needs, with the current config (measured on 2025-10-27, 12 CPU cores, ~60 MB/s):

- **Disk:** about 2.1 GB per day stored (measured on an October day; summer days hold more
  daylight imagery), so ~1.12–1.24 TB for the 534 days of the configured range.
- **Network:** about 4.9 GB per day downloaded. A day takes 95–110 s, so the range takes 14–16 h.
- **Memory:** about 10 GB: 7.2–7.7 GB for the 48 byte-range workers and up to 1.9 GB for the main
  process, which holds a day's GLM events until the day is written. Lower `byte_range_processes`
  to use less.

## Running a download

```bash
python download_raw.py --config config.json
```

Days already marked `complete` are skipped unless `overwrite_existing` is `true`, so running the
same command again continues where the last run stopped and retries incomplete days.

While it runs, the last line shows a progress bar, days done, data downloaded, data saved to
disk (added as each day finishes), download speed (last 30 s) and an ETA. The ETA uses each
day's exact size once the day is listed and the average listed day for days not reached yet.
When output is redirected to a file, a progress line is printed once a minute instead. When the
run finishes, it prints a summary and then the data report (see
[Checking the data](#checking-the-data)); `--no-report` skips the report.

Ctrl-C stops the run within a few seconds: files being decoded finish, the days in progress are
left as `.zarr.tmp` folders, and running the same command again rebuilds those days and continues.

For a long run, use `screen` or `tmux` so it survives closing the terminal, and keep a log:

```bash
screen -S goes                     # detach with Ctrl-A D, reattach with: screen -r goes
source .venv/bin/activate
python download_raw.py --config config.json 2>&1 | tee -a goes_download.log
```

Lines starting with `!!` in the log are errors (`grep '!!' goes_download.log`). Don't edit the
code while a run is going: a worker process that dies is replaced, and the replacement loads
whatever the code is at that moment.

## Checking the data

`report.py` reads every day store in the config's date range and lists the days that need
attention: not complete (or missing, or interrupted), an ABI product with less than 98% of its
scans, or less than 99% GLM coverage. The download prints the same report when a run finishes,
covering the whole date range including days done by earlier runs (not after Ctrl-C).

```bash
python report.py --config config.json                  # summary and days needing attention
python report.py --config config.json --all --csv days.csv
```

## Config basics

- `gcs_bucket`: the public bucket to read (`gcp-public-data-goes-19`).
- `start_date_utc`, `end_date_utc` (`YYYY-MM-DD`): the UTC days to download, both included. The
  config starts on 2025-04-10, just after NOAA declared GOES-19 operational as GOES-East on
  2025-04-07. Its data is on the operational 75.0°W grid from 2025-04-02, and before 2025-03-20 it
  came from the 89.5°W test position (don't mix the two). It ends on the latest day that had ended
  24 hours before the last update; move the end date forward and run again to add newer days.
- `raw_output_root_dir` (default `./goes_raw`, relative to where you run the command): where the
  day stores go. The config uses `./goes_raw2`; `goes_raw*/` folders are git-ignored.
- `overwrite_existing` (default `false`): rebuild days that are already complete, e.g. after
  changing the products or `bounds`.
- `download`: the product keys to fetch (see below).

A key the downloader doesn't know is reported when the run starts (`!! ... ignoring keys the
downloader doesn't use`), since a misspelled setting would otherwise silently fall back to its
default. Keys starting with `_`, such as `"_comment"`, are for notes and are skipped silently.

Days that ended less than 24 hours before the run are stored but not marked complete (the report
shows them as "too recent"); any later run rebuilds them, and marks them complete once the
24 hours have passed.

## Choosing products

`abi_products` and `glm_product` in `config.json` define the products that can be downloaded
(all 16 CMIP bands in two sets, plus the CONUS L2 products), each with a `description`. Only the
keys listed in `download` are fetched, for example:

```json
"download": ["mcmip_c02", "mcmip_c05", "mcmip_c13", "mcmip_c15", "acha", "tpw", "dsi_cape", "dsi_li", "dsi_ki", "glm"]
```

The 16 CMIP bands come in two sets. `cmip_c01`–`cmip_c16` read each band's own file at its native
resolution (0.5 km for band 2, 1 km for bands 1, 3 and 5, 2 km for the rest). `mcmip_c01`–
`mcmip_c16` read the band from the all-band MCMIPC file, where every band is at 2 km. With
`"download_method": "byte_ranges"` only the chosen bands are fetched from it (about 1–4 MB per band
per scan inside `bounds`); with `"whole_files"` every ~59 MB file is downloaded in full. Band 2 at
native resolution is ~19.5 GB/day to download, so the MCMIPC set is the practical way to get the
visible and near-infrared bands. For bands 13 and 15 both sets give identical data.

Products that read different variables from the same files, like the `mcmip_*` bands or the five
`dsi_*` stability indices (CAPE, lifted index, K-index, Showalter index, total totals), share one
download per scan. Leave out `"glm"` to skip lightning. Changing the list does not touch days
already marked complete; delete them or set `overwrite_existing` to rebuild them with the new
products.

## Reading the data

Reading needs `xarray` and `dask`, which the downloader itself does not install
(`uv pip install xarray dask`; checked with xarray 2026.7 and dask 2026.8). Each day store opens
directly with xarray, one product (or GLM table) at a time, and several days combine into one
continuous time series:

```python
import glob, xarray as xr

days = sorted(glob.glob("goes_raw2/*/*.zarr"))  # the config's raw_output_root_dir
c13 = xr.open_mfdataset(days, engine="zarr", group="abi/mcmip_c13", combine="by_coords")
flashes = xr.open_mfdataset(days, engine="zarr", group="glm/flashes", combine="nested", concat_dim="flash")
coverage = xr.open_mfdataset(days, engine="zarr", group="glm/coverage", combine="by_coords")
events = xr.open_zarr(days[0], group="glm/events")  # one day: flash_index refers to that day's flashes
```

`coverage.covered` has one entry per 20-second window: 0 means there is no GLM data for that
window, so no flashes there does not mean no lightning. NOAA misses a few GLM files on most days,
and on 2026-07-15 published no GLM data from 20:22:40 UTC to midnight. A flash's `file_index` is
its row in the same day's `glm/files`, so it no longer identifies a file once several days are
combined; use the flash times instead.

`glm/events` holds every GLM event (one lit pixel of the lightning sensor at one moment) with its
`lat`, `lon` and `flash_index`, the row of its flash in the same day's `glm/flashes`. Events are
what shows how far each flash spread, which a flash-extent density needs; the flash table only has
each flash's centre and area. Like `file_index`, `flash_index` only works within one day's store.

ABI values are the file's packed numbers, not physical units: the physical value is
`value * scale_factor + add_offset` (both stored per time slot). xarray turns fill values into NaN,
which makes `value` a float array; pass `mask_and_scale=False` to get the stored integers. Time
slots without a scan read as all NaN, and the ABI x/y coordinates are `x_scan_angle_rad` and
`y_scan_angle_rad`.

## Assumptions about the source data

**Checked on every scan.** A scan that breaks one of these counts as a file error, which leaves the
day incomplete so it is retried and visible in the log:

- Within a day, each product's grid (shape and x/y scan angles), fill value and quality-flag data
  type stay the same as in its first scan.
- Data values are packed integers that fit in 16 bits; quality flags fit in 8 or 16 bits.
- In each GLM file, event, group and flash ids are unique, and every event's parent group and
  every group's parent flash are in the same file.

**Recorded, but expected to stay constant.** `scale_factor` and `add_offset` are stored for every
scan, so a change would be captured. For GOES-19 they have never changed: one file per month for
every current product, from the start of the data (Oct 2024 for CMIP and MCMIPC, Dec 2024 for the
others) to Sep 2026, all had the same scale, offset and fill value. The feature step can treat
them as constant per product but should check.

**Assumed, not checked:**

- ABI stays in scan mode 6 (CONUS every 5 minutes). Every product's filename filter contains
  `-M6`, so scans taken in other modes are not downloaded. Each 5-minute slot holds at most one
  scan; an extra scan in the same slot is skipped and counted in `n_abi_slot_collisions`.
- The bucket stores files under `<product>/<year>/<day-of-year>/<hour>/` by scan start time, and
  file names carry `_s`, `_e` and `_c` (start, end, creation) times. When NOAA republishes a scan,
  the file with the newest creation time is used.
- All ABI products share one projection (the GOES-East fixed grid, centred on 75.0°W) since
  GOES-19 moved to the GOES-East position in April 2025. Earlier files use its 89.5°W test
  position, so don't mix those dates with later ones. The `bounds` crop is computed from the first
  ABI scan decoded each day and applied to every ABI product.
- GLM flash times are "seconds since" offsets in the file. A GLM file belongs to the day its start
  time falls in, so a flash in the first file of a day can begin a second or two before midnight;
  select flashes by their own times, not by which day store holds them.
- NOAA publishes a day's files within 24 hours of the day ending; later files are missed once
  the day is marked complete.
- In a complete day, ABI `present` = 0 and GLM `covered` = 0 mean NOAA did not publish that scan
  or file; a failed download would have left the day incomplete.

## Config options for storage

- `cmip_precision` (K, `0.25` in the config): rounds the infrared CMIP bands (7–16) to save space.
  The step is the largest power-of-two number of packed counts that is at most this many K (0.246 K
  for band 13, 0.238 K for band 15) and is stored per scan in `quantization_step`; each value moves
  by at most half a step. This cuts CMIP storage by about 26%; `null` keeps full precision. The
  visible and near-infrared bands (1–6) are reflectance and keep full precision.
- `precision` (per ABI product, in the product's units): overrides `cmip_precision` for that
  product, or rounds any other product. Categorical products such as cloud top phase must not set it.
- `bounds` (`lat_min`, `lat_max`, `lon_min`, `lon_max`): **applies to ABI only.** Crops every ABI
  product to the smallest scan-angle rectangle containing the box; `null` keeps the full CONUS
  sector. GLM is never cropped: every flash and event in GLM's field of view is kept (about
  240 MB/day, mostly events), and the feature step selects what its grid needs.
  The config uses 24–50°N, 125–66°W, the contiguous-US bounding box NOAA NCEI publishes for its
  nClimGrid-Daily dataset. That box keeps about 65% of the ABI pixels.

Measured on 2025-10-27 with the current products and that box: 2.1 GB/day. Bands 13 and 15 take
0.46 GB each, bands 2 and 5 0.37 GB each and GLM 0.24 GB. Bands 2 and 5 are reflectance, so they
shrink to about a tenth at night; days with more daylight store more.

## Config options for speed

- `download_method` (`"whole_files"` if not set; the config uses `"byte_ranges"`): how files are
  fetched.
  - `"whole_files"` downloads every file in full and decodes it in `max_decode_processes` processes.
  - `"byte_ranges"` downloads only the parts that are used: each file's internal HDF5 index, the
    variables that are stored (for GLM the flash and event variables and the ids linking them, not
    the rest of the group data), and for ABI only the bands and image blocks that overlap
    `bounds`. With the current products that is 4.9 GB per day out of 16.1 GB of source files.
    Each file then takes a few requests in a row, so the requests run in `byte_range_processes`
    worker processes. The stored data is identical either way.
- `byte_range_processes` (48): with `"byte_ranges"`, worker processes that each fetch and decode
  one file at a time. Together they peaked at 7.2–7.7 GB of memory with the current products.
- `max_concurrent_http_requests` (32): open connections shared by all days, for listing and, with
  `"whole_files"`, downloads. Byte-range workers use their own connections.
- `http_timeout_seconds` (60): time limit for one request, including reading the whole file. With
  `"byte_ranges"` it limits each network read instead.
- `http_retries` (5): attempts per request. Timeouts, connection errors and HTTP 408, 429 and 5xx
  are retried with backoff; other errors fail at once.
- `max_decode_processes` (4): with `"whole_files"`, worker processes that decode the NetCDF files.
- `max_days_in_flight` (2): days processed at the same time, so the next day's listing overlaps the
  current day's downloads.
- `max_inflight_payloads` (optional): files per day that can be between download start and being
  written to Zarr, which caps memory use. Defaults to `max_concurrent_http_requests`, or to
  `byte_range_processes` with `"byte_ranges"`.

Download speed is usually limited by the internet connection. On a connection that delivers about
60 MB/s, 64 connections with 8 decode processes was no faster than the defaults, 128 connections
was slower, and decoding used only about 1.3 CPU cores.

Measured on one full day (2025-10-27) on that connection with 12 CPU cores. With the current
products, `"byte_ranges"` with 48 processes took 95–110 s, downloaded 4.9 GB, used 288 s of CPU
and 7.2–7.7 GB for the workers; `"whole_files"` would download 16.1 GB. An earlier comparison,
with single-band files for bands 8, 11, 13 and 15 and GLM flashes only:

| `download_method` | Time per day | Downloaded | CPU time | Worker memory |
|---|---|---|---|---|
| `"whole_files"` | 106–107 s | 6.9 GB | 125 s | 0.5 GB |
| `"byte_ranges"`, 32 processes | 90 s | 4.7 GB | not measured | 6.3 GB |
| `"byte_ranges"`, 48 processes | 79–82 s | 4.7 GB | 438 s | 9.5 GB |

Byte ranges save about a quarter of the time, but only with enough worker processes. GLM files are
small, so fetching them is limited by request round trips, not bandwidth. They overlap with the
bandwidth-bound ABI files only because files start in scan-time order.

## Tests

```bash
uv pip install -e ".[dev]" -c requirements.lock
pytest              # offline, under a second: small synthetic files stand in for the GOES ones
pytest --network    # also real files and a full test hour both ways (~0.5 GB, ~30 s)
```

GitHub runs pyflakes and the offline tests on Python 3.12 and 3.13 for every push and pull
request (`.github/workflows/tests.yml`); the network tests are run by hand.

## Troubleshooting

- **"requires a different Python" when installing:** use Python 3.12 or 3.13.
- **A day stays incomplete:** run the same command again; failed files are retried. The `!!` log
  lines and `report.py` show which days and files failed. A file that fails on every run points
  to a problem with that file at the source.
- **"too recent" in the report:** the day ended less than 24 hours before the run; run again later.
- **Running out of memory:** lower `byte_range_processes` (32 used ~6.3 GB and was ~10% slower).
- **Slower than expected:** check the connection's speed first. With `"byte_ranges"`, fewer than
  ~48 workers leaves bandwidth unused.
