# goes_downloader

Downloads GOES-19 ABI (CONUS L2) and GLM (LCFA) data from NOAA's public bucket on AWS
(`s3://noaa-goes19`) or Google Cloud (`gs://gcp-public-data-goes-19`) into one raw Zarr store per
UTC day, for building a lightning-prediction training dataset. The download does no time binning
or spatial aggregation. A separate feature step (see [Building features](#building-features)) turns
the raw stores into CNN inputs and lightning labels on one grid.

For each UTC day it:

1. Lists the day's ABI and GLM files, retrying transient errors.
2. ABI: stores every 5-minute scan on the product's own grid, cropped to `bounds`, as the file's
   packed integers plus its quality flags (DQF).
3. GLM: stores every flash (time, lat/lon, area, energy, quality flag), every event (lat/lon and
   its flash) and a coverage flag for every 20-second window, so outages show up as gaps rather
   than as "no lightning".
4. Writes `<raw_output_root_dir>/YYYY/YYYYDDD.zarr`, marked `complete` only if every listing and
   file succeeded and the day ended at least 24 hours ago (NOAA publishes some files late).
   Incomplete days are rebuilt on the next run.

The docstring of `goes_downloader/store.py` describes the store layout and every attribute. Stores
are Zarr format 3; format-2 stores from earlier versions count as missing and are rebuilt.

## Files

| Path | What it is |
|---|---|
| `download_raw.py` | Runs the download |
| `download_mrms.py` | Downloads MRMS radar products into their own day stores (see [MRMS radar](#mrms-radar)) |
| `report.py` | Shows which days the stores hold and how complete they are |
| `build_geography.py` | Puts terrain, land/water, viewing angle and country borders on the feature grid, once |
| `build_features.py` | Builds the per-day feature stores (CNN inputs and lightning labels) |
| `config.json` | Date range, output folder, products to download and all settings |
| `goes_downloader/` | The code; `store.py` documents the raw store layout, `features/__init__.py` the feature stores |
| `tests/` | The test suite (see [Tests](#tests)) |
| `requirements.lock` | The exact dependency versions the code was tested with |

## Setup

Use Python 3.12, 3.13 or 3.14.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e . -c requirements.lock    # -c pins the tested versions; uv pip works the same way
```

What a run needs with the example config (one October day, read from Google Cloud on a 12-core
machine with a ~60 MB/s connection):

- **Disk:** about 1.7 GB per day (2025-10-27); summer days hold more daylight imagery: 2.0 GB on
  2025-07-15.
- **Network:** about 4.9 GB per day. A day took 95–110 s.
- **Memory:** about 10 GB: 7.2–7.7 GB for the 48 byte-range workers and up to 1.9 GB for the main
  process, which holds a day's GLM events until the day is written. Lower `byte_range_processes` to
  use less.

## Running a download

```bash
python download_raw.py --config config.json
```

Days already marked `complete` are skipped unless `overwrite_existing` is `true`, so running the same
command again continues where the last run stopped and retries incomplete days.

The last line shows a progress bar, days done, data downloaded and saved, download speed and an ETA
(a line a minute when output goes to a file). At the end it prints a summary and the data report
(see [Checking the data](#checking-the-data)); `--no-report` skips the report.

Ctrl-C stops the run within a few seconds; days in progress are left as `.zarr.tmp` folders and
rebuilt on the next run. For a long run, use `screen` or `tmux` and keep a log:

```bash
python download_raw.py --config config.json 2>&1 | tee -a goes_download.log
```

Lines starting with `!!` are errors. Don't edit the code during a run: a replacement for a dead
worker process loads the code as it is at that moment.

## Checking the data

`report.py` reads every day store in the config's date range and lists the days that are not
complete, missing or interrupted, where an ABI product has less than 98% of its scans, or with less
than 99% GLM coverage.

```bash
python report.py --config config.json                  # summary and days needing attention
python report.py --config config.json --all --csv days.csv
```

## Config basics

- `bucket`: `s3://<name>` or `gs://<name>`. NOAA publishes the same files under the same names on AWS
  (`s3://noaa-goes19`) and Google Cloud (`gs://gcp-public-data-goes-19`), read anonymously over HTTPS,
  so the stored data is identical either way (see [Choosing a cloud](#choosing-a-cloud)).
- `start_date_utc`, `end_date_utc` (`YYYY-MM-DD`): the UTC days to download, both included. To add
  newer days, move the end date forward and run again. NOAA declared GOES-19 operational as
  GOES-East on 2025-04-07; its data is on the operational 75.0°W grid from 2025-04-02, and before
  2025-03-20 came from the 89.5°W test position. Don't mix the two positions in one dataset.
- `raw_output_root_dir` (default `./goes_raw`): where the day stores go, relative to the directory
  you run the command from.
- `overwrite_existing` (default `false`): rebuild days that are already complete, for example after
  changing the products or `bounds`.
- `download`: the product keys to fetch (see [Choosing products](#choosing-products)).

A run reports any config key it doesn't know (`!! ... ignoring keys`), since a misspelled setting
would otherwise fall back to its default. Keys starting with `_`, such as `"_comment"`, are notes.

Days that ended less than 24 hours before the run are stored but not marked complete; the report shows
them as "too recent", and a later run completes them.

## Choosing products

`abi_products` and `glm_product` in `config.json` define the products that can be downloaded, each
with a `description`. Only the keys listed in `download` are fetched, for example:

```json
"download": ["mcmip_c02", "mcmip_c05", "mcmip_c13", "mcmip_c15", "acha", "tpw", "dsi_cape", "dsi_li", "dsi_ki", "glm"]
```

The 16 CMIP bands come in two sets:

- `cmip_c01`–`cmip_c16` read each band's own file at its native resolution: 0.5 km for band 2,
  1 km for bands 1, 3 and 5, and 2 km for the rest. Band 2 is then about 19.5 GB per day to download.
- `mcmip_c01`–`mcmip_c16` read the band from the all-band MCMIPC file, where every band is at 2 km.
  With `"download_method": "byte_ranges"` only the chosen bands are fetched, about 1–4 MB per band
  per scan inside `bounds`; with `"whole_files"`, every ~59 MB file is downloaded in full.

Products that read the same files, such as the `mcmip_*` bands and the five `dsi_*` stability
indices, share one download per scan. Leave out `"glm"` to skip lightning. Changing `download`
doesn't touch days already complete: delete them or set `overwrite_existing`.

## Reading the data

Reading needs `xarray` and `dask` in releases that read Zarr format 3 with zarr 3 (checked with
xarray 2026.9 and dask 2026.8), and `numcodecs[pcodec]` outside the downloader's environment:
`pip install xarray dask "numcodecs[pcodec]"`. Each day store opens with xarray, one product or GLM
table at a time, and several days combine into one time series:

```python
import glob, xarray as xr

days = sorted(glob.glob("goes_raw/*/*.zarr"))  # your raw_output_root_dir
c13 = xr.open_mfdataset(days, engine="zarr", group="abi/mcmip_c13", combine="by_coords")
flashes = xr.open_mfdataset(days, engine="zarr", group="glm/flashes", combine="nested", concat_dim="flash")
coverage = xr.open_mfdataset(days, engine="zarr", group="glm/coverage", combine="by_coords")
events = xr.open_zarr(days[0], group="glm/events")  # one day: flash_index refers to that day's flashes
```

**ABI values** are the file's packed integers: `physical = value * scale_factor + add_offset`, both
stored per time slot. xarray turns fill values into NaN, so `value` reads as floats and slots without
a scan as all NaN; pass `mask_and_scale=False` for the stored integers. The x/y coordinates are
`x_scan_angle_rad` and `y_scan_angle_rad`.

**GLM coverage:** `coverage.covered` has one entry per 20-second window; 0 means no GLM data, so no
flashes there doesn't mean no lightning. NOAA misses a few GLM files on most days, and occasionally
hours at a time.

**GLM events:** `glm/events` holds every event (one lit pixel of the lightning sensor at one moment)
with its `lat`, `lon` and `flash_index`, the row of its flash in the same day's `glm/flashes`. Events
show how far each flash spread, which a flash-extent density needs. A few orphan events, whose group or
flash is missing from NOAA's file or ambiguous because 16-bit flash ids wrap around, have
`flash_index` -1 and are counted in `n_orphan_events`. `flash_index` and a flash's `file_index` are
only valid within one day's store; across days, use the flash times.

## Assumptions about the source data

**Checked on every scan**, as a file error that leaves the day incomplete:

- Within a day, each product's grid (shape and scan angles), fill value and quality-flag data type
  stay as in its first scan.
- Data values are packed integers that fit in 16 bits, and quality flags fit in 8 or 16 bits.

**Recorded, not checked:** `scale_factor` and `add_offset` are stored for every scan. For GOES-19 they
have not changed for any current product from the start of its data through September 2026.

**Assumed:**

- ABI stays in scan mode 6, with CONUS every 5 minutes; every filename filter contains `-M6`. A second
  scan in one 5-minute slot is skipped and counted in `n_abi_slot_collisions`.
- Files are stored under `<product>/<year>/<day-of-year>/<hour>/`, with `_s`, `_e` and `_c` times
  (scan start, scan end, creation) in their names. When NOAA republishes a scan, the newest file is used.
- All ABI products share the GOES-East fixed grid; the `bounds` crop is computed from the day's first
  ABI scan.
- A GLM file belongs to the day its start time falls in, so a flash in a day's first file can begin a
  second or two before midnight. Select flashes by their own times.
- NOAA publishes a day's files within 24 hours of the day ending; later files are missed. So in a
  complete day, ABI `present` = 0 and GLM `covered` = 0 mean NOAA didn't publish that file.

## Config options for storage

- `cmip_precision` (K): rounds the infrared CMIP bands (7–16) to the largest power-of-two number of
  packed counts that is at most this many K (0.246 K for band 13 at `0.25`, stored per scan in
  `quantization_step`). At `0.25` this cuts CMIP storage by about 26%; `null` keeps full precision.
  The reflectance bands (1–6) always keep full precision.
- `precision` (per ABI product, in its units): overrides `cmip_precision`, or rounds any other
  product. Categorical products such as cloud top phase must not set it.
- `bounds` (`lat_min`, `lat_max`, `lon_min`, `lon_max`), **ABI only:** crops every ABI product to the
  smallest scan-angle rectangle containing the box; `null` keeps the full CONUS sector. The example
  config uses NOAA NCEI's contiguous-US box (24–50°N, 125–66°W), which keeps about 65% of the pixels.
  GLM is never cropped (about 240 MB per day); the feature step selects what its grid needs.

With the example config a day stores about 1.7 GB (2025-10-27):

| Product | Per day |
|---|---|
| Bands 13 and 15 | 0.37 GB each |
| Bands 2 and 5 | 0.36–0.37 GB each (they shrink to about a tenth at night) |
| ACHA, TPW and the three DSI indices (10 km) | 0.13 GB together |
| GLM | 0.13 GB |

ABI frames and GLM columns are stored with pcodec and the 8-bit quality flags with Blosc zstd, which
stored 18% less than Blosc zstd 3 with bitshuffle everywhere for the same CPU time (2025-07-15).

## Config options for speed

- `download_method` (default `"whole_files"`; the example config uses `"byte_ranges"`). The stored
  data is identical either way.
  - `"whole_files"` downloads every file in full and decodes it in `max_decode_processes` (4)
    processes.
  - `"byte_ranges"` downloads only each file's HDF5 index, the stored variables and, for ABI, the
    bands and image blocks overlapping `bounds`: 4.9 GB per day instead of 16.1 GB with the example
    products. Each file takes several requests in a row, so these run in `byte_range_processes` (48)
    worker processes, each handling one file at a time.
- `max_concurrent_http_requests` (32): open connections shared by all days, for listing and, with
  `"whole_files"`, downloads.
- `http_timeout_seconds` (60): time limit for one request (with `"byte_ranges"`, for each read).
- `http_retries` (5): attempts per request. Timeouts, connection errors and HTTP 408, 429 and 5xx are
  retried with backoff.
- `max_days_in_flight` (2): days processed at once, so the next day's listing overlaps downloads.
- `max_inflight_payloads` (optional): files per day between download start and being written, which
  caps memory. Defaults to `max_concurrent_http_requests`, or `byte_range_processes` with
  `"byte_ranges"`.

Speed is usually limited by the connection. With `"byte_ranges"`, much fewer than ~48 workers leaves
bandwidth unused: GLM files are small, so their fetches are limited by round trips.

## MRMS radar

`download_mrms.py` mirrors MRMS radar mosaics from NOAA's AWS bucket (`s3://noaa-mrms-pds`) into their
own day stores, `<mrms.output_root_dir>/YYYY/YYYYDDD.zarr`, with the same progress display as
`download_raw.py`:

```bash
python download_mrms.py --config config.json
```

It uses the config's dates, `bounds`, `overwrite_existing` and HTTP settings, and its `mrms` section:

- `products`: each a `key` (the store group), a `prefix` (the bucket folder holding one folder per
  day) and a `description`. The example config downloads `CONUS/Reflectivity_-10C_00.50`, the
  reflectivity at the -10 °C isotherm that LightningCast v2 added to its inputs.
- `threads` (8): files decoded at once, each holding a ~50 MB field.
- `bucket` (`s3://noaa-mrms-pds`) and `output_root_dir` (`./mrms_raw`).

Every file is kept (every 2 minutes for the CONUS mosaics) on its 0.01° lat/lon grid, cropped to the
pixels whose cell overlaps `bounds`, with values packed as published (16-bit integers with each file's
scale and offset). MRMS marks its own gaps: in the reflectivity products -999 is outside radar coverage
and -99 is covered with no echo. Days are marked `complete` by the same rule as the GOES stores; the
docstring of `goes_downloader/mrms.py` describes the layout. The files are GRIB2 with the field packed
as a PNG image, decoded by Pillow; the values match eccodes exactly (five files, 2025-04-10 to
2026-09-25).

Measured with 8 threads on a 12-core workstation (2026-10-28), 720 files a day:

| Day | Downloaded | Stored | Time | Peak memory |
|---|---|---|---|---|
| 2026-07-14 | 0.94 GB | 0.70 GB | 24 s (one run) | 2.2 GB |
| 2025-10-27 | 1.08 GB | 0.74 GB | 25–28 s (three runs) | 2.2 GB |

So the configured 571 days take about 4 hours and 400–420 GB. Frames are stored with plain zstd 9,
which stored 10% less than Blosc zstd 3 with byte shuffle.

## Building features

The feature step reads the raw stores and writes one store per UTC day to
`<features.output_root_dir>/YYYY/YYYYDDD.zarr`, holding the configured channels and lightning labels
on one grid every `time_step_minutes`. It is configured by the `features` section of `config.json` and
reuses the download's `raw_output_root_dir`, dates and products. The layout is in the docstring of
`goes_downloader/features/__init__.py`.

```bash
python build_geography.py --config config.json    # once: terrain, land/water, viewing angle, borders
python build_features.py --config config.json     # every day in the range
python build_features.py --config config.json --start 2025-04-10 --end 2025-12-31 --stats
```

- `--start`, `--end`, `--workers` and `--overwrite` override the config for one run.
- Days already complete are skipped, so the command can run again as the download adds days.
- `--stats` builds nothing. It combines the per-day sums over the range into
  `stats_<start>_<end>.json`: mean, std, min, max and valid fraction per channel, and positive and
  known fraction per label (counted on the label's own cells).

A day is `complete` when its raw store and every neighbouring raw day it read are complete: the day
before for lags and gap filling, the day after for labels. Neighbours outside the download date range
don't hold a day back; their slots are just missing, and the days next to them are rebuilt if the
range is later extended.

Ctrl-C stops a build within a second or two; days in progress are rebuilt on the next run. Changing
the `features` config doesn't touch days already complete: pass `--overwrite` or delete the output.

### Grid and time

- `grid.resolution_km` (4): cell size at the sub-satellite point, a multiple of the finest product's
  pixel size (2 km). Cells are blocks of ABI fixed-grid pixels, so a 4 km cell is 2x2 pixels of a 2 km
  band, and a 10 km cell exactly one DSI pixel. Over CONUS a 4 km cell is about 4.4 km east-west by
  5.8 km north-south.
- `grid.bounds`: lat/lon box of the grid. Left out, the download `bounds` are used; `null` uses
  everything stored.
- `time_step_minutes` (15): how often a sample is written. Channels are always computed on the
  5-minute ABI slots, so `lag`, `sum_past` and gap filling work at 5-minute resolution.
- `dtype` (`float16`), `chunk_yx` (`null`: one chunk per time step) and `compression` (Blosc `codec`,
  `level`, `shuffle`): how the frames are stored. Training reads whole frames, which one chunk per
  frame makes fast. LZ4HC 5 (the default) stores 13% less than LZ4 9 and reads a frame of 11 channels
  as fast (2.7 ms against 3.1 ms, 2025-07-15); zstd 3 and pcodec stored 3% less but read in 5.4 and
  9.1 ms.

Time `t` is the start of a 5-minute ABI slot. The **issue time** is `t + 5 min`: every input ends
there, and every label window starts there.

### Channels

Each entry of `features.channels` has a `name` (usable in expressions), an optional `description`,
and `"output": false` for helper channels that other channels read but that aren't stored. It takes
its values from exactly one of these sources:

- **`product`**: an ABI product from the `download` list, in physical units.
  - `aggregation` applies when the product is at least as fine as the grid: `mean`, `min`, `max`,
    `std`, or `fraction` (the share of pixels whose value is in `values`). A coarser product takes
    the pixel under the cell centre.
  - `valid_dqf` (`[0]`) lists the quality flags kept.
  - `min_valid_fraction` (0.5) is the share of valid pixels a cell needs.
  - `fill_gaps_minutes` keeps each cell's last valid value for that long, which suits clear-sky-only
    products such as TPW and DSI.
- **`mrms`**: a product of the config's `mrms` section, in physical units (dBZ for reflectivity). See
  [MRMS channels](#mrms-channels).
  - `aggregation`: `mean`, `min` or `max` over the 0.01° pixels placed in the cell (about 25 on the
    4 km grid).
  - `no_echo` (−20): the value of pixels covered by radar with no echo (MRMS's −99). Pixels outside
    radar coverage (MRMS's −999) are missing.
  - `min_valid_fraction` (0.5): the share of the cell's pixels in radar coverage that it needs.
- **`glm`**: per 5-minute slot and cell, `count` (flash centroids), `extent` (flashes over the cell,
  each weighted by the share of the cell its GLM pixels cover: flash-extent density before rounding;
  see [Flash footprints](#flash-footprints)), `energy_fj` or `area_km2`. The value is missing when GLM
  covered less than `glm.min_coverage` of the slot.
- **`solar`**: `cos_zenith`, `local_hour_sin` or `local_hour_cos` at the middle of the slot, computed
  with NOAA's solar-calculator algorithm.
- **`expr`**: an expression over other channels, with `+ - * / **`, numbers and the functions
  `lag(x, minutes)`, `sum_past`/`mean_past`/`min_past`/`max_past(x, minutes)`, `abs`, `sqrt`,
  `log1p`, `exp`, `minimum`, `maximum`, `clip(x, low, high)` and `mask(x, m)` (x where m > 0, missing
  elsewhere). Any missing input makes the result missing.

For example, the example config's visible channels divide reflectance by cos(solar zenith) and are
missing when the sun is low: `mask(c02 / cos_sza, cos_sza - 0.1)`.

#### MRMS channels

The example config's `refl_m10c_max` is the strongest reflectivity at −10 °C in each cell, LightningCast
v2's radar input (Cintineo et al. 2026, Weather and Forecasting 41).

- **Time.** Each slot takes the newest MRMS file before its issue time, if under 10 minutes old;
  otherwise the slot is missing. MRMS writes a file every 2 minutes, with occasional gaps of 4.
- **No echo.** −20 dBZ sits below every echo (the lowest in 72 files of 2025-07-14 to 16 was −16 dBZ),
  so a `max` is the cell's strongest echo.
- **Missing days.** A feature day is complete only if every MRMS day it read is (`mrms_days` in the
  store), and is rebuilt when a missing one arrives. So download MRMS before building, or days are
  built twice.
- **Placement.** Radar sees the ground under a storm; ABI and GLM see it from the satellite, where a
  storm top 10 km up appears 6–20 km (2–5 cells) further from the sub-satellite point than the radar
  column under it. GLM reports each flash at about the ground position under it, so each MRMS pixel's
  lat/lon goes through the same mapping as GLM's flashes (read on `glm.lightning_ellipsoid_km`): a
  flash and the radar column under it land on the same cell, and so do the labels. With `null` both
  stay on the ground.

The placement was measured on 2025-07-15 and 16 by matching radar cores (≥ 35 dBZ) with flash
footprints of the 15 minutes before and after the radar time, so that storm motion cancels. Each entry
is the offset of the flashes from the cores in cells, rows (+ south) then columns (+ east), on 07-15 /
07-16:

| Radar placed | West (< 105° W) | Central | East (≥ 90° W) |
|---|---|---|---|
| On the ground | −0.86, −1.82 / −0.80, −1.82 | −1.34, −1.36 / −1.24, −1.28 | −0.40, −0.79 / −0.69, −0.38 |
| Like GLM's flashes (used) | −0.39, −0.04 / −0.39, −0.02 | −0.66, +0.26 / −0.61, +0.34 | +0.00, −0.15 / −0.16, +0.02 |
| At the ellipsoid's height above the pixel | +0.92, −0.68 / +0.99, −0.62 | +0.41, −0.17 / +0.55, −0.42 | +1.23, −0.25 / +1.09, −0.09 |

On the ground the flashes sit north-west of the cores, as parallax predicts. The last row, where the
satellite would see a point at the lightning ellipsoid's height above each pixel, differs from GLM's
mapping by 4–8 km because that ellipsoid is flatter than the Earth, and overshoots. On 07-15, 78.9% of
core cells held a flash cell with the mapping used, against 77.5% for the last row and 73.5% on the
ground.

### Labels and GLM

Each entry of `features.labels` is:

- 1 where at least `min_flashes` (default 1) flashes occurred within `radius_cells` (default 0)
  cells, between `start_minutes` (default 0) and `end_minutes` after the issue time
- 0 where none did
- 255 (unknown) where GLM covered less than `glm.min_coverage` (0.95) of the window

With `"source": "extent"` (the default), flashes are counted by the window's flash-extent density,
rounded, in each cell (see [Flash footprints](#flash-footprints)). With `"centroid"`, a flash counts
only in the cell of its centroid. Orphan events belong to no flash and are left out.

Three options exist for matching another definition rather than for training:

- `start_minutes` may be -5, which starts the window at the scan's own slot, overlapping the inputs.
- `fed_minutes` takes the density in windows of that many minutes and counts the largest, instead of
  one density over the whole window.
- `resolution_km` puts the label on finer cells that nest in the grid's (2 km on the 4 km grid). Such
  labels go to `labels_fine`; all of them must share one resolution.

The example config's `lightningcast_60min` uses all three to reproduce LightningCast's truth
(Cintineo et al. 2022): at least one flash in a 1-minute glmtools flash-extent density of the 2 km
pixel, within 60 minutes of the scan. Its window starts at the slot rather than the scan itself, 77 s
earlier in mode 6.

`flash_count` (centroids) and `flash_extent` (footprints) hold the flashes in each time step's window
ending at the issue time, for deriving other labels, persistence baselines and "new lightning" scores
later.

- `glm.quality_flags` (`null` = all) filters flashes. Flag 3 marks large flashes, not false ones.
- `glm.lightning_ellipsoid_km` (14/6) is the ellipsoid GLM geolocates flashes on. Reading their lat/lon
  back on it recovers the direction GLM saw, which lines flashes up with the cloud tops ABI sees.
  `null` uses the ground ellipsoid, which shifts flashes by about 2 km over CONUS.

#### Flash footprints

A GLM event is one lit pixel of GLM's sensor, 8 km across at nadir and 8–15 km over the grid,
reported at the pixel's centre. The cells under those centres form a lattice, since GLM's pixels are
1.3–2 cells of 4 km apart: on 2026-07-15 only 40% of the cells in the busiest 60 x 60 area were touched
all day, and 13% of the cells labelled lightning had an unlit cell boxed in by lit ones.

Footprints are therefore built as glmtools builds flash-extent density, the method behind
LightningCast's truth and NOAA's gridded GLM products: each event's pixel is rebuilt from glmtools'
table of pixel-corner offsets (`goes_downloader/features/glm_pixel_corners.npz`, see the licence file
beside it), events of one flash closer than 28 µrad are one pixel, and the pixel is spread onto the
cells by area. A cell's density over a window is the sum of those shares over the window's flashes,
rounded half up; one flash covers a cell at most once. `fed`-type channels keep the sum before
rounding, so it adds up over any window; `flash_extent` and the labels round.

Checks, on 2026-07-15 unless stated:

- Holes in the 30-minute label fell from 13.2% to 0.04% of lit cells, and the busiest area's daily
  coverage rose from 40% to 97%. The label's base rate rose from 0.0136 to 0.0190.
- The table is GOES-16's, which glmtools uses for every satellite. GOES-19's pixel spacing, measured on
  four days of 2026 after merging each pixel's repeated positions, is 0.996 of the table's along each
  axis, within 0.5% in every ninth of the grid.
- GLM reports one pixel at positions a step or two of NOAA's lat/lon packing apart (5.6 µrad each) as
  the navigation drifts, and glmtools' 28 µrad bins leave some repeats as separate pixels. Merging them
  at 40 µrad instead changed the 60-minute label's base rate from 0.0248 to 0.0247, the LightningCast
  label's not at all, and the footprint area by 1.5%, so the footprints keep glmtools' bins, as
  LightningCast's truth did.
- Footprint area is 1.17 times GLM's own `flash_area`. NOAA's area per pixel barely varies (66–77 km²)
  while the pixels' ground area grows away from nadir from 69 to 114 km², so `flash_area`, and the
  `area_km2` channel built from it, is not a ground area towards the northern edge of the grid.

### Geography

`build_geography.py` writes `geography.zarr` into the features output folder, once: each of
`features.geography.fields` on the feature grid, with statistics for normalisation.
`build_features.py` stops if it was built on a different grid; rebuild it with `--overwrite`.

- Sources are downloaded once into `geography.cache_dir` (`./goes_static`, about 66 MB); OPeNDAP URLs
  are read for the grid's area only.
- A source finer than the cells is aggregated (`mean`, `std`, `min`, `max`); a coarser one is
  interpolated. `scale`, `offset` and `clip` apply to the source values first.
- **Borders fields** (`"borders"`) come from a GeoJSON file of country polygons keyed by `ADM0_A3`.
  The polygons of `countries` (default all) are drawn on a 0.01° raster, where holes such as the Great
  Lakes stay water, and aggregated to cells. `"fraction"` is the share of the cell on that land,
  `"distance_km"` the distance to the nearest cell at least half on it (0 on it). `within` keeps only
  polygons with a vertex in a lat/lon box, which leaves Alaska and Hawaii out of the contiguous US.
  These fields are meant for choosing where to score forecasts, not as model inputs.

The example config defines these fields:

| Field | Source | Notes |
|---|---|---|
| `elevation` | [NOAA ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model), 60 arc-seconds | Cell mean (m), clipped at 0 so the sea is 0. Large lakes show their floor; `land_fraction` tells them apart. |
| `elevation_std` | the same | Terrain roughness (m). |
| `land_fraction` | [NASA IMERG land/sea mask](https://gpm.nasa.gov/data/directory/imerg-land-sea-mask-netcdf), 0.1° | 1 land, 0 water; interpolated. |
| `satellite_zenith` | grid geometry | Degrees between the local vertical and the direction to GOES. |
| `conus_land` | [Natural Earth](https://www.naturalearthdata.com/) 1:10m countries with large lakes removed (public domain) | Share of the cell on US land within 24–50° N, 126–66° W. |
| `conus_distance_km` | the same | km to the nearest cell at least half on that land, 0 on it. |
| `any_land` | the same | Share of the cell on any country's land. |

### Size and speed

With the example config before `refl_m10c_max` was added (11 stored channels, 4 km cells, every 15
minutes, float16, LZ4HC 5), a day stores about 0.32 GB (0.35 GB on 2025-07-15). The MRMS channel adds
9 MB a day.

Building is CPU-bound, but memory sets `max_workers`. A worker peaks while it puts the day's flash
footprints on the 4 km and 2 km grids, at 2.45 GB on 2026-07-04 (the day with the most flashes over the
grid), then holds about 1.7 GB while it builds the frames. On a 6-core (12-thread) machine with 29 GB,
24 days of July 2025 took 6 min 34 s with 8 workers (16 s a day), using at most 15.9 GB between them.
In a paired run of older code, 12 workers were only 5% faster and pushed 3.8 GB of other programs into
swap, so the config uses 8. On its own 2025-07-15 took 51 s, and the MRMS channel adds 6–14 s to a
day. Building next to a running
download is much slower, so use far fewer `--workers` then. The geography store takes about 16 s once
its sources are cached.

## Choosing a cloud

AWS (`s3://noaa-goes19`, in `us-east-1`) and Google Cloud (`gs://gcp-public-data-goes-19`) hold the
same files, so switching `bucket` changes only where they come from; days already complete are not
downloaded again. Byte-range downloads make several requests in a row per file, so round-trip time
matters as much as bandwidth. In one test from a home connection, a day took 2 min 53 s from AWS and
4 min 01 s from Google Cloud (8-core Mac, 24 workers), but which is faster depends on where you run it.
A machine in AWS `us-east-1` is next to the data.

## Tests

```bash
pip install -e ".[dev]" -c requirements.lock
pytest              # offline, about 10 s: small synthetic files and stores stand in for the GOES ones
pytest --network    # also real files and a full test hour both ways (~0.5 GB, ~1 min)
```

The network tests read from the configured `bucket` and check that both clouds list the same files
for the test hour. GitHub Actions runs pyflakes and the offline tests on Python 3.12, 3.13 and 3.14.

## Troubleshooting

- **"requires a different Python" when installing:** use Python 3.12, 3.13 or 3.14.
- **`CERTIFICATE_VERIFY_FAILED` (every listing fails):** the python.org installer for macOS comes
  without CA certificates. Run its `Install Certificates.command` once, or start the commands with
  `SSL_CERT_FILE=/etc/ssl/cert.pem`.
- **A day stays incomplete:** run the same command again; failed files are retried. The `!!` log
  lines and `report.py` show which days and files failed. A file that fails on every run points
  to a problem with that file at the source.
- **"too recent" in the report:** the day ended less than 24 hours before the run; run again later.
- **Running out of memory:** lower `byte_range_processes` (or `max_workers` for the feature step).
- **Slower than expected:** check the connection's speed first. With `"byte_ranges"`, much fewer
  than ~48 workers leaves bandwidth unused.
