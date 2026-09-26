# goes_downloader

Downloads GOES-19 ABI (CONUS L2) and GLM (LCFA) data from the public GCS bucket
`gcp-public-data-goes-19` into one raw Zarr store per UTC day, for building a
lightning-prediction training dataset. No time binning or spatial aggregation happens
here; that is the job of `build_features.py`, which turns the raw stores into CNN inputs and
lightning labels on one grid (see [Building features](#building-features)).

For each UTC day it:

1. Lists ABI and GLM files over the GCS JSON API, retrying transient errors.
2. ABI: stores every 5-min scan on the product's native grid (CMIP bands 8, 11, 13 and 15 at
   2 km; ACHA cloud height, TPW and the DSI stability indices at 10 km) as the file's packed
   integers plus its DQF flags.
3. GLM: stores every flash (time, lat/lon, area, energy, quality flag) plus a coverage flag for
   every 20-second window, so outages (including files NOAA never published) show up as gaps
   rather than as "no lightning".
4. Writes `goes_raw/YYYY/YYYYDDD.zarr` and marks it `complete` only if every listing and file
   succeeded and the day ended at least 24 hours ago, since NOAA publishes some files late.
   Incomplete days are rebuilt on the next run.

The full layout is described in the docstring at the top of `download_raw.py`.

## Usage

```bash
pip install -r requirements.txt
python download_raw.py --config config.json
```

Days already marked `complete` are skipped unless `overwrite_existing` is `true`.

While it runs, the last line shows a progress bar, days done, data downloaded, data saved to
disk (added as each day finishes), download speed (last 30 s) and an ETA. The ETA uses each
day's exact size once the day is listed and the average listed day for days not reached yet.
When output is redirected to a file, a progress line is printed once a minute instead.

## Choosing products

`abi_products` and `glm_product` in `config.json` define every available product (all 16 CMIP
bands plus the CONUS L2 products), each with a `description`. Only the keys
listed in `download` are fetched, for example:

```json
"download": ["cmip_c08", "cmip_c11", "cmip_c13", "cmip_c15", "acha", "tpw", "actp", "glm"]
```

Products that read different variables from the same files, like the five `dsi_*` stability
indices (CAPE, lifted index, K-index, Showalter index, total totals), share one download per
scan. Leave out `"glm"` to skip lightning. Changing the list does not touch days already marked
complete; delete them or set `overwrite_existing` to rebuild them with the new products.

## Reading the data

Each day store opens directly with xarray, one product (or GLM table) at a time, and several days
combine into one continuous time series:

```python
import glob, xarray as xr

days = sorted(glob.glob("goes_raw/*/*.zarr"))
c13 = xr.open_mfdataset(days, engine="zarr", group="abi/cmip_c13", combine="by_coords")
flashes = xr.open_mfdataset(days, engine="zarr", group="glm/flashes", combine="nested", concat_dim="flash")
coverage = xr.open_mfdataset(days, engine="zarr", group="glm/coverage", combine="by_coords")
```

`coverage.covered` has one entry per 20-second window: 0 means there is no GLM data for that
window, so no flashes there does not mean no lightning. NOAA misses a few GLM files on most days,
and on 2026-07-15 published no GLM data from 20:22:40 UTC to midnight. A flash's `file_index` is
its row in the same day's `glm/files`, so it no longer identifies a file once several days are
combined; use the flash times instead.

ABI values are the file's packed numbers, not physical units: the physical value is
`value * scale_factor + add_offset` (both stored per time slot). xarray turns fill values into NaN,
which makes `value` a float array; pass `mask_and_scale=False` to get the stored integers.

## Assumptions about the source data

**Checked on every scan.** A scan that breaks one of these counts as a file error, which leaves the
day incomplete so it is retried and visible in the log:

- Within a day, each product's grid (shape and x/y scan angles), fill value and quality-flag data
  type stay the same as in its first scan.
- Data values are packed integers that fit in 16 bits; quality flags fit in 8 or 16 bits.

**Recorded, but expected to stay constant.** `scale_factor` and `add_offset` are stored for every
scan, so a change would be captured. For GOES-19 they have never changed: one file per month for
every current product, from the start of the data (Oct 2024 for CMIP, Dec 2024 for the others) to
Sep 2026, all had the same scale, offset and fill value. The feature step can treat them as
constant per product but should check.

**Assumed, not checked:**

- ABI stays in scan mode 6 (CONUS every 5 minutes). Every product's filename filter contains
  `-M6`, so scans taken in other modes are not downloaded. Each 5-minute slot holds at most one
  scan; an extra scan in the same slot is skipped and counted in `n_abi_slot_collisions`.
- The bucket stores files under `<product>/<year>/<day-of-year>/<hour>/` by scan start time, and
  file names carry `_s`, `_e` and `_c` (start, end, creation) times. When NOAA republishes a scan,
  the file with the newest creation time is used.
- All ABI products share one projection (the GOES-East fixed grid, centred on 75.0°W) since
  GOES-19 moved to the GOES-East position in 2025. Earlier files (e.g. January 2025) use its
  89.5°W test position, so don't mix those dates with later ones. The `bounds` crop is computed
  from the first ABI scan decoded each day and applied to every ABI product.
- GLM flash times are "seconds since" offsets in the file. A GLM file belongs to the day its start
  time falls in, so a flash in the first file of a day can begin a second or two before midnight;
  select flashes by their own times, not by which day store holds them.
- NOAA publishes a day's files within 24 hours of the day ending; later files are missed once
  the day is marked complete.
- In a complete day, ABI `present` = 0 and GLM `covered` = 0 mean NOAA did not publish that scan
  or file; a failed download would have left the day incomplete.

## Config options for storage

- `cmip_precision` (K): rounds the infrared CMIP bands (7–16) to this step to save space. 0.25 K is
  below the sensor noise and cuts CMIP storage by about 26%; `null` keeps full precision. The
  visible bands (1–6) are reflectance and always keep full precision.
- `precision` (per ABI product, in the product's units): overrides `cmip_precision` for that
  product, or rounds any other product. Categorical products such as cloud top phase must not set it.
- `bounds` (`lat_min`, `lat_max`, `lon_min`, `lon_max`): **applies to ABI only.** Crops every ABI
  product to the smallest scan-angle rectangle containing the box; `null` keeps the full CONUS
  sector. GLM is never cropped: every flash in GLM's field of view is kept (about 36 MB/day), and
  the feature step selects the flashes its grid needs.
  The config uses 24–50°N, 125–66°W, the contiguous-US bounding box NOAA NCEI publishes for its
  nClimGrid-Daily dataset.

Measured on one hour of data with the current products: about 3.3 GB/day for full CONUS.
That box keeps about 65% of the ABI pixels.

## Config options for speed

- `max_concurrent_http_requests` (32): open connections shared by all days, for listing and
  downloads.
- `http_timeout_seconds` (60): time limit for one request, including reading the whole file.
- `http_retries` (5): attempts per request. Timeouts, connection errors and HTTP 408, 429 and 5xx
  are retried with backoff; other errors fail at once.
- `max_decode_processes` (4): worker processes that decode the NetCDF files.
- `max_days_in_flight` (2): days processed at the same time, so the next day's listing overlaps the
  current day's downloads.
- `max_inflight_payloads` (optional, defaults to `max_concurrent_http_requests`): files per day
  that can be between download start and being written to Zarr, which caps memory use.

Download speed is usually limited by the internet connection, not by these settings. On a
connection that delivers about 60 MB/s, 64 connections with 8 decode processes was no faster than
the defaults, 128 connections was slower, and decoding used only about 1.3 CPU cores.

## Building features

`build_features.py` reads the raw day stores and writes one feature store per UTC day to
`goes_features/YYYY/YYYYDDD.zarr`: the configured channels and lightning labels on one grid, every
`time_step_minutes`. It is configured by the `features` section of `config.json` and reuses the raw
settings (`raw_output_root_dir`, dates, products). The full layout is in the docstring at the top of
`build_features.py`.

```bash
python build_features.py --config config.json                   # every day in the range
python build_features.py --config config.json --start 2025-10-16 --end 2026-05-31 --stats
```

`--start`, `--end`, `--workers` and `--overwrite` override the config for one run. `--stats` builds
nothing: it combines the per-day sums of the days in the range into
`goes_features/stats_<start>_<end>.json` (mean, std, min, max and valid fraction per channel, positive
fraction per label). Run it over the training days only, so normalisation doesn't see the test period.

A day is `complete` when its raw store and every neighbouring raw day it read are complete: the day
before for lags and gap filling, the day after for labels. Neighbours outside the download date range
never arrive, so they don't hold a day back; their slots are just missing. Incomplete days are rebuilt
on the next run. As with the downloader, changing the `features` config doesn't touch days already
complete: pass `--overwrite` (or delete `goes_features/`) to rebuild them.

### Grid and time

- `grid.resolution_km` (4): cell size at the sub-satellite point, a multiple of the finest product's
  (2 km for the CMIP bands, 10 km if only 10 km products are used). Cells are blocks of ABI fixed-grid
  pixels, so a 4 km cell is 2x2 CMIP pixels and a 10 km cell is exactly one DSI pixel. Over CONUS, cells
  are larger than nominal because the satellite sees the US at an angle: a 4 km cell is about 4.4 km
  east-west by 5.8 km north-south in the middle, up to 7–9 km at the northern edge.
- `grid.bounds`: lat/lon box of the grid. Left out, it is the download `bounds`; `null` keeps
  everything the raw stores hold.
- `grid.reference_product`: the product whose grid the cells are built from; default the finest one used.
- `time_step_minutes` (15): how often a sample is written (a multiple of 5 that divides a day). Channels
  are always computed on the 5-minute ABI slots, so `lag`, `sum_past` and gap filling work at 5 minutes
  whatever the step.

Time `t` is the start of a 5-minute ABI slot. The **issue time** is `t + 5 min`: every input ends there,
and every label window starts there.

### Channels

Each entry of `features.channels` has a `name` (usable in expressions), an optional `description`, and
`"output": false` for helper channels that other channels read but that aren't stored. It takes its
values from exactly one of:

- **`product`**: an ABI product from the `download` list, converted to physical units.
  - `aggregation` (`mean`): how source pixels become a cell when the product is at least as fine as
    the grid: `mean`, `min`, `max`, `std`, or `fraction` of pixels whose value is in `values` (for
    categorical products, e.g. `"product": "actp", "aggregation": "fraction", "values": [4]` for the ice
    fraction). A product coarser than the grid (10 km DSI on a 4 km grid) takes the pixel containing the
    cell centre instead.
  - `valid_dqf` (`[0]`): quality flag values kept; other pixels count as missing. `null` ignores the DQF.
  - `min_valid_fraction` (0.5): a cell is missing unless this fraction of its pixels is valid.
  - `fill_gaps_minutes` (0): each cell keeps its last valid value for up to this long. Useful for
    clear-sky-only products (TPW, DSI), which disappear under the clouds that matter most.
- **`glm`**: GLM flashes starting in each 5-minute slot, per cell: `count`, `energy_fj` (total optical
  energy, femtojoules) or `area_km2` (total flash area). Missing when GLM covered less than
  `glm.min_coverage` of the slot.
- **`expr`**: an expression over other channels with `+ - * / **`, numbers and these functions:
  `lag(x, minutes)` (the value `minutes` earlier), `sum_past`, `mean_past`, `min_past`, `max_past(x, minutes)`
  (over the slots of the last `minutes`, the current one included), `abs`, `sqrt`, `log1p`, `exp`,
  `minimum(a, b)`, `maximum(a, b)` and `clip(x, low, high)`. Any missing input makes the result missing.

The default channels are the four IR bands, the coldest pixel of band 13 in each cell, the 6.2 − 10.3 µm
difference, the 15-minute change of band 13, cloud-top height, TPW, CAPE, lifted index, K-index (gap-filled
up to 60 minutes) and the flashes of the last 15 minutes.

Other options: `dtype` (`float16`, or `float32`; float16 keeps about 3 significant digits and overflows
above 65504, so sums such as `energy_fj` should go through `log1p` or use `float32`; values that overflow
are stored as NaN and counted in the store's `float16_overflow` attribute), `chunk_yx` (`[256, 256]`,
spatial chunk size: smaller suits random patches, larger suits whole frames), `max_workers` (4, days
built in parallel, up to about 1.3 GB of memory each).

### Labels

Each entry of `features.labels` is 1 where at least `min_flashes` (1) flashes started within
`radius_cells` (0) cells between `start_minutes` (0) and `end_minutes` after the issue time. The radius is
in cells, so it scales with `resolution_km`; the configured labels use 2 cells, about 10 km on the 4 km
grid, roughly GLM's location accuracy plus its pixel size. A label is 255 (unknown) where GLM covered less
than `glm.min_coverage` (0.95) of the window, so outages never become "no lightning". Lightning is rare:
in the October test hours 0.5% of known cells were positive for 30 minutes, so weight the loss accordingly.

`flash_count` holds the flashes of each `time_step_minutes` window ending at the issue time, so other
labels (longer windows, other thresholds) can be derived later without rebuilding.

`glm.quality_flags` (`null` = all) filters flashes by quality flag. About 8% of flashes are flagged 3
(more events than a threshold): they are large flashes, not false detections, so they are kept by
default. `glm.lightning_ellipsoid_km` (14/6) places flashes where ABI sees their cloud tops: GLM
geolocates each flash on this ellipsoid (the Earth raised 14 km at the equator, 6 km at the poles; NOAA's
operational value since 2018), so reading the lat/lon back on it recovers the direction GLM saw, the same
direction ABI, on the same satellite, views the cloud top from. `null` reads them on the ground ellipsoid
instead, which moves flashes about 2 km (up to 4 km) over CONUS; the shift is small because the lightning
ellipsoid is flatter, which moves a given latitude south and cancels most of the height's parallax.

### Reading features for training

`xarray.open_mfdataset` combines days into one time axis (see [Reading the data](#reading-the-data)).
Features are float16 on disk: convert to float32 before any arithmetic, since float16 sums overflow.
A PyTorch dataset that stacks the channels at `t`, `t - 15` and `t - 30` minutes and returns one label as
the target:

```python
import glob, json
import numpy as np
import torch
import xarray as xr

stats = json.load(open("goes_features/stats_2025-10-16_2026-05-31.json"))["channels"]


class LightningDataset(torch.utils.data.Dataset):
    """Channels at t, t-15 and t-30 min stacked as the input, one label at t as the target, random patches."""

    def __init__(self, days, label="lightning_next_60min", history=(0, 1, 2), patch=256, min_valid=0.9):
        ds = xr.open_mfdataset(days, engine="zarr", combine="by_coords", mask_and_scale=False)
        self.x, self.y, self.patch = ds.features, ds.labels.sel(label=label), patch
        names = list(ds.channel.values)
        self.mean = np.array([stats[c]["mean"] for c in names], np.float32)[:, None, None]
        self.std = np.array([stats[c]["std"] for c in names], np.float32)[:, None, None]
        # Samples whose history steps all exist (a missing day leaves a gap in time) and have their ABI scan;
        # c13 is NaN only where the scan is missing, unlike cloud height or the clear-sky products
        step = np.timedelta64(int(ds.attrs["time_step_minutes"]), "m")
        times = ds.time.values
        index = {t: i for i, t in enumerate(times)}
        ok = ds.channel_valid_fraction.sel(channel="c13").values >= min_valid
        self.samples = [
            [index[t - h * step] for h in history]
            for t in times
            if all(t - h * step in index and ok[index[t - h * step]] for h in history)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        steps = self.samples[i]
        y0 = np.random.randint(0, self.x.sizes["y"] - self.patch + 1)
        x0 = np.random.randint(0, self.x.sizes["x"] - self.patch + 1)
        win = dict(y=slice(y0, y0 + self.patch), x=slice(x0, x0 + self.patch))
        # float16 on disk: convert before any arithmetic
        frames = [(self.x.isel(time=t, **win).values.astype(np.float32) - self.mean) / self.std for t in steps]
        x = np.nan_to_num(np.concatenate(frames), nan=0.0)  # missing -> the channel mean
        y = self.y.isel(time=steps[0], **win).values  # uint8: 1 lightning, 0 none, 255 unknown (mask in the loss)
        return torch.from_numpy(x), torch.from_numpy(y)


train = LightningDataset(sorted(glob.glob("goes_features/2025/*.zarr")))
```

Reading patch by patch through xarray is simple but slow; for long training runs, load whole days
into memory or read the Zarr arrays directly (`zarr.open_consolidated(day)["features"][t, :, y0:y1, x0:x1]`).

### Size and speed

With the default channels (13 channels, 4 km, every 15 minutes, float16) a day takes about 0.5 GB,
about 160 GB for the configured date range. Size grows with channels, samples per day and cells:
5-minute steps triple it, 8 km cells divide it by four. A day builds in about 35 seconds per worker, so
4 workers get through the range in about an hour. Both numbers are extrapolated from 5 hours of one day.
