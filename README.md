# goes_downloader

Downloads GOES-19 ABI (CONUS L2) and GLM (LCFA) data from the public GCS bucket
`gcp-public-data-goes-19` into one raw Zarr store per UTC day, for building a
lightning-prediction training dataset. No time binning or spatial aggregation happens
here; that belongs in a separate feature-building step.

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
