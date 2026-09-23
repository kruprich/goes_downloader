# goes_downloader

Downloads GOES-19 ABI (CONUS L2) and GLM (LCFA) data from the public GCS bucket
`gcp-public-data-goes-19` into one raw Zarr store per UTC day, for building a
lightning-prediction training dataset. No time binning or spatial aggregation happens
here; that belongs in a separate feature-building step.

For each UTC day it:

1. Lists ABI and GLM files over the GCS JSON API, retrying transient errors.
2. ABI: stores every 5-min scan on the product's native grid (CMIP bands 8, 11, 13 and 15
   and cloud top phase at 2 km, ACHA cloud height and TPW at 10 km) as the file's packed
   integers plus its DQF flags.
3. GLM: stores every flash (time, lat/lon, area, energy, quality flag) and which
   20 s files were read, so missing files show up as coverage gaps.
4. Writes `goes_raw/YYYY/YYYYDDD.zarr` and marks it `complete` only if every listing and
   file succeeded. Incomplete days are rebuilt on the next run.

The full layout is described in the docstring at the top of `download_raw.py`.

## Usage

```bash
pip install -r requirements.txt
python download_raw.py --config config.json
```

Days already marked `complete` are skipped unless `overwrite_existing` is `true`.

## Config options for storage

- `precision` (per ABI product, in the product's units): rounds stored values to save space.
  The CMIP bands are set to 0.25 K, which is below the sensor noise and cuts CMIP storage by
  about 26%. Categorical products such as cloud top phase must not set it.
- `bounds` (`lat_min`, `lat_max`, `lon_min`, `lon_max`): crops ABI to the smallest scan-angle
  rectangle containing the box and keeps GLM flashes inside it. `null` keeps the full CONUS sector.

Measured on one hour of data with the current products: about 3.1 GB/day for full CONUS.
A lower-48 box (24–50°N, 125–66°W) keeps about 65% of the ABI pixels.
