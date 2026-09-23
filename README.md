# goes_downloader

Downloads GOES-19 ABI (CONUS L2) and GLM (LCFA) data from the public GCS bucket
`gcp-public-data-goes-19` into one raw Zarr store per UTC day, for building a
lightning-prediction training dataset. No time binning or spatial aggregation happens
here; that belongs in a separate feature-building step.

For each UTC day it:

1. Lists ABI and GLM files over the GCS JSON API, retrying transient errors.
2. ABI: stores every 5-min scan on the product's native grid (CMIP bands 8, 11, 13 and 15
   and cloud top phase at 2 km; ACHA cloud height, TPW and the DSI stability indices at
   10 km) as the file's packed integers plus its DQF flags.
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
```

ABI values are the file's packed numbers, not physical units: the physical value is
`value * scale_factor + add_offset` (both stored per time slot). xarray turns fill values into NaN,
which makes `value` a float array; pass `mask_and_scale=False` to get the stored integers.

## Config options for storage

- `precision` (per ABI product, in the product's units): rounds stored values to save space.
  The CMIP bands are set to 0.25 K, which is below the sensor noise and cuts CMIP storage by
  about 26%. Categorical products such as cloud top phase must not set it.
- `bounds` (`lat_min`, `lat_max`, `lon_min`, `lon_max`): crops ABI to the smallest scan-angle
  rectangle containing the box and keeps GLM flashes inside it. `null` keeps the full CONUS sector.

Measured on one hour of data with the current products: about 3.3 GB/day for full CONUS.
A lower-48 box (24–50°N, 125–66°W) keeps about 65% of the ABI pixels.
