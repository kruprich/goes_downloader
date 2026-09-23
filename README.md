# goes_downloader

Builds a per-day Zarr dataset from GOES-19 ABI (CONUS L2) and GLM (LCFA) data in the
public GCS bucket `gcp-public-data-goes-19`.

For each UTC day it:

1. Lists ABI and GLM files over the GCS JSON API and assigns each file to a time bin
   (strict: the file's start and end must both fall within the bin).
2. Downloads and decodes files in a bounded async + process-pool pipeline.
3. ABI: averages DQF==0 pixels across the files in each bin, then block-averages
   onto a coarser grid.
4. GLM: projects good-quality flashes onto the ABI fixed grid and counts them per cell.
5. Writes `goes_abi_glm/merged/YYYY/YYYYDDD.zarr` with values, valid-pixel fractions
   and file/pixel accounting under `validation/`.

## Usage

```bash
pip install -r requirements.txt
python generate_dataset.py --config config.json
```

Days already marked `complete` are skipped unless `overwrite_existing` is `true`.

## Known issue

ACHA and TPW are 10 km products, while CMIP band 13 is 2 km. The grid comes from the
first ABI product found, so the 10 km products are padded into the corner of the 2 km
grid instead of being resampled. Fix planned: give each product a `resolution_km` in the
config and aggregate all products to the coarsest resolution.
