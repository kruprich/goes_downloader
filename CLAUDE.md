# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-script downloader (`download_raw.py`, driven by `config.json`) that mirrors GOES-19 ABI
CONUS L2 products and GLM lightning flashes from the public bucket `gcp-public-data-goes-19` into
one raw Zarr store per UTC day. It is the raw layer for a lightning-prediction training dataset: it
deliberately does **no** time binning, regridding or quality filtering. That belongs in a separate
feature-building step that does not exist yet. The docstring at the top of `download_raw.py` is the
authoritative description of the store layout; README.md covers usage, reading and assumptions.

## Commands

```bash
pip install -r requirements.txt            # Python 3.10 (pyenv); zarr is pinned <3 (v2 API)
python download_raw.py --config config.json
```

There are no tests or linters. Verify changes by running a short real download. `LIST_HOURS`
limits listing to chosen hours, and the harness must be a script file: the decode
`ProcessPoolExecutor` uses spawn on macOS, so code fed through stdin (`python3 - <<EOF`) fails in
every worker.

```python
# run_test.py: one hour into a scratch dir (copy config.json, override dates/output/overwrite)
import sys, asyncio
import download_raw as dr
if __name__ == "__main__":
    dr.LIST_HOURS = [18]
    asyncio.run(dr.run_all_days(dr.load_config(sys.argv[1])))
```

The usual reference hour is 2025-10-27 18 UTC (~5 s, ~216 files). Check results against the
original NetCDF files and/or the previous version's output array by array. Days that ended less
than 24 h ago are never marked complete, so test with past dates.

## Architecture

**Config (`config.json`).** `abi_products` and `glm_product` define every available product (all
16 CMIP bands plus single-variable CONUS L2 products, each with a `description`). The `download`
list selects which to fetch. `cmip_precision` applies to infrared CMIP bands 7–16 (detected from
the band number in `filename_must_contain`); a product's own `precision` overrides it. `bounds`
crops ABI only.

**Per-day flow (`process_single_day`).**
1. List each distinct `gcs_product_prefix` once (24 hourly prefixes). `select_files` filters by
   `filename_must_contain` and keeps the newest creation time when NOAA republishes a scan.
2. Group ABI products by source file (`(gcs_product_prefix, filename_must_contain)`), so products
   reading different variables from one file (e.g. the `dsi_*` indices) share one download and one
   decode. Each file goes to a 5-minute slot (288/day) by scan start.
3. Download (`http_get`, retries on transient errors) → decode in `DecodePool` (returns packed
   integers + DQF) → validate every product of the file (`accept_first_or_validate`: grid, fill,
   DQF dtype must match the day's first scan) before writing any → write one Zarr chunk per slot
   from a thread.
4. GLM flashes accumulate in memory and are written at the end with per-file rows and a
   `glm/coverage` flag per 20-second window.
5. Build in `<day>.zarr.tmp`, consolidate metadata, then move into place. `complete` is true only
   if no listing/file errors occurred and the day ended ≥24 h ago; incomplete days are rebuilt
   from scratch on the next run.

**Invariants that are easy to break:**
- Arrays written in full must use `_write_array` (`fill_value=None` + `_ARRAY_DIMENSIONS`).
  zarr's default fill of 0 makes xarray read every legitimate 0 as missing.
- Frame arrays (`value`, `dqf`) keep a real fill value because unwritten slots must read as
  missing; products whose files lack `_FillValue` use 65535.
- Values stay packed (`physical = value * scale_factor + add_offset`, stored per slot). The user
  decided against storing physical values or CF packing attributes.
- The `bounds` rectangle comes from the first ABI scan decoded each day (all ABI products share
  the GOES-East fixed grid). GLM is deliberately never cropped (~36 MB/day).
- `DecodePool` replaces the process pool when a worker dies, and closes the dead pool's call-queue
  reader to avoid CPython gh-94777, a hang at exit on Python < 3.12.
- Missing data is explicit: ABI `present` = 0 and GLM `covered` = 0. In a complete day these mean
  NOAA never published the file.

## Conventions

- Output stores are regenerated, not migrated. Don't add format-version checks or migration code
  for old stores.
- Keep README.md (usage, reading, assumptions about the source data, storage options) and the
  module docstring (store layout) in sync with behaviour changes. Assumptions that only matter for
  interpreting the data belong in the future feature step, not the README.
- Output goes to `raw_output_root_dir` (default `./goes_raw`, git-ignored). Current config: about
  2.1 GB/day stored with the NOAA CONUS bounds (~700 GB over the date range) and 6–8 GB/day
  downloaded, since whole files are fetched regardless of bounds.
