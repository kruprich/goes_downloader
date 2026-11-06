# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`download_raw.py` (→ the `goes_downloader` package, driven by `config.json`) mirrors GOES-19 ABI CONUS
L2 products and GLM flashes and events from NOAA's public bucket on AWS (`s3://noaa-goes19`) or Google
Cloud (`gs://gcp-public-data-goes-19`; same files) into one raw Zarr store per UTC day.
`download_mrms.py` (`goes_downloader/mrms.py`, the config's `mrms` section) does the same for MRMS
radar from `s3://noaa-mrms-pds`. The raw layer deliberately does no time binning, regridding or quality
filtering: that is the feature step (`goes_downloader/features/`, run by `build_geography.py` once and
`build_features.py`), which writes CNN inputs and lightning labels on one grid, one store per day, for
the separate `goes_lightning_model` repository. `report.py` summarizes the raw stores.

The store layouts are authoritative in the docstrings of `store.py` (raw), `mrms.py` (MRMS),
`features/__init__.py` (features) and `features/geography.py`. README.md covers setup, usage,
source-data assumptions and measured sizes and speeds.

## Commands

Shells here don't activate the venv, so call its Python directly (Python 3.12–3.14). On the Mac
checkout the venv uses python.org's 3.12, which has no CA certificates: prefix anything that touches
the network with `SSL_CERT_FILE=/etc/ssl/cert.pem`.

```bash
.venv/bin/python -m pip install -e ".[dev]" -c requirements.lock   # without [dev] to only run it
.venv/bin/python download_raw.py --config config.json     # a full run; prints a data report at the end
.venv/bin/python download_mrms.py --config config.json    # MRMS radar day stores (~25 s a day)
.venv/bin/python report.py --config config.json           # which days are complete / need attention
.venv/bin/python build_geography.py --config config.json  # once: static fields on the feature grid
.venv/bin/python build_features.py --config config.json   # feature stores; --start/--end/--workers/--stats
.venv/bin/python -m pytest                                # offline tests, about 10 s
.venv/bin/python -m pytest --network                      # + real files and the test hour (~0.5 GB, ~30 s)
.venv/bin/python -m pyflakes goes_downloader tests download_raw.py download_mrms.py report.py build_features.py build_geography.py
```

`requirements.lock` pins the tested runtime versions and is used as constraints. When a dependency
changes, regenerate it from the installed versions and run the tests on 3.12 and 3.13 (CI runs
3.12–3.14). There are deliberately no console-script entry points: their generated scripts import the
package at top level, and spawned workers re-run the main script.

## Testing and verifying changes

- Before editing, check that nothing runs from this checkout (`pgrep -af 'download_|build_features'`):
  a replacement worker imports the code as it is then.
- Offline tests build small synthetic NetCDF, GRIB2 and Zarr data (`conftest.py`). The fake remote
  starts its buffers as random bytes (`junk_fill`), so decoding anything not fetched fails.
  `test_features.py` writes its raw and MRMS days once per module (`raw_days`, `mrms_days`) and links
  them into each test: a test that adds days must copy them first.
- Network tests (`--network`) decode real files both ways, run the reference hour through the pipeline
  with both `download_method` values (stores must be identical), compare listings on both clouds and
  download an MRMS hour. `decode_mrms` was checked once against eccodes, which isn't a dependency:
  repeat that in a scratch venv if it changes.
- Fetching or decoding changes: compare stores with the previous version's array by array, with both
  `download_method` values; they must be identical.
- Feature-step changes: also build a few real days into a scratch `output_root_dir` (2–3 `--workers`
  while a download runs; skip raw days with a `.tmp` beside them) and compare with values recomputed
  from the raw arrays: block means of 2 km frames, the 10 km pixel under each cell centre, footprints
  from `glm/events` mapped with NOAA's formula instead of pyproj (skipping `flash_index` -1), and MRMS
  cell maxima with pixels placed on the lightning ellipsoid the same way. Check MRMS placement by
  matching radar cores with flash footprints from before and after the radar time (README, "MRMS
  channels").
- Speed or memory changes: time a full day with the real command, sample the process group's RSS from
  `/proc`, and update README's figures saying what was measured.

A short real download needs a harness *file*: the pool uses spawn and workers re-run the main script,
so code fed through stdin fails in every worker. The package is installed in `.venv`, so the file can
live anywhere.

```python
# run_test.py: one hour into a scratch dir (copy config.json, override dates/output/overwrite)
import sys, asyncio
from goes_downloader import config, remote, runner
if __name__ == "__main__":
    remote.LIST_HOURS = [18]
    asyncio.run(runner.run_all_days(config.load_config(sys.argv[1])))
```

The usual reference hour is 2025-10-27 18 UTC (~5 s, 228 files); `mrms` honours `LIST_HOURS` too. Use
past dates (days that ended less than 24 h ago are never complete) and scratch output, never
`goes_raw2/` or `goes_raw/`: a store built from a few hours is still marked complete, so the real run
would skip that day.

## Architecture

`goes_downloader/`: `config` (dataclasses, `load_config`), `remote` (names, object URLs, S3 XML and GCS
JSON listing, `select_files`, `http_get`; the only module that knows the clouds apart), `decode` (NetCDF
→ packed ABI arrays and GLM tables; `DecodePool`), `crop`, `ranges` (byte-range fetching in the pool
workers, `_RangeFile`), `store` (layout, Zarr writing), `day` (`process_single_day`), `runner`, `report`,
`progress`, `common`, and `mrms` (the MRMS downloader on its own, decoding in threads without a process
pool).

**Per day** (`process_single_day`): each distinct `product_prefix` is listed once; `select_files` keeps
the newest creation time when NOAA republishes a scan. ABI products are grouped by source file, so the
`mcmip_*` bands or the `dsi_*` indices share one download and decode, and each file goes to a 5-minute
slot by scan start. Download → decode in `DecodePool` → validate every product of the file against the
day's first scan (`accept_first_or_validate`, `check_window`) before writing any → one chunk per slot.
With `"byte_ranges"` the workers fetch too: h5py reads the HDF5 metadata through `_RangeFile`, only
decoded chunks are fetched, and the unchanged decoders read the file-sized buffer. GLM tables stay in
memory (up to ~1.9 GB) until the end of the day. Days are built in `<day>.zarr.tmp` and moved into
place, `complete` only if nothing failed and the day ended ≥24 h ago; incomplete days are rebuilt from
scratch.

**Config**: `abi_products` and `glm_product` define every product; `download` picks which to fetch.
`mcmip_*` products are only practical with `"byte_ranges"` (~59 MB per whole MCMIPC file).
`load_config` logs keys it doesn't read (`_`-prefixed keys are notes), so a new setting must be added
to `TOP_LEVEL_KEYS` (or the product/bounds key sets) as well as read.

**Feature step** (`features/`): `settings` (unknown keys are errors), `expressions` (derived channels,
the `Plan`), `grid`, `raw` (`RawTimeline`), `mrms` (`MrmsTimeline`), `glm` (`GlmTimeline`), `geography`,
`build` (`build_day` in spawn workers, `run_build`, `run_stats`). `build_day` goes slot by slot from
`lookback` slots before the day; only `Plan.every_slot` channels are computed at every slot, the rest at
output slots. It publishes by renaming the old store to `<day>.zarr.old` first, so a killed worker can't
leave a store that reads as complete without its arrays. `run_build` rebuilds a complete day when a raw
or MRMS day it read as missing or incomplete has completed (`raw_neighbour_arrived`); on Ctrl-C it
terminates the workers, which exit at once (`_exit_now`), and always shuts the pool down.

## Invariants that are easy to break

- Pool workers import only `ranges`/`decode` and what they need, not aiohttp or zarr (~146 MB each):
  entry scripts import nothing outside their main guard, and `remote` imports aiohttp inside
  `http_get` (`test_entry_script_imports_nothing_outside_its_main_guard`).
- Every store is Zarr format 3, and exists if it has a `zarr.json` (format-2 stores count as missing).
  Every array names its dimensions and passes `chunk_key_encoding=CHUNK_KEYS` (flat `c.<slot>.0.0`
  files; zarr's default nested directories cost ~28 GB over the range). Arrays written in full use
  `_write_array`; consolidate with `store.consolidate`. zarr 3 refuses numpy scalars in attributes
  (except `np.float64`), and a failed write leaves the group unable to write attributes: convert with
  `int()`/`float()` first.
- xarray masks only a `_FillValue` attribute, not a Zarr v3 fill value. Arrays with a real fill (ABI
  `value`/`dqf`, 65535 when the file has none; MRMS `value`; `labels`, `labels_fine`) repeat it as
  `_FillValue`, so unwritten slots read as missing. Arrays written in full with fill 0 (`flash_count`,
  `flash_extent`) must not have one; NaN fills need none.
- Codecs: raw ABI frames and GLM columns use `frame_codecs(dtype)` (pcodec; Blosc zstd for 8-bit arrays,
  which pcodec can't encode), MRMS plain zstd 9, features the configured Blosc.
- Values stay packed (`physical = value * scale_factor + add_offset`, per slot). The user decided
  against storing physical values or CF packing attributes.
- The `bounds` crop comes from the day's first ABI scan; with byte ranges each worker computes it from
  its own file and `check_window` requires a match. GLM is never cropped.
- With byte ranges unfetched bytes are zero, so decoders read only what the ranged functions fetch
  (GLM: `GLM_READ_VARIABLES`; ABI: `x`, `y` and the data/DQF inside the window). Reading more fails for
  compressed variables and silently gives zeros for uncompressed ones. `_RangeFile` keeps a range list
  and an anonymous `mmap`, never per-byte arrays (2.5× slower and 19 GB across the workers). Module
  settings such as `RANGE_MERGE_GAP_BYTES` are read at call time (tests change them).
- `DecodePool` replaces the pool when a worker dies. Workers ignore SIGINT; the main process handles
  Ctrl-C (cancels queued files, lets running ones finish, exits with 130).
- Missing data is explicit: ABI `present` = 0, GLM `covered` = 0; in a complete day they mean NOAA never
  published the file. `report.py` reads the root and group attributes (`complete`, `too_recent`, `n_*`,
  `n_present`, `abi_product_keys`, `glm_included`); keep them.
- Feature cells are blocks of the finest product's pixels counted from the uncropped sector origin, so
  they nest in the 2 km and 10 km grids; the geography store must match (`check_geography`). Coarser
  sources take the pixel under the cell centre, ties to the lower one.
- GLM is read on the lightning ellipsoid (+14/+6 km) to recover GLM's view angle. A flash is stored in
  its file's day but counts at its first-event time, so `GlmTimeline` reads the files starting within
  `FILE_MARGIN_NS` of the range from whichever day holds them. Event `flash_index` is -1 for orphans:
  mask it before indexing, or -1 picks the day's last flash.
- MRMS pixels go through GLM's mapping (`glm_projection`), so radar lands on the cells of the flashes
  above it. Placing them at the ellipsoid's height above the pixel sounds more correct but put flashes
  up to 1.2 rows south of the radar cores (README, "MRMS channels").
- **A footprint is pixels, never event points**: one cell per event leaves a lattice of holes that the
  labels and the model learn (README, "Flash footprints"). `pixel_cover` spreads glmtools' pixel
  rectangles by area, each flash at most once per cell, per minute. Round once, over the reported
  window: `sums("extent")` is unrounded (it adds up over slots), `extent_peak` rounded. The store's
  `glm.footprint` records the method.
- Inputs end at the issue time (slot start + 5 min) and label windows start there (`start_minutes` -5
  is for evaluation only). Labels are 255 and GLM channels NaN where coverage < `glm.min_coverage`.
  Finer labels go to `labels_fine` and must nest (`TargetGrid.finer`).
- A feature day is complete only if every raw and MRMS day it read inside the download range is.
- Memory sets the feature build's `max_workers` (8): a worker peaks at ~2.45 GB while it spreads
  footprints, and 12 workers ran the 29 GB machine into swap. That is why extents are int32 and
  `COVER_BATCH` is 100k. Search them with int32 values (`_range`): a Python int makes numpy convert the
  whole array to int64. Keep BLAS out of `build_day`: OpenBLAS threads in every worker (`np.dot`) took
  43% of the CPU.

## Conventions

- Output stores are regenerated, not migrated: no format-version checks or migration code.
- Keep README.md, the layout docstrings and this file in sync with behaviour changes. README's sizes,
  speeds and memory figures are measurements: re-measure when behaviour changes.
- Behaviour changes come with tests; keep `pytest` and pyflakes clean.
- Output (git-ignored): raw in `goes_raw2/` (the config's `raw_output_root_dir`), MRMS in `mrms_raw/`,
  features and `geography.zarr` in `goes_features/`, geography sources in `goes_static/`. `goes_raw/`
  holds 28 days from an older product list that the user kept; don't mix it with `goes_raw2/`.
- The config covers 2025-04-10 (GOES-19 became GOES-East on 2025-04-07) to 2026-11-01, 571 days. Per
  day: raw 1.7–2.0 GB stored (~1.1 TB in all), 4.9 GB downloaded with `"byte_ranges"` (16.1 GB with
  `"whole_files"`), ~100 s; MRMS ~0.66–0.72 GB (~400 GB), ~25 s; features ~0.56 GB with 15 channels
  (~320 GB), ~16 s on 8 workers, far slower next to a download (use fewer `--workers` then). Download
  MRMS before building features, or feature days are built twice.
