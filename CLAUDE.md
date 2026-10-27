# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A downloader (`download_raw.py` → the `goes_downloader` package, driven by `config.json`) that
mirrors GOES-19 ABI CONUS L2 products and GLM lightning flashes and events from NOAA's public bucket
on AWS (`s3://noaa-goes19`, the config's `bucket`) or Google Cloud (`gs://gcp-public-data-goes-19`;
same files and names) into one raw Zarr store per UTC day. It is the raw layer for a
lightning-prediction training dataset: it deliberately does **no** time binning, regridding or
quality filtering. That is the feature step (`goes_downloader/features/`, run by
`build_geography.py` once and `build_features.py`), which turns the raw stores into CNN inputs and
lightning labels on one grid, one feature store per day, for the model in the separate
`goes_lightning_model` repository. `report.py` summarizes what the raw stores hold. A separate
downloader (`download_mrms.py` → `goes_downloader/mrms.py`, the config's `mrms` section) mirrors MRMS
radar mosaics from `s3://noaa-mrms-pds` into their own raw day stores the same way; the feature step
doesn't read them yet. The docstrings of `goes_downloader/store.py` (raw stores),
`goes_downloader/mrms.py` (MRMS stores), `goes_downloader/features/__init__.py` (feature stores)
and `goes_downloader/features/geography.py` (the geography store) are the authoritative layouts;
README.md covers setup, usage, reading, assumptions about the source data, and measured sizes and
speeds.

## Commands

Shells here don't activate the virtual environment, so call its Python directly. Python 3.12–3.14
(`requires-python` in pyproject.toml): the locked numpy and pyproj need 3.12. On the Mac checkout the venv uses python.org's 3.12, which has no CA
certificates: prefix anything that touches the network with `SSL_CERT_FILE=/etc/ssl/cert.pem`.

```bash
.venv/bin/python -m pip install -e ".[dev]" -c requirements.lock   # without [dev] to only run it
.venv/bin/python download_raw.py --config config.json     # a full run; prints a data report at the end
.venv/bin/python download_mrms.py --config config.json    # MRMS radar day stores (about 40 s a day)
.venv/bin/python report.py --config config.json           # which days are complete / need attention
.venv/bin/python build_geography.py --config config.json  # once: static fields on the feature grid
.venv/bin/python build_features.py --config config.json   # feature stores; --start/--end/--workers/--stats
.venv/bin/python -m pytest                                # offline tests, about 10 s
.venv/bin/python -m pytest --network                      # + real files and the test hour (~0.5 GB, ~30 s)
.venv/bin/python -m pyflakes goes_downloader tests download_raw.py download_mrms.py report.py build_features.py build_geography.py
```

`pyproject.toml` holds the package definition, `requires-python`, the direct dependencies and the
pytest settings; `requirements.lock` pins the tested versions and is used as constraints (`-c`).
When a dependency changes, regenerate the lock from the environment's installed versions (runtime
packages only) and run the tests on 3.12 and 3.13, as `.github/workflows/tests.yml` does on every
push. There are deliberately no console-script entry points: generated scripts import the package
at top level, and spawned workers re-run the main script, so they would load aiohttp and zarr.

## Testing and verifying changes

Before editing, check that no download is running from this checkout (`pgrep -af download_raw`):
the decode pool replaces a worker that dies, and the replacement imports the code as it is then.

**Offline tests** (`tests/`) build small MCMIPC- and LCFA-like NetCDF files (`conftest.py`) and
serve them to the byte-range code through a fake remote whose buffers start as random bytes
(`junk_fill`), so decoding anything that was not fetched fails the comparison.
- `test_ranges.py`: byte-range decoding equals whole-file decoding; chunks of unread bands or
  outside the crop aren't fetched; the fetched-range bookkeeping against a per-byte model;
  `_get_range` splits the object URL into host and path (fake `HTTPSConnection`).
- `test_remote.py`: object URLs and `bucket` validation for both clouds; S3 (XML) and GCS (JSON)
  listing pages through continuation tokens.
- `test_decode.py`: rounding of packed values; linking GLM events to flashes and its errors.
- `test_store.py`: ABI frames (unwritten slots read as the fill, which is also their `_FillValue`;
  dimension names; pcodec, Blosc for 8-bit; `consolidate` raises no warning); GLM tables (flash order,
  event links after sorting, counts, coverage, codecs; a day without GLM data).
- `test_config.py`: config validation and IR-band precision; `crop_slice`; the entry scripts
  importing nothing outside their main guards.
- `test_mrms.py`: GRIB2 files built like MRMS's (lat/lon grid, a real 16-bit PNG, signed packing
  exponents) decode to their packed values; other packings are refused; the crop; a day against a
  fake bucket (a failed download leaves the day incomplete and its row at the fill, the rerun
  completes it, a complete day downloads nothing; a file on another grid is an error); config errors.
- `test_report.py`, `test_runner.py`: the report's rules; the report after a run, `--no-report`,
  Ctrl-C.
- `test_features.py`: the feature step on two synthetic raw days on the real fixed grid (a 2 km
  and a 10 km product with bad-DQF, fill and missing scans; hand-placed flashes and events): block
  aggregation and nearest-pixel sampling against the pixels, `lag` across midnight, `mask` against
  the sun, flash centroids and footprints, labels (including a flash stored in the next day's first
  file and a GLM gap), completeness, geography regridding, borders rasterized by the even-odd rule
  (holes, a `within` filter) and their distances against brute force, sun and satellite geometry against
  astropy/spherical values, config errors. The raw days are written once per module (`raw_days`) and
  each test's `raw/` links to them: a test that adds raw days must copy them first.

For changes to the feature step, also build a few real days (with 2–3 `--workers` while a download
runs) and compare against values recomputed from the raw arrays: block means of the 2 km frames,
the 10 km pixel under each cell centre, and flash footprints mapped from `glm/events` with NOAA's
formula instead of pyproj (skipping orphan events, `flash_index` -1). Write them to a scratch `output_root_dir`; avoid raw days that have a
`.tmp` next to them (the downloader is rebuilding them).

**Network tests** (`test_network.py`, `--network`) decode two real files of every configured
product both ways and run the reference hour through the whole pipeline with both
`download_method` values, requiring identical stores, all from the configured `bucket`; and list the
reference hour on both clouds in 50-object pages, requiring the same names and sizes; and download
the reference hour of every configured MRMS product from AWS (30 files on the 0.01° grid, cropped).
The MRMS decoder was also checked once against eccodes (identical values on five files, 2025–2026);
eccodes isn't a dependency, so repeat that in a scratch venv if `decode_mrms` changes.

For a short real download, write a harness script that sets `remote.LIST_HOURS` and run it with
`.venv/bin/python` (the package is installed in `.venv`, so the script can live anywhere). It must
be a file: the pool uses spawn (Linux would otherwise fork) and spawned workers re-run the main
script, so code fed through stdin (`python3 - <<EOF`) fails in every worker.

```python
# run_test.py: one hour into a scratch dir (copy config.json, override dates/output/overwrite)
import sys, asyncio
from goes_downloader import config, remote, runner
if __name__ == "__main__":
    remote.LIST_HOURS = [18]
    asyncio.run(runner.run_all_days(config.load_config(sys.argv[1])))
```

- The usual reference hour is 2025-10-27 18 UTC (~5 s, 228 files). Days that ended less than 24 h
  ago are never marked complete, so test with past dates.
- Write test output to scratch dirs, never the configured output (`goes_raw2/`) or the older
  `goes_raw/`: a store built from a few hours is still
  marked complete, so the real run would skip that day (`report.py` would flag it).
- For changes to fetching or decoding, compare against the previous version's store array by
  array and run both `download_method` values; the stores must be identical.
- For changes that can affect speed or memory, time a full day with the real command and sample
  the process group's RSS from `/proc`; README.md's numbers come from such runs.

## Architecture

**Modules (`goes_downloader/`).**
- `config`: the settings dataclasses and `load_config`.
- `remote`: file names, object URLs (`object_url`), listing (`list_object_names` for S3's
  ListObjectsV2 XML or the GCS JSON API, `list_day_names`, `select_files`, `LIST_HOURS`,
  `LIST_PAGE_SIZE`), whole-file downloads (`http_get`, aiohttp). The only module that knows the
  clouds apart: the main process builds each object's HTTPS URL, and byte-range workers get that.
- `decode`: NetCDF → packed ABI arrays and GLM flash/event tables; `DecodePool`.
- `crop`: the bounds → scan-angle rectangle → pixel window.
- `ranges`: byte-range fetching and decoding in the pool workers (`_RangeFile`).
- `store`: the layout docstring; Zarr writing (`AbiProductState`, `write_glm_tables`, `consolidate`,
  `frame_codecs`).
- `day`: `process_single_day`.
- `runner`: `run_all_days`; `main` handles Ctrl-C and prints the report unless `--no-report`.
- `report`: `day_rows`, `problems`, `print_report`.
- `common`: shared constants. `progress`: `log` and the progress line (`progress.current`).
- `mrms`: the MRMS downloader on its own: its config section (`load_mrms_config`, sharing the GOES
  dates, bounds and HTTP settings), a GRIB2 reader for MRMS's one layout (`decode_mrms`: template 3.0
  grid, 5.41 PNG packing decoded by Pillow, no bitmap; anything else is refused), `crop_window`,
  `process_mrms_day` and `main`. It reuses `remote`'s listing and `http_get` and decodes in threads
  (gzip, Pillow and zstd release the GIL), so it has no process pool and none of the worker import
  rules below. It filters files by `remote.LIST_HOURS` too, so the short-run harness works for it.

Pool workers import only `ranges`/`decode` and what they need: `download_raw.py` keeps its imports
inside the main guard, and `remote` imports aiohttp inside `http_get`, so workers skip aiohttp and
zarr (~146 MB each). Keep it that way (`test_entry_script_imports_nothing_outside_its_main_guard`).

**Config (`config.json`).** `bucket` is `s3://<name>` or `gs://<name>` (validated by
`BUCKET_PATTERN`; stored as the root attr `source_bucket`), read anonymously over HTTPS. Each
product's `product_prefix` is the top-level folder, the same on both clouds.
`abi_products` and `glm_product` define the downloadable products (all
16 CMIP bands twice, plus single-variable CONUS L2 products, each with a `description`): `cmip_*`
read each band's own file, `mcmip_*` read one band from the all-band MCMIPC file at 2 km, which
is only practical with `"byte_ranges"` (~59 MB per whole file; `run_all_days` warns otherwise).
The `download` list selects which to fetch; it currently holds LightningCast's bands 2, 5, 13 and
15 (`mcmip_*`), the 10 km `acha`, `tpw`, `dsi_cape`, `dsi_li` and `dsi_ki`, and `glm`. `cmip_precision` applies to infrared CMIP bands 7–16 (band number from
`filename_must_contain`, or from the `CMI_Cnn` data variable for MCMIPC); a product's own
`precision` overrides it. `bounds` crops ABI only. `load_config` logs every key it doesn't read
(`unknown_keys`; `_`-prefixed keys are notes): a new setting must be added to `TOP_LEVEL_KEYS` (or
the product/bounds key sets) as well as read.

**Per-day flow (`process_single_day`).**
1. List each distinct `product_prefix` once (24 hourly prefixes). `select_files` filters by
   `filename_must_contain` and keeps the newest creation time when NOAA republishes a scan.
2. Group ABI products by source file (`(product_prefix, filename_must_contain)`), so products
   reading different variables from one file (e.g. the `mcmip_*` bands, the `dsi_*` indices) share
   one download and one decode. Each file goes to a 5-minute slot (288/day) by scan start.
3. Download (`http_get`, retries on transient errors) → decode in `DecodePool` (returns packed
   integers + DQF) → validate every product of the file (`accept_first_or_validate`: grid, fill,
   DQF dtype must match the day's first scan; `check_window`) before writing any → write one Zarr
   chunk per slot from a thread. With `download_method` `"byte_ranges"` the pool workers fetch too
   (`ranged_decode_abi_file` / `ranged_decode_glm_file`): h5py reads the HDF5 metadata through
   `_RangeFile`, only the chunks that are decoded are fetched (ABI: those overlapping the crop
   window, which the worker computes from its own file), and the unchanged NetCDF decoders read the
   resulting file-sized buffer. Files start in scan-time order so latency-bound GLM fetches overlap
   bandwidth-bound ABI ones.
4. GLM flashes and events (~70 M events/day, up to ~1.9 GB in the main process) accumulate in
   memory and are written at the end with per-file rows and a `glm/coverage` flag per 20-second
   window. `decode_glm_file` links each event to its flash through the file's event → group →
   flash ids (`_parent_rows`, `_through`; an orphan gets -1: its group or flash is missing from the
   file, or the id is used twice because 16-bit flash ids wrap around in very active files).
   `write_glm_tables` turns that into `flash_index`, the row in the day's time-sorted
   `glm/flashes`, and counts orphans (`n_orphan_events`). No id problem fails a file.
5. Build in `<day>.zarr.tmp`, consolidate metadata, then move into place. `complete` is true only
   if no listing/file errors occurred and the day ended ≥24 h ago; incomplete days are rebuilt
   from scratch on the next run.

**Feature step (`goes_downloader/features/`).** `settings` (the `features` section; unknown keys are
errors), `expressions` (derived channels and the evaluation `Plan`), `grid` (`TargetGrid`,
`make_grid`, `axis_map`/`reduce2d`/`aggregate`, `SolarGeometry`, `satellite_zenith_deg`), `raw`
(`RawTimeline`: raw frames by global 5-min slot across days), `glm` (`GlmTimeline`: centroids,
footprints from events, coverage), `geography` (the one-time static fields, including country borders
rasterized from GeoJSON by scanline: `rasterize_rings`, `distance_to_km`), `build` (`build_day` in
spawn workers, `run_build`, `run_stats`, the CLIs). `build_day` goes slot by slot from `lookback`
slots before the day to its end. Only `Plan.every_slot` channels (read at earlier slots, carrying
values forward with `fill_gaps`, or read by such a channel) are computed at every slot; the rest only
at output slots, which cut a day from 42–44 s to 32 s with identical output. The cache keeps only
the slots `lag`/`*_past` still need. `run_build` rebuilds a complete day when a raw day it recorded
as missing or incomplete has since completed, or when its store can't be read (`raw_neighbour_arrived`);
on Ctrl-C it terminates the workers, which exit at once (`_exit_now`: exiting through Python would let
the pool's worker loop start the next day), and the pool is always shut down so its atexit hook can't
join the queued days. `build_day` publishes by renaming any existing store to `<day>.zarr.old`
before moving `.tmp` into place, so a worker killed mid-publish can't leave a store that reads as
complete with its arrays deleted; `run_build` removes leftover `.old` directories.

**Invariants that are easy to break:**
- Every store is Zarr format 3 (`zarr.open_group(..., zarr_format=3)`); code tells a store exists by
  its `zarr.json`, so format-2 stores from before count as missing and are rebuilt. Every array names
  its dimensions (`dimension_names`); arrays written in full use `_write_array`. Consolidate with
  `store.consolidate`, which silences zarr's warning that consolidated metadata isn't in the v3 spec.
- Frame arrays (`value`, `dqf`) keep a real fill value because unwritten slots must read as
  missing; products whose files lack `_FillValue` use 65535.
- Raw ABI frames and GLM columns take their codec from `frame_codecs(dtype)`: pcodec 8, except 8-bit
  arrays (DQF, `quality_flag`), which pcodec can't encode and keep Blosc zstd 3 with bitshuffle. MRMS
  frames use plain zstd 9 and feature frames the configured Blosc (`features.compression`).
- xarray ignores a Zarr v3 array's fill value and masks only a `_FillValue` attribute. Integer frames
  with a fill (ABI `value`/`dqf`, MRMS `value`, `labels`, `labels_fine`) repeat it as `_FillValue`,
  so xarray reads fills as NaN as it did with format 2. Arrays written in full (fill 0) must not have
  one, or legitimate zeros would read as missing; NaN fills need none.
- Values stay packed (`physical = value * scale_factor + add_offset`, stored per slot). The user
  decided against storing physical values or CF packing attributes.
- The `bounds` rectangle comes from the first ABI scan decoded each day (all ABI products share
  the GOES-East fixed grid). With byte ranges each worker computes the window from its own file,
  and `check_window` requires it to equal the kept crop. GLM is deliberately never cropped
  (~240 MB/day with events).
- `DecodePool` replaces the process pool when a worker dies, and closes the dead pool's call-queue
  reader to avoid CPython gh-94777, a hang at exit on Python < 3.12. Workers ignore SIGINT; the
  main process handles Ctrl-C (cancels queued files, lets running ones finish, exits with 130).
- With byte ranges, every byte that is not fetched is zero, so a decoder must read only what the
  ranged functions fetch: GLM only `GLM_READ_VARIABLES`, ABI only `x`, `y` and the data/DQF
  variables inside the window. Reading anything else fails for compressed variables and silently
  gives zeros for uncompressed ones such as the GLM variables. `_RangeFile` keeps a list of fetched
  ranges and an anonymous `mmap`, never per-byte arrays: MCMIPC files are ~59 MB, and file-sized
  numpy bookkeeping made a day 2.5× slower and took 19 GB across the workers. Module-level
  settings such as `RANGE_MERGE_GAP_BYTES` are read at call time (tests change them).
- Missing data is explicit: ABI `present` = 0 and GLM `covered` = 0. In a complete day these mean
  NOAA never published the file.
- `report.py` reads root and group attributes (`complete`, `too_recent`, `n_*`, `n_present`,
  `abi_product_keys`, `glm_included`); keep them when changing the writer.
- Feature grid cells are blocks of the finest product's pixels counted from the uncropped sector
  origin, so they nest in the 2 km and 10 km grids; the geography store must have the same cells
  (`check_geography`). Coarser sources take the pixel under the cell centre, ties to the lower one.
- GLM flashes and events are read on the lightning ellipsoid (+14/+6 km), which recovers GLM's view
  angle. A flash belongs to its file's day store but counts at its own first-event time: one that
  starts just before midnight is in the next day's first file. So `GlmTimeline` reads the files that
  start within 20 s of the range (`FILE_MARGIN_NS`), whichever day holds them, and only their events
  (offsets from `glm/files/n_events`). Event `flash_index` is -1 for orphans (no flash in their
  file): mask it before indexing flashes, or -1 silently picks the day's last flash.
- **A footprint is pixels, never event points.** An event marks the centre of a GLM pixel 1.3–2 cells
  across; putting each in the one cell under it leaves a lattice of holes that the labels, the `fed`
  input and so the model all learn (README, "Flash footprints"). `pixel_cover` spreads glmtools'
  pixel rectangles by area, one flash at most once per cell, and works per minute so a label can take
  1-minute densities. Rounding happens once, over the window being reported: `sums("extent")` is the
  unrounded density (it adds up over slots), `extent_peak` the rounded one. Rounding per slot and then
  summing gives a different answer. The method is recorded in the store's `glm.footprint`, so stores
  built before it can be told apart.
- Feature inputs end at the issue time (slot start + 5 min); label windows start there, or at the
  scan's own slot for a label with `start_minutes` -5, which is for evaluation only. Labels are 255
  where GLM covered less than `glm.min_coverage`; GLM channels are NaN likewise. A label with a finer
  `resolution_km` goes to `labels_fine`; its cells must nest in the grid's (`TargetGrid.finer`).
- `flash_count` and `flash_extent` are written in full (fill 0, no `_FillValue`); `features` uses NaN
  and `labels` 255. A feature day is complete only if every raw day it read inside the download range is;
  days next to the range's ends are rebuilt once a missing neighbour arrives (`raw_neighbour_arrived`).
- The entry scripts import nothing outside their main guard (spawn workers re-run them).

## Conventions

- Output stores are regenerated, not migrated. Don't add format-version checks or migration code
  for old stores.
- Keep README.md, the `store.py` docstring (layout and every attribute) and this file in sync with
  behaviour changes. README.md's sizes, speeds and memory figures are measurements: re-measure when
  behaviour changes and say what was measured. Assumptions that only matter for interpreting the
  data belong in the feature step's documentation.
- Behaviour changes come with tests; keep `pytest` and pyflakes clean.
- Output goes to `raw_output_root_dir` (default `./goes_raw`; the config uses `./goes_raw2`;
  `goes_raw*/` is git-ignored). `goes_raw/` holds 28 days from an older product list (bands 8, 11,
  13, 15; no GLM events) that the user kept; don't mix it with `goes_raw2/`. Current config:
  2025-04-10 (GOES-19 became operational as GOES-East on 2025-04-07) to 2026-09-25, 534 days, about
  1.7 GB/day stored (2025-10-27; 2.0 GB on 2025-07-15) with the NOAA CONUS bounds and pcodec, so
  ~0.9–1.0 TB for the range (Blosc zstd 3 with bitshuffle stored 18% more, the same CPU). With `"byte_ranges"` a day downloads 4.9 GB and
  took 95–110 s from Google Cloud on a ~60 MB/s connection (14–16 h for the range); `"whole_files"`
  would download 16.1 GB/day because of the MCMIPC files. README's speed and memory figures are from
  Google Cloud except its "Choosing a cloud" section: on the 8-core, 16 GB Mac with 24 processes a day
  took 2 min 53 s from AWS and 4 min 01 s from Google Cloud (one run each, identical stores).
- Features go to `features.output_root_dir` (`./goes_features`, git-ignored) with `geography.zarr`
  in it; geography sources are cached in `./goes_static`. Current config: about 0.32 GB/day
  (~175 GB for the range), stored as whole-frame LZ4HC 5 chunks. On 2025-07-15 with zarr 3 that stored
  13% less than LZ4 9 and read a whole frame in 2.7 ms against 3.1 ms, for ~7 more CPU-seconds a day;
  zstd 3 stored 3% less but read in 5.4 ms, pcodec as little but in 9.1 ms (zarr 2: whole-frame chunks
  read 11x faster than zstd in 256x256 chunks).
  Building is CPU-bound and zarr 3 runs Blosc on one thread, so `max_workers` is the core count (12).
  Measured with zarr 2 and one Blosc thread per worker, that built a day in 11.9 s against 15.8 s for
  4 workers with Blosc threading over every core, and a day on its own took ~27 s and 113 CPU-seconds.
  zarr 3 measured 3–9% slower than zarr 2 on identical output (October 2026, 2025-07-15 alone: 47.0 s
  against 43.2 s; 3 days on 3 workers: 65.3 s against 62.6 s). Much slower next to a download: use
  far fewer `--workers` then.
- MRMS goes to `mrms.output_root_dir` (`./mrms_raw`, git-ignored). Current config: the -10 °C
  reflectivity only, 720 files a day, ~1 GB/day downloaded and ~0.72 GB stored with plain zstd 9
  (Blosc zstd 3 stored 10% more with byte shuffle and 67% more with bitshuffle, pcodec 31% more),
  45–47 s a day with 8 threads and ~1.1 GB peak memory (zstd 9 costs ~1.5 s of that; on 2026-10-26
  the network was slower than for the earlier 37–39 s), so about 7 h and 370–400 GB for the range.
