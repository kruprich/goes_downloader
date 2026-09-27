# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A downloader (`download_raw.py` → the `goes_downloader` package, driven by `config.json`) that
mirrors GOES-19 ABI CONUS L2 products and GLM lightning flashes and events from the public bucket
`gcp-public-data-goes-19` into one raw Zarr store per UTC day. It is the raw layer for a
lightning-prediction training dataset: it deliberately does **no** time binning, regridding or
quality filtering. That belongs in a separate feature-building step that does not exist yet.
`report.py` summarizes what the stores hold. The docstring at the top of `goes_downloader/store.py`
is the authoritative description of the store layout and its attributes; README.md covers setup,
usage, reading, assumptions about the source data, and measured sizes and speeds.

## Commands

Shells here don't activate the virtual environment, so call its Python directly. Python 3.11–3.13
only: 3.14 can't install the pins (`numcodecs<0.16` has no 3.14 wheel).

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # requirements.lock alone to only run it
.venv/bin/python download_raw.py --config config.json     # a full run; prints a data report at the end
.venv/bin/python report.py --config config.json           # which days are complete / need attention
.venv/bin/python -m pytest                                # offline tests, under a second
.venv/bin/python -m pytest --network                      # + real files and the test hour (~0.5 GB, ~30 s)
.venv/bin/python -m pyflakes goes_downloader tests download_raw.py report.py   # the only linter
```

`requirements.lock` holds the tested versions; regenerate it from the environment's installed
versions (runtime packages only) when a dependency changes.

## Testing and verifying changes

Before editing, check that no download is running from this checkout (`pgrep -af download_raw`):
the decode pool replaces a worker that dies, and the replacement imports the code as it is then.

**Offline tests** (`tests/`) build small MCMIPC- and LCFA-like NetCDF files (`conftest.py`) and
serve them to the byte-range code through a fake remote whose buffers start as random bytes
(`junk_fill`), so decoding anything that was not fetched fails the comparison.
- `test_ranges.py`: byte-range decoding equals whole-file decoding; chunks of unread bands or
  outside the crop aren't fetched; the fetched-range bookkeeping against a per-byte model.
- `test_decode.py`: rounding of packed values; linking GLM events to flashes and its errors.
- `test_store.py`: GLM tables (flash order, event links after sorting, counts, coverage).
- `test_config.py`: config validation and IR-band precision; `crop_slice`; the entry script
  importing nothing outside its main guard.
- `test_report.py`, `test_runner.py`: the report's rules; the report after a run, `--no-report`,
  Ctrl-C.

**Network tests** (`test_network.py`, `--network`) decode two real files of every configured
product both ways and run the reference hour through the whole pipeline with both
`download_method` values, requiring identical stores.

For a short real download, write a harness script that sets `remote.LIST_HOURS`. It must be a
file: the pool uses spawn (Linux would otherwise fork) and spawned workers re-run the main script,
so code fed through stdin (`python3 - <<EOF`) fails in every worker.

```python
# run_test.py: one hour into a scratch dir (copy config.json, override dates/output/overwrite)
import sys, asyncio
sys.path.insert(0, "/path/to/goes_downloader")  # not needed when the script is in the repo root
from goes_downloader import config, remote, runner
if __name__ == "__main__":
    remote.LIST_HOURS = [18]
    asyncio.run(runner.run_all_days(config.load_config(sys.argv[1])))
```

- The usual reference hour is 2025-10-27 18 UTC (~5 s, 228 files). Days that ended less than 24 h
  ago are never marked complete, so test with past dates.
- Write test output to scratch dirs, never `goes_raw/`: a store built from a few hours is still
  marked complete, so the real run would skip that day (`report.py` would flag it).
- For changes to fetching or decoding, compare against the previous version's store array by
  array and run both `download_method` values; the stores must be identical.
- For changes that can affect speed or memory, time a full day with the real command and sample
  the process group's RSS from `/proc`; README.md's numbers come from such runs.

## Architecture

**Modules (`goes_downloader/`).**
- `config`: the settings dataclasses and `load_config`.
- `remote`: file names, listing (`list_day_names`, `select_files`, `LIST_HOURS`), whole-file
  downloads (`http_get`, aiohttp).
- `decode`: NetCDF → packed ABI arrays and GLM flash/event tables; `DecodePool`.
- `crop`: the bounds → scan-angle rectangle → pixel window.
- `ranges`: byte-range fetching and decoding in the pool workers (`_RangeFile`).
- `store`: the layout docstring; Zarr writing (`AbiProductState`, `write_glm_tables`).
- `day`: `process_single_day`.
- `runner`: `run_all_days`; `main` handles Ctrl-C and prints the report unless `--no-report`.
- `report`: `day_rows`, `problems`, `print_report`.
- `common`: shared constants. `progress`: `log` and the progress line (`progress.current`).

Pool workers import only `ranges`/`decode` and what they need: `download_raw.py` keeps its imports
inside the main guard, and `remote` imports aiohttp inside `http_get`, so workers skip aiohttp and
zarr (~146 MB each). Keep it that way (`test_entry_script_imports_nothing_outside_its_main_guard`).

**Config (`config.json`).** `abi_products` and `glm_product` define the downloadable products (all
16 CMIP bands twice, plus single-variable CONUS L2 products, each with a `description`): `cmip_*`
read each band's own file, `mcmip_*` read one band from the all-band MCMIPC file at 2 km, which
is only practical with `"byte_ranges"` (~59 MB per whole file; `run_all_days` warns otherwise).
The `download` list selects which to fetch; it currently holds LightningCast's bands 2, 5, 13 and
15 (`mcmip_*`). `cmip_precision` applies to infrared CMIP bands 7–16 (band number from
`filename_must_contain`, or from the `CMI_Cnn` data variable for MCMIPC); a product's own
`precision` overrides it. `bounds` crops ABI only. Unknown top-level keys are ignored.

**Per-day flow (`process_single_day`).**
1. List each distinct `gcs_product_prefix` once (24 hourly prefixes). `select_files` filters by
   `filename_must_contain` and keeps the newest creation time when NOAA republishes a scan.
2. Group ABI products by source file (`(gcs_product_prefix, filename_must_contain)`), so products
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
   flash ids (`_parent_rows`; a missing parent is a file error); `write_glm_tables` turns that into
   `flash_index`, the row in the day's time-sorted `glm/flashes`.
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

## Conventions

- Output stores are regenerated, not migrated. Don't add format-version checks or migration code
  for old stores.
- Keep README.md, the `store.py` docstring (layout and every attribute) and this file in sync with
  behaviour changes. README.md's sizes, speeds and memory figures are measurements: re-measure when
  behaviour changes and say what was measured. Assumptions that only matter for interpreting the
  data belong in the future feature step, not the README.
- Behaviour changes come with tests; keep `pytest` and pyflakes clean.
- Output goes to `raw_output_root_dir` (default `./goes_raw`, git-ignored). Current config: about
  2.1 GB/day stored with the NOAA CONUS bounds (~675 GB over the date range). With
  `"byte_ranges"` a day downloads 4.9 GB and takes 95–110 s on a ~60 MB/s connection (8.5–10 h
  for the range); `"whole_files"` would download 16.1 GB/day because of the MCMIPC files.
