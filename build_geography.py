"""
build_geography.py: terrain, land/water and viewing angle on the feature grid, once.

  python build_geography.py --config config.json [--overwrite]

Downloads the configured sources into features.geography.cache_dir (once) and writes
<features.output_root_dir>/geography.zarr; see goes_downloader/features/geography.py.
"""

if __name__ == "__main__":
    from goes_downloader.features.build import geography_main

    geography_main()
