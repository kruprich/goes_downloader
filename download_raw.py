"""
download_raw.py: mirror GOES-19 ABI + GLM into one raw Zarr store per UTC day.

  python download_raw.py --config config.json [--no-report]

goes_downloader/store.py describes the store layout; README.md covers the config and usage.
"""

# Imports stay inside the guard: spawned workers re-run this file, and should load only what their
# tasks need, not aiohttp or zarr.
if __name__ == "__main__":
    from goes_downloader.runner import main

    main()
