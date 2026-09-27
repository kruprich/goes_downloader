"""
download_raw.py: mirror GOES-19 ABI + GLM into one raw Zarr store per UTC day.

  python download_raw.py --config config.json

The code lives in the goes_downloader package; goes_downloader/store.py describes the store layout
and README.md covers usage.

Config:
  abi_products / glm_product define every product; "download" lists the keys to fetch.
  Products that read different variables from the same files (e.g. the dsi_* indices) share one
  download per scan, and each bucket prefix is listed once per day.
  download_method "byte_ranges" fetches only the HDF5 metadata and the chunks that are decoded
  (ABI chunks inside the crop) instead of whole files; the stored data is the same. The mcmip_*
  products read single bands from the ~59 MB all-band files and are only practical this way.
"""

# Imports stay inside the guard: every worker process re-runs this file on start (spawn), and this
# way the workers load only the modules their tasks need, not aiohttp or zarr.
if __name__ == "__main__":
    from goes_downloader.runner import main

    main()
