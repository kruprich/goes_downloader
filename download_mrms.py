"""
download_mrms.py: mirror MRMS radar products from NOAA's AWS bucket into one raw Zarr store per UTC day.

  python download_mrms.py --config config.json

The config's "mrms" section lists the products; goes_downloader/mrms.py describes the store layout.
"""

# Imports stay inside the guard, as in the other entry scripts
if __name__ == "__main__":
    from goes_downloader.mrms import main

    main()
