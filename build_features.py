"""
build_features.py: raw day stores -> CNN input features and lightning labels, one Zarr store per UTC day.

  python build_features.py --config config.json
  python build_features.py --config config.json --start 2025-04-10 --end 2025-12-31 --stats

goes_downloader/features/__init__.py describes the store layout; README.md covers the options.
Run build_geography.py once first for the static fields.
"""

# Imports stay inside the guard: every worker process re-runs this file on start (spawn)
if __name__ == "__main__":
    from goes_downloader.features.build import features_main

    features_main()
