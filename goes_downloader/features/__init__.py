"""
The feature step: raw day stores -> CNN inputs and lightning labels on one grid, one Zarr store per UTC day.

  python build_geography.py --config config.json    # once: terrain, land/water, viewing angle, borders
  python build_features.py --config config.json     # every day in the range (days already complete are skipped)
  python build_features.py --config config.json --start D --end D --stats   # normalisation statistics

Everything is set by the "features" section of config.json (README.md lists the options and the methods).

Grid: blocks of coarsen x coarsen pixels of the finest product used, counted from the uncropped CONUS
sector's first pixel, so cells nest exactly in the 2 km and 10 km ABI grids. Finer products are aggregated
over the pixels whose centres fall in a cell; coarser ones take the pixel containing the cell centre. GLM
flashes and events, and MRMS pixels, go to the cell of their view angle on GLM's lightning ellipsoid.

Time: slot t holds the ABI scan starting in [t, t+5 min). The issue time is the end of the slot
(t + 5 min): every input ends there and every label window starts there.

GLM: "count" channels and centroid labels use each flash's centroid; "extent" channels and labels its
footprint, built as glmtools builds flash-extent density (glm.py). "extent" channels hold that density
before rounding; flash_extent and extent labels round it half up over their window. Orphan events
(flash_index -1) are skipped. A label with min_flashes > 1 counts flash-cell hits within its radius.

Layout of <features.output_root_dir>/YYYY/YYYYDDD.zarr (Zarr format 3, like the raw stores)
  time[time]                           datetime64 slot start, every time_step_minutes
  channel[channel], label[label]       names in config order (channels with "output": false excluded)
  features[time, channel, y, x]        float16 (or float32); NaN = missing. One chunk per time step (or
                                       chunk_yx), compressed with Blosc features.compression
  labels[time, label, y, x]            uint8 1 = lightning in the window, 0 = none, 255 = unknown (GLM covered
                                       less than glm.min_coverage of the window; also the _FillValue attribute,
                                       so xarray reads it as NaN)
  labels_fine[time, label_fine, y_fine, x_fine]   the same for labels with a finer resolution_km (for
                                       evaluation), each grid cell holding factor x factor of their cells;
                                       label_fine[label_fine], x_fine[x_fine], y_fine[y_fine] as below
  flash_count[time, y, x]              uint16 flashes (quality-filtered) whose centroid is in the cell and whose
                                       first event is in [issue - time_step, issue)
  flash_extent[time, y, x]             uint16 flash-extent density over the same window, rounded (from GLM
                                       events); these two let other labels, baselines and scores be derived later
  glm_coverage[time]                   float32 fraction of that window covered by GLM files
  channel_valid_fraction[time, channel] float32 fraction of cells that are not NaN
  x[x], y[y]                           float64 cell-centre scan angles (rad) on the ABI fixed grid
  lat[y, x], lon[y, x]                 float32 cell centres (deg)

Root attrs:
  complete = true only if the day's raw store and every neighbouring raw day that was read (lookback for
  lags and gap filling, lookahead for labels) are complete, and every MRMS day read. A neighbour outside the
  raw config's date range never will be, so it counts as unavailable instead: its slots are NaN / unknown.
  Incomplete days are rebuilt from scratch on the next run.
  raw_days, mrms_days (with MRMS channels): {YYYY-MM-DD: complete, incomplete or missing} of the days read;
  a complete store is rebuilt when one of them has since completed.
  channel_stats / label_stats: per-day sums for normalisation and class balance (combine with --stats).
  label_stats counts each label's cells on its own grid (n_cells, n_known, n_positive), so a labels_fine
  label has factor x factor as many as the grid.
  grid, fine_grid (with labels_fine), channels, helper_channels, labels, glm (with the footprint method),
  mrms (with MRMS channels: lightning_ellipsoid_km, max_age_minutes), time_step_minutes, geography_path:
  how the store was built.

The geography store (<output_root_dir>/geography.zarr, one for all days) is described in geography.py.
"""
