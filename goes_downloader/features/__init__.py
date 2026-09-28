"""
The feature step: raw day stores -> CNN inputs and lightning labels on one grid, one Zarr store per UTC day.

  python build_geography.py --config config.json    # once: terrain, land/water, viewing angle on the grid
  python build_features.py --config config.json     # every day in the range (days already complete are skipped)
  python build_features.py --config config.json --start D --end D --stats   # normalisation statistics

Everything is set by the "features" section of config.json (README.md lists the options). Modules:
settings (the config), expressions (derived channels), grid (target grid, regridding, sun and satellite
geometry), raw (reading the raw stores), glm (flashes, flash footprints, labels), geography (the one-time
static fields), build (the day builder and the runs).

Grid: the ABI fixed grid, coarsened by an integer factor from the finest product used. Each cell is a block
of coarsen x coarsen reference pixels counted from the uncropped CONUS sector's first pixel, so cells nest
exactly in the 2 km and 10 km ABI grids. Products at least as fine as the cell are aggregated over the
source pixels whose centres fall in it (mean/min/max/std/fraction); coarser products take the pixel
containing the cell centre. GLM flashes and events go to the cell containing their view angle, recovered
with the lightning ellipsoid GLM navigates to, so they line up with the cloud tops ABI sees.

Time: slot t holds the ABI scan starting in [t, t+5 min). The issue time is the end of the slot
(t + 5 min): every input ends there and every label window starts there. Solar channels are computed for
the middle of the slot.

GLM: "count" channels and centroid labels use each flash's centroid. "extent" channels and labels use its
footprint: every cell one of its events fell in (the flash-extent density of LightningCast's labels).
Orphan events (flash_index -1 in the raw store: NOAA's file lacks their group or flash) are skipped.
A label with min_flashes > 1 counts flash-cell hits within its radius.

Layout of <features.output_root_dir>/YYYY/YYYYDDD.zarr
  time[time]                           datetime64 slot start, every time_step_minutes
  channel[channel], label[label]       names in config order (channels with "output": false excluded)
  features[time, channel, y, x]        float16 (or float32); NaN = missing. Chunked (1, all channels, chunk_yx)
  labels[time, label, y, x]            uint8 1 = lightning in the window, 0 = none, 255 = unknown (GLM covered
                                       less than glm.min_coverage of the window)
  flash_count[time, y, x]              uint16 flashes (quality-filtered) whose centroid is in the cell and whose
                                       first event is in [issue - time_step, issue)
  flash_extent[time, y, x]             uint16 flashes in the same window with an event in the cell (with GLM
                                       events); these two let other labels, baselines and scores be derived later
  glm_coverage[time]                   float32 fraction of that window covered by GLM files
  channel_valid_fraction[time, channel] float32 fraction of cells that are not NaN
  x[x], y[y]                           float64 cell-centre scan angles (rad) on the ABI fixed grid
  lat[y, x], lon[y, x]                 float32 cell centres (deg)

Root attrs:
  complete = true only if the day's raw store and every neighbouring raw day that was read (lookback for
  lags and gap filling, lookahead for labels) are complete. A neighbour outside the raw config's date range
  never will be, so it counts as unavailable instead: its slots are NaN / unknown. Incomplete days are
  rebuilt from scratch on the next run.
  channel_stats / label_stats: per-day sums for normalisation and class balance (combine with --stats).
  grid, channels, helper_channels, labels, glm, time_step_minutes, geography_path: how the store was built.

The geography store (<output_root_dir>/geography.zarr, one for all days) is described in geography.py.
"""
