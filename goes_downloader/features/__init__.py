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
footprint, built as glmtools builds flash-extent density (LightningCast's truth): each event's GLM pixel,
rebuilt from glmtools' pixel-corner table, spread onto the cells by area, each flash at most once per
cell. "extent" channels hold that density before rounding; flash_extent and extent labels round it half
up over their window (README, "Flash footprints"). Orphan events (flash_index -1 in the raw store: NOAA's
file lacks their group or flash) are skipped. A label with min_flashes > 1 counts flash-cell hits within
its radius.

Layout of <features.output_root_dir>/YYYY/YYYYDDD.zarr
  time[time]                           datetime64 slot start, every time_step_minutes
  channel[channel], label[label]       names in config order (channels with "output": false excluded)
  features[time, channel, y, x]        float16 (or float32); NaN = missing. One chunk per time step (or
                                       chunk_yx), compressed with Blosc features.compression
  labels[time, label, y, x]            uint8 1 = lightning in the window, 0 = none, 255 = unknown (GLM covered
                                       less than glm.min_coverage of the window)
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
  lags and gap filling, lookahead for labels) are complete. A neighbour outside the raw config's date range
  never will be, so it counts as unavailable instead: its slots are NaN / unknown. Incomplete days are
  rebuilt from scratch on the next run.
  channel_stats / label_stats: per-day sums for normalisation and class balance (combine with --stats).
  grid, fine_grid (with labels_fine), channels, helper_channels, labels, glm (with the footprint method),
  time_step_minutes, geography_path: how the store was built.

The geography store (<output_root_dir>/geography.zarr, one for all days) is described in geography.py.
"""
