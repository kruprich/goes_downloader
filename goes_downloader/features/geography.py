"""
Geography: fields that don't change with time (terrain, land/water, viewing angle, country borders), put on the feature grid once.

  python build_geography.py --config config.json

Writes <features.output_root_dir>/<features.geography.path> (default geography.zarr) from the first raw
day that defines the grid. Lat/lon sources are downloaded once into features.geography.cache_dir and kept;
OPeNDAP URLs (.../dodsC/...) are read for the grid's area only. A source finer than the cells is aggregated
over the source pixels whose centres fall in each cell; a coarser one is interpolated bilinearly. Rebuild
(--overwrite) only when the grid or the fields change; build_features.py checks that its grid matches.

Borders fields come from a GeoJSON file of country polygons (Natural Earth's admin-0 countries, keyed by
ADM0_A3). The polygons are rasterized at BORDERS_RESOLUTION_DEG (even-odd rule, so holes such as the
large lakes stay water) and aggregated like any fine source: "fraction" is the share of the cell on the
chosen countries' land; "distance_km" is the great-circle distance from the cell centre to the nearest
centre of a cell at least half on that land (0 for those cells). "within" keeps only polygons with a
vertex in a lat/lon box, e.g. the contiguous US without Alaska and Hawaii.

Layout of geography.zarr
  <field>[y, x]      float32 one array per configured field, in config order (NaN off the Earth's disk)
  x[x], y[y]         float64 cell-centre scan angles (rad), the same as the feature stores'
  lat[y, x], lon[y, x] float32 cell centres (deg)
  attrs: fields (names, in order), definitions (as configured), stats (count, sum, sum_sq, min, max per
  field, for normalisation), grid (the TargetGrid), built_utc
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import zarr

from ..crop import latlon_to_scan_angle
from ..progress import log
from ..store import COMPRESSOR, _write_array
from .grid import TargetGrid, grid_latlon, satellite_zenith_deg
from .settings import FeatureConfig, GeographySpec, geography_store_path

BORDERS_RESOLUTION_DEG = 0.01  # the raster that country polygons are drawn on before aggregating to cells
EARTH_RADIUS_KM = 6371.0


def _latlon_names(ds: Any) -> Tuple[str, str]:
    names = set(ds.variables)
    lat = next((n for n in ("lat", "latitude") if n in names), None)
    lon = next((n for n in ("lon", "longitude") if n in names), None)
    if lat is None or lon is None:
        raise ValueError(f"no lat/lon (or latitude/longitude) variables among {sorted(names)}")
    return lat, lon


def _box_slices(lat: np.ndarray, lon: np.ndarray, box: Tuple[float, float, float, float]) -> Tuple[slice, slice]:
    """Rows and columns of a regular lat/lon grid covering box = (lat_min, lat_max, lon_min, lon_max)."""
    lat_min, lat_max, lon_min, lon_max = box
    if float(np.nanmax(lon)) > 180:  # 0..360 source
        lon_min, lon_max = lon_min % 360, lon_max % 360
        if lon_min > lon_max:
            raise ValueError("the grid crosses the source's 0/360 longitude seam, which is not supported")
    rows = np.nonzero((lat >= lat_min) & (lat <= lat_max))[0]
    cols = np.nonzero((lon >= lon_min) & (lon <= lon_max))[0]
    if rows.size < 2 or cols.size < 2:
        raise ValueError(f"the source does not cover lat {lat_min:.1f}..{lat_max:.1f}, lon {box[2]:.1f}..{box[3]:.1f}")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def source_path(cfg: FeatureConfig, spec: GeographySpec) -> str:
    if spec.file:
        return spec.file
    name = os.path.basename(urllib.parse.urlparse(spec.url).path)
    name = name[:-3] if name.endswith(".gz") else name
    if "/dodsC/" in spec.url:  # OPeNDAP: only a subset is saved
        name = os.path.splitext(name)[0] + "_subset.nc"
    return os.path.join(cfg.geography_cache_dir, name)


def fetch_source(url: str, path: str, variable: str, box: Tuple[float, float, float, float]) -> None:
    """
    Saves a lat/lon source locally: OPeNDAP URLs (".../dodsC/...") are read over the network for the
    box only and saved as NetCDF; anything else is downloaded whole (and gunzipped if it ends in .gz).
    """
    import netCDF4 as nc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    log(f"downloading geography source {url}")
    if "/dodsC/" in url:
        with nc.Dataset(url) as src:
            lat_name, lon_name = _latlon_names(src)
            lat, lon = src[lat_name][:], src[lon_name][:]
            rows, cols = _box_slices(np.asarray(lat), np.asarray(lon), box)
            values = np.ma.filled(np.ma.asarray(src[variable][rows, cols], dtype=np.float32), np.nan)
            with nc.Dataset(tmp, "w") as out:
                out.source_url = url
                out.createDimension("lat", values.shape[0])
                out.createDimension("lon", values.shape[1])
                out.createVariable("lat", "f8", ("lat",))[:] = lat[rows]
                out.createVariable("lon", "f8", ("lon",))[:] = lon[cols]
                var = out.createVariable(variable, "f4", ("lat", "lon"), zlib=True, fill_value=np.float32(np.nan))
                with warnings.catch_warnings():  # netCDF4 sets an array's shape, which NumPy >= 2.5 deprecates
                    warnings.simplefilter("ignore", DeprecationWarning)
                    var[:] = values
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "goes_downloader"})
        with urllib.request.urlopen(request, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(gzip.GzipFile(fileobj=r) if url.endswith(".gz") else r, f)
    os.replace(tmp, path)


def read_source(path: str, variable: str, box: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat, lon, values[lat, lon]) of a regular lat/lon source over the box; fill values become NaN."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        lat_name, lon_name = _latlon_names(ds)
        lat, lon = np.asarray(ds[lat_name][:], np.float64), np.asarray(ds[lon_name][:], np.float64)
        rows, cols = _box_slices(lat, lon, box)
        var = ds[variable]
        if var.dimensions == (lon_name, lat_name):
            values = var[cols, rows].T
        elif var.dimensions == (lat_name, lon_name):
            values = var[rows, cols]
        else:
            raise ValueError(f"{path}: {variable} has dimensions {var.dimensions}, expected ({lat_name}, {lon_name})")
        return lat[rows], lon[cols], np.ma.filled(np.ma.asarray(values, dtype=np.float32), np.nan)


def _covers(lat: np.ndarray, lon: np.ndarray, box: Tuple[float, float, float, float]) -> bool:
    lon180 = (lon + 180) % 360 - 180
    return lat.min() <= box[0] + 0.1 and lat.max() >= box[1] - 0.1 and lon180.min() <= box[2] + 0.1 and lon180.max() >= box[3] - 0.1


def _bilinear(lat: np.ndarray, lon: np.ndarray, values: np.ndarray, qlat: np.ndarray, qlon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a regular lat/lon grid, ignoring NaN corners; NaN outside the grid."""
    if lon.max() > 180:
        qlon = qlon % 360
    fi = (qlat - lat[0]) / (lat[1] - lat[0])
    fj = (qlon - lon[0]) / (lon[1] - lon[0])
    inside = (fi >= 0) & (fi <= len(lat) - 1) & (fj >= 0) & (fj <= len(lon) - 1)
    i0 = np.clip(np.floor(np.nan_to_num(fi)), 0, len(lat) - 2).astype(np.int64)
    j0 = np.clip(np.floor(np.nan_to_num(fj)), 0, len(lon) - 2).astype(np.int64)
    wi, wj = np.clip(np.nan_to_num(fi) - i0, 0, 1), np.clip(np.nan_to_num(fj) - j0, 0, 1)
    total = np.zeros(qlat.shape)
    weight = np.zeros(qlat.shape)
    for di, dj, w in ((0, 0, (1 - wi) * (1 - wj)), (0, 1, (1 - wi) * wj), (1, 0, wi * (1 - wj)), (1, 1, wi * wj)):
        v = values[i0 + di, j0 + dj]
        ok = np.isfinite(v)
        total += np.where(ok, v * w, 0.0)
        weight += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / weight
    out[~inside | (weight == 0)] = np.nan
    return out


def regrid_source(specs: List[GeographySpec], lat: np.ndarray, lon: np.ndarray, values: np.ndarray,
                  grid: TargetGrid, cell_lat: np.ndarray, cell_lon: np.ndarray) -> List[np.ndarray]:
    """
    Puts one lat/lon source on the grid for each spec (which may differ in aggregation, clip, scale).
    A source finer than the cells is aggregated over the source pixels whose centres fall in each cell;
    a coarser one is interpolated bilinearly at the cell centres.
    """
    n_cells = grid.ny * grid.nx
    lon180 = (lon + 180) % 360 - 180

    def transform(spec: GeographySpec, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float64) * spec.scale + spec.offset
        low, high = spec.clip
        return np.clip(v, -np.inf if low is None else low, np.inf if high is None else high)

    n_pixels = np.zeros(n_cells)
    acc = [dict(n=np.zeros(n_cells), s=np.zeros(n_cells), ss=np.zeros(n_cells),
                mn=np.full(n_cells, np.inf), mx=np.full(n_cells, -np.inf)) for _ in specs]
    chunk = max(1, 2_000_000 // len(lon))
    for r0 in range(0, len(lat), chunk):
        la, lo = np.meshgrid(lat[r0 : r0 + chunk], lon180, indexing="ij")
        x, y = latlon_to_scan_angle(la.ravel(), lo.ravel(), grid.projection)
        cells = grid.cells_of(np.asarray(x), np.asarray(y))
        inside = cells >= 0
        c = cells[inside]
        n_pixels += np.bincount(c, minlength=n_cells)
        raw = values[r0 : r0 + chunk].ravel()[inside]
        for spec, a in zip(specs, acc):
            v = transform(spec, raw)
            ok = np.isfinite(v)
            a["n"] += np.bincount(c[ok], minlength=n_cells)
            a["s"] += np.bincount(c[ok], weights=v[ok], minlength=n_cells)
            a["ss"] += np.bincount(c[ok], weights=v[ok] * v[ok], minlength=n_cells)
            if spec.aggregation in ("min", "max"):
                np.fmin.at(a["mn"], c[ok], v[ok]) if spec.aggregation == "min" else np.fmax.at(a["mx"], c[ok], v[ok])

    on_earth = np.isfinite(cell_lat).ravel()
    fine = (n_pixels[on_earth] > 0).mean() >= 0.99 if on_earth.any() else False
    out = []
    for spec, a in zip(specs, acc):
        interp = transform(spec, _bilinear(lat, lon, values, cell_lat.ravel().astype(np.float64), cell_lon.ravel().astype(np.float64)))
        if not fine:
            if spec.aggregation != "mean":
                raise ValueError(f"geography field {spec.name!r}: the source is coarser than the grid, so only aggregation \"mean\" (interpolation) works")
            field = interp
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = a["s"] / a["n"]
                field = {
                    "mean": mean,
                    "std": np.sqrt(np.maximum(a["ss"] / a["n"] - mean * mean, 0.0)),
                    "min": np.where(np.isfinite(a["mn"]), a["mn"], np.nan),
                    "max": np.where(np.isfinite(a["mx"]), a["mx"], np.nan),
                }[spec.aggregation]
            if spec.aggregation == "mean":  # the odd cell without a source pixel
                field = np.where(a["n"] > 0, field, interp)
        field = np.where(on_earth, field, np.nan)
        out.append(field.reshape(grid.ny, grid.nx).astype(np.float32))
    return out


def read_borders(path: str, countries: Any, within: Any) -> List[np.ndarray]:
    """Rings ([n, 2] lon/lat, closed) of the countries' polygons (every country if None), outer and holes alike."""
    with open(path, "r", encoding="utf-8") as f:
        features = json.load(f)["features"]
    rings = []
    for feature in features:
        if countries is not None and feature["properties"].get("ADM0_A3") not in countries:
            continue
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            parts = [np.asarray(r, dtype=np.float64)[:, :2] for r in polygon]
            if within is not None:
                lat_min, lat_max, lon_min, lon_max = within
                outer = parts[0]
                if not ((outer[:, 1] >= lat_min) & (outer[:, 1] <= lat_max) & (outer[:, 0] >= lon_min) & (outer[:, 0] <= lon_max)).any():
                    continue
            rings += parts
    return rings


def rasterize_rings(rings: List[np.ndarray], lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    uint8 [lat, lon], 1 where the pixel centre is inside the rings by the even-odd rule, on the regular
    ascending grid lat x lon: every edge's crossings with the pixel-centre rows, sorted, filled in pairs.
    """
    n_rows, n_cols = len(lat), len(lon)
    x1 = np.concatenate([r[:-1, 0] for r in rings]) if rings else np.zeros(0)
    y1 = np.concatenate([r[:-1, 1] for r in rings]) if rings else np.zeros(0)
    x2 = np.concatenate([r[1:, 0] for r in rings]) if rings else np.zeros(0)
    y2 = np.concatenate([r[1:, 1] for r in rings]) if rings else np.zeros(0)
    # Rows whose centre v satisfies min(y1, y2) <= v < max(y1, y2): the same half-open rule at every vertex,
    # so each closed ring crosses each row an even number of times
    first = np.searchsorted(lat, np.minimum(y1, y2), "left")
    stop = np.searchsorted(lat, np.maximum(y1, y2), "left")
    count = stop - first
    edge = np.repeat(np.arange(len(x1)), count)
    row = first[edge] + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
    v = lat[row]
    x = x1[edge] + (v - y1[edge]) * (x2[edge] - x1[edge]) / (y2[edge] - y1[edge])
    order = np.lexsort((x, row))
    row, x = row[order], x[order]
    if len(row) % 2 or np.any(row[0::2] != row[1::2]):
        raise ValueError("country polygons are not closed rings")
    start = np.searchsorted(lon, x[0::2], "left")  # pixels with start <= centre < end
    end = np.searchsorted(lon, x[1::2], "left")
    diff = np.zeros((n_rows, n_cols + 1), np.int32)
    np.add.at(diff, (row[0::2], start), 1)
    np.add.at(diff, (row[0::2], end), -1)
    return (np.cumsum(diff[:, :-1], axis=1) > 0).astype(np.uint8)


def distance_to_km(member: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from each cell centre to the nearest member cell's centre; 0 for members."""
    on_earth = np.isfinite(lat)
    member = member & on_earth
    out = np.full(lat.shape, np.nan, np.float32)
    if not member.any():
        return out
    padded = np.pad(member, 1)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge = member & ~inner  # the nearest member of any other cell is on the members' edge

    def unit(mask: np.ndarray) -> np.ndarray:
        phi, lam = np.radians(lat[mask].astype(np.float64)), np.radians(lon[mask].astype(np.float64))
        return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=1)

    targets = unit(edge).T
    others = on_earth & ~member
    points = unit(others)
    nearest = np.empty(len(points))
    for a in range(0, len(points), 2048):
        nearest[a : a + 2048] = (points[a : a + 2048] @ targets).max(axis=1)
    out[member] = 0.0
    out[others] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip((1 - nearest) / 2, 0, 1)))
    return out


def border_fields(specs: List[GeographySpec], path: str, grid: TargetGrid, cell_lat: np.ndarray, cell_lon: np.ndarray,
                  box: Tuple[float, float, float, float]) -> List[np.ndarray]:
    """The borders fields of one (file, countries, within): the land fraction per cell, and distances from it."""
    res = BORDERS_RESOLUTION_DEG
    lat = np.arange(box[0] + res / 2, box[1], res)
    lon = np.arange(box[2] + res / 2, box[3], res)
    inside = rasterize_rings(read_borders(path, specs[0].countries, specs[0].within), lat, lon)
    (fraction,) = regrid_source([GeographySpec("fraction", variable="inside")], lat, lon, inside.astype(np.float32), grid, cell_lat, cell_lon)
    distance = distance_to_km(fraction >= 0.5, cell_lat, cell_lon) if any(s.borders == "distance_km" for s in specs) else None
    return [fraction if s.borders == "fraction" else distance for s in specs]


def compute_fields(cfg: FeatureConfig, grid: TargetGrid) -> Dict[str, np.ndarray]:
    """Every configured geography field on the grid, downloading sources that aren't cached yet."""
    lat, lon = grid_latlon(grid)
    ok = np.isfinite(lat)
    box = (float(lat[ok].min()) - 1, float(lat[ok].max()) + 1, float(lon[ok].min()) - 1, float(lon[ok].max()) + 1)
    fields: Dict[str, np.ndarray] = {}
    by_source: Dict[Tuple[str, str], List[GeographySpec]] = {}
    by_borders: Dict[Tuple[str, Any, Any], List[GeographySpec]] = {}
    for spec in cfg.geography:
        if spec.borders:
            by_borders.setdefault((source_path(cfg, spec), spec.countries, spec.within), []).append(spec)
        elif spec.geometry == "satellite_zenith":
            fields[spec.name] = satellite_zenith_deg(lat, lon, grid.projection)
        elif spec.geometry in ("latitude", "longitude"):
            fields[spec.name] = (lat if spec.geometry == "latitude" else lon).astype(np.float32)
        else:
            by_source.setdefault((source_path(cfg, spec), spec.variable), []).append(spec)

    for (path, variable), specs in by_source.items():
        url = next((s.url for s in specs if s.url), None)
        if not os.path.exists(path):
            if url is None:
                raise ValueError(f"geography source {path} does not exist and no url is configured")
            fetch_source(url, path, variable, box)
        src_lat, src_lon, values = read_source(path, variable, box)
        if not _covers(src_lat, src_lon, box) and url is not None and "/dodsC/" in url:
            fetch_source(url, path, variable, box)  # a subset saved for a smaller grid
            src_lat, src_lon, values = read_source(path, variable, box)
        for spec, field in zip(specs, regrid_source(specs, src_lat, src_lon, values, grid, lat, lon)):
            fields[spec.name] = field

    for (path, _, _), specs in by_borders.items():
        if not os.path.exists(path):
            url = next((s.url for s in specs if s.url), None)
            if url is None:
                raise ValueError(f"geography source {path} does not exist and no url is configured")
            fetch_source(url, path, "", box)
        for spec, field in zip(specs, border_fields(specs, path, grid, lat, lon, box)):
            fields[spec.name] = field
    return {s.name: fields[s.name] for s in cfg.geography}


def field_stats(v: np.ndarray) -> Dict[str, Any]:
    f = v[np.isfinite(v)].astype(np.float64)
    if not f.size:
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None}
    return {"count": int(f.size), "sum": float(f.sum()), "sum_sq": float(np.dot(f, f)), "min": float(f.min()), "max": float(f.max())}


def definition(f: GeographySpec) -> Dict[str, Any]:
    """A field's definition in config.json terms."""
    if f.geometry:
        d: Dict[str, Any] = {"name": f.name, "geometry": f.geometry}
    elif f.borders:
        d = {"name": f.name, "borders": f.borders, "countries": None if f.countries is None else list(f.countries),
             "within": None if f.within is None else list(f.within), "url": f.url, "file": f.file}
        d = {k: v for k, v in d.items() if v is not None or k == "countries"}
    else:
        d = {"name": f.name, "url": f.url, "file": f.file, "variable": f.variable, "aggregation": f.aggregation,
             "clip": list(f.clip), "scale": f.scale, "offset": f.offset}
        d = {k: v for k, v in d.items() if v is not None}
    return {**d, "description": f.description} if f.description else d


def build_geography(cfg: FeatureConfig, grid: TargetGrid) -> str:
    """Writes the geography store (built at <path>.tmp and moved into place)."""
    if not cfg.geography:
        raise ValueError("features.geography.fields is empty")
    fields = compute_fields(cfg, grid)
    lat, lon = grid_latlon(grid)
    path = geography_store_path(cfg)
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    root = zarr.open_group(zarr.DirectoryStore(tmp), mode="w")
    for name, v in fields.items():
        _write_array(root, name, v.astype(np.float32), ["y", "x"], coordinates="lat lon", zarr=dict(compressor=COMPRESSOR))
    _write_array(root, "x", grid.x(), ["x"], units="rad", long_name="ABI fixed grid E/W scan angle")
    _write_array(root, "y", grid.y(), ["y"], units="rad", long_name="ABI fixed grid N/S scan angle")
    _write_array(root, "lat", lat, ["y", "x"], units="degrees_north")
    _write_array(root, "lon", lon, ["y", "x"], units="degrees_east")
    root.attrs.update({
        "fields": list(fields),
        "definitions": [definition(f) for f in cfg.geography],
        "stats": {name: field_stats(v) for name, v in fields.items()},
        "grid": asdict(grid),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    })
    zarr.consolidate_metadata(root.store)
    shutil.rmtree(path, ignore_errors=True)
    shutil.move(tmp, path)
    return path


def geography_grid(path: str) -> Any:
    """The TargetGrid a geography store was built on, or None if there is no store."""
    if not os.path.exists(os.path.join(path, ".zgroup")):
        return None
    return TargetGrid(**zarr.open_consolidated(path, mode="r").attrs["grid"])
