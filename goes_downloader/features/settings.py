"""The "features" section of config.json: grid, time step, channels, labels, GLM and geography settings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..common import SLOT_SECONDS, SLOTS_PER_DAY
from ..config import LatLonBounds, RawConfig, _require, load_config

SLOT_MINUTES = SLOT_SECONDS // 60

AGGREGATIONS = ("mean", "min", "max", "std", "fraction")
# count/energy_fj/area_km2: flash centroids; extent: flashes whose events (footprint) touched the cell
GLM_STATS = ("count", "extent", "energy_fj", "area_km2")
LABEL_SOURCES = ("extent", "centroid")
SOLAR_VALUES = ("cos_zenith", "local_hour_sin", "local_hour_cos")
GEOMETRY_VALUES = ("satellite_zenith", "latitude", "longitude")
GEOGRAPHY_AGGREGATIONS = ("mean", "std", "min", "max")
BORDERS_VALUES = ("fraction", "distance_km")


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: str  # "product", "glm", "solar" or "expr"
    output: bool = True
    description: str = ""
    # kind == "product"
    product: Optional[str] = None
    aggregation: str = "mean"
    values: Tuple[float, ...] = ()
    valid_dqf: Optional[Tuple[int, ...]] = (0,)
    min_valid_fraction: float = 0.5
    fill_gaps_slots: int = 0
    # kind == "glm"
    glm_stat: Optional[str] = None
    # kind == "solar"
    solar: Optional[str] = None
    # kind == "expr"
    expr: Optional[str] = None


@dataclass(frozen=True)
class LabelSpec:
    name: str
    start_slots: int
    end_slots: int
    min_flashes: int
    radius_cells: float
    source: str = "extent"
    description: str = ""


@dataclass(frozen=True)
class GeographySpec:
    """
    A field that doesn't change with time: computed from the grid, regridded from a lat/lon file, or from
    country borders (a GeoJSON file of country polygons such as Natural Earth's).
    """

    name: str
    description: str = ""
    geometry: Optional[str] = None
    borders: Optional[str] = None  # "fraction" of the cell on these countries' land, or "distance_km" to it
    countries: Optional[Tuple[str, ...]] = None  # ADM0_A3 codes; None = every country
    within: Optional[Tuple[float, float, float, float]] = None  # keep polygons with a vertex in (lat_min, lat_max, lon_min, lon_max)
    url: Optional[str] = None
    file: Optional[str] = None
    variable: Optional[str] = None
    aggregation: str = "mean"
    clip: Tuple[Optional[float], Optional[float]] = (None, None)
    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class FeatureConfig:
    raw: RawConfig
    output_root_dir: str
    start_date_utc: str
    end_date_utc: str
    overwrite_existing: bool
    max_workers: int
    resolution_km: float
    bounds: Optional[LatLonBounds]
    reference_product: Optional[str]
    time_step_slots: int
    dtype: str
    chunk_yx: Tuple[int, int]
    glm_quality_flags: Optional[Tuple[int, ...]]
    glm_ellipsoid_km: Optional[Tuple[float, float]]  # (equatorial, polar) heights above the ground ellipsoid
    glm_min_coverage: float
    channels: Tuple[ChannelSpec, ...]
    labels: Tuple[LabelSpec, ...]
    geography: Tuple[GeographySpec, ...]
    geography_path: str
    geography_cache_dir: str

    @property
    def uses_glm(self) -> bool:
        return self.raw.glm_product is not None

    @property
    def uses_events(self) -> bool:
        """Whether flash footprints (GLM events) are needed; flash_extent is stored whenever GLM is."""
        return self.uses_glm


CHANNEL_KEYS = {
    "product": {"product", "aggregation", "values", "valid_dqf", "min_valid_fraction", "fill_gaps_minutes"},
    "glm": {"glm"},
    "solar": {"solar"},
    "expr": {"expr"},
}
LABEL_KEYS = {"name", "description", "start_minutes", "end_minutes", "min_flashes", "radius_cells", "source"}
GEOGRAPHY_FIELD_KEYS = {"name", "description", "geometry", "url", "file", "variable", "aggregation", "clip", "scale", "offset",
                        "borders", "countries", "within"}
FEATURE_KEYS = {"output_root_dir", "start_date_utc", "end_date_utc", "overwrite_existing", "max_workers", "grid",
                "time_step_minutes", "dtype", "chunk_yx", "glm", "channels", "labels", "geography"}


def minutes_to_slots(v: Any, what: str, positive: bool = False) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v % SLOT_MINUTES or (positive and v == 0):
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"{what} must be a {kind} multiple of {SLOT_MINUTES} minutes, got {v!r}")
    return int(v) // SLOT_MINUTES


def _fraction(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        raise ValueError(f"{what} must be a number between 0 and 1, got {v!r}")
    return float(v)


def _int_list(v: Any, what: str) -> Tuple[int, ...]:
    if not isinstance(v, list) or not v or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in v):
        raise ValueError(f"{what} must be a non-empty list of non-negative integers or null, got {v!r}")
    return tuple(v)


def _unknown(d: Dict[str, Any], known: set, where: str) -> None:
    unknown = sorted(k for k in d if k not in known and not str(k).startswith("_"))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")


def parse_channel(i: int, c: Any, abi_keys: List[str], has_glm: bool) -> ChannelSpec:
    from .expressions import FUNCTION_NAMES, parse_expression

    where = f"features.channels[{i}]"
    if not isinstance(c, dict):
        raise ValueError(f"{where} must be an object")
    name = c.get("name")
    if not isinstance(name, str) or not name.isidentifier() or name in FUNCTION_NAMES:
        raise ValueError(f"{where}.name must be an identifier (letters, digits, _) that is not a function name, got {name!r}")
    where = f"channel {name!r}"
    kinds = [k for k in CHANNEL_KEYS if k in c]
    if len(kinds) != 1:
        raise ValueError(f"{where} needs exactly one of " + ", ".join(f'"{k}"' for k in CHANNEL_KEYS))
    kind = kinds[0]
    _unknown(c, {"name", "output", "description"} | CHANNEL_KEYS[kind], f"{where} ({kind} channel)")
    common = dict(name=name, kind=kind, output=bool(c.get("output", True)), description=str(c.get("description", "")))

    if kind == "product":
        product = c["product"]
        if product not in abi_keys:
            raise ValueError(f"{where}: product {product!r} is not in the download list (downloaded ABI products: {abi_keys})")
        aggregation = c.get("aggregation", "mean")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"{where}: aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        values = c.get("values")
        if (aggregation == "fraction") != (values is not None):
            raise ValueError(f"{where}: \"values\" (the categories to count) goes with aggregation \"fraction\" only")
        if values is not None and (not isinstance(values, list) or not values or not all(isinstance(v, (int, float)) for v in values)):
            raise ValueError(f"{where}: values must be a non-empty list of numbers")
        valid_dqf = c.get("valid_dqf", [0])
        return ChannelSpec(
            **common, product=product, aggregation=aggregation, values=tuple(float(v) for v in values or ()),
            valid_dqf=None if valid_dqf is None else _int_list(valid_dqf, f"{where}: valid_dqf"),
            min_valid_fraction=_fraction(c.get("min_valid_fraction", 0.5), f"{where}: min_valid_fraction"),
            fill_gaps_slots=minutes_to_slots(c.get("fill_gaps_minutes", 0), f"{where}: fill_gaps_minutes"),
        )
    if kind == "glm":
        if not has_glm:
            raise ValueError(f"{where}: GLM channels need \"glm\" in the download list")
        if c["glm"] not in GLM_STATS:
            raise ValueError(f"{where}: glm must be one of {GLM_STATS}, got {c['glm']!r}")
        return ChannelSpec(**common, glm_stat=c["glm"])
    if kind == "solar":
        if c["solar"] not in SOLAR_VALUES:
            raise ValueError(f"{where}: solar must be one of {SOLAR_VALUES}, got {c['solar']!r}")
        return ChannelSpec(**common, solar=c["solar"])
    if not isinstance(c["expr"], str):
        raise ValueError(f"{where}: expr must be a string")
    parse_expression(c["expr"], where)
    return ChannelSpec(**common, expr=c["expr"])


def parse_label(i: int, l: Any) -> LabelSpec:
    where = f"features.labels[{i}]"
    if not isinstance(l, dict):
        raise ValueError(f"{where} must be an object")
    _unknown(l, LABEL_KEYS, where)
    name = l.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}.name must be a non-empty string")
    where = f"label {name!r}"
    start = minutes_to_slots(l.get("start_minutes", 0), f"{where}: start_minutes")
    end = minutes_to_slots(l.get("end_minutes"), f"{where}: end_minutes", positive=True)
    if end <= start:
        raise ValueError(f"{where}: end_minutes must be greater than start_minutes")
    min_flashes = l.get("min_flashes", 1)
    if not isinstance(min_flashes, int) or isinstance(min_flashes, bool) or min_flashes < 1:
        raise ValueError(f"{where}: min_flashes must be a positive integer")
    radius = l.get("radius_cells", 0)
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius < 0:
        raise ValueError(f"{where}: radius_cells must be a non-negative number")
    source = l.get("source", "extent")
    if source not in LABEL_SOURCES:
        raise ValueError(f"{where}: source must be one of {LABEL_SOURCES}, got {source!r}")
    return LabelSpec(name, start, end, min_flashes, float(radius), source, str(l.get("description", "")))


def parse_geography_field(i: int, f: Any) -> GeographySpec:
    where = f"features.geography.fields[{i}]"
    if not isinstance(f, dict):
        raise ValueError(f"{where} must be an object")
    _unknown(f, GEOGRAPHY_FIELD_KEYS, where)
    name = f.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{where}.name must be an identifier, got {name!r}")
    where = f"geography field {name!r}"
    description = str(f.get("description", ""))
    if "geometry" in f:
        extra = set(f) - {"name", "description", "geometry"}
        if extra:
            raise ValueError(f"{where}: a geometry field takes no {sorted(extra)}")
        if f["geometry"] not in GEOMETRY_VALUES:
            raise ValueError(f"{where}: geometry must be one of {GEOMETRY_VALUES}, got {f['geometry']!r}")
        return GeographySpec(name, description, geometry=f["geometry"])
    if "borders" in f:
        extra = set(f) - {"name", "description", "borders", "countries", "within", "url", "file"}
        if extra:
            raise ValueError(f"{where}: a borders field takes no {sorted(extra)}")
        if f["borders"] not in BORDERS_VALUES:
            raise ValueError(f"{where}: borders must be one of {BORDERS_VALUES}, got {f['borders']!r}")
        if not (f.get("url") or f.get("file")):
            raise ValueError(f"{where}: a borders field needs the \"url\" and/or \"file\" of a GeoJSON file of country polygons")
        countries = f.get("countries")
        if countries is not None and (not isinstance(countries, list) or not countries or not all(isinstance(c, str) and c for c in countries)):
            raise ValueError(f"{where}: countries must be a non-empty list of ADM0_A3 codes (e.g. [\"USA\"]) or null for every country")
        within = f.get("within")
        if within is not None and not (isinstance(within, list) and len(within) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in within)
                                       and within[0] < within[1] and within[2] < within[3]):
            raise ValueError(f"{where}: within must be [lat_min, lat_max, lon_min, lon_max]")
        if f["borders"] == "distance_km" and countries is None:
            raise ValueError(f"{where}: distance_km needs countries")
        return GeographySpec(name, description, borders=f["borders"], countries=None if countries is None else tuple(countries),
                             within=None if within is None else tuple(float(v) for v in within), url=f.get("url"), file=f.get("file"))
    if not (f.get("url") or f.get("file")) or not isinstance(f.get("variable"), str):
        raise ValueError(f"{where} needs \"geometry\", \"borders\", or a \"variable\" with a \"url\" and/or \"file\"")
    aggregation = f.get("aggregation", "mean")
    if aggregation not in GEOGRAPHY_AGGREGATIONS:
        raise ValueError(f"{where}: aggregation must be one of {GEOGRAPHY_AGGREGATIONS}, got {aggregation!r}")
    clip = f.get("clip", [None, None])
    if not (isinstance(clip, list) and len(clip) == 2 and all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in clip)):
        raise ValueError(f"{where}: clip must be [low, high] with numbers or null")
    for k in ("scale", "offset"):
        if k in f and (isinstance(f[k], bool) or not isinstance(f[k], (int, float))):
            raise ValueError(f"{where}: {k} must be a number")
    return GeographySpec(
        name, description, url=f.get("url"), file=f.get("file"), variable=f["variable"], aggregation=aggregation,
        clip=tuple(None if v is None else float(v) for v in clip), scale=float(f.get("scale", 1.0)), offset=float(f.get("offset", 0.0)),
    )


def load_feature_config(config_path: str) -> FeatureConfig:
    """Reads the "features" section of the shared config.json; the raw settings come from load_config."""
    raw = load_config(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        section = json.load(f).get("features")
    if not isinstance(section, dict):
        raise ValueError(f"{config_path} has no \"features\" object")
    _unknown(section, FEATURE_KEYS, "features")

    abi_keys = [p.key for p in raw.abi_products]
    has_glm = raw.glm_product is not None
    channels = tuple(parse_channel(i, c, abi_keys, has_glm) for i, c in enumerate(_require(section, "channels", list)))
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise ValueError(f"channel names must be unique, got {names}")
    if not any(c.output for c in channels):
        raise ValueError("at least one channel must have output true")
    if not any(c.kind == "product" for c in channels):
        raise ValueError("at least one channel must read an ABI product (it defines the grid)")

    labels = tuple(parse_label(i, l) for i, l in enumerate(section.get("labels", [])))
    if labels and not has_glm:
        raise ValueError("labels need \"glm\" in the download list")
    if len({l.name for l in labels}) != len(labels):
        raise ValueError("label names must be unique")

    geo = section.get("geography", {})
    if not isinstance(geo, dict):
        raise ValueError("features.geography must be an object")
    _unknown(geo, {"path", "cache_dir", "fields"}, "features.geography")
    geography = tuple(parse_geography_field(i, f) for i, f in enumerate(geo.get("fields", [])))
    if len({g.name for g in geography}) != len(geography):
        raise ValueError("geography field names must be unique")

    grid = section.get("grid", {})
    _unknown(grid, {"resolution_km", "bounds", "reference_product"}, "features.grid")
    resolution_km = grid.get("resolution_km", 4)
    if isinstance(resolution_km, bool) or not isinstance(resolution_km, (int, float)) or resolution_km <= 0:
        raise ValueError("features.grid.resolution_km must be a positive number")
    reference_product = grid.get("reference_product")
    if reference_product is not None and reference_product not in abi_keys:
        raise ValueError(f"features.grid.reference_product {reference_product!r} is not a downloaded ABI product")
    # "bounds" absent -> the download bounds; null -> the whole area the raw stores hold
    bounds = raw.bounds
    if "bounds" in grid:
        b = grid["bounds"]
        bounds = None if b is None else LatLonBounds(**{k: float(_require(b, k, (int, float))) for k in ("lat_min", "lat_max", "lon_min", "lon_max")})

    step = minutes_to_slots(section.get("time_step_minutes", SLOT_MINUTES), "features.time_step_minutes", positive=True)
    if SLOTS_PER_DAY % step:
        raise ValueError("features.time_step_minutes must divide a day (5, 10, 15, 20, 30, 60, ...)")
    dtype = section.get("dtype", "float16")
    if dtype not in ("float16", "float32"):
        raise ValueError("features.dtype must be \"float16\" or \"float32\"")
    chunk_yx = section.get("chunk_yx", [256, 256])
    if not (isinstance(chunk_yx, list) and len(chunk_yx) == 2 and all(isinstance(v, int) and v > 0 for v in chunk_yx)):
        raise ValueError("features.chunk_yx must be two positive integers")

    glm = section.get("glm", {})
    _unknown(glm, {"quality_flags", "lightning_ellipsoid_km", "min_coverage"}, "features.glm")
    flags = glm.get("quality_flags")
    ellipsoid = glm.get("lightning_ellipsoid_km", {"equatorial": 14.0, "polar": 6.0})
    if ellipsoid is not None and not (isinstance(ellipsoid, dict) and set(ellipsoid) == {"equatorial", "polar"}):
        raise ValueError("features.glm.lightning_ellipsoid_km must be {\"equatorial\": km, \"polar\": km} or null")

    start = section.get("start_date_utc") or raw.start_date_utc
    end = section.get("end_date_utc") or raw.end_date_utc
    datetime.strptime(start, "%Y-%m-%d")
    datetime.strptime(end, "%Y-%m-%d")
    output_root = str(section.get("output_root_dir", "./goes_features"))

    cfg = FeatureConfig(
        raw=raw,
        output_root_dir=output_root,
        start_date_utc=start,
        end_date_utc=end,
        overwrite_existing=bool(section.get("overwrite_existing", False)),
        max_workers=max(1, int(section.get("max_workers", 4))),
        resolution_km=float(resolution_km),
        bounds=bounds,
        reference_product=reference_product,
        time_step_slots=step,
        dtype=dtype,
        chunk_yx=(chunk_yx[0], chunk_yx[1]),
        glm_quality_flags=None if flags is None else _int_list(flags, "features.glm.quality_flags"),
        glm_ellipsoid_km=None if ellipsoid is None else (float(ellipsoid["equatorial"]), float(ellipsoid["polar"])),
        glm_min_coverage=_fraction(glm.get("min_coverage", 0.95), "features.glm.min_coverage"),
        channels=channels,
        labels=labels,
        geography=geography,
        # A path relative to output_root_dir, so the geography travels with the features
        geography_path=str(geo.get("path", "geography.zarr")),
        geography_cache_dir=str(geo.get("cache_dir", "./goes_static")),
    )
    from .expressions import make_plan

    make_plan(cfg.channels, [l.end_slots for l in cfg.labels])  # checks expression references and cycles
    return cfg


def geography_store_path(cfg: FeatureConfig) -> str:
    return os.path.join(cfg.output_root_dir, cfg.geography_path)
