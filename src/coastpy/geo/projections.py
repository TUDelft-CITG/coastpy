from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy import float32
from pyproj import Transformer
from scipy.stats import norm
from shapely import ops
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform

from coastpy.geo.geoms import create_offset_rectangle
from coastpy.geo.ops import extend_transect


def normal_p5_p95(mean: float, std: float) -> tuple[float, float]:
    """
    Computes the 5th and 95th percentiles for a normal distribution.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.

    Returns:
        Tuple of (p5, p95) percentiles.
    """
    if np.isnan(std):
        return np.nan, np.nan
    p5 = norm.ppf(0.05, loc=mean, scale=std)
    p95 = norm.ppf(0.95, loc=mean, scale=std)
    return p5, p95


def project_trend(rate: float, years: float | list[float]) -> float | list[float]:
    """
    Computes projected shoreline change distance(s) based on a given trend rate.

    This function calculates the projected change in shoreline position over a
    specified number of years, using a constant rate of change. Negative rates
    indicate erosion (landward movement), while positive rates indicate accretion
    (seaward movement).

    Args:
        rate (float): The rate of shoreline change per year. Negative for erosion,
            positive for accretion.
        years (float | list[float]): The number of years (or a list of years) over
            which to project the shoreline change.

    Returns:
        float | list[float]: The projected shoreline change distance(s). If `years`
        is a single float, returns a single float. If `years` is a list, returns a
        list of projected distances corresponding to each year in the input list.
    """

    def compute_offset(y):
        return rate * y

    if isinstance(years, list):
        return [compute_offset(y) for y in years]
    return compute_offset(years)


def project_along_transect(
    row: pd.Series | gpd.GeoSeries,
    shoreline_change: float | list[float],
    reference_point: Point | tuple[float, float] | None = None,
    accommodation_buffer: float = 0.0,
    extend_to: float = 10_000.0,
) -> Point | list[Point] | None:
    """
    Projects a point or list of points along a transect.

    Args:
        row (dict): GeoDataFrame row containing 'geometry' (LineString) and 'utm_epsg' (int).
        shoreline_change (float | list[float]): Offset(s) in meters along the transect.
            Positive values indicate seaward movement, negative values indicate landward movement.
        reference_point (Point | tuple[float, float] | None): Optional reference point.
            Defaults to the row's reference coordinates if None.
        accommodation_buffer (float): Landward buffer distance in meters. Always applied as negative.
        extend_to (float): Minimum transect length in meters for projection.

    Returns:
        Point | list[Point] | None: Projected point(s) or None if projection fails.
    """
    transect: LineString = row["geometry"]  # type: ignore
    utm_epsg = row["utm_epsg"]

    if not isinstance(transect, LineString) or len(transect.coords) < 2:
        return None

    if reference_point is None:
        reference_point = Point(row["sds:reference_lon"], row["sds:reference_lat"])  # type: ignore
    elif isinstance(reference_point, tuple):
        reference_point = Point(reference_point)

    # Ensure accommodation_buffer is always negative
    accommodation_buffer = -abs(accommodation_buffer)

    # Create CRS transformers
    to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
    ).transform
    to_wgs = Transformer.from_crs(
        f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
    ).transform

    transect_utm = transform(to_utm, transect)
    ref_pt_utm = transform(to_utm, reference_point)

    transect_utm = extend_transect(transect_utm, extend_to, side="both")
    base_dist = transect_utm.project(ref_pt_utm)

    def compute_point(offset: float) -> Point:
        total_offset = offset + accommodation_buffer
        proj_dist = max(0, min(base_dist + total_offset, transect_utm.length))
        pt = transect_utm.interpolate(proj_dist)
        return transform(to_wgs, pt)

    if isinstance(shoreline_change, list):
        return [compute_point(offset) for offset in shoreline_change]
    return compute_point(shoreline_change)


def resolve_input(
    df: pd.DataFrame,
    val,
    name: str,
    default=None,
    cast=None,
    required: bool = False,
):
    if isinstance(val, str):
        if val not in df.columns:
            if required:
                raise ValueError(
                    f"[ERROR] Required column '{val}' not found for '{name}'"
                )
            print(f"[WARN] Column '{val}' not found. Using default for '{name}'")
            return [default] * len(df)

        series = df[val]

        if default is not None and not isinstance(default, list):
            series = series.fillna(default)

        if cast == "boolean":
            return series.astype("boolean").tolist()
        elif cast in [float, int, str]:
            return series.astype(cast).tolist() if cast else series.tolist()
        elif cast is list:
            return series.tolist()
        return series.tolist()

    elif isinstance(val, list):
        if len(val) != len(df):
            raise ValueError(
                f"[ERROR] Length of list for '{name}' does not match DataFrame length"
            )
        return val

    return [default] * len(df)


def apply_projection_from_values(
    df: gpd.GeoDataFrame,
    reference_lons: str | list[float],
    reference_lats: str | list[float],
    accommodation_buffer: float = 0.0,
    ambient_change_is_valid: str | list[bool] | None = None,
    ambient_change_mean: str | list[float] | None = None,
    ambient_change_std: str | list[float] | None = None,
    sea_level_rise_samples: str | list[list[float]] | None = None,
    total_change_samples: str | list[list[float]] | None = None,
) -> gpd.GeoDataFrame:
    """
    Projects future shoreline positions based on ambient and sea level rise change samples.

    Args:
        df: GeoDataFrame of transects with UTM geometries and metadata.
        reference_lons: Column or list of reference longitudes.
        reference_lats: Column or list of reference latitudes.
        accommodation_buffer: Optional landward buffer (meters).
        ambient_change_is_valid: Optional column or list of validity flags (True/False/<NA>).
        ambient_change_mean: Optional column or list of ambient change means (m).
        ambient_change_std: Optional column or list of ambient change std devs (m).
        sea_level_rise_samples: Optional column or list of sea level rise samples.
        total_change_samples: Optional column or list of total change samples (for diagnostics only).

    Returns:
        GeoDataFrame with projected geometries and change metadata.
    """
    lons = resolve_input(
        df, reference_lons, "reference_lons", required=True, cast=float
    )
    lats = resolve_input(
        df, reference_lats, "reference_lats", required=True, cast=float
    )
    is_valid = resolve_input(
        df,
        ambient_change_is_valid,
        "ambient_change_is_valid",
        default=pd.NA,
        cast="boolean",
    )
    ac_mean = resolve_input(
        df, ambient_change_mean, "ambient_change_mean", default=0.0, cast=float
    )
    ac_std = resolve_input(
        df, ambient_change_std, "ambient_change_std", default=np.nan, cast=float
    )

    slr_samples = resolve_input(
        df, sea_level_rise_samples, "sea_level_rise_samples", default=[], cast=list
    )
    total_samples = resolve_input(
        df, total_change_samples, "total_change_samples", default=[], cast=list
    )

    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        tid = row["transect_id"]
        lon, lat = lons[i], lats[i]
        ac_valid = is_valid[i]
        ac_mu = float(ac_mean[i])  # type: ignore
        ac_sigma = float(ac_std[i])  # type: ignore
        slr_vals = slr_samples[i]
        total_vals = total_samples[i]

        ac_p5, ac_p95 = normal_p5_p95(ac_mu, ac_sigma)

        # Handle sea level rise samples
        if isinstance(slr_vals, list | np.ndarray) and len(slr_vals) > 0:
            slr_arr = np.asarray(slr_vals, dtype="float32")
            slr_p5, slr_p50, slr_p95 = np.percentile(slr_arr, [5, 50, 95])
            slr_vals_out = slr_arr.tolist()
        else:
            slr_p5 = slr_p50 = slr_p95 = 0.0
            slr_vals_out = [0.0]

        # Handle total change samples (optional diagnostics)
        if isinstance(total_vals, list | np.ndarray) and len(total_vals) > 0:
            total_arr = np.asarray(total_vals, dtype="float32")
            total_p5, total_p50, total_p95 = np.percentile(total_arr, [5, 50, 95])
            total_vals_out = total_arr.tolist()
        else:
            total_p5 = total_p50 = total_p95 = np.nan
            total_vals_out = [np.nan]

        # Compute shoreline change (ac + slr)
        shoreline_change = ac_mu + slr_p50

        if lon is None or lat is None:
            raise ValueError(
                f"Invalid coordinates for transect {tid}: lon={lon}, lat={lat}"
            )
        ref_point = Point(float(lon), float(lat))

        geom = project_along_transect(
            row,
            shoreline_change=shoreline_change,
            reference_point=ref_point,
            accommodation_buffer=accommodation_buffer,
        )

        if geom is None:
            continue

        records.append(
            {
                "transect_id": tid,
                "ssp": pd.NA,
                "datetime": row.get("datetime", pd.NaT),
                "ambient_change_is_valid": ac_valid,
                "ambient_change_mean": np.float32(ac_mu),
                "ambient_change_std": np.float32(ac_sigma),
                "ambient_change_p5": np.float32(ac_p5),
                "ambient_change_p95": np.float32(ac_p95),
                "sea_level_rise_samples": slr_vals_out,
                "sea_level_rise_p5": np.float32(slr_p5),
                "sea_level_rise_p50": np.float32(slr_p50),
                "sea_level_rise_p95": np.float32(slr_p95),
                "total_change_samples": total_vals_out,
                "total_change_p5": np.float32(total_p5),
                "total_change_p50": np.float32(total_p50),
                "total_change_p95": np.float32(total_p95),
                "accommodation_buffer": np.float32(accommodation_buffer),
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=4326).astype(
        {
            "ambient_change_mean": "float32",
            "ambient_change_std": "float32",
            "ambient_change_p5": "float32",
            "ambient_change_p95": "float32",
            "sea_level_rise_p5": "float32",
            "sea_level_rise_p50": "float32",
            "sea_level_rise_p95": "float32",
            "total_change_p5": "float32",
            "total_change_p50": "float32",
            "total_change_p95": "float32",
            "accommodation_buffer": "float32",
        }
    )


def add_building_count(
    areas_at_risk_of_erosion: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Annotate areas at risk of coastal erosion polygons with the count of intersecting buildings.

    - If an area has no geometry, the count is NaN.
    - If it has geometry but no intersecting buildings, the count is 0.
    - Otherwise, the count reflects the number of intersections.

    Args:
        areas_at_risk_of_erosion (gpd.GeoDataFrame): Polygons representing areas of interest.
        buildings (gpd.GeoDataFrame): Building footprints as polygons.

    Returns:
        gpd.GeoDataFrame: Copy of areas_at_risk_of_erosion with an added 'building_count' column.
    """
    if areas_at_risk_of_erosion.crs != buildings.crs:
        if areas_at_risk_of_erosion.crs is not None:
            buildings = buildings.to_crs(areas_at_risk_of_erosion.crs)
        else:
            raise ValueError(
                "CRS of areas_at_risk_of_erosion is None. Cannot reproject building_gdf."
            )

    valid_areas = areas_at_risk_of_erosion[
        areas_at_risk_of_erosion.geometry.notnull()
    ].copy()
    missing_geom = areas_at_risk_of_erosion[
        areas_at_risk_of_erosion.geometry.isnull()
    ].copy()

    valid_areas = valid_areas.reset_index(drop=True)
    valid_areas["__area_id"] = valid_areas.index

    joined = gpd.sjoin(
        buildings,
        valid_areas[["__area_id", "geometry"]],
        how="left",
        predicate="intersects",
    )

    counts = joined.groupby("__area_id").size().rename("building_count")
    valid_areas = valid_areas.join(counts, on="__area_id")
    valid_areas.drop(columns="__area_id", inplace=True)
    valid_areas["building_count"] = valid_areas["building_count"].fillna(0).astype(int)

    missing_geom["building_count"] = pd.NA

    return gpd.GeoDataFrame(
        pd.concat([valid_areas, missing_geom], axis=0).sort_index(),
        geometry="geometry",
        crs=areas_at_risk_of_erosion.crs,
    )


######################################################## @ JUUL ##############################
############### FROM HERE ITS OLD CODE` ##############################


def apply_projection_trend(
    df: gpd.GeoDataFrame,
    target_datetimes: list[datetime],
    reference_lons: str | list[float],
    reference_lats: str | list[float],
    reference_datetimes: str | list[datetime],
    accommodation_buffer: float = 100.0,
) -> gpd.GeoDataFrame:
    """
    Projects transects forward in time using change rate trends and adds optional landward buffer.

    Args:
        df: GeoDataFrame containing transect geometries and metadata.
        target_datetimes: List of datetime targets to project to.
        reference_lons: Column name (str) or list of longitudes (floats).
        reference_lats: Column name (str) or list of latitudes (floats).
        reference_datetimes: Column name (str) or list of datetime objects.
        accommodation_buffer: Buffer distance in meters applied in landward direction.

    Returns:
        GeoDataFrame with projected point geometries and structured shoreline change fields.
    """

    def resolve(ref: str | list, name: str):
        if isinstance(ref, str):
            if ref not in df.columns:
                raise ValueError(f"Column '{ref}' not found in DataFrame for '{name}'")
            return df[ref].tolist()
        if isinstance(ref, list):
            if len(ref) != len(df):
                raise ValueError(f"Length of list '{name}' must match DataFrame length")
            return ref
        raise TypeError(f"Argument '{name}' must be a string or a list")

    lons = resolve(reference_lons, "reference_lons")
    lats = resolve(reference_lats, "reference_lats")
    ref_dts = resolve(reference_datetimes, "reference_datetimes")

    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        tid = row["transect_id"]
        rate = row["sds:change_rate"]
        ref_dt = ref_dts[i]

        if pd.isnull(ref_dt):
            print(f"[WARN] Missing reference_datetime for transect {tid}. Skipping.")
            continue

        year_deltas = [target.year - ref_dt.year for target in target_datetimes]
        ac_values = [float(rate * delta) for delta in year_deltas]

        ref_point = Point(lons[i], lats[i])
        projected_geoms = project_along_transect(
            row,
            shoreline_change=ac_values,
            reference_point=ref_point,
            accommodation_buffer=accommodation_buffer,
        )

        if not projected_geoms:
            continue

        for j, (target_dt, geom) in enumerate(
            zip(target_datetimes, projected_geoms, strict=False)
        ):
            if geom:
                ac_val = ac_values[j]

                records.append(
                    {
                        "transect_id": tid,
                        "ssp": pd.NA,
                        "datetime": target_dt,
                        "ambient_change_is_valid": pd.NA,
                        "ambient_change_mean": float32(ac_val),
                        "ambient_change_std": float32("nan"),
                        "ambient_change_p5": float32("nan"),
                        "ambient_change_p95": float32("nan"),
                        "ambient_change_samples": [float32(ac_val)],
                        "sea_level_rise_samples": None,
                        "sea_level_rise_p5": float32("nan"),
                        "sea_level_rise_p50": float32("nan"),
                        "sea_level_rise_p95": float32("nan"),
                        "total_change_samples": [float32(ac_val)],
                        "total_change_p5": float32("nan"),
                        "total_change_p50": float32("nan"),
                        "total_change_p95": float32("nan"),
                        "geometry": geom,
                    }
                )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).astype(
        {
            "ambient_change_mean": "float32",
            "ambient_change_std": "float32",
            "ambient_change_p5": "float32",
            "ambient_change_p95": "float32",
            "sea_level_rise_p5": "float32",
            "sea_level_rise_p50": "float32",
            "sea_level_rise_p95": "float32",
            "total_change_p5": "float32",
            "total_change_p50": "float32",
            "total_change_p95": "float32",
        }
    )
    return gdf


def compute_rectangle_projection(
    row: dict,
    shoreline_change: float,
    accommodation_buffer: float = 100.0,
    reference_point: Point | tuple[float, float] | None = None,
    extend_to: float = 10_000.0,
    alongshore_buffer: float = 50.0,
) -> Polygon | None:
    """
    Projects a rectangle polygon along a transect using shoreline change and optional landward buffer.

    Args:
        row: A row from a GeoDataFrame, must contain 'geometry' and 'utm_epsg'.
        shoreline_change: Offset in meters along the transect (positive = seaward, negative = landward).
        accommodation_buffer: Landward buffer distance in meters (applied as -abs(buffer)).
        reference_point: Optional reference point. If None, uses row's reference lon/lat.
        extend_to: Minimum transect length in meters for safe projection.
        alongshore_buffer: Width of the projected rectangle alongshore (perpendicular to transect).

    Returns:
        Polygon geometry of the projected rectangle, or None on failure.
    """
    try:
        transect: LineString = row["geometry"]
        utm_epsg: int = row["utm_epsg"]

        if not isinstance(transect, LineString) or len(transect.coords) < 2:
            return None

        if reference_point is None:
            reference_point = Point(row["sds:reference_lon"], row["sds:reference_lat"])
        elif isinstance(reference_point, tuple):
            reference_point = Point(reference_point)

        # Setup reprojection
        to_utm = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
        ).transform
        to_wgs = Transformer.from_crs(
            f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
        ).transform

        transect_utm = LineString([to_utm(*pt) for pt in transect.coords])
        ref_pt_utm = Point(to_utm(*reference_point.coords[0]))

        transect_utm = extend_transect(transect_utm, extend_to, side="both")
        base_dist = transect_utm.project(ref_pt_utm)
        proj_dist = base_dist + shoreline_change - abs(accommodation_buffer)

        # Clamp to transect bounds
        proj_dist = max(0, min(proj_dist, transect_utm.length))

        # Define segment between reference and projected point
        start = transect_utm.interpolate(base_dist)
        end = transect_utm.interpolate(proj_dist)
        segment = LineString([start, end])

        rectangle = create_offset_rectangle(segment, distance=alongshore_buffer)
        return ops.transform(to_wgs, rectangle)

    except Exception as e:
        print(
            f"[WARN] Rectangle projection failed for {row.get('transect_id', 'unknown')}: {e}"
        )
        return None


def apply_rectangle_projection_trend(
    df: gpd.GeoDataFrame,
    target_datetimes: list[datetime],
    reference_lons: str | list[float],
    reference_lats: str | list[float],
    reference_datetimes: str | list[datetime],
    accommodation_buffer: float = 100.0,
    alongshore_buffer: float = 50.0,
) -> gpd.GeoDataFrame:
    """
    Projects rectangle polygons forward in time along transects using trend rates,
    and produces schema-compliant outputs aligned with future shoreline change schema.

    Args:
        df: GeoDataFrame containing transects and metadata.
        target_datetimes: List of future datetimes to project to.
        reference_lons: Column name (str) or list of longitude values.
        reference_lats: Column name (str) or list of latitude values.
        reference_datetimes: Column name (str) or list of datetime objects.
        accommodation_buffer: Landward buffer distance in meters (applied as -abs(buffer)).
        alongshore_buffer: Width of the rectangle in meters, alongshore.

    Returns:
        GeoDataFrame with projected POLYGON geometries and structured shoreline change outputs.
    """

    def resolve(ref: str | list, name: str):
        if isinstance(ref, str):
            if ref not in df.columns:
                raise ValueError(f"Column '{ref}' not found in DataFrame for '{name}'")
            return df[ref].tolist()
        if isinstance(ref, list):
            if len(ref) != len(df):
                raise ValueError(f"Length of list '{name}' must match DataFrame length")
            return ref
        raise TypeError(f"Argument '{name}' must be a string or a list")

    lons = resolve(reference_lons, "reference_lons")
    lats = resolve(reference_lats, "reference_lats")
    ref_dts = resolve(reference_datetimes, "reference_datetimes")

    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        tid = row["transect_id"]
        rate = row["sds:change_rate"]
        ref_dt = ref_dts[i]

        if pd.isnull(ref_dt):
            print(f"[WARN] Missing reference_datetime for transect {tid}. Skipping.")
            continue

        year_deltas = [target.year - ref_dt.year for target in target_datetimes]
        ac_values = [float(rate * delta) for delta in year_deltas]
        ref_point = Point(lons[i], lats[i])

        for j, target_dt in enumerate(target_datetimes):
            geom = compute_rectangle_projection(
                row,
                shoreline_change=ac_values[j],
                accommodation_buffer=accommodation_buffer,
                reference_point=ref_point,
                alongshore_buffer=alongshore_buffer,
            )
            if not geom:
                continue

            ac_val = ac_values[j]

            records.append(
                {
                    "transect_id": tid,
                    "ssp": pd.NA,
                    "datetime": target_dt,
                    "ambient_change_is_valid": pd.NA,
                    "ambient_change_mean": float32(ac_val),
                    "ambient_change_std": float32("nan"),
                    "ambient_change_p5": float32("nan"),
                    "ambient_change_p95": float32("nan"),
                    "ambient_change_samples": [float32(ac_val)],
                    "sea_level_rise_samples": None,
                    "sea_level_rise_p5": float32("nan"),
                    "sea_level_rise_p50": float32("nan"),
                    "sea_level_rise_p95": float32("nan"),
                    "total_change_samples": [float32(ac_val)],
                    "total_change_p5": float32("nan"),
                    "total_change_p50": float32("nan"),
                    "total_change_p95": float32("nan"),
                    "geometry": geom,
                }
            )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).astype(
        {
            "ambient_change_mean": "float32",
            "ambient_change_std": "float32",
            "ambient_change_p5": "float32",
            "ambient_change_p95": "float32",
            "sea_level_rise_p5": "float32",
            "sea_level_rise_p50": "float32",
            "sea_level_rise_p95": "float32",
            "total_change_p5": "float32",
            "total_change_p50": "float32",
            "total_change_p95": "float32",
        }
    )
    return gdf


def apply_projection_from_future_df(
    df: gpd.GeoDataFrame,
    futures: pd.DataFrame,
    reference_lons: str | list[float],
    reference_lats: str | list[float],
    accommodation_buffer: float = 100.0,
) -> gpd.GeoDataFrame:
    """
    Projects future shoreline positions for each transect using externally provided future offsets.

    Args:
        df: GeoDataFrame with transect geometries and metadata.
        futures: DataFrame with future shoreline change data per transect_id.
                 Must include 'transect_id', 'datetime', 'ssp',
                 'ambient_change_samples', 'sea_level_rise_samples', 'total_change_samples'.
        reference_lons: Column name (str) or list of float longitudes.
        reference_lats: Column name (str) or list of float latitudes.
        accommodation_buffer: Optional landward buffer to subtract from shoreline change.

    Returns:
        GeoDataFrame with projected Point geometries and structured shoreline change fields and metadata.
    """

    def resolve(ref: str | list, name: str):
        if isinstance(ref, str):
            if ref not in df.columns:
                raise ValueError(f"Column '{ref}' not found in DataFrame for '{name}'")
            return df[ref].tolist()
        if isinstance(ref, list):
            if len(ref) != len(df):
                raise ValueError(f"Length of list '{name}' must match DataFrame length")
            return ref
        raise TypeError(f"Argument '{name}' must be a string or a list")

    lons = resolve(reference_lons, "reference_lons")
    lats = resolve(reference_lats, "reference_lats")

    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        tid = row["transect_id"]
        matched = futures[futures["transect_id"] == tid]
        if matched.empty:
            continue

        ref_point = Point(lons[i], lats[i])

        grouped = matched.groupby(["datetime", "ssp"])

        for (target_dt, ssp), group in grouped:
            ambient_samples = (
                group["ambient_change_samples"].explode().astype(float).tolist()
            )
            slr_samples = (
                group["sea_level_rise_samples"].explode().astype(float).tolist()
            )
            total_samples = (
                group["total_change_samples"].explode().astype(float).tolist()
            )

            pt = project_along_transect(
                row,
                shoreline_change=total_samples,
                reference_point=ref_point,
                accommodation_buffer=accommodation_buffer,
            )

            if pt:
                records.append(
                    {
                        "transect_id": tid,
                        "ssp": ssp,
                        "datetime": pd.to_datetime(target_dt),
                        "ambient_change_is_valid": pd.NA,
                        "ambient_change_mean": float32(
                            pd.Series(ambient_samples).mean()
                        ),
                        "ambient_change_std": float32(pd.Series(ambient_samples).std()),
                        "ambient_change_p5": float32(
                            pd.Series(ambient_samples).quantile(0.05)
                        ),
                        "ambient_change_p95": float32(
                            pd.Series(ambient_samples).quantile(0.95)
                        ),
                        "ambient_change_samples": [float32(v) for v in ambient_samples],
                        "sea_level_rise_samples": [float32(v) for v in slr_samples],
                        "sea_level_rise_p5": float32(
                            pd.Series(slr_samples).quantile(0.05)
                        ),
                        "sea_level_rise_p50": float32(
                            pd.Series(slr_samples).quantile(0.50)
                        ),
                        "sea_level_rise_p95": float32(
                            pd.Series(slr_samples).quantile(0.95)
                        ),
                        "total_change_samples": [float32(v) for v in total_samples],
                        "total_change_p5": float32(
                            pd.Series(total_samples).quantile(0.05)
                        ),
                        "total_change_p50": float32(
                            pd.Series(total_samples).quantile(0.50)
                        ),
                        "total_change_p95": float32(
                            pd.Series(total_samples).quantile(0.95)
                        ),
                        "geometry": pt,
                    }
                )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).astype(
        {
            "ambient_change_mean": "float32",
            "ambient_change_std": "float32",
            "ambient_change_p5": "float32",
            "ambient_change_p95": "float32",
            "sea_level_rise_p5": "float32",
            "sea_level_rise_p50": "float32",
            "sea_level_rise_p95": "float32",
            "total_change_p5": "float32",
            "total_change_p50": "float32",
            "total_change_p95": "float32",
        }
    )
    return gdf


def apply_rectangle_projection_from_future_df(
    df: gpd.GeoDataFrame,
    futures: pd.DataFrame,
    reference_lons: str | list[float],
    reference_lats: str | list[float],
    accommodation_buffer: float = 100.0,
    alongshore_buffer: float = 50.0,
) -> gpd.GeoDataFrame:
    """
    Projects shoreline rectangles using externally provided future shoreline change offsets.

    Args:
        df: GeoDataFrame with transect geometries and metadata.
        futures: DataFrame containing future shoreline change for each transect_id.
                 Must include 'transect_id', 'offset_m', 'datetime', 'ssp'.
        reference_lons: Column name (str) or list of longitude values.
        reference_lats: Column name (str) or list of latitude values.
        accommodation_buffer: Landward buffer in meters (applied as -abs(buffer)).
        alongshore_buffer: Width of the rectangle (in meters) along the transect.

    Returns:
        GeoDataFrame with projected rectangle (POLYGON) geometries and schema-compliant shoreline change fields.
    """

    def resolve(ref: str | list, name: str):
        if isinstance(ref, str):
            if ref not in df.columns:
                raise ValueError(f"Column '{ref}' not found in DataFrame for '{name}'")
            return df[ref].tolist()
        if isinstance(ref, list):
            if len(ref) != len(df):
                raise ValueError(f"Length of list '{name}' must match DataFrame length")
            return ref
        raise TypeError(f"Argument '{name}' must be a string or a list")

    lons = resolve(reference_lons, "reference_lons")
    lats = resolve(reference_lats, "reference_lats")

    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        tid = row["transect_id"]
        matched = futures[futures["transect_id"] == tid]
        if matched.empty:
            continue

        ref_point = Point(lons[i], lats[i])

        for _, fut in matched.iterrows():
            shoreline_change = fut["offset_m"]
            poly = compute_rectangle_projection(
                row,
                shoreline_change=shoreline_change,
                accommodation_buffer=accommodation_buffer,
                reference_point=ref_point,
                alongshore_buffer=alongshore_buffer,
            )
            if not poly:
                continue

            offset = float32(shoreline_change)

            records.append(
                {
                    "transect_id": tid,
                    "ssp": fut.get("ssp", pd.NA),
                    "datetime": pd.to_datetime(fut["datetime"]),
                    "ambient_change_is_valid": pd.NA,
                    "ambient_change_mean": float32("nan"),
                    "ambient_change_std": float32("nan"),
                    "ambient_change_p5": float32("nan"),
                    "ambient_change_p95": float32("nan"),
                    "ambient_change_samples": None,
                    "sea_level_rise_samples": None,
                    "sea_level_rise_p5": float32("nan"),
                    "sea_level_rise_p50": float32("nan"),
                    "sea_level_rise_p95": float32("nan"),
                    "total_change_samples": [offset],
                    "total_change_p5": float32("nan"),
                    "total_change_p50": float32("nan"),
                    "total_change_p95": float32("nan"),
                    "geometry": poly,
                }
            )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).astype(
        {
            "ambient_change_mean": "float32",
            "ambient_change_std": "float32",
            "ambient_change_p5": "float32",
            "ambient_change_p95": "float32",
            "sea_level_rise_p5": "float32",
            "sea_level_rise_p50": "float32",
            "sea_level_rise_p95": "float32",
            "total_change_p5": "float32",
            "total_change_p50": "float32",
            "total_change_p95": "float32",
        }
    )
    return gdf


if __name__ == "__main__":
    import pathlib
    from datetime import datetime

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString

    df = gpd.read_parquet(
        pathlib.Path.home() / "Downloads" / "sample_dataframe.parquet"
    )

    # --- Run your function ---
    result = apply_projection_from_values(
        df=df,
        reference_lons="sds:reference_lon",
        reference_lats="sds:reference_lat",
        ambient_change_is_valid="ambient_change_is_valid",
        ambient_change_mean="ambient_change_mean",
        ambient_change_std="ambient_change_std",
        sea_level_rise_samples="sea_level_rise_samples",
        total_change_samples="total_change_samples",
        accommodation_buffer=0,  # meters
    )

    # --- Inspect the result ---
    print(
        result[
            [
                "transect_id",
                "ambient_change_mean",
                "sea_level_rise_p50",
                "total_change_p50",
                "geometry",
            ]
        ]
    )
