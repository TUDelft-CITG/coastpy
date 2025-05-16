from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import norm
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform as shapely_transform

from coastpy.geo.geoms import create_offset_rectangle
from coastpy.geo.ops import extend_transect


def ambient_change_from_trend(
    df: gpd.GeoDataFrame,
    target_datetimes: Sequence[datetime],
    reference_datetime: str = "sds:reference_datetime",
    change_rate: str = "sds:change_rate",
) -> gpd.GeoDataFrame:
    """
    Computes projected ambient change for each target year based on a fixed trend rate.

    Args:
        df: GeoDataFrame containing 'sds:change_rate' and reference datetime columns.
        target_datetimes: List of future datetime objects (e.g., [2050-01-01, 2100-01-01]).
        reference_datetime: Column with the base year of observation.
        change_rate: Column with shoreline change rate in meters/year.

    Returns:
        Expanded GeoDataFrame with one row per target datetime, including:
            - 'ambient_change_mean': projected trend-based change in meters.
            - 'datetime': target datetime.
    """
    if reference_datetime not in df.columns:
        raise ValueError(f"Missing required column: {reference_datetime}")
    if change_rate not in df.columns:
        raise ValueError(f"Missing required column: {change_rate}")

    # Resolve datetime and rate columns
    base_years = pd.to_datetime(df[reference_datetime]).dt.year
    rates = df[change_rate].astype(float)

    # Store results per target year
    records = []
    for target_dt in target_datetimes:
        target_year = target_dt.year
        deltas = target_year - base_years
        ambient_mean = rates * deltas

        df_copy = df.copy()
        df_copy["ambient_change_mean"] = ambient_mean.astype("float32")
        df_copy["datetime"] = target_dt
        records.append(df_copy)

    return gpd.GeoDataFrame(
        pd.concat(records, ignore_index=True), geometry="geometry", crs=df.crs
    )


def ambient_change_is_valid(df: gpd.GeoDataFrame) -> list[bool]:
    """
    Marks all rows as valid for ambient trend-based change. Placeholder for future logic.

    Args:
        df: GeoDataFrame (ambient projection data).

    Returns:
        List of True values indicating validity.
    """
    return [True] * len(df)


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


def prepare_transect_context(
    row: dict,
    reference_point: Point | tuple[float, float] | None,
    extend_to_m: float = 10_000.0,
) -> tuple[LineString, Point, callable, float]:
    """
    Prepare UTM reprojection context and compute base distance along the transect.
    """
    transect = row.get("geometry")
    utm_epsg = row.get("utm_epsg")

    if not isinstance(transect, LineString) or len(transect.coords) < 2:
        raise ValueError(
            f"Invalid or empty geometry for transect {row.get('transect_id', '?')}."
        )

    if reference_point is None:
        reference_point = Point(row["sds:reference_lon"], row["sds:reference_lat"])
    elif isinstance(reference_point, tuple):
        reference_point = Point(reference_point)

    to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
    ).transform
    to_wgs = Transformer.from_crs(
        f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
    ).transform

    transect_utm = shapely_transform(to_utm, transect)
    transect_utm = extend_transect(transect_utm, extend_to_m, side="both")
    ref_point_utm = shapely_transform(to_utm, reference_point)
    base_distance_m = transect_utm.project(ref_point_utm)

    return transect_utm, ref_point_utm, to_wgs, base_distance_m


def compute_projection_stats(
    ambient_change_mean: float,
    ambient_change_std: float,
    retreat_samples: list[float] | np.ndarray,
    total_change_samples: list[float] | np.ndarray,
) -> dict:
    """
    Computes change quantiles and shoreline change statistics using ambient, retreat, and total samples.

    Args:
        ambient_change_mean: Mean ambient (trend-based) change in meters.
        ambient_change_std: Standard deviation of ambient change.
        retreat_samples: Sea-level rise sample values (list or array).
        total_change_samples: Optional total shoreline change samples (ambient + retreat).

    Returns:
        Dict of float32-compatible statistics and input-normalized sample lists.
    """
    # Ambient change quantiles (normal distribution)

    ambient_change_p5, ambient_change_p95 = normal_p5_p95(
        ambient_change_mean, ambient_change_std
    )

    # Retreat samples (e.g. SLR)
    if isinstance(retreat_samples, list | np.ndarray) and len(retreat_samples) > 0:
        retreat_arr = np.asarray(retreat_samples, dtype="float32")

        if np.ma.isMaskedArray(retreat_arr) and isinstance(
            retreat_arr, np.ma.MaskedArray
        ):
            retreat_arr = retreat_arr.filled(np.nan)

        retreat_p5, retreat_p50, retreat_p95 = np.percentile(retreat_arr, [5, 50, 95])
        retreat_samples_out = retreat_arr.tolist()
    else:
        retreat_p5 = retreat_p50 = retreat_p95 = 0.0
        retreat_samples_out = [0.0]

    # Total change samples (optional diagnostics)
    if (
        isinstance(total_change_samples, list | np.ndarray)
        and len(total_change_samples) > 0
    ):
        total_arr = np.asarray(total_change_samples, dtype="float32")
        if np.ma.isMaskedArray(total_arr) and isinstance(total_arr, np.ma.MaskedArray):
            total_arr = total_arr.filled(np.nan)  # replace masked values with np.nan

        total_p5, total_p50, total_p95 = np.percentile(total_arr, [5, 50, 95])
        total_samples_out = total_arr.tolist()
    else:
        total_p5 = total_p50 = total_p95 = np.nan
        total_samples_out = [np.nan]

    return {
        "ambient_change_p5": np.float32(ambient_change_p5),
        "ambient_change_p95": np.float32(ambient_change_p95),
        "retreat_samples": retreat_samples_out,
        "retreat_p5": np.float32(retreat_p5),
        "retreat_p50": np.float32(retreat_p50),
        "retreat_p95": np.float32(retreat_p95),
        "total_change_samples": total_samples_out,
        "total_change_p5": np.float32(total_p5),
        "total_change_p50": np.float32(total_p50),
        "total_change_p95": np.float32(total_p95),
    }


def point_projection(
    row: dict,
    shoreline_change_m: float,
    reference_point: Point | tuple[float, float] | None = None,
    accommodation_buffer_m: float = 0.0,
    extend_to_m: float = 10_000.0,
) -> Point | None:
    """
    Project a point along a transect.
    """
    try:
        transect_utm, _, to_wgs, base_distance_m = prepare_transect_context(
            row, reference_point, extend_to_m
        )
        accommodation_buffer_m = -abs(accommodation_buffer_m)
        offset_m = shoreline_change_m + accommodation_buffer_m
        projection_distance = np.clip(
            base_distance_m + offset_m, 0, transect_utm.length
        )
        point_utm = transect_utm.interpolate(projection_distance)
        return shapely_transform(to_wgs, point_utm)

    except Exception as e:
        print(f"[WARN] Point projection failed for {row.get('transect_id', '?')}: {e}")
        return None


def rectangle_projection(
    row: dict,
    shoreline_change_m: float,
    reference_point: Point | tuple[float, float] | None = None,
    accommodation_buffer_m: float = 0.0,
    alongshore_buffer_m: float = 50.0,
    extend_to_m: float = 10_000.0,
) -> Polygon | None:
    """
    Project a rectangle (polygon) along a transect.
    """
    try:
        transect_utm, _, to_wgs, base_distance_m = prepare_transect_context(
            row, reference_point, extend_to_m
        )
        offset_m = shoreline_change_m - abs(accommodation_buffer_m)
        projection_distance = np.clip(
            base_distance_m + offset_m, 0, transect_utm.length
        )
        pt_start = transect_utm.interpolate(base_distance_m)
        pt_end = transect_utm.interpolate(projection_distance)
        segment = LineString([pt_start, pt_end])
        rect_utm = create_offset_rectangle(segment, distance=alongshore_buffer_m)
        return shapely_transform(to_wgs, rect_utm)

    except Exception as e:
        print(
            f"[WARN] Rectangle projection failed for {row.get('transect_id', '?')}: {e}"
        )
        return None


def apply_projection_from_values(
    df: gpd.GeoDataFrame,
    reference_lons: str | list[float],
    reference_lats: str | list[float],
    accommodation_buffer: float = 0.0,
    ambient_change_is_valid: str | list[bool] | None = None,
    ambient_change_mean: str | list[float] | None = None,
    ambient_change_std: str | list[float] | None = None,
    retreat_samples: str | list[list[float]] | None = None,
    total_change_samples: str | list[list[float]] | None = None,
    result_type: Literal["point", "rectangle"] = "point",
    alongshore_buffer: float = 50.0,
) -> gpd.GeoDataFrame:
    """
    Apply shoreline projection (point or rectangle) using ambient and sea level rise inputs.
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
        df, retreat_samples, "retreat_samples", default=[], cast=list
    )
    total_samples = resolve_input(
        df, total_change_samples, "total_change_samples", default=[], cast=list
    )

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        tid = row["transect_id"]
        ref_point = Point(lons[i], lats[i])  # type: ignore

        stats = compute_projection_stats(
            ambient_change_mean=ac_mean[i],  # type: ignore
            ambient_change_std=ac_std[i],  # type: ignore
            retreat_samples=slr_samples[i],  # type: ignore
            total_change_samples=total_samples[i],  # type: ignore
        )

        shoreline_change = ac_mean[i] + stats["retreat_p50"]

        if result_type == "rectangle":
            geom = rectangle_projection(
                row=row,  # type: ignore
                shoreline_change_m=shoreline_change,
                accommodation_buffer_m=accommodation_buffer,
                reference_point=ref_point,
                alongshore_buffer_m=alongshore_buffer,
            )
        else:
            geom = point_projection(
                row=row,  # type: ignore
                shoreline_change_m=shoreline_change,
                accommodation_buffer_m=accommodation_buffer,
                reference_point=ref_point,
            )

        if geom is None:
            continue

        records.append(
            {
                "transect_id": tid,
                "ssp": row.get("ssp", pd.NA),
                "datetime": row.get("datetime", pd.NaT),
                "ambient_change_is_valid": is_valid[i],
                "ambient_change_mean": np.float32(ac_mean[i]),
                "ambient_change_std": np.float32(ac_std[i]),
                "ambient_change_p5": np.float32(stats["ambient_change_p5"]),
                "ambient_change_p95": np.float32(stats["ambient_change_p95"]),
                "retreat_samples": stats["retreat_samples"],
                "retreat_p5": np.float32(stats["retreat_p5"]),
                "retreat_p50": np.float32(stats["retreat_p50"]),
                "retreat_p95": np.float32(stats["retreat_p95"]),
                "total_change_samples": stats["total_change_samples"],
                "total_change_p5": np.float32(stats["total_change_p5"]),
                "total_change_p50": np.float32(stats["total_change_p50"]),
                "total_change_p95": np.float32(stats["total_change_p95"]),
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
            "retreat_p5": "float32",
            "retreat_p50": "float32",
            "retreat_p95": "float32",
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


if __name__ == "__main__":
    import os
    import pathlib

    import dotenv
    import fsspec
    import geopandas as gpd

    from coastpy.io.cloud import write_parquet
    from coastpy.io.utils import name_data

    # --- LOAD ENV + STORAGE CONFIG ---
    dotenv.load_dotenv()
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {
        "account_name": "coclico",
        "sas_token": sas_token,
    }

    # --- FILESYSTEM + PATHS ---
    input_path = "file://" + str(
        pathlib.Path.home() / "Downloads" / "sample_dataframe.parquet"
    )
    output_prefix = "file://" + str(pathlib.Path.home() / "tmp" / "projections")

    fs = fsspec.filesystem(fsspec.utils.get_protocol(output_prefix), **storage_options)
    if not fs.exists(output_prefix):
        fs.mkdirs(output_prefix, exist_ok=True)

    # --- LOAD INPUT ---
    with fsspec.open(input_path, mode="rb") as f:
        df = gpd.read_parquet(f)

    # --- NORMALIZE COLUMN NAMES ---
    df.rename(
        columns=lambda col: col.replace("sea_level_rise", "retreat")
        if "sea_level_rise" in col
        else col,
        inplace=True,
    )
    df.rename(columns={"scenario": "ssp"}, inplace=True)

    # --- PROJECTION RUNS ---
    configurations = [
        (0, "point", "noaccom"),
        (0, "rectangle", "noaccom"),
        (50, "point", "accom"),
        (50, "rectangle", "accom"),
    ]

    for buffer, result_type, postfix in configurations:
        print(f"Running {result_type} projection (buffer: {buffer}m)...")

        gdf = apply_projection_from_values(
            df=df,
            reference_lons="sds:reference_lon",
            reference_lats="sds:reference_lat",
            ambient_change_is_valid="ambient_change_is_valid",
            ambient_change_mean="ambient_change_mean",
            ambient_change_std="ambient_change_std",
            retreat_samples="retreat_samples",
            total_change_samples="total_change_samples",
            accommodation_buffer=buffer,
            result_type=result_type,
        )

        urlpath = name_data(
            gdf,
            prefix=output_prefix,
            filename_prefix=f"projection_{result_type}",
            postfix=postfix,
        )

        write_parquet(gdf, urlpath, storage_options=storage_options)
