import warnings

import antimeridian
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, box


def create_antimeridian_buffer(
    max_distance: float, buffer_factor: float = 1.5
) -> gpd.GeoDataFrame:
    """
    Creates a buffered zone around the antimeridian to account for spatial
    operations near the -180/180 longitude line.

    Args:
        max_distance (float): Maximum distance in meters for the buffer around the antimeridian.
        buffer_factor (float, optional): Factor to scale the buffer size. Defaults to 1.5.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the buffered zone around the antimeridian.
    """
    # Create a GeoDataFrame representing the antimeridian line
    antimeridian_line = gpd.GeoDataFrame(
        geometry=[LineString([[-180, -85], [-180, 85]])], crs=4326
    )

    # Apply buffer in a projected CRS (e.g., Web Mercator) and convert back to lat/lon
    buffer_zone = antimeridian_line.to_crs(3857).buffer(max_distance * buffer_factor)
    buffer_zone = gpd.GeoDataFrame(geometry=buffer_zone, crs=3857).to_crs(4326)

    # Suppress FixWindingWarning from the antimeridian package
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", antimeridian.FixWindingWarning)
        fixed_geometry = shapely.geometry.shape(
            antimeridian.fix_geojson(
                shapely.geometry.mapping(buffer_zone.iloc[0].geometry)
            )
        )

    # Return as GeoDataFrame
    return gpd.GeoDataFrame(geometry=[fixed_geometry], crs=4326)


def add_attributes_from_gdf(
    df: gpd.GeoDataFrame,
    other_gdf: gpd.GeoDataFrame,
    max_distance: float = 20000,
    buffer_factor: float = 1.5,
) -> gpd.GeoDataFrame:
    """
    Adds attributes from a source GeoDataFrame to a target GeoDataFrame based on nearest spatial join.

    Args:
        df (gpd.GeoDataFrame): The target GeoDataFrame to which attributes will be added.
        other_gdf (gpd.GeoDataFrame): The other GeoDataFrame from which attributes will be extracted.
        max_distance (float): The maximum distance for nearest neighbor consideration, in meters.
        buffer_factor (float): Factor to increase the buffer area. Defaults to 1.5.

    Returns:
        gpd.GeoDataFrame: The target GeoDataFrame with added attributes from the source GeoDataFrame.
    """
    # Ensure the transect GeoDataFrame has a point geometry to avoid double intersection.
    transect_origins = gpd.GeoDataFrame(
        df[["transect_id"]],
        geometry=gpd.points_from_xy(df.lon, df.lat, crs=4326),
    )

    antimeridian_buffer = create_antimeridian_buffer(
        max_distance, buffer_factor=buffer_factor
    )

    # Optimization: define the region of interest with a buffer that only works in areas far away from
    # the antimeridian
    if gpd.overlay(transect_origins, antimeridian_buffer).empty:
        roi = gpd.GeoDataFrame(geometry=[box(*transect_origins.total_bounds)], crs=4326)
        roi = gpd.GeoDataFrame(
            geometry=roi.to_crs(3857).buffer(max_distance * 1.5).to_crs(4326)
        )

        # Filter source GeoDataFrame within the region of interest
        other_gdf = gpd.sjoin(other_gdf, roi).drop(columns=["index_right"])

    # Perform nearest neighbor spatial join
    joined = gpd.sjoin_nearest(
        transect_origins.to_crs(3857),
        other_gdf.to_crs(3857),
        max_distance=max_distance,
    ).drop(columns=["index_right", "geometry"])

    # Merge the attributes into the original target GeoDataFrame
    result = pd.merge(df, joined, on="transect_id", how="left").drop_duplicates(
        "transect_id"
    )
    return result


def add_attributes_from_gdfs(
    df: gpd.GeoDataFrame,
    other_gdfs: list[gpd.GeoDataFrame],
    max_distance: float = 20000,
) -> gpd.GeoDataFrame:
    """
    Adds attributes from multiple other GeoDataFrames to a target GeoDataFrame.

    Args:
        df (gpd.GeoDataFrame): The target GeoDataFrame to which attributes will be added.
        other_gdfs (List[gpd.GeoDataFrame]): A list of other GeoDataFrames from which attributes will be extracted.
        max_distance (float): The maximum distance for nearest neighbor consideration, in meters.

    Returns:
        gpd.GeoDataFrame: The target GeoDataFrame with added attributes from all source GeoDataFrames.
    """
    for source_gdf in other_gdfs:
        df = add_attributes_from_gdf(df, source_gdf, max_distance)
    return df
