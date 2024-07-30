import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def add_attributes_from_gdf(
    df: gpd.GeoDataFrame,
    other_gdf: gpd.GeoDataFrame,
    max_distance: float = 20000,
) -> gpd.GeoDataFrame:
    """
    Adds attributes from a source GeoDataFrame to a target GeoDataFrame based on nearest spatial join.

    Args:
        df (gpd.GeoDataFrame): The target GeoDataFrame to which attributes will be added.
        other_gdf (gpd.GeoDataFrame): The other GeoDataFrame from which attributes will be extracted.
        max_distance (float): The maximum distance for nearest neighbor consideration, in meters.

    Returns:
        gpd.GeoDataFrame: The target GeoDataFrame with added attributes from the source GeoDataFrame.
    """
    # Ensure the transect GeoDataFrame has a point geometry to avoid double intersection.
    transect_origins = gpd.GeoDataFrame(
        df[["tr_name"]],
        geometry=gpd.points_from_xy(df.lon, df.lat, crs=4326),
    )

    # Optimization: define the region of interest with a buffer
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
    result = pd.merge(df, joined, on="tr_name", how="left").drop_duplicates("tr_name")
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
