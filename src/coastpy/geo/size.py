import sys

import geopandas as gpd
import numpy as np
import pandas as pd


def get_geometry_memory_usage(series: gpd.GeoSeries) -> np.ndarray:
    """
    Get memory usage for each geometry in the GeoSeries.

    Args:
        series (gpd.GeoSeries): GeoPandas GeoSeries containing geometries.

    Returns:
        np.ndarray: Memory usage for each geometry.
    """
    return np.array([sys.getsizeof(geom.wkb) for geom in series])


def estimate_memory_usage_per_row(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.Series:
    """
    Estimate the memory usage of each row in a DataFrame or GeoDataFrame.

    Args:
        df (Union[pd.DataFrame, gpd.GeoDataFrame]): DataFrame or GeoDataFrame whose row memory usages are to be computed.

    Returns:
        pd.Series: Memory usage for each row.
    """
    # Compute memory usage for geometry columns
    geom_usage_per_row = pd.Series(
        np.sum(
            [
                get_geometry_memory_usage(df[col])
                for col in df.columns
                if isinstance(df[col].dtype, gpd.array.GeometryDtype)
            ],
            axis=0,
        )
    )

    # Estimate non-geometry usage per row
    non_geom_usage_per_col = df.select_dtypes(
        exclude=[gpd.geoseries.GeoSeries]
    ).memory_usage(deep=True, index=False)
    non_geom_usage_per_row = non_geom_usage_per_col.sum() / len(df)

    total_usage_per_row = geom_usage_per_row + non_geom_usage_per_row

    return total_usage_per_row.astype(int)


def estimate_memory_usage(df: pd.DataFrame | gpd.GeoDataFrame) -> int:
    """
    Estimate the memory usage of a DataFrame or GeoDataFrame.

    Args:
        df (Union[pd.DataFrame, gpd.GeoDataFrame]): DataFrame or GeoDataFrame to estimate memory usage.

    Returns:
        int: Estimated total memory usage in bytes.
    """
    mem_usage_per_row = estimate_memory_usage_per_row(df)
    return mem_usage_per_row.sum()
