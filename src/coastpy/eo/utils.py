from typing import Any

import geopandas as gpd
import odc.geo
from odc.geo.geobox import GeoBox


def geobox_from_data_extent(
    region: gpd.GeoDataFrame,
    data_extent: gpd.GeoDataFrame,
    crs: Any,
    resolution: int | float,
) -> GeoBox:
    """
    Creates a geobox optimized to the maximum extent of available data.

    Args:
        region (gpd.GeoDataFrame): Input region geometry to constrain.
        data_extent (gpd.GeoDataFrame): Available data coverage geometry.
        crs (str): Target CRS for the geobox (e.g., 'EPSG:32633').
        resolution (int, optional): Spatial resolution in crs (meters, degrees).

    Returns:
        odc.geo.GeoBox: Geobox covering the intersection of region and data.

    Raises:
        ValueError: If the region does not overlap with available data.
    """
    # Find intersection between region and data extent
    intersection = gpd.overlay(region, data_extent).dissolve()

    if intersection.empty:
        raise ValueError(
            "No overlap between the provided region and data extent. "
            "Ensure both inputs are correct and overlapping."
        )

    # Create geobox from the intersected area
    intersected_geom = odc.geo.geom.Geometry(
        intersection.geometry.iloc[0], crs=region.crs
    )
    return GeoBox.from_geopolygon(intersected_geom.to_crs(crs), resolution=resolution)
