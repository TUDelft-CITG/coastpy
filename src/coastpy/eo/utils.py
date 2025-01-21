from typing import Any

import geopandas as gpd
import odc.geo
import pystac
import stac_geoparquet
from odc.geo.geobox import GeoBox


def data_extent_from_stac_items(
    items: list[pystac.Item],
    dissolve_kwargs: dict | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute the data extent by dissolving geometries in a STAC collection.

    Args:
        items (list[dict]): List of STAC items as dictionaries.
        group_by (str, optional): Column name for grouping.
            If None, geometries are grouped by their unique representation.
        dissolve_kwargs (dict, optional): Additional parameters for GeoDataFrame `dissolve`.

    Returns:
        gpd.GeoDataFrame: Dissolved GeoDataFrame.
    """
    if not items:
        raise ValueError("No items provided.")

    items_as_json = [item.to_dict() for item in items]

    gdf = stac_geoparquet.to_geodataframe(items_as_json, dtype_backend="pyarrow")
    dissolve_kwargs = dissolve_kwargs or {}
    return gdf.dissolve(by=gdf.geometry.apply(lambda geom: geom.wkt), **dissolve_kwargs)


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
