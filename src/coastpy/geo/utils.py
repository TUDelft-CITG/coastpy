import warnings

import geopandas as gpd
import shapely.geometry
from ipyleaflet import Map


def get_region_of_interest_from_map(
    m: Map, default_extent: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """
    Extract ROI from an ipyleaflet map or use the provided extent.

    Args:
        m (ipyleaflet.Map): Interactive map object.
        extent (tuple): Fallback (west, south, east, north) if map is not rendered.

    Returns:
        gpd.GeoDataFrame: ROI as a GeoDataFrame in EPSG:4326.
    """
    try:
        west, south, east, north = m.west, m.south, m.east, m.north
    except AttributeError:
        warnings.warn("Map not rendered. Using provided default extent.", stacklevel=2)
        west, south, east, north = default_extent

    if not west:
        warnings.warn("Map not rendered. Using provided default extent.", stacklevel=2)
        west, south, east, north = default_extent

    center_lon = (west + east) / 2
    center_lat = (north + south) / 2

    print(f"m.center = ({center_lat:.2f}, {center_lon:.2f})")
    print(f"m.zoom = {m.zoom:.1f}")
    print(
        f"west, south, east, north = ({west:.3f}, {south:.3f}, {east:.3f}, {north:.3f})"
    )

    roi = shapely.geometry.box(west, south, east, north)
    return gpd.GeoDataFrame(geometry=[roi], crs="EPSG:4326")
