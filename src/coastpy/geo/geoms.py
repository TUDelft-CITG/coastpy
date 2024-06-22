from typing import Any

import geopandas as gpd
import shapely
from shapely import LineString, Polygon

from coastpy.geo.ops import generate_offset_line
from coastpy.geo.transform import (
    bbox_to_geojson,
)


def bbox_from_coords(coords: list[float | list[Any]]) -> list[float]:
    """
    Extract the bounding box from a list of coordinates.

    This function recursively goes through a list of coordinates, which can be
    nested to various levels, and returns the bounding box for the given coordinates.

    Args:
        coords (List[Union[float, List[Any]]]): A list of coordinates which can be
            nested lists representing complex geometries.

    Returns:
        List[float]: Bounding box in the format [min_lat, min_lon, max_lat, max_lon].

    Raises:
        AssertionError: If there's a type mismatch in the coordinates.
    """

    lats, lons = [], []

    def _extract(coords: list[float | int | list[Any]]) -> None:
        for x in coords:
            # This handles points
            if isinstance(x, float):
                assert isinstance(
                    coords[0], float
                ), f"Type mismatch: {coords[0]} is not a float"
                assert isinstance(
                    coords[1], float
                ), f"Type mismatch: {coords[1]} is not a float"
                lats.append(coords[0])
                lons.append(coords[1])
                return
            # This handles nested lists of coordinates
            if isinstance(x, list) and isinstance(x[0], list):
                _extract(x)
            else:
                lat, lon = x
                lats.append(lat)
                lons.append(lon)

    _extract(coords)
    lats.sort()
    lons.sort()

    return [lats[0], lons[0], lats[-1], lons[-1]]


def geo_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """GeoDataFrame with a bounding box that can be used for various gis operations.
    Args:
        min_lon (float): most eastern longitude
        min_lat (float): most northern latitude
        max_lon (float): most wester longitude
        max_lat (float): most soutern latitude
        src_crs (str, optional): Valid EPSG string or number (int). Defaults to "EPSG:4326".
        dst_crs (str, optional): Valid EPSG string or number (int). Defaults to "EPSG:4326".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with bounding box as geometry.
    """

    bbox = [min_lon, min_lat, max_lon, max_lat]
    bbox = bbox_to_geojson(bbox)
    bbox = shapely.geometry.shape(bbox)
    return gpd.GeoDataFrame(geometry=[bbox], crs=src_crs).to_crs(dst_crs)


def create_offset_rectangle(line: LineString, distance: float) -> Polygon:
    """
    Construct a rectangle polygon using the original line and an offset distance.

    Args:
        line (LineString): The original line around which the polygon is constructed.
        distance (float): The offset distance used to create the sides of the polygon.

    Returns:
        Polygon: The constructed rectangle-shaped polygon.
    """

    # Create the offset lines
    left_offset_line = generate_offset_line(line, distance)
    right_offset_line = generate_offset_line(line, -distance)

    # Retrieve end points
    left_start, left_end = left_offset_line.coords[:]
    right_start, right_end = right_offset_line.coords[:]

    # Construct the polygon using the end points
    polygon = Polygon([left_start, left_end, right_end, right_start])

    return polygon
