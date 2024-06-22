from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point


def linestring_to_coords(
    data: gpd.GeoDataFrame | gpd.GeoSeries | LineString,
    columns: list[str] | None = None,
) -> tuple[pd.Series, pd.Series] | gpd.GeoDataFrame:
    """
    Explode linestring geometry and return longitude and latitude coordinates
    as separate variables.

    Args:
        data (Union[gpd.GeoDataFrame, gpd.GeoSeries, LineString]): The input geometry.
        columns (List[str], optional): List containing names of longitude and latitude, defaults to ["lon", "lat"].

    Returns:
        Union[Tuple[pd.Series, pd.Series], gpd.GeoDataFrame]: If input is a GeoSeries,
            returns two Series with lon and lat values. If input is a GeoDataFrame,
            returns the dataframe with exploded coordinates.
    """

    if columns is None:
        columns = ["lon", "lat"]

    if isinstance(data, LineString):
        data = gpd.GeoSeries(data)

    exploded = data.geometry.apply(lambda x: x.coords).explode()

    if isinstance(data, gpd.GeoSeries):
        gs = gpd.GeoSeries(exploded.apply(Point))
        return gs.x, gs.y

    elif isinstance(data, gpd.GeoDataFrame):
        df = pd.DataFrame(exploded.tolist(), columns=columns, index=exploded.index)
        return data.join(df).reset_index().rename(columns={"index": "group"})

    msg = "data must be a GeoDataFrame, GeoSeries, or LineString"
    raise TypeError(msg)


def geojson_to_bbox(geometry: dict[str, Any]) -> list[float]:
    """
    Extract the bounding box from a geojson geometry.

    Args:
        geometry (Dict[str, Any]): GeoJSON geometry dictionary.

    Returns:
        List[float]: Bounding box of the geojson geometry, formatted according to:

    Note: https://tools.ietf.org/html/rfc7946#section-5

    Raises:
        ValueError: If the provided geometry type is not supported.
    """

    geom_type = geometry["type"]
    coords = geometry["coordinates"]

    if geom_type == "Point":
        lon, lat = coords
        return [lon, lat, lon, lat]

    lats, lons = [], []

    if geom_type in ["LineString", "MultiPoint"]:
        for lon, lat in coords:
            lons.append(lon)
            lats.append(lat)

    elif geom_type == "Polygon":
        for linear_ring in coords:
            for lon, lat in linear_ring:
                lons.append(lon)
                lats.append(lat)

    elif geom_type in ["MultiLineString", "MultiPolygon"]:
        for part in coords:
            for component in part:
                if isinstance(component[0], float | int):  # Handle MultiLineString
                    lon, lat = component
                    lons.append(lon)
                    lats.append(lat)
                else:  # Handle MultiPolygon
                    for lon, lat in component:
                        lons.append(lon)
                        lats.append(lat)

    else:
        msg = f"Unsupported geometry type: {geom_type}"
        raise ValueError(msg)

    return [min(lons), min(lats), max(lons), max(lats)]


def bbox_to_geojson(
    bbox: list[float] | tuple[float, float, float, float],
) -> dict[str, str | list[list[list[float]]]]:
    """
    Convert a bounding box into a Polygon geometry in dictionary format.

    Args:
        bbox (List[float]): A bounding box in the format [min_lat, min_lon, max_lat, max_lon].

    Returns:
        Dict[str, Union[str, List[List[List[float]]]]]: Polygon geometry represented as a dictionary.

    Note:
        The function doesn't use `shapely.geometry.shape` directly but constructs the polygon
        based on the provided bounding box.
    """

    return {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
            ]
        ],
    }
