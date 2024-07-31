from typing import Literal

import geopandas as gpd
import mercantile
from shapely import Point
from shapely.geometry import shape


def bbox_to_quadkey(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> str | None:
    """
    Converts a bounding box to a single quadkey, if possible, for the highest zoom level where the bounding box
    fits into a single tile.

    Args:
    - min_lon (float): The minimum longitude of the bounding box.
    - min_lat (float): The minimum latitude of the bounding box.
    - max_lon (float): The maximum longitude of the bounding box.
    - max_lat (float): The maximum latitude of the bounding box.

    Returns:
    - Optional[str]: The quadkey string if the bounding box fits into a single tile at any zoom level from 12 down to 0,
      None otherwise.
    """
    for zoom in range(12, -1, -1):
        tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))
        if len(tiles) == 1:
            return mercantile.quadkey(tiles[0])
    return None


def geojson_to_quadkey(geojson_data: dict) -> str | None:
    """
    Converts GeoJSON geometry to a quadkey string, if the geometry fits into a single tile at any zoom level.

    Args:
    - geojson_data (Dict): The GeoJSON data containing the geometry.

    Returns:
    - Optional[str]: The quadkey string if the geometry fits into a single tile, None otherwise.
    """
    geom = shape(geojson_data["geometry"])
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    return bbox_to_quadkey(min_lon, min_lat, max_lon, max_lat)


def quadkey_to_geojson(quadkey: str) -> dict:
    """
    Converts a quadkey string to its GeoJSON polygon representation.

    Args:
    - quadkey (str): The quadkey string representing a single tile.

    Returns:
    - Dict: The GeoJSON representation of the tile's bounding box as a polygon.
    """
    tile = mercantile.quadkey_to_tile(quadkey)
    bbox = mercantile.bounds(tile)
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox.west, bbox.south],
                    [bbox.east, bbox.south],
                    [bbox.east, bbox.north],
                    [bbox.west, bbox.north],
                    [bbox.west, bbox.south],
                ]
            ],
        },
    }


def add_geo_columns(
    df: gpd.GeoDataFrame,
    geo_columns: list[Literal["bbox", "bounding_quadkey", "quadkey"]],
    quadkey_zoom_level: int = 12,
) -> gpd.GeoDataFrame:
    """
    Augments a GeoDataFrame with bounding box, optional bounding quadkey, and optional quadkey columns.

    Args:
        df: A GeoDataFrame with geometries.
        geo_columns: A list of strings specifying which columns to add. Possible values are 'bbox', 'bounding_quadkey', and 'quadkey'.
        quadkey_zoom_level: Zoom level for quadkey calculation, default is 12.

    Returns:
        A GeoDataFrame possibly with added 'bbox', 'bounding_quadkey', and 'quadkey' columns.
    """
    df = df.copy()
    src_crs = df.crs.to_epsg()
    if src_crs != 4326:
        df = df.to_crs(epsg=4326)

    def calculate_bbox(geom):
        """Calculates the bounding box of a geometry."""
        minx, miny, maxx, maxy = geom.bounds
        return {"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy}

    def calculate_bounding_quadkey(bbox):
        """Calculates the quadkey for the bounding box."""
        return mercantile.quadkey(
            mercantile.bounding_tile(
                bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
            )
        )

    def get_point_from_geometry(geom):
        """Get a representative point from different geometry types."""
        if geom.geom_type == "Point":
            return geom
        elif geom.geom_type in ["Polygon", "MultiPolygon"]:
            return geom.centroid
        elif geom.geom_type in ["LineString"]:
            return geom.interpolate(0.5, normalized=True)
        # NOTE: The following code is commented out because it is not used in the current implementation.
        # elif geom.geom_type == "MultiPoint":
        #     return geom[0]
        # elif geom.geom_type in ["MultiLineString", "GeometryCollection"]:
        #     return geom[0].interpolate(0.5, normalized=True)
        else:
            msg = f"Unsupported geometry type: {geom.geom_type}"
            raise ValueError(msg)

    # Add bounding box column
    if "bbox" in geo_columns:
        df["bbox"] = df.geometry.apply(calculate_bbox)

    # Add bounding quadkey column
    if "bounding_quadkey" in geo_columns:
        if "bbox" not in df.columns:
            df["bbox"] = df.geometry.apply(calculate_bbox)
        df["bounding_quadkey"] = df["bbox"].apply(calculate_bounding_quadkey)

    # Add quadkey column
    if "quadkey" in geo_columns:
        if quadkey_zoom_level is None:
            msg = (
                "quadkey_zoom_level must be provided when 'quadkey' is in geo_columns."
            )
            raise ValueError(msg)
        if "lon" in df.columns and "lat" in df.columns:
            points = gpd.GeoSeries(
                [Point(xy) for xy in zip(df.lon, df.lat, strict=False)], crs="EPSG:4326"
            )
        else:
            points = df.geometry.apply(get_point_from_geometry).to_crs("EPSG:4326")

        quadkeys = points.apply(
            lambda point: mercantile.quadkey(
                mercantile.tile(point.x, point.y, quadkey_zoom_level)
            )
        )
        df["quadkey"] = quadkeys

    # Revert CRS if necessary
    if src_crs != 4326:
        df = df.to_crs(epsg=src_crs)

    return df
