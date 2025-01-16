import math
from itertools import product, repeat
from typing import Any

import geopandas as gpd
import mercantile
import numpy as np
import shapely.geometry as sgeom
from mercantile import Tile
from shapely.geometry import Polygon, box


def tile_to_quadkey(tile: Tile) -> str:
    """
    Convert a Tile to its corresponding quadkey.

    Args:
        tile (Tile): A mercantile Tile object.

    Returns:
        str: The quadkey corresponding to the tile.
    """
    quadkey = []
    for i in range(tile.z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tile.x & mask) != 0:
            digit += 1
        if (tile.y & mask) != 0:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


def make_mercantiles(zoom_level: int) -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame of tiles for a given zoom level with their corresponding quadkeys.

    Args:
        zoom_level (int): The zoom level for which to generate tiles.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing tile geometries and their quadkeys.
    """
    tiles_data: list[dict[str, Any]] = []
    num_tiles = 2**zoom_level

    for x in range(num_tiles):
        for y in range(num_tiles):
            tile = mercantile.Tile(x=x, y=y, z=zoom_level)
            bounds = mercantile.bounds(tile)
            tile_polygon = box(bounds.west, bounds.south, bounds.east, bounds.north)
            quadkey = mercantile.quadkey(tile)
            tiles_data.append({"geometry": tile_polygon, "quadkey": quadkey})

    df = gpd.GeoDataFrame(tiles_data, columns=["geometry", "quadkey"], crs=4326)
    return df


def deg2num(lon_deg: float, lat_deg: float, zoom: int) -> tuple:
    """Derive tile number for point in WGS 84 given certain zoom level.

    Args:
        lon_deg (float): Longitude as float in WGS 84.
        lat_deg (float):Latitude as float in WGS 84.
        zoom (int): Zoom following OSM slippy tile names conventions.

    Returns:
        _type_: tile number (xtile, ytile) following OSM slippy tile names conventions.
    """

    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple:
    """Tile number (xtile, ytile) to WGS 84 point (longitude, latitude) of northwest tile corner.

    Args:
        xtile (int): xtile number as int.
        ytile (int): ytile number as int int.
        zoom (int): zoom level following OSM slippy map names convention.

    Returns:
        tuple: point (longitude, latitude) in WGS 84.
    """
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def unique_items(tile_numbers: list) -> list:
    """Filter unique tuples from list of tuples.

    Args:
        tile_numbers (list): List of tuples.

    Returns:
        list: List of tuples.
    """
    seen = set()
    result = []
    for item in tile_numbers:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def coords2num(lons: list, lats: list, zoom: int) -> list:
    """List of longitude / latitude coordinates to OpenStreetMap tile names given certain zoom level.

    Args:
        lons (list): List of longitude as floats in WGS 84.
        lats (list): List of latitudes as floats in WGS 84.
        zoom (int): Zoom level following OSM slippy tile names.

    Returns:
        list: List of tuples with tile numbers (xtile, ytile).
    """
    tile_numbers = list(map(deg2num, lons, lats, repeat(zoom)))

    return unique_items(tile_numbers)


def zoom2num(zoom: int) -> list:
    """Generate tile numbers for zoom level.

    Args:
        zoom (int): zoom level in accordance with OpenStreetMap slippy tile names.

    Returns:
        list: list with tuples of tile numbers.
    """
    # tiles (zoom level) = 2**zoom * 2**zoom
    ntiles = 2**zoom
    tile_arange = np.arange(ntiles)
    return list(product(tile_arange, tile_arange))


def tile2box(xtile: int, ytile: int, zoom: int) -> dict:
    """Generate box for tile (xtile, ytile) at specified zoom level following OpenStreetMap
    slippy tile names conventions.

    Args:
        xtile int: xtile coordinate
        ytile int: ytile coordinate (int)
        zoom int: zoom level

    Returns:
        dict: with box name (zoom + xtile + ytile) as key and box (shapely.geometry.Polygon) as value.
    """
    nw = num2deg(xtile, ytile, zoom)
    ne = num2deg(xtile + 1, ytile, zoom)
    sw = num2deg(xtile, ytile + 1, zoom)
    se = num2deg(xtile + 1, ytile + 1, zoom)
    boxname = f"z{zoom}x{xtile}y{ytile}"
    return {boxname: sgeom.Polygon([nw, ne, sw, se]).envelope}


def slippy_map_tiles_to_quadkey(xtile: int, ytile: int, zoom: int) -> dict[str, Any]:
    """Convert tile coordinates to a bounding box with an associated quadkey.

    This function computes the bounding box for a given tile, represented by its
    xtile, ytile, and zoom values. The resulting box is associated with a quadkey
    following the Bing Maps Tile System.

    Args:
        xtile (int): The x-coordinate of the tile.
        ytile (int): The y-coordinate of the tile.
        zoom (int): The zoom level.

    Returns:
        dict: A dictionary with the quadkey as the key and the bounding box (as a
        shapely Polygon) as the value.
    """

    def _quadkey(x: int, y: int, z: int) -> str:
        """Convert tile coordinates to a quadkey."""
        quadkey = ""
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quadkey += str(digit)
        return quadkey

    nw = num2deg(xtile, ytile, zoom)
    ne = num2deg(xtile + 1, ytile, zoom)
    sw = num2deg(xtile, ytile + 1, zoom)
    se = num2deg(xtile + 1, ytile + 1, zoom)

    quadkey = _quadkey(xtile, ytile, zoom)
    return {quadkey: Polygon([nw, ne, sw, se]).envelope}


def make_slippy_map_tiles(zoom_level: int, crs: str) -> gpd.GeoDataFrame:
    """Generate boxes for a given zoom level.

    Args:
        zoom_level (int): Zoom level in accordance with OSM slippy tile conventions.
        crs (str): Coordinate reference system string.

    Returns:
        gpd.GeoDataFrame: Geodataframe containing the boxes.
    """
    tile_numbers = zoom2num(zoom_level)

    # Convert the tile coordinates to boxes by deriving the corners.
    boxes = [tile2box(*xytile, zoom=zoom_level) for xytile in tile_numbers]

    # Merge the list of dictionaries into one dict.
    boxes = {k: v for d in boxes for k, v in d.items()}

    # Load into GeoPandas and convert to desired CRS
    boxes = gpd.GeoDataFrame(
        boxes.items(), columns=["tilekey", "geometry"], crs="EPSG:4326"
    ).to_crs(crs)
    return boxes
