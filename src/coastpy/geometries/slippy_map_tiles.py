#!/usr/bin/env python
# coding: utf-8


import math
import pathlib
from itertools import product, repeat

import geopandas as gpd
import numpy as np
import shapely.geometry as sgeom


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


# TODO: implement dynamic zero padding
def tile2box_qk(xtile: int, ytile: int, zoom: int) -> dict:
    nw = num2deg(xtile, ytile, zoom)
    ne = num2deg(xtile + 1, ytile, zoom)
    sw = num2deg(xtile, ytile + 1, zoom)
    se = num2deg(xtile + 1, ytile + 1, zoom)
    quadkey = str(xtile) + str(ytile)
    return {quadkey: sgeom.Polygon([nw, ne, sw, se]).envelope}


def make_boxes(zoom_level, crs):
    tile_numbers = zoom2num(zoom_level)

    # Convert the tile coordinates (xtile, ytile) to boxes by deriving the corners.
    # Note, tile2box function is wrapped within lambda function to unpack tuples
    # (xtile, ytile) in map
    boxes = list(map(lambda xytile: tile2box(*xytile, zoom=zoom_level), tile_numbers))

    # Result from previous function is a list of dictionaries. Here, merge into one dict.
    boxes = {k: v for d in boxes for k, v in d.items()}

    # Load into GeoPandas so that the results, for example, can be exported to GeoJSON
    boxes = gpd.GeoDataFrame(
        boxes.items(), columns=["quadkey", "geometry"], crs="EPSG:4326"
    ).to_crs(crs)
    return boxes


if __name__ == "__main__":
    # Set zoom level and generate xtiles and ytiles
    boxes = make_boxes(zoom_level=7, crs="epsg:3857")
    print("Done")
