import argparse
import datetime
import itertools
import json
import logging
import os
import random
import uuid
from functools import partial
from typing import Literal

import dotenv
import fsspec
import geopandas as gpd
import pystac

from coastpy.geo.quadtiles import make_mercantiles
from coastpy.geo.quadtiles_utils import add_geo_columns
from coastpy.io.utils import name_bounds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def short_id(seed: str, length: int = 3) -> str:
    """Generate a short deterministic ID based on a seed."""
    # Use hashlib for a consistent deterministic hash
    random.seed(seed)
    return uuid.UUID(int=random.getrandbits(128)).hex[:length]


def add_proc_id(geometry, crs):
    """Create a unique deterministic proc_id for a geometry."""
    bounds_name = name_bounds(geometry.bounds, crs)
    # Use geometry bounds as a stable input for the seed
    deterministic_suffix = short_id(str(geometry.bounds))
    return f"{bounds_name}-{deterministic_suffix}"


def load_data(
    uri: str, storage_options=None, to_crs: str | None = None
) -> gpd.GeoDataFrame:
    """
    Load a geospatial dataset and optionally transform its CRS.

    Args:
        uri (str): URI of the dataset.
        storage_options (dict, optional): Storage options for fsspec.
        to_crs (str, optional): CRS to transform the dataset.

    Returns:
        GeoDataFrame: The loaded dataset.
    """
    with fsspec.open(uri, "rb", **(storage_options or {})) as f:
        df = gpd.read_parquet(f)
    if to_crs:
        df = df.to_crs(to_crs)
    logger.info(f"Loaded {uri} with {len(df)} features.")
    return df


def clip_and_filter(grid, coastal_zone):
    """
    Clip the grid tiles by the coastal zone.
    """
    # Filter the grid tiles by intersecting coastline
    filtered_tiles = (
        gpd.sjoin(grid, coastal_zone, how="inner", predicate="intersects")
        .drop_duplicates(subset="coastal_grid:quadkey")
        .drop(columns=["index_right"])
    )
    logger.info(f"Filtered grid tiles: {len(filtered_tiles)} features.")

    # Clip the tiles by the coastal zone
    clipped_tiles = gpd.overlay(filtered_tiles, coastal_zone, how="intersection")
    clipped_tiles = clipped_tiles.explode(index_parts=False).reset_index(drop=True)
    logger.info(f"Clipped tiles: {len(clipped_tiles)} features.")
    return clipped_tiles


def _aggregate_spatial_data(tiles, spatial_data, groupby_column, attributes):
    """Perform spatial join and aggregate attributes."""
    joined = gpd.sjoin(tiles, spatial_data, how="inner").drop(columns=["index_right"])
    aggregated = joined.groupby(groupby_column).agg(attributes)
    return tiles.merge(aggregated, on=groupby_column, how="left")


def add_divisions(tiles, countries):
    """Add country and continent info to tiles."""
    attributes = {
        "admin:countries": lambda x: json.dumps(list(set(x))),
        "admin:continents": lambda x: json.dumps(list(set(x))),
    }
    countries = countries[["country", "continent", "geometry"]].rename(
        columns={"country": "admin:countries", "continent": "admin:continents"}
    )
    return _aggregate_spatial_data(
        tiles,
        countries,
        "coastal_grid:id",
        attributes,
    )


def add_mgrs(tiles, s2grid):
    """Add Sentinel-2 MGRS names to tiles."""
    attributes = {"Name": lambda x: json.dumps(list(set(x)))}
    tiles = _aggregate_spatial_data(
        tiles, s2grid[["Name", "geometry"]], "coastal_grid:id", attributes
    )
    tiles = tiles.rename(columns={"Name": "s2:mgrs_tile"})
    return tiles


def read_coastal_zone(
    buffer_size: Literal["500m", "1000m", "2000m", "5000m", "10000m", "15000m"],
):
    """
    Load the coastal zone data layer for a specific buffer size.
    """
    coclico_catalog = pystac.Catalog.from_file(
        "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
    )
    coastal_zone_collection = coclico_catalog.get_child("coastal-zone")
    if coastal_zone_collection is None:
        msg = "Coastal zone collection not found"
        raise ValueError(msg)
    item = coastal_zone_collection.get_item(f"coastal_zone_{buffer_size}")
    if item is None:
        msg = f"Coastal zone item for {buffer_size} not found"
        raise ValueError(msg)
    href = item.assets["data"].href
    with fsspec.open(href, mode="rb") as f:
        coastal_zone = gpd.read_parquet(f)
    return coastal_zone


VALID_BUFFER_SIZES = ["500m", "1000m", "2000m", "5000m", "10000m", "15000m"]
COLUMN_ORDER = [
    "coastal_grid:id",
    "coastal_grid:quadkey",
    "coastal_grid:bbox",
    "admin:countries",
    "admin:continents",
    "s2:mgrs_tile",
    "geometry",
]


def main(
    buffer_sizes: Literal["500m", "1000m", "2000m", "5000m", "10000m", "15000m"],
    zooms: list[int],
    release,
):
    """
    Main function to process coastal grids for given buffer sizes and zoom levels.
    """
    dotenv.load_dotenv()
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}

    for buffer_size, zoom in itertools.product(buffer_sizes, zooms):
        # Validate zoom and buffer_size
        if buffer_size not in VALID_BUFFER_SIZES:
            msg = f"Invalid buffer size: {buffer_size}"
            raise ValueError(msg)
        if not (1 <= zoom <= 18):
            msg = f"Zoom level must be between 1 and 18. Got: {zoom}"
            raise ValueError(msg)

        print(f"Processing buffer size: {buffer_size}, zoom level: {zoom}")

        # Construct storage path
        storage_urlpath = f"az://coastal-grid/release/{release}/coastal_grid_z{zoom}_{buffer_size}.parquet"

        # Load datasets
        coastal_zone = read_coastal_zone(buffer_size)  # type: ignore
        with fsspec.open(
            "https://coclico.blob.core.windows.net/public/countries.parquet", "rb"
        ) as f:
            countries = gpd.read_parquet(f)
        with fsspec.open(
            "https://coclico.blob.core.windows.net/tiles/S2A_OPER_GIP_TILPAR_MPC.parquet",
            "rb",
        ) as f:
            mgrs = gpd.read_parquet(f).to_crs(4326)

        grid = make_mercantiles(zoom).rename(
            columns={"quadkey": "coastal_grid:quadkey"}
        )

        # Clip and filter
        tiles = clip_and_filter(grid, coastal_zone)

        # Add geographical columns and unique IDs
        tiles = add_geo_columns(tiles, geo_columns=["bbox"]).rename(
            columns={"bbox": "coastal_grid:bbox"}
        )
        tiles = tiles.sort_values("coastal_grid:quadkey")
        add_proc_id_partial = partial(add_proc_id, crs=tiles.crs)
        tiles["coastal_grid:id"] = tiles.geometry.map(add_proc_id_partial)

        # Aggregate tiles with country and continent data
        tiles = add_divisions(tiles, countries)
        tiles = add_mgrs(tiles, mgrs)
        tiles = tiles[COLUMN_ORDER]

        # Save the processed tiles
        with fsspec.open(storage_urlpath, mode="wb", **storage_options) as f:
            tiles.to_parquet(f)

        logger.info(f"Saved: {storage_urlpath}")


if __name__ == "__main__":

    def validate_date(date_string):
        """
        Validates the date string to ensure it's in the 'yyyy-mm-dd' format.
        """
        try:
            # Attempt to parse the date in 'yyyy-mm-dd' format
            datetime.datetime.strptime(date_string, "%Y-%m-%d")
            return date_string  # Return the valid date string
        except ValueError:
            msg = f"Invalid date format: {date_string}. Expected format: yyyy-mm-dd."
            raise ValueError(msg) from None

    parser = argparse.ArgumentParser(description="Generate coastal grid datasets.")
    parser.add_argument(
        "--buffer_size",
        nargs="+",
        type=str,
        required=True,
        help="List of buffer sizes (e.g., 500m 1000m 2000m 5000m 10000m 15000m).",
    )
    parser.add_argument(
        "--zoom",
        nargs="+",
        type=int,
        required=True,
        help="List of zoom levels (e.g., 4 6 8).",
    )
    parser.add_argument(
        "--release",
        type=validate_date,
        required=True,
        help="Release date in yyyy-mm-dd format (e.g., 2024-12-09).",
    )
    args = parser.parse_args()

    main(args.buffer_size, args.zoom, args.release)
