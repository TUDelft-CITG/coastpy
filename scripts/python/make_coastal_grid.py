import argparse
import datetime
import itertools
import json
import logging
import os
import random
import re
import uuid
from collections import defaultdict
from functools import partial
from typing import Literal

import dotenv
import fsspec
import geopandas as gpd
import pystac

from coastpy.geo.quadtiles import make_mercantiles
from coastpy.geo.quadtiles_utils import add_geo_columns
from coastpy.io.utils import name_bounds, name_bounds_with_hash  # noqa

# Configure logging for your specific module
logger = logging.getLogger(__name__)  # Replace __name__ with your module name


def configure_logging(verbosity: int):
    """Set logging level based on verbosity."""
    level = logging.DEBUG if verbosity else logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def short_id(seed: str, length: int = 3) -> str:
    """Generate a short deterministic ID based on a seed."""
    # Use hashlib for a consistent deterministic hash
    random.seed(seed)
    return uuid.UUID(int=random.getrandbits(128)).hex[:length]


def add_bounds(geometry, crs):
    """Create a unique deterministic proc_id for a geometry."""
    # NOTE: leave here because this would also be an option
    # bounds_name = name_bounds(geometry.bounds, crs)
    # NOTE: leave here because this would also be an option
    bounds_name = name_bounds_with_hash(geometry.bounds, crs)
    # NOTE: leave here because this was the old approach of adding a deterministic suffix
    # Use geometry bounds as a stable input for the seed
    # deterministic_suffix = short_id(str(geometry.bounds))
    # return f"{bounds_name}-{deterministic_suffix}"
    return bounds_name


# NOTE: leave here because this was the old approach of adding a grouped suffix
def add_grouped_index(df, id_column="coastal_grid:id"):
    """Enhance the ID column by appending a zero-padded group index."""
    df["new_index"] = df.groupby(id_column).cumcount() + 1
    df[id_column] = df[id_column] + "-" + df["new_index"].astype(str).str.zfill(3)
    df = df.drop(columns=["new_index"])
    return df


def add_proc_id(df, crs, zoom):
    """Add a unique deterministic proc_id to a DataFrame."""
    df["coastal_grid:id"] = df["geometry"].map(partial(add_bounds, crs=crs))
    # df = add_grouped_index(df)
    df["coastal_grid:id"] = f"z{zoom}" + "-" + df["coastal_grid:id"]
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
    with fsspec.open(href, mode="rb", account_name="coclico") as f:
        coastal_zone = gpd.read_parquet(f)
    return coastal_zone


def load_data():
    """Load static resources like countries and MGRS datasets."""
    try:
        logging.info("Loading static resources...")

        # Check if the countries file exists before reading
        countries_url = "https://coclico.blob.core.windows.net/public/countries.parquet"
        if not fsspec.filesystem("https").exists(countries_url):
            raise FileNotFoundError(f"Countries file not found: {countries_url}")

        with fsspec.open(countries_url, "rb") as f:
            countries = gpd.read_parquet(f)
        logging.info("Loaded countries dataset.")

        mgrs_url = "https://coclico.blob.core.windows.net/tiles/S2A_OPER_GIP_TILPAR_MPC.parquet"
        if not fsspec.filesystem("https").exists(mgrs_url):
            raise FileNotFoundError(f"MGRS file not found: {mgrs_url}")

        with fsspec.open(mgrs_url, "rb") as f:
            mgrs = gpd.read_parquet(f).to_crs(4326)
        logging.info("Loaded MGRS dataset.")

        utm_grid_url = "https://coclico.blob.core.windows.net/public/utm_grid.parquet"
        if not fsspec.filesystem("https").exists(utm_grid_url):
            raise FileNotFoundError(f"UTM Grid file not found: {utm_grid_url}")

        with fsspec.open(utm_grid_url, "rb") as f:
            utm_grid = gpd.read_parquet(f, columns=["geometry", "epsg"])
        utm_grid = utm_grid.dissolve("epsg").reset_index()

        return countries, mgrs, utm_grid

    except Exception as e:
        logging.error(f"Failed to load static resources: {e}")
        raise


VALID_BUFFER_SIZES = ["500m", "1000m", "2000m", "5000m", "10000m", "15000m"]
COLUMN_ORDER = [
    "coastal_grid:id",
    "coastal_grid:quadkey",
    "coastal_grid:bbox",
    "coastal_grid:utm_epsg",
    "admin:countries",
    "admin:continents",
    "s2:mgrs_tile",
    "geometry",
]


def main(
    buffer_sizes: Literal["500m", "1000m", "2000m", "5000m", "10000m", "15000m"],
    zooms: list[int],
    release,
    verbose: bool = False,
):
    """
    Main function to process coastal grids for given buffer sizes and zoom levels.
    """
    configure_logging(verbose)

    dotenv.load_dotenv()
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}

    # Load static resources
    countries, mgrs, utm_grid = load_data()

    # Retrieve all existing files in the release directory
    release_path = f"az://coastal-grid/release/{release}"
    fs = fsspec.filesystem("az", **storage_options)
    existing_files = {
        f"az://{file}" for file in fs.glob(f"{release_path}/coastal_grid_z*_*.parquet")
    }
    logger.info(f"Found {len(existing_files)} existing files.")

    # Generate expected file paths
    expected_files = {
        f"{release_path}/coastal_grid_z{zoom}_{buffer_size}.parquet"
        for buffer_size, zoom in itertools.product(buffer_sizes, zooms)
    }

    # Filter files to process
    files_to_process = expected_files - existing_files
    logger.info(f"{len(files_to_process)} files to process.")

    # Parse file paths to group by buffer size and zoom levels
    pattern = re.compile(r"coastal_grid_z(\d+)_(\d+m)\.parquet")
    buffer_to_zooms = defaultdict(list)

    for file_path in files_to_process:
        file_name = file_path.split("/")[-1]
        match = pattern.match(file_name)
        if match:
            zoom, buffer_size = match.groups()
            buffer_to_zooms[buffer_size].append(int(zoom))
        else:
            logger.error(f"Failed to parse file name {file_name}. Skipping.")

    # Process each buffer size
    for buffer_size, zoom_levels in buffer_to_zooms.items():
        logger.info(f"Processing buffer size: {buffer_size}")
        try:
            # Load the coastal zone data once for the buffer size
            coastal_zone = read_coastal_zone(buffer_size)
        except Exception as e:
            logger.error(
                f"Failed to read coastal zone for buffer size {buffer_size}: {e}"
            )
            continue

        # Process each zoom level for the current buffer size
        for zoom in zoom_levels:
            logger.info(f"Processing zoom level: {zoom}")
            try:
                # Generate grid and process tiles
                grid = make_mercantiles(zoom).rename(
                    columns={"quadkey": "coastal_grid:quadkey"}
                )
                tiles = clip_and_filter(grid, coastal_zone)
                tiles = add_geo_columns(tiles, geo_columns=["bbox"]).rename(
                    columns={"bbox": "coastal_grid:bbox"}
                )

                # # Add unique IDs and UTM EPSG codes
                tiles = add_proc_id(tiles, tiles.crs, zoom)

                if tiles["coastal_grid:id"].duplicated().any():
                    raise ValueError("Duplicate IDs found in the tiles.")

                tiles = tiles.sort_values("coastal_grid:quadkey")

                points = tiles.representative_point().to_frame("geometry")
                utm_epsg = gpd.sjoin(points, utm_grid).drop(columns=["index_right"])[
                    "epsg"
                ]
                tiles["coastal_grid:utm_epsg"] = utm_epsg

                # Aggregate and add additional columns
                tiles = add_divisions(tiles, countries)
                tiles = add_mgrs(tiles, mgrs)
                tiles = tiles[COLUMN_ORDER]

                # Save the processed tiles
                output_path = (
                    f"{release_path}/coastal_grid_z{zoom}_{buffer_size}.parquet"
                )
                with fsspec.open(output_path, mode="wb", **storage_options) as f:
                    tiles.to_parquet(f)

                logger.info(f"Saved: {output_path}")

            except Exception as e:
                logger.error(
                    f"Failed to process zoom level {zoom} for buffer size {buffer_size}: {e}"
                )


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

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for this module"
    )
    args = parser.parse_args()

    main(args.buffer_size, args.zoom, args.release, args.verbose)
