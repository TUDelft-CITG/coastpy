import ast
import json
from typing import Literal

import fsspec
import geopandas as gpd
import pandas as pd
import pystac
import pystac.item
import pystac.media_type
import rioxarray  # noqa


def read_coastal_grid(zoom: int, buffer_size: str) -> gpd.GeoDataFrame:
    """Load coastal zone data layer for a specific buffer size."""
    catalog = pystac.Catalog.from_file(
        "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
    )
    collection = catalog.get_child("coastal-grid")
    if not collection:
        raise ValueError("Coastal zone collection not found")

    item = collection.get_item(f"coastal_grid_z{zoom}_{buffer_size}")
    if not item:
        raise ValueError(
            f"Coastal zone item for zoom {zoom} with {buffer_size} not found"
        )
    href = item.assets["data"].href
    with fsspec.open(href, mode="rb") as f:
        grid = gpd.read_parquet(f)

        # Apply JSON parsing for specific columns
    grid["admin:continents"] = grid["admin:continents"].apply(
        lambda x: json.loads(x) if x else []
    )
    grid["admin:countries"] = grid["admin:countries"].apply(
        lambda x: json.loads(x) if x else []
    )
    return grid


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


def filter_by_admins(
    grid: gpd.GeoDataFrame,
    include_countries=None,
    include_continents=None,
    exclude_countries=None,
    exclude_continents=None,
) -> gpd.GeoDataFrame:
    """Filter the coastal grid based on countries and continents.

    Exclude logic: only exclude rows if ALL countries/continents in the row are in the exclude list.
    """
    if include_continents:
        grid = grid[
            grid["admin:continents"].apply(
                lambda x: any(c in include_continents for c in x)
            )
        ]
    if include_countries:
        grid = grid[
            grid["admin:countries"].apply(
                lambda x: any(c in include_countries for c in x)
            )
        ]
    if exclude_continents:
        grid = grid[
            ~grid["admin:continents"].apply(
                lambda x: set(x).issubset(set(exclude_continents))
            )
        ]
    if exclude_countries:
        grid = grid[
            ~grid["admin:countries"].apply(
                lambda x: set(x).issubset(set(exclude_countries))
            )
        ]
    return grid


def coastal_grid_by_mgrs_tile(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid["s2:mgrs_tile"] = grid["s2:mgrs_tile"].apply(ast.literal_eval)
    grid = grid.explode("s2:mgrs_tile", ignore_index=True)
    grid["tile_id"] = grid[["s2:mgrs_tile", "coastal_grid:id"]].agg("_".join, axis=1)
    return grid


def dissolve_by_mgrs_tile(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Union geometries per unique MGRS tile."""
    grid = coastal_grid_by_mgrs_tile(grid)
    return (
        grid[["geometry", "s2:mgrs_tile"]]
        .dissolve(by="s2:mgrs_tile", as_index=False)
        .reset_index(drop=True)
    )


def add_tile_id(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add unique `tile_id` to each geometry in the MGRS tile."""
    exploded = grid.explode(index_parts=False)
    exploded["tile_id"] = (
        exploded["s2:mgrs_tile"]
        + "-"
        + exploded.groupby("s2:mgrs_tile").cumcount().apply(lambda x: f"{x:02d}")
    )
    exploded = exploded.reset_index(drop=True)
    return exploded


def add_utm_epsg(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add `utm_epsg` column to the coastal grid using `s2:mgrs_tile`."""

    def compute_epsg(mgrs_tile: str) -> int | None:
        if len(mgrs_tile) < 3:
            return None
        zone = int(mgrs_tile[:2])
        hemisphere = 326 if mgrs_tile[2] >= "N" else 327
        return hemisphere * 100 + zone

    grid = grid.copy()
    grid["utm_epsg"] = grid["s2:mgrs_tile"].apply(compute_epsg)
    return grid


def make_coastal_mgrs_grid(
    buffer_size: str,
    zoom: int,
    include_countries=None,
    include_continents=None,
    exclude_countries=None,
    exclude_continents=None,
) -> gpd.GeoDataFrame:
    """
    Wrapper to a coastal MGRS tile system as a GeoDataFrame.
    """

    grid = read_coastal_grid(zoom=zoom, buffer_size=buffer_size)

    grid = filter_by_admins(
        grid,
        include_countries,
        include_continents,
        exclude_countries,
        exclude_continents,
    )

    grid = dissolve_by_mgrs_tile(grid)
    grid = add_tile_id(grid)
    grid = add_utm_epsg(grid)
    return grid


def get_default_processing_priorities(grid: gpd.GeoDataFrame) -> dict[str, int]:
    """
    Generate a dictionary mapping tile IDs to processing priorities.

    Processing priority is assigned in descending order as follows:
    - 10: EU countries except Russia, Sweden, and Norway
    - 9:  EU countries except Russia
    - 8:  Remaining European continent (within latitude -70 to 70)
    - 7:  Rest of the world (within latitude -70 to 70)
    - 6:  All remaining tiles

    Args:
        grid (gpd.GeoDataFrame): GeoDataFrame containing grid information with `tile_id` as index.

    Returns:
        dict[str, int]: Mapping of tile IDs to processing priority.
    """
    priority_mapping = {}

    # Ensure necessary columns exist
    required_columns = {"admin:continents", "admin:countries"}
    if not required_columns.issubset(grid.columns):
        raise ValueError(
            f"Missing required columns: {required_columns - set(grid.columns)}"
        )

    # Define masks for each priority level
    eu_mask = grid["admin:continents"].apply(lambda x: "EU" in x)
    russia_mask = grid["admin:countries"].apply(lambda x: "RU" in x)
    norway_sweden_mask = grid["admin:countries"].apply(
        lambda x: any(c in ["NO", "SE"] for c in x)
    )
    latitude_mask = grid.index.isin(grid.cx[:, -70:70].index)

    def assign_priority(priority: int, mask: pd.Series):
        """Assign priority only if the tile ID is not already assigned."""
        for tid in grid.loc[mask].index:
            if tid not in priority_mapping:
                priority_mapping[tid] = priority

    # Assign priorities in descending order, ensuring no overwrites
    assign_priority(10, eu_mask & ~russia_mask & ~norway_sweden_mask)
    assign_priority(9, eu_mask & ~russia_mask)
    assign_priority(8, eu_mask & latitude_mask)
    assign_priority(7, ~eu_mask & latitude_mask)

    # Assign remaining tiles to lowest priority (6)
    assigned_ids = set(priority_mapping.keys())
    remaining_ids = set(grid.index) - assigned_ids
    for tid in remaining_ids:
        priority_mapping[tid] = 6

    return priority_mapping
