import ast
import json

import fsspec
import geopandas as gpd
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


def filter_by_admins(
    grid: gpd.GeoDataFrame,
    include_countries=None,
    include_continents=None,
    exclude_countries=None,
    exclude_continents=None,
) -> gpd.GeoDataFrame:
    """Filter the coastal grid based on countries and continents."""
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
                lambda x: any(c in exclude_continents for c in x)
            )
        ]
    if exclude_countries:
        grid = grid[
            ~grid["admin:countries"].apply(
                lambda x: any(c in exclude_countries for c in x)
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
    include_countries=None,
    include_continents=None,
    exclude_countries=None,
    exclude_continents=None,
) -> gpd.GeoDataFrame:
    """
    Wrapper to a coastal MGRS tile system as a GeoDataFrame.
    """

    grid = read_coastal_grid(zoom=9, buffer_size=buffer_size)

    # Apply JSON parsing for specific columns
    grid["admin:continents"] = grid["admin:continents"].apply(
        lambda x: json.loads(x) if x else []
    )
    grid["admin:countries"] = grid["admin:countries"].apply(
        lambda x: json.loads(x) if x else []
    )

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
