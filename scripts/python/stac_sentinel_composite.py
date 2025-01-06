import datetime
import os
import re
from collections import defaultdict

import dotenv
import fsspec
import pystac
import xarray as xr
from pystac import Collection, Provider, ProviderRole

from coastpy.stac.item import (
    add_cog_asset,
    create_cog_item,
)

dotenv.load_dotenv()

sas_token = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
storage_options = {"account_name": "coclico", "sas_token": sas_token}


# Configuration
STORAGE_ACCOUNT_NAME = "coclico"
VERSION = "2025-01-05"
CONTAINER_NAME = "tmp/typology/composite/release/slurm-2025-01-05"
COLLECTION_ID = "sentinel-composite"
COLLECTION_TITLE = "Sentinel-2 Annual Composites"
COLLECTION_DESCRIPTION = """
A collection of composite Sentinel-2 imagery processed using median compositing techniques.
"""
DATETIME_DATA_CREATED = datetime.datetime(2025, 1, 5, tzinfo=datetime.UTC)
DATETIME_STAC_CREATED = datetime.datetime.now(datetime.UTC)
LICENSE = "CC-BY-4.0"

# Providers
PROVIDERS = [
    Provider(
        name="Deltares",
        roles=[
            ProviderRole.PRODUCER,
            ProviderRole.PROCESSOR,
            ProviderRole.HOST,
        ],
        url="https://deltares.nl",
    ),
    Provider(
        name="Sentinel-2 L2A on Planetary Computer",
        roles=[ProviderRole.HOST],
        url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
    ),
]


def create_collection():
    """
    Create the STAC collection for composite Sentinel-2 data.
    """
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180, -90, 180, 90]]),
        pystac.TemporalExtent([[DATETIME_DATA_CREATED, None]]),
    )

    collection = Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=COLLECTION_DESCRIPTION,
        providers=PROVIDERS,
        license=LICENSE,
        extent=extent,
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
    )

    return collection


def group_tif_files_by_tile(paths: list[str]) -> dict[str, dict[str, str]]:
    """
    Group .tif file paths by tile ID.

    The tile IDs consist of an MGRS S2 tile name (e.g., 31UFU) followed by a numeric index (e.g., 00),
    separated by a dash. Each grouped dictionary entry contains band-specific file paths.

    Args:
        paths (List[str]): List of file paths to group.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary grouping file paths by tile ID, with bands as sub-keys.
    """
    # Define a regex pattern to extract tile ID and band
    tile_pattern = re.compile(
        r"(?P<tile_id>\d{2}[A-Z]{3}-\d{2})_(?P<band>\w+)_\d{2}m\.tif$"
    )

    grouped_files = defaultdict(dict)

    for path in paths:
        filename = path.split("/")[-1]  # Extract the filename from the path
        match = tile_pattern.search(filename)

        if match:
            tile_id = match.group("tile_id")
            band = match.group("band")
            grouped_files[tile_id][band] = path
        else:
            raise ValueError(
                f"File '{path}' does not match the expected naming convention."
            )

    return dict(grouped_files)


if __name__ == "__main__":
    fs = fsspec.filesystem("az", **storage_options)
    paths = fs.glob(f"{CONTAINER_NAME}/*.tif")

    tiles = group_tif_files_by_tile(paths)

    for tile_id, band_paths in tiles.items():
        print(f"Tile: {tile_id}")
        for _, path in band_paths.items():
            with fs.open(path) as f:
                ds = xr.open_dataset(f, engine="rasterio", chunks={})
                print("done")

    # Create collection and add items
    collection = create_collection()

    for tile_id, band_paths in tiles.items():
        # Combine bands into an xarray Dataset
        datasets = []
        for band_name, path in band_paths.items():
            with fs.open(path) as f:
                ds = xr.open_dataset(f, engine="rasterio").expand_dims(
                    {"band": [band_name]}
                )
                datasets.append(ds)

        combined_dataset = xr.concat(datasets, dim="band")

        # Create a STAC item for the combined dataset
        base_url = f"az://{CONTAINER_NAME}/{tile_id}"
        item = create_cog_item(
            combined_dataset, base_url, storage_options=storage_options
        )

        # Add each band as an asset
        for band_name, path in band_paths.items():
            add_cog_asset(item, combined_dataset.sel(band=band_name), band_name, path)

        collection.add_item(item)

    # Save and validate collection
    collection.normalize_hrefs("./stac-output")
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    collection.validate_all()

    print("Composite Sentinel-2 STAC collection created successfully!")
