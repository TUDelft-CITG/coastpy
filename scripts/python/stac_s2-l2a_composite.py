import fsspec.utils
from analytics.config import configure_dask, configure_rio, load_env

load_env()
configure_dask()
configure_rio(cloud=True)
import argparse
import datetime
import json
import os
import re
import time
from collections import defaultdict
from typing import Any

import dask.bag as db
import fsspec
import pandas as pd
import pystac
import pystac.media_type
import xarray as xr
from dask import delayed
from dask.distributed import Client
from distributed import LocalCluster
from pystac import Collection, Item, Provider, ProviderRole
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.scientific import ScientificExtension
from pystac.extensions.version import VersionExtension
from pystac.stac_io import DefaultStacIO

from coastpy.stac.item import add_gpq_snapshot, create_cog_item
from coastpy.stac.layouts import COGLayout
from coastpy.utils.dask_utils import summarize_dask_cluster


def format_date_range(date_range: str) -> str:
    """Convert ISO date range to YYYYMMDD_YYYYMMDD format."""
    return "_".join(pd.to_datetime(date_range.split("/")).strftime("%Y%m%d"))


COLLECTION_ID = "s2-l2a-composite"
COLLECTION_TITLE = "Sentinel-2 L2A Annual Composites"
COLLECTION_DESCRIPTION = """
A collection of composite Sentinel-2 L2A imagery processed using a median compositing technique.
"""

ASSET_DESCRIPTION = "Sentinel-2 L2A Annual Composites"
ASSET_EXTRA_FIELDS = {
    "xarray:storage_options": {"account_name": "coclico"},
}

DATETIME_STAC_CREATED = datetime.datetime.now(datetime.UTC)
GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"
LICENSE = "CC-BY-4.0"

# Data specifics
NODATA_VALUE = -9999
SCALE_FACTOR = 0.0001
RESOLUTION = 10
DATA_TYPE = "int16"
UNIT = "m"

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"

DESCRIPTION = """
An annual composite dataset created using CoastPy Simple Composite from Sentinel-2 Level-2A data hosted on the Planetary Computer.
The composite includes 6 bands (RGB, NIR, SWIR16, SWIR22) and is derived by selecting the least cloudy 10 scenes per orbital ground track.
"""

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
        name="Microsoft",
        roles=[ProviderRole.HOST],
        url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
    ),
]

# Thumbnail URL
THUMBNAIL_URL = "https://example.com/thumbnail.jpeg"

# Extra fields for items
ASSET_EXTRA_FIELDS = {
    "raster:bands": [
        {"name": "red", "description": "Red band"},
        {"name": "green", "description": "Green band"},
        {"name": "blue", "description": "Blue band"},
        {"name": "nir", "description": "Near Infrared band"},
        {"name": "swir16", "description": "Shortwave Infrared 1.6 µm band"},
        {"name": "swir22", "description": "Shortwave Infrared 2.2 µm band"},
    ]
}

CITATION_TEXT = "Calkoen, Floris et al., 'In progress', "


def add_citation_extension(collection: pystac.Collection) -> pystac.Collection:
    """
    Add citation-related metadata using the Scientific Extension.
    """
    ScientificExtension.add_to(collection)
    sci_ext = ScientificExtension.ext(collection, add_if_missing=True)
    # sci_ext.doi = CITATION_DOI
    sci_ext.citation = CITATION_TEXT
    # sci_ext.publications = [
    #     Publication(doi=CITATION_DOI, citation=CITATION_TEXT)
    # ]
    return collection


def create_collection(
    description: str | None = None, extra_fields: dict[str, Any] | None = None
) -> pystac.Collection:
    """
    Create the STAC collection for Sentinel-2 Annual Composite data.
    """
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[DATETIME_STAC_CREATED, None]]),
    )

    if description is None:
        description = DESCRIPTION

    collection = Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=description,
        providers=PROVIDERS,
        license=LICENSE,
        extent=extent,
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
    )

    collection.add_asset(
        "thumbnail",
        pystac.Asset(
            href=THUMBNAIL_URL,
            title="Thumbnail",
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
        ),
    )

    # Add item assets extension
    ItemAssetsExtension.add_to(collection)
    collection.extra_fields["item_assets"] = {
        "data": {
            "title": "Annual Composite Bands",
            "description": "Cloud-optimized GeoTIFF containing composite bands.",
            "roles": ["data"],
            "type": pystac.MediaType.COG,
            **ASSET_EXTRA_FIELDS,
        }
    }

    # Add citation metadata
    collection = add_citation_extension(collection)

    # Add version extension
    VersionExtension.add_to(collection)
    version_ext = VersionExtension.ext(collection, add_if_missing=True)
    version_ext.version = "1.0"

    # Allow additional custom fields
    if extra_fields:
        collection.extra_fields.update(extra_fields)

    return collection


def extract_groups(pattern, urlpath: str) -> dict[str, str | None]:
    """Extract groups from a file path using the regex pattern."""
    match = pattern.search(urlpath)
    if not match:
        raise ValueError(f"Cannot extract groups from urlpath: {urlpath}")
    groups = match.groupdict()
    return {k: v for k, v in groups.items() if v is not None}


def group_tif_files_by_tile(
    storage_pattern: str, filename_pattern, storage_options
) -> dict[str, dict[str, str]]:
    """
    Group tiles by their Tile IDs and associated bands based on files in storage.

    Args:
        storage_pattern (str): The storage pattern to list files from storage.
        filename_pattern (str): The regex pattern to extract tile ID and band from the filename.
        storage_options (dict): Storage options to pass to fsspec.

    Returns:
        Dict[str, Set[str]]: Mapping of tile IDs to their associated bands.
    """
    protocol = fsspec.utils.get_protocol(storage_pattern)
    fs = fsspec.filesystem(protocol, **storage_options)
    files = fs.glob(storage_pattern)

    pattern = re.compile(filename_pattern)

    if not files:
        print("No files found.")
        return {}

    tiles = defaultdict(dict)
    for f in files:
        try:
            groups = extract_groups(pattern, f)
            tile_id = groups["tile_id"]
            band = groups.get("band")
            if tile_id and band:
                tiles[tile_id][band] = f
            elif tile_id:
                tiles[tile_id] = f
        except ValueError as e:
            print(f"Warning: Could not parse file {f}: {e}")

    return tiles


def update_attributes(ds: xr.Dataset) -> xr.Dataset:
    """
    Update attribute names in the dataset and its variables by replacing the first underscore
    with a semicolon for attributes starting with 'eo_' or 'composite_'.

    Args:
        ds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Dataset with updated attribute names.
    """
    # Update global attributes
    attrs_copy = ds.attrs.copy()
    for attr in attrs_copy:
        if attr.startswith(("eo_", "composite_")):
            new_attr = attr.replace("_", ":", 1)
            ds.attrs[new_attr] = ds.attrs.pop(attr)

    # Update variable-specific attributes
    for var in ds.data_vars:
        var_attrs_copy = ds[var].attrs.copy()
        for attr in var_attrs_copy:
            if attr.startswith(("eo_", "composite_")):
                new_attr = attr.replace("_", ":", 1)
                ds[var].attrs[new_attr] = ds[var].attrs.pop(attr)

    return ds


def load_bands_to_dataset(
    band_paths: dict[str, str],
    storage_options: dict,
) -> xr.Dataset:
    """
    Load multiple GeoTIFF files as a single xarray.Dataset with bands as variables.
    """
    fs = fsspec.filesystem("az", **storage_options)

    # Initialize lists for data arrays and common attributes
    data_arrays = []
    common_attrs = None

    for band_name, path in band_paths.items():
        # Open each GeoTIFF lazily
        with fs.open(path) as f:
            ds = xr.open_dataset(f, engine="rasterio", chunks={})

        # Extract the "band_data" variable and rename it to the band name
        da = ds["band_data"].squeeze(dim="band", drop=True).rename(band_name)

        # Copy band-level attributes
        da.attrs["band_name"] = band_name

        # Collect dataset-level attributes from the first dataset
        if common_attrs is None:
            common_attrs = da.attrs.copy()

        data_arrays.append(da)

    # Combine all DataArrays into a single Dataset
    combined_dataset = xr.merge(data_arrays)

    # Ensure consistent coordinates (x, y, spatial_ref)
    reference_coords = data_arrays[0].coords
    for coord in ["x", "y", "spatial_ref"]:
        if not all(
            da.coords[coord].identical(reference_coords[coord]) for da in data_arrays
        ):
            raise ValueError(f"Inconsistent '{coord}' coordinates across bands.")

    combined_dataset = combined_dataset.assign_coords(
        {
            "x": reference_coords["x"],
            "y": reference_coords["y"],
            "spatial_ref": reference_coords["spatial_ref"],
        }
    )

    # Add common dataset-level attributes
    if common_attrs:
        combined_dataset.attrs.update(common_attrs)

    del combined_dataset.attrs["band_name"]

    # Replaces underscores in attribute names with colons to comply with STAC
    combined_dataset = update_attributes(combined_dataset)

    return combined_dataset


# Functions for the workflow
@delayed
def load_bands_to_dataset_delayed(
    band_paths: dict, storage_options: dict
) -> xr.Dataset:
    """Lazy loading of datasets using Dask Delayed."""
    return load_bands_to_dataset(band_paths, storage_options)


@delayed
def create_stac_item_delayed(
    ds: xr.Dataset, urlpath: str, storage_options
) -> Item | None:
    """Create a STAC item from a dataset lazily."""
    try:
        return create_cog_item(
            ds,
            urlpath,
            storage_options=storage_options,
            nodata=NODATA_VALUE,
            data_type=DATA_TYPE,
            scale_factor=SCALE_FACTOR,
            unit=UNIT,
        )
    except Exception as e:
        print(f"Error creating item: {e}")
        return None


@delayed
def write_stac_item_to_storage(
    item: Item, stac_item_container: str, storage_options: dict
) -> str | None:
    """Write a STAC item to cloud storage as a JSON file."""
    if item:
        item_path = f"{stac_item_container}/{item.id}.json"
        try:
            with fsspec.open(item_path, mode="w", **storage_options) as f:
                json.dump(item.to_dict(), f)
            return item_path
        except Exception as e:
            print(f"Error writing item to storage: {e}")
            return None


def read_stac_item_from_storage(file: str, filesystem) -> Item:
    with filesystem.open(file) as f:
        item_dict = json.load(f)
    return Item.from_dict(item_dict)


def filter_existing_tiles(
    tiles: dict[str, dict], stac_item_container, storage_options: dict[str, str]
) -> dict[str, dict]:
    """Filter out tiles that already have STAC items in the cloud storage."""
    fs = fsspec.filesystem("az", **storage_options)
    existing_files = fs.glob(f"{stac_item_container}/*.json")
    TILE_ID_PATTERN = (
        r"(?P<tile_id>\d{2}[A-Za-z]{3}_z\d+-(?:n|s)\d{2}(?:w|e)\d{3}-[a-z0-9]{6})"
    )
    pattern = re.compile(TILE_ID_PATTERN)

    existing_ids = set()
    for file in existing_files:
        match = pattern.search(file)
        if match:
            tile_id = match.group("tile_id")
            existing_ids.add(tile_id)
        else:
            raise ValueError(f"Cannot extract tile ID from file: {file}")

    return {
        tile_id: bands
        for tile_id, bands in tiles.items()
        if tile_id not in existing_ids
    }


def create_stac_items(
    s2a_composite_container,
    s2a_filename_pattern,
    s2a_required_bands,
    stac_item_container,
    client,
    storage_options: dict[str, str],
):
    """Main function for creating STAC items."""
    start_time = time.time()

    # List and filter tiles
    storage_pattern = f"{s2a_composite_container}/*.tif"
    tiles = group_tif_files_by_tile(
        storage_pattern, s2a_filename_pattern, storage_options
    )
    print(f"Found {len(tiles)} tiles.")

    tiles = {
        tile_id: bands
        for tile_id, bands in tiles.items()
        if set(s2a_required_bands).issubset(bands.keys())
    }
    print(f"Found {len(tiles)} tiles with all required bands.")

    # Filter out tiles with existing STAC items
    tiles = filter_existing_tiles(tiles, stac_item_container, storage_options)
    print(f"Processing {len(tiles)} new tiles.")

    stac_id_pattern = r"(?P<stac_id>\d{2}[A-Za-z]{3}_z\d+-(?:n|s)\d{2}(?:w|e)\d{3}-[a-z0-9]{6}_\d{8}_\d{8})"

    # Create tasks
    tasks = []
    keys = list(tiles.keys())[104:106]
    tiles = {k: v for k, v in tiles.items() if k in keys}

    for _, band_paths in tiles.items():
        ds = load_bands_to_dataset_delayed(band_paths, storage_options)
        first_href = band_paths[next(iter(band_paths.keys()))]
        match = re.search(stac_id_pattern, first_href)
        if not match:
            raise ValueError(f"Could not extract STAC ID from URL: {first_href}")
        stac_id = match.group("stac_id")
        urlpath = f"{s2a_composite_container}/{stac_id}.tif"
        item = create_stac_item_delayed(ds, urlpath, storage_options)
        task = write_stac_item_to_storage(item, stac_item_container, storage_options)
        tasks.append(task)

    # Execute tasks

    futures = client.compute(tasks, retries=2, sync=False)
    results = client.gather(futures, errors="skip")  # skip failures

    print(
        f"Create STAC items (n={len(results)}) completed in {time.time() - start_time:.2f} seconds."
    )


def create_collection_with_items(
    stac_item_container, stac_collection_directory, client, storage_options
):
    fs = fsspec.filesystem("az", **storage_options)
    files = list(fs.glob(f"{stac_item_container}/*.json"))
    print("Found", len(files), "STAC items.")

    # Create a Dask bag from the list of files
    bag = db.from_sequence(files, npartitions=10)

    # Map the read_stac_item_from_storage function over the bag
    items_bag = bag.map(lambda file: read_stac_item_from_storage(file, fs))

    def validate(item):
        if not isinstance(item, Item):
            raise ValueError(f"Item is not a Pystac.Item: {item}")
        if not item.validate():
            raise ValueError(f"Invalid item: {item}")
        return item

    items_bag = items_bag.map(validate)

    # Compute the bag to get the list of pystac.Item objects
    items = client.compute(items_bag, sync=True, retries=3)

    stac_io = DefaultStacIO()  # CoCliCoStacIO()
    layout = COGLayout()

    collection = create_collection()

    for item in items:
        collection.add_item(item)

    collection.update_extent_from_items()

    collection = add_gpq_snapshot(
        collection, GEOPARQUET_STAC_ITEMS_HREF, storage_options
    )

    catalog = pystac.Catalog.from_file(str(stac_collection_directory / "catalog.json"))

    # TODO: there should be a cleaner method to remove the previous stac catalog and its items
    try:
        if catalog.get_child(collection.id):
            catalog.remove_child(collection.id)
            print(f"Removed child: {collection.id}.")
    except Exception:
        pass

    catalog.add_child(collection)

    collection.normalize_hrefs(
        str(stac_collection_directory / collection.id), strategy=layout
    )

    catalog.save(
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
        dest_href=str(stac_collection_directory),
        stac_io=stac_io,
    )


def create_local_client(**kwargs):
    """
    Create a Dask local cluster and client with an optional custom config file.
    """
    cluster = LocalCluster(**kwargs)
    client = Client(cluster)
    return cluster, client


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a STAC collection for Sentinel-2 L2A composites."
    )

    parser.add_argument(
        "container", help="Source container with the S2A Composite imagery."
    )
    parser.add_argument("release", help="Release version.")
    parser.add_argument("stac_dir", help="Directory to store the STAC collection.")

    parser.add_argument(
        "--create-items",
        action="store_true",
        help="Create STAC items.",
    )
    parser.add_argument(
        "--create-collection",
        action="store_true",
        help="Create a STAC collection with items.",
    )

    parser.add_argument(
        "--n-workers", type=int, default=None, help="Number of Dask workers."
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Number of threads per worker.",
    )
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = parse_args()

    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "credential": sas_token}

    # Configuration
    VERSION = args.release
    CONTAINER_NAME = args.container
    S2A_COMPOSITE_CONTAINER = f"az://{CONTAINER_NAME}/release/{VERSION}"
    STAC_ITEM_CONTAINER = (
        f"az://tmp/stac/{S2A_COMPOSITE_CONTAINER.replace('az://', '')}/items"
    )
    DATETIME_RANGE = "2023-01-01/2024-01-01"
    BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "SCL"]
    S2A_REQUIRED_BANDS = [b for b in BANDS if b != "SCL"]

    date_range = format_date_range(DATETIME_RANGE)

    # NOTE: this is very important to get right
    S2A_FILENAME_PATTERN = (
        rf"(?P<tile_id>\d{{2}}[A-Za-z]{{3}}_z\d+-(?:n|s)\d{{2}}(?:w|e)\d{{3}}-[a-z0-9]{{6}})"  # Tile ID
        rf"(?:_{date_range})?"  # Optional date range
        r"(?:_(?P<band>[a-z0-9]+))?"  # Optional band
        r"(?:_10m\.tif)?"  # Optional resolution
    )

    # STAC DEFINITIONS
    STAC_DIR = args.stac_dir

    cluster, client = create_local_client(
        n_workers=args.n_workers, threads_per_worker=args.threads_per_worker
    )
    summarize_dask_cluster(client)

    if args.create_items:
        print("Creating STAC items...")
        create_stac_items(
            s2a_composite_container=S2A_COMPOSITE_CONTAINER,
            s2a_filename_pattern=S2A_FILENAME_PATTERN,
            s2a_required_bands=S2A_REQUIRED_BANDS,
            stac_item_container=STAC_ITEM_CONTAINER,
            client=client,
            storage_options=storage_options,
        )
    if args.create_collection:
        print("Creating STAC collection with items...")
        create_collection_with_items(
            stac_item_container=STAC_ITEM_CONTAINER,
            stac_collection_directory=STAC_DIR,
            client=client,
            storage_options=storage_options,
        )
        print(f"STAC collection created and saved successfully at {STAC_DIR}.")

    print(f"Workflow completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
