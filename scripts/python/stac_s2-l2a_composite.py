import datetime
import logging
import os
import pathlib
import re
from collections import defaultdict
from typing import Any

import fsspec
import pystac
import pystac.media_type
import stac_geoparquet
import xarray as xr
from coclicodata.coclico_stac.layouts import CoCliCoCOGLayout
from dotenv import load_dotenv
from pystac import Collection, Provider, ProviderRole
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.scientific import ScientificExtension
from pystac.extensions.version import VersionExtension
from pystac.stac_io import DefaultStacIO

from coastpy.io.utils import PathParser
from coastpy.stac.item import (
    create_cog_item,
)

# Load the environment variables from the .env file
load_dotenv(override=True)

logging.getLogger("azure").setLevel(logging.WARNING)

# Get the SAS token and storage account name from environment variables
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
STORAGE_ACCOUNT_NAME = "coclico"
storage_options = {"account_name": STORAGE_ACCOUNT_NAME, "credential": sas_token}

# CoCliCo STAC
STAC_DIR = pathlib.Path.home() / "dev" / "coclicodata" / "current"

COLLECTION_ID = "s2-l2a-composite"
COLLECTION_TITLE = "Sentinel-2 L2A Annual Composites"
COLLECTION_DESCRIPTION = """
A collection of composite Sentinel-2 L2A imagery processed using a median compositing technique.
"""

ASSET_DESCRIPTION = "Sentinel-2 L2A Annual Composites"
ASSET_EXTRA_FIELDS = {
    "xarray:storage_options": {"account_name": "coclico"},
}

# Configuration
STORAGE_ACCOUNT_NAME = "coclico"
CONTAINER_NAME = "s2-l2a-composite"
VERSION = "2025-01-11"
CONTAINER_URI = f"az://{CONTAINER_NAME}/release/{VERSION}"
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
        r"(?P<tile_id>\d{2}[A-Z]{3}-\d{2})_[0-9]{8}_[0-9]{8}_(?P<band>\w+)_\d{2}m\.tif$"
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
            common_attrs = ds.attrs.copy()

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

    return combined_dataset


if __name__ == "__main__":
    stac_io = DefaultStacIO()  # CoCliCoStacIO()
    layout = CoCliCoCOGLayout()

    collection = create_collection()

    fs = fsspec.filesystem("az", **storage_options)
    paths = fs.glob(f"{CONTAINER_URI}/*.tif")
    tiles = group_tif_files_by_tile(paths)

    for tile_id, band_paths in tiles.items():
        ds = load_bands_to_dataset(band_paths, storage_options)

        # Create a STAC item for the combined dataset
        urlpath = f"{CONTAINER_URI}/{tile_id}.tif"
        item = create_cog_item(
            ds,
            urlpath,
            storage_options=storage_options,
            nodata=NODATA_VALUE,
            data_type=DATA_TYPE,
            scale_factor=SCALE_FACTOR,
            unit=UNIT,
        )

        collection.add_item(item)
    collection.update_extent_from_items()

    items = list(collection.get_all_items())
    items_as_json = [i.to_dict() for i in items]
    item_extents = stac_geoparquet.to_geodataframe(items_as_json)

    with fsspec.open(GEOPARQUET_STAC_ITEMS_HREF, mode="wb", **storage_options) as f:
        item_extents.to_parquet(f)

    snapshot_pp = PathParser(
        GEOPARQUET_STAC_ITEMS_HREF, account_name=STORAGE_ACCOUNT_NAME
    )
    with fsspec.open(snapshot_pp.to_cloud_uri(), mode="wb", **storage_options) as f:
        item_extents.to_parquet(f)

    gpq_items_asset = pystac.Asset(
        snapshot_pp.to_https_url(),
        title="GeoParquet STAC items",
        description="Snapshot of the collection's STAC items exported to GeoParquet format.",
        media_type=PARQUET_MEDIA_TYPE,
        roles=["metadata"],
    )
    collection.add_asset("geoparquet-stac-items", gpq_items_asset)

    catalog = pystac.Catalog.from_file(str(STAC_DIR / "catalog.json"))

    # TODO: there should be a cleaner method to remove the previous stac catalog and its items
    try:
        if catalog.get_child(collection.id):
            catalog.remove_child(collection.id)
            print(f"Removed child: {collection.id}.")
    except Exception:
        pass

    catalog.add_child(collection)

    collection.normalize_hrefs(str(STAC_DIR / collection.id), strategy=layout)

    collection.validate_all()

    catalog.save(
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
        dest_href=str(STAC_DIR),
        stac_io=stac_io,
    )
