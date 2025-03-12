import datetime
import os
import pathlib
from typing import Any

import fsspec
import pystac
import tqdm
from dotenv import load_dotenv
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.scientific import ScientificExtension
from pystac.extensions.version import VersionExtension
from pystac.provider import ProviderRole
from pystac.stac_io import DefaultStacIO

from coastpy.libs import stac_table
from coastpy.libs.stac_table import InferDatetimeOptions
from coastpy.stac import ParquetLayout
from coastpy.stac.item import add_gpq_snapshot, create_tabular_item

# Load the environment variables from the .env file
load_dotenv()

# Get the SAS token and storage account name from environment variables
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
STORAGE_ACCOUNT_NAME = "coclico"
storage_options = {"account_name": STORAGE_ACCOUNT_NAME, "credential": sas_token}

# Container and URI configuration
VERSION = "2025-02-21"
DATETIME_STAC_CREATED = datetime.datetime.now(datetime.UTC)
DATETIME_DATA_CREATED = datetime.datetime(2025, 2, 21)
CONTAINER_NAME = "cet"
PREFIX = f"release/{VERSION}"
CONTAINER_URI = f"az://{CONTAINER_NAME}/{PREFIX}"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
LICENSE = "CC-BY-4.0"

# Collection information
COLLECTION_ID = "cet"
COLLECTION_TITLE = (
    "The CoCliCo coastal characteristics database to support erosion analysis"
)

# Transect and zoom configuration
TRANSECT_LENGTH = 2000
ZOOM = 9

DESCRIPTION = """
This database brings together existing information on erosion and relevant coastal characteristics in a GIS environment, facilitating broad-scale analysis.
"""
# Asset details
ASSET_TITLE = "CET"
ASSET_DESCRIPTION = "Parquet dataset with coastal erosion data for this region."
GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"

COLUMN_DESCRIPTIONS = [
    {
        "name": "source",
        "type": "string",
        "description": "Source of coastline data used in analysis.",
    },
    {
        "name": "country",
        "type": "string",
        "description": "ISO 2-letter country code where the segment is located.",
    },
    {
        "name": "covered",
        "type": "string",
        "description": "Indicates if the coastal classification has been applied: 'Y' (Yes), 'N' (No), 'N/A' (Not included).",
    },
    {
        "name": "seg_id",
        "type": "string",
        "description": "Unique segment ID assigned within each country.",
    },
    {
        "name": "seg_length",
        "type": "int",
        "description": "Length of the coastal segment in meters.",
    },
    {
        "name": "associated_floodplain",
        "type": "string",
        "description": "Indicates if the segment is associated with a floodplain ('Y' for Yes, 'N' for No).",
    },
    {
        "name": "local_floodplain",
        "type": "string",
        "description": "ID of the local floodplain adjacent to the coastline segment.",
    },
    {
        "name": "remote_floodplain_1",
        "type": "string",
        "description": "ID of the first remote floodplain associated with the coastline segment.",
    },
    {
        "name": "remote_floodplain_2",
        "type": "string",
        "description": "ID of the second remote floodplain associated with the coastline segment.",
    },
    {
        "name": "onshore_structure",
        "type": "string",
        "description": "Indicates presence of onshore engineered structures affecting coastal evolution ('Y' for Yes, 'N' for No).",
    },
    {
        "name": "offshore_structure",
        "type": "string",
        "description": "Indicates presence of offshore structures like breakwaters affecting coastal evolution ('Y' for Yes, 'N' for No).",
    },
    {
        "name": "harbour",
        "type": "string",
        "description": "Indicates presence of a permanent port or harbour structure ('Y' for Yes, 'N' for No).",
    },
    {
        "name": "geomorphological_class",
        "type": "string",
        "description": "Classification of the coastal geomorphology (e.g., Beach, Erodible cliffs, Dune system, Wetlands).",
    },
    {
        "name": "barrier",
        "type": "string",
        "description": "Description of the broad-scale coastal barrier feature, if present (e.g., Spit, Barrier Island, Tombolo).",
    },
    {
        "name": "primary_sediment_type",
        "type": "string",
        "description": "Primary sediment type of the coastal segment (e.g., Sand, Mud, Rock, Sand/Gravel).",
    },
    {
        "name": "secondary_sediment_type",
        "type": "string",
        "description": "Secondary sediment type, if applicable (e.g., rock platforms in a sandy beach).",
    },
    {
        "name": "historical_shoreline_change_regime",
        "type": "string",
        "description": "Historical trend of shoreline movement from 1984 to 2021: 'Ero' (Erosion), 'Acc' (Accretion), 'Sta' (Stable).",
    },
    {
        "name": "corine_code_18",
        "type": "string",
        "description": "Corine 2018 land cover classification code for the segment.",
    },
    {
        "name": "corine_code_simplified",
        "type": "string",
        "description": "Simplified reclassification of Corine 2018 land cover codes.",
    },
    {
        "name": "Future_erosion",
        "type": "float",
        "description": "Probability of future erosion occurring at the segment location.",
    },
    {
        "name": "Future_accretion",
        "type": "float",
        "description": "Probability of future accretion occurring at the segment location.",
    },
    {
        "name": "Future_stable",
        "type": "float",
        "description": "Probability of future stability at the segment location.",
    },
    {
        "name": "Notes",
        "type": "string",
        "description": "Additional notes or comments on the segment.",
    },
    {
        "name": "Local_floodplain_area_km2",
        "type": "float",
        "description": "Total area (km²) of the associated local floodplain.",
    },
    {
        "name": "geometry",
        "type": "geometry",
        "description": "Geospatial representation of the coastal segment in a MULTILINESTRING format.",
    },
]


ASSET_EXTRA_FIELDS = {
    "table:storage_options": {"account_name": "coclico"},
    "table:columns": COLUMN_DESCRIPTIONS,
}


def add_citation_extension(collection):
    """
    Add citation-related metadata to the STAC collection using the Citation Extension.
    """
    # Add the Scientific Extension to the collection
    ScientificExtension.add_to(collection)

    # Define the DOI and citation
    CITATION = "Hanson et al., in progress."

    # Add the DOI and citation to the collection's extra fields
    sci_ext = ScientificExtension.ext(collection, add_if_missing=True)
    sci_ext.citation = CITATION

    return collection


def create_collection(
    description: str | None = None, extra_fields: dict[str, Any] | None = None
) -> pystac.Collection:
    providers = [
        pystac.Provider(
            name="Deltares",
            roles=[
                ProviderRole.PROCESSOR,
                ProviderRole.LICENSOR,
            ],
            url="https://deltares.nl",
        ),
        pystac.Provider(
            name="CoCliCo",
            roles=[ProviderRole.PRODUCER, ProviderRole.LICENSOR, ProviderRole.HOST],
            url="https://coclicoservices.eu/",
        ),
    ]

    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, 90.0, 180.0, -90.0]]),
        pystac.TemporalExtent([[DATETIME_DATA_CREATED, None]]),
    )

    links = [
        pystac.Link(
            pystac.RelType.LICENSE,
            target="https://creativecommons.org/licenses/by/4.0/",
            media_type="text/html",
            title="CC BY 4.0 ",
        ),
    ]

    keywords = [
        "Coastal analytics",
        "Coastal erosion",
        "Coastal floodingDeltares",
        "CoCliCo",
        "GeoParquet",
    ]
    if description is None:
        description = DESCRIPTION

    collection = pystac.Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=description,
        license=LICENSE,
        providers=providers,
        extent=extent,
        catalog_type=pystac.CatalogType.RELATIVE_PUBLISHED,
    )

    collection.add_asset(
        "thumbnail",
        pystac.Asset(
            "https://coclico.blob.core.windows.net/assets/thumbnails/cet-thumbnail.jpeg",
            title="Thumbnail",
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
        ),
    )
    collection.links = links
    collection.keywords = keywords

    ItemAssetsExtension.add_to(collection)

    collection.extra_fields["item_assets"] = {
        "data": {
            "title": ASSET_TITLE,
            "description": ASSET_DESCRIPTION,
            "roles": ["data"],
            "type": stac_table.PARQUET_MEDIA_TYPE,
            **ASSET_EXTRA_FIELDS,
        }
    }

    if extra_fields:
        collection.extra_fields.update(extra_fields)

    collection = add_citation_extension(collection)
    version_ext = VersionExtension.ext(collection, add_if_missing=True)
    version_ext.version = VERSION

    # NOTE: Add schema validation after making a PR to the stac-table repo
    # collection.stac_extensions.append(stac_table.SCHEMA_URI)

    return collection


if __name__ == "__main__":
    storage_options = {"account_name": "coclico", "credential": sas_token}
    fs, token, [root] = fsspec.get_fs_token_paths(
        CONTAINER_URI, storage_options=storage_options
    )
    paths = fs.glob(CONTAINER_URI + "/**/*.parquet")
    uris = ["az://" + p for p in paths]

    STAC_DIR = pathlib.Path.home() / "dev" / "coclicodata" / "current"
    catalog = pystac.Catalog.from_file(str(STAC_DIR / "catalog.json"))

    stac_io = DefaultStacIO()
    layout = ParquetLayout()
    collection = create_collection(
        extra_fields={"storage_pattern": CONTAINER_URI + "/*.parquet"}
    )
    collection.validate_all()

    for uri in tqdm.tqdm(uris, desc="Processing files"):
        item = create_tabular_item(
            urlpath=uri,
            asset_title=ASSET_TITLE,
            asset_description=ASSET_DESCRIPTION,
            storage_options=storage_options,
            properties=None,
            item_extra_fields=None,
            asset_extra_fields=None,
            datetime=DATETIME_DATA_CREATED,
            infer_datetime=InferDatetimeOptions.no,
            alternate_links={"cloud": True},
        )

        item.validate()

        collection.add_item(item)

    collection.update_extent_from_items()

    collection = add_gpq_snapshot(
        collection, GEOPARQUET_STAC_ITEMS_HREF, storage_options
    )
    # TODO: there should be a cleaner method to remove the previous stac catalog and its items
    try:
        if catalog.get_child(collection.id):
            catalog.remove_child(collection.id)
            print(f"Removed child: {collection.id}.")
    except Exception:
        pass

    catalog.add_child(collection)

    collection.normalize_hrefs(str(STAC_DIR / collection.id), layout)

    collection.validate_all()

    catalog.save(
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
        dest_href=str(STAC_DIR),
        stac_io=stac_io,
    )
