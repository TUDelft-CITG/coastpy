import datetime
import os
import pathlib
import re
from typing import Any

import fsspec
import pystac
import tqdm
from dotenv import load_dotenv
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.scientific import Publication, ScientificExtension
from pystac.extensions.version import VersionExtension
from pystac.provider import ProviderRole
from pystac.stac_io import DefaultStacIO
from pystac.utils import now_in_utc

from coastpy.io.utils import PathParser
from coastpy.libs import stac_table
from coastpy.stac import ParquetLayout
from coastpy.stac.item import PARQUET_MEDIA_TYPE, add_gpq_snapshot

# Load the environment variables from the .env file
load_dotenv()

# Get the SAS token and storage account name from environment variables
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
STORAGE_ACCOUNT_NAME = "coclico"
storage_options = {"account_name": STORAGE_ACCOUNT_NAME, "credential": sas_token}

# Container and URI configuration
VERSION = "2025-01-01"
DATETIME_STAC_CREATED = datetime.datetime.now(datetime.UTC)
DATETIME_DATA_CREATED = datetime.datetime(2023, 2, 9)
CONTAINER_NAME = "coastal-grid"
PREFIX = f"release/{VERSION}"
CONTAINER_URI = f"az://{CONTAINER_NAME}/{PREFIX}"
LICENSE = "CC-BY-4.0"

# Collection information
COLLECTION_ID = "coastal-grid"
COLLECTION_TITLE = "Coastal Grid"

DESCRIPTION = """
The Coastal Grid dataset provides a global tiling system for geospatial analytics in coastal areas.
It supports scalable data processing workflows by offering structured grids at varying zoom levels
(5, 6, 7, 8, 9, 10) and buffer sizes (500m, 1000m, 2000m, 5000m, 10000m, 15000m).

Each tile contains information on intersecting countries, continents, and Sentinel-2 MGRS tiles
as nested JSON lists. The dataset is particularly suited for applications requiring global coastal
coverage, such as satellite-based coastal monitoring, spatial analytics, and large-scale data processing.

Key Features:
- Global coverage of the coastal zone, derived from OpenStreetMap's generalized coastline (2023-02).
- Precomputed intersections with countries, continents, and MGRS tiles.
- Designed for use in scalable geospatial workflows.

This dataset is structured as a STAC collection, with individual items for each zoom level and buffer
size combination. Users can filter items by the `zoom` and `buffer_size` fields in the STAC metadata.

Please consider the following citation when using this dataset:

Floris Reinier Calkoen, Arjen Pieter Luijendijk, Kilian Vos, Etiënne Kras, Fedor Baart,
Enabling coastal analytics at planetary scale, Environmental Modelling & Software, 2024,
106257, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2024.106257.
"""

ASSET_TITLE = "Coastal Grid"
ASSET_DESCRIPTION = (
    "Parquet dataset providing a global structured coastal grid for coastal analytics"
)

GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"

COLUMN_DESCRIPTIONS = [
    {
        "name": "coastal_grid:id",
        "type": "string",
        "description": "Unique identifier for each tile, derived from bounds and a deterministic hex suffix.",
    },
    {
        "name": "coastal_grid:quadkey",
        "type": "string",
        "description": "Mercator quadkey for each tile, indicating its spatial location and zoom level.",
    },
    {
        "name": "coastal_grid:bbox",
        "type": "object",
        "description": "Bounding box of the tile in WGS84 coordinates, represented as a dictionary.",
    },
    {
        "name": "coastal_grid:utm_epsg",
        "type": "int32",
        "description": "EPSG code for the UTM zone of the tile estimated by GeoPandas.estimate_utm_crs().",
    },
    {
        "name": "admin:countries",
        "type": "string",
        "description": """JSON list of countries intersecting the tile (e.g., '["CA", "US"]').""",
    },
    {
        "name": "admin:continents",
        "type": "string",
        "description": """JSON list of continents intersecting the tile (e.g., '["NA", "SA"]').""",
    },
    {
        "name": "s2:mgrs_tile",
        "type": "string",
        "description": """JSON list of Sentinel-2 MGRS tiles intersecting the tile (e.g., '["15XWL", "16XDR"]').""",
    },
    {
        "name": "geometry",
        "type": "geometry",
        "description": "Polygon geometry defining the spatial extent of the tile.",
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
    DOI = "10.1016/j.envsoft.2024.106257"
    CITATION = (
        "Floris Reinier Calkoen, Arjen Pieter Luijendijk, Kilian Vos, Etiënne Kras, Fedor Baart, "
        "Enabling coastal analytics at planetary scale, Environmental Modelling & Software, 2024, "
        "106257, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2024.106257."
    )

    # Add the DOI and citation to the collection's extra fields
    sci_ext = ScientificExtension.ext(collection, add_if_missing=True)
    sci_ext.doi = DOI
    sci_ext.citation = CITATION

    # Optionally add publications (if applicable)
    sci_ext.publications = [Publication(doi=DOI, citation=CITATION)]

    return collection


def create_collection(
    description: str | None = None, extra_fields: dict[str, Any] | None = None
) -> pystac.Collection:
    providers = [
        pystac.Provider(
            name="Deltares",
            roles=[
                ProviderRole.PRODUCER,
                ProviderRole.PROCESSOR,
                ProviderRole.HOST,
                ProviderRole.LICENSOR,
            ],
            url="https://deltares.nl",
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
        "Coastal change",
        "Coastal monitoring",
        "Satellite-derived shorelines",
        "Coastal zone",
        "Deltares",
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
            "https://coclico.blob.core.windows.net/assets/thumbnails/coastal-grid-thumbnail.jpeg",
            title="Thumbnail",
            media_type=pystac.MediaType.JPEG,
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


def create_item(
    urlpath: str,
    storage_options: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
    item_extra_fields: dict[str, Any] | None = None,
    asset_extra_fields: dict[str, Any] | None = None,
) -> pystac.Item:
    """Create a STAC Item"""

    if item_extra_fields is None:
        item_extra_fields = {}

    if properties is None:
        properties = {}

    pp = PathParser(urlpath, account_name=STORAGE_ACCOUNT_NAME)

    template = pystac.Item(
        id=pp.stac_item_id,
        properties=properties,
        geometry=None,
        bbox=None,
        datetime=DATETIME_DATA_CREATED,
        stac_extensions=[],
    )
    template.common_metadata.created = now_in_utc()

    item = stac_table.generate(
        uri=pp.to_cloud_uri(),
        template=template,
        infer_bbox=True,
        proj=True,
        infer_geometry=False,
        infer_datetime=stac_table.InferDatetimeOptions.no,
        datetime_column=None,
        metadata_created=DATETIME_STAC_CREATED,
        datetime=DATETIME_DATA_CREATED,
        count_rows=True,
        asset_key="data",
        asset_href=pp.to_cloud_uri(),
        asset_title=ASSET_TITLE,
        asset_description=ASSET_DESCRIPTION,
        asset_media_type=PARQUET_MEDIA_TYPE,
        asset_roles=["data"],
        asset_extra_fields=asset_extra_fields,
        storage_options=storage_options,
        validate=False,
    )
    assert isinstance(item, pystac.Item)

    # add descriptions to item properties
    if "table:columns" in ASSET_EXTRA_FIELDS and "table:columns" in item.properties:
        source_lookup = {
            col["name"]: col for col in ASSET_EXTRA_FIELDS["table:columns"]
        }

    for target_col in item.properties["table:columns"]:
        source_col = source_lookup.get(target_col["name"])
        if source_col:
            target_col.setdefault("description", source_col.get("description"))

    # Optionally add an HTTPS link if the URI uses a 'cloud protocol'
    if not item.assets["data"].href.startswith("https://"):
        item.add_link(
            pystac.Link(
                rel="alternate",
                target=pp.to_https_url(),
                title="HTTPS access",
                media_type=PARQUET_MEDIA_TYPE,
            )
        )
    return item


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
        match = re.search(r"_z([0-9]+)_([0-9]+m)\.parquet$", uri)
        if match:
            zoom_level = match.group(1)  # Capture the zoom level
            buffer_size = match.group(2)  # Capture the buffer size
            item_extra_fields = {"zoom_level": zoom_level, "buffer_size": buffer_size}

        item = create_item(
            uri,
            storage_options=storage_options,
            asset_extra_fields=ASSET_EXTRA_FIELDS,
            item_extra_fields=item_extra_fields,
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
