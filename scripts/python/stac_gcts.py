import dataclasses
import datetime
import os
import pathlib
import re
from typing import Any

import fsspec
import pystac
import stac_geoparquet
from dotenv import load_dotenv
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.scientific import ScientificExtension
from pystac.extensions.version import VersionExtension
from pystac.provider import ProviderRole
from pystac.stac_io import DefaultStacIO

from coastpy.libs import stac_table
from coastpy.stac import ParquetLayout

# Load the environment variables from the .env file
load_dotenv()

# Get the SAS token and storage account name from environment variables
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
storage_account_name = "coclico"
storage_options = {"account_name": storage_account_name, "credential": sas_token}

# NOTE:
TEST_RELEASE = True

# Container and URI configuration
CONTAINER_NAME = "gcts"
RELEASE_DATE = "2024-08-02"
PREFIX = f"release/{RELEASE_DATE}"
CONTAINER_URI = f"az://{CONTAINER_NAME}/{PREFIX}"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
LICENSE = "CC-BY-4.0"

# Collection information
COLLECTION_ID = "gcts"
COLLECTION_TITLE = "Global Coastal Transect System (GCTS)"

# Transect and zoom configuration
TRANSECT_LENGTH = 2000
ZOOM = 9

DESCRIPTION = """
Cross-shore coastal transects are essential to coastal monitoring, offering a consistent
reference line to measure coastal change, while providing a robust foundation to map
coastal characteristics and derive coastal statistics thereof. The Global Coastal Transect
System consists of more than 11 million cross-shore coastal transects uniformly spaced at
100-m intervals alongshore, for all OpenStreetMap coastlines that are longer than 5 kilometers.
The dataset is more extensively described Calkoen et al., 2024. "Enabling Coastal Analytics
at Planetary Scale" available [here](https://doi.org/10.1016/j.envsoft.2024.106257).
"""
# Asset details
ASSET_TITLE = "GCTS"
ASSET_DESCRIPTION = f"Parquet dataset with coastal transects ({TRANSECT_LENGTH} m) at 100 m alongshore resolution for this region."

# GeoParquet STAC items
if TEST_RELEASE:
    GEOPARQUET_STAC_ITEMS_HREF = f"az://items-test/{COLLECTION_ID}.parquet"
else:
    GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"

COLUMN_DESCRIPTIONS = [
    {
        "name": "transect_id",
        "type": "string",
        "description": "A unique identifier for each transect, constructed from three key components: the 'coastline_id', 'segment_id', and 'interpolated_distance'. The 'coastline_id' corresponds to the FID in OpenStreetMap (OSM) and is prefixed with 'cl'. The 'segment_id' indicates the segment of the OSM coastline split by a UTM grid, prefixed with 's'. The 'interpolated_distance' represents the distance from the starting point of the coastline to the transect, interpolated along the segment, and is prefixed with 'tr'. The complete structure is 'cl[coastline_id]s[segment_id]tr[interpolated_distance]', exemplified by 'cl32946s04tr08168547'. This composition ensures each transect name is a distinct and informative representation of its geographical and spatial attributes.",
    },
    {
        "name": "lon",
        "type": "float",
        "description": "Longitude of the transect origin.",
    },
    {
        "name": "lat",
        "type": "float",
        "description": "Latitude of the transect origin.",
    },
    {
        "name": "bearing",
        "type": "float",
        "description": "North bearing of the transect from the landward side in degrees, with the north as reference.",
    },
    {
        "name": "geometry",
        "type": "byte_array",
        "description": "Well-Known Binary (WKB) representation of the transect as a linestring geometry.",
    },
    {
        "name": "osm_coastline_is_closed",
        "type": "bool",
        "description": "Indicates whether the source OpenStreetMap (OSM) coastline, from which the transects were derived, forms a closed loop. A value of 'true' suggests that the coastline represents an enclosed area, such as an island.",
    },
    {
        "name": "osm_coastline_length",
        "type": "int32",
        "description": "Represents the total length of the source OpenStreetMap (OSM) coastline, that is summed across various UTM regions. It reflects the aggregate length of the original coastline from which the transects are derived.",
    },
    {
        "name": "utm_epsg",
        "type": "int32",
        "description": "EPSG code representing the UTM Coordinate Reference System for the transect.",
    },
    {
        "name": "bbox",
        "type": "struct<minx: double, miny: double, maxx: double, maxy: double>",
        "description": "Bounding box of the transect geometry, given by minimum and maximum coordinates in x (longitude) and y (latitude).",
    },
    {
        "name": "quadkey",
        "type": "string",
        "description": "QuadKey corresponding to the transect origin location at zoom 12, following the Bing Maps Tile System for spatial indexing.",
    },
    {
        "name": "continent",
        "type": "string",
        "description": "Name of the continent in which the transect is located.",
    },
    {
        "name": "country",
        "type": "string",
        "description": "ISO alpha-2 country code for the country in which the transect is located. The country data are extracted from Overture Maps (divisions).",
    },
    {
        "name": "common_country_name",
        "type": "string",
        "description": "Common country name (EN) in which the transect is located. The country data are extracted from Overture Maps (divisions).",
    },
    {
        "name": "common_region_name",
        "type": "string",
        "description": "Common region name (EN) in which the transect is located. The regions are extracted from Overture Maps (divisions).",
    },
]


ASSET_EXTRA_FIELDS = {
    "table:storage_options": {"account_name": "coclico"},
    "table:columns": COLUMN_DESCRIPTIONS,
}


@dataclasses.dataclass
class PathParts:
    """
    Parses a path into its component parts, supporting variations with and without hive partitioning,
    and with and without geographical bounds.

    Attributes:
        path (str): The full path to parse.
        container (str | None): The storage container or bucket name.
        prefix (str | None): The path prefix between the container and the file name.
        name (str | None): The file name including its extension.
        stac_item_id (str | None): The identifier used for STAC, which is the file name without the .parquet extension.
    """

    path: str
    container: str | None = None
    prefix: str | None = None
    name: str | None = None
    stac_item_id: str | None = None

    def __post_init__(self) -> None:
        # Strip any protocol pattern like "az://"
        stripped_path = re.sub(r"^\w+://", "", self.path)
        split = stripped_path.rstrip("/").split("/")

        # Extract the container which is the first part of the path
        self.container = split[0]

        # The file name is the last element in the split path
        self.name = split[-1]

        # The prefix is everything between the container and the filename
        self.prefix = "/".join(split[1:-1]) if len(split) > 2 else None

        # stac_item_id is the filename without the .parquet extension
        self.stac_item_id = self.name.replace(".parquet", "")


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

    start_datetime = datetime.datetime.strptime(RELEASE_DATE, "%Y-%m-%d")

    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, 90.0, 180.0, -90.0]]),
        pystac.TemporalExtent([[start_datetime, None]]),
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
        "Cloud technology",
        "Coastal change",
        "Coastal monitoring",
        "Satellite-derived shorelines",
        "Low elevation coastal zone",
        "Data management",
        "Transects",
        "GCTS",
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
            "https://coclico.blob.core.windows.net/assets/thumbnails/gcts-thumbnail.jpeg",
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

    ScientificExtension.add_to(collection)
    # TODO: revise citation upon publication

    CITATION = """
    Floris Reinier Calkoen, Arjen Pieter Luijendijk, Kilian Vos, Etiënne Kras, Fedor Baart,
    Enabling coastal analytics at planetary scale, Environmental Modelling & Software, 2024,
    106257, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2024.106257.
    (https://www.sciencedirect.com/science/article/pii/S1364815224003189)
    """

    # NOTE: we could make a separate DOI for the transects and then link the paper in
    # sci:citations as a feature. However, for now I (Floris) prefer to use the same DOI.
    collection.extra_fields["sci:citation"] = CITATION
    # collection.extra_fields["sci:doi"] = "https://doi.org/10.1016/j.envsoft.2024.106257"

    collection.stac_extensions.append(stac_table.SCHEMA_URI)

    VersionExtension.add_to(collection)
    collection.extra_fields["version"] = RELEASE_DATE

    return collection


def create_item(
    asset_href: str,
    storage_options: dict[str, Any] | None = None,
    asset_extra_fields: dict[str, Any] | None = None,
) -> pystac.Item:
    """Create a STAC Item

    For

    Args:
        asset_href (str): The HREF pointing to an asset associated with the item

    Returns:
        Item: STAC Item object
    """

    parts = PathParts(asset_href)

    properties = {
        "title": ASSET_TITLE,
        "description": ASSET_DESCRIPTION,
    }

    dt = datetime.datetime.strptime(RELEASE_DATE, "%Y-%m-%d")
    # shape = shapely.box(*bbox)
    # geometry = shapely.geometry.mapping(shape)
    template = pystac.Item(
        id=parts.stac_item_id,
        properties=properties,
        geometry=None,
        bbox=None,
        datetime=dt,
        stac_extensions=[],
    )

    item = stac_table.generate(
        uri=asset_href,
        template=template,
        infer_bbox=True,
        infer_geometry=None,
        datetime_column=None,
        infer_datetime=stac_table.InferDatetimeOptions.no,
        count_rows=True,
        asset_key="data",
        asset_extra_fields=asset_extra_fields,
        proj=True,
        storage_options=storage_options,
        validate=False,
    )
    assert isinstance(item, pystac.Item)

    item.common_metadata.created = datetime.datetime.now(datetime.UTC)

    # add descriptions to item properties
    if "table:columns" in ASSET_EXTRA_FIELDS and "table:columns" in item.properties:
        source_lookup = {
            col["name"]: col for col in ASSET_EXTRA_FIELDS["table:columns"]
        }

    for target_col in item.properties["table:columns"]:
        source_col = source_lookup.get(target_col["name"])
        if source_col:
            target_col.setdefault("description", source_col.get("description"))

    # TODO: make configurable upstream
    item.assets["data"].title = ASSET_TITLE
    item.assets["data"].description = ASSET_DESCRIPTION

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

    collection = create_collection(extra_fields={"container_uri": CONTAINER_URI})
    collection.validate_all()

    for uri in uris:
        item = create_item(uri, storage_options=storage_options)
        item.validate()
        collection.add_item(item)

    collection.update_extent_from_items()

    items = list(collection.get_all_items())
    items_as_json = [i.to_dict() for i in items]
    item_extents = stac_geoparquet.to_geodataframe(items_as_json)

    with fsspec.open(GEOPARQUET_STAC_ITEMS_HREF, mode="wb", **storage_options) as f:
        item_extents.to_parquet(f)

    collection.add_asset(
        "geoparquet-stac-items",
        pystac.Asset(
            GEOPARQUET_STAC_ITEMS_HREF,
            title="GeoParquet STAC items",
            description="Snapshot of the collection's STAC items exported to GeoParquet format.",
            media_type=PARQUET_MEDIA_TYPE,
            roles=["data"],
        ),
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
