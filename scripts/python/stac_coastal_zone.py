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
STORAGE_ACCOUNT_NAME = "coclico"
storage_options = {"account_name": STORAGE_ACCOUNT_NAME, "credential": sas_token}

# Container and URI configuration
CONTAINER_NAME = "coastal-zone"
RELEASE_DATE = "2024-12-08"
PREFIX = f"release/{RELEASE_DATE}"
CONTAINER_URI = f"az://{CONTAINER_NAME}/{PREFIX}"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
LICENSE = "CC-BY-4.0"

# Collection information
COLLECTION_ID = "coastal-zone"
COLLECTION_TITLE = "Coastal Zone"

DESCRIPTION = """
The Coastal Zone dataset provides a vectorized representation of coastal zones at multiple buffer distances.
It is derived from a generalized version of the OpenStreetMap coastline (2023-02) and serves as a valuable
tool for masking other datasets or for spatial analysis in coastal regions.

This STAC collection includes multiple layers, each corresponding to a specific buffer distance:
500m, 1000m, 2000m, 5000m, 10000m, and 15000m. The buffer distance defines the zone's extent, with the
total width being twice the buffer distance (e.g., a 5000m buffer results in a zone 10km wide).

Each layer in the collection is stored as a separate item and can be filtered using the `buffer_size`
field in the item's properties. These layers contain only the geometry, enabling seamless integration with
other geospatial data.

Please consider the following citation when using this dataset:

Floris Reinier Calkoen, Arjen Pieter Luijendijk, Kilian Vos, Etiënne Kras, Fedor Baart,
Enabling coastal analytics at planetary scale, Environmental Modelling & Software, 2024,
106257, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2024.106257.
(https://www.sciencedirect.com/science/article/pii/S1364815224003189)

"""

ASSET_TITLE = "Coastal Zone"
ASSET_DESCRIPTION = (
    "Parquet dataset with coastal zone geometries for multiple buffer distances."
)

GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"

COLUMN_DESCRIPTIONS = [
    {
        "name": "geometry",
        "type": "byte_array",
        "description": "Well-Known Binary (WKB) representation of the transect as a linestring geometry.",
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
    href: str | None = None

    _base_href = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

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

        self.href = f"{self._base_href}/{self.container}/{self.prefix}/{self.name}"


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
            "https://coclico.blob.core.windows.net/assets/thumbnails/coastal-zone-thumbnail.jpeg",
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
    extra_properties: dict[str, Any] | None = None,
) -> pystac.Item:
    """Create a STAC Item

    For

    Args:
        asset_href (str): The HREF pointing to an asset associated with the item

    Returns:
        Item: STAC Item object
    """

    parts = PathParts(asset_href)

    if extra_properties is None:
        extra_properties = {}

    dt = datetime.datetime.strptime(RELEASE_DATE, "%Y-%m-%d")
    # shape = shapely.box(*bbox)
    # geometry = shapely.geometry.mapping(shape)
    template = pystac.Item(
        id=parts.stac_item_id,
        properties=extra_properties,
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
        asset_href=parts.href,
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
        match = re.search(r"_([0-9]+m)\.parquet$", uri)
        if match:
            buffer_size = match.group(1)
            extra_properties = {"buffer_size": buffer_size}
        item = create_item(
            uri,
            storage_options=storage_options,
            asset_extra_fields=ASSET_EXTRA_FIELDS,
            extra_properties=extra_properties,
        )
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
