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

from coastpy.libs import stac_table
from coastpy.stac import ParquetLayout
from coastpy.stac.item import add_gpq_snapshot, create_tabular_item

# Load the environment variables from the .env file
load_dotenv()

# Get the SAS token and storage account name from environment variables
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
STORAGE_ACCOUNT_NAME = "coclico"
storage_options = {"account_name": STORAGE_ACCOUNT_NAME, "credential": sas_token}

# Container and URI configuration
VERSION = "2025-08-06"
DATETIME_STAC_CREATED = datetime.datetime.now(datetime.UTC)
DATETIME_DATA_CREATED = datetime.datetime(2025, 4, 9)
CONTAINER_NAME = "edito/typology"
PREFIX = f"release/{VERSION}"
CONTAINER_URI = f"az://{CONTAINER_NAME}/{PREFIX}"
LICENSE = "CC-BY-4.0"

# Collection information
COLLECTION_ID = "global-coastal-typology"
COLLECTION_TITLE = "Global Coastal Typology"

DESCRIPTION = """
This dataset provides a globally consistent, high-resolution (100 m) coastal typology, derived from satellite imagery and elevation data using a deep learning model. It provides a foundational dataset for coastal-change analysis, erosion assessment, and more broader coastal vulnerability mapping and coastal adaptation in the face of accelerating climate change.

Using a supervised multi-task convolutional neural network, we classified four coastal attributes along the cross-shore profile for nearly 10 million transects from the Global Coastal Transect System (GCTS):
1.  **Sediment Type**: e.g., sandy, gravel or shingle; muddy; rocky; or, no sediment.
2.  **Coastal Type**: e.g., cliffed or steep, sediment plain, wetlands, or dune systems.
3.  **Built Environment**: Presence or absence of human development.
4.  **Coastal Defenses**: Presence or absence of human-made coastal defenses.

The model achieves strong predictive performance (F1 scores: 0.67-0.83). Results show that ~61% of the global coastline consists of soft, potentially erodible sediments. Among sandy, gravel, or shingle coasts, 20% are cliff-backed and 16.5% are located on built-up coasts.

Data are stored in a cloud-optimized, partitioned Parquet format. Tutorials and usage examples are available via the `coastpy` Python library: https://github.com/TUDelft-CITG/coastpy.

Please cite the associated publication when using this dataset:
Calkoen et al., 2025, *Mapping the world's coast: a global 100 m coastal typology derived from satellite data using deep learning*, Earth System Science Data (in review at https://essd.copernicus.org/preprints/essd-2025-388/).

DOI paper: https://doi.org/10.5194/essd-2025-388
DOI dataset: https://doi.org/10.5281/zenodo.15599096
"""

ASSET_TITLE = "Global Coastal Typology"

ASSET_DESCRIPTION = "Parquet dataset providing a global coastal typology derived from satellite imagery and elevation data. "

GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"

COLUMN_DESCRIPTIONS = [
    {
        "name": "transect_id",
        "type": "string",
        "description": "Unique identifier for each transect, constructed from coastline, segment, and distance.",
    },
    {
        "name": "geometry",
        "type": "geometry",
        "description": "Transect origin (Point Geometry) as a Well-Known Binary (WKB) LineString.",
    },
    {
        "name": "utm_epsg",
        "type": "int32",
        "description": "EPSG code representing the UTM Coordinate Reference System of the transect.",
    },
    {
        "name": "bbox",
        "type": "object",
        "description": "Bounding box of the transect geometry, defined by minimum and maximum x (longitude) and y (latitude) coordinates.",
    },
    {
        "name": "quadkey",
        "type": "string",
        "description": "Spatial index for the transect origin, following the Bing Maps Tile System.",
    },
    {
        "name": "continent",
        "type": "string",
        "description": "Name of the continent where the transect is located.",
    },
    {
        "name": "country",
        "type": "string",
        "description": "ISO alpha-2 country code for the transect's location.",
    },
    {
        "name": "common_country_name",
        "type": "string",
        "description": "Common name of the country where the transect is located.",
    },
    {
        "name": "common_region_name",
        "type": "string",
        "description": "Common name of the region where the transect is located.",
    },
    # Coastal Typology Variables
    {
        "name": "class:shore_type",
        "type": "str",
        "description": "Predicted shore type classification, which describes the material composing the shore (e.g., sandy sediments, rocky formations, or muddy sediments).",
    },
    {
        "name": "class:coastal_type",
        "type": "str",
        "description": "Predicted coastal type classification, which refers to the geomorphological features of the coast, which may be natural (e.g., cliffs, dunes) or human-influenced (e.g., engineered structures).",
    },
    {
        "name": "class:has_defense",
        "type": "boolean",
        "description": "Boolean flag indicating whether coastal defense structures (e.g., sea walls, breakwaters) are present to protect against erosion and flooding.",
    },
    {
        "name": "class:is_built_environment",
        "type": "boolean",
        "description": "Boolean flag indicating whether the coastal area is dominated by human-made structures or remains largely natural.",
    },
    # Shore Type Class Probabilities
    {
        "name": "class:prob_sandy_gravel_or_small_boulder_sediments",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the shore is composed of sandy sediments, gravel, or small boulders. Derived from the shore type classification.",
    },
    {
        "name": "class:prob_muddy_sediments",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the shore is composed of muddy sediments. Derived from the shore type classification.",
    },
    {
        "name": "class:prob_rocky_shore_platform_or_large_boulders",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the shore consists of rocky platforms or large boulders. Derived from the shore type classification.",
    },
    {
        "name": "class:prob_no_sediment_or_shore_platform",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the shore lacks sediment and consists mainly of shore platform features. Derived from the shore type classification.",
    },
    # Coastal Type Class Probabilities
    {
        "name": "class:prob_cliffed_or_steep",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast is cliffed or steep. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_moderately_sloped",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast is moderately sloped. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_bedrock_plain",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast is characterized by a bedrock plain. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_sediment_plain",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast is characterized by a sediment plain. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_dune",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast contains dune systems. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_wetland",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast includes wetland environments. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_inlet",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast contains an inlet or estuarine feature. Derived from the coastal type classification.",
    },
    {
        "name": "class:prob_engineered_structures",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coast includes engineered structures. Derived from the coastal type classification.",
    },
    # Binary Task Probabilities
    {
        "name": "class:prob_has_defense",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that coastal defense structures are present. Complements the binary 'class:has_defense' label.",
    },
    {
        "name": "class:prob_is_built_environment",
        "type": "float32",
        "description": "Model confidence (probability between 0 and 1) that the coastal area is built-up or dominated by human-made infrastructure. Complements the binary 'class:is_built_environment' label.",
    },
    {
        "name": "class:model",
        "type": "str",
        "description": "Identifier for the machine learning model used for coastal typology predictions.",
    },
    {
        "name": "class:datetime_created",
        "type": "datetime64[ms]",
        "description": "Timestamp indicating when the coastal typology predictions were generated.",
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
    # DOI = "https://doi.org/10.5194/essd-2025-388"
    CITATION = (
        "Calkoen, F. R., Luijendijk, A. P., Hanson, S., Nicholls, R. J., "
        "Moreno-Rodenas, A., De Heer, H., and Baart, F.: Mapping the world's coast: "
        "a global 100-m coastal typology derived from satellite data using deep learning, "
        "Earth Syst. Sci. Data Discuss. [preprint], "
        "https://doi.org/10.5194/essd-2025-388, in review, 2025."
    )

    # Add the DOI and citation to the collection's extra fields
    sci_ext = ScientificExtension.ext(collection, add_if_missing=True)

    # Add the DOI and citation to the collection's extra fields
    sci_ext = ScientificExtension.ext(collection, add_if_missing=True)
    sci_ext.citation = CITATION
    # sci_ext.doi = DOI  # NOTE: using doi brreaks

    # Optionally add publications (if applicable)
    # sci_ext.publications = [Publication(doi=DOI, citation=CITATION)]

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
        "Coastal Typology",
        "Coastal Erosion",
        "Coastal Vulnerability",
        "Coastal AdaptationCoastal Classification",
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

    # collection.add_asset(
    #     "thumbnail",
    #     pystac.Asset(
    #         "https://coclico.blob.core.windows.net/assets/thumbnails/coastal-grid-thumbnail.jpeg",
    #         title="Thumbnail",
    #         media_type=pystac.MediaType.JPEG,
    #     ),
    # )

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
    fs = fsspec.filesystem("az", **storage_options)
    paths = fs.glob(CONTAINER_URI + "/**/*.parquet")
    uris = ["az://" + p for p in paths]
    print(f"Found {len(uris)} Parquet files in {CONTAINER_URI}")

    STAC_DIR = pathlib.Path.home() / "dev" / "coclicodata" / "current"
    catalog = pystac.Catalog.from_file(str(STAC_DIR / "catalog.json"))
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
            item_extra_fields={"table:columns": COLUMN_DESCRIPTIONS},
            asset_extra_fields=ASSET_EXTRA_FIELDS,
            datetime=DATETIME_DATA_CREATED,
            infer_datetime=stac_table.InferDatetimeOptions.no,
            alternate_links={"CLOUD": True},
            force_bbox=False,
        )
        item.validate()
        collection.add_item(item)

    collection.update_extent_from_items()
    collection = add_gpq_snapshot(
        collection, GEOPARQUET_STAC_ITEMS_HREF, storage_options
    )

    catalog.add_child(collection)
    collection.normalize_hrefs(str(STAC_DIR / collection.id), layout)
    collection.validate_all()
    catalog.save(
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
        dest_href=str(STAC_DIR),
    )
