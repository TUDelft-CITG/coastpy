import datetime
import logging
import os
import pathlib
from typing import Any

import fsspec
import pystac
import pystac.media_type
import pystac.utils
import rasterio
import shapely
import stac_geoparquet
import xarray as xr

# uncomment these lines if you do not have coclicodata in development mode installed
# dev_dir = pathlib.Path.home() / "dev"  # set the path to the location where you would like to clone the package
# dev_dir.mkdir(parents=True, exist_ok=True)
# # Clone the repository
# os.system(f"git clone https://github.com/openearth/coclicodata.git {dev_dir / 'coclicodata'}")
# # Install the package in development mode
# os.system(f"pip install -e {dev_dir / 'coclicodata'}")
from coclicodata.coclico_stac.layouts import CoCliCoCOGLayout
from dotenv import load_dotenv
from pystac.extensions import raster
from pystac.extensions.file import FileExtension
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.scientific import ScientificExtension
from pystac.extensions.version import VersionExtension
from pystac.provider import ProviderRole
from pystac.stac_io import DefaultStacIO
from stactools.core.utils import antimeridian
from tqdm import tqdm

from coastpy.io.utils import PathParser
from coastpy.stac.item import PARQUET_MEDIA_TYPE

# Load the environment variables from the .env file
load_dotenv(override=True)

logging.getLogger("azure").setLevel(logging.WARNING)

# Get the SAS token and storage account name from environment variables
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
STORAGE_ACCOUNT_NAME = "coclico"
storage_options = {"account_name": STORAGE_ACCOUNT_NAME, "credential": sas_token}

# CoCliCo STAC
STAC_DIR = pathlib.Path.home() / "dev" / "coclicodata" / "current"

COLLECTION_ID = "deltares-delta-dtm"
COLLECTION_TITLE = "DeltaDTM: A global coastal digital terrain model"

ASSET_TITLE = "DeltaDTM"
ASSET_DESCRIPTION = "Deltares DeltaDTM with terrain elevation data for this region."
ASSET_EXTRA_FIELDS = {
    "xarray:storage_options": {"account_name": "coclico"},
}

# Container and URI configuration
VERSION = "v1.1"
DATETIME_STAC_CREATED = datetime.datetime.now(datetime.UTC)
DATETIME_DATA_CREATED = pystac.utils.now_in_utc()
CONTAINER_NAME = "deltares-delta-dtm"
CONTAINER_URI = f"az://{CONTAINER_NAME}/{VERSION}"
GEOPARQUET_STAC_ITEMS_HREF = f"az://items/{COLLECTION_ID}.parquet"
LICENSE = "CC-BY-4.0"

# Data specifics
NODATA_VALUE = -9999
RESOLUTION = 30
DATA_TYPE = raster.DataType.FLOAT32
UNIT = "m"

# NOTE: Since December 2024 we have changed to https hrefs instead of az blob storage paths
EXAMPLE_HREF = "https://coclico.blob.core.windows.net/deltares-delta-dtm/v1.1/DeltaDTM_v1_1_N03W052.tif"


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
            title="CC License",
        )
    ]

    keywords = [
        "Elevation",
        "...",
    ]

    description = """
    A global coastal digital terrain model, based on CopernicusDEM, ESA WorldCover, ICESat-2 and
    GEDI data. For more information, see Pronk et al. (2024) DeltaDTM: A global coastal digital terrain model.
    """

    collection = pystac.Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=description,
        license="CC-BY-4.0",
        providers=providers,
        extent=extent,
        catalog_type=pystac.CatalogType.RELATIVE_PUBLISHED,
    )

    collection.add_asset(
        "thumbnail",
        pystac.Asset(
            "https://coclico.blob.core.windows.net/assets/thumbnails/deltares-delta-dtm-thumbnail.jpeg",
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
            "type": pystac.MediaType.COG,
            **ASSET_EXTRA_FIELDS,
        }
    }

    if extra_fields:
        collection.extra_fields.update(extra_fields)

    ScientificExtension.add_to(collection)
    collection.extra_fields["sci:citation"] = (
        """Pronk, Maarten. 2024. “DeltaDTM v1.1: A Global Coastal Digital Terrain Model.” 4TU.ResearchData. https://doi.org/10.4121/21997565.V3."""
    )
    collection.extra_fields["sci:doi"] = "10.4121/21997565"
    collection.extra_fields["sci:publications"] = [
        {
            "doi": "10.1038/s41597-024-03091-9",
            "citation": """Pronk, Maarten, Aljosja Hooijer, Dirk Eilander, Arjen Haag, Tjalling de Jong, Michalis Vousdoukas, Ronald Vernimmen, Hugo Ledoux, and Marieke Eleveld. 2024. “DeltaDTM: A Global Coastal Digital Terrain Model.” Scientific Data 11 (1): 273. https://doi.org/10.1038/s41597-024-03091-9.""",
        }
    ]

    VersionExtension.add_to(collection)
    collection.extra_fields["version"] = "1.1"

    return collection


def create_item(block, item_id, antimeridian_strategy=antimeridian.Strategy.SPLIT):
    bbox = rasterio.warp.transform_bounds(block.rio.crs, 4326, *block.rio.bounds())
    geometry = shapely.geometry.mapping(shapely.make_valid(shapely.geometry.box(*bbox)))
    bbox = shapely.make_valid(shapely.box(*bbox)).bounds

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=DATETIME_DATA_CREATED,
        properties={},
    )
    item.common_metadata.created = DATETIME_STAC_CREATED

    antimeridian.fix_item(item, antimeridian_strategy)

    ext = ProjectionExtension.ext(item, add_if_missing=True)
    ext.bbox = block.rio.bounds()  # these are provided in the crs of the data
    ext.shape = [v for k, v in block.sizes.items() if k in ["y", "x"]]
    ext.epsg = block.rio.crs.to_epsg()
    ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*ext.bbox))
    ext.transform = list(block.rio.transform())[:6]
    ext.add_to(item)

    return item


def create_asset(
    item, asset_title, asset_href, nodata, resolution, data_type, nbytes=None, unit=None
):
    asset = pystac.Asset(
        href=asset_href,
        media_type=pystac.MediaType.COG,
        title=asset_title,
        roles=["data"],
    )
    asset.common_metadata.created = DATETIME_DATA_CREATED

    item.add_asset("data", asset)

    FileExtension.ext(asset, add_if_missing=True)

    if nbytes:
        asset.extra_fields["file:size"] = nbytes

    raster.RasterExtension.ext(asset, add_if_missing=True).bands = [
        raster.RasterBand.create(
            nodata=nodata,
            spatial_resolution=resolution,
            data_type=data_type,
            unit=unit,
        )
    ]

    return item


if __name__ == "__main__":
    fs, token, [root] = fsspec.get_fs_token_paths(
        CONTAINER_URI, storage_options=storage_options
    )

    fps = fs.glob(f"{root}/**/*.tif")

    stac_io = DefaultStacIO()  # CoCliCoStacIO()
    layout = CoCliCoCOGLayout()

    collection = create_collection()

    # NOTE: adjust logging level to avoid this warning:
    # INFO:rasterio._filepath:Object not found in virtual filesystem:
    # filename=b'ed4e8723-89d8-4075-b946-1ae1a21cca03/ed4e8723-89d8-4075-b946-1ae1a21cca03.aux'
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    for fp in tqdm(fps, desc="Processing file paths"):
        href = "az://" + fp
        urlpath = PathParser(href, account_name=STORAGE_ACCOUNT_NAME)

        with fs.open(fp, "rb") as f:
            block = xr.open_dataset(f, engine="rasterio")["band_data"].squeeze()

        item = create_item(block, urlpath.stac_item_id)
        item = create_asset(
            item,
            ASSET_TITLE,
            urlpath.to_https_url(),
            nodata=NODATA_VALUE,
            resolution=RESOLUTION,
            data_type=DATA_TYPE,
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
    gpq_items_asset.common_metadata.created = DATETIME_DATA_CREATED
    collection.add_asset("geoparquet-stac-items", gpq_items_asset)

    # Currently this is a link for testing purpose. When the data is available on the WMS
    # server it will be updated here.
    collection.add_link(
        pystac.Link(
            rel="service",
            target="https://coclico.avi.deltares.nl/geoserver/cfhp/wms?bbox={bbox-epsg-3857}&format=image/png&service=WMS&version=1.1.1&request=GetMap&srs=EPSG:3857&transparent=true&width=256&height=256&layers=cfhp:HIGH_DEFENDED_MAPS_Mean_spring_tide",
            title="Geoserver WMS Link",
            media_type="application/vnd.ogc.wms_xml",
        )
    )

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
