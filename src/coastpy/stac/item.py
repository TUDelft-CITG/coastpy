from datetime import datetime
from enum import Enum
from typing import Any

import antimeridian
import fsspec
import pystac
import pystac.catalog
import pystac.common_metadata
import pystac.utils
import rasterio
import rasterio.warp
import shapely
import stac_geoparquet
import xarray as xr
from affine import Affine
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
from shapely.geometry import Polygon, box, mapping
from shapely.validation import make_valid

from coastpy.io.utils import PathParser, get_datetimes
from coastpy.libs.stac_table import InferDatetimeOptions, generate
from coastpy.utils.xarray_utils import get_nodata

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


class AssetProtocol(Enum):
    HTTPS = "https"
    CLOUD = "cloud"
    FILE = "file"

    def get_uri(self, pp: PathParser, file_dir: str | None = None) -> str:
        """Returns the URI based on the selected protocol."""
        if self == AssetProtocol.HTTPS:
            return pp.to_https_url()
        elif self == AssetProtocol.CLOUD:
            return pp.to_cloud_uri()
        elif self == AssetProtocol.FILE:
            return pp.to_filepath() if file_dir is None else pp.to_filepath(file_dir)
        raise ValueError(f"Unsupported protocol: {self}")


def add_alternate_links(
    item: pystac.Item, pp: PathParser, alternate_links: dict[str, bool | str]
) -> pystac.Item:
    """Adds alternate links to a STAC Item."""
    for key, value in alternate_links.items():
        if key in AssetProtocol.__members__:
            alt_proto = AssetProtocol[key]
            target = (
                alt_proto.get_uri(pp)
                if value is True
                else alt_proto.get_uri(pp, value)
                if isinstance(value, str)
                else None
            )
        else:
            target = value

        if isinstance(target, str):
            item.add_link(
                pystac.Link(
                    rel="alternate",
                    target=target,
                    title=f"{key.capitalize()} Access",
                    media_type=pystac.MediaType.COG,
                )
            )
    return item


def get_rotated_bbox(
    dataset: xr.Dataset | xr.DataArray,
) -> tuple[float, float, float, float]:
    """Computes the bounding box of a rotated raster dataset in EPSG:4326.

    Args:
        dataset (xr.Dataset | xr.DataArray): Raster dataset with georeferencing.

    Returns:
        tuple[float, float, float, float]: Bounding box (min_lon, min_lat, max_lon, max_lat).

    Raises:
        ValueError: If dataset has no CRS assigned.
    """
    if dataset.rio.crs is None:
        raise ValueError("Dataset must have a valid CRS.")

    transform: Affine = dataset.rio.transform()
    crs = dataset.rio.crs
    height, width = dataset.rio.height, dataset.rio.width

    # Corner coordinates in pixel space (row, col)
    pixel_corners = [(0, 0), (width, 0), (width, height), (0, height)]

    # Convert pixel to world coordinates using affine transformation
    world_corners = [transform * (x, y) for x, y in pixel_corners]

    # Convert to EPSG:4326
    lon, lat = rasterio.warp.transform(
        crs, "EPSG:4326", *zip(*world_corners, strict=False)
    )

    return min(lon), min(lat), max(lon), max(lat)


def is_rotated(dataset: xr.Dataset | xr.DataArray) -> bool:
    """Checks if a raster dataset is rotated."""
    transform = dataset.rio.transform()
    return transform[1] != 0 or transform[3] != 0  # Shear_x or Shear_y is non-zero


def create_cog_item(
    dataset: xr.Dataset | xr.DataArray,
    urlpath: str,
    protocol: AssetProtocol = AssetProtocol.HTTPS,
    alternate_links: dict[str, bool | str] | None = None,
    storage_options: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
    item_extra_fields: dict[str, Any] | None = None,
    asset_extra_fields: dict[str, Any] | None = None,
    nodata: Any | None = None,
    data_type: Any = None,
    scale_factor: float | None = None,
    unit: str | None = None,
    resolution: int | None = None,
    extra_extensions: list[str] | None = None,
) -> pystac.Item:
    """Creates a STAC Item for a Cloud-Optimized GeoTIFF dataset.

    Args:
        dataset (xr.Dataset | xr.DataArray): Input dataset.
        urlpath (str): Base URL for constructing HREFs for assets.
        protocol (AssetProtocol, optional): Protocol for asset HREFs. Defaults to HTTPS.
        alternate_links (dict[str, bool | str] | None, optional): Alternate asset access URIs.
        storage_options (dict[str, Any] | None, optional): Storage configuration.
        properties (dict[str, Any] | None, optional): STAC item properties.
        item_extra_fields (dict[str, Any] | None, optional): Extra fields for the item.
        asset_extra_fields (dict[str, Any] | None, optional): Extra fields for the asset.
        nodata (Any | None): Nodata value. Inferred when this is None. Defaults to None.
        data_type (Any, optional): Data type. Defaults to None.
        scale_factor (float | None, optional): Scale factor. Defaults to None.
        unit (str | None, optional): Unit of measurement. Defaults to None.
        resolution (int | None, optional): Resolution of the asset. Defaults to None.
        extra_extensions (list[str] | None, optional): Additional STAC extensions.

    Returns:
        pystac.Item: A STAC item containing metadata and COG assets.
    """
    storage_options = storage_options or {}
    properties = properties or {}
    item_extra_fields = item_extra_fields or {}
    asset_extra_fields = asset_extra_fields or {}

    account_name = storage_options.get("account_name", "")
    pp = PathParser(urlpath, account_name=account_name)
    stac_id = pp.stac_item_id

    # Extract CRS, BBOX, and Geometry
    crs = dataset.rio.crs
    if is_rotated(dataset):
        bbox = get_rotated_bbox(dataset)  # Use accurate method for rotated rasters
        geometry = Polygon.from_bounds(*bbox)
    else:
        bounds = dataset.rio.bounds()  # Use default method for non-rotated rasters
        bbox = list(rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds))
        geometry = shapely.geometry.shape(antimeridian.fix_shape(mapping(box(*bbox))))
        geometry = make_valid(geometry)

    # NOTE: credits to stac-packages sentinel2 for the following code snippet and antimeridian
    # sometimes, antimeridian and/or polar crossing scenes on some platforms end up
    # with geometries that cover the inverse area that they should, so nearly the
    # entire globe. This has been seen to have different behaviors on different
    # architectures and dependent library versions. To prevent these errors from
    # resulting in a wildly-incorrect geometry, we fail here if the geometry
    # is unreasonably large. Typical areas will no greater than 3, whereas an
    # incorrect globe-covering geometry will have an area for 61110.
    if (ga := geometry.area) > 100:
        raise Exception(f"Area of geometry is {ga}, which is too large to be correct.")

    # Extract datetime information
    datetimes = get_datetimes(dataset)
    if datetimes:
        dt = datetimes["datetime"]
        start_datetime = datetimes.get("start_datetime")
        end_datetime = datetimes.get("end_datetime")
    else:
        dt = pystac.utils.now_in_utc()
        start_datetime = None
        end_datetime = None

    attributes = dataset.attrs.copy()
    standard_metadata = {"datetime", "start_datetime", "end_datetime"}
    properties.update(
        {
            k.replace(":", "_"): v
            for k, v in attributes.items()
            if k not in standard_metadata
        }
    )

    # Create STAC item
    item = pystac.Item(
        id=stac_id,
        geometry=shapely.geometry.mapping(geometry),
        bbox=bbox,  # type: ignore
        datetime=dt,
        properties=properties,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    item.common_metadata.created = pystac.utils.now_in_utc()

    # Add STAC extensions
    if extra_extensions:
        for ext in extra_extensions:
            item.stac_extensions.append(ext)

    proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
    proj_ext.code = int(dataset.rio.crs.to_epsg())
    proj_ext.shape = [dataset.sizes["y"], dataset.sizes["x"]]
    proj_ext.transform = list(dataset.rio.transform())
    proj_ext.bbox = dataset.rio.bounds()
    proj_ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*proj_ext.bbox))

    # Handle dataset vs. data array
    if isinstance(dataset, xr.Dataset):
        for var_name, var in dataset.data_vars.items():
            pp.band = var_name

            # Resolution can only be extracted with rio when we have square (non-rotated) data.
            if resolution is None:
                resolution = int(var.rio.resolution()[0])
            pp.resolution = f"{int(resolution)}m"

            asset_href = protocol.get_uri(pp)

            var_nodata = get_nodata(var) if nodata is None else nodata

            item = add_cog_asset(
                item,
                var,
                var_name,
                asset_href,
                nodata=var_nodata,
                data_type=data_type,
                scale_factor=scale_factor,
                unit=unit,
                resolution=resolution,
            )

            if alternate_links:
                item = add_alternate_links(item, pp, alternate_links)
    else:
        asset_href = protocol.get_uri(pp)

        var_nodata = get_nodata(dataset) if nodata is None else nodata

        add_cog_asset(
            item,
            dataset,
            None,
            asset_href,
            nodata=var_nodata,
            data_type=data_type,
            scale_factor=scale_factor,
            unit=unit,
        )

        if alternate_links:
            add_alternate_links(item, pp, alternate_links)

    return item


def extract_eo_bands(data: xr.DataArray) -> list[Band]:
    """
    Extract EO bands from the 'band' dimension of an xarray DataArray.
    Args:
        data (xr.DataArray): Input DataArray with a 'band' dimension.

    Returns:
        List[Band]: List of EO Bands with names and common names.
    """
    if "band" not in data.dims:
        raise ValueError("The input DataArray does not contain a 'band' dimension.")

    if "band" not in data.coords:
        raise ValueError("The DataArray is missing a 'band' coordinate.")

    band_names = [str(band) for band in data["band"].values]

    return [
        Band.create(name=band_name, common_name=band_name) for band_name in band_names
    ]


def add_cog_asset(
    item: pystac.Item,
    data: xr.DataArray,
    var_name: str | None,
    href: str,
    nodata: Any | None = None,
    data_type: Any | None = None,
    scale_factor: float | None = None,
    unit: str | None = None,
    resolution: int | None = None,
) -> pystac.Item:
    """
    Add a Cloud Optimized GeoTIFF (COG) asset to a STAC item.

    Args:
        item (pystac.Item): The STAC item to which the asset will be added.
        data (xr.DataArray): The data array representing the band(s).
        var_name (Optional[str]): Name of the variable (band name).
        href (str): HREF for the asset.
        nodata (Any, optional): Nodata value. When None, the value is extracted using get_nodata(). Defaults to None.
        scale_factor (float, optional): Scale factor for the asset. Defaults to None.
        unit (str, optional): Unit for the asset. Defaults to None.
        resolution (int, optional): Resolution of the asset. Defaults to None.

    Returns:
        pystac.Item: The updated STAC item with the asset.
    """
    if not data.rio.crs:
        raise ValueError("CRS information is missing in the data array.")

    if nodata is None:
        nodata = get_nodata(data)

    if data_type is None:
        data_type = data.dtype.name

    resolution = resolution or abs(data.rio.resolution()[0])
    transform = list(data.rio.transform())
    shape = [data.sizes["y"], data.sizes["x"]]
    bbox = data.rio.bounds()

    asset = pystac.Asset(
        href=href,
        media_type=pystac.MediaType.COG,
        roles=["data"],
        title=f"{var_name} band" if var_name else "COG Asset",
    )
    asset_key = var_name or "data"
    item.add_asset(asset_key, asset)

    proj_ext = ProjectionExtension.ext(asset, add_if_missing=True)
    proj_ext.code = int(data.rio.crs.to_epsg())
    proj_ext.shape = shape
    proj_ext.transform = transform
    proj_ext.bbox = bbox
    proj_ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*proj_ext.bbox))

    raster_ext = RasterExtension.ext(asset, add_if_missing=True)
    raster_ext.bands = [
        RasterBand.create(
            nodata=nodata,
            spatial_resolution=resolution,
            data_type=DataType(data_type),
            unit=unit,
            scale=scale_factor,
        )
    ]

    if "band" in data.dims:
        eo_ext = EOExtension.ext(asset, add_if_missing=True)
        eo_ext.bands = extract_eo_bands(data)

    return item


def enrich_table_columns(
    item: pystac.Item, asset_extra_fields: dict[str, Any]
) -> pystac.Item:
    """Add descriptions to table columns in item properties."""
    if "table:columns" in asset_extra_fields and "table:columns" in item.properties:
        source_lookup = {
            col["name"]: col for col in asset_extra_fields["table:columns"]
        }
        for target_col in item.properties["table:columns"]:
            source_col = source_lookup.get(target_col["name"])
            if source_col:
                target_col.setdefault("description", source_col.get("description"))
    return item


def create_tabular_item(
    urlpath: str,
    asset_title: str,
    asset_description: str,
    storage_options: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
    item_extra_fields: dict[str, Any] | None = None,
    asset_extra_fields: dict[str, Any] | None = None,
    datetime: datetime | None = None,
    infer_datetime: InferDatetimeOptions = InferDatetimeOptions.range,
    protocol: AssetProtocol = AssetProtocol.HTTPS,
    alternate_links: dict[str, bool | str] | None = None,
) -> pystac.Item:
    """Creates a STAC Item for a tabular dataset with configurable metadata and alternate access links.

    Args:
        urlpath (str): The dataset's URI or file path.
        asset_title (str): The title of the asset.
        asset_description (str): The description of the asset.
        storage_options (dict[str, Any] | None, optional): Storage configuration for remote access.
        properties (dict[str, Any] | None, optional): Additional STAC Item properties.
        item_extra_fields (dict[str, Any] | None, optional): Extra metadata fields for the item.
        asset_extra_fields (dict[str, Any] | None, optional): Extra metadata fields for the asset.
        datetime (datetime | None, optional): Explicit datetime for the dataset.
        infer_datetime (InferDatetimeOptions, optional): Strategy for inferring datetime from data. Defaults to 'range'.
        protocol (AssetProtocol, optional): The protocol to use for the asset href. Defaults to HTTPS.
        alternate_links (dict[str, bool | str] | None, optional): Alternate access links.

    Returns:
        pystac.Item: A fully populated STAC Item representing the tabular dataset.
    """
    if (datetime and infer_datetime != InferDatetimeOptions.no) or (
        datetime is None and infer_datetime == InferDatetimeOptions.no
    ):
        raise ValueError("Specify either 'datetime' or 'infer_datetime', but not both.")

    storage_options = storage_options or {}
    properties = properties or {}
    item_extra_fields = item_extra_fields or {}
    asset_extra_fields = asset_extra_fields or {}

    account_name = storage_options.get("account_name", "")
    pp = PathParser(urlpath, account_name=account_name)

    template = pystac.Item(
        id=pp.stac_item_id,
        properties=properties,
        geometry=None,
        bbox=None,
        datetime=datetime or pystac.utils.now_in_utc(),
        stac_extensions=[],
    )
    template.common_metadata.created = pystac.utils.now_in_utc()

    if infer_datetime != InferDatetimeOptions.no:
        datetime_column = "datetime"

    item = generate(
        uri=AssetProtocol.CLOUD.get_uri(pp),
        template=template,
        infer_bbox=True,
        proj=True,
        infer_geometry=False,
        datetime=datetime,
        infer_datetime=infer_datetime,
        datetime_column=datetime_column,
        metadata_created=pystac.utils.now_in_utc(),
        count_rows=True,
        asset_key="data",
        asset_href=protocol.get_uri(pp),
        asset_title=asset_title,
        asset_description=asset_description,
        asset_media_type=PARQUET_MEDIA_TYPE,
        asset_roles=["data"],
        asset_extra_fields=asset_extra_fields,
        storage_options=storage_options,
        validate=False,
    )
    assert isinstance(item, pystac.Item)

    # Enrich table column descriptions
    item = enrich_table_columns(item, asset_extra_fields)

    # Handle alternate links
    if alternate_links:
        for key, value in alternate_links.items():
            if key in AssetProtocol.__members__:
                proto = AssetProtocol[key]
                target = (
                    proto.get_uri(pp)
                    if value is True
                    else proto.get_uri(pp, value)
                    if isinstance(value, str)
                    else None
                )
            else:
                target = value

            if isinstance(target, str):
                item.add_link(
                    pystac.Link(
                        rel="alternate",
                        target=target,
                        title=f"{key.capitalize()} Access",
                        media_type=PARQUET_MEDIA_TYPE,
                    )
                )

    return item


if __name__ == "__main__":
    import os
    import pathlib

    import dotenv
    import geopandas as gpd
    import odc
    import odc.geo
    import pystac
    import shapely
    from rasterio.enums import Resampling

    from coastpy.eo.typology import TypologyCollection
    from coastpy.io.utils import name_data
    from coastpy.stac.layouts import COGLayout

    def create_collection() -> pystac.Collection:
        extent = pystac.Extent(
            pystac.SpatialExtent([[-180.0, 90.0, 180.0, -90.0]]),
            pystac.TemporalExtent([[pystac.utils.now_in_utc(), None]]),
        )

        collection = pystac.Collection(
            id="typology-train-cube",
            title="Typology Train Cube",
            description="A collection of coastal typology training data.",
            extent=extent,
            catalog_type=pystac.CatalogType.RELATIVE_PUBLISHED,
        )
        return collection

    dotenv.load_dotenv()
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}

    TRANSECT_ATTRIBUTES = [
        "transect_id",
        "lon",
        "lat",
        "utm_epsg",
        "continent",
        "country",
        "common_region_name",
        "bearing",
        "dist_b0",
        "dist_b30",
        "dist_b60",
        "dist_b90",
        "dist_b120",
        "dist_b150",
        "dist_b180",
        "dist_b210",
        "dist_b240",
        "dist_b270",
        "dist_b300",
        "dist_b330",
        "uuid",
        "user",
        "shore_type",
        "coastal_type",
        "landform_type",
        "is_built_environment",
        "has_defense",
        "datetime_created",
        "datetime_updated",
        "is_challenging",
        "comment",
        "link",
        "confidence",
        "is_validated",
    ]

    TARGET_AXIS = "horizontal-right-aligned"
    RESAMPLING = Resampling.cubic
    TRANSECT_LENGTH = 2000
    OFFSET_DISTANCE = 200
    RESOLUTION = 10
    y_shape = int((OFFSET_DISTANCE * 2) / RESOLUTION)
    x_shape = int(TRANSECT_LENGTH / RESOLUTION)

    transects = gpd.read_parquet(
        "/Users/calkoen/data/tmp/typology/train/transect.parquet"
    )

    for i in range(len(transects)):
        transect = transects.iloc[[i]]
        transect_properties = transect[TRANSECT_ATTRIBUTES].iloc[0].to_dict()
        transect_properties["datetime_created"] = transect_properties[
            "datetime_created"
        ].isoformat()
        transect_properties["datetime_updated"] = transect_properties[
            "datetime_updated"
        ].isoformat()

        # Define region of interest based on buffered transects
        region_of_interest = gpd.GeoDataFrame(
            geometry=[
                shapely.box(
                    *gpd.GeoSeries.from_xy(transect.lon, transect.lat, crs=4326)
                    .to_crs(transect.utm_epsg.item())
                    .buffer(1500)
                    .to_crs(4326)
                    .total_bounds
                )
            ],
            crs=4326,
        )

        coastal_zone = odc.geo.geom.Geometry(
            region_of_interest.geometry.item(), crs=region_of_interest.crs
        )

        ds = (
            TypologyCollection()
            .search(region_of_interest, sas_token=sas_token)
            .load()
            .execute()
        )
        ds = ds.compute()

        chip = TypologyCollection.chip_from_transect(
            ds,
            transect,
            y_shape,
            x_shape,
            RESAMPLING,
            rotate=True,
            target_axis=TARGET_AXIS,
            offset_distance=OFFSET_DISTANCE,
            resolution=RESOLUTION,
        )

        OUT_STORAGE = "/Users/calkoen/data/tmp/typology/train/release/2025-02-02"
        STAC_DIR = pathlib.Path("/Users/calkoen/data/tmp/typology/stac")

        pathlike = name_data(chip, prefix=OUT_STORAGE)

        pp = PathParser(pathlike, account_name="coclico")
        for var_name, var in chip.data_vars.items():
            pp.band = var_name
            pp.resolution = "10m"
            pathlike2 = AssetProtocol.FILE.get_uri(pp, OUT_STORAGE)
            var.rio.to_raster(pathlike2)

        collection = create_collection()

        item = create_cog_item(
            chip,
            pathlike,
            properties=transect_properties,
            protocol=AssetProtocol.FILE,
            storage_options=storage_options,
            resolution=10,
        )
        item.validate()

        collection.add_item(item)

        layout = COGLayout()
        collection.normalize_hrefs(str(STAC_DIR / collection.id), layout)

        collection.save()


def add_gpq_snapshot(
    collection: pystac.Collection,
    storage_path: str,
    storage_options: dict[str, Any],
) -> pystac.Collection:
    """
    Writes a GeoParquet snapshot to cloud storage and adds it as a STAC asset.

    Args:
        df (pd.DataFrame): The DataFrame to write to Parquet.
        collection (pystac.Collection): The STAC Collection to attach the asset to.
        storage_path (str): Cloud storage path (e.g., "az://bucket/path/to/file.parquet").
        storage_options (dict): Storage options for `fsspec.open()`, must contain `account_name`.

    Returns:
        pystac.Collection: The updated STAC collection with the added asset.

    Raises:
        ValueError: If `account_name` is missing from `storage_options`.
    """

    # Validate `account_name` in storage options
    if "account_name" not in storage_options:
        raise ValueError("`storage_options` must contain `account_name`.")

    # Parse paths using PathParser
    path_parser = PathParser(storage_path, account_name=storage_options["account_name"])
    cloud_uri = path_parser.to_cloud_uri()
    asset_href = path_parser.to_https_url()

    items = list(collection.get_all_items())
    items_as_json = [i.to_dict() for i in items]
    item_extents = stac_geoparquet.to_geodataframe(items_as_json)

    # Write GeoParquet to cloud storage
    with fsspec.open(cloud_uri, mode="wb", **storage_options) as f:
        item_extents.to_parquet(f)

    # Create STAC Asset with hardcoded metadata
    asset = pystac.Asset(
        href=asset_href,
        title="GeoParquet STAC items",
        description="Snapshot of the collection's STAC items exported to GeoParquet format.",
        media_type=PARQUET_MEDIA_TYPE,
        roles=["metadata"],
    )

    # Set creation timestamp
    asset.common_metadata.created = pystac.utils.now_in_utc()

    # Attach asset to STAC collection
    collection.add_asset("geoparquet-stac-items", asset)

    return collection
