import warnings
from datetime import datetime
from typing import Any

import pystac
import pystac.utils
import rasterio
import rasterio.warp
import xarray as xr
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
from shapely.geometry import box, mapping

from coastpy.io.utils import PathParser, extract_datetimes
from coastpy.libs.stac_table import InferDatetimeOptions, generate
from coastpy.utils.xarray_utils import get_nodata


def create_cog_item(
    dataset: xr.Dataset | xr.DataArray,
    pathlike: str,
    nodata: Any = "infer",
    data_type: Any = None,
    scale_factor: float | None = None,
    unit: str | None = None,
    extra_extensions: list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> pystac.Item:
    """
    Create a generic STAC item from an xarray.Dataset or xarray.DataArray.

    Args:
        dataset (Union[xr.Dataset, xr.DataArray]): Input dataset or data array.
        urlpath (str): Base URL for constructing HREFs for assets.
        storage_options (Optional[Dict[str, Any]]): Options for cloud storage (e.g., account name for Azure).
        nodata (Any, optional): Nodata value. Use "infer" to automatically extract it. Defaults to "infer".
        scale_factor (float, optional): Scale factor for the data. Defaults to None.
        unit (str, optional): Unit of the data. Defaults to None.
        extra_extensions (list[str], optional): Additional STAC extensions to add. Defaults to None.

    Returns:
        pystac.Item: A STAC item containing metadata and placeholders for assets.
    """
    storage_options = storage_options or {}
    # Extract CRS and reproject bounding box to EPSG:4326
    crs = dataset.rio.crs
    bounds = dataset.rio.bounds()
    bbox = list(rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds))
    geometry = mapping(box(*bbox))

    # Generate STAC ID
    account_name = storage_options.get("account_name", "")
    path_parser = PathParser(pathlike, account_name=account_name)
    stac_id = path_parser.stac_item_id

    # Extract datetime information
    datetimes = extract_datetimes(dataset)
    dt = datetimes["datetime"]
    start_datetime = datetimes.get("start_datetime")
    end_datetime = datetimes.get("end_datetime")

    # Extract and prepare metadata
    attributes = dataset.attrs.copy()
    standard_metadata = ["datetime", "start_datetime", "end_datetime"]
    properties = {k: v for k, v in attributes.items() if k not in standard_metadata}
    properties.update({k.replace(":", "_"): v for k, v in attributes.items()})

    # Create the STAC item
    item = pystac.Item(
        id=stac_id,
        geometry=geometry,
        bbox=bbox,
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

    # Add projection metadata
    proj = ProjectionExtension.ext(item, add_if_missing=True)
    proj.epsg = crs.to_epsg()
    proj.shape = [dataset.sizes["y"], dataset.sizes["x"]]
    proj.bbox = bbox
    proj.transform = list(dataset.rio.transform())

    # Add assets
    if isinstance(dataset, xr.Dataset):
        for var_name, var in dataset.data_vars.items():
            path_parser.band = var_name
            path_parser.resolution = f"{int(var.rio.resolution()[0])}m"
            pathlike = path_parser.to_https_url()

            # Handle nodata
            var_nodata = nodata if nodata != "infer" else get_nodata(var)

            # Add as asset
            add_cog_asset(
                item,
                var,
                var_name,
                pathlike,
                nodata=var_nodata,
                data_type=data_type,
                scale_factor=scale_factor,
                unit=unit,
            )
    else:
        pathlike = path_parser.to_cloud_uri()
        var_nodata = nodata if nodata != "infer" else get_nodata(dataset)
        add_cog_asset(
            item,
            dataset,
            None,
            pathlike,
            nodata=var_nodata,
            scale_factor=scale_factor,
            unit=unit,
        )

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
    nodata: Any = "infer",
    data_type: Any | None = None,
    scale_factor: float | None = None,
    unit: str | None = None,
) -> pystac.Item:
    """
    Add a Cloud Optimized GeoTIFF (COG) asset to a STAC item.

    Args:
        item (pystac.Item): The STAC item to which the asset will be added.
        data (xr.DataArray): The data array representing the band(s).
        var_name (Optional[str]): Name of the variable (band name).
        href (str): HREF for the asset.
        nodata (Any, optional): Nodata value. Use "infer" to extract from data. Defaults to "infer".
        scale_factor (float, optional): Scale factor for the asset. Defaults to None.
        unit (str, optional): Unit for the asset. Defaults to None.

    Returns:
        pystac.Item: The updated STAC item with the asset.
    """
    if not data.rio.crs:
        raise ValueError("CRS information is missing in the data array.")

    nodata_value = nodata if nodata != "infer" else get_nodata(data)

    if data_type is None:
        warnings.warn(
            "DataType is missing. Inferring from dataset.", UserWarning, stacklevel=2
        )
        data_type = data.dtype.name

    resolution = abs(data.rio.resolution()[0])  # Assumes square pixels
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
    proj_ext.shape = shape
    proj_ext.bbox = bbox
    proj_ext.transform = transform

    raster_ext = RasterExtension.ext(asset, add_if_missing=True)
    raster_ext.bands = [
        RasterBand.create(
            nodata=nodata_value,
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


def enrich_table_columns(item: pystac.Item, asset_extra_fields: dict[str, Any]) -> None:
    """Add descriptions to table columns in item properties."""
    if "table:columns" in asset_extra_fields and "table:columns" in item.properties:
        source_lookup = {
            col["name"]: col for col in asset_extra_fields["table:columns"]
        }
        for target_col in item.properties["table:columns"]:
            source_col = source_lookup.get(target_col["name"])
            if source_col:
                target_col.setdefault("description", source_col.get("description"))


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
    alternate_links: dict[str, bool | str] | None = None,
) -> pystac.Item:
    """Creates a STAC Item for a tabular dataset with configurable metadata and alternate access links.

    This function generates a STAC Item for a Parquet-based tabular dataset, allowing for flexible
    datetime handling, metadata enrichment, and multiple access links. The item integrates with
    cloud-based storage solutions and follows best practices for STAC Geoparquet collections.

    Args:
        urlpath (str): The dataset's URI or file path.
        asset_title (str): The title of the asset.
        asset_description (str): The description of the asset.
        storage_options (dict[str, Any] | None, optional): Storage configuration for remote access.
        properties (dict[str, Any] | None, optional): Additional STAC Item properties.
        item_extra_fields (dict[str, Any] | None, optional): Extra metadata fields for the item.
        asset_extra_fields (dict[str, Any] | None, optional): Extra metadata fields for the asset.
        datetime (datetime | None, optional): Explicit datetime for the dataset. If None, inference is applied.
        infer_datetime (InferDatetimeOptions, optional): Strategy for inferring datetime from data. Defaults to 'range'.
        alternate_links (dict[str, bool | str] | None, optional): Alternate access links.
            - `"https"`: `True` to auto-generate, or a custom string.
            - `"cloud"`: `True` to auto-generate, or a custom string.
            - `"file"`: `True` to auto-generate, or a custom string.
            - Custom keys allow specifying other URIs.

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
        uri=pp.to_cloud_uri(),
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
        asset_href=pp.to_https_url(),
        asset_title=asset_title,
        asset_description=asset_description,
        asset_media_type="application/x-parquet",
        asset_roles=["data"],
        asset_extra_fields=asset_extra_fields,
        storage_options=storage_options,
        validate=False,
    )
    assert isinstance(item, pystac.Item)

    # Enrich table column descriptions
    enrich_table_columns(item, asset_extra_fields)

    # Handle alternate links
    if alternate_links:
        uri_methods = {
            "https": pp.to_https_url(),
            "cloud": pp.to_cloud_uri(),
            "file": lambda dir=None: pp.to_filepath()
            if dir is None
            else pp.to_filepath(dir),
        }

        for key, value in alternate_links.items():
            if key in uri_methods:
                target = (
                    uri_methods[key]()
                    if value is True
                    else uri_methods[key](value)
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
                        media_type="application/x-parquet",
                    )
                )

    return item
