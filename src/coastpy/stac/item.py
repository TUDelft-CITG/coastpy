import warnings
from datetime import datetime
from typing import Any

import pandas as pd
import pystac
import pystac.utils
import rasterio.warp
import xarray as xr
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
from shapely.geometry import box, mapping

from coastpy.io.utils import PathParser
from coastpy.utils.xarray import get_nodata


def _extract_datetimes(
    ds: xr.Dataset | xr.DataArray,
) -> dict[str, datetime | None]:
    """
    Extract datetime information (datetime, start_datetime, end_datetime) from an xarray Dataset or DataArray.

    Args:
        ds (Union[xr.Dataset, xr.DataArray]): Input xarray object.

    Returns:
        Dict[str, Optional[datetime]]: A dictionary containing 'datetime', 'start_datetime', and 'end_datetime'.
    """

    # Check for a time dimension
    if "time" in ds.dims:
        time_values = ds.coords["time"].values
        if len(time_values) > 1:
            return {
                "datetime": pd.Timestamp(time_values[0]).to_pydatetime(),
                "start_datetime": pd.Timestamp(time_values[0]).to_pydatetime(),
                "end_datetime": pd.Timestamp(time_values[-1]).to_pydatetime(),
            }
        else:
            return {
                "datetime": pd.Timestamp(time_values[0]).to_pydatetime(),
                "start_datetime": None,
                "end_datetime": None,
            }

    # Check for time or datetime coordinate
    if "time" in ds.coords or "datetime" in ds.coords:
        time_coord = ds.coords.get("time") or ds.coords.get("datetime")
        if time_coord:
            time_value = pd.Timestamp(time_coord.values).to_pydatetime()
        return {"datetime": time_value, "start_datetime": None, "end_datetime": None}

    # Check for datetime attributes in variables
    start_times = []
    end_times = []
    for var in ds.data_vars.values():
        if "start_datetime" in var.attrs:
            start_times.append(
                pd.Timestamp(var.attrs["start_datetime"]).to_pydatetime()
            )
        if "end_datetime" in var.attrs:
            end_times.append(pd.Timestamp(var.attrs["end_datetime"]).to_pydatetime())

    if start_times and end_times:
        return {
            "datetime": min(start_times),
            "start_datetime": min(start_times),
            "end_datetime": max(end_times),
        }

    if "start_datetime" in ds.attrs and "end_datetime" in ds.attrs:
        return {
            "datetime": pd.Timestamp(ds.attrs["start_datetime"]).to_pydatetime(),
            "start_datetime": pd.Timestamp(ds.attrs["start_datetime"]).to_pydatetime(),
            "end_datetime": pd.Timestamp(ds.attrs["end_datetime"]).to_pydatetime(),
        }

    # Check for 'datetime' in attributes
    if "datetime" in ds.attrs:
        return {
            "datetime": pd.Timestamp(ds.attrs["datetime"]).to_pydatetime(),
            "start_datetime": None,
            "end_datetime": None,
        }

    # If no datetime information is found, raise an error
    raise ValueError("Unable to determine datetime information from dataset.")


from datetime import datetime

import pystac
import pystac.utils
import rasterio
import rasterio.warp
import xarray as xr

from coastpy.stac.item import _extract_datetimes


def create_cog_item(
    dataset: xr.Dataset | xr.DataArray,
    urlpath: str,
    storage_options: dict[str, Any] | None = None,
    nodata: Any = "infer",
    data_type: Any = None,
    scale_factor: float | None = None,
    unit: str | None = None,
    extra_extensions: list[str] | None = None,
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
    # Extract CRS and reproject bounding box to EPSG:4326
    crs = dataset.rio.crs
    bounds = dataset.rio.bounds()
    bbox = list(rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds))
    geometry = mapping(box(*bbox))

    # Generate STAC ID
    account_name = (
        storage_options.get("account_name", None) if storage_options else None
    )
    path_parser = PathParser(urlpath, account_name=account_name)
    stac_id = path_parser.stac_item_id

    # Extract datetime information
    datetimes = _extract_datetimes(dataset)
    datetime = datetimes["datetime"]
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
        datetime=datetime,
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
            urlpath = path_parser.to_https_url()

            # Handle nodata
            var_nodata = nodata if nodata != "infer" else get_nodata(var)

            # Add as asset
            add_cog_asset(
                item,
                var,
                var_name,
                urlpath,
                nodata=var_nodata,
                data_type=data_type,
                scale_factor=scale_factor,
                unit=unit,
            )
    else:
        urlpath = path_parser.to_cloud_uri()
        var_nodata = nodata if nodata != "infer" else get_nodata(dataset)
        add_cog_asset(
            item,
            dataset,
            None,
            urlpath,
            nodata=var_nodata,
            scale_factor=scale_factor,
            unit=unit,
        )

    return item


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
        data (xr.DataArray): The data array representing the band.
        var_name (Optional[str]): Name of the variable (band name).
        href (str): HREF for the asset.
        nodata (Any, optional): Nodata value. Use "infer" to extract from data. Defaults to "infer".
        scale_factor (float, optional): Scale factor for the asset. Defaults to None.
        unit (str, optional): Unit for the asset. Defaults to None.

    Returns:
        None: The function updates the item in place.

    Raises:
        ValueError: If required attributes like CRS or resolution are missing.
    """
    # Validate CRS
    if not data.rio.crs:
        raise ValueError("CRS information is missing in the data array.")

    # Infer nodata value if needed
    nodata_value = nodata if nodata != "infer" else get_nodata(data)
    if nodata_value is None:
        warnings.warn(
            f"Nodata value is missing for variable '{var_name}'.",
            UserWarning,
            stacklevel=2,
        )

    if data_type is None:
        warnings.warn(
            f"DataType value is missing for variable '{var_name}'.",
            UserWarning,
            stacklevel=2,
        )
        data_type = data.dtype.name

    # Extract spatial resolution
    resolution = abs(data.rio.resolution()[0])  # Assumes square pixels

    # Extract transform, shape, and bounds
    transform = list(data.rio.transform())
    shape = [data.sizes["y"], data.sizes["x"]]
    bbox = data.rio.bounds()

    # Create the asset
    asset = pystac.Asset(
        href=href,
        media_type=pystac.MediaType.COG,
        roles=["data"],
        title=f"{var_name} Band" if var_name else "Data Asset",
    )
    # Attach the asset to the item
    asset_key = var_name or "data"
    item.add_asset(asset_key, asset)

    # Add Raster extension to the asset
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

    # Add Projection extension to the asset
    proj_ext = ProjectionExtension.ext(asset, add_if_missing=True)
    proj_ext.shape = shape
    proj_ext.bbox = bbox
    proj_ext.transform = transform

    return item
