import warnings
from datetime import datetime
from typing import Any

import pandas as pd
import pystac
import rasterio.warp
import xarray as xr
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension
from shapely.geometry import box, mapping

from coastpy.io.utils import PathParser, name_data
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


def create_cog_item(
    dataset: xr.Dataset | xr.DataArray,
    base_url: str,
    storage_options: dict[str, Any] | None = None,
) -> pystac.Item:
    """
    Create a generic STAC item from an xarray.Dataset or xarray.DataArray.

    Args:
        dataset (Union[xr.Dataset, xr.DataArray]): Input dataset or data array.
        base_url (str): Base URL for constructing HREFs for assets.
        storage_options (Optional[Dict[str, Any]]): Options for cloud storage (e.g., account name for Azure).

    Returns:
        pystac.Item: A STAC item containing metadata and placeholders for assets.
    """

    # Extract CRS and reproject bounding box to EPSG:4326
    crs = dataset.rio.crs
    bounds = dataset.rio.bounds()
    bbox = list(rasterio.warp.transform_bounds(crs, "EPSG:4326", *bounds))
    geometry = mapping(box(*bbox))

    # Generate STAC ID
    name = name_data(dataset, include_random_hex=False)
    account_name = storage_options.get("account_name") if storage_options else None
    url = f"{base_url}/{name}"
    pp = PathParser(url, account_name=account_name)
    stac_id = pp.stac_item_id

    # Extract datetimes
    datetimes = _extract_datetimes(dataset)
    datetime = datetimes["datetime"]
    start_datetime = datetimes.get("start_datetime")
    end_datetime = datetimes.get("end_datetime")

    # Copy dataset attributes as item properties
    properties = dataset.attrs.copy()
    for dt in ["datetime", "start_datetime", "end_datetime"]:
        del properties[dt]
    properties["composite:stac_ids"] = list(properties["composite:stac_ids"])
    properties["composite:groups"] = list(properties["composite:groups"])

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
    print(item.validate())

    # Add projection metadata
    proj = ProjectionExtension.ext(item, add_if_missing=True)
    proj.epsg = crs.to_epsg()
    proj.shape = [dataset.sizes["y"], dataset.sizes["x"]]
    proj.bbox = bbox
    proj.transform = list(dataset.rio.transform())

    # Add assets
    https_url = pp.to_https_url()
    if isinstance(dataset, xr.Dataset):
        for var_name, var in dataset.data_vars.items():
            href = _generate_href(https_url, var_name)
            add_cog_asset(item, var, var_name, href)
    else:
        add_cog_asset(item, dataset, None, https_url)

    return item


def add_cog_asset(
    item: pystac.Item,
    var: xr.DataArray,
    var_name: str | None,
    href: str,
) -> pystac.Item:
    """
    Add a Cloud Optimized GeoTIFF (COG) asset to a STAC item.

    Args:
        item (pystac.Item): The STAC item to which the asset will be added.
        var (xr.DataArray): The variable to describe as an asset.
        var_name (Optional[str]): The name of the variable (band name).
        href (str): The HREF for the asset.

    Returns:
        pystac.Item: The updated STAC item with the new asset.
    """
    # Validate nodata value
    nodata = get_nodata(var)
    if nodata is None:
        raise ValueError(f"Nodata value is missing for variable '{var_name}'.")

    # Validate scale factor
    scale_factor = var.attrs.get("scale_factor", None)
    if scale_factor is None:
        warnings.warn(
            f"Scale factor is not set for variable '{var_name}'.",
            UserWarning,
            stacklevel=2,
        )

    # Derive resolution and projection
    resolution = abs(var.rio.resolution()[0])  # Assumes square pixels
    transform = list(var.rio.transform())
    shape = [var.sizes["y"], var.sizes["x"]]
    bbox = var.rio.bounds()

    # Create the asset
    asset = pystac.Asset(
        href=href,
        media_type=pystac.MediaType.COG,
        roles=["data"],
        title=f"{var_name.capitalize()} Band" if var_name else "Data Asset",
    )

    # Add the asset to the STAC item after it is fully populated
    item.add_asset(var_name or "data", asset)

    # Add Raster extension
    raster = RasterExtension.ext(asset, add_if_missing=True)
    raster.bands = [
        RasterBand.create(
            nodata=nodata,
            spatial_resolution=resolution,
            data_type=DataType(str(var.dtype)),
        )
    ]

    # Add Projection extension to asset
    asset_proj = ProjectionExtension.ext(asset, add_if_missing=True)
    asset_proj.shape = shape
    asset_proj.bbox = bbox
    asset_proj.transform = transform

    return item


def _generate_href(base_url: str, var_name: str) -> str:
    """
    Generate an HREF for a STAC item with multiple assets (e.g., bands).

    Args:
        base_url (str): Base URL for the asset.
        var_name (str): The variable name (band).

    Returns:
        str: The generated HREF.
    """
    prefix, name = base_url.rsplit("/", 1)
    return f"{prefix}/{var_name}_{name}"
