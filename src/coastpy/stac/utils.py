import copy
import itertools
import logging
import operator

import fsspec
import geopandas as gpd
import pystac
import xarray as xr

logger = logging.getLogger(__name__)


def collate(items: xr.DataArray) -> list[pystac.Item]:
    """
    Collate many items by id, gathering together assets with the same item ID.

    Args:
        items (xr.DataArray): A DataArray containing STAC items.

    Returns:
        List[pystac.Item]: A list of unique pystac Items, with assets collated by item ID.

    Example:
        >>> items_data = xr.DataArray(...)
        >>> collated_items = collate(items_data)
        >>> print(collated_items)
        [<pystac.Item object at ...>, ...]
    """

    # Convert data array into a flat list of stac items
    items_list = items.data.ravel().tolist()

    collated_items = []
    key_func = operator.attrgetter("id")

    # Group by item ID
    for _, group in itertools.groupby(sorted(items_list, key=key_func), key=key_func):
        grouped_items = list(group)
        main_item = grouped_items[0].clone()
        collated_items.append(main_item)

        # Merge assets from all items in the group
        for other_item in grouped_items:
            cloned_item = other_item.clone()
            for asset_key, asset in cloned_item.assets.items():
                main_item.add_asset(asset_key, asset)

    return collated_items


def stackstac_to_dataset(stack: xr.DataArray) -> xr.Dataset:
    """
    Convert a stackstac DataArray to an xarray Dataset.

    Args:
        stack (xr.DataArray): The input stackstac DataArray.

    Returns:
        xr.Dataset: The output dataset after conversion.

    Example:
        >>> # Assuming 'stack' is an existing stackstac DataArray.
        >>> ds = stackstac_to_dataset(stack)
    """
    ds = stack.to_dataset("band")

    # Loop over coordinates with band dimension and save those as attrs in variables
    for coord, da in ds.band.coords.items():
        if "band" in da.dims:
            for i, band in enumerate(stack.band.values):
                ds[band].attrs[coord] = da.values[i]

    ds = ds.drop_dims("band")
    return ds


def read_snapshot(collection, columns=None, storage_options=None):
    """
    Reads the extent of items from a STAC collection and returns a GeoDataFrame with specified columns.

    Args:
        collection: A STAC collection object that contains assets.
        columns: List of columns to return. Default is ["geometry", "assets", "href"].
        storage_options: Storage options to pass to fsspec. Default is None.

    Returns:
        GeoDataFrame containing the specified columns.
    """
    if storage_options is None:
        storage_options = {"account_name": "coclico"}

    # Set default columns
    if columns is None:
        columns = ["geometry", "assets", "href"]

    columns_ = copy.deepcopy(columns)

    # Ensure 'assets' is always in the columns
    if "assets" not in columns:
        columns.append("assets")
        logger.debug("'assets' column added to the list of columns")

    # Open the parquet file and read the specified columns
    href = collection.assets["geoparquet-stac-items"].href
    with fsspec.open(href, mode="rb", **storage_options) as f:
        extents = gpd.read_parquet(f, columns=[c for c in columns if c != "href"])

    # If 'href' is requested, extract it from the 'assets' column
    if "href" in columns:
        extents["href"] = extents["assets"].apply(lambda x: x["data"]["href"])
        logger.debug("'href' column extracted from 'assets'")

    # Drop 'assets' if it was not originally requested
    if "assets" not in columns_:
        extents = extents.drop(columns=["assets"])
        logger.debug("'assets' column dropped from the GeoDataFrame")

    return extents
