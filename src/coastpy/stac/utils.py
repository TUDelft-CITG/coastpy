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


def read_snapshot(collection, columns=None, add_href=True, storage_options=None):
    """
    Reads the extent of items from a STAC collection and returns a GeoDataFrame with specified columns.

    Args:
        collection: A STAC collection object that contains assets.
        columns: List of columns to return. If None, all columns will be read.
        add_href: Boolean indicating whether to extract and add the 'href' column from 'assets'. Default is True.
        storage_options: Storage options to pass to fsspec. Default is {"account_name": "coclico"}.

    Returns:
        GeoDataFrame containing the specified columns.
    """
    # Set default storage options
    if storage_options is None:
        storage_options = {"account_name": "coclico"}

    if columns is not None and "assets" not in columns:
        columns.append("assets")

    # Open the Parquet file and read the specified columns
    href = collection.assets["geoparquet-stac-items"].href
    with fsspec.open(href, mode="rb", **storage_options) as f:
        extents = gpd.read_parquet(f, columns=columns)

    if add_href:
        if "assets" not in extents.columns:
            msg = "The 'assets' column is required to extract 'href'."
            raise ValueError(msg)
        extents["href"] = extents["assets"].apply(lambda x: x["data"]["href"])

    return extents
