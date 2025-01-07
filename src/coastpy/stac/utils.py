import itertools
import logging
import operator

import fsspec
import geopandas as gpd
import pandas as pd
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
        add_href: Boolean indicating whether to extract and add the 'href' columns from 'assets'. Default is True.
        storage_options: Storage options to pass to fsspec. Default is {"account_name": "coclico"}.

    Returns:
        GeoDataFrame containing the specified columns.
    """
    # Set default storage options
    if storage_options is None:
        storage_options = {"account_name": "coclico"}

    if columns is not None:
        columns = list({*columns, "assets"})

    href = collection.assets["geoparquet-stac-items"].href

    if href.startswith("https://"):
        with fsspec.open(href, mode="rb") as f:
            extents = gpd.read_parquet(f, columns=columns)
    else:
        with fsspec.open(href, mode="rb", **storage_options) as f:
            extents = gpd.read_parquet(f, columns=columns)

    if add_href:
        if "assets" not in extents.columns:
            raise ValueError(
                "The 'assets' column is required to extract 'href' values."
            )

        # Determine whether we are dealing with a single or multiple assets
        first_assets = extents["assets"].iloc[0]

        if (
            isinstance(first_assets, dict)
            and "data" in first_assets
            and len(first_assets) == 1
        ):
            # Single asset under "data"
            extents["href"] = extents["assets"].apply(
                lambda x: x.get("data", {}).get("href")
            )
        else:
            # Multiple assets
            def extract_hrefs(assets_dict):
                if not isinstance(assets_dict, dict):
                    return {}
                return {
                    f"href_{key}": value.get("href", None)
                    for key, value in assets_dict.items()
                }

            hrefs_df = extents["assets"].apply(extract_hrefs).apply(pd.Series)
            extents = pd.concat([extents, hrefs_df], axis=1)

    return extents


if __name__ == "__main__":
    import logging
    import os
    import time

    import fsspec
    import geopandas as gpd
    import pystac
    import pystac.item
    import rioxarray  # noqa
    import xarray as xr
    from dotenv import load_dotenv
    from odc.stac import configure_rio

    from coastpy.stac.utils import read_snapshot

    load_dotenv()

    # Configure cloud and Dask settings
    configure_rio(cloud_defaults=True)
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}

    VERSION = "slurm-2025-01-02"
    RESAMPLING = "cubic"
    OUT_STORAGE = f"az://tmp/typology/composite/release/{VERSION}"
    START_TIME = time.time()

    BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
    SPECTRAL_INDICES = ["NDWI", "NDVI", "MNDWI", "NDMI"]

    catalog = pystac.Catalog.from_file(
        "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
    )
    s2_col = catalog.get_child("s2-l2a-composite")

    s2_extents = read_snapshot(
        s2_col,
        columns=None,
        storage_options=storage_options,
    )

    print("Done")
