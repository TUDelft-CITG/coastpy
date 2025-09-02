import itertools
import logging
import operator
from collections.abc import Callable
from contextlib import suppress

import fsspec
import geopandas as gpd
import numpy as np
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


def get_alternate_href(links):
    """Extracts the 'alternate' href from a list of STAC links."""
    if not isinstance(links, list | tuple | np.ndarray):
        return None  # Return None if links are missing or not iterable

    for link in links:
        if isinstance(link, dict) and link.get("rel") == "alternate" and "href" in link:
            return link["href"]

    return None  # Default return if no alternate href is found


def list_parquet_columns_from_stac(
    collection: pystac.Collection, asset_key: str = "data"
) -> list[str]:
    """Extract available column names from a STAC Collection using the table:columns extension."""
    if not isinstance(collection, pystac.Collection):
        raise TypeError("Expected a STAC Collection.")

    asset = collection.item_assets.get(asset_key)
    if asset is None:
        raise KeyError(f"Asset '{asset_key}' not found in item_assets.")

    table_columns = asset.properties.get("table:columns")
    if not table_columns:
        raise ValueError(f"Asset '{asset_key}' does not declare 'table:columns'.")

    return [col["name"] for col in table_columns if "name" in col]


def read_snapshot(
    collection,
    columns=None,
    add_href=True,
    storage_options=None,
    patch_url: Callable[[str, dict[str, str]], str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Reads the extent of items from a STAC collection and returns a GeoDataFrame with specified columns.

    Args:
        collection: A STAC collection object that contains assets.
        columns: List of columns to return. If None, all columns will be read.
        add_href: Boolean indicating whether to extract and add the 'href' columns from 'assets'. Default is True.
        storage_options: Storage options to pass to fsspec. Default is {"account_name": "coclico"}.
        patch_url: Function to patch the URL. If None, no patching is done. This is useful to reference private cloud storage or local files.

    Returns:
        GeoDataFrame containing the specified columns.
    """

    # Set default storage options
    storage_options = storage_options or {}

    if columns is not None:
        columns = list({*columns, "assets"})

    href = collection.assets["geoparquet-stac-items"].href
    href = patch_url(href, storage_options) if patch_url else href

    if href.startswith("https://"):
        try:
            with fsspec.open(href, mode="rb") as f:
                extents = gpd.read_parquet(f, columns=columns)
        except FileNotFoundError:
            token = storage_options.get("sas_token") or storage_options.get(
                "credential"
            )
            if token is None:
                raise ValueError(
                    "The provided href is not accessible. Please check the URL or provide a valid SAS token."
                ) from None
            signed_href = f"{href}?{token}"
            with fsspec.open(signed_href, mode="rb") as f:
                extents = gpd.read_parquet(f, columns=columns)
    else:
        with fsspec.open(href, mode="rb", **storage_options) as f:
            extents = gpd.read_parquet(f, columns=columns)

    if add_href:
        if "assets" not in extents.columns:
            msg = "The 'assets' column is required to extract 'href' values."
            raise ValueError(msg)

        # Determine whether we are dealing with a single or multiple assets
        first_assets = extents["assets"].iloc[0]

        if (
            isinstance(first_assets, dict)
            and "data" in first_assets
            and len(first_assets) == 1
        ):
            # Extract primary href from assets
            extents["href"] = extents["assets"].apply(
                lambda x: x.get("data", {}).get("href")
            )

            with suppress(KeyError):
                extents["alternate_href"] = extents["links"].apply(get_alternate_href)
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
            from typing import cast

            extents = cast(gpd.GeoDataFrame, pd.concat([extents, hrefs_df], axis=1))

    return extents


def get_matching_hrefs(
    region_of_interest: gpd.GeoDataFrame,
    collection_id: str,
    catalog_url: str = "https://coclico.blob.core.windows.net/stac/v1/catalog.json",
    use_alternate_href: bool = False,
    patch_url: Callable[[str], str] | None = None,
) -> list[str]:
    """
    Returns (optionally patched) HREFs from a STAC collection that intersect a region of interest.

    Args:
        region_of_interest (GeoDataFrame): Region to spatially filter STAC items.
        collection_id (str): STAC Collection ID (e.g., 'gcts', 'overture-buildings').
        catalog_url (str): STAC Catalog URL.
        use_alternate_href (bool): Whether to use 'alternate_href' column if present.
        patch_url (Callable): Optional function to patch URLs (used for snapshot + asset hrefs).

    Returns:
        list[str]: List of (optionally patched) HREFs intersecting the region.
    """
    # Load STAC collection
    catalog = pystac.Catalog.from_file(catalog_url)
    collection = catalog.get_child(collection_id)
    if collection is None:
        raise ValueError(f"Collection '{collection_id}' not found in catalog.")

    # Read snapshot (patch_url used to optionally patch GeoParquet snapshot href)
    snapshot = read_snapshot(
        collection,
        columns=(
            ["geometry", "assets", "links"]
            if use_alternate_href
            else ["geometry", "assets"]
        ),
        add_href=True,
    )

    # Spatial filter
    matched = gpd.sjoin(snapshot, region_of_interest.to_crs(4326), how="inner")

    key = "alternate_href" if use_alternate_href else "href"
    if key not in matched.columns:
        raise ValueError(f"Column '{key}' not found in snapshot.")

    raw_hrefs = matched[key].dropna().unique()

    if patch_url:
        return [patch_url(href) for href in raw_hrefs]

    return raw_hrefs.tolist()
