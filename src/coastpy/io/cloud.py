import logging
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr


def write_block(
    block: xr.DataArray,
    blob_name: str,
    storage_options: dict[str, Any] | None = None,
    profile_options: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> int:
    """
    Write a given block of data to a specified cloud storage location.

    Args:
        block (xr.DataArray): The data block to be written to cloud storage.
        blob_name (str): The target storage path, including the blob's name.
        storage_options (Optional[Dict[str, Any]]): Configuration options for the specific
            storage backend (e.g., authentication details). Defaults to an empty dictionary.
        profile_options (Optional[Dict[str, Any]]): Configuration options for rasterio's
            `to_raster` method when writing the block. Defaults to an empty dictionary.
        overwrite (bool): If True, overwrites the existing blob. Defaults to False.

    Returns:
        int: The number of bytes written, or the size of the existing blob if overwrite is False.

    Raises:
        ValueError: If more than one storage path is identified or no valid components
            are found to form a name.

    Example:
        >>> block = xr.DataArray(...)
        >>> write_block(block, "s3://mybucket/data.tif")
        1024  # hypothetical size of the written data in bytes.
    """
    import rioxarray  # noqa: F401

    if profile_options is None:
        profile_options = {}
    if storage_options is None:
        storage_options = {}

    fs, _, paths = fsspec.get_fs_token_paths(blob_name, storage_options=storage_options)

    if len(paths) > 1:
        msg = "too many paths"
        raise ValueError(msg, paths)

    path = paths[0]

    # Check if the blob exists and overwrite is False, then return its size.
    if not overwrite:  # noqa : SIM103
        if fs.exists(path):
            logging.info("Blob already exists and overwrite is set to False.")
            return fs.info(path)["size"]

    memfs = fsspec.filesystem("memory")

    with memfs.open("data", "wb") as buffer:
        block.squeeze().rio.to_raster(buffer, **profile_options)
        buffer.seek(0)

        if fs.protocol == "file":
            nbytes = len(buffer.getvalue())
        else:
            nbytes = buffer.getbuffer().nbytes

        # use the open method from fsspec to write the buffer
        with fs.open(path, "wb") as f:
            f.write(buffer.getvalue())

    return nbytes


def write_table(
    df: pd.DataFrame | gpd.GeoDataFrame,
    blob_name: str,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> int | None:
    """
    Write a pandas or geopandas DataFrame to a specified cloud storage location.

    Args:
        df (pd.DataFrame | gpd.GeoDataFrame): The DataFrame to be written to cloud storage.
        blob_name (str): The target storage path, including the blob's name.
        storage_options (Optional[Dict[str, Any]]): Configuration options for the specific
            storage backend (e.g., authentication details). Defaults to an empty dictionary.
        overwrite (bool): If True, overwrites the existing blob. Defaults to False.

    Returns:
        int: The number of bytes written, or the size of the existing blob if overwrite is False.

    Raises:
        ValueError: If more than one storage path is identified.

    Example:
        >>> df = pd.DataFrame(...)
        >>> write_table(df, "s3://mybucket/data.parquet")
    """
    if df.empty:
        return 0

    if storage_options is None:
        storage_options = {}

    fs, _, paths = fsspec.get_fs_token_paths(blob_name, storage_options=storage_options)

    if len(paths) > 1:
        msg = "Too many paths identified"
        raise ValueError(msg, paths)

    path = paths[0]

    # Check if the blob exists and overwrite is False, then return its size.
    if not overwrite and fs.exists(path):
        logging.info("Blob already exists and overwrite is set to False.")
        return fs.info(path)["size"]

    # TODO: return size of table when written to storage
    # Write DataFrame to a buffer
    with fsspec.open(blob_name, "wb", **storage_options) as f:
        df.to_parquet(f)
