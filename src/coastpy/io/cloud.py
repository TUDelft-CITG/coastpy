import logging
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.cog import save_cog_with_dask, to_cog


def write_block(
    block: xr.DataArray,
    blob_name: str,
    storage_options: dict[str, Any] | None = None,
    nodata: float | None = None,
    compression: str = "DEFLATE",
    blocksize: int = 512,
    overwrite: bool = True,
    use_dask: bool = True,
    overview_resampling: str | None = None,
    overview_levels: list[int] | None = None,
    tags: dict[str, str] | None = None,
    intermediate_compression: bool | str | dict[str, Any] = False,
    predictor: int | bool = 3,
    stats: bool = True,
    client: Any = None,
    extra_rio_opts: dict[str, Any] | None = None,
) -> int:
    """
    Save an xarray.DataArray as a Cloud Optimized GeoTIFF (COG), handling both
    in-memory and Dask-backed data.

    Args:
        block (xr.DataArray): Data to save.
        blob_name (str): Path or URL for the output COG.
        storage_options (Optional[Dict]): Storage backend options.
        nodata (Optional[float]): No-data value for the output raster.
        compression (str): Compression method for the GeoTIFF. Defaults to "DEFLATE".
        blocksize (int): Tile size for the COG. Defaults to 512.
        overwrite (bool): Whether to overwrite existing files.
        use_dask (bool): Use Dask for saving.
        overview_resampling (Optional[str]): Resampling method for overviews.
        overview_levels (Optional[list[int]]): Shrink factors for overviews.
        tags (Optional[Dict[str, str]]): Metadata tags for the output file.
        intermediate_compression (Union[bool, str, Dict]): Intermediate compression settings.
        predictor (Optional[int]): Compression predictor for floating-point data.
        stats (bool): Compute stats for GIS compatibility.
        client (Any): Dask client for distributed workflows.
        extra_rio_opts (Optional[Dict]): Additional `rasterio` options.

    Returns:
        int: Number of bytes written.

    Raises:
        ValueError: If CRS is missing or invalid.
    """
    # Ensure CRS is set
    if not block.rio.crs:
        raise ValueError("CRS is missing. Set a valid CRS using `rio.write_crs`.")

    if overview_levels is None:
        overview_levels = [2, 4, 8, 16, 32]

    # Ensure nodata is properly defined
    block = block.rio.write_nodata(nodata or float("nan"))

    # Determine storage backend
    storage_options = storage_options or {}
    fs, _, paths = fsspec.get_fs_token_paths(blob_name, storage_options=storage_options)
    if len(paths) > 1:
        raise ValueError("Too many paths specified.")
    path = paths[0]

    # Check if the file exists and overwrite is False
    if not overwrite and fs.exists(path):
        logging.info(f"File already exists at {path} and overwrite is disabled.")
        return fs.info(path)["size"]

    # Use in-memory writing if not using Dask
    if not use_dask or not block.chunks:
        cog_bytes = to_cog(
            block,
            blocksize=blocksize,
            compression=compression,
            nodata=nodata,
            overview_resampling=overview_resampling,
            overview_levels=overview_levels,
            intermediate_compression=intermediate_compression,
            tags=tags,
            **(extra_rio_opts or {}),
        )
        # Write bytes to target storage
        with fs.open(path, "wb") as f:
            f.write(cog_bytes)
    else:
        # Use Dask for large datasets
        save_cog_with_dask(
            block,
            dst=path,
            compression=compression,
            blocksize=blocksize,
            predictor=predictor,
            overview_resampling=overview_resampling,
            overview_levels=overview_levels,
            stats=stats,
            client=client,
            bigtiff=True,
            **(extra_rio_opts or {}),
        )

    # Return file size
    return fs.info(path)["size"]


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
