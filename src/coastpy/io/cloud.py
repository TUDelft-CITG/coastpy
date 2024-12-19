import logging
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.cog import save_cog_with_dask, to_cog

from coastpy.utils.xarray import get_nodata, set_nodata


def write_block(
    block: xr.DataArray,
    blob_name: str,
    storage_options: dict[str, Any] | None = None,
    nodata: float | None = None,
    compression: str = "DEFLATE",
    blocksize: int = 512,
    overwrite: bool = True,
    use_dask: bool = True,
    overview_resampling: str = "nearest",
    overview_levels: list[int] | None = None,
    tags: dict[str, str] | None = None,
    predictor: int = 3,
    stats: bool = True,
    client: Any = None,
    extra_rio_opts: dict[str, Any] | None = None,
    extra_dask_opts: dict[str, Any] | None = None,
) -> int:
    """
    Save an xarray.DataArray as a Cloud Optimized GeoTIFF (COG) using either
    in-memory or distributed workflows.

    Args:
        block (xr.DataArray): Data to save.
        blob_name (str): Path or URL for the output COG.
        storage_options (Optional[Dict]): Storage backend options.
        nodata (Optional[float]): No-data value for the output raster.
        compression (str): Compression method for the GeoTIFF. Defaults to "DEFLATE".
        blocksize (int): Tile size for the COG. Defaults to 512.
        overwrite (bool): Whether to overwrite existing files.
        use_dask (bool): Use Dask for saving. Defaults to True.
        overview_resampling (str): Resampling method for overviews. Defaults to "nearest".
        overview_levels (Optional[list[int]]): Shrink factors for overviews. Defaults to [2, 4, 8, 16, 32].
        tags (Optional[Dict[str, str]]): Metadata tags for the output file.
        predictor (int): Compression predictor for floating-point data. Defaults to 3.
        stats (bool): Compute stats for GIS compatibility. Defaults to True.
        client (Any): Dask client for distributed workflows.
        use_windowed_writes (bool): Enable windowed writes for large images. Defaults to False.
        intermediate_compression (Union[bool, str, Dict]): Intermediate compression settings.
        extra_rio_opts (Optional[Dict]): Additional `rasterio` options.
        extra_dask_opts (Optional[Dict]): Additional Dask-specific options.

    Returns:
        int: Number of bytes written.

    Raises:
        ValueError: If CRS is missing or invalid.
    """
    # Validate CRS
    if not block.rio.crs:
        raise ValueError("CRS is missing. Set a valid CRS using `rio.write_crs`.")

    # Default overview levels
    if overview_levels is None:
        overview_levels = [2, 4, 8, 16, 32]

    # Set nodata value
    if nodata is None:
        nodata = get_nodata(block)
        nodata = nodata or float("nan")
    block = set_nodata(block, nodata)
    block = block.rio.write_nodata(nodata)

    # Determine storage backend
    storage_options = storage_options or {}
    fs, _, paths = fsspec.get_fs_token_paths(blob_name, storage_options=storage_options)
    if len(paths) > 1:
        raise ValueError("Too many paths specified.")
    path = paths[0]

    # Check if file exists
    if not overwrite and fs.exists(path):
        logging.info(f"File already exists at {path} and overwrite is disabled.")
        return fs.info(path)["size"]

    # Save data
    if not use_dask or not block.chunks:
        if extra_rio_opts is None:
            extra_rio_opts = {}

        if "compress" not in extra_rio_opts:
            extra_rio_opts["compress"] = compression

        cog_bytes = to_cog(
            block,
            blocksize=blocksize,
            overview_resampling=overview_resampling,
            overview_levels=overview_levels,
            tags=tags,
            nodata=nodata,
            # NOTE: these options are not yet supported by write_block
            # intermediate_compression=intermediate_compression,
            # use_windowed_writes=use_windowed_writes,
            **(extra_rio_opts or {}),
        )
        with fs.open(path, "wb") as f:
            f.write(cog_bytes)
    else:
        logging.info("Saving using `save_cog_with_dask` (distributed).")
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
            **(extra_dask_opts or {}),
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
