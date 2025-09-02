import logging
import os
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.cog import save_cog_with_dask, to_cog

from coastpy.utils.xarray_utils import get_nodata, set_nodata


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
    block = set_nodata(block, nodata)  # type: ignore
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


def write_parquet(
    df: pd.DataFrame | gpd.GeoDataFrame,
    urlpath: str,
    storage_options: dict | None = None,
    overwrite: bool = True,
) -> int:
    """
    Write a DataFrame or GeoDataFrame to Parquet using fsspec-compatible storage.

    Args:
        df: The (Geo)DataFrame to write.
        urlpath: Destination path (e.g., 'file:///tmp/file.parquet', 'abfs://container/file.parquet').
        storage_options: Optional dictionary of storage backend options (e.g., credentials).
        overwrite: If False and file exists, skips writing and returns size.

    Returns:
        Size in bytes of the written file, or existing file if overwrite is False.
    """
    if df.empty:
        return 0

    storage_options = storage_options or {}
    fs, _, paths = fsspec.get_fs_token_paths(urlpath, storage_options=storage_options)

    if len(paths) != 1:
        raise ValueError("Expected exactly one destination path", paths)

    path = paths[0]

    if not overwrite and fs.exists(path):
        return fs.info(path)["size"]

    if fs.protocol == "file":
        local_path = fsspec.utils.strip_protocol(path)
        parent_dir = os.path.dirname(local_path)
        if not fs.exists(parent_dir):
            fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(path, mode="wb") as f:
        df.to_parquet(f)

    return fs.info(path)["size"]
