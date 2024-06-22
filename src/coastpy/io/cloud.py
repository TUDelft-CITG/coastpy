import logging
import pathlib
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr

from coastpy.io.utils import is_local_file_path


def to_https_url(href: str, storage_options: dict | None = None) -> str:
    """Converts a cloud storage href to its corresponding HTTPS URL.

    Args:
        href (str): The href string to be converted.
        storage_options (dict, optional): Dictionary of storage options that can contain
            platform-specific settings. For Azure Blob Storage, it should contain
            the 'account_name' key.

    Returns:
        str: The corresponding HTTPS URL.

    Raises:
        ValueError: If the protocol in href is unknown or unsupported or if required
            options are missing in 'storage_options'.
    """

    # Local file system
    if storage_options is None:
        storage_options = {}
    if pathlib.Path(href).exists():
        return f"file://{href}"

    # Google Cloud Storage
    for protocol in ["gs://", "gcs://"]:
        if href.startswith(protocol):
            return href.replace(protocol, "https://storage.googleapis.com/")

    # Azure Blob Storage
    if href.startswith(("az://", "azure://", "abfs://")):
        if "account_name" not in storage_options:
            msg = (
                "For Azure Blob Storage hrefs, the 'account_name' is required in"
                " 'storage_options'."
            )
            raise ValueError(msg)

        _, path = href.split("://", 1)
        account_name = storage_options["account_name"]
        return f"https://{account_name}.blob.core.windows.net/{path}"

    # AWS S3
    if href.startswith("s3://"):
        _, rest = href.split("://", 1)
        bucket, key = rest.split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"

    msg = f"Unknown or unsupported protocol in href: {href}"
    raise ValueError(msg)


def to_uri(href_or_uri: str, protocol: str | None = None) -> str:
    """
    Converts an HTTPS URL to its corresponding cloud storage URI based on the specified protocol.
    If the input is already a valid cloud URI, it adjusts the URI based on the provided protocol or returns it as is.

    Args:
        href_or_uri (str): The HTTPS URL or cloud URI to be converted or validated.
        protocol (Optional[str]): The desired cloud protocol ("gs", "gcs", "az", "azure", "abfs", "s3").

    Returns:
        str: The corresponding URI in the desired protocol.

    Raises:
        ValueError: If the protocol is unknown, unsupported, or href_or_uri does not match the expected format.

    Example:
        >>> to_uri_protocol("https://storage.googleapis.com/my-bucket/my-file.txt", protocol="gs")
        'gs://my-bucket/my-file.txt'
    """
    valid_protocols = ["gs", "gcs", "az", "azure", "abfs", "s3"]
    href_or_uri = str(href_or_uri)

    # Google Cloud Storage
    gcs_formats = [
        "https://storage.googleapis.com/",
        "https://storage.cloud.google.com/",
    ]

    for fmt in gcs_formats:
        if href_or_uri.startswith(fmt):
            prefix = href_or_uri.replace(fmt, "")
            if protocol in [None, "gs"]:
                return f"gs://{prefix}"
            elif protocol == "gcs":
                return f"gcs://{prefix}"
            else:
                msg = (
                    "For Google Cloud Storage href, valid protocols are 'gs' and"
                    f" 'gcs', but got '{protocol}'."
                )
                raise ValueError(msg)

    # Azure Blob Storage
    if ".blob.core.windows.net/" in href_or_uri:
        # Split the href to get the container and path
        container_and_path = href_or_uri.split(".blob.core.windows.net/", 1)[1]
        if protocol in [None, "az", "azure"]:
            return f"az://{container_and_path}"
        elif protocol == "abfs":
            return f"abfs://{container_and_path}"
        msg = (
            "For Azure Blob Storage href, valid protocols are 'az', 'azure', and"
            f" 'abfs', but got '{protocol}'."
        )
        raise ValueError(msg)

    # AWS S3
    if href_or_uri.startswith("https://s3.") and ".amazonaws.com/" in href_or_uri:
        parts = href_or_uri.replace("https://", "").split(".amazonaws.com/")
        bucket = parts[0].replace("s3.", "")
        key = parts[1]
        if protocol in [None, "s3"]:
            return f"s3://{bucket}/{key}"
        msg = f"For AWS S3 href, the valid protocol is 's3', but got '{protocol}'."
        raise ValueError(msg)

    # If the input is already a valid cloud URI
    for uri_prefix in ["gs://", "gcs://", "az://", "azure://", "abfs://", "s3://"]:
        if href_or_uri.startswith(uri_prefix):
            if protocol is None:
                return href_or_uri
            else:
                return to_uri(href_or_uri.replace(uri_prefix, "https://", 1), protocol)

    if is_local_file_path(href_or_uri):
        return href_or_uri

    msg = (
        f"Unknown or unsupported protocol: {protocol}. Valid protocols are"
        f" {', '.join(valid_protocols)}.\nOr href_or_uri does not match expected"
        f" format: {href_or_uri}"
    )

    raise ValueError(msg)


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
