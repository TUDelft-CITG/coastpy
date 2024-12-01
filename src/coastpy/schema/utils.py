import logging

import fsspec
import geopandas as gpd
import pandas as pd
from fsspec.utils import get_protocol

from coastpy.schema import BaseModel

logger = logging.getLogger(__name__)


def resolve_path(pathlike: str, fs: fsspec.AbstractFileSystem) -> str:
    """
    Resolve the path for either local or cloud storage using the provided filesystem object.

    Args:
        pathlike (str): Path to the file or pattern.
        fs (fsspec.AbstractFileSystem): Filesystem object for storage access.

    Returns:
        str: Resolved path (signed URL for cloud storage or the local path).
    """
    protocol = fs.protocol
    if protocol in {"az", "abfs", "s3", "gcs"}:  # Cloud storage protocols
        storage_options = fs.storage_options
        account_name = storage_options.get("account_name")
        sas_token = storage_options.get(
            "sas_token", storage_options.get("credential", "")
        )
        if not account_name:
            msg = "Missing 'account_name' in storage options for cloud storage."
            raise ValueError(msg)
        base_url = f"https://{account_name}.blob.core.windows.net"
        return (
            f"{base_url}/{pathlike}?{sas_token}"
            if sas_token
            else f"{base_url}/{pathlike}"
        )
    return pathlike


def write_record(
    record: BaseModel,
    pathlike: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """
    Read a single record from cloud storage and parse it into the specified model.

    Args:
        pathlike (str): Path to the specific record inside the container.
        model (Type[BaseModel]): The model class to decode the record into.
        fs (fsspec.AbstractFileSystem): Filesystem object for storage access.

    Returns:
        BaseModel: Parsed instance of the specified data model.
    """
    try:
        with fs.open(pathlike, mode="w") as f:
            f.write(record.to_json())
    except Exception as e:
        logger.error(f"Failed to write or encode record at {pathlike}: {e}")
        msg = f"Error writing record at {pathlike}: {e}"
        raise ValueError(msg) from e


def read_record(
    pathlike: str, model: type[BaseModel], fs: fsspec.AbstractFileSystem
) -> BaseModel:
    """
    Read a single record from cloud storage and parse it into the specified model.

    Args:
        pathlike (str): Path to the specific record in storage.
        model (type[BaseModel]): The model class to decode the record into.
        fs (fsspec.AbstractFileSystem): Filesystem object for accessing cloud storage.

    Returns:
        BaseModel: An instance of the parsed data model.

    Raises:
        ValueError: If reading or parsing fails.
    """
    pathlike = resolve_path(pathlike, fs=fs)
    try:
        with fs.open(pathlike, mode="r") as f:
            return model().decode(f.read())
    except Exception as e:
        logger.error(f"Failed to read or decode record at {pathlike}: {e}")
        msg = f"Error reading record at {pathlike}: {e}"
        raise ValueError(msg) from e


def read_records(
    paths: list[str], model: type[BaseModel], fs: fsspec.AbstractFileSystem
) -> list[BaseModel]:
    """
    Read and parse multiple records from cloud storage into a list of model instances.

    Args:
        paths (list[str]): List of paths to the records.
        model (type[BaseModel]): The model class to decode the records into.
        fs (fsspec.AbstractFileSystem): Filesystem object for accessing cloud storage.

    Returns:
        list[BaseModel]: A list of parsed model instances.

    Raises:
        ValueError: If no valid records are found.
    """
    records = []
    for path in paths:
        try:
            record = read_record(path, model, fs)
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to load record from {path}: {e}")

    if not records:
        msg = "No valid records found."
        raise ValueError(msg)

    return records


def read_records_to_pandas(
    container: str,
    model: type[BaseModel],
    storage_options: dict,
    **kwargs,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Load multiple records from cloud storage and convert them into a Pandas/GeoPandas DataFrame.

    Args:
        container (str): Path or pattern to the container (e.g., "az://container/*.json").
        model (type[BaseModel]): The model class to decode the records into.
        storage_options (dict): Configuration for accessing cloud storage.
        **kwargs: Additional arguments passed to Pandas/GeoPandas for DataFrame creation.

    Returns:
        pd.DataFrame | gpd.GeoDataFrame: A DataFrame containing the records.

    Raises:
        ValueError: If no valid records are found.
    """
    # Validate inputs
    if not isinstance(container, str):
        msg = f"Expected 'container' to be a string, got {type(container).__name__}."
        raise ValueError(msg)
    if not issubclass(model, BaseModel):
        msg = f"Expected 'model' to be a subclass of BaseModel, got {type(model).__name__}."
        raise ValueError(msg)

    protocol = get_protocol(container)
    fs = fsspec.filesystem(protocol, **storage_options)
    paths = fs.glob(container)

    if not paths:
        msg = f"No records found in the container: {container}"
        raise ValueError(msg)

    # Read records
    records = read_records(paths, model, fs)

    # Convert records to DataFrame
    data_frames = [r.to_frame() for r in records]
    combined_df = pd.concat(data_frames, **kwargs).reset_index(drop=True)

    # Handle geometries
    if "geometry" in combined_df.columns:
        return gpd.GeoDataFrame(combined_df, geometry="geometry", crs="EPSG:4326")

    return combined_df
