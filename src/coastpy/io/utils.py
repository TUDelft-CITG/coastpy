import json
import pathlib
import uuid
import warnings
from datetime import datetime
from posixpath import join as urljoin
from urllib.parse import urlsplit

import fsspec
import geopandas as gpd
import pandas as pd
import xarray
import xarray as xr
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform


def is_local_file_path(path: str | pathlib.Path) -> bool:
    """
    Determine if a given path is a local filesystem path using urlsplit.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if it's a local file path, False otherwise.
    """
    parsed = urlsplit(str(path))

    return parsed.scheme in ["", "file"]


def name_block(
    block: xr.DataArray,
    storage_prefix: str = "",
    name_prefix: str | None = None,
    include_band: str | None = None,
    time_dim: str | None = "time",
    x_dim: str | None = "x",
    y_dim: str | None = "y",
) -> str:
    """Create a name for a block based on its coordinates and other options.

    Args:
        block (xr.DataArray): The DataArray block whose name is to be generated.
        storage_prefix (str | None, optional): Storage prefix to prepend to blob name. Defaults to None.
        name_prefix (Optional[str], optional): String prefix to prepend to the name. Defaults to None.
        include_band (Optional[str], optional): Name of the band to include in the name. Defaults to None.
        time_dim (Optional[str], optional): Name of the time dimension. Defaults to "time".
        x_dim (str | None, optional): Name of the x dimension. Defaults to "x".
        y_dim (str | None, optional): Name of the y dimension. Defaults to "y".

    Returns:
        str: A string name constructed from the block's coordinates.

    Raises:
        ValueError: If no valid components are available to form a name.
    """

    storage_prefix = str(storage_prefix) if storage_prefix else ""

    components = []

    epsg = block.rio.crs.to_epsg()

    if name_prefix:
        components.append(name_prefix)

    if (x_dim or y_dim) and (epsg != 4326 and epsg != 3857):
        components.append(f"epsg={block.rio.crs.to_epsg()}")

    if include_band:
        components.append(include_band)

    if time_dim and time_dim in block.coords:
        time = pd.Timestamp(block.coords[time_dim].item()).isoformat()
        components.append(f"{time_dim}={time}")

    if x_dim and x_dim in block.coords:
        if epsg == 4326:
            x_val = round(block.coords[x_dim].min().item(), 2)
        else:
            x_val = int(round(block.coords[x_dim].min().item(), 2))
        components.append(f"{x_dim}={x_val}")

    if y_dim and y_dim in block.coords:
        if epsg == 4326:
            y_val = round(block.coords[y_dim].min().item(), 2)
        else:
            y_val = int(round(block.coords[y_dim].min().item(), 2))
        components.append(f"{y_dim}={y_val}")

    if not components:
        msg = (
            "No valid components to form a name. Check the input DataArray and provided"
            " arguments."
        )
        raise ValueError(msg)

    name = f"{'_'.join(components)}.tif"

    if not is_local_file_path(storage_prefix):  # cloud storage
        return str(urljoin(storage_prefix, name))
    else:  # local storage
        return str(pathlib.Path(storage_prefix) / name)


def name_table(
    gdf: gpd.GeoDataFrame,
    storage_prefix: str = "",
    name_prefix: str | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
) -> str:
    """Create a name for a Parquet file based on the total_bounds of a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame whose file name is to be generated.
        storage_prefix (str, optional): Storage prefix to prepend to file name. Defaults to "".
        name_prefix (str | None, optional): String prefix to prepend to the name. Defaults to None.
        x_dim (str, optional): Name of the x dimension. Defaults to "x".
        y_dim (str, optional): Name of the y dimension. Defaults to "y".

    Returns:
        str: A string name constructed from the GeoDataFrame's total bounds.

    Raises:
        ValueError: If no valid components are available to form a name.
    """

    storage_prefix = str(storage_prefix) if storage_prefix else ""

    components = []

    if name_prefix:
        components.append(name_prefix)

    epsg = gdf.crs.to_epsg()
    if epsg and epsg not in [4326, 3857]:
        components.append(f"epsg={epsg}")
        warnings.warn(
            "It is recommended to save GeoSpatial tabular data in EPSG 4326 or 3857 to"
            " ease global analyses.",
            stacklevel=2,
        )

    # Using total bounds for naming
    try:
        total_bounds = gdf.total_bounds
        x_min, y_min, _, _ = map(int, total_bounds)
        components.append(f"{x_dim}={x_min}_{y_dim}={y_min}")
    except ValueError:
        components.append("")

    if not components:
        msg = (
            "No valid components to form a name. Check the input GeoDataFrame and"
            " provided arguments."
        )
        raise ValueError(msg)

    name = f"{'_'.join(components)}.parquet"

    if not is_local_file_path(storage_prefix):
        return str(urljoin(storage_prefix, name))
    else:
        return str(pathlib.Path(storage_prefix) / name)


def name_data(
    data: pd.DataFrame | gpd.GeoDataFrame | xarray.Dataset | xarray.DataArray,
    prefix: str | None = None,
    filename_prefix: str | None = None,
    include_bounds: bool = True,
    include_random_hex: bool = True,
) -> str:
    """
    Generates a unique filename for a dataset with optional geographic bounds and random hex string.
    If neither bounds nor hex string are included, raises a ValueError.

    Args:
        data: A DataFrame, GeoDataFrame, or xarray dataset/data array representing spatial or non-spatial data.
        prefix: An optional prefix for the filename, typically used for higher-level grouping.
        filename_prefix: An additional prefix for the filename, used for further categorization.
        include_bounds: Flag to include geographic bounds in the filename for geospatial datasets.
        include_random_hex: Flag to include a random hex string in the filename.
        include_random_hex: The zoom level of the QuadKey prefix to include.

    Returns:
        A string representing the uniquely generated filename based on the provided options.

    Raises:
        ValueError: If both include_bounds and include_random_hex are False, or if the data type is unsupported.
    """
    if not include_bounds and not include_random_hex:
        msg = "At least one of include_bounds, include_random_hex or quadkey_prefix must be set."
        raise ValueError(msg)

    suffix = ".parquet"  # Default suffix for non-spatial data
    parts = []

    # Generate bounds part
    bounds_part = ""
    if include_bounds and isinstance(
        data, gpd.GeoDataFrame | xarray.Dataset | xarray.DataArray
    ):
        if isinstance(data, gpd.GeoDataFrame):
            bounds = data.total_bounds
            crs = data.crs.to_string()
        else:  # xarray
            bounds = data.rio.bounds()
            crs = data.rio.crs.to_string()

        def name_bounds(bounds, crs):
            bounds_geometry = box(*bounds)
            if crs != "EPSG:4326":
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                bounds_geometry = transform(transformer.transform, bounds_geometry)
            minx, miny, _, _ = bounds_geometry.bounds
            lon_prefix = "e" if minx >= 0 else "w"
            lat_prefix = "n" if miny >= 0 else "s"
            formatted_minx = f"{abs(minx):03.0f}"
            formatted_miny = f"{abs(miny):02.0f}"
            return f"{lat_prefix}{formatted_miny}{lon_prefix}{formatted_minx}"

        bounds_part = name_bounds(bounds, crs)
        if bounds_part:
            parts.append(bounds_part)

    # Generate random hex part
    random_hex = ""
    if include_random_hex:
        random_hex = uuid.uuid4().hex[:12]
        parts.append(f"part-{random_hex}")

    if not parts:
        msg = "No valid parts were generated for the filename."
        raise ValueError(msg)

    name_part = "_".join(parts) + suffix

    # Constructing the final filename
    components = [comp for comp in [filename_prefix, name_part] if comp]
    filename = "_".join(components)

    if prefix:
        return f"{prefix}/{filename}"

    return filename


def read_items_extent(collection, columns=None, storage_options=None):
    if storage_options is None:
        storage_options = {}

    if columns is None:
        columns = ["geometry", "assets"]

    required_cols = ["geometry", "assets"]

    for col in required_cols:
        if col not in columns:
            columns = [*columns, col]

    href = collection.assets["geoparquet-stac-items"].href
    with fsspec.open(href, mode="rb", **storage_options) as f:
        extents = gpd.read_parquet(f, columns=columns)
        extents["href"] = extents.assets.map(lambda x: x["data"]["href"])
    return extents


def write_log_entry(
    container: str,
    name: str,
    status: str,
    storage_options: dict[str, str] | None = None,
    prefix: str | None = None,
    time: str | datetime | None = None,
) -> None:
    """
    Adds a new entry to a JSON log file with a given name (can be URI), time, and status.
    The log file is stored in the specified container and optionally prefixed with the given prefix.

    Args:
        container (str): The base path to the log container.
        name (str): The name to log.
        status (str): The status of the processing (e.g., 'success', 'failed').
        storage_options (Optional[Dict[str, str]]): Authentication and configuration options for Azure storage.
        prefix (Optional[str]): The prefix for the log file path.
        time (Optional[Union[str, datetime]]): The time of logging. If None, the current time is used.
    """
    if storage_options is None:
        storage_options = {}
    # Generate a UUID for the log file name
    log_id = uuid.uuid4().hex[:16]
    log_filename = f"{log_id}.json"

    # Construct the full path to the log file
    log_path = (
        f"{container}/{prefix}/{log_filename}"
        if prefix
        else f"{container}/{log_filename}"
    )

    # Ensure the time is in ISO format
    if time is None:
        time = datetime.now().isoformat()
    else:
        try:
            if isinstance(time, str):
                time = datetime.fromisoformat(time).isoformat()
            elif isinstance(time, datetime):
                time = time.isoformat()
            else:
                raise ValueError
        except ValueError:
            time = datetime.now().isoformat()

    # Create the log entry
    log_entry = {"name": name, "time": time, "status": status}

    # Write the log entry to the specified path in JSON format
    with fsspec.open(log_path, "w", **storage_options) as f:
        json.dump(log_entry, f)


def read_log_entries(
    base_uri: str, storage_options: dict[str, str | None], prefix: str | None = None
) -> pd.DataFrame:
    """
    Lists all JSON log files under a specified prefix, reads them into a DataFrame,
    and sorts them by ascending time.

    Args:
        base_uri (str): The base URI to search for JSON log files.
        storage_options (Dict[str, str]): Authentication and configuration options for Azure storage.
        prefix (Optional[str]): A prefix to filter the log files.

    Returns:
        pd.DataFrame: A DataFrame of log entries sorted by ascending date.
    """
    protocol = base_uri.split("://")[0]
    fs = fsspec.filesystem(protocol, **storage_options)

    # Use glob to list all JSON files under the base URI with the optional prefix
    search_pattern = f"{base_uri}/{prefix}/*.json" if prefix else f"{base_uri}/*.json"
    json_files = fs.glob(search_pattern)

    # Read JSON files into a list of dictionaries
    logs: list[dict] = []
    for f in json_files:
        with fs.open(f, "r", **storage_options) as f2:
            log_entry = json.load(f2)
            logs.append(log_entry)

    # If no log entries are found, return an empty DataFrame
    if not logs:
        return pd.DataFrame(columns=["name", "time", "status"])

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(logs)

    # Ensure the time column is in datetime format, raise an error if parsing fails
    try:
        df["time"] = pd.to_datetime(df["time"], errors="raise")
    except Exception as e:
        msg = f"Error parsing datetime: {e}"
        raise ValueError(msg)  # noqa: B904

    # Sort DataFrame by time in ascending order
    df = df.sort_values(by="time", ascending=True)

    return df
