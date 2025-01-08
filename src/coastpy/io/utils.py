import dataclasses
import datetime
import hashlib
import logging
import pathlib
from typing import Any
from urllib.parse import urlparse, urlsplit

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr
from pyproj import Transformer
from pystac.utils import datetime_to_str
from shapely.geometry import box
from shapely.ops import transform

logger = logging.getLogger(__name__)


def is_file(urlpath: str | pathlib.Path) -> bool:
    """
    Determine if a urlpath is a filesystem path by using urlsplit.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if it's a local file path, False otherwise.
    """
    parsed = urlsplit(str(urlpath))

    return parsed.scheme in ["", "file"]


@dataclasses.dataclass
class PathParser:
    """
    Parses cloud storage paths into components, supporting multiple protocols.

    Attributes:
        urlpath (str): Full URL or URI to parse.
        scheme (str): Protocol from the URL (e.g., "https", "az", "gs", "s3").
        container (str): Storage container, bucket, or top-level directory.
        prefix (str | None): Path prefix inside the container/bucket.
        name (str): File name including its extension.
        suffix (str): File extension (e.g., ".parquet", ".tif").
        stac_item_id (str): Identifier for STAC, derived from the file name.
        base_url (str): Base HTTPS URL for accessing the resource.
        base_uri (str): Base URI for the cloud storage provider.
        href (str): Full HTTPS URL for accessing the resource.
        uri (str): Full URI for cloud-specific access.
        account_name (str | None): Account name for cloud storage (required for Azure).
        path (str): Full path within the container, excluding the container name.
    """

    urlpath: str
    scheme: str = ""
    container: str = ""
    path: str = ""
    name: str = ""
    suffix: str = ""
    stac_item_id: str = ""
    https_url: str = ""
    cloud_uri: str = ""
    account_name: str | None = None
    cloud_protocol: str = ""
    base_dir: str | pathlib.Path = ""
    band: str = ""
    resolution: str = ""
    _base_https_url: str = ""
    _base_cloud_uri: str = ""

    SUPPORTED_PROTOCOLS = {"https", "az", "gs", "s3"}
    DEFAULT_BASE_DIR = pathlib.Path.home() / "data" / "tmp"

    def __post_init__(self):
        if is_file(self.urlpath):
            self._parse_path()
        else:
            self._parse_url()

    def _parse_url(self):
        parsed = urlparse(self.urlpath)

        # Basic parsing
        self.scheme = parsed.scheme
        self.name = parsed.path.split("/")[-1]
        self.suffix = f".{self.name.split('.')[-1]}"
        self.stac_item_id = self._extract_stac_item_id(self.name)

        if not self.base_dir:
            self.base_dir = self.DEFAULT_BASE_DIR

        # Check for supported protocols
        if self.scheme not in self.SUPPORTED_PROTOCOLS:
            msg = f"Unsupported protocol: {self.scheme}"
            raise ValueError(msg)

        # Protocol-specific parsing
        if self.scheme == "https":
            self.container = parsed.path.split("/")[1]
            self._base_https_url = (
                self.scheme + "://" + parsed.netloc + "/" + self.container
            )
            self.path = "/".join(parsed.path.split("/")[2:])

            if "windows" in self._base_https_url:
                if not self.cloud_protocol:
                    self.cloud_protocol = "az"
            elif "google" in self._base_https_url:
                if not self.cloud_protocol:
                    self.cloud_protocol = "gs"
            elif "amazon" in self._base_https_url:  # noqa: SIM102
                if not self.cloud_protocol:
                    self.cloud_protocol = "s3"

        elif self.scheme in {"az", "gs", "s3"}:
            self.path = parsed.path.lstrip("/")  # Remove leading slash
            self.container = parsed.netloc
            self.cloud_protocol = self.scheme

            if self.scheme == "az":
                if not self.account_name:
                    msg = "For 'az://' URIs, 'account_name' must be provided."
                    raise ValueError(msg)
                self._base_https_url = f"https://{self.account_name}.blob.core.windows.net/{self.container}"
            elif self.scheme == "gs":
                self._base_https_url = (
                    f"https://storage.googleapis.com/{self.container}"
                )
            elif self.scheme == "s3":
                self._base_https_url = f"https://{self.container}.s3.amazonaws.com"

    def _extract_stac_item_id(self, filename: str) -> str:
        """
        Extracts the stac_item_id from a filename, optionally removing a band prefix.

        Args:
            filename (str): The filename to process.

        Returns:
            str: The extracted stac_item_id.
        """
        if self.band and filename.startswith(self.band + "_"):
            return filename[len(self.band) + 1 :]  # Remove band prefix
        return filename.split(".")[0]  # Default behavio

    def _parse_path(self):
        path = pathlib.Path(self.urlpath)

        if not self.base_dir:
            msg = "For local file paths, 'base_dir' must be provided."
            raise ValueError(msg)

        if not self.cloud_protocol:
            msg = "For local file paths, 'cloud_protocol' must be provided."
            raise ValueError(msg)

        path = path.relative_to(self.base_dir)
        parts = path.parts

        if len(parts) < 2:
            msg = "Local file paths must have at least two components."
            raise ValueError(msg)

        self.container = parts[0]
        self.path = "/".join(parts[1:])
        self.name = path.name
        self.suffix = f".{path.suffix}"
        self.stac_item_id = self._extract_stac_item_id(path.stem)

        if self.cloud_protocol == "az":
            if not self.account_name:
                msg = "For 'az://' URIs, 'account_name' must be provided."
                raise ValueError(msg)
            self._base_https_url = (
                f"https://{self.account_name}.blob.core.windows.net/{self.container}"
            )

        elif self.cloud_protocol == "gs":
            self._base_https_url = f"https://storage.googleapis.com/{self.container}"

        elif self.cloud_protocol == "s3":
            self._base_https_url = f"https://{self.container}.s3.amazonaws.com"

    def to_filepath(
        self, base_dir: pathlib.Path | str | None = None, capitalize=False
    ) -> pathlib.Path:
        "Convert to local file path."
        if base_dir is None:
            base_dir = self.base_dir

        name_parts = [self.name.split(".")[0]]
        if self.band:
            name_parts.append(self.band)
        if self.resolution:
            name_parts.append(self.resolution)
        name_with_band_and_res = "_".join(name_parts)

        if capitalize:
            name_with_band_and_res = name_with_band_and_res.upper()

        name_with_band_and_res += self.suffix

        path_parts = self.path.rsplit("/", 1)
        path_with_band_and_res = (
            f"{path_parts[0]}/{name_with_band_and_res}"
            if len(path_parts) > 1
            else name_with_band_and_res
        )

        return pathlib.Path(base_dir) / self.container / path_with_band_and_res

    def _construct_url(self, base_url: str, capitalize: bool = False) -> str:
        "Construct URL or URI."
        name_parts = [self.name.split(".")[0]]

        if self.band:
            name_parts.append(self.band)

        if self.resolution:
            name_parts.append(self.resolution)

        file_name = "_".join(name_parts)

        if capitalize:
            file_name = file_name.upper()

        file_name += self.suffix

        return f"{base_url}/{self.path.rsplit('/', 1)[0]}/{file_name}"

    def to_https_url(self, capitalize: bool = False) -> str:
        "Convert to HTTPS URL."
        return self._construct_url(self._base_https_url, capitalize=capitalize)

    def to_cloud_uri(self, capitalize=False) -> str:
        "Convert to cloud storage URI."
        return self._construct_url(
            self.cloud_protocol + "://" + self.container, capitalize=capitalize
        )


def extract_datetimes(
    data: pd.DataFrame | gpd.GeoDataFrame | xr.Dataset | xr.DataArray,
) -> dict[str, datetime.datetime | None]:
    """
    Extract datetime information (datetime, start_datetime, end_datetime) from a dataset.

    Args:
        data: Input dataset (pandas DataFrame, GeoDataFrame, or xarray object).

    Returns:
        dict[str, datetime | None]: Dictionary with keys 'datetime', 'start_datetime', and 'end_datetime'.

    Raises:
        ValueError: If datetime information cannot be determined.
    """
    # Handle pandas DataFrame or GeoDataFrame
    if isinstance(data, (pd.DataFrame | gpd.GeoDataFrame)):
        # Check for 'start_datetime' and 'end_datetime' columns
        if "start_datetime" in data.columns and "end_datetime" in data.columns:
            start_datetime = pd.Timestamp(data["start_datetime"].min()).to_pydatetime()  # type: ignore
            end_datetime = pd.Timestamp(data["end_datetime"].max()).to_pydatetime()  # type: ignore
            return {
                "datetime": start_datetime,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
            }
        # Check for a single 'datetime' column
        elif "datetime" in data.columns:
            datetime_value = pd.Timestamp(data["datetime"].iloc[0]).to_pydatetime()
            return {
                "datetime": datetime_value,
                "start_datetime": None,
                "end_datetime": None,
            }

    # Handle xarray Dataset or DataArray
    elif isinstance(data, (xr.Dataset | xr.DataArray)):
        # Check for a time dimension
        if "time" in data.dims:
            time_values = data.coords["time"].values
            if len(time_values) > 1:
                return {
                    "datetime": pd.Timestamp(time_values[0]).to_pydatetime(),
                    "start_datetime": pd.Timestamp(time_values[0]).to_pydatetime(),
                    "end_datetime": pd.Timestamp(time_values[-1]).to_pydatetime(),
                }
            else:
                return {
                    "datetime": pd.Timestamp(time_values[0]).to_pydatetime(),
                    "start_datetime": None,
                    "end_datetime": None,
                }
        # Check for start and end datetime attributes
        if "start_datetime" in data.attrs and "end_datetime" in data.attrs:
            return {
                "datetime": pd.Timestamp(data.attrs["start_datetime"]).to_pydatetime(),
                "start_datetime": pd.Timestamp(
                    data.attrs["start_datetime"]
                ).to_pydatetime(),
                "end_datetime": pd.Timestamp(
                    data.attrs["end_datetime"]
                ).to_pydatetime(),
            }
        # Check for a single datetime attribute
        if "datetime" in data.attrs:
            return {
                "datetime": pd.Timestamp(data.attrs["datetime"]).to_pydatetime(),
                "start_datetime": None,
                "end_datetime": None,
            }

        # Check for datetime attributes in variables
        start_times = []
        end_times = []
        for var in data.data_vars.values():
            if "start_datetime" in var.attrs:
                start_times.append(
                    pd.Timestamp(var.attrs["start_datetime"]).to_pydatetime()
                )
            if "end_datetime" in var.attrs:
                end_times.append(
                    pd.Timestamp(var.attrs["end_datetime"]).to_pydatetime()
                )
        if start_times and end_times:
            return {
                "datetime": min(start_times),
                "start_datetime": min(start_times),
                "end_datetime": max(end_times),
            }

    # Raise error if no datetime information is found
    raise ValueError("Unable to determine datetime information from the dataset.")


def short_id(seed: str, length: int = 6) -> str:
    """
    Generate a short deterministic hash ID.

    Args:
        seed (str): Input string to hash.
        length (int): Length of the output hash.

    Returns:
        str: Short hash string.
    """
    return hashlib.md5(seed.encode()).hexdigest()[:length]


def transform_bounds_to_epsg4326(
    bounds: tuple[float, float, float, float], crs: Any
) -> tuple[float, float, float, float]:
    """
    Transform bounding box coordinates to EPSG:4326.

    Args:
        bounds (tuple): Bounding box as (minx, miny, maxx, maxy).
        crs (str): Coordinate reference system of the bounds.

    Returns:
        tuple: Transformed bounding box in EPSG:4326.
    """
    bounds_geometry = box(*bounds)
    if crs != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        bounds_geometry = transform(transformer.transform, bounds_geometry)
    return bounds_geometry.bounds


def format_bounds(
    minx: float, miny: float, maxx: float, maxy: float, precision: int = 6
) -> str:
    """
    Format bounding box coordinates with specified precision.

    Args:
        minx, miny, maxx, maxy (float): Bounding box coordinates.
        precision (int): Decimal precision for the bounds.

    Returns:
        str: Formatted bounding box string.
    """
    return f"{minx:.{precision}f}_{miny:.{precision}f}_{maxx:.{precision}f}_{maxy:.{precision}f}"


def name_bounds(
    bounds: tuple[float, float, float, float], crs: Any, precision: int = 0
) -> str:
    """
    Generate a location-based name for bounding box coordinates.

    Args:
        bounds (tuple): Bounding box as (minx, miny, maxx, maxy).
        crs (str): Coordinate reference system of the bounds.
        precision (int): Decimal precision for naming (default: integer degrees).

    Returns:
        str: Formatted name, e.g., "n45e123".
    """
    minx, miny, _, _ = transform_bounds_to_epsg4326(bounds, crs)

    lon_prefix = "e" if minx >= 0 else "w"
    lat_prefix = "n" if miny >= 0 else "s"

    formatted_minx = f"{abs(minx):0{3 + precision}.{precision}f}".replace(".", "")
    formatted_miny = f"{abs(miny):0{2 + precision}.{precision}f}".replace(".", "")

    return f"{lat_prefix}{formatted_miny}{lon_prefix}{formatted_minx}"


def name_bounds_with_hash(
    bounds: tuple[float, float, float, float],
    crs: Any,
    precision: int = 6,
    length: int = 6,
) -> str:
    """
    Generate a deterministic hash-based name for bounding box coordinates.

    Args:
        bounds (tuple): Bounding box as (minx, miny, maxx, maxy).
        crs (str): Coordinate reference system of the bounds.
        precision (int): Decimal precision for the bounds.
        length (int): Length of the hash suffix.

    Returns:
        str: Formatted name, e.g., "n45e123-abc".
    """
    minx, miny, maxx, maxy = transform_bounds_to_epsg4326(bounds, crs)

    # Format bounds with specified precision
    formatted_bounds = format_bounds(minx, miny, maxx, maxy, precision)

    # Generate a deterministic short hash
    hash_suffix = short_id(formatted_bounds, length=length)

    lon_prefix = "e" if minx >= 0 else "w"
    lat_prefix = "n" if miny >= 0 else "s"

    formatted_minx = f"{abs(minx):02.0f}"
    formatted_miny = f"{abs(miny):02.0f}"

    return f"{lat_prefix}{formatted_miny}{lon_prefix}{formatted_minx}-{hash_suffix}"


def name_data(
    data: pd.DataFrame | gpd.GeoDataFrame | xr.Dataset | xr.DataArray,
    prefix: str | None = None,
    filename_prefix: str | None = None,
    include_bounds: bool = True,
    add_deterministic_hash: bool = True,
    postfix: str | None = None,
    include_time: str | bool | None = None,
) -> str:
    """
    Generate a unique filename for a dataset based on bounds, hash, and optional components.

    Args:
        data: Spatial or non-spatial dataset (DataFrame, GeoDataFrame, or xarray).
        prefix: Optional prefix for the filename.
        filename_prefix: Additional prefix for further categorization.
        include_bounds: Include geographic bounds in the filename (default: True).
        add_deterministic_hash: Use deterministic hash for naming (default: True).
        postfix: Optional postfix to add after bounds/hash (e.g., "01", "02").
        include_time: Time component to include in the filename:
            - None (default): No time added.
            - str: Custom time string (e.g., "2023-2024").
            - True: Infer time from the data.

    Returns:
        str: A unique filename based on the provided options.

    Raises:
        ValueError: If no valid parts were generated for the filename.
    """
    # Synchronize include_bounds and add_deterministic_hash
    if not include_bounds:
        add_deterministic_hash = False

    parts = []

    # Generate bounds or bounds with hash
    if include_bounds and isinstance(
        data, gpd.GeoDataFrame | xr.Dataset | xr.DataArray
    ):
        if isinstance(data, gpd.GeoDataFrame):
            bounds = data.total_bounds
            crs = data.crs.to_string()
        else:
            bounds = data.rio.bounds()
            crs = data.rio.crs.to_string()

        if add_deterministic_hash:
            bounds_part = name_bounds_with_hash(bounds, crs)  # type: ignore
        else:
            bounds_part = name_bounds(bounds, crs)  # type: ignore

        if bounds_part:
            parts.append(bounds_part)

    # Add optional postfix
    if postfix:
        if parts:
            parts[-1] = f"{parts[-1]}-{postfix}"
        elif filename_prefix:
            filename_prefix = f"{filename_prefix}-{postfix}"
        else:
            parts.append(postfix)

    # Add time component
    if include_time:
        if isinstance(include_time, str):
            time_part = include_time
        elif isinstance(include_time, bool):
            datetime_info = extract_datetimes(data)
            if datetime_info["start_datetime"] and datetime_info["end_datetime"]:
                start = datetime_to_str(
                    datetime_info["start_datetime"], timespec="days"
                )
                end = datetime_to_str(datetime_info["end_datetime"], timespec="days")
                time_part = f"{start}_{end}"
            else:
                time_part = datetime_to_str(datetime_info["datetime"], timespec="auto")  # type: ignore
        else:
            raise ValueError(
                "Invalid value for `include_time`. Use None, str, or bool."
            )

        parts.append(time_part)

    if not parts:
        raise ValueError("No valid parts were generated for the filename.")

    # Determine file suffix
    suffix = ".parquet" if isinstance(data, pd.DataFrame | gpd.GeoDataFrame) else ".tif"
    name_part = "_".join(parts) + suffix

    # Construct filename components
    components = [comp for comp in [filename_prefix, name_part] if comp]
    filename = "_".join(components)

    return f"{prefix}/{filename}" if prefix else filename


def rm_from_storage(
    pattern: str,
    storage_options: dict[str, str] | None = None,
    confirm: bool = True,
    verbose: bool = True,
) -> None:
    """
    Deletes all blobs/files in the specified storage location that match the given prefix.

    Args:
        pattern (str): The pattern or path pattern (including wildcards) for the blobs/files to delete.
        storage_options (Dict[str, str], optional): A dictionary containing storage connection details.
        confirm (bool): Whether to prompt for confirmation before deletion.
        verbose (bool): Whether to display detailed log messages.

    Returns:
        None
    """
    if storage_options is None:
        storage_options = {}

    # Create a local logger
    logger = logging.getLogger(__name__)
    if verbose:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if storage_options is None:
        storage_options = {}

    # Get filesystem, token, and resolved paths
    fs, _, paths = fsspec.get_fs_token_paths(pattern, storage_options=storage_options)

    if paths:
        if verbose:
            logger.info(
                f"Warning: You are about to delete the following {len(paths)} blobs/files matching '{pattern}'."
            )
            for path in paths:
                logger.info(path)

        if confirm:
            confirmation = input(
                f"Type 'yes' to confirm deletion of {len(paths)} blobs/files matching '{pattern}': "
            )
        else:
            confirmation = "yes"

        if confirmation.lower() == "yes":
            for path in paths:
                try:
                    if verbose:
                        logger.info(f"Deleting blob/file: {path}")
                    fs.rm(path)
                    if verbose:
                        logger.info(f"Blob/file {path} deleted successfully.")
                except Exception as e:
                    if verbose:
                        logger.error(f"Failed to delete blob/file: {e}")
            if verbose:
                logger.info("All specified blobs/files have been deleted.")
        else:
            if verbose:
                logger.info("Blob/file deletion cancelled.")
    else:
        if verbose:
            logger.info(f"No blobs/files found matching '{pattern}'.")

    # Remove the handler after use
    if verbose:
        logger.removeHandler(handler)
        handler.close()
