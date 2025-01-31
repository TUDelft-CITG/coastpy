import dataclasses
import datetime
import hashlib
import logging
import pathlib
from pathlib import Path
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
    """A robust parser for local file paths, HTTPS URLs, and Azure cloud storage URIs."""

    original_path: str  # Input path (local, HTTPS, or cloud URI)
    account_name: str = ""  # Required for Azure (az://)
    band: str = ""  # Band information (e.g., "nir", "green")
    resolution: str = ""  # Resolution (e.g., "10m", "30m")

    # Internal attributes for path management
    _protocol: str = ""  # Internal protocol (file, https, az)
    _https_netloc: str = ""  # Standardized HTTPS base domain
    _cloud_netloc: str = ""  # Cloud provider URI prefix (az://)
    _bucket: str = ""  # Cloud storage bucket/container
    _key: str = ""  # Path inside the bucket
    _directory: Path = Path()  # Local parent directory
    _stem: str = ""  # Base filename (without suffix)
    _suffix: str = ""  # File extension (e.g., .tif)

    def __post_init__(self):
        """Automatically parse the given path based on its protocol."""
        self.protocol = fsspec.utils.get_protocol(self.original_path)  # Triggers setter

    ### **🔹 Protocol Handling**
    @property
    def protocol(self) -> str:
        """Getter for protocol."""
        return self._protocol

    @protocol.setter
    def protocol(self, new_protocol: str):
        """Setter for protocol. Updates attributes dynamically when protocol changes."""
        self._protocol = new_protocol
        self._parse_path()  # Re-parse with updated protocol

    ### **🔹 Path Parsing**
    def _parse_path(self):
        """Parses the path into components based on protocol."""
        parsed = urlparse(self.original_path)

        if self.protocol == "file":
            self._parse_local_path()
        elif self.protocol in {"https", "az"}:
            self._parse_azure_path(parsed)
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")

    def _parse_local_path(self):
        """Extracts components for local file paths."""
        path_obj = Path(self.original_path)
        self._directory = path_obj.parent
        self._stem = path_obj.stem  # Extract base name (without extension)
        self._suffix = path_obj.suffix  # Extract extension

        # Reset cloud attributes
        self._bucket = ""
        self._key = ""
        self._cloud_netloc = ""
        self._https_netloc = ""

    def _parse_azure_path(self, parsed):
        """Extracts components for Azure cloud storage paths."""
        self._key = parsed.path.lstrip("/")  # Remove leading slash
        filename = Path(self._key).name
        self._stem = Path(filename).stem
        self._suffix = Path(filename).suffix

        if self.protocol == "https":
            # Extract account_name and enforce proper parsing
            if "blob.core.windows.net" not in parsed.netloc:
                raise ValueError("Invalid Azure HTTPS URL format.")

            self.account_name = parsed.netloc.split(".")[0]
            self._bucket, self._key = self._key.split("/", 1)
            self._cloud_netloc = "az://"
            self._https_netloc = f"https://{self.account_name}.blob.core.windows.net"

        elif self.protocol == "az":
            # Cloud URI format: az://<bucket>/<key>
            self._bucket = parsed.netloc
            self._key = parsed.path.lstrip("/")
            self._cloud_netloc = "az://"

            # Allow account_name to be set later for HTTPS conversion
            if self.account_name:
                self._https_netloc = (
                    f"https://{self.account_name}.blob.core.windows.net"
                )

    ### **🔹 Properties for Path Components**
    @property
    def bucket(self) -> str:
        """Getter for bucket/container name."""
        return self._bucket

    @bucket.setter
    def bucket(self, new_bucket: str):
        """Setter for bucket. Ensures attributes remain consistent."""
        self._bucket = new_bucket

    @property
    def key(self) -> str:
        """Returns the directory portion of the key if a filename exists, otherwise returns the full key.

        Returns:
            str: The directory portion of the key, or the full key if no filename exists.
        """
        if (
            self._key and self._stem
        ):  # Only strip if we have both a directory and a filename
            return "/".join(self._key.split("/")[:-1]) if "/" in self._key else ""
        return self._key  # If no filename, return full key

    @key.setter
    def key(self, new_key: str):
        """Sets the key, ensuring the filename (if present) is stored separately.

        Args:
            new_key (str): The new key path (can be a full path with filename or just a directory).
        """
        if "." in Path(new_key).name:  # If it has an extension, treat it as a filename
            self._key = (
                "/".join(new_key.split("/")[:-1]) if "/" in new_key else ""
            )  # Directory portion
            self._stem = Path(new_key).stem  # Extract filename stem
            self._suffix = Path(new_key).suffix  # Extract file extension
        else:
            self._key = new_key  # Entire key is just a directory

    @property
    def filename(self) -> str:
        """Dynamically constructs the filename based on band and resolution."""
        name_parts = [self._stem]
        if self.band:
            name_parts.append(self.band)
        if self.resolution:
            name_parts.append(self.resolution)

        return f"{'_'.join(name_parts)}{self._suffix}"

    @property
    def https_netloc(self) -> str:
        """Getter for HTTPS netloc."""
        return self._https_netloc

    @property
    def cloud_netloc(self) -> str:
        """Getter for Cloud netloc."""
        return self._cloud_netloc

    @property
    def stac_item_id(self) -> str:
        if self.band and self._stem.startswith(self.band + "_"):
            return self._stem[len(self.band) + 1 :]
        return self._stem

    def _update_attributes(self, **kwargs):
        """Updates multiple attributes dynamically based on provided kwargs."""
        if "protocol" in kwargs:
            self.protocol = kwargs["protocol"]
        if "bucket" in kwargs:
            self.bucket = kwargs["bucket"]
        if "key" in kwargs:
            self.key = kwargs["key"]
        if "account_name" in kwargs:
            self.account_name = kwargs["account_name"]
        if "band" in kwargs:
            self.band = kwargs["band"]
        if "resolution" in kwargs:
            self.resolution = kwargs["resolution"]

    def to_https_url(self, **kwargs) -> str:
        """Constructs a valid Azure HTTPS URL from cloud components."""
        self._update_attributes(**kwargs)

        if not self.bucket:
            raise ValueError("Cannot generate HTTPS URL. Bucket is missing.")

        if not self.account_name:
            raise ValueError(
                "Azure requires an `account_name` for HTTPS URLs. "
                "Provide one via `to_https_url(account_name='...')`"
            )

        # Correctly construct the URL, ensuring no extra `/`
        key_part = f"{self.key}/" if self.key else ""
        return f"https://{self.account_name}.blob.core.windows.net/{self.bucket}/{key_part}{self.filename}"

    def to_cloud_uri(self, **kwargs) -> str:
        """Constructs a valid Azure cloud URI."""
        self._update_attributes(**kwargs)

        if not self.bucket:
            raise ValueError("Cannot generate cloud URI. Bucket is missing.")

        # Correctly construct the URI, ensuring no extra `/`
        key_part = f"{self.key}/" if self.key else ""
        return f"{self.cloud_netloc}{self.bucket}/{key_part}{self.filename}"

    def to_filepath(self, directory: str | Path = "") -> Path:
        """Converts to a local file path."""
        directory = Path(directory) if directory else self._directory
        return directory / self.filename


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
                    "datetime": pd.Timestamp(time_values).to_pydatetime(),
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
