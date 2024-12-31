import dataclasses
import hashlib
import logging
import pathlib
import uuid
import warnings
from posixpath import join as urljoin
from typing import Any
from urllib.parse import urlparse, urlsplit

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr
from pyproj import Transformer
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

    if not is_file(storage_prefix):  # cloud storage
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

    if not is_file(storage_prefix):
        return str(urljoin(storage_prefix, name))
    else:
        return str(pathlib.Path(storage_prefix) / name)


def name_bounds(bounds: tuple, crs: Any):
    """
    Generate a location-based name for bounding box coordinates.

    Args:
        bounds (tuple): Bounding box as (minx, miny, maxx, maxy).
        crs (str): Coordinate reference system of the bounds.

    Returns:
        str: Formatted name, e.g., "n45e123".
    """
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


def name_data(
    data: pd.DataFrame | gpd.GeoDataFrame | xr.Dataset | xr.DataArray,
    prefix: str | None = None,
    filename_prefix: str | None = None,
    include_bounds: bool = True,
    include_random_hex: bool = False,
) -> str:
    """
    Generate a unique filename for a dataset based on bounds and a deterministic hash.

    Args:
        data: Spatial or non-spatial dataset (DataFrame, GeoDataFrame, or xarray).
        prefix: Optional prefix for the filename.
        filename_prefix: Additional prefix for further categorization.
        include_bounds: Include geographic bounds in the filename.
        include_random_hex: Include a random hex string (overridden by deterministic hash).

    Returns:
        str: A unique filename based on the provided options.

    Raises:
        ValueError: If no valid parts are generated for the filename.
    """
    if not include_bounds and not include_random_hex and not filename_prefix:
        msg = "At least one of include_bounds or include_random_hex or filename_prefix must be set."
        raise ValueError(msg)

    parts = []

    # Generate bounds part with deterministic hash
    bounds_part = ""
    if include_bounds and isinstance(
        data, gpd.GeoDataFrame | xr.Dataset | xr.DataArray
    ):
        if isinstance(data, gpd.GeoDataFrame):
            bounds = data.total_bounds
            crs = data.crs.to_string()
        else:
            bounds = data.rio.bounds()
            crs = data.rio.crs.to_string()

        bounds_part = name_bounds_with_hash(bounds, crs)  # type: ignore
        if bounds_part:
            parts.append(bounds_part)

    # Generate random hex part (optional, for fallback)
    if include_random_hex:
        random_hex = uuid.uuid4().hex[:3]
        parts.append(random_hex)

    if not parts:
        msg = "No valid parts were generated for the filename."
        raise ValueError(msg)

    # Construct filename components
    suffix = ".parquet" if isinstance(data, pd.DataFrame | gpd.GeoDataFrame) else ".tif"
    name_part = "_".join(parts) + suffix
    components = [comp for comp in [filename_prefix, name_part] if comp]
    filename = "_".join(components)

    return f"{prefix}/{filename}" if prefix else filename


def name_bounds_with_hash(
    bounds: tuple, crs: Any, precision: int = 6, length: int = 6
) -> str:
    """
    Generate a deterministic hash-based name for bounding box coordinates.

    Args:
        bounds (tuple): Bounding box as (minx, miny, maxx, maxy).
        crs (str): Coordinate reference system of the bounds.
        precision (int): Decimal precision for the bounds.

    Returns:
        str: Formatted name, e.g., "n45e123-abc".
    """
    bounds_geometry = box(*bounds)
    if crs != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        bounds_geometry = transform(transformer.transform, bounds_geometry)

    minx, miny, maxx, maxy = bounds_geometry.bounds

    # Format bounds with specified precision
    formatted_bounds = f"{minx:.{precision}f}_{miny:.{precision}f}_{maxx:.{precision}f}_{maxy:.{precision}f}"

    # Generate a deterministic short hash
    hash_seed = formatted_bounds
    hash_suffix = short_id(hash_seed, length=length)

    lon_prefix = "e" if minx >= 0 else "w"
    lat_prefix = "n" if miny >= 0 else "s"

    return f"{lat_prefix}{abs(miny):02.0f}{lon_prefix}{abs(minx):03.0f}-{hash_suffix}"


def short_id(seed: str, length: int = 6) -> str:
    """Generate a short deterministic ID based on a seed."""
    return hashlib.md5(seed.encode()).hexdigest()[:length]


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


if __name__ == "__main__":
    az_href = "az://my-container/my-folder/my-file.tif"
    az_https = (
        "https://my-account.blob.core.windows.net/my-container/my-folder/my-file.tif"
    )

    pp = PathParser(az_href, account_name="coclico")

    print("done")
