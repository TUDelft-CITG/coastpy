from typing import Any

import dask
import geopandas as gpd
import pandas as pd
from distributed import Client
from pandas.api.types import CategoricalDtype

from coastpy.utils.config import ComputeInstance


class DaskClientManager:
    """Manager for creating Dask clients based on compute instance type.

    This class supports the creation of local and SLURM Dask clusters,
    with optional configuration from external files.

    Attributes:
        config_path (Optional[str]): Path to a Dask configuration file.
    """

    def __init__(self):
        """Initialize the DaskClientManager, optionally loading a config file.

        Args:
            config_path (Optional[str]): Path to the configuration file.
        """
        dask.config.refresh()

    def create_client(self, instance_type: ComputeInstance, *args: Any, **kwargs: Any):
        """Create a Dask client based on the instance type.

        Args:
            instance_type (ComputeInstance): The type of the compute instance.
            *args: Additional positional arguments for client creation.
            **kwargs: Additional keyword arguments for client creation.

        Returns:
            Client: The Dask client.

        Raises:
            ValueError: If the instance type is not recognized.
        """

        if instance_type.name == "LOCAL":
            return self._create_local_client(*args, **kwargs)
        elif instance_type.name == "SLURM":
            return self._create_slurm_client(*args, **kwargs)
        else:
            msg = "Unknown compute instance type."
            raise ValueError(msg)

    def _create_local_client(self, *args: Any, **kwargs: Any) -> Client:
        """Create a local Dask client with potential overrides.

        Args:
            *args: Additional positional arguments for client creation.
            **kwargs: Additional keyword arguments for client creation.

        Returns:
            Client: The Dask local client.
        """
        # Set default values
        from distributed import Client

        configs = {
            "threads_per_worker": 1,
            "processes": True,
            "n_workers": 5,
            "local_directory": "/tmp",
        }

        # Update defaults with any overrides provided in kwargs
        configs.update(kwargs)

        # Create and return the Dask Client using the updated parameters
        return Client(*args, **configs)

    def _create_slurm_client(self, *args: Any, **kwargs: Any) -> Client:
        """Create a SLURM Dask client for a 32GB RAM node."""
        from dask_jobqueue import SLURMCluster

        slurm_configs = {
            "cores": 1,  # 1 core per worker
            # "memory": "8GB",  # Memory allocated per worker
            "processes": 1,  # Single process per worker
            "walltime": "00:20:00",  # Maximum runtime per worker
            # "job_extra_directives": [
            #     "--output=/scratch/${USER}/dask_logs/%x_%j.out",  # Log file
            #     "--error=/scratch/${USER}/dask_logs/%x_%j.err",
            # ],
            # "local_directory": "/scratch/${USER}/dask_tmp",  # Worker temp storage
            # "log_directory": "/scratch/${USER}/dask_logs",  # Dask logs
        }
        slurm_configs.update(kwargs)

        # Initialize SLURM cluster
        cluster = SLURMCluster(*args, **slurm_configs)

        # Scale cluster to 4 workers
        cluster.scale(jobs=4)

        print(f"Cluster job script:\n{cluster.job_script()}")

        return Client(cluster)


def silence_shapely_warnings() -> None:
    """Suppress specific warnings commonly encountered in Shapely geometry operations."""
    import warnings

    warnings_to_ignore: list[str] = [
        "invalid value encountered in buffer",
        "invalid value encountered in intersection",
        "invalid value encountered in unary_union",
    ]

    for warning in warnings_to_ignore:
        warnings.filterwarnings("ignore", message=warning)


def silence_numpy_warnings() -> None:
    """Suppress specific warnings commonly encountered in NumPy operations."""
    import warnings

    warnings_to_ignore: list[str] = [
        "All-NaN slice encountered",
    ]

    for warning in warnings_to_ignore:
        warnings.filterwarnings("ignore", message=warning)


def make_meta_from_dtypes(
    dtypes: dict, geometry_column: str | None = None
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Create an empty (Geo)DataFrame with specified data types.

    Args:
        dtypes (dict): Dictionary where keys are column names and values are data types.
            Supported types include:
            - Standard pandas types (e.g., "int", "float32", "datetime64[ns]")
            - Geometry type (for GeoDataFrames, use 'geometry')
            - CategoricalDtype (for categorical columns).
        geometry_column (str, optional): Name of the geometry column. If None, it is inferred.
            Raises an error if multiple geometry columns exist and none is specified.

    Returns:
        pd.DataFrame | gpd.GeoDataFrame: An empty DataFrame or GeoDataFrame with the specified schema.
            Returns a GeoDataFrame if a "geometry" column is present in `dtypes`.

    Raises:
        ValueError: If unsupported data types are provided or if multiple geometry columns exist without specification.
    """
    data = {}
    detected_geometry_columns = []

    # Iterate through dtypes and initialize empty columns
    for column, dtype in dtypes.items():
        if dtype == "geometry":
            data[column] = gpd.GeoSeries(
                [], dtype="geometry"
            )  # Use GeoSeries for geometries
            detected_geometry_columns.append(column)
        elif dtype == "datetime64[ns]":
            data[column] = pd.Series([], dtype="datetime64[ns]")
        elif dtype == "string":
            data[column] = pd.Series([], dtype="string")
        elif isinstance(dtype, type | str | CategoricalDtype):
            data[column] = pd.Series([], dtype=dtype)
        else:
            raise ValueError(f"Unsupported dtype for column '{column}': {dtype}")

    # Handle geometry column selection
    if geometry_column is None:
        if len(detected_geometry_columns) > 1:
            raise ValueError(
                f"Multiple geometry columns detected: {detected_geometry_columns}. "
                f"Please specify the geometry column using the `geometry_column` argument."
            )
        elif len(detected_geometry_columns) == 1:
            geometry_column = detected_geometry_columns[0]

    # Create the base DataFrame
    df = pd.DataFrame(data)

    # Convert to GeoDataFrame if a geometry column is present
    if geometry_column:
        if geometry_column not in dtypes:
            raise ValueError(
                f"Specified geometry column '{geometry_column}' is not in `dtypes`."
            )
        return gpd.GeoDataFrame(df, geometry=geometry_column)

    return df
