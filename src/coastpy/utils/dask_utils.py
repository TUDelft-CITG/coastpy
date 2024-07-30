from typing import Any

import dask

from coastpy.utils.config import ComputeInstance, configure_instance


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
        self.instance_type = configure_instance()
        dask.config.refresh()

    def create_client(
        self, instance_type: ComputeInstance | None, *args: Any, **kwargs: Any
    ):
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
        if instance_type is None:
            instance_type = self.instance_type

        if instance_type.name == "LOCAL":
            return self._create_local_client(*args, **kwargs)
        elif instance_type.name == "SLURM":
            return self._create_slurm_client(*args, **kwargs)
        else:
            msg = "Unknown compute instance type."
            raise ValueError(msg)

    def _create_local_client(self, *args: Any, **kwargs: Any):
        """Create a local Dask client with potential overrides.

        Args:
            *args: Additional positional arguments for client creation.
            **kwargs: Additional keyword arguments for client creation.

        Returns:
            Client: The Dask local client.
        """
        from distributed import Client

        return Client(*args, **kwargs)

    def _create_slurm_client(self, *args: Any, **kwargs: Any):
        """Create a SLURM Dask client with potential overrides.

        Args:
            *args: Additional positional arguments for client creation.
            **kwargs: Additional keyword arguments for client creation.

        Returns:
            Client: The Dask SLURM client.
        """
        from dask_jobqueue import SLURMCluster

        min_jobs = kwargs.pop(
            "minimum_jobs", dask.config.get("jobqueue.adaptive.minimum", 1)
        )
        max_jobs = kwargs.pop(
            "maximum_jobs", dask.config.get("jobqueue.adaptive.maximum", 30)
        )

        cluster = SLURMCluster(*args, **kwargs)
        cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
        return cluster.get_client()


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
