from dask_jobqueue import SLURMCluster

from coastpy.utils.config import ComputeInstance


def create_dask_client(instance_type: ComputeInstance):
    """Create a Dask client based on the instance type.

    Args:
        instance_type (ComputeInstance): The type of the compute instance.

    Returns:
        Client: The Dask client.
    """

    if instance_type.name == "LOCAL":
        from distributed import Client

        return Client(
            threads_per_worker=1,
            processes=True,
            local_directory="/tmp",
        )
    elif instance_type.name == "SLURM":
        cluster = SLURMCluster(memory="16GB")
        cluster.adapt(minimum_jobs=1, maximum_jobs=30)
        return cluster.get_client()
    else:
        msg = "Unknown compute instance type."
        raise ValueError(msg)


def silence_shapely_warnings() -> None:
    """
    Suppress specific warnings commonly encountered in Shapely geometry operations.

    Warnings being suppressed:
    - Invalid value encountered in buffer
    - Invalid value encountered in intersection
    - Invalid value encountered in unary_union
    """

    warning_messages = [
        "invalid value encountered in buffer",
        "invalid value encountered in intersection",
        "invalid value encountered in unary_union",
    ]
    import warnings

    for message in warning_messages:
        warnings.filterwarnings("ignore", message=message)
