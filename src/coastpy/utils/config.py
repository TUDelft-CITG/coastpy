import logging
import os
import pathlib
from enum import Enum, auto
from sysconfig import get_python_version


class ComputeInstance(Enum):
    LOCAL = auto()
    SLURM = auto()
    COILED = auto()


def is_interactive() -> bool:
    """
    Check if the code is running in a Jupyter Notebook environment.
    """
    try:
        shell = get_python_version().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter Notebook or JupyterLab
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal or IPython cognsole
        else:
            return False  # Other interactive shells
    except NameError:
        return False  # Not in an interactive shell


def get_proj_dir(proj_name) -> pathlib.Path:
    """
    Get the project directory path.

    Returns:
        A `pathlib.Path` object representing the project directory path.
    """

    def resolve_proj_path(cwd: pathlib.Path, proj_name: str) -> pathlib.Path:
        root = cwd.root
        proj_dir = cwd
        while proj_dir.name != proj_name:
            proj_dir = proj_dir.parent
            if str(proj_dir) == root:
                msg = "Reached root depth - cannot resolve project path."
                raise ValueError(msg)

        return proj_dir

    cwd = pathlib.Path().resolve() if is_interactive() else pathlib.Path(__file__)

    proj_dir = resolve_proj_path(cwd, proj_name=proj_name)
    return proj_dir


def detect_instance_type() -> ComputeInstance:
    """Detect if running on a local machine, SLURM cluster, or Coiled."""
    if "COILED_RUNTIME" in os.environ:
        return ComputeInstance.COILED
    elif (
        "SLURM_JOB_ID" in os.environ
        or "SLURM_NODELIST" in os.environ
        or "SLURM_JOB_NAME" in os.environ
    ):
        return ComputeInstance.SLURM
    else:
        return ComputeInstance.LOCAL


def configure_instance() -> ComputeInstance:
    """Configure the instance based on its type.

    This function detects whether the code is running on a local machine or a SLURM cluster.
    It logs the type of instance and returns the detected instance type. Use this function
    to add logic based on the compute environment.

    Returns:
        ComputeInstance: The type of the compute instance.

    Raises:
        ValueError: If the compute instance type is unknown.
    """
    instance = detect_instance_type()
    if instance == ComputeInstance.SLURM:
        logging.info("Running on a SLURM cluster.")
        return instance
    elif instance == ComputeInstance.LOCAL:
        logging.info("Running on a local machine.")
        return instance
    else:
        msg = "Unknown compute instance type."
        raise ValueError(msg)
