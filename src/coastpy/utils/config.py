import getpass
import logging
import os
import pathlib
import warnings
from enum import Enum, auto
from sysconfig import get_python_version

from dotenv import load_dotenv


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


def fetch_sas_token(sas_token: str | None = None) -> str:
    """
    Retrieves the Azure Storage SAS token from a provided argument, .env file, or user input.

    The function follows this order:
    1. Validates the provided `sas_token` argument if given.
    2. Attempts to load the `AZURE_STORAGE_SAS_TOKEN` from the `.env` file.
    3. Prompts the user for manual input if the token is still not found.
    4. Raises a ValueError if no valid token is provided.

    Args:
        sas_token (Optional[str]): Optional SAS token provided directly.

    Returns:
        str: A valid Azure Storage SAS token.

    Raises:
        ValueError: If the SAS token is not found or is invalid.
    """

    def _sanitize_token(token: str) -> str:
        """Remove surrounding quotes and whitespace from the token."""
        return token.strip().strip("'\"")

    def _validate_token(token: str) -> bool:
        """Ensure the token starts with 'sv='."""
        return token.startswith("sv=")

    # 1. Check the provided `sas_token` argument
    if sas_token:
        sas_token = _sanitize_token(sas_token)
        if _validate_token(sas_token):
            return sas_token
        else:
            warnings.warn(
                "The provided SAS token is invalid. Falling back to .env lookup or manual input.",
                stacklevel=2,
            )

    # 2. Load token from the `.env` file
    load_dotenv(override=True)
    env_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")

    if env_token:
        env_token = _sanitize_token(env_token)
        if _validate_token(env_token):
            return env_token
        else:
            warnings.warn(
                "The SAS token found in the .env file is invalid. Please verify the token.",
                stacklevel=2,
            )

    # 3. Prompt the user for secure input
    warnings.warn(
        "Azure Storage SAS token not found.\n"
        "This dataset is available upon reasonable request.\n"
        "Please contact the data providers to obtain an access token.",
        stacklevel=2,
    )

    user_input = getpass.getpass("Please enter your Azure Storage SAS token: ").strip()
    user_input = _sanitize_token(user_input)

    if user_input and _validate_token(user_input):
        return user_input

    # 4. Raise an error if no valid token is provided
    raise ValueError(
        "No valid SAS token provided. Access denied.\n"
        "Ensure your token starts with 'sv=' or contact the data providers for access."
    )
