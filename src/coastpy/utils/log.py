import json
import os
import sys
import uuid
from datetime import datetime
from enum import Enum

import fsspec
import pandas as pd


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"


def get_log_urlpath_prefix(base_dir: str) -> str:
    """Generate a log file path with timestamp and full script path."""
    base_dir = base_dir.rstrip("/")
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    # Get the full path to the current script
    full_script_path = (
        os.path.abspath(sys.argv[0]).replace("/", "_").replace(":", "-").split(".py")[0]
    )
    return f"{base_dir}/{timestamp}_{full_script_path}"


def log(
    urlpath: str,
    name: str,
    status: Status,
    message: str = "",
    storage_options: dict[str, str] | None = None,
) -> None:
    """Log an entry as a JSON record to the specified location.

    Args:
        urlpath (str): Path to the log file (supports local and cloud storage).
        name (str): Name of the entry.
        status (Status): Status of the entry (SUCCESS or FAILED).
        storage_options (dict[str, str] | None): Options for accessing storage.
    """
    storage_options = storage_options or {}
    entry = {
        "id": uuid.uuid4().hex,
        "name": name,
        "status": status.value,
        "datetime": datetime.now().isoformat(),
        "message": message,
    }

    with fsspec.open(urlpath, "w", **storage_options) as f:
        json.dump(entry, f)


def read_logs(
    urlpath: str, storage_options: dict[str, str | None] | None = None
) -> pd.DataFrame:
    """Read log entries from a JSON file and return as a DataFrame.

    Args:
        urlpath (str): Path to the log file or directory (supports local and cloud storage).
        storage_options (dict[str, str | None]): Options for accessing storage.

    Returns:
        pd.DataFrame: A DataFrame containing log entries sorted by datetime.
    """
    storage_options = storage_options or {}
    protocol = urlpath.split("://")[0]
    fs = fsspec.filesystem(protocol, **storage_options)

    # Gather all JSON files
    json_files = fs.glob(f"{urlpath}/*.json")

    logs = []
    for file in json_files:
        try:
            with fs.open(file, "r", **storage_options) as f:
                log_entry = json.load(f)
                logs.append(log_entry)
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not logs:
        return pd.DataFrame(columns=["id", "name", "status", "datetime", "message"])

    # Convert to DataFrame
    df = pd.DataFrame(logs)

    # Parse 'time' column
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values(by="datetime", ascending=True)

    return df
