import datetime
import json
import re
import uuid
from enum import Enum

import fsspec
import pandas as pd


class Status(Enum):
    """Enum for processing status of items."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class FileLogger:
    """
    FileLogger for tracking processing statuses of items.

    Attributes:
        log_path (str): Path to the log file (cloud/local).
        ids (List[str]): List of IDs to initialize the log with.
        pattern (str): Regex pattern to extract IDs from file paths.
        storage_opts (dict): Storage options for fsspec.
    """

    def __init__(
        self,
        log_path: str,
        ids: list[str],
        pattern: str,
        storage_opts: dict | None = None,
    ) -> None:
        self.log_path = log_path
        self.ids = ids
        self.pattern = pattern
        self.storage_opts = storage_opts or {}
        self.fs = fsspec.filesystem(log_path.split("://")[0], **self.storage_opts)
        self.log_df: pd.DataFrame

        if self.fs.exists(self.log_path):
            self.read()
            self._validate_ids()
        else:
            self._init_log()

    def _validate_ids(self) -> None:
        """Validate that all provided IDs exist in the log."""
        missing = set(self.ids) - set(self.log_df.index)  # type: ignore
        if missing:
            raise ValueError(f"Missing IDs in log: {missing}")

    def _init_log(self) -> None:
        """Initialize the log DataFrame with PENDING status."""
        self.log_df = pd.DataFrame(
            {
                "id": self.ids,
                "status": Status.PENDING.value,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")
        self.write()

    def _extract_id(self, urlpath: str) -> str:
        """Extract the ID from a file path.

        Args:
            urlpath (str): File path to extract ID from.

        Returns:
            str: Extracted ID.

        Raises:
            ValueError: If the ID cannot be extracted.
        """
        match = re.search(self.pattern, urlpath)
        if not match:
            raise ValueError(f"Cannot extract ID from urlpath: {urlpath}")
        return match.group(1)

    def read(self) -> None:
        """Load the log from storage."""
        try:
            with self.fs.open(self.log_path, "rb") as f:
                self.log_df = pd.read_parquet(f)
        except Exception as e:
            raise OSError(f"Failed to read log file: {e}")  # noqa: B904

    def write(self) -> None:
        """Write the log DataFrame back to storage."""
        try:
            with self.fs.open(self.log_path, "wb") as f:
                self.log_df.to_parquet(f, index=True)
        except Exception as e:
            raise OSError(f"Failed to write log file: {e}")  # noqa: B904

    def sample(self, n: int, statuses: list[Status] | None = None) -> list[str]:
        """
        Sample up to `n` IDs with specified statuses.

        Args:
            n (int): Number of IDs to sample.
            statuses (List[Status], optional): Statuses to sample from. Defaults to [PENDING].

        Returns:
            List[str]: Sampled IDs.

        Raises:
            ValueError: If no items are available with the given statuses.
        """
        statuses = statuses or [Status.PENDING]
        filtered = self.log_df[self.log_df["status"].isin([s.value for s in statuses])]

        if filtered.empty:
            raise ValueError(f"No items with statuses: {[s.value for s in statuses]}")

        return filtered.sample(n=min(n, len(filtered))).index.tolist()

    def update(self, item_id: str, status: Status, message: str = "") -> None:
        """
        Update the status and message of an item in the log.

        Args:
            item_id (str): ID of the item to update.
            status (Status): New status.
            message (str, optional): Optional message. Defaults to "".

        Raises:
            KeyError: If the ID is not in the log.
        """
        if item_id not in self.log_df.index:
            raise KeyError(f"ID '{item_id}' not found in the log.")
        self.log_df.at[item_id, "status"] = status.value
        self.log_df.at[item_id, "datetime"] = pd.Timestamp.utcnow().isoformat()
        self.log_df.at[item_id, "message"] = message

    def bulk_update(self, results: list[tuple[str, Status, str]]) -> None:
        """
        Bulk update the log with processing results.

        Args:
            results (List[Tuple[str, Status, str]]): List of (urlpath, Status, message).

        Raises:
            ValueError: If ID extraction fails.
        """
        for urlpath, status, message in results:
            try:
                item_id = self._extract_id(urlpath)
                self.update(item_id, status, message)
            except KeyError:
                print(f"Warning: ID '{item_id}' not found in the log.")


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
        "datetime": datetime.datetime.now().isoformat(),
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
