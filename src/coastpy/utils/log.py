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
    FileLogger for tracking processing statuses of items with retry tracking.

    Attributes:
        log_path (str): Path to the log file (cloud/local).
        ids (List[str] | None): List of IDs to initialize the log with. Defaults to None.
        pattern (str): Regex pattern to extract IDs from file paths. Defaults to r".*".
        storage_options (dict): Storage options for fsspec.
        add_missing (bool): Add missing IDs to the log if True. Defaults to False.
    """

    def __init__(
        self,
        log_path: str,
        ids: list[str] | None = None,
        pattern: str = r".*",
        storage_options: dict | None = None,
        add_missing: bool = False,
    ) -> None:
        self.log_path = log_path
        self.ids = ids
        self.pattern = pattern
        self.add_missing = add_missing
        self.storage_options = storage_options or {}
        self.fs = fsspec.filesystem(log_path.split("://")[0], **self.storage_options)
        self.df: pd.DataFrame

        if self.fs.exists(self.log_path):
            self.read()
            if self.ids is None:
                self.ids = self.df.index.tolist()
            else:
                self._validate_ids()
        elif self.ids:
            self._init_log()
        else:
            raise ValueError(
                "Either 'ids' must be provided, or a valid log must exist at 'log_path'."
            )

    def _validate_ids(self) -> None:
        """Validate that all provided IDs exist in the log."""
        if self.ids is None:
            raise ValueError("Cannot validate IDs: 'ids' is not initialized.")
        missing = set(self.ids) - set(self.df.index)
        if missing:
            if self.add_missing:
                self._add_missing_ids(missing)
            else:
                raise ValueError(f"Missing IDs in log: {missing}")

    def _add_missing_ids(self, missing_ids: set[str]) -> None:
        """Add missing IDs to the log with PENDING status."""
        new_entries = pd.DataFrame(
            {
                "id": list(missing_ids),
                "status": Status.PENDING.value,
                "retries": 0,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")
        self.df = pd.concat([self.df, new_entries])

    def _init_log(self) -> None:
        """Initialize the log DataFrame with PENDING status."""
        if not self.ids:
            raise ValueError("Cannot initialize log: 'ids' is required.")
        self.df = pd.DataFrame(
            {
                "id": self.ids,
                "status": Status.PENDING.value,
                "retries": 0,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")

    def _extract_id(self, urlpath: str) -> str:
        """Extract the ID from a file path."""
        match = re.search(self.pattern, urlpath)
        if not match:
            raise ValueError(f"Cannot extract ID from urlpath: {urlpath}")
        return match.group(1)

    def read(self) -> None:
        """Load the log from storage."""
        with self.fs.open(self.log_path, "rb") as f:
            self.df = pd.read_parquet(f)

    def write(self) -> None:
        """Write the log DataFrame back to storage."""
        with self.fs.open(self.log_path, "wb") as f:
            self.df.to_parquet(f, index=True)

    def reset(self, statuses: list | None = None) -> None:
        """Reset the statuses of items in the log to Status.PENDING."""
        if statuses is None:
            statuses = [Status.PROCESSING]

        for i, status in enumerate(statuses):
            if isinstance(status, Status):
                statuses[i] = status.value

        to_reset = self.df[self.df.status.isin(statuses)].index.to_list()
        for item_id in to_reset:
            self.update(item_id, Status.PENDING, "")

    def update(self, item_id: str, status: Status, message: str = "") -> None:
        """
        Update the status, message, and retry count of an item in the log.

        Args:
            item_id (str): ID of the item to update.
            status (Status): New status.
            message (str, optional): Optional message. Defaults to "".

        Raises:
            KeyError: If the ID is not in the log.
        """
        if item_id not in self.df.index:
            raise KeyError(f"ID '{item_id}' not found in the log.")

        if status == Status.FAILED:
            self.df.at[item_id, "retries"] += 1

        elif status == Status.SUCCESS:
            self.df.at[item_id, "retries"] = 0

        self.df.at[item_id, "status"] = status.value
        self.df.at[item_id, "datetime"] = pd.Timestamp.utcnow().isoformat()
        self.df.at[item_id, "message"] = message

    def bulk_update(self, results: list[tuple[str, Status, str]]) -> None:
        """
        Bulk update the log with processing results.

        Args:
            results (List[Tuple[str, Status, str]]): List of (urlpath, Status, message).
        """
        for urlpath, status, message in results:
            try:
                item_id = self._extract_id(urlpath)
                self.update(item_id, status, message)
            except KeyError:
                print(f"Warning: ID '{item_id}' not found in the log.")

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
        filtered = self.df[
            (self.df["status"].isin([s.value for s in statuses]))
            & (self.df.index.isin(self.ids))
        ]

        if filtered.empty:
            raise ValueError(f"No items with statuses: {[s.value for s in statuses]}")

        return filtered.sample(n=min(n, len(filtered))).index.tolist()


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


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}
    OUT_STORAGE = "az://tmp/s2-l2a-composite/release/2025-01-17"

    file_logger = FileLogger(
        log_path=f"{OUT_STORAGE.replace('az://', 'az://log/')}/log.parquet",
        pattern=r"(\d{2}[A-Za-z]{3}_z\d+-(?:n|s)\d{2}(?:w|e)\d{3}-[a-z0-9]{6})",
        storage_options=storage_options,
    )
    log_df = file_logger.df.copy()
    storage_pattern = OUT_STORAGE + "/*.tif"

    file_logger.reset()

    print("Done")
