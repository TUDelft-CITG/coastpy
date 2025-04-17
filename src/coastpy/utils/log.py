import datetime
import json
import re
import time
import uuid
from collections import defaultdict
from enum import Enum
from re import Pattern

import fsspec
import pandas as pd


class Status(Enum):
    """Enum for processing status of items."""

    PENDING = "pending"
    PROCESSING = "processing"
    PARTLY = "partly"
    SUCCESS = "success"
    FAILED = "failed"


class TileLogger:
    """
    TileLogger for tracking tiles with optional band-level granularity, sampling, and retry handling.
    """

    def __init__(
        self,
        log_path: str,
        ids: list[str],
        pattern: str | Pattern[str],
        required_bands: list[str] | None = None,
        storage_options: dict[str, str] | None = None,
        add_missing: bool = False,
    ):
        self.log_path = log_path
        self.ids = ids
        self.pattern = re.compile(pattern)
        self.required_bands = sorted(required_bands or [])
        self.add_missing = add_missing
        self.storage_options = storage_options or {}
        self.fs = fsspec.filesystem(log_path.split("://")[0], **self.storage_options)
        self.df: pd.DataFrame

        if self.fs.exists(self.log_path):
            self.read()
            self._validate_ids()
        else:
            self._init_log()

    def _init_log(self) -> None:
        """Initialize the log DataFrame."""
        bands_col = [frozenset()] * len(self.ids) if self.required_bands else None
        self.df = pd.DataFrame(
            {
                "id": self.ids,
                "status": Status.PENDING.value,
                "bands": bands_col,
                "retries": 0,
                "priority": 0,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")

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
        bands_col = [frozenset()] * len(missing_ids) if self.required_bands else None
        new_entries = pd.DataFrame(
            {
                "id": list(missing_ids),
                "status": Status.PENDING.value,
                "bands": bands_col,
                "retries": 0,
                "priority": 0,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")
        self.df = pd.concat([self.df, new_entries])

    def _extract_groups(self, urlpath: str) -> dict[str, str | None]:
        """Extract groups from a file path using the regex pattern."""
        match = self.pattern.search(urlpath)
        if not match:
            raise ValueError(f"Cannot extract groups from urlpath: {urlpath}")
        groups = match.groupdict()
        return {k: v for k, v in groups.items() if v is not None}

    def _update_tile_status(self, tile_id: str) -> None:
        """Update tile status based on band completeness."""
        if self.required_bands:
            bands = self.df.at[tile_id, "bands"]
            missing = set(self.required_bands) - bands
            if not missing:
                self.df.at[tile_id, "status"] = Status.SUCCESS.value
                self.df.at[tile_id, "bands"] = bands
                self.df.at[tile_id, "message"] = "All bands processed."

            else:
                self.df.at[tile_id, "status"] = Status.PARTLY.value
                self.df.at[tile_id, "bands"] = bands
                self.df.at[tile_id, "message"] = (
                    f"Missing bands: {', '.join(sorted(missing))}"
                )

    def _group_tiles_by_band(self, storage_path: str) -> dict[str, set[str]]:
        """
        Group tiles by their IDs and associated bands based on files in storage.

        Args:
            storage_path (str): The path pattern to list files from storage.

        Returns:
            Dict[str, Set[str]]: Mapping of tile IDs to their associated bands.
        """
        fs = fsspec.filesystem(storage_path.split("://")[0], **self.storage_options)
        files = fs.glob(storage_path)

        if not files:
            print("No files found.")
            return {}

        tiles = defaultdict(set)
        for f in files:
            try:
                groups = self._extract_groups(f)
                tile_id = groups["tile_id"]
                band = groups.get("band")
                if tile_id and band:
                    tiles[tile_id].add(band)
                elif tile_id:
                    tiles[tile_id]  # Ensure the tile exists even if no band is found
            except ValueError as e:
                print(f"Warning: Could not parse file {f}: {e}")

        return tiles

    def assign_priority(self, priority_mapping: dict[str, int]) -> "TileLogger":
        """
        Assign priority levels to tiles in a Pandas DataFrame **in place**.

        Args:
            priority_mapping (dict[str, int]): Dictionary mapping `tile_id` to priority levels.

        Example Usage:
            logger.assign_priority({"tile_001": 10, "tile_002": 8})
        """
        if not isinstance(priority_mapping, dict):
            raise ValueError("Expected a dictionary with {tile_id: priority_level}")

        # Ensure the 'priority' column exists
        if "priority" not in self.df.columns:
            self.df["priority"] = 0  # Default priority level

        # Convert dictionary keys to a list for compatibility
        valid_ids = self.df.index.intersection(list(priority_mapping.keys()))

        # Assign priorities using .map() properly
        self.df.loc[valid_ids, "priority"] = [
            priority_mapping[tile] for tile in valid_ids
        ]
        return self

    def update(
        self,
        urlpath: str,
        status: Status,
        message: str = "",
        dt: str | None = None,
        bands: set[str] | None = None,
    ) -> None:
        """Update the log based on the given URL path."""
        groups = self._extract_groups(urlpath)
        tile_id = groups["tile_id"]
        band = groups.get("band")
        dt = dt or self.now_to_isoformat()

        if tile_id not in self.df.index:
            raise KeyError(f"Tile ID '{tile_id}' not found in the log.")

        # Load current bands (stored internally as frozenset)
        current_bands = self.df.at[tile_id, "bands"] or frozenset()

        # Patch: add band from file path if needed
        if bands is None and band is not None:
            updated_bands = current_bands | frozenset([band])
        elif bands is None:
            updated_bands = current_bands
        else:
            updated_bands = frozenset(bands)

        self.df.at[tile_id, "bands"] = updated_bands
        self._update_tile_status(tile_id)  # type: ignore

        self.df.at[tile_id, "datetime"] = dt
        if status in {Status.FAILED, Status.PARTLY}:
            self.df.at[tile_id, "retries"] += 1
            self.df.at[tile_id, "message"] = (
                f"{message} (retry count: {self.df.at[tile_id, 'retries']})"
            )
        else:
            self.df.at[tile_id, "status"] = status.value
            self.df.at[tile_id, "message"] = message
            if status == Status.SUCCESS:
                self.df.at[tile_id, "retries"] = 0

    def bulk_update(self, results: list[tuple[str, Status, str, str]]) -> None:
        """Bulk update the log with multiple results."""
        for urlpath, status, message, dt in results:
            try:
                self.update(urlpath, status, message, dt)
            except Exception as e:
                print(f"Error updating {urlpath}: {e}")

    def update_from_storage(self, storage_pattern: str) -> None:
        """
        Update the logger based on files found in storage.

        Args:
            storage_pattern (str): The pattern to list files from storage.
        """
        tiles = self._group_tiles_by_band(storage_pattern)

        for tile_id, bands in tiles.items():
            if tile_id not in self.df.index:
                print(f"Warning: Tile ID '{tile_id}' not found in the log.")
                continue

            # Check for completeness if required bands are defined
            if self.required_bands:
                missing_bands = set(self.required_bands) - bands
                if not missing_bands:
                    self.update(
                        tile_id, Status.SUCCESS, "All bands processed.", bands=bands
                    )
                else:
                    self.update(
                        tile_id,
                        Status.PARTLY,
                        f"Missing bands: {', '.join(sorted(missing_bands))}",
                        bands=bands,
                    )
            else:
                # No bands required; mark tile as success
                self.update(tile_id, Status.SUCCESS, "Tile found in storage.")

    def reset(self, statuses: list[Status] | None = None) -> None:
        """Reset statuses to PENDING for specific tiles."""
        statuses = statuses or [Status.PROCESSING]
        for status in statuses:
            to_reset = self.df[self.df["status"] == status.value].index.tolist()
            for tile_id in to_reset:
                self.update(
                    tile_id,
                    Status.PENDING,
                    "",
                    self.now_to_isoformat(),
                )

    def sample(
        self,
        n: int,
        statuses: list[Status] | None = None,
        min_priority: int | None = None,
        max_retries: int | None = None,
        sort_by_priority: bool = True,
    ) -> list[str]:
        """
        Sample up to `n` IDs with specified statuses, with optional priority sorting.

        Args:
            n (int): Number of IDs to sample.
            statuses (List[Status], optional): Statuses to sample from. Defaults to [PENDING].
            min_priority (int, optional): Minimum priority threshold. Defaults to highest available priority.
            max_retries (int, optional): Maximum allowed retries for a task. Defaults to no restriction.
            sort_by_priority (bool, optional): Whether to prioritize higher-priority items first. Defaults to True.

        Returns:
            List[str]: Sampled IDs.
        """
        statuses = statuses or [Status.PENDING]

        # Filter based on statuses and ensure only provided `ids` are considered
        filtered = self.df[
            self.df["status"].isin([s.value for s in statuses])
            & self.df.index.isin(self.ids)
        ].reindex(self.ids)

        if filtered.empty:
            return []

        # Determine the highest available priority if min_priority is not set
        if min_priority is None:
            min_priority = filtered["priority"].max()

        if max_retries is not None:
            filtered = filtered[filtered["retries"] < max_retries]

        # Apply priority filter
        filtered = filtered[filtered["priority"] >= min_priority]

        if filtered.empty:
            return []

        # If sort_by_priority is False, we shuffle the data immediately
        if not sort_by_priority:
            return filtered.sample(n=min(n, len(filtered))).index.tolist()

        # Sort by priority (descending) to process higher-priority items first
        filtered = filtered.sort_values(by="priority", ascending=False)

        # Shuffle within each priority group
        filtered = filtered.groupby("priority", group_keys=False).apply(
            lambda x: x.sample(frac=1)
        )

        sampled_ids = []

        # Iterate over unique priority levels (highest to lowest)
        for priority_level in sorted(filtered["priority"].unique(), reverse=True):
            available_samples = filtered[
                filtered["priority"] == priority_level
            ].index.tolist()

            if available_samples:
                to_sample = min(n - len(sampled_ids), len(available_samples))
                sampled_ids.extend(available_samples[:to_sample])

            # Stop once we have enough samples
            if len(sampled_ids) >= n:
                break

        return sampled_ids

    def n_to_process(
        self, statuses: list[Status] | None = None, min_priority: int | None = None
    ) -> int:
        """
        Get the number of samples to process based on the current status and priority.

        Args:
            statuses (List[Status], optional): Statuses to sample from. Defaults to [PENDING].
            min_priority (int, optional): Minimum priority threshold. Defaults to highest available priority.

        Returns:
            int: Number of tiles still pending processing.
        """
        statuses = statuses or [Status.PENDING]

        # Filter by status
        filtered = self.df[self.df["status"].isin([s.value for s in statuses])]

        if filtered.empty:
            return 0

        if min_priority is None:
            min_priority = filtered["priority"].max()

        if min_priority > filtered["priority"].max():
            return 0

        filtered = filtered[filtered["priority"] >= min_priority]

        return filtered.shape[0]

    def read(self) -> None:
        """Load the log from storage."""
        with self.fs.open(self.log_path, "rb") as f:
            self.df = pd.read_parquet(f)

        # Convert strings back to frozensets
        if "bands" in self.df.columns:
            self.df["bands"] = self.df["bands"].apply(self._string_to_frozenset)

    def write(self) -> None:
        """Write the log DataFrame back to storage."""
        df_copy = self.df.copy()
        # Convert frozensets to strings
        if "bands" in df_copy.columns:
            df_copy["bands"] = df_copy["bands"].apply(self._frozenset_to_string)

        with self.fs.open(self.log_path, "wb") as f:
            df_copy.to_parquet(f, index=True)
        time.sleep(0.5)

    @staticmethod
    def _frozenset_to_string(bands: frozenset | None) -> str:
        """Convert a frozenset to a comma-separated string."""
        return ",".join(sorted(bands)) if bands else ""

    @staticmethod
    def _string_to_frozenset(bands_str: str) -> frozenset:
        """Convert a comma-separated string to a frozenset."""
        return frozenset(bands_str.split(",")) if bands_str else frozenset()

    @staticmethod
    def now_to_isoformat():
        """Return the current datetime in ISO format."""
        return datetime.datetime.now(datetime.UTC).isoformat()


class ParquetLogger:
    """
    ParquetLogger for tracking the processing status of parquet files.
    """

    def __init__(
        self,
        log_path: str,
        pattern: str | Pattern[str],
        ids: list[str] | None = None,
        storage_options: dict[str, str] | None = None,
        add_missing: bool = False,
    ):
        self.log_path = log_path
        self.ids = ids or []
        self.pattern = re.compile(pattern)
        self.add_missing = add_missing
        self.storage_options = storage_options or {}
        self.fs = fsspec.filesystem(log_path.split("://")[0], **self.storage_options)
        self.df: pd.DataFrame

        if self.fs.exists(self.log_path):
            self.read()
            if self.ids:
                self._validate_ids()
        else:
            self._init_log()

    def _init_log(self) -> None:
        """Initialize the log DataFrame."""
        self.df = pd.DataFrame(
            {
                "id": self.ids,
                "status": Status.PENDING.value,
                "retries": 0,
                "priority": 0,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")

    def _validate_ids(self) -> None:
        """Validate that all provided IDs exist in the log."""
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
                "priority": 0,
                "datetime": pd.Timestamp.utcnow().isoformat(),
                "message": "",
            }
        ).set_index("id")
        self.df = pd.concat([self.df, new_entries])

    def _extract_tile_id(self, urlpath: str) -> str:
        """Extract tile ID from the file path using the regex pattern."""
        match = self.pattern.search(urlpath)
        if not match:
            raise ValueError(f"Cannot extract tile ID from urlpath: {urlpath}")
        return match.group(0)

    def update(
        self, urlpath: str, status: Status, message: str = "", dt: str | None = None
    ) -> None:
        """Update the log based on the given URL path."""
        tile_id = self._extract_tile_id(urlpath)
        dt = dt or self.now_to_isoformat()

        if tile_id not in self.df.index:
            if self.add_missing:
                self._add_missing_ids({tile_id})
            else:
                raise KeyError(f"Tile ID '{tile_id}' not found in the log.")

        self.df.at[tile_id, "status"] = status.value
        self.df.at[tile_id, "message"] = message
        self.df.at[tile_id, "datetime"] = dt

        if status == Status.FAILED:
            self.df.at[tile_id, "retries"] += 1
            self.df.at[tile_id, "message"] = (
                f"{message} (retry count: {self.df.at[tile_id, 'retries']})"
            )
        elif status == Status.SUCCESS:
            self.df.at[tile_id, "retries"] = 0

    def assign_priority(self, priority_mapping: dict[str, int]) -> None:
        """
        Assign priority levels to tiles in a Pandas DataFrame **in place**.

        Args:
            priority_mapping (dict[str, int]): Dictionary mapping `tile_id` to priority levels.

        Example Usage:
            logger.assign_priority({"tile_001": 10, "tile_002": 8})
        """
        if not isinstance(priority_mapping, dict):
            raise ValueError("Expected a dictionary with {tile_id: priority_level}")

        # Ensure the 'priority' column exists
        if "priority" not in self.df.columns:
            self.df["priority"] = 0  # Default priority level

        # Convert dictionary keys to a list for compatibility
        valid_ids = self.df.index.intersection(list(priority_mapping.keys()))

        # Assign priorities using .map() properly
        self.df.loc[valid_ids, "priority"] = [
            priority_mapping[tile] for tile in valid_ids
        ]

    def bulk_update(self, results: list[tuple[str, Status, str, str]]) -> None:
        """Bulk update the log with multiple results."""
        for urlpath, status, message, dt in results:
            try:
                self.update(urlpath, status, message, dt)
            except Exception as e:
                print(f"Error updating {urlpath}: {e}")

    def update_from_storage(self, storage_pattern: str) -> None:
        """
        Update the logger based on parquet files found in storage.

        Args:
            storage_pattern (str): The pattern to list parquet files from storage.
        """
        files = self.fs.glob(storage_pattern)

        if not files:
            print("No parquet files found.")
            return

        for f in files:
            try:
                tile_id = self._extract_tile_id(f)
                if tile_id in self.df.index:
                    self.update(f, Status.SUCCESS, "Parquet file processed.")
                elif self.add_missing:
                    self._add_missing_ids({tile_id})
                    self.update(f, Status.SUCCESS, "Parquet file processed.")
                else:
                    print(f"Warning: Tile ID '{tile_id}' not found in the log.")
            except ValueError as e:
                print(f"Warning: Could not parse file {f}: {e}")

    def reset(self, statuses: list[Status] | None = None) -> None:
        """Reset statuses to PENDING for specific tiles."""
        statuses = statuses or [Status.PROCESSING]
        to_reset = self.df[
            self.df["status"].isin([s.value for s in statuses])
        ].index.tolist()
        for tile_id in to_reset:
            self.update(tile_id, Status.PENDING, "", self.now_to_isoformat())

    def sample(
        self,
        n: int,
        statuses: list[Status] | None = None,
        min_priority: int | None = None,
        max_retries: int | None = None,
        sort_by_priority: bool = True,
        sort_by_grid_idx: bool = False,
    ) -> list[str]:
        """
        Sample up to `n` IDs with specified statuses, with optional sorting controls.

        Args:
            n (int): Number of IDs to sample.
            statuses (List[Status], optional): Statuses to sample from. Defaults to [PENDING].
            min_priority (int, optional): Minimum priority threshold. Defaults to highest available.
            max_retries (int, optional): Maximum allowed retries. Defaults to no restriction.
            sort_by_priority (bool, optional): Whether to sort by priority (descending). Defaults to True.
            sort_by_grid_idx (bool, optional): Whether to sample in the order of index (e.g., quadkey). Defaults to False.

        Returns:
            List[str]: Sampled IDs.
        """
        statuses = statuses or [Status.PENDING]

        # Filter based on statuses and ensure only provided `ids` are considered
        filtered = self.df[
            self.df["status"].isin([s.value for s in statuses])
            & self.df.index.isin(self.ids)
        ].reindex(self.ids)

        if filtered.empty:
            return []

        # Determine the highest available priority if min_priority is not set
        if min_priority is None:
            min_priority = filtered["priority"].max()

        if max_retries is not None:
            filtered = filtered[filtered["retries"] < max_retries]

        # Error: cannot enable both priority and grid-index sorting
        if sort_by_priority and sort_by_grid_idx:
            raise ValueError("Cannot sort by both priority and grid index.")

        # Apply priority filter
        filtered = filtered[filtered["priority"] >= min_priority]

        if filtered.empty:
            return []

        # If sort_by_priority is False AND we do not want it filtered by ids, we shuffle the data immediately
        if not sort_by_priority and not sort_by_grid_idx:
            return filtered.sample(n=min(n, len(filtered))).index.tolist()

        # Sort by grid index (e.g., quadkey-ordered index)
        if sort_by_grid_idx:
            return filtered.head(n).index.tolist()

        # Sort by priority (descending) to process higher-priority items first
        filtered = filtered.sort_values(by="priority", ascending=False)

        # Shuffle within each priority group
        filtered = filtered.groupby("priority", group_keys=False).apply(
            lambda x: x.sample(frac=1)
        )

        sampled_ids = []

        # Iterate over unique priority levels (highest to lowest)
        for priority_level in sorted(filtered["priority"].unique(), reverse=True):
            available_samples = filtered[
                filtered["priority"] == priority_level
            ].index.tolist()

            if available_samples:
                to_sample = min(n - len(sampled_ids), len(available_samples))
                sampled_ids.extend(available_samples[:to_sample])

            # Stop once we have enough samples
            if len(sampled_ids) >= n:
                break

        return sampled_ids

    def n_to_process(
        self, statuses: list[Status] | None = None, min_priority: int | None = None
    ) -> int:
        """
        Get the number of samples to process based on the provided ids, status and processing priority.

        Args:
            statuses (List[Status], optional): Statuses to sample from. Defaults to [PENDING].
            min_priority (int, optional): Minimum priority threshold. Defaults to highest available priority.

        Returns:
            int: Number of tiles still pending processing.
        """
        statuses = statuses or [Status.PENDING]

        # Filter by status
        filtered = self.df[self.df["status"].isin([s.value for s in statuses])]

        # Filter by provided ids
        filtered = filtered[filtered.index.isin(self.ids)]

        if filtered.empty:
            return 0

        if min_priority is None:
            min_priority = filtered["priority"].max()

        if min_priority > filtered["priority"].max():
            return 0

        filtered = filtered[filtered["priority"] >= min_priority]

        return filtered.shape[0]

    def read(self) -> None:
        """Load the log from storage."""
        with self.fs.open(self.log_path, "rb") as f:
            self.df = pd.read_parquet(f)

    def write(self) -> None:
        """Write the log DataFrame back to storage."""
        with self.fs.open(self.log_path, "wb") as f:
            self.df.to_parquet(f, index=True)

    @staticmethod
    def now_to_isoformat() -> str:
        """Return the current datetime in ISO format."""
        return datetime.datetime.now(datetime.UTC).isoformat()


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
