import datetime
import json
import re
import time
import uuid
from collections import defaultdict
from enum import Enum

import fsspec
import pandas as pd


class Status(Enum):
    """Enum for processing status of items."""

    PENDING = "pending"
    PROCESSING = "processing"
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
        pattern: str,
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
                self.df.at[tile_id, "message"] = "All bands processed."

            else:
                self.df.at[tile_id, "status"] = Status.FAILED.value
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

    def update(
        self, urlpath: str, status: Status, message: str = "", dt: str | None = None
    ) -> None:
        """Update the log based on the given URL path."""
        groups = self._extract_groups(urlpath)
        tile_id = groups["tile_id"]
        if not tile_id:
            raise ValueError(f"Tile ID not found in urlpath: {urlpath}")
        band = groups.get("band")
        dt = dt or self.now_to_isoformat()

        if tile_id not in self.df.index:
            raise KeyError(f"Tile ID '{tile_id}' not found in the log.")

        # Tile-level update
        if not band:
            self.df.at[tile_id, "status"] = status.value
            self.df.at[tile_id, "message"] = message
        # Band-level update
        else:
            current_bands = self.df.at[tile_id, "bands"] or frozenset()
            if status == Status.SUCCESS:
                self.df.at[tile_id, "bands"] = current_bands | {band}
            self._update_tile_status(tile_id)

        # Update datetime
        self.df.at[tile_id, "datetime"] = dt

        # Add retry logic
        if status == Status.FAILED:
            self.df.at[tile_id, "retries"] += 1
            self.df.at[tile_id, "message"] = (
                f"{message} (retry count: {self.df.at[tile_id, 'retries']})"
            )
        elif status == Status.SUCCESS:
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
                    self.update(tile_id, Status.SUCCESS, "All bands processed.")
                else:
                    self.update(
                        tile_id,
                        Status.FAILED,
                        f"Missing bands: {', '.join(sorted(missing_bands))}",
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
        pattern: str,
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
        filtered = self.df[self.df["status"].isin([s.value for s in statuses])]

        if filtered.empty:
            raise ValueError(f"No items with statuses: {[s.value for s in statuses]}")

        return filtered.sample(n=min(n, len(filtered))).index.tolist()

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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import ast
    import json
    import os
    import re
    import time
    from collections import defaultdict

    import fsspec
    import pandas as pd

    from coastpy.utils.grid import read_coastal_grid

    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}
    VERSION = "2025-01-17"
    OUT_STORAGE = f"az://tmp/s2-l2a-composite/release/{VERSION}"

    DATETIME_RANGE = "2023-01-01/2024-01-01"
    BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
    SPECTRAL_INDICES = ["NDWI", "NDVI", "MNDWI", "NDMI"]

    ZOOM = 9
    BUFFER_SIZE = "10000m"

    def format_date_range(date_range: str) -> str:
        """Convert ISO date range to YYYYMMDD_YYYYMMDD format."""
        return "_".join(pd.to_datetime(date_range.split("/")).strftime("%Y%m%d"))

    date_range = format_date_range(DATETIME_RANGE)
    required_bands = [b for b in BANDS if b != "SCL"]

    # NOTE: this is very important to get right
    FILENAME_PATTERN = (
        rf"(?P<tile_id>\d{{2}}[A-Za-z]{{3}}_z\d+-(?:n|s)\d{{2}}(?:w|e)\d{{3}}-[a-z0-9]{{6}})"  # Tile ID
        rf"(?:_{date_range})?"  # Optional date range
        r"(?:_(?P<band>[a-z0-9]+))?"  # Optional band
        r"(?:_10m\.tif)?"  # Optional resolution
    )

    grid = read_coastal_grid(buffer_size=BUFFER_SIZE, zoom=ZOOM)

    # Apply JSON parsing for specific columns
    grid["admin:continents"] = grid["admin:continents"].apply(
        lambda x: json.loads(x) if x else []
    )
    grid["admin:countries"] = grid["admin:countries"].apply(
        lambda x: json.loads(x) if x else []
    )
    grid["s2:mgrs_tile"] = grid["s2:mgrs_tile"].apply(ast.literal_eval)
    grid = grid.explode("s2:mgrs_tile", ignore_index=True)
    grid["tile_id"] = grid[["s2:mgrs_tile", "coastal_grid:id"]].agg("_".join, axis=1)
    print(f"Shape grid: {grid.shape}")

    log_urlpath = f"{OUT_STORAGE.replace('az://', 'az://log/')}/log.parquet"

    tile_logger = TileLogger(
        log_urlpath,
        grid.tile_id.to_list(),
        FILENAME_PATTERN,
        required_bands,
        storage_options,
    )

    log_df = tile_logger.df.copy()

    storage_pattern = f"{OUT_STORAGE}/*.tif"
    df1 = tile_logger.df.copy()
    print(df1.status.value_counts())
    tile_logger.update_from_storage(storage_pattern)
    df2 = tile_logger.df.copy()
    tile_logger.write()
    print(df2.status.value_counts())
    print(f"Shape log_df: {log_df.shape}")
