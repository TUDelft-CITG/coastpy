import logging
from datetime import timedelta

import pandas as pd
import pystac
import stac_geoparquet
import stac_geoparquet.arrow


def filter_and_sort_stac_items(
    items: list[pystac.Item],
    max_items: int,
    group_by: list[str],
    sort_by: str,
    time_window: timedelta = timedelta(days=5),
) -> list[pystac.Item]:
    """
    Filter and sort STAC items by grouping spatially (tile and relative orbit),
    then grouping temporally, and ranking items within each group by quality.

    Args:
        items (list[pystac.Item]): List of STAC items to process.
        max_items (int): Maximum number of items to return per group.
        group_by (list[str]): Properties to group by (e.g., ['s2:mgrs_tile', 'sat:relative_orbit']).
        sort_by (str): Property to sort by within each group (e.g., 'eo:cloud_cover').
        time_window (timedelta): Time window for temporal grouping. Defaults to 5 days.

    Returns:
        list[pystac.Item]: Filtered and sorted list of STAC items.
    """
    try:
        # Convert STAC items to a DataFrame
        df = (
            stac_geoparquet.arrow.parse_stac_items_to_arrow(items)
            .read_all()
            .to_pandas()
        )

        # Ensure 'datetime' column is available and parse it
        if "datetime" not in df.columns:
            raise ValueError("STAC items must include a 'datetime' property.")
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Group by spatial properties (e.g., 's2:mgrs_tile' and 'sat:relative_orbit')
        def process_group(group):
            # Temporal grouping within each spatial group
            group = group.sort_values("datetime")
            group["time_bin"] = (
                group["datetime"].dt.floor(f"{time_window.days}D").astype(str)
            )
            # Group by temporal bins and sort within each bin by cloud coverage
            grouped = group.groupby("time_bin", group_keys=False).apply(
                lambda g: g.sort_values(sort_by).head(1)
            )
            return grouped

        # Apply spatial and temporal grouping
        spatial_group = df.groupby(group_by, group_keys=False).apply(process_group)

        # Limit the number of items per spatial group to max_items
        final_selection = spatial_group.groupby(group_by, group_keys=False).apply(
            lambda g: g.sort_values(sort_by).head(max_items)
        )

        # Reconstruct the filtered list of items
        return [items[idx] for idx in final_selection.index]

    except Exception as err:
        logging.error(f"Error filtering and sorting items: {err}")
        return []
