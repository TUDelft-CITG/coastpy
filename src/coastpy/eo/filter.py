import logging
from datetime import timedelta

import pandas as pd
import pystac
import stac_geoparquet
import stac_geoparquet.arrow

# NOTE: keep original function for reference
# def filter_and_sort_stac_items(
#     items: list[pystac.Item],
#     max_items: int,
#     group_by: list[str],
#     sort_by: str,
#     time_window: timedelta = timedelta(days=5),
# ) -> list[pystac.Item]:
#     """
#     Filter and sort STAC items by grouping spatially (tile and relative orbit),
#     then grouping temporally, and ranking items within each group by quality.

#     Args:
#         items (list[pystac.Item]): List of STAC items to process.
#         max_items (int): Maximum number of items to return per group.
#         group_by (list[str]): Properties to group by (e.g., ['s2:mgrs_tile', 'sat:relative_orbit']).
#         sort_by (str): Property to sort by within each group (e.g., 'eo:cloud_cover').
#         time_window (timedelta): Time window for temporal grouping. Defaults to 5 days.

#     Returns:
#         list[pystac.Item]: Filtered and sorted list of STAC items.
#     """
#     try:
#         # Convert STAC items to a DataFrame
#         df = (
#             stac_geoparquet.arrow.parse_stac_items_to_arrow(items)
#             .read_all()
#             .to_pandas()
#         )

#         # Ensure 'datetime' column is available and parse it
#         if "datetime" not in df.columns:
#             raise ValueError("STAC items must include a 'datetime' property.")
#         df["datetime"] = pd.to_datetime(df["datetime"])

#         # Group by spatial properties (e.g., 's2:mgrs_tile' and 'sat:relative_orbit')
#         def process_group(group):
#             # Temporal grouping within each spatial group
#             group = group.sort_values("datetime")
#             group["time_bin"] = (
#                 group["datetime"].dt.floor(f"{time_window.days}D").astype(str)
#             )
#             # Group by temporal bins and sort within each bin by cloud coverage
#             grouped = group.groupby("time_bin", group_keys=False).apply(
#                 lambda g: g.sort_values(sort_by).head(1)
#             )
#             return grouped

#         # Apply spatial and temporal grouping
#         spatial_group = df.groupby(group_by, group_keys=False).apply(process_group)

#         # Limit the number of items per spatial group to max_items
#         final_selection = spatial_group.groupby(group_by, group_keys=False).apply(
#             lambda g: g.sort_values(sort_by).head(max_items)
#         )

#         # Reconstruct the filtered list of items
#         items = [items[idx] for idx in final_selection.index]
#         print(len(items))
#         return items

#     except Exception as err:
#         logging.error(f"Error filtering and sorting items: {err}")
#         return []


# Define cloud cover bins and corresponding image selection strategy
CLOUD_THRESHOLD_MAPPING: dict[float, int] = {
    15: 10,
    100: 10,
}


def filter_and_sort_stac_items(
    items: list[pystac.Item],
    group_by: list[str],
    sort_by: str,
    time_window: timedelta = timedelta(days=5),
) -> list[pystac.Item]:
    """
    Filter and sort STAC items by grouping spatially (tile and relative orbit),
    then grouping temporally, and ranking items within each group by cloud cover.

    Args:
        items (list[pystac.Item]): List of STAC items to process.
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

        # Ensure required columns are available
        if "datetime" not in df.columns or sort_by not in df.columns:
            raise ValueError(
                "STAC items must include 'datetime' and cloud cover properties."
            )

        df["datetime"] = pd.to_datetime(df["datetime"])

        def process_group(group):
            """Apply temporal binning and sort within bins."""
            group = group.sort_values("datetime")
            group["time_bin"] = (
                group["datetime"].dt.floor(f"{time_window.days}D").astype(str)
            )

            # Group by time bin, then sort by cloud cover within each bin
            grouped = group.groupby("time_bin", group_keys=False).apply(
                lambda g: g.sort_values(sort_by).head(1)  # Take least cloudy per bin
            )
            return grouped

        spatiotemporal_groups = df.groupby(group_by, group_keys=False).apply(
            process_group
        )

        def compute_max_avg_cloud(group):
            """Compute the average cloud cover per group and return the max across all groups."""
            top_10 = group.sort_values(sort_by).head(10)
            return top_10[sort_by].mean()

        # Compute average cloud cover per group
        max_avg_cloud = (
            spatiotemporal_groups.groupby(group_by).apply(compute_max_avg_cloud).max()
        )
        mean_avg_cloud = (
            spatiotemporal_groups.groupby(group_by).apply(compute_max_avg_cloud).mean()
        )
        median_avg_cloud = (
            spatiotemporal_groups.groupby(group_by)
            .apply(compute_max_avg_cloud)
            .median()
        )
        print(f"Max average cloud cover: {max_avg_cloud}")
        print(f"Mean average cloud cover: {mean_avg_cloud}")
        print(f"Median average cloud cover: {median_avg_cloud}")
        print(f"Unique MGRS tiles: {spatiotemporal_groups['s2:mgrs_tile'].unique()}")
        print(
            f"Unique relative tracks: {spatiotemporal_groups['sat:relative_orbit'].unique()}"
        )
        minx, miny, maxx, maxy = stac_geoparquet.to_geodataframe(
            [i.to_dict() for i in items], dtype_backend="pyarrow"
        ).total_bounds
        print(f"minx, miny, maxx, maxy = {minx, miny, maxx, maxy}")

        for threshold, max_items in CLOUD_THRESHOLD_MAPPING.items():  # noqa
            if max_avg_cloud < threshold:
                break

        final_selection = spatiotemporal_groups.groupby(
            group_by, group_keys=False
        ).apply(lambda g: g.sort_values(sort_by).head(max_items))

        # Reconstruct the filtered list of items
        items = [items[idx] for idx in final_selection.index]
        print(
            f"Number of items: {len(items)}; groups: {len(spatiotemporal_groups.groupby(group_by).groups)}"
        )
        return items

    except Exception as err:
        logging.error(f"Error filtering and sorting items: {err}")
        return []
