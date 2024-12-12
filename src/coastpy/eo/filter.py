import logging

import pystac
import stac_geoparquet


def filter_and_sort_stac_items(
    items: list[pystac.Item],
    max_items: int,
    group_by: str,
    sort_by: str,
) -> list[pystac.Item]:
    """
    Filter and sort STAC items by grouping and ranking within each group.

    Args:
        items (list[pystac.Item]): List of STAC items to process.
        max_items (int): Maximum number of items to return per group.
        group_by (str): Property to group by (e.g., 's2:mgrs_tile').
        sort_by (str): Property to sort by within each group (e.g., 'eo:cloud_cover').

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

        # Group by the specified property and sort within groups
        df = df.groupby(group_by, group_keys=False).apply(
            lambda group: group.sort_values(sort_by).head(max_items)
        )

        # Reconstruct the filtered list of items from indices
        return [items[idx] for idx in df.index]

    except Exception as err:
        logging.error(f"Error filtering and sorting items: {err}")
        return []
