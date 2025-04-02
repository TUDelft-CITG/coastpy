import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import pystac
import stac_geoparquet
import stac_geoparquet.arrow

CLOUD_THRESHOLD_LOOKUP = {
    10: 10,  # low cloud -> few items
    20: 20,  # moderate cloud -> more items
    100: 30,  # high cloud -> max items
}


def get_items_per_group(cloud_mean: float, threshold_map: dict[int, int]) -> int:
    for threshold in sorted(threshold_map):
        if cloud_mean <= threshold:
            return threshold_map[threshold]
    return max(threshold_map.values())


def filter_and_sort_stac_items(
    items: list[pystac.Item],
    group_by: list[str],
    time_window: timedelta = timedelta(days=5),
    max_num_groups: int = 4,
    max_items: int = 30,
    cloud_threshold_lookup: dict[int, int] = CLOUD_THRESHOLD_LOOKUP,
) -> list[pystac.Item]:
    try:
        sort_by = "eo:cloud_cover"

        df = (
            stac_geoparquet.arrow.parse_stac_items_to_arrow(items)
            .read_all()
            .to_pandas()
        )
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Group spatially and assign synthetic groups
        spatial_keys = df[group_by].drop_duplicates().reset_index(drop=True)
        spatial_keys["synthetic_group"] = (
            np.random.choice(
                range(max_num_groups), size=len(spatial_keys), replace=True
            )
            if len(spatial_keys) > max_num_groups
            else range(len(spatial_keys))
        )
        df = df.merge(spatial_keys, on=group_by, how="left")

        # Temporal binning and least cloudy per bin per group
        def process_group(g):
            g = g.sort_values("datetime")
            g["time_bin"] = g["datetime"].dt.floor(f"{time_window.days}D")
            return (
                g.groupby("time_bin", group_keys=False)
                .apply(lambda x: x.sort_values(sort_by).head(1))
                .reset_index(drop=True)
            )

        df = df.groupby("synthetic_group", group_keys=False).apply(process_group)

        # --- Global cloud mean and per-group sampling ---
        global_mean = df[sort_by].mean()
        n_items_per_group = get_items_per_group(global_mean, cloud_threshold_lookup)

        df_sampled = df.groupby("synthetic_group", group_keys=False).apply(
            lambda g: g.sort_values(sort_by).head(n_items_per_group)
        )
        df_sampled_ids = set(df_sampled["id"])

        # Backfill if under max_items
        if len(df_sampled) < max_items:
            df_remaining = df[~df["id"].isin(df_sampled_ids)].copy()
            n_missing = max_items - len(df_sampled)

            backfill = df_remaining.sort_values(sort_by).head(n_missing)
            df_sampled = pd.concat([df_sampled, backfill], ignore_index=True)

        # Enforce final limit
        df_sampled = df_sampled.sort_values(sort_by).head(max_items)

        # Reconstruct list of STAC Items
        item_dict = {item.id: item for item in items}
        selected_items = [
            item_dict[row["id"]]
            for _, row in df_sampled.iterrows()
            if row["id"] in item_dict
        ]

        # Diagnostics
        print(f"Total selected items: {len(selected_items)}")
        print(f"Global cloud mean: {global_mean:.2f}")
        print(f"n_items_per_group (from lookup): {n_items_per_group}")

        return selected_items

    except Exception as err:
        logging.error(f"Error filtering and sorting items: {err}")
        return []
