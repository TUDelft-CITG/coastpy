from collections.abc import Iterable, Mapping

import geopandas as gpd
import pandas as pd

# ---------------------
# Label encoding schema
# ---------------------
LABEL_ENCODING: dict[str, dict[str, int]] = {
    "class:shore_type": {
        "sandy_gravel_or_small_boulder_sediments": 0,
        "muddy_sediments": 1,
        "rocky_shore_platform_or_large_boulders": 2,
        "no_sediment_or_shore_platform": 3,
    },
    "class:coastal_type": {
        "cliffed_or_steep": 0,
        "moderately_sloped": 1,
        "bedrock_plain": 2,
        "sediment_plain": 3,
        "dune": 4,
        "wetland": 5,
        "coral": 6,
        "inlet": 7,
        "engineered_structures": 8,
    },
    "class:is_built_environment": {"false": 0, "true": 1},
    "class:has_defense": {"false": 0, "true": 1},
    "class:landform_type": {
        "mainland_coast": 0,
        "estuary": 1,
        "barrier_island": 2,
        "barrier": 3,
        "pocket_beach": 4,
        "spit": 5,
        "enh:bay": 6,
    },
}

# ------------------------
# Hex color definitions
# ------------------------
LABEL_COLORS_HEX: dict[str, dict[str, str]] = {
    "class:shore_type": {
        "sandy_gravel_or_small_boulder_sediments": "#FFD700",
        "muddy_sediments": "#8B5A2B",
        "rocky_shore_platform_or_large_boulders": "#A9A9A9",
        "no_sediment_or_shore_platform": "#1E90FF",
        "ice_or_tundra": "#F0F8FF",
    },
    "class:coastal_type": {
        "cliffed_or_steep": "#E24A33",
        "moderately_sloped": "#1A9850",
        "bedrock_plain": "#A9A9A9",
        "sediment_plain": "#FFD700",
        "dune": "#D45D9F",
        "wetland": "#8B5A2B",
        "coral": "#DA70D6",
        "inlet": "#1E90FF",
        "engineered_structures": "#4D4D4D",
    },
    "class:is_built_environment": {
        "false": "#1A9850",
        "true": "#4D4D4D",
    },
    "class:has_defense": {
        "false": "#1A9850",
        "true": "#4D4D4D",
    },
    "class:landform_type": {
        "mainland_coast": "#d8b365",
        "estuary": "#f6e8c3",
        "barrier_island": "#5ab4ac",
        "barrier": "#66c2a5",
        "pocket_beach": "#c7eae5",
        "spit": "#8c510a",
        "enh:bay": "#c2a5cf",
    },
}


def get_ordered_color_list(
    label: str, encoding: dict[str, dict[str, int]], colors: dict[str, dict[str, str]]
) -> list[str]:
    """Return hex colors ordered by encoded class index.

    Args:
        label: Full column name (e.g. 'class:shore_type').
        encoding: Full label → class → int encoding.
        colors: Full label → class → hex color.

    Returns:
        List of hex colors sorted by encoding index.
    """
    sorted_items = sorted(encoding[label].items(), key=lambda x: x[1])
    return [colors[label][k] for k, _ in sorted_items]


def get_encoded_color_mapping(label: str) -> dict[int, str]:
    """Return encoded integer → color map for a label.

    Args:
        label: Full column name (e.g. 'class:shore_type').

    Returns:
        Mapping from int code to hex color.
    """
    enc = LABEL_ENCODING[label]
    col = LABEL_COLORS_HEX[label]
    return {enc[k]: col[k] for k in enc}


def encode_categorical(
    gdf: gpd.GeoDataFrame,
    column: str,
    encoding: dict[str, dict[str, int]],
) -> gpd.GeoDataFrame:
    """Encode a single categorical column using the provided encoding.

    Args:
        gdf: GeoDataFrame with raw class column (e.g. 'class:shore_type').
        column: Column name to encode.
        encoding: Full column → class → integer mapping.

    Returns:
        GeoDataFrame with new '<column>_code' column containing integer values.
    """
    df = gdf.copy()
    if column not in df.columns or column not in encoding:
        return df  # No-op if column is missing or not in encoding

    mapping = encoding[column]
    code_column = f"{column}_code"

    df[code_column] = df[column].map(
        lambda x: mapping.get(str(x).lower()) if pd.notnull(x) else None
    )
    return df


def filter_label_encoding(
    base_encoding: dict[str, dict[str, int]],
    observed: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, int]]:
    """Filter and reindex class encodings based on observed class values.

    Args:
        base_encoding: Full label → class → int.
        observed: Full column name → observed class names.

    Returns:
        Filtered encoding dict with reindexed class integers.
    """
    filtered = {}
    for column, classes in observed.items():
        if column not in base_encoding:
            continue
        valid = {k: v for k, v in base_encoding[column].items() if k in classes}
        filtered[column] = {k: i for i, k in enumerate(valid)}
    return filtered


def filter_label_colors(
    base_colors: dict[str, dict[str, str]],
    observed: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, str]]:
    """Filter label colors to match observed class values.

    Args:
        base_colors: Full label → class → hex color.
        observed: Full column name → observed class names.

    Returns:
        Filtered color dict using full column names.
    """
    filtered = {}
    for column, classes in observed.items():
        if column not in base_colors:
            continue
        filtered[column] = {
            k: base_colors[column][k] for k in classes if k in base_colors[column]
        }
    return filtered


def get_encoding_and_colors_for_values(
    observed_classes: Mapping[str, Iterable[str]],
    normalize_keys: bool = True,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    """Get encoding and color map for observed class values.

    Args:
        observed_classes: Full column name → class values.
        normalize_keys: Lowercase keys in color map.

    Returns:
        Tuple of (encoding, color_map) dictionaries.
    """
    encoding = filter_label_encoding(LABEL_ENCODING, observed_classes)
    color_map = filter_label_colors(LABEL_COLORS_HEX, observed_classes)

    if normalize_keys:
        color_map = {
            col: {str(k).lower(): v for k, v in mapping.items()}
            for col, mapping in color_map.items()
        }

    return encoding, color_map


def get_encoding_and_colors_from_gdf(
    gdf: gpd.GeoDataFrame,
    normalize_keys: bool = True,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    """Extract encoding and color map directly from a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with 'class:...' columns.
        normalize_keys: Lowercase color dict keys.

    Returns:
        Filtered (encoding, color_map) dictionaries.

    Raises:
        ValueError: If none of the expected columns are found.
    """
    expected_cols = [
        "class:shore_type",
        "class:coastal_type",
        "class:is_built_environment",
        "class:has_defense",
        "class:landform_type",
    ]

    observed: dict[str, list[str]] = {}
    for col in expected_cols:
        if col in gdf.columns:
            values = gdf[col].dropna().unique().tolist()
            str_values = [str(v).lower() for v in values]
            observed[col] = str_values

    if not observed:
        raise ValueError("No expected 'class:...' columns found in GeoDataFrame.")

    return get_encoding_and_colors_for_values(observed, normalize_keys=normalize_keys)
