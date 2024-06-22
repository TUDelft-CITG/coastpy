import pathlib

import pystac
from pystac import Item
from pystac.layout import BestPracticesLayoutStrategy
from pystac.utils import JoinType, join_path_or_url, safe_urlparse


class ListItemLayout(BestPracticesLayoutStrategy):
    """
    Custom layout for CoCliCo STAC collections.

    This layout strategy modifies the default item path format. Instead of creating an item
    with the path format `/variable-mapbox-dim-value-dim-value/variable-mapbox-dim-value-dim-value.json`,
    it uses the format `variable-mapbox/variable-mapbox-dim-value-dim-value.json`.

    Attributes:
        item: The STAC item for which the href needs to be generated.
        parent_dir: The parent directory where the STAC item will reside.

    Returns:
        str: The href path for the given STAC item based on the custom layout.
    """

    def get_item_href(self, item: pystac.Item, parent_dir: str) -> str:
        """
        Generate the href path for a given STAC item based on the custom layout.

        Args:
            item (STACItem): The STAC item for which the href needs to be generated.
            parent_dir (str): The parent directory where the STAC item will reside.

        Returns:
            str: The href path for the given STAC item based on the custom layout.
        """
        parsed_parent_dir = safe_urlparse(parent_dir)
        join_type = JoinType.from_parsed_uri(parsed_parent_dir)
        items_dir = "items"
        custom_id = pathlib.Path(item.id).with_suffix(".json")
        return join_path_or_url(join_type, parent_dir, items_dir, str(custom_id))


class DynamicPrefixLayout(BestPracticesLayoutStrategy):
    """
    Custom layout for STAC collections based on dynamic prefixes from item IDs.

    This layout strategy generates item paths based on the first two segments of the
    item's ID. The directory is determined dynamically based on the item ID's structure,
    leading to a layout that looks like: `segment1-segment2/segment1-segment2-segment3-segment4.json`
    compared to a constant "items" directory.

    Attributes:
        item (pystac.Item): The STAC item for which the href needs to be generated.
        parent_dir (str): The parent directory where the STAC item will reside.

    Returns:
        str: The href path for the given STAC item based on the custom layout.
    """

    def get_item_href(self, item: pystac.Item, parent_dir: str) -> str:
        parsed_parent_dir = safe_urlparse(parent_dir)
        join_type = JoinType.from_parsed_uri(parsed_parent_dir)

        # Use the first two segments from the item's ID for the directory name
        custom_id = "-".join(item.id.split("-")[0:2])
        item_root = join_path_or_url(join_type, parent_dir, custom_id)
        return join_path_or_url(join_type, item_root, f"{item.id}.json")


class ParquetLayout(BestPracticesLayoutStrategy):
    """
    Custom layout for CoGs within CoCliCo STAC collections.

    Modifies the item path to:
        items/variable-mapbox-dim-value-dim-value.json
    Instead of the default:
        /variable-mapbox-dim-value-dim-value/variable-mapbox-dim-value-dim-value.json
    """

    def get_item_href(self, item: Item, parent_dir: str) -> str:
        """
        Determines the item href based on the custom layout for STAC items.

        Args:
            item (Item): The STAC item.
            parent_dir (str): The parent directory path.

        Returns:
            str: The constructed item href.
        """
        # Parse the parent directory URL or path
        parsed_parent_dir = safe_urlparse(parent_dir)
        join_type = JoinType.from_parsed_uri(parsed_parent_dir)

        # Use the `stac_item_id` directly from the item's ID, which is expected to be in the correct format
        stac_item_id = item.id

        # Define the directory where items will be stored relative to the parent directory
        items_dir = "items"

        # Construct the filename for the item's JSON representation
        filename = f"{stac_item_id}.json"

        # Join the parent directory, items directory, and filename to form the full item href
        return join_path_or_url(join_type, parent_dir, items_dir, filename)
