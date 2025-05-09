from urllib.parse import urlparse, urlunparse

import geopandas as gpd
import ibis
import pystac
from ibis import _

from coastpy.stac.utils import list_parquet_columns_from_stac, read_snapshot


def fetch_transects(
    region_of_interest: gpd.GeoDataFrame,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Fetch GCTS transects intersecting a region, with optional column selection."""
    import geopandas as gpd
    import ibis
    from ibis import _

    def to_url(uri: str) -> str:
        protocol, value = uri.split("://")
        return f"{protocol}://coclico.blob.core.windows.net/{value}"

    region_of_interest = region_of_interest.to_crs(4326)

    # Read STAC collection
    stac_catalog_url = "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
    collection = pystac.Catalog.from_file(stac_catalog_url).get_child("gcts")
    all_columns = list_parquet_columns_from_stac(collection)  # type: ignore[no-untyped-call]

    # Validate user-specified columns
    if columns is not None:
        unknown = set(columns) - set(all_columns)
        if unknown:
            raise ValueError(f"Unknown column(s): {unknown}")
        if "geometry" not in columns:
            columns = [*columns, "geometry"]
    else:
        columns = all_columns

    # Spatial snapshot to find matching hrefs
    snapshot = read_snapshot(
        collection,
        columns=["geometry"],
    )
    gcts_hrefs = list(gpd.sjoin(snapshot, region_of_interest).href.unique())
    urls = [to_url(uri) for uri in gcts_hrefs]

    # Bounding box filter
    west, south, east, north = list(region_of_interest.total_bounds)

    con = ibis.duckdb.connect(extensions=["spatial"])
    t = con.read_parquet(urls, table_name="gcts")

    t = t.filter(
        _.bbox.xmin > west,
        _.bbox.ymin > south,
        _.bbox.xmax < east,
        _.bbox.ymax < north,
    )

    t = t.select(columns)

    gcts = t.to_pandas().set_crs(4326, allow_override=True)

    # Final precise spatial join
    gcts = gpd.sjoin(gcts, region_of_interest).drop(columns="index_right")

    return gcts


def sign_href(
    href: str, account_name: str = "coclico", sas_token: str | None = None
) -> str:
    """Return a fully qualified and signed HTTPS URL from a STAC asset HREF.

    Supports az://, abfs://, or https:// inputs.

    Args:
        href: Original STAC href.
        account_name: Azure account name.
        sas_token: Optional SAS token to append as query string.

    Returns:
        Signed https:// URL.
    """
    parsed = urlparse(href)

    # Convert az:// or abfs:// to https://
    if parsed.scheme in {"az", "abfs"}:
        # Example: az://container/path/to/file -> https://{account}.blob.core.windows.net/container/path/to/file
        new_netloc = f"{account_name}.blob.core.windows.net"
        return sign_href(
            urlunparse(("https", new_netloc, parsed.path, "", "", "")),
            account_name,
            sas_token,
        )

    if parsed.scheme != "https":
        raise ValueError(f"Unsupported scheme in HREF: {href}")

    # If already has query or is already signed, append SAS token if provided
    if sas_token:
        sep = "&" if parsed.query else "?"
        return f"{href}{sep}{sas_token}"

    return href


def fetch_data(
    region_of_interest: gpd.GeoDataFrame,
    collection_id: str,
    columns: list[str] | None = None,
    catalog_url: str = "https://coclico.blob.core.windows.net/stac/v1/catalog.json",
    storage_options: dict[str, str] | None = None,
) -> gpd.GeoDataFrame:
    """Fetch spatial data from a STAC collection intersecting a region.

    Args:
        region_of_interest: Region to spatially intersect.
        collection_id: STAC collection ID (e.g., "gcts", "overture-buildings").
        columns: Optional list of columns to select.
        catalog_url: Root STAC catalog URL.
        storage_options: Optional dict with Azure `account_name` and `sas_token`.

    Returns:
        GeoDataFrame of filtered results.
    """
    region_of_interest = region_of_interest.to_crs(4326)

    # Load collection
    catalog = pystac.Catalog.from_file(catalog_url)
    collection = catalog.get_child(collection_id)
    if collection is None:
        raise ValueError(f"Collection '{collection_id}' not found in catalog.")

    # List and validate columns
    all_columns = list_parquet_columns_from_stac(collection)  # type: ignore
    if columns is not None:
        unknown = set(columns) - set(all_columns)
        if unknown:
            raise ValueError(f"Unknown column(s): {unknown}")
        if "geometry" not in columns:
            columns = [*columns, "geometry"]
    else:
        columns = all_columns

    # Read snapshot and extract intersecting HREFs
    snapshot = read_snapshot(collection, columns=["geometry"])
    hrefs = list(gpd.sjoin(snapshot, region_of_interest).href.unique())

    # Azure config
    account_name = (
        storage_options.get("account_name", "coclico") if storage_options else "coclico"
    )
    sas_token = storage_options.get("sas_token") if storage_options else None
    urls = [
        sign_href(href, account_name=account_name, sas_token=sas_token)
        for href in hrefs
    ]

    # Bounding box filter
    west, south, east, north = region_of_interest.total_bounds

    con = ibis.duckdb.connect(extensions=["spatial"])
    t = con.read_parquet(urls, table_name=collection_id)

    t = t.filter(
        _.bbox.xmin > west,
        _.bbox.ymin > south,
        _.bbox.xmax < east,
        _.bbox.ymax < north,
    )

    t = t.select(columns)

    df = t.to_pandas().set_crs(4326, allow_override=True)

    return gpd.sjoin(df, region_of_interest).drop(columns="index_right")
