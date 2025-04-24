import geopandas as gpd
import pystac

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
