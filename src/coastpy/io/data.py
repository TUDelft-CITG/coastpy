import geopandas as gpd
import pystac

from coastpy.stac.utils import read_snapshot


def fetch_transects(region_of_interest: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    import ibis
    from ibis import _

    def to_url(uri):
        protocol, value = uri.split("://")
        return f"{protocol}://coclico.blob.core.windows.net/{value}"

    region_of_interest = region_of_interest.to_crs(4326)

    snapshot = read_snapshot(
        pystac.Catalog.from_file(
            "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
        ).get_child("gcts"),
        columns=["geometry"],
    )

    gcts_hrefs = list(gpd.sjoin(snapshot, region_of_interest).href.unique())

    urls = [to_url(uri) for uri in gcts_hrefs]

    west, south, east, north = list(region_of_interest.total_bounds)

    con = ibis.duckdb.connect(extensions=["spatial"])

    t = con.read_parquet(urls, table_name="gcts")
    expr = t.filter(
        _.bbox.xmin > west,
        _.bbox.ymin > south,
        _.bbox.xmax < east,
        _.bbox.ymax < north,
    )
    gcts = expr.to_pandas().set_crs(4326, allow_override=True)
    gcts = gpd.sjoin(gcts, region_of_interest).drop(columns="index_right")
    return gcts
