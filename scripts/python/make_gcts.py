import logging
import pathlib
import time
import warnings
from functools import partial

import dask

# NOTE: explicitly set query-planning to False to avoid issues with dask-geopandas
dask.config.set({"dataframe.query-planning": False})
import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd
import mercantile
import pandas as pd
import pyproj
from distributed import Client
from geopandas.array import GeometryDtype
from shapely import LineString
from shapely.geometry import Point

from coastpy.geo.ops import crosses_antimeridian
from coastpy.geo.quadtiles_utils import add_geo_columns
from coastpy.geo.transect import (
    generate_transects_from_coastline,
)
from coastpy.io.partitioner import QuadKeyEqualSizePartitioner
from coastpy.io.retrieve import retrieve_rois
from coastpy.utils.dask_utils import (
    silence_shapely_warnings,
)

DATA_DIR = pathlib.Path.home() / "data"
TMP_DIR = DATA_DIR / "tmp"
SRC_DIR = DATA_DIR / "src"
PRC_DIR = DATA_DIR / "prc"
RES_DIR = DATA_DIR / "res"
LIVE_DIR = DATA_DIR / "live"

# TODO: make cli using argsparse
# transect configuration settings
MIN_COASTLINE_LENGTH = 5000
SMOOTH_DISTANCE = 1.0e-3
START_DISTANCE = 50
COASTLINE_SEGMENT_LENGTH = 1e4
TRANSECT_LENGTH = 2000

FILENAMER = "part.{number}.parquet"
COASTLINE_ID_COLUMN = "FID"  # FID (OSM) or OBJECTID (Sayre)
COLUMNS = [COASTLINE_ID_COLUMN, "geometry"]
COASTLINE_ID_RENAME = "FID"

PRC_CRS = "EPSG:3857"
DST_CRS = "EPSG:4326"

prc_epsg = pyproj.CRS.from_user_input(PRC_CRS).to_epsg()
dst_epsg = pyproj.CRS.from_user_input(DST_CRS).to_epsg()

# dataset specific settings
COASTLINES_DIR = SRC_DIR / "coastlines_osm_generalized_v2023" / "coast_3857_gen9.shp"

UTM_GRID_FP = LIVE_DIR / "tiles" / "utm.parquet"
ADMIN1_FP = LIVE_DIR / "overture" / "2024-02-15" / "admin_bounds_level_1.parquet"
ADMIN2_FP = LIVE_DIR / "overture" / "2024-02-15" / "admin_bounds_level_2.parquet"

# OUT_DIR = PRC_DIR / COASTLINES_DIR.stem.replace(
#     "coast", f"transects_{TRANSECT_LENGTH}_test"
# )
OUT_DIR = PRC_DIR / f"gcts-{TRANSECT_LENGTH}m.parquet"

# To drop transects at meridonal boundary
SPACING = 100
MAX_PARTITION_SIZE = (
    "500MB"  # compressed parquet is usually order two smaller, so multiply this
)
MIN_ZOOM_QUADKEY = 2

DTYPES = {
    "tr_name": str,
    "lon": "float32",
    "lat": "float32",
    "bearing": "float32",
    "geometry": GeometryDtype(),
    # NOTE: leave here because before we used to store the coastline name
    # "coastline_name": str,
    "coastline_is_closed": bool,
    "coastline_length": "int32",
    "utm_crs": "int32",
    "bbox": object,
    "quadkey": str,
    # NOTE: leave here because before we used to store the bounding quadkey
    # "bounding_quadkey": str,
    "isoCountryCodeAlpha2": str,
    "admin_level_1_name": str,
    "isoSubCountryCode": str,
    "admin_level_2_name": str,
}


def silence_warnings():
    """
    Silence specific warnings for a cleaner output.

    """

    # Silencing specific warnings
    warnings.filterwarnings(
        "ignore",
        message=(
            r"is_sparse is deprecated and will be removed in a future version. Check"
            r" `isinstance\(dtype, pd.SparseDtype\)` instead."
        ),
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"is_datetime64tz_dtype is deprecated and will be removed in a future"
            r" version. Check `isinstance\(dtype, pd.DatetimeTZDtype\)` instead."
        ),
    )


def zero_pad_tr_name(tr_names: pd.Series) -> pd.Series:
    """
    Zero-pads the numerical parts of transect names to ensure logical sorting.

    This function takes a pandas Series containing transect names with the format
    "cl{coastline_id}tr{transect_id}", extracts the numeric parts, zero-pads them based
    on the maximum length of any coastline_id and transect_id in the Series, and
    reconstructs the transect names with zero-padded ids.

    Args:
        tr_names (pd.Series): A Series of transect names in the format "cl{coastline_id}tr{transect_id}".

    Returns:
        pd.Series: A Series of zero-padded transect names for logical sorting.
    """
    # Extract and rename IDs
    ids = tr_names.str.extract(r"cl(\d+)s(\d+)tr(\d+)").rename(
        columns={0: "coastline_id", 1: "segment_id", 2: "transect_id"}
    )
    ids = ids.astype({"coastline_id": str, "segment_id": str, "transect_id": str})

    # Calculate max lengths for zero-padding
    max_coastline_id_len = ids["coastline_id"].str.len().max()
    max_segment_id_len = ids["segment_id"].str.len().max()
    max_transect_id_len = ids["transect_id"].str.len().max()

    # Apply zero-padding
    ids["coastline_id"] = ids["coastline_id"].str.zfill(max_coastline_id_len)
    ids["segment_id"] = ids["segment_id"].str.zfill(max_segment_id_len)
    ids["transect_id"] = ids["transect_id"].str.zfill(max_transect_id_len)

    # Reconstruct the transect names with zero-padded IDs
    zero_padded_names = (
        "cl" + ids["coastline_id"] + "s" + ids["segment_id"] + "tr" + ids["transect_id"]
    )

    return pd.Series(zero_padded_names, index=tr_names.index)


def sort_line_segments(segments, original_line):
    """
    Sorts segments of a LineString based on their order in the original LineString,
    while ensuring that the sorted segments maintain the original direction.

    Parameters:
        segments (GeoDataFrame): A GeoDataFrame containing LineString segments in arbitrary order.
        original_line (LineString): The original LineString from which the segments were derived.

    Returns:
        GeoDataFrame: A GeoDataFrame of the sorted LineString segments, retaining all original data columns.
    """
    # Reset index to merge results properly
    segments = segments.reset_index(drop=True)

    sorted_indices = []
    # Use the second point of the original linestring for direction accuracy
    direction_point = Point(original_line.coords[1])

    while not segments.empty:
        if not sorted_indices:
            # Initialize sorting with the segment closest to the second point of the original line
            segments["dist_to_dir_point"] = segments.apply(
                lambda row: Point(row.geometry.coords[1]).distance(direction_point),
                axis=1,
            )
            closest_idx = segments["dist_to_dir_point"].idxmin()
        else:
            # Continue sorting based on the proximity to the last segment's end point
            last_end_point = Point(sorted_indices[-1].geometry.coords[-1])
            segments["dist_to_last_end"] = segments.apply(
                lambda row: last_end_point.distance(Point(row.geometry.coords[0])),
                axis=1,
            )
            closest_idx = segments["dist_to_last_end"].idxmin()

        sorted_indices.append(segments.loc[closest_idx])
        segments = segments.drop(closest_idx)

    sorted_segments = pd.DataFrame(sorted_indices).drop(
        columns=["dist_to_last_end", "dist_to_dir_point"]
    )
    return sorted_segments


if __name__ == "__main__":
    silence_warnings()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Transects will be written to {OUT_DIR}")

    if not OUT_DIR.exists():
        OUT_DIR.mkdir(exist_ok=True, parents=True)

    start_time = time.time()

    client = Client(
        threads_per_worker=1, processes=True, local_directory="/tmp", n_workers=8
    )
    client.run(silence_shapely_warnings)
    logging.info(f"Client dashboard link: {client.dashboard_link}")

    utm_grid = (
        gpd.read_parquet(UTM_GRID_FP).dissolve("epsg").to_crs(prc_epsg).reset_index()
    )
    [utm_grid_scattered] = client.scatter(
        [utm_grid.loc[:, ["geometry", "epsg", "utm_code"]]], broadcast=True
    )

    coastlines = (
        dask_geopandas.read_file(COASTLINES_DIR, npartitions=10)
        # .sample(frac=0.01)
        .to_crs(prc_epsg)
    )

    def is_closed(geometry):
        """Check if a LineString geometry is closed."""
        return geometry.is_closed

    def wrap_is_closed(df):
        df["coastline_is_closed"] = df.geometry.astype(object).apply(is_closed)
        return df

    META = gpd.GeoDataFrame(
        {
            "FID": pd.Series([], dtype="i8"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "coastline_is_closed": pd.Series([], dtype="bool"),
        }
    )

    coastlines = coastlines.map_partitions(wrap_is_closed, meta=META).set_crs(
        coastlines.crs
    )

    # Apply the function to add the 'is_closed_island' column to the GeoDataFrame
    rois = retrieve_rois()
    utm_extent = (
        rois.loc[["GLOBAL_UTM_EXTENT"]]
        .to_crs(prc_epsg)
        .drop(columns=["processing_priority"])
    )
    [utm_extent_scattered] = client.scatter([utm_extent], broadcast=True)

    def overlay_by_grid(df, grid):
        return gpd.overlay(
            df,
            grid,
            keep_geom_type=False,
        ).explode(column="geometry", index_parts=False)

    META = gpd.GeoDataFrame(
        {
            "FID": pd.Series([], dtype="i8"),
            "coastline_is_closed": pd.Series([], dtype="bool"),
            "epsg": pd.Series([], dtype="i8"),
            "utm_code": pd.Series([], dtype=object),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
        }
    )

    lazy_values = []
    for partition in coastlines.to_delayed():
        lazy_value = dask.delayed(overlay_by_grid)(partition, utm_extent_scattered)
        lazy_value = dask.delayed(overlay_by_grid)(lazy_value, utm_grid_scattered)
        lazy_values.append(lazy_value)  # Note the change here

    coastlines = dd.from_delayed(lazy_values, meta=META).set_crs(coastlines.crs)
    # rename fid because they are no longer unique after overlay
    coastlines = coastlines.rename(columns={"FID": "FID_osm"}).astype(
        {"FID_osm": "i4", "epsg": "i4"}
    )  # type: ignore

    # TODO: use coastpy.geo.utils add_geometry_lengths
    def add_lengths(df, utm_crs):
        silence_shapely_warnings()
        # compute geometry length in local utm crs
        df = (
            df.to_crs(utm_crs)
            .assign(geometry_length=lambda df: df.geometry.length)
            .to_crs(df.crs)
        )
        # compute total coastline length per FID
        coastline_lengths = (
            df.groupby("FID_osm")["geometry_length"]
            .sum()
            .rename("coastline_length")
            .reset_index()
        )
        # add to dataframe
        return pd.merge(
            df.drop(columns=["geometry_length"]), coastline_lengths, on="FID_osm"
        )

    META = gpd.GeoDataFrame(
        {
            "FID_osm": pd.Series([], dtype="i4"),
            "coastline_is_closed": pd.Series([], dtype="bool"),
            "epsg": pd.Series([], dtype="i4"),
            "utm_code": pd.Series([], dtype="string"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "coastline_length": pd.Series([], dtype="f8"),
        }
    )

    coastlines = coastlines.map_partitions(
        lambda partition: partition.groupby("epsg", group_keys=False).apply(
            lambda gr: add_lengths(gr, gr.name)
        ),
        meta=META,
    ).set_crs(coastlines.crs)
    # the coastlines have been clipped to a utm grid, so add a new name id

    def add_coastline_names(df):
        segment_ids = df.groupby("FID_osm").cumcount()
        names = [
            f"cl{fid}s{seg}" for fid, seg in zip(df.FID_osm, segment_ids, strict=False)
        ]
        df["coastline_name"] = names
        return df

    META = gpd.GeoDataFrame(
        {
            "FID_osm": pd.Series([], dtype="i4"),
            "coastline_is_closed": pd.Series([], dtype="bool"),
            "epsg": pd.Series([], dtype="i4"),
            "utm_code": pd.Series([], dtype="string"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "coastline_length": pd.Series([], dtype="f8"),
            "coastline_name": pd.Series([], dtype="string"),
        }
    )
    coastlines = coastlines.map_partitions(add_coastline_names, meta=META).set_crs(
        coastlines.crs
    )

    # coastlines = (
    #     coastlines.assign(coastline_name=1)
    #     .assign(coastline_name=lambda df: df.coastline_name.cumsum())
    #     .persist()
    # ).set_crs(coastlines.crs)

    # coastline_names = coastlines.coastline_name.value_counts().compute()

    # drop coastlines that are too short
    coastlines = coastlines.loc[
        coastlines.coastline_length > MIN_COASTLINE_LENGTH
    ].persist()

    def generate_filtered_transects(
        coastline: LineString,
        transect_length: float,
        spacing: float | int,
        coastline_name: str,
        coastline_is_closed: bool,
        coastline_length: int,
        src_crs: int,
        utm_crs: int,
        dst_crs: int,
        smooth_distance: float = 1e-3,
    ) -> gpd.GeoDataFrame:
        transects = generate_transects_from_coastline(
            coastline,
            transect_length,
            spacing,
            coastline_name,
            coastline_is_closed,
            coastline_length,
            src_crs,
            utm_crs,
            dst_crs,
            smooth_distance,
        )

        # Drop transects that cross the antimeridian
        crosses = crosses_antimeridian(transects)

        transects = transects.loc[~crosses].copy()

        # TODO: adding transects like this works only for the eastern side of the
        # transect in the utm projection. The other segment will still be corrupt. So
        # when we would generate the transects it results in transects shorter than 2km.
        # tr_corrected = generate_transects_from_coastline_with_antimeridian_correction(
        #     coastline,
        #     transect_length,
        #     coastline_name,
        #     src_crs,
        #     utm_crs,
        #     dst_crs,
        #     crosses=crosses,
        #     utm_grid=utm_grid_scattered.result().set_index("epsg"),
        # )
        # transects.loc[crosses, "geometry"] = tr_corrected

        return transects

    # Order of columns in the coastlines dataframe
    # ['FID_osm', 'epsg', 'utm_code', 'geometry', 'coastline_length','coastline_name']
    # create a partial function with arguments that do not change
    partial_generate_filtered_transects = partial(
        generate_filtered_transects,
        transect_length=TRANSECT_LENGTH,
        spacing=SPACING,
        src_crs=coastlines.crs.to_epsg(),
        dst_crs=dst_epsg,
        smooth_distance=SMOOTH_DISTANCE,
    )

    # Repartition and persist the coastlines
    bag = coastlines.repartition(npartitions=100).to_bag().persist()
    # Using lambda to pick items from the bag and map them to the right parameters
    transects = bag.map(
        lambda b: partial_generate_filtered_transects(
            coastline=b[4],
            coastline_name=b[6],
            coastline_is_closed=b[1],
            coastline_length=int(b[5]),
            utm_crs=b[2],
        )
    )

    transects = pd.concat(transects.compute())
    transects = add_geo_columns(
        transects, geo_columns=["bbox", "quadkey"], quadkey_zoom_level=12
    )

    # NOTE: in next gcts release move this out of processing and add from countries (divisions) seperately
    admin1 = (
        gpd.read_parquet(ADMIN1_FP)
        .to_crs(transects.crs)
        .drop(columns=["id"])
        .rename(columns={"primary_name": "admin_level_1_name"})
    )
    admin2 = (
        gpd.read_parquet(ADMIN2_FP)
        .to_crs(transects.crs)
        .drop(columns=["id", "isoCountryCodeAlpha2"])
        .rename(columns={"primary_name": "admin_level_2_name"})
    )

    # NOTE: zoom level 5 is hard-coded here because I believe spatial join will be faster
    quadkey_grouper = "quadkey_z5"
    transects[quadkey_grouper] = transects.apply(
        lambda r: mercantile.quadkey(mercantile.tile(r.lon, r.lat, 5)), axis=1
    )

    def add_admin_bounds(df, admin_df, max_distance=15000):
        points = gpd.GeoDataFrame(
            df[["tr_name"]], geometry=gpd.GeoSeries.from_xy(df.lon, df.lat, crs=4326)
        ).to_crs(3857)
        joined = gpd.sjoin_nearest(
            points, admin_df.to_crs(3857), max_distance=20000
        ).drop(columns=["index_right", "geometry"])

        df = pd.merge(df, joined, on="tr_name", how="left")
        return df

    transects = transects.groupby(quadkey_grouper, group_keys=False).apply(
        lambda gr: add_admin_bounds(gr, admin1),
    )
    transects = transects.groupby(quadkey_grouper, group_keys=False).apply(
        lambda gr: add_admin_bounds(gr, admin2),
    )

    transects = transects.drop(columns=[quadkey_grouper])

    transects["tr_name"] = zero_pad_tr_name(transects["tr_name"])

    partitioner = QuadKeyEqualSizePartitioner(
        transects,
        out_dir=OUT_DIR,
        max_size=MAX_PARTITION_SIZE,
        min_quadkey_zoom=MIN_ZOOM_QUADKEY,
        sort_by="quadkey",
        geo_columns=["bbox", "quadkey"],
        column_order=list(DTYPES.keys()),
        dtypes=DTYPES,
    )
    partitioner.process()

    logging.info("Done!")
    elapsed_time = time.time() - start_time
    logging.info(
        f"Time (H:M:S): {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
    )
