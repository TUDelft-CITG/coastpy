import datetime
import logging
import os
import time
import warnings
from functools import partial

import dask
import dask_geopandas
import fsspec
import geopandas as gpd
import pandas as pd
import shapely
from dotenv import load_dotenv
from geopandas.array import GeometryDtype
from shapely.geometry import LineString, Point

from coastpy.geo.ops import crosses_antimeridian
from coastpy.geo.quadtiles_utils import add_geo_columns
from coastpy.geo.transect import generate_transects_from_coastline
from coastpy.io.partitioner import QuadKeyEqualSizePartitioner
from coastpy.utils.config import configure_instance
from coastpy.utils.dask import (
    DaskClientManager,
    silence_shapely_warnings,
)
from coastpy.utils.pandas import add_attributes_from_gdfs

load_dotenv(override=True)

sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
storage_options = {"account_name": "coclico", "credential": sas_token}

# NOTE: The generalized coastline used here cannot be made publicly available contact
# authors for access.
osm_coastline_uri = "az://coastlines-osm/release/2023-02-09/coast_3857_gen9.parquet"
utm_grid_uri = "az://grid/utm.parquet"
countries_uri = "az://public/countries.parquet"  # From overture maps 2024-07-22
regions_uri = "az://public/regions.parquet"  # From overture maps 2024-07-22

today = datetime.datetime.now().strftime("%Y-%m-%d")
OUT_BASE_URI = f"az://gcts/release/{today}"
TMP_BASE_URI = OUT_BASE_URI.replace("az://", "az://tmp/")

# TODO: make cli using argsparse
# transect configuration settings
MIN_COASTLINE_LENGTH = 5000
SMOOTH_DISTANCE = 1.0e-3
TRANSECT_LENGTH = 2000
SPACING = 100

FILENAMER = "part.{numb2er}.parquet"
COASTLINE_ID_COLUMN = "FID"  # FID (OSM) or OBJECTID (Sayre)
COLUMNS = [COASTLINE_ID_COLUMN, "geometry"]
COASTLINE_ID_RENAME = "FID"

PRC_EPSG = 3857
DST_EPSG = 4326

MAX_PARTITION_SIZE = (
    "500MB"  # compressed parquet is usually order two smaller, so multiply this
)

MIN_ZOOM_QUADKEY = 2

DTYPES = {
    "transect_id": str,
    "lon": "float32",
    "lat": "float32",
    "bearing": "float32",
    "geometry": GeometryDtype(),
    # NOTE: leave here because before we used to store the coastline name
    # "osm_coastline_id": str,
    "osm_coastline_is_closed": bool,
    "osm_coastline_length": "int32",
    "utm_epsg": "int32",
    "bbox": object,
    "quadkey": str,
    # NOTE: leave here because before we used to store the bounding quadkey
    # "bounding_quadkey": str,
    "continent": str,
    "country": str,
    "common_country_name": str,
    "common_region_name": str,
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
    warnings.filterwarnings(
        "ignore",
        r"DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated,"
        r"and in a future version of pandas the grouping columns will be excluded from the operation."
        r"Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping"
        r"action= columns after groupby to silence this warning.",
    )


def zero_pad_transect_id(transect_ids: pd.Series) -> pd.Series:
    """
    Zero-pads the numerical parts of transect names to ensure logical sorting.

    This function takes a pandas Series containing transect names with the format
    "cl{coastline_id}tr{transect_id}", extracts the numeric parts, zero-pads them based
    on the maximum length of any coastline_id and transect_id in the Series, and
    reconstructs the transect names with zero-padded ids.

    Args:
        transect_ids (pd.Series): A Series of transect names in the format "cl{coastline_id}tr{transect_id}".

    Returns:
        pd.Series: A Series of zero-padded transect names for logical sorting.
    """
    # Extract and rename IDs
    ids = transect_ids.str.extract(r"cl(\d+)s(\d+)tr(\d+)").rename(
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

    return pd.Series(zero_padded_names, index=transect_ids.index)


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
                lambda row: last_end_point.distance(Point(row.geometry.coords[0])),  # noqa: B023
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
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.info(f"Transects will be written to {OUT_BASE_URI}")

    start_time = time.time()

    instance_type = configure_instance()
    client = DaskClientManager().create_client(
        instance_type,
    )
    client.run(silence_shapely_warnings)
    logging.info(f"Client dashboard link: {client.dashboard_link}")

    with fsspec.open(utm_grid_uri, **storage_options) as f:
        utm_grid = gpd.read_parquet(f)

    utm_grid = utm_grid.dissolve("epsg").to_crs(PRC_EPSG).reset_index()

    [utm_grid_scattered] = client.scatter(
        [utm_grid.loc[:, ["geometry", "epsg", "utm_code"]]], broadcast=True
    )

    coastlines = (
        dask_geopandas.read_parquet(osm_coastline_uri, storage_options=storage_options)
        .repartition(npartitions=10)
        .persist()
        .sample(frac=0.02)
        .to_crs(PRC_EPSG)
    )

    def is_closed(geometry):
        """Check if a LineString geometry is closed."""
        return geometry.is_closed

    def wrap_is_closed(df):
        df["osm_coastline_is_closed"] = df.geometry.astype(object).apply(is_closed)
        return df

    META = gpd.GeoDataFrame(
        {
            "FID": pd.Series([], dtype="i8"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "osm_coastline_is_closed": pd.Series([], dtype="bool"),
        }
    )

    coastlines = coastlines.map_partitions(wrap_is_closed, meta=META).set_crs(
        coastlines.crs
    )

    utm_extent = gpd.GeoDataFrame(
        geometry=[
            shapely.box(
                xmin=-179.99998854,
                ymin=-80.00000006,
                xmax=179.99998854,
                ymax=84.00000009,
            )
        ],
        crs="EPSG:4326",
    ).to_crs(PRC_EPSG)

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
            "osm_coastline_is_closed": pd.Series([], dtype="bool"),
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

    import dask.dataframe as dd

    coastlines = dd.from_delayed(lazy_values, meta=META).set_crs(coastlines.crs)
    # rename fid because they are no longer unique after overlay
    coastlines = coastlines.rename(columns={"FID": "FID_osm"}).astype(
        {"FID_osm": "i4", "epsg": "i4"}
    )  # type: ignore

    # TODO: use coastpy.geo.utils add_geometry_lengths
    def add_lengths(df, utm_epsg):
        silence_shapely_warnings()
        # compute geometry length in local utm crs
        df = (
            df.to_crs(utm_epsg)
            .assign(geometry_length=lambda df: df.geometry.length)
            .to_crs(df.crs)
        )
        # compute total coastline length per FID
        coastline_lengths = (
            df.groupby("FID_osm")["geometry_length"]
            .sum()
            .rename("osm_coastline_length")
            .reset_index()
        )
        # add to dataframe
        return pd.merge(
            df.drop(columns=["geometry_length"]), coastline_lengths, on="FID_osm"
        )

    META = gpd.GeoDataFrame(
        {
            "FID_osm": pd.Series([], dtype="i4"),
            "osm_coastline_is_closed": pd.Series([], dtype="bool"),
            "epsg": pd.Series([], dtype="i4"),
            "utm_code": pd.Series([], dtype="string"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "osm_coastline_length": pd.Series([], dtype="f8"),
        }
    )

    # NOTE: check how to handle the group keys with Pandas > 2.2.2
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
        df["osm_coastline_id"] = names
        return df

    META = gpd.GeoDataFrame(
        {
            "FID_osm": pd.Series([], dtype="i4"),
            "osm_coastline_is_closed": pd.Series([], dtype="bool"),
            "epsg": pd.Series([], dtype="i4"),
            "utm_code": pd.Series([], dtype="string"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "osm_coastline_length": pd.Series([], dtype="f8"),
            "osm_coastline_id": pd.Series([], dtype="string"),
        }
    )
    coastlines = coastlines.map_partitions(add_coastline_names, meta=META).set_crs(
        coastlines.crs
    )

    # coastlines = (
    #     coastlines.assign(osm_coastline_id=1)
    #     .assign(osm_coastline_id=lambda df: df.osm_coastline_id.cumsum())
    #     .persist()
    # ).set_crs(coastlines.crs)

    # coastline_names = coastlines.osm_coastline_id.value_counts().compute()

    # drop coastlines that are too short
    coastlines = coastlines.loc[
        coastlines.osm_coastline_length > MIN_COASTLINE_LENGTH
    ].persist()

    def generate_filtered_transects(
        coastline: LineString,
        transect_length: float,
        spacing: float | int,
        osm_coastline_id: str,
        osm_coastline_is_closed: bool,
        osm_coastline_length: int,
        src_crs: int,
        utm_epsg: int,
        dst_crs: int,
        smooth_distance: float = 1e-3,
    ) -> gpd.GeoDataFrame:
        transects = generate_transects_from_coastline(
            coastline,
            transect_length,
            spacing,
            osm_coastline_id,
            osm_coastline_is_closed,
            osm_coastline_length,
            src_crs,
            utm_epsg,
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
        #     osm_coastline_id,
        #     src_crs,
        #     utm_epsg,
        #     dst_crs,
        #     crosses=crosses,
        #     utm_grid=utm_grid_scattered.result().set_index("epsg"),
        # )
        # transects.loc[crosses, "geometry"] = tr_corrected

        return transects

    # Order of columns in the coastlines dataframe
    # ['FID_osm', 'epsg', 'utm_code', 'geometry', 'osm_coastline_length','osm_coastline_id']
    # create a partial function with arguments that do not change
    partial_generate_filtered_transects = partial(
        generate_filtered_transects,
        transect_length=TRANSECT_LENGTH,
        spacing=SPACING,
        src_crs=coastlines.crs.to_epsg(),
        dst_crs=DST_EPSG,
        smooth_distance=SMOOTH_DISTANCE,
    )

    # Repartition and persist the coastlines
    bag = coastlines.repartition(npartitions=100).to_bag().persist()
    # Using lambda to pick items from the bag and map them to the right parameters
    transects = bag.map(
        lambda b: partial_generate_filtered_transects(
            coastline=b[4],
            osm_coastline_id=b[6],
            osm_coastline_is_closed=b[1],
            osm_coastline_length=int(b[5]),
            utm_epsg=b[2],
        )
    )

    transects = pd.concat(transects.compute())
    transects = add_geo_columns(
        transects, geo_columns=["bbox", "quadkey"], quadkey_zoom_level=12
    )

    transects["transect_id"] = zero_pad_transect_id(transects["transect_id"])

    partitioner = QuadKeyEqualSizePartitioner(
        transects,
        out_dir=TMP_BASE_URI,
        max_size="1GB",
        min_quadkey_zoom=4,
        sort_by="quadkey",
        geo_columns=["bbox", "quadkey"],
        storage_options=storage_options,
    )
    partitioner.process()

    # with fsspec.open(TMP_BASE_URI, "wb", **storage_options) as f:
    #     transects.to_parquet(f, index=False)

    logging.info(f"Transects written to {TMP_BASE_URI}")

    transects = dask_geopandas.read_parquet(
        TMP_BASE_URI, storage_options=storage_options
    )
    # zoom = 5
    # quadkey_grouper = f"quadkey_{zoom}"
    # transects[quadkey_grouper] = transects.quadkey.str[:zoom]

    def process(transects_group):
        with fsspec.open(countries_uri, **storage_options) as f:
            countries = gpd.read_parquet(
                f, columns=["country", "common_country_name", "continent", "geometry"]
            )
        with fsspec.open(regions_uri, **storage_options) as f:
            regions = gpd.read_parquet(f, columns=["common_region_name", "geometry"])

        r = add_attributes_from_gdfs(
            transects_group, [countries, regions], max_distance=20000
        )
        return r

    logging.info("Part 2: adding attributes to transects...")
    # logging.info(f"Grouping the transects by quadkey zoom level {zoom}")

    tasks = []
    for group in transects.to_delayed():
        t = dask.delayed(process)(group, countries_uri, regions_uri, max_distance=20000)
        tasks.append(t)

    logging.info("Computing the submitted tasks..")
    transects = pd.concat(dask.compute(*tasks))
    # transects = transects.drop(columns=[quadkey_grouper])

    logging.info(
        f"Partitioning into equal partitions by quadkey at zoom level {MIN_ZOOM_QUADKEY}"
    )
    partitioner = QuadKeyEqualSizePartitioner(
        transects,
        out_dir=OUT_BASE_URI,
        max_size=MAX_PARTITION_SIZE,
        min_quadkey_zoom=MIN_ZOOM_QUADKEY,
        sort_by="quadkey",
        geo_columns=["bbox", "quadkey"],
        column_order=list(DTYPES.keys()),
        dtypes=DTYPES,
        storage_options=storage_options,
        naming_function_kwargs={"include_random_hex": True},
    )
    partitioner.process()

    logging.info("Closing client.")
    client.close()

    logging.info("Done!")
    elapsed_time = time.time() - start_time
    logging.info(
        f"Time (H:M:S): {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
    )
