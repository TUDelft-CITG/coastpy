#!/usr/bin/env python
import logging
import warnings

import dask
import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd
import pandas as pd
import shapely
from distributed import Client
from geopandas.array import GeometryDtype
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from coastpy.utils.dask import silence_shapely_warnings

BUFFER_SIZE = 15000
TOLERANCE = 100
MIN_COASTLINE_LENGTH = 0

RELEASE = "2024-12-08"

OSM_COASTLINE_URI = "az://coastlines-osm/release/2023-02-09/coast_3857_gen9.parquet"
OSM_COASTLINE_SAMPLE_URI = (
    "az://public/coastlines-osm-sample/release/2023-02-09/coast_3857_gen9.parquet"
)
UTM_GRID_URI = "az://public/utm_grid.parquet"

PRC_EPSG = 3857

STORAGE_URLPATH = (
    f"az://coastal-zone/release/{RELEASE}/coastal_zone_{BUFFER_SIZE}m.parquet"
)


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
        df.groupby("FID")["geometry_length"].sum().rename("FID_length").reset_index()
    )
    # add to dataframe
    df = pd.merge(df.drop(columns=["geometry_length"]), coastline_lengths, on="FID")
    return df


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


if __name__ == "__main__":
    import dotenv
    import fsspec

    dotenv.load_dotenv()
    storage_options = {
        "account_name": "coclico",
        "sas_token": "AZURE_STORAGE_SAS_TOKEN",
    }

    client = Client()
    logging.info(client.dashboard_link)

    silence_warnings()

    with fsspec.open(UTM_GRID_URI, **storage_options) as f:
        utm_grid = gpd.read_parquet(f)

    utm_grid = utm_grid.dissolve("epsg").to_crs(PRC_EPSG).reset_index()

    [utm_grid_scattered] = client.scatter(
        [utm_grid.loc[:, ["geometry", "epsg", "utm_code"]]], broadcast=True
    )

    try:
        # NOTE: the source coastline cannot be made public. So here we attempt to
        # read the full dataset, but when the access keys are not available, we fallback
        # to a sample of the dataset.
        coastlines = (
            dask_geopandas.read_parquet(
                OSM_COASTLINE_URI, storage_options=storage_options
            )
            .repartition(npartitions=10)
            .persist()
            # .sample(frac=0.1)
            .to_crs(PRC_EPSG)
        )
        logging.info("Successfully loaded the full private dataset.")

    except Exception as e:
        # Log the exception and fallback to the sample dataset
        logging.warning(
            f"Failed to load private dataset: {e}. Defaulting to sample dataset."
        )
        coastlines = (
            dask_geopandas.read_parquet(OSM_COASTLINE_SAMPLE_URI)
            .repartition(npartitions=10)
            .persist()
            .to_crs(PRC_EPSG)
        )
        logging.info("Successfully loaded the sample dataset.")

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

    # drop coastlines that are too short
    coastlines = coastlines.loc[
        coastlines.osm_coastline_length > MIN_COASTLINE_LENGTH
    ].persist()

    def buffer_geometry(df, utm_crs, buffer_size):
        silence_shapely_warnings()
        df["buffer"] = gpd.GeoDataFrame(
            geometry=df.to_crs(utm_crs).buffer(buffer_size), crs=utm_crs
        ).to_crs(df.crs)
        return df

    def fix_self_intersecting_polygons(row):
        """
        Corrects geometries where buffering creates self-intersecting or invalid polygons.
        Ensures proper winding (CCW) for exteriors and removes invalid geometries.

        Args:
            row: A DataFrame row with a 'buffer' geometry to correct.

        Returns:
            The corrected row with a valid and properly oriented geometry.
        """
        silence_shapely_warnings()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row = row.copy()

            try:
                if row.buffer.geom_type == "Polygon":
                    # Fix winding of the exterior ring
                    if not row.geometry.within(row.buffer):
                        row.buffer = orient(Polygon(row.buffer.exterior), sign=1.0)

                elif row.buffer.geom_type == "MultiPolygon":
                    # Process each polygon in the multipolygon
                    parts = []
                    for polygon in row.buffer.geoms:
                        if not row.geometry.within(polygon):
                            polygon = orient(Polygon(polygon.exterior), sign=1.0)  # noqa
                        parts.append(polygon)
                    row.buffer = MultiPolygon(parts)

            except Exception as e:
                logging.error(f"Failed to fix self-intersecting polygon: {e}")
                return None

            # Simplify geometry to remove unnecessary complexity
            row.buffer = row.buffer.simplify(
                tolerance=TOLERANCE, preserve_topology=True
            )

            return row

    def compute_buffers(df):
        silence_shapely_warnings()
        df = df.groupby("epsg", group_keys=False).apply(
            lambda p: buffer_geometry(p, p.name, BUFFER_SIZE)
        )
        df = df.apply(fix_self_intersecting_polygons, axis=1)
        df = df.dropna(subset=["geometry"])
        df = (
            df.drop(columns="geometry")
            .rename(columns={"buffer": "geometry"})
            .set_geometry("geometry", crs=df.crs)
        )
        columns_to_drop = ["osm_coastline_id", "osm_coastline_is_closed", "utm_code"]

        df = df.drop(columns=columns_to_drop)
        df = df.rename(columns={"FID_osm": "FID", "osm_coastline_length": "FID_length"})
        return df

    META = gpd.GeoDataFrame(
        {
            "FID": pd.Series([], dtype="i4"),
            "epsg": pd.Series([], dtype="i4"),
            "FID_length": pd.Series([], dtype="f8"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
        }
    )

    buffer = (
        coastlines.map_partitions(compute_buffers, meta=META)
        .set_crs(coastlines.crs)
        .persist()
    )

    def crosses_antimeridian(geometry):
        """
        Check if a geometry crosses the antimeridian.

        Args:
            geometry (shapely.geometry.base.BaseGeometry): The input geometry.

        Returns:
            bool: True if the geometry crosses the antimeridian, False otherwise.
        """
        minx, miny, maxx, maxy = geometry.bounds
        return maxx - minx > 180

    def map_crosses_antimeridian(df):
        src_crs = df.crs
        df = df.to_crs(4326)
        df["crosses_antimeridian"] = df["geometry"].apply(crosses_antimeridian)
        df = df.to_crs(src_crs)
        return df

    def correct_antimeridian_cross(row):
        """
        Correct geometries crossing the antimeridian using the antimeridian library.

        Args:
            row (pd.Series): Row containing the geometry to correct.

        Returns:
            shapely.geometry.base.BaseGeometry: Corrected geometry.
        """
        geom = row.geometry

        try:
            # Fix geometry using antimeridian library
            import antimeridian

            if geom.geom_type == "Polygon":
                fixed = antimeridian.fix_polygon(geom, fix_winding=True)
                fixed = make_valid(fixed)
                return fixed
            elif geom.geom_type == "MultiPolygon":
                fixed = antimeridian.fix_multi_polygon(geom, fix_winding=True)
                fixed = make_valid(fixed)
                return fixed
        except Exception as e:
            logging.error(f"Failed to correct antimeridian crossing: {e}")
            return geom

    def correct_antimeridian_crosses_in_df(df):
        """
        Correct geometries that cross the antimeridian.

        Args:
            df (gpd.GeoDataFrame): Input GeoDataFrame with `crosses_antimeridian` column.
            utm_grid (gpd.GeoDataFrame): UTM grid for overlay.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with corrected geometries.
        """
        df = df.copy()
        crs = df.crs
        df = df.to_crs(4326)

        # Create a boolean mask for rows to correct
        rows_to_correct = df["crosses_antimeridian"]

        # Apply the correction only to rows where `crosses_antimeridian` is True
        df.loc[rows_to_correct, "geometry"] = df.loc[rows_to_correct].apply(
            lambda row: correct_antimeridian_cross(row), axis=1
        )
        df = df.to_crs(crs)
        return df

    META = gpd.GeoDataFrame(
        {
            "FID": pd.Series([], dtype="i4"),
            "epsg": pd.Series([], dtype="i4"),
            "FID_length": pd.Series([], dtype="f8"),
            "geometry": gpd.GeoSeries([], dtype=GeometryDtype),
            "crosses_antimeridian": pd.Series([], dtype=bool),
        }
    )

    lazy_values = []
    for partition in buffer.to_delayed():
        t = dask.delayed(map_crosses_antimeridian)(partition)
        t = dask.delayed(correct_antimeridian_crosses_in_df)(t)
        lazy_values.append(t)
    buffer = dd.from_delayed(lazy_values, meta=META).set_crs(buffer.crs)

    # NOTE: this is useful
    union = (buffer.unary_union).compute()

    # NOTE: maybe add datetime and other metadata?
    buffer = (
        gpd.GeoDataFrame(geometry=[union], crs=buffer.crs)
        .explode(index_parts=False)
        .reset_index(drop=True)
    ).set_geometry("geometry", crs=buffer.crs)

    # NOTE: There will be 1 remaining LineString exactly at the antimeridian
    buffer = buffer[
        (buffer.geom_type == "Polygon") | (buffer.geom_type == "MultiPolygon")
    ]

    logging.info(f"Writing buffer to {STORAGE_URLPATH}..")
    with fsspec.open(STORAGE_URLPATH, mode="wb", **storage_options) as f:
        buffer.to_parquet(f)
