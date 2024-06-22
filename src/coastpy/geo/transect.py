from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from pyproj import Transformer
from shapely import LineString, Point, wkt
from shapely.ops import transform

from coastpy.geo.ops import (
    calculate_point,
    extract_coordinates,
    extract_endpoints,
    get_angle,
    get_planar_bearing,
)


@dataclass
class Transect:
    """Dataclass for transects"""

    tr_name: str
    tr_origin: Point
    tr_length: int
    tr_angle: float
    utm_crs: int | str
    dst_crs: int | str

    _geometry: LineString = field(init=False, repr=False)
    _lon: float = field(init=False, repr=False)
    _lat: float = field(init=False, repr=False)
    _bearing: float = field(init=False, repr=False)

    @property
    # @lru_cache(maxsize=1)  # does not work when using dask distributed
    def geometry(self):
        utm_crs_epsg = pyproj.CRS.from_user_input(self.utm_crs).to_epsg()
        dst_crs_epsg = pyproj.CRS.from_user_input(self.dst_crs).to_epsg()

        pt1 = calculate_point(self.tr_origin, self.tr_angle + 90, 0.5 * self.tr_length)
        opposite_angle = get_angle(pt1, self.tr_origin)
        pt2 = calculate_point(pt1, opposite_angle, self.tr_length)
        self._bearing = get_planar_bearing(pt1, pt2)

        tf = Transformer.from_crs(utm_crs_epsg, dst_crs_epsg, always_xy=True)
        geometry = transform(tf.transform, LineString([pt1, pt2]))

        tf_4326 = Transformer.from_crs(utm_crs_epsg, 4326, always_xy=True)
        lon, lat = next(iter(transform(tf_4326.transform, self.tr_origin).coords))
        tr_origin = transform(tf.transform, self.tr_origin)

        self._geometry = geometry
        self._tr_origin = tr_origin
        self._lon = lon
        self._lat = lat

        return geometry

    def to_dict(self):
        return {
            "tr_name": self.tr_name,
            "geometry": self.geometry,
            "lon": self._lon,
            "lat": self._lat,
            "tr_origin": wkt.dumps(self._tr_origin),
            "bearing": self._bearing,
            "utm_crs": self.utm_crs,
            "src_crs": self.dst_crs,
        }

    def __hash__(self):
        return hash(
            (
                self.tr_name,
                self.tr_origin,
                self.tr_length,
                self.utm_crs,
                self.dst_crs,
            )
        )


def transect_geometry(
    start_datum: Point,
    tr_origin: Point,
    end_datum: Point,
    transect_length: float,
    rot_anti: np.ndarray,
    rot_clock: np.ndarray,
) -> LineString:
    """
    Generate a transect based on the input coastline segment and transect length.

    Args:
        start_datum (Point): Start point of the coastal segment.
        tr_origin (Point): Middle point of the coastal segment (transect origin).
        end_datum (Point): End point of the coastal segment.
        transect_length (float): Transect length in meters.
        rot_anti (np.ndarray): Anti-clockwise rotation matrix.
        rot_clock (np.ndarray): Clockwise rotation matrix.

    Returns:
        LineString: Transect as Shapely LineString.

    Example:
        >>> start = Point(0, 0)
        >>> mid = Point(1, 1)
        >>> end = Point(2, 2)
        >>> transect_length = 1
        >>> ROT_ANTI = np.array([[0, -1], [1, 0]])
        >>> ROT_CLOCK = np.array([[0, 1], [-1, 0]])
        >>> transect_geometry(start, mid, end, transect_length, ROT_ANTI, ROT_CLOCK)
        <shapely.geometry.linestring.LineString object at ...>
    """

    transect_length = transect_length / 2

    segment_vector = np.array(
        [[end_datum.x - start_datum.x], [end_datum.y - start_datum.y]]
    )
    perpendicular_anti = np.dot(rot_anti, segment_vector)
    perpendicular_clock = np.dot(rot_clock, segment_vector)

    # Normalize and scale the perpendicular vectors
    perpendicular_anti = (
        perpendicular_anti / np.linalg.norm(perpendicular_anti)
    ) * transect_length
    perpendicular_clock = (
        perpendicular_clock / np.linalg.norm(perpendicular_clock)
    ) * transect_length

    transect_start = (
        tr_origin.x + float(perpendicular_anti[0]),
        tr_origin.y + float(perpendicular_anti[1]),
    )
    transect_end = (
        tr_origin.x + float(perpendicular_clock[0]),
        tr_origin.y + float(perpendicular_clock[1]),
    )

    return LineString([transect_start, transect_end])


def make_transect_origins(
    line: shapely.geometry.LineString, spacing: float
) -> list[float]:
    """
    Generate transect origins along a LineString with a given spacing.

    Args:
        line: LineString representing the transect line.
        spacing: Distance between transect origins.

    Returns:
        List of transect origins as distances from the start of the LineString.
    """
    n_transects = int(line.length / spacing)
    leftover = line.length % spacing

    if leftover < spacing * 0.5:
        n_transects = n_transects - 1

    if n_transects == 0:
        return [line.length / 2]

    start = (line.length - n_transects * spacing) * 0.5
    end = start + n_transects * spacing
    points = np.arange(start, end, spacing)
    return points.tolist()


def make_transects(
    coastline: gpd.GeoSeries,
    coastline_name: str,
    src_crs: str,
    utm_crs: str,
    dst_crs: str,
    spacing: float,
    transect_length: int = 2000,
    smooth_distance: float = 1.0e-3,
) -> gpd.GeoDataFrame:
    """
    Generate transects along a coastline.

    Args:
        coastline: GeoSeries representing the coastline.
        coastline_name: Name of the coastline.
        src_crs: Source CRS of the coastline.
        utm_crs: UTM CRS for local coordinate transformation.
        dst_crs: Destination CRS for the transects.
        spacing: Distance between transects.
        idx: Index of the function call.
        transect_length: Length of the transects.
        smooth_distance: Distance used to smooth the coastline.

    Returns:
        GeoDataFrame containing transects with their attributes and geometry.
    """
    # Convert coastline to local UTM CRS
    tf = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    coastline = transform(tf.transform, coastline)

    origins = make_transect_origins(coastline, spacing)

    transects = []
    for origin in origins:
        # Derive transect angle from nearby points of transect origin
        tr_origin = coastline.interpolate(origin)
        pt1 = coastline.interpolate(origin - smooth_distance)
        pt2 = coastline.interpolate(origin + smooth_distance)

        angle = get_angle(pt1, pt2)
        tr_id = f"cl{int(coastline_name)}tr{int(origin)}"

        tr = Transect(
            tr_name=tr_id,
            tr_angle=angle,
            tr_origin=tr_origin,
            tr_length=transect_length,
            utm_crs=utm_crs,
            dst_crs=dst_crs,
        )

        transects.append(tr.to_dict())

    if not transects:
        return None

    # when pyarrow is more stable, use pyarrow dtypes instead
    # column_datatypes = {
    #     "tr_name": "string[pyarrow]",
    #     "lon": "float64[pyarrow]",
    #     "lat": "float64[pyarrow]",
    #     "tr_origin": "string[pyarrow]",
    #     "bearing": "float64[pyarrow]",
    #     "utm_crs": "int32[pyarrow]",
    #     "src_crs": "int32[pyarrow]",
    #     "coastline_name": "int32[pyarrow]",
    # }

    column_datatypes = {
        "tr_name": "string",
        "lon": "float64",
        "lat": "float64",
        "tr_origin": "string",
        "bearing": "float64",
        "utm_crs": "int32",
        "src_crs": "int32",
        "coastline_name": "int32",
    }

    # TODO: instead of dropping transects that cross date line, create MultiLinestrings?
    return (
        gpd.GeoDataFrame(transects, geometry="geometry", crs=dst_crs)
        .reset_index(drop=True)
        # .pipe(drop_transects_crossing_antimeridian)
        .assign(coastline_name=coastline_name)
        .astype(column_datatypes)
    )


def generate_transects_from_coastline(
    coastline: LineString,
    transect_length: float,
    spacing: float | int,
    coastline_name: int,
    coastline_is_closed: bool,
    coastline_length: int,
    src_crs: str | int,
    utm_crs: str | int,
    dst_crs: str | int,
    smooth_distance: float = 1e-3,
) -> gpd.GeoDataFrame:
    """
    Generate a list of transects from a coastline geometry.

    Args:
        coastline (LineString): The coastline geometry.
        transect_length (float): Length of the transects.
        coastline_name (int): ID for the coastline.
        coastline_is_closed (bool): If the source OSM coastline is closed.
        coastline_length (int): length of the coastline.
        src_crs (str): Source CRS of the coastline geometry.
        utm_crs (str): UTM CRS for local coordinate transformation.
        dst_crs (str): Target CRS for the transects.
        smooth_distance (float, optional): Smoothing distance. Defaults to 1e-3.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with transects and related attributes.
    """
    # Define a template empty geodataframe with specified datatypes
    META = gpd.GeoDataFrame(
        {
            "tr_name": pd.Series([], dtype="string"),
            "lon": pd.Series([], dtype="float32"),
            "lat": pd.Series([], dtype="float32"),
            "bearing": pd.Series([], dtype="float32"),
            "utm_crs": pd.Series([], dtype="int32"),
            # NOTE: leave here because before we used to store the coastline name
            # "coastline_name": pd.Series([], dtype="string"),
            "coastline_is_closed": pd.Series([], dtype="bool"),
            "coastline_length": pd.Series([], dtype="int32"),
            "geometry": gpd.GeoSeries([], dtype="geometry"),
        },
        crs=dst_crs,
    )

    ROT_ANTI = np.array([[0, -1], [1, 0]])
    ROT_CLOCK = np.array([[0, 1], [-1, 0]])

    dtypes = META.dtypes.to_dict()
    column_order = META.columns.to_list()

    tf = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    coastline = transform(tf.transform, coastline)

    origins = make_transect_origins(coastline, spacing)
    tr_origins = [coastline.interpolate(o) for o in origins]

    if not tr_origins:
        return META

    start_datums = [coastline.interpolate(o - smooth_distance) for o in origins]
    end_datums = [coastline.interpolate(o + smooth_distance) for o in origins]

    transects = [
        transect_geometry(start, origin, end, transect_length, ROT_ANTI, ROT_CLOCK)
        for start, origin, end in zip(start_datums, tr_origins, end_datums, strict=True)
    ]

    endpoints = [extract_endpoints(tr) for tr in transects]
    pt1s, pt2s = zip(*endpoints, strict=False)
    bearings = [
        get_planar_bearing(pt1, pt2) for pt1, pt2 in zip(pt1s, pt2s, strict=True)
    ]

    tf_4326 = Transformer.from_crs(utm_crs, 4326, always_xy=True)
    tr_origins_4326 = [
        transform(tf_4326.transform, tr_origin) for tr_origin in tr_origins
    ]

    lons, lats = zip(*[extract_coordinates(p) for p in tr_origins_4326], strict=True)

    tr_names = [f"{coastline_name}tr{int(o)}" for o in origins]

    return (
        gpd.GeoDataFrame(
            {
                "tr_name": tr_names,
                "lon": lons,
                "lat": lats,
                "bearing": bearings,
                "geometry": transects,
            },
            crs=utm_crs,
        )
        .to_crs(dst_crs)
        # NOTE: leave here because before we used to store the coastline name
        # .assign(coastline_name=coastline_name)
        .assign(utm_crs=utm_crs)
        .assign(coastline_is_closed=coastline_is_closed)
        .assign(coastline_length=coastline_length)
        .loc[:, column_order]
        .astype(dtypes)
    )
