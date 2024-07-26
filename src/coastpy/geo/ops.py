import logging
import math
import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.ops
from pyproj import Transformer, transform
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.ops import snap, split

from coastpy.utils.dask_utils import silence_shapely_warnings


def shift_point(
    c1: tuple[float, float], c2: tuple[float, float], offset: float
) -> Point:
    """
    Shift a point (`c1`) towards another point (`c2`) by a specified offset distance.
    The new point will lie on the line connecting `c1` and `c2`.

    Args:
        c1 (Tuple[float, float]): The initial point, given as a tuple (x1, y1).
        c2 (Tuple[float, float]): The target point towards which `c1` will be shifted, given as a tuple (x2, y2).
        offset (float): Distance to shift `c1` towards `c2`. If the offset is greater than the distance between
                        `c1` and `c2`, the result will be `c2`.

    Returns:
        Point: A Shapely Point object representing the shifted point.

    Note:
        Adopted from Dirk Eilander at:
        https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/sandbox/osmmodelbuilding/shapely_tools.py
    """
    x1, y1 = c1
    x2, y2 = c2

    # Check for zero length line
    if (x1 - x2) == 0 and (y1 - y2) == 0:
        return Point(x1, y1)

    rel_length = np.minimum(offset / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 1)
    x_new = x1 + (x2 - x1) * rel_length
    y_new = y1 + (y2 - y1) * rel_length

    return Point(x_new, y_new)


def extend_line(line: LineString, offset: float, side: str = "both") -> LineString:
    """
    Extend a LineString in the same orientation on either the start, end, or both sides.

    Args:
        line (LineString): The input line to be extended.
        offset (float): The distance to extend the line by.
        side (str, optional): The side of the line to be extended. It can be "start", "end", or "both".
                              Default is "both".

    Returns:
        LineString: The extended LineString.

    Note:
        Adopted from Dirk Eilander at:
        https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/sandbox/osmmodelbuilding/shapely_tools.py
    """
    sides = ["start", "end"] if side == "both" else [side]

    coords = list(line.coords)
    for s in sides:
        if s == "start":
            p_new = shift_point(coords[0], coords[1], -1.0 * offset)
            coords.insert(0, tuple(p_new.coords[0]))
        elif s == "end":
            p_new = shift_point(coords[-1], coords[-2], -1.0 * offset)
            coords.append(tuple(p_new.coords[0]))
    return LineString(coords)


def extend_transect(line: LineString, length: float, side: str = "both") -> LineString:
    """
    Extend a transect LineString up to a specified total length.

    Args:
        line (LineString): The input transect line to be extended.
        length (float): The total desired length of the transect after extension.
        side (str, optional): The side of the transect to be extended. It can be "start", "end", or "both".
                              Default is "both".

    Returns:
        LineString: The extended LineString transect.

    Raises:
        NotImplementedError: If the desired length is shorter than the current transect length.

    Note:
        This function relies on the `extend_line` function.
    """
    offset = (length - line.length) / 2
    if offset < 0:
        msg = "Transects can only be extended."
        raise NotImplementedError(msg)
    # The extended line contains the original points plus the new points, so we
    # extract the coordinates and use the first and last point to create the new transect.
    line = extend_line(line, offset, side)
    coords = line.coords
    return LineString([coords[0], coords[-1]])


def extract_coordinates(pt: Point | tuple[float, float]) -> tuple[float, float]:
    """Extract coordinates from a Point or tuple.

    Args:
        pt (Union[Point, Tuple[float, float]]): A point object or tuple.

    Returns:
        Tuple[float, float]: The x and y coordinates of the point.
    """
    if isinstance(pt, Point):
        return pt.x, pt.y
    elif (
        isinstance(pt, tuple)
        and len(pt) == 2
        and all(isinstance(coord, (float, int)) for coord in pt)
    ):
        return float(pt[0]), float(pt[1])
    else:
        msg = f"Invalid point provided. Expected a Point object or a tuple of two floats. Received: {pt}"
        raise ValueError(msg)


def extract_endpoints(line: LineString) -> tuple[Point, Point]:
    """Extract the start and end points of a linestring.

    Args:
        line (LineString): A linestring.

    Returns:
        tuple[Point, Point]: The start and end points of the linestring.
    """

    # Check if line is a MultiLineString
    if isinstance(line, MultiLineString):
        msg = "MultiLineString provided instead of LineString."
        raise ValueError(msg)

    # Check validity of LineString
    if not line.is_valid:
        msg = "Invalid LineString provided."
        raise ValueError(msg)

    # Check if LineString has at least two coordinates
    coords = list(line.coords)
    if len(coords) < 2:
        msg = "LineString should have at least two coordinates."
        raise ValueError(msg)

    # Extract start and end points
    start_point = Point(coords[0])
    end_point = Point(coords[-1])

    return start_point, end_point


def get_angle(
    pt1: Point | tuple[float, float], pt2: Point | tuple[float, float]
) -> float:
    """Calculate the angle in degrees between two points.

    The angle is computed in in counter-clockwise direction from the positive x-axis,
    (or east) to the line segment connecting the two points.

    Args:
        pt1 (Union[Point, Tuple[float, float]]): The first point.
        pt2 (Union[Point, Tuple[float, float]]): The second point.

    Returns:
        float: The angle in degrees.
    """
    x1, y1 = extract_coordinates(pt1)
    x2, y2 = extract_coordinates(pt2)
    dy = y2 - y1
    dx = x2 - x1
    return math.degrees(math.atan2(dy, dx))


def get_planar_bearing(
    pt1: Point | tuple[float, float], pt2: Point | tuple[float, float]
) -> float:
    """Calculate the bearing (angle normalized to [0, 360) degrees) between two points.

    The normalization is required to handle cases where the resulting angle is negative,
    which can occur when pt1 is to the right of pt2.

    Args:
        pt1 (Union[Point, Tuple[float, float]]): The first point.
        pt2 (Union[Point, Tuple[float, float]]): The second point.

    Returns:
        float: The bearing in degrees, normalized to [0, 360].

    Example with Points:
        >>> pt1 = Point(1, 1)
        >>> pt2 = Point(2, 2)
        >>> get_bearing(pt1, pt2)
        45.0

    Example with tuples:
        >>> pt1 = (1, 1)
        >>> pt2 = (2, 2)
        >>> get_bearing(pt1, pt2)
        45.0
    """
    angle = get_angle(pt1, pt2)  # angle counter-clock wise from east
    bearing = 90 - angle  # angle to bearing
    return (bearing + 360) % 360


def get_spherical_bearing(
    pt1: tuple[float, float] | Point,
    pt2: tuple[float, float] | Point,
    src_crs: str | int,
    ellps: str = "WGS84",
) -> float:
    """
    Calculate the geodesic bearing between two points.

    Given two points and the source coordinate reference system (CRS),
    this function computes the geodesic bearing between the two points.

    Args:
        pt1 (Union[Tuple[float, float], Point]): The first point.
        pt2 (Union[Tuple[float, float], Point]): The second point.
        src_crs str | int : The source coordinate reference system as an EPSG string or code.

    Returns:
        float: The geodesic bearing in degrees, normalized to [0, 360).

    Example:
        >>> pt1 = (4.9033, 52.3680)
        >>> pt2 = (4.8977, 52.3676)
        >>> src_crs = "EPSG:4326"
        >>> get_geodesic_bearing(pt1, pt2, src_crs)
        295.1521

    Warning:
        This function has not been extensively tested. Use with caution.
    """

    # Issue a user warning
    warnings.warn(
        "This function has not been extensively tested. Use with caution.",
        UserWarning,
        stacklevel=2,
    )

    if not isinstance(pt1, Point):
        pt1 = Point(pt1)
    if not isinstance(pt2, Point):
        pt2 = Point(pt2)

    tf = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    pt1 = transform(tf.transform, pt1)  # type: ignore
    pt2 = transform(tf.transform, pt2)  # type: ignore

    lon1, lat1 = pt1[0], pt1[1]
    lon2, lat2 = pt2[0], pt2[1]

    geodesic = pyproj.Geod(ellps=ellps)
    fwd_azimuth, _, _ = geodesic.inv(lon1, lat1, lon2, lat2)
    return (fwd_azimuth + 360) % 360


def calculate_point(pt: Point, angle: float, dist: float) -> Point:
    """
    Define a point by distance and angle from a reference point.

    Args:
        pt (Point): The reference point.
        angle (float): The angle in degrees from the reference point.
        dist (float): The distance from the reference point.

    Returns:
        Point: The new point calculated by distance and angle from the reference point.
    """
    bearing = math.radians(angle)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)


def split_line_by_points(
    line: LineString, points: MultiPoint, tolerance: float = 0.1
) -> GeometryCollection:
    """
    Split a line by provided points into several segments.

    This function creates segments held in a Shapely GeometryCollection, which is iterable.
    Providing a tolerance is essential due to the float nature of points which might have
    many decimals, causing them not to be found precisely along the line. The snap() function
    from Shapely resolves this by allowing a tolerance.

    Args:
        line (LineString): The linestring to be split.
        points (MultiPoint): Points along the linestring where the line will be split.
        tolerance (float, optional): Tolerance for interpolating points. Defaults to 0.1 meters.

    Returns:
        GeometryCollection: Iterable Shapely line sequence using the `geoms` method.
    """

    return split(snap(line, points, tolerance), points)


def split_line_by_distance(
    line: LineString, distance: float, tolerance: float = 0.1
) -> GeometryCollection:
    """
    Split a line into segments of a specified distance.

    The function ensures global transects are equally spaced. It's especially useful for coastlines that are long and
    cover multiple CRS UTM regions. To describe such coastlines in the local UTM CRS, they should be split into
    several shorter segments.

    The function creates segments held in a Shapely GeometryCollection, which is iterable. It uses a helper function
    `split_line_by_points()` to perform the actual split based on computed points.

    In some cases, like islands that don't have a defined boundary, the last point is extracted directly from the
    list of coordinates. This method is slower, and is used as a fallback.

    Args:
        line (LineString): Coastline described as a Shapely LineString.
        distance (float): Desired segment length in meters.
        tolerance (float, optional): Tolerance for the split operation. Defaults to 0.1 meters.

    Returns:
        GeometryCollection: An iterable collection of linestring segments.
    """

    distances = np.arange(0, line.length, distance)

    # Check for boundary availability, use a fallback if not present
    last_point = (
        line.boundary.geoms[1] if hasattr(line.boundary, "geoms") else line.coords[-1]
    )

    points = MultiPoint([line.interpolate(dist) for dist in distances] + [last_point])

    return split_line_by_points(line, points, tolerance=tolerance)


def merge_lines(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge lines in the GeoDataFrame into a continuous line.

    This function takes in a GeoDataFrame containing line geometries, and
    attempts to merge them into a continuous line using shapely's linemerge
    operation.

    Args:
        df (GeoDataFrame): Input GeoDataFrame with line geometries.

    Returns:
        GeoDataFrame: GeoDataFrame containing the merged line.

    Note:
        If the input dataframe is empty, an empty GeoDataFrame is returned.
    """

    # Check for an empty dataframe
    if df.empty:
        return gpd.GeoDataFrame(geometry=[], crs=df.crs)

    merged_line = shapely.ops.linemerge(
        df.explode(index_parts=False).geometry.to_list()
    )

    return (
        gpd.GeoDataFrame(geometry=[merged_line], crs=df.crs)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )


def overlay_by_grid(df: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Overlay a GeoDataFrame with a given grid and return the intersected geometries.

    Args:
        df (gpd.GeoDataFrame): Input GeoDataFrame to be overlaid.
        grid (gpd.GeoDataFrame): The grid used for overlay.

    Returns:
        gpd.GeoDataFrame: Overlaid GeoDataFrame with intersected geometries.
    """

    return gpd.overlay(
        df,
        grid,
        keep_geom_type=False,
    ).explode(column="geometry", index_parts=False)


def generate_offset_line(line: LineString, offset: float) -> LineString:
    """
    Generate an offset line from the original line at a specified distance using offset_curve method.

    Args:
        line (LineString): The original line from which the offset is generated.
        offset (float): The distance for the offset. Positive values offset to the left,
            and negative values offset to the right.

    Returns:
        LineString: The offset line generated from the original line.
    """
    return line.offset_curve(offset) if offset != 0 else line


def determine_rotation_angle(
    pt1: Point | tuple[float, float],
    pt2: Point | tuple[float, float],
    target_axis: Literal["closest", "vertical", "horizontal"] = "closest",
) -> float:
    """
    Determines the correct rotation angle to align the orientation of a cross-shore transect
    either vertically, horizontally or to the closest.

    Args:
        pt1: The starting point of the transect. Can be either a Point object or a tuple of floats.
        pt2: The ending point of the transect. Can be either a Point object or a tuple of floats.
        target_axis: The target axis to align the transect to. Can be either "closest", "vertical" or "horizontal".

    Returns:
        float: The rotation angle in degrees. Positive values represent counterclockwise rotation.

    Example:
    >>> determine_rotation_angle((0,0), (1,1), target_axis="horizontal")
    -45.0

    Raises:
    - ValueError: If the computed angle is not within the expected range [-180, 180].
    """

    x1, y1 = extract_coordinates(pt1)
    x2, y2 = extract_coordinates(pt2)

    angle = get_angle(pt1, pt2)
    bearing = get_planar_bearing(pt1, pt2)
    logging.info(f"Angle between points: {angle} degrees.")
    logging.info(f"Bearing between points: {bearing} degrees.")

    if x1 == x2 or y1 == y2:  # combines the two conditions as the result is the same
        return 0

    if target_axis == "closest":
        angle_rotations = {
            (0, 45): lambda b: b,
            (45, 90): lambda b: -(90 - b),
            (90, 135): lambda b: b - 90,
            (135, 180): lambda b: -(180 - b),
            (180, 225): lambda b: b - 180,
            (225, 270): lambda b: -(270 - b),
            (270, 315): lambda b: b - 270,
            (315, 360): lambda b: -(360 - b),
        }

    elif target_axis == "horizontal":
        angle_rotations = {
            (0, 90): lambda b: 90 + b,
            (90, 180): lambda b: -(270 - b),
            (180, 270): lambda b: b - 270,
            (270, 360): lambda b: b - 270,
        }

    elif target_axis == "vertical":
        angle_rotations = {
            (0, 90): lambda b: b,
            (90, 180): lambda b: b,
            (180, 270): lambda b: -(270 - b),
            (270, 360): lambda b: -(360 - b),
        }

    else:
        msg = (
            f"Invalid target_axis: {target_axis}. Must be one of 'closest', 'vertical'"
            " or 'horizontal'."
        )
        raise ValueError(msg)

    for (lower_bound, upper_bound), rotation_func in angle_rotations.items():
        if lower_bound <= bearing < upper_bound:
            return rotation_func(bearing)

    msg = "Invalid bearing computed. Expected bearing within range [0, 360]."
    raise ValueError(msg)


def crosses_antimeridian(df: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Determines whether linestrings in a GeoDataFrame cross the International Date Line.

    Args:
        df (gpd.GeoDataFrame): Input GeoDataFrame with LineString geometries.

    Returns:
        gpd.GeoSeries: Series indicating whether each LineString crosses the antimeridian.

    Example:
        >>> df = gpd.read_file('path_to_file.geojson')
        >>> df['crosses'] = crosses_antimeridian(df)
        >>> print(df['crosses'])

    Note:
        Assumes the input GeoDataFrame uses a coordinate system in meters.
        If using a degree-based system like EPSG:4326, the results may not be accurate.
    """
    TEMPLATE = pd.Series([], dtype="bool")

    if df.crs.to_epsg() != 3857:
        df = df.to_crs(3857)

    if df.empty:
        return TEMPLATE

    coords = df.geometry.astype(object).apply(lambda x: (x.coords[0], x.coords[-1]))
    return coords.apply(lambda x: x[0][0] * x[1][0] < 0)


def buffer_in_utm(
    geom: Polygon,
    src_crs: str | int,
    buffer_distance: float | int,
    utm_crs: str | int | None = None,
) -> Polygon:
    """
    Apply a buffer to a geometry in its appropriate UTM projection.

    Args:
        geom (shapely.geometry.Polygon): Input geometry.
        src_crs (str): The coordinate reference system of the input geometry in PROJ string format or EPSG code.
        buffer_distance (float | int): Buffer distance in metres.
        utm_crs (str): The UTM zone of the input geometry in PROJ String or EPSG code.


    Returns:
        shapely.geometry.Polygon: Buffered geometry.

    Example:
        from shapely.geometry import Point
        buffered_geom = buffer_in_utm(Point(12.4924, 41.8902), src_crs="EPSG:4326", buffer_distance=-100)
    """
    if not utm_crs:
        utm_crs = gpd.GeoSeries(geom, crs=src_crs).estimate_utm_crs()

    # Set up the transformers for forward and reverse transformations
    transformer_to_utm = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    transformer_from_utm = Transformer.from_crs(utm_crs, src_crs, always_xy=True)

    # Perform the transformations
    geom_utm = transform(transformer_to_utm.transform, geom)  # type: ignore
    geom_buffered_utm = geom_utm.buffer(buffer_distance)
    geom_buffered = transform(transformer_from_utm.transform, geom_buffered_utm)  # type: ignore

    return geom_buffered


def add_line_length(
    gdf: gpd.GeoDataFrame, dst_crs: str | int, length_col: str
) -> gpd.GeoDataFrame:
    """
    Adds a column to the GeoDataFrame `gdf` with the length of each LineString in meters,
    in the target CRS `target_crs`. The column will be named `length_col`.

    Parameters:
    -----------
    gdf : GeoDataFrame
        The input GeoDataFrame containing LineString geometries.
    target_crs : str
        The target CRS to convert the geometries to, in EPSG format (e.g. 'EPSG:4326', 4326).
    length_col : str
        The name of the new column to add to the GeoDataFrame.

    Returns:
    --------
    GeoDataFrame
        The input GeoDataFrame with the new length column added.
    """
    src_crs = gdf.crs
    gdf = gdf.to_crs(dst_crs)
    gdf[length_col] = gdf.geometry.length
    gdf = gdf.to_crs(src_crs)
    return gdf


def get_xy_range(
    gdf: gpd.GeoDataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Extracts the min and max longitude (x) and latitude (y) from a GeoDataFrame
    and returns them as a formatted xy range suitable for geoviews.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]:
            The x range (min lon, max lon) and y range (min lat, max lat).
    """

    x_range = tuple(gdf.total_bounds[[0, 2]])
    y_range = tuple(gdf.total_bounds[[1, 3]])
    return x_range, y_range


def add_geometry_lengths(
    df: gpd.GeoDataFrame,
    crs: str,
    group_col: str,
    geometry_length_col: str = "geometry_length",
    total_length_col: str = "partial_FID_length",
) -> pd.DataFrame:
    """
    Computes and adds geometry length for a given GeoDataFrame in a specified CRS.

    Args:
        df (gpd.GeoDataFrame): The input GeoDataFrame.
        crs (str): The CRS used for computing the length.
        group_col (str, optional): Column used for grouping and calculating total lengths.
        geometry_length_col (str, optional): Name of the column where individual geometry lengths will be stored. Defaults to 'geometry_length'.
        total_length_col (str, optional): Name of the column where total lengths per group will be stored. Defaults to 'partial_FID_length'.

    Returns:
        pd.DataFrame: A DataFrame with added length columns.

    Warnings:
        This function might provide inaccurate total lengths for 'FID's that extend across multiple CRSs. It is recommended to overlay the input data with a UTM grid and group it by the appropriate CRS before computing lengths using this function.
    """

    silence_shapely_warnings()

    # Warning about potential inaccuracies
    warnings.warn(
        (
            "This function might provide inaccurate total lengths for 'FID's that"
            " extend across multiple CRSs. It is recommended to overlay the input data"
            " with a UTM grid and group it by the appropriate CRS before computing"
            " lengths using this function."
        ),
        UserWarning,
        stacklevel=2,
    )

    # Compute geometry length in the specified CRS
    df_temp = df.to_crs(crs)
    df_temp[geometry_length_col] = df_temp.geometry.length
    df = df_temp.to_crs(df.crs)

    # Compute total length per group
    total_lengths = (
        df.groupby(group_col)[geometry_length_col]
        .sum()
        .rename(total_length_col)
        .reset_index()
    )

    # Merge to the main DataFrame and drop temporary geometry_length column
    df = pd.merge(df.drop(columns=[geometry_length_col]), total_lengths, on=group_col)

    return df
