import datetime
import uuid

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

# Mapping Python types to Pandas dtypes
PANDAS_TYPE_MAP = {
    str: "object",
    int: "int64",
    float: "float64",
    bool: "bool",
    datetime.datetime: "datetime64[ns]",
    datetime.date: "datetime64[ns]",
    datetime.time: "object",
    dict: "object",
    list: "object",
    tuple: "object",
    uuid.UUID: "object",
    Point: "object",
    LineString: "object",
    Polygon: "object",
    MultiPoint: "object",
    MultiLineString: "object",
    MultiPolygon: "object",
    GeometryCollection: "object",
    object: "object",
}

# Mapping Python types to GeoParquet dtypes
GEOPARQUET_TYPE_MAP = {
    str: "STRING",
    int: "INTEGER",
    float: "DOUBLE",
    bool: "BOOLEAN",
    datetime.datetime: "DATETIME",
    datetime.date: "DATE",
    datetime.time: "TIME",
    dict: "STRUCT",
    list: "ARRAY",
    tuple: "ARRAY",
    uuid.UUID: "STRING",
    Point: "GEOMETRY",
    LineString: "GEOMETRY",
    Polygon: "GEOMETRY",
    MultiPoint: "GEOMETRY",
    MultiLineString: "GEOMETRY",
    MultiPolygon: "GEOMETRY",
    GeometryCollection: "GEOMETRY",
    object: "STRING",
}
