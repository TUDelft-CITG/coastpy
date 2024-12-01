import datetime
from typing import Any

from shapely import wkt
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def custom_schema_hook(typ) -> dict[str, Any]:
    """Provide JSON schema for custom types."""
    if typ is LineString:
        # Represent LineString as a WKT string
        return {"type": "string", "description": "A WKT representation of a LineString"}
    if typ is dict:  # Example for bbox as dict
        return {
            "type": "object",
            "properties": {
                "xmin": {"type": "number"},
                "ymin": {"type": "number"},
                "xmax": {"type": "number"},
                "ymax": {"type": "number"},
            },
            "required": ["xmin", "ymin", "xmax", "ymax"],
        }
    # Raise an error if typ is not handled instead of returning None
    msg = f"Unsupported type for schema generation: {typ}"
    raise ValueError(msg)


def encode_custom(obj):
    """Encode custom data types for serialization."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(
        obj,
        GeometryCollection
        | LineString
        | Point
        | Polygon
        | MultiPolygon
        | MultiPoint
        | MultiLineString,
    ):
        return obj.wkt
    msg = f"Type {type(obj)} not supported"
    raise TypeError(msg)


def decode_custom(type, obj):
    """Decode custom data types for deserialization."""
    if type is datetime.datetime:
        return datetime.datetime.fromisoformat(obj)
    elif type in {
        GeometryCollection,
        LineString,
        Point,
        Polygon,
        MultiPolygon,
        MultiPoint,
        MultiLineString,
    }:
        try:
            return wkt.loads(obj)
        except Exception as e:
            msg = f"Failed to decode geometry: {e}"
            raise ValueError(msg) from e
    return obj
