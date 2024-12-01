import datetime
from abc import abstractmethod
from typing import (
    Any,
    Literal,
    get_args,
    get_origin,
)

import geopandas as gpd
import msgspec
import numpy as np
import pandas as pd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from coastpy.schema.schema_hooks import decode_custom, encode_custom
from coastpy.schema.type_enums import (
    CoastalType,
    HasDefense,
    IsBuiltEnvironment,
    LandformType,
    ShoreType,
)
from coastpy.schema.type_mappings import GEOPARQUET_TYPE_MAP, PANDAS_TYPE_MAP


class BaseModel(
    msgspec.Struct,
    kw_only=True,  # Set all fields as keyword-only to avoid ordering issues
    tag=str.lower,
    tag_field="type",
    dict=True,
    omit_defaults=True,
    repr_omit_defaults=True,
):
    @property
    def __defined_struct_fields__(self) -> tuple[str, ...]:
        """Return tuple of fields explicitly defined with non-default values."""
        defined_fields = [
            field.name
            for field in msgspec.structs.fields(self)
            if getattr(self, field.name, None) != field.default
        ]
        return tuple(defined_fields)

    @property
    def __field_types__(self) -> dict[str, type]:
        """
        Generate a flat dictionary of field names and their types, consistent with `to_dict()`.
        Nested BaseModel fields are flattened, with conflicts raising a KeyError.
        """
        # Start extraction from the current class
        return self._get_field_types(self.__class__)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseModel":
        """Create an instance from JSON string."""
        decoder = msgspec.json.Decoder(cls, dec_hook=decode_custom)
        return decoder.decode(json_str.encode())

    @classmethod
    def from_frame(cls, frame: gpd.GeoDataFrame) -> "BaseModel":
        """Create an instance from a GeoDataFrame row."""
        if len(frame) != 1:
            msg = "Input GeoDataFrame must contain exactly one row."
            raise ValueError(msg)
        data = frame.iloc[0].to_dict()
        return cls.from_dict(data)

    @classmethod
    def from_series(cls, series: pd.Series | gpd.GeoSeries) -> "BaseModel":
        """Create an instance from a Series."""
        return cls.from_dict(series.to_dict())

    @classmethod
    def _get_field_types(
        cls, struct_cls: type[msgspec.Struct], parent_key: str = "", sep: str = "."
    ) -> dict[str, type]:
        """
        Recursively extract field types for a given `msgspec.Struct` class, supporting nested fields.

        Args:
            struct_cls (Type[msgspec.Struct]): The `msgspec.Struct` class to extract types from.
            parent_key (str): Prefix for namespacing keys.
            sep (str): Separator for namespaced keys.

        Returns:
            Dict[str, Type]: A dictionary of field names (namespaced) and their resolved types.
        """
        field_types = {}

        for field_ in msgspec.structs.fields(struct_cls):
            field_name = field_.name
            field_type = field_.type
            new_key = f"{parent_key}{sep}{field_name}" if parent_key else field_name

            # Check if the field type is a Literal
            if get_origin(field_type) is Literal:
                # Extract the first literal value and determine its base type
                literal_values = get_args(field_type)
                resolved_type = type(literal_values[0]) if literal_values else str

            # Handle nested BaseModel fields
            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                # Recursively fetch nested field types
                nested_types = cls._get_field_types(
                    field_type, parent_key=new_key, sep=sep
                )
                for nested_name, nested_type in nested_types.items():
                    if nested_name in field_types:
                        msg = f"Key conflict detected: '{nested_name}' already exists in the field types."
                        raise KeyError(msg) from None
                    field_types[nested_name] = nested_type
                continue  # Skip adding the current field_name as it is handled recursively

            # Handle dictionaries (e.g., bbox)
            elif get_origin(field_type) is dict:
                key_type, value_type = get_args(field_type)
                if key_type is str and value_type in {int, float}:
                    # Special handling for bbox-like dictionaries
                    resolved_type = dict
                else:
                    resolved_type = dict

            # Handle other types
            else:
                resolved_type = (
                    get_args(field_type)[0] if get_args(field_type) else field_type
                )

            # Add the resolved type to the dictionary
            if new_key in field_types:
                msg = f"Key conflict detected: '{new_key}' already exists in the field types."
                raise KeyError(msg) from None
            field_types[new_key] = resolved_type

        return field_types

    @classmethod
    def null(cls) -> "BaseModel":
        """Create an instance with null values for each field."""
        null_values = {}
        field_types = cls._get_field_types(cls)

        for field_ in msgspec.structs.fields(cls):
            field_name = field_.name
            field_type = field_.type

            try:
                base_type = field_types[field_name]
            except KeyError:
                base_type = field_type

            if base_type is str:
                null_values[field_name] = ""
            elif base_type in {int, float}:
                null_values[field_name] = np.nan
            elif base_type is bool:
                null_values[field_name] = False
            elif base_type in {datetime.datetime, pd.Timestamp}:
                null_values[field_name] = pd.NaT
            elif get_origin(base_type) is dict:
                # Handle structured data types like bbox
                key_type, value_type = get_args(base_type)
                if key_type is str and value_type in {int, float}:
                    null_values[field_name] = {
                        key: np.nan for key in ["xmin", "ymin", "xmax", "ymax"]
                    }
                else:
                    null_values[field_name] = {}
            elif issubclass(
                base_type,
                GeometryCollection
                | LineString
                | Point
                | Polygon
                | MultiPolygon
                | MultiPoint
                | MultiLineString,
            ):
                null_values[field_name] = GeometryCollection()

            elif isinstance(base_type, type) and issubclass(base_type, BaseModel):
                # Initialize nested BaseModel with its null values
                null_values[field_name] = base_type.null()

            else:
                msg = f"Unhandled field type '{base_type}' for field '{field_name}'. Add support for this type."
                raise TypeError(msg)

        return cls(**null_values)

    def to_dict(self, flatten: bool = False) -> dict[str, Any]:
        """
        Convert instance to a dictionary format.

        Args:
            flatten (bool): Whether to return a flattened dictionary.

        Returns:
            Dict[str, Any]: The instance represented as a dictionary.
        """
        if flatten:
            return self.to_flat_dict()
        return msgspec.structs.asdict(self)

    def to_frame(
        self, geometry="geometry", bbox="bbox", crs="EPSG:4326"
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Convert the instance to a DataFrame or GeoDataFrame, flattening nested fields.

        Args:
            geometry (str): Dot-separated path to the geometry attribute.
            bbox (str): Dot-separated path to the bbox attribute.
            crs (str): Coordinate reference system for GeoDataFrame.

        Returns:
            pd.DataFrame | gpd.GeoDataFrame: Flattened DataFrame or GeoDataFrame.
        """
        flat_dict = self.to_dict(flatten=True)

        geometry_ = flat_dict.pop(geometry, None)
        bbox_ = flat_dict.pop(bbox, None)

        df = pd.DataFrame([flat_dict])

        if bbox_ is not None:
            df[bbox] = [bbox_]

        if geometry_ is not None:
            df[geometry] = [geometry_]

        if geometry_ is not None:
            return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

        return df

    def to_series(
        self, geometry="geometry", bbox="bbox", crs="EPSG:4326"
    ) -> pd.Series | gpd.GeoSeries:
        """
        Convert the instance to a Series or GeoSeries, flattening nested fields.

        Args:
            geometry (str): Dot-separated path to the geometry attribute.
            bbox (str): Dot-separated path to the bbox attribute.
            crs (str): Coordinate reference system for GeoDataFrame.

        Returns:
            pd.Series | gpd.Series: Flattened Series or GeoSeries.
        """
        flat_dict = self.to_dict(flatten=True)

        geometry_ = flat_dict.pop(geometry, None)
        bbox_ = flat_dict.pop(bbox, None)

        s = pd.Series(flat_dict)

        if bbox_ is not None:
            s[bbox] = [bbox_]

        if geometry_ is not None:
            s[geometry] = [geometry_]

        if geometry_ is not None:
            return gpd.GeoSeries(s, geometry=geometry, crs=crs)

        return s

    def to_flat_dict(self, sep=".", ignore_fields: list | None = None) -> dict:
        """
        Flatten the instance into a dictionary with namespaced keys,
        using class-level tags for nested BaseModel fields.

        Args:
            sep (str): Separator for nested keys.
            ignore_fields (list | None): Attributes to exclude from flattening.

        Returns:
            dict: A flattened dictionary with only explicitly defined fields.
        """
        if ignore_fields is None:
            ignore_fields = ["bbox"]

        def _extract_fields_with_tag(instance, tag=None):
            """
            Extracts fields from a BaseModel instance, optionally tagging keys.

            Args:
                instance (BaseModel): The BaseModel instance to process.
                tag (str | None): Optional tag to prefix keys.

            Returns:
                dict: Extracted fields with appropriate tags.
            """
            flat_data = {}

            for field_name in instance.__defined_struct_fields__:
                field_value = getattr(instance, field_name, None)

                # Handle nested BaseModel instances
                if isinstance(field_value, BaseModel):
                    # Use the nested BaseModel's tag
                    nested_tag = getattr(
                        field_value.__class__.__struct_config__, "tag", field_name
                    )
                    nested_data = _extract_fields_with_tag(field_value, tag=nested_tag)
                    flat_data.update(nested_data)

                # Handle special attributes (e.g., bbox)
                elif field_name in ignore_fields and isinstance(field_value, dict):
                    flat_data[field_name] = field_value

                # Handle dictionaries (non-special attributes)
                elif isinstance(field_value, dict):
                    for k, v in field_value.items():
                        flat_data[f"{field_name}{sep}{k}"] = v

                # Handle lists
                elif isinstance(field_value, list):
                    for i, v in enumerate(field_value):
                        flat_data[f"{field_name}{sep}{i}"] = v

                # Handle primitive values
                else:
                    key = f"{tag}{sep}{field_name}" if tag else field_name
                    flat_data[key] = field_value

            return flat_data

        # Extract fields from the current instance
        return _extract_fields_with_tag(self)

    def encode(self) -> bytes:
        """Encode instance as JSON bytes."""
        encoder = msgspec.json.Encoder(enc_hook=encode_custom)
        return encoder.encode(self)

    def decode(self, data: bytes):
        """Decode JSON bytes to an instance."""
        decoder = msgspec.json.Decoder(ModelUnion, dec_hook=decode_custom)
        return decoder.decode(data)

    def to_json(self) -> str:
        """Encode instance as JSON string."""
        return self.encode().decode()

    def validate(self) -> bool:
        """
        Validate the current instance against its schema using JSON serialization.

        Returns:
            bool: True if the instance is valid, otherwise False.
        """
        try:
            # Serialize the instance to JSON and attempt to decode it back
            json_data = self.to_json()
            self.decode(json_data.encode())
            return True
        except msgspec.ValidationError as e:
            print(f"Validation Error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return False

    @abstractmethod
    def example(cls) -> "BaseModel":
        """Create an example instance with predefined example values."""
        ...

    def to_meta(
        self, mode: Literal["pandas", "geoparquet"] = "pandas"
    ) -> dict[str, str]:
        """
        Generate a dictionary mapping field names to their corresponding types.

        Parameters:
            mode (str): Either 'pandas' (default) or 'geoparquet'. Determines the type system to use.

        Returns:
            Dict[str, str]: A dictionary mapping field names to their respective types.

        Raises:
            KeyError: If duplicate keys are detected in nested fields.
            ValueError: If the `mode` is not supported.
        """
        # Select the appropriate type map
        if mode == "pandas":
            type_map = PANDAS_TYPE_MAP
        elif mode == "geoparquet":
            type_map = GEOPARQUET_TYPE_MAP
        else:
            msg = f"Unsupported mode '{mode}'. Use 'pandas' or 'geoparquet'."
            raise ValueError(msg)

        meta = {}
        for field_name, field_type in self.__field_types__.items():
            try:
                # Map the type to the selected system
                meta[field_name] = type_map.get(field_type, type_map[object])
            except KeyError:
                msg = f"Type '{field_type}' for field '{field_name}' is not supported in the {mode} type map."
                raise KeyError(msg) from None

        return meta

    def empty_frame(
        self, geometry: str = "geometry", crs="EPSG:4326"
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Create an empty DataFrame with appropriate column types.
        """
        _meta = self.to_meta()

        if geometry in _meta:
            empty_data = {
                col: pd.Series(dtype=col_type)
                if col_type != GeometryCollection()
                else []
                for col, col_type in _meta.items()
            }
            return gpd.GeoDataFrame(empty_data, geometry=geometry, crs=crs)

        # For non-geometric data, return a regular DataFrame
        empty_data = {col: pd.Series(dtype=col_type) for col, col_type in _meta.items()}
        return pd.DataFrame(empty_data)

    @classmethod
    def from_flat_dict(cls, record: dict, separator=".") -> "BaseModel":
        """
        Reconstruct a BaseModel instance from a flat dictionary with namespaced keys.

        Args:
            record (dict): Input dictionary with flattened, namespaced keys.

        Returns:
            BaseModel: Reconstructed instance.
        """
        nested_data = {}
        fields = {field.name: field.type for field in msgspec.structs.fields(cls)}

        for field_name, field_type in fields.items():
            # Extract flat keys using namespace convention
            prefix = f"{field_name}{separator}"
            nested_keys = {
                k[len(prefix) :]: v for k, v in record.items() if k.startswith(prefix)
            }

            # NOTE: Handle nested BaseModel instances
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                if nested_keys:  # Nested dictionary found
                    nested_data[field_name] = field_type.from_flat_dict(nested_keys)
            elif isinstance(field_type, dict) and get_origin(field_type) is dict:
                # NOTE: Directly map dictionaries (e.g., bbox)
                nested_data[field_name] = nested_keys or record.get(field_name)
            else:
                # Assign primitive or list values directly
                if field_name in record:
                    nested_data[field_name] = record[field_name]

        return cls(**nested_data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModel":
        """
        Create an instance from a dictionary, handling both directly instantiable
        and flattened structures.

        Args:
            data (Dict[str, Any]): The input data dictionary.

        Returns:
            BaseModel: The instantiated object.

        Raises:
            ValueError: If the dictionary is invalid or improperly structured.
        """
        try:
            # Attempt direct instantiation
            return cls(**data)
        except TypeError as e:  # noqa: F841
            # If direct instantiation fails, assume flattened structure and attempt reconstruction
            try:
                return cls.from_flat_dict(data)
            except Exception as flat_error:
                # Re-raise with detailed context if reconstruction fails
                msg = (
                    f"Failed to initialize {cls.__name__} from the provided data. "
                    f"Ensure the input is correctly structured. Original error: {flat_error}"
                )
                raise ValueError(msg) from flat_error
        except Exception as e:
            # Catch any unexpected errors and re-raise with meaningful context
            msg = (
                f"Unexpected error occurred while initializing {cls.__name__}. "
                f"Original error: {e}"
            )
            raise ValueError(msg) from e


class Transect(BaseModel, tag="transect"):
    transect_id: str
    geometry: LineString
    lon: float | None = None
    lat: float | None = None
    bearing: float | None = None
    osm_coastline_is_closed: bool | None = None
    osm_coastline_length: int | None = None
    utm_epsg: int | None = None
    bbox: dict[str, float] | None = None
    quadkey: str | None = None
    continent: str | None = None
    country: str | None = None
    common_country_name: str | None = None
    common_region_name: str | None = None

    @classmethod
    def example(cls):
        _EXAMPLE_VALUES = {
            "transect_id": "cl32408s01tr00223948",
            "geometry": LineString(
                [
                    [4.287529606158882, 52.106643659044614],
                    [4.266728801968574, 52.11926398930266],
                ]
            ),
            "lon": 4.277131,
            "lat": 52.112953,
            "bearing": 313.57275,
            "osm_coastline_is_closed": False,
            "osm_coastline_length": 1014897,
            "utm_epsg": 32631,
            "bbox": {
                "xmax": 4.287529606158882,
                "xmin": 4.266728801968574,
                "ymax": 52.11926398930266,
                "ymin": 52.106643659044614,
            },
            "quadkey": "020202113000",
            "continent": "EU",
            "country": "NL",
            "common_country_name": "Netherlands",
            "common_region_name": "South Holland",
        }
        return cls(**_EXAMPLE_VALUES)


class TypologyTrainSample(BaseModel, tag="typology"):
    transect: Transect
    user: str  # TODO: annotate/literal by regex to make empty strings invalid
    uuid: str  # universal_unique_id = uuid.uuid4().hex[:12]
    datetime_created: datetime.datetime
    datetime_updated: datetime.datetime
    shore_type: ShoreType
    coastal_type: CoastalType
    landform_type: LandformType
    is_built_environment: IsBuiltEnvironment
    has_defense: HasDefense
    is_challenging: bool
    comment: str
    link: str
    confidence: str
    is_validated: bool

    def to_frame(
        self, geometry="transect.geometry", bbox="transect.bbox", crs="EPSG:4326"
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Convert the instance to a DataFrame or GeoDataFrame, flattening nested fields.

        Args:
            geometry (str): Dot-separated path to the geometry attribute.
            bbox (str): Dot-separated path to the bbox attribute.
            crs (str): Coordinate reference system for GeoDataFrame.

        Returns:
            pd.DataFrame | gpd.GeoDataFrame: Flattened DataFrame or GeoDataFrame.
        """
        return super().to_frame(geometry=geometry, bbox=bbox, crs=crs)

    @classmethod
    def example(cls) -> "TypologyTrainSample":
        _EXAMPLE_VALUES = {
            "transect": Transect.example(),
            "user": "floris-calkoen",
            "uuid": "3b984582ecd6",
            "datetime_created": datetime.datetime(2024, 1, 9, 12, 0),
            "datetime_updated": datetime.datetime(2024, 1, 11, 12, 0),
            "shore_type": "sandy_gravel_or_small_boulder_sediments",
            "coastal_type": "sediment_plain",
            "landform_type": "mainland_coast",
            "is_built_environment": "true",
            "has_defense": "true",
            "is_challenging": False,
            "comment": "This is an example transect including a comment.",
            "link": "https://example.com/link-to-google-street-view",
            "confidence": "high",
            "is_validated": True,
        }
        return cls(**_EXAMPLE_VALUES)


class TypologyTestSample(BaseModel, tag="typologytestsample"):
    train_sample: TypologyTrainSample
    pred_shore_type: ShoreType
    pred_coastal_type: CoastalType
    pred_has_defense: HasDefense
    pred_is_built_environment: IsBuiltEnvironment

    # def to_frame(
    #     self, geometry="transect.geometry", bbox="transect.bbox", crs="EPSG:4326"
    # ) -> pd.DataFrame | gpd.GeoDataFrame:
    #     """
    #     Convert the instance to a DataFrame or GeoDataFrame, flattening nested fields.

    #     Args:
    #         geometry (str): Dot-separated path to the geometry attribute.
    #         bbox (str): Dot-separated path to the bbox attribute.
    #         crs (str): Coordinate reference system for GeoDataFrame.

    #     Returns:
    #         pd.DataFrame | gpd.GeoDataFrame: Flattened DataFrame or GeoDataFrame.
    #     """
    #     return super().to_frame(geometry=geometry, bbox=bbox, crs=crs)

    @classmethod
    def example(cls) -> "TypologyTestSample":
        _EXAMPLE_VALUES = {
            "train_sample": TypologyTrainSample.example(),
            "pred_shore_type": "sandy_gravel_or_small_boulder_sediments",
            "pred_coastal_type": "sediment_plain",
            "pred_has_defense": "true",
            "pred_is_built_environment": "false",
        }
        return cls(**_EXAMPLE_VALUES)


class TypologyInferenceSample(BaseModel, tag="typologyinferencesample"):
    transect: Transect
    pred_shore_type: ShoreType
    pred_coastal_type: CoastalType
    pred_has_defense: HasDefense
    pred_is_built_environment: IsBuiltEnvironment

    @classmethod
    def example(cls) -> "TypologyInferenceSample":
        _EXAMPLE_VALUES = {
            "transect": Transect.example(),
            "pred_shore_type": "sandy_gravel_or_small_boulder_sediments",
            "pred_coastal_type": "cliffed_or_steep",
            "pred_has_defense": "true",
            "pred_is_built_environment": "false",
        }
        return cls(**_EXAMPLE_VALUES)


ModelUnion = (
    Transect | TypologyTrainSample | TypologyTestSample | TypologyInferenceSample
)
