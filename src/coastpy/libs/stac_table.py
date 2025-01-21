# NOTE: this is copied from Tom Augspurger' stac-table repository. We had to adopt some
# change some parts and installing the package on remote machines from gh was a bit tedious.
# Try to keep this code in sync with the original and remove this note when the package
# can be installed. Also please acknowledge the original author.
"""
Generate STAC Collections for tabular datasets.
"""

import copy
import datetime
import enum
from typing import Any, TypeVar

import dask
import dask_geopandas
import fsspec
import pandas as pd
import pyarrow as pa
import pyproj
import pystac
import pystac.utils
import shapely.geometry
from pystac.asset import Asset
from pystac.extensions import projection
from shapely.ops import transform

T = TypeVar("T", pystac.Collection, pystac.Item)
SCHEMA_URI = "https://stac-extensions.github.io/table/v1.2.0/schema.json"
# https://issues.apache.org/jira/browse/PARQUET-1889: parquet doesn't officially
# have a type yet.
# NOTE: follow discussion at: https://github.com/opengeospatial/geoparquet/issues/115
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


class InferDatetimeOptions(str, enum.Enum):
    no = "no"
    midpoint = "midpoint"
    unique = "unique"
    range = "range"


def generate(
    uri: str,
    template: pystac.Item,
    infer_bbox: bool = True,
    proj: bool | dict | None = None,
    infer_geometry: bool = False,
    infer_datetime: str = "no",
    datetime_column: str | None = None,
    datetime: datetime.datetime | None = None,
    metadata_created: datetime.datetime | None = None,
    count_rows: bool = True,
    asset_href: str | None = None,
    asset_key: str = "data",
    asset_title: str = "Dataset root",
    asset_description: str = "Root of the dataset",
    asset_media_type: str = "application/vnd.apache.parquet",
    asset_roles: list[str] | None = None,
    asset_extra_fields: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
    validate: bool = True,
) -> pystac.Item:
    """
    Generate a STAC Item from a Parquet dataset.

    Args:
        uri (str): URI of the dataset (e.g., "az://path/to/file.parquet").
        template (pystac.Item): Template STAC item to clone and populate.
        infer_bbox (bool, optional): Whether to infer the bounding box.
        proj (bool or dict, optional): Include projection information.
            - Pass `True` for default projection metadata (e.g., `proj:crs`).
            - Pass a dictionary for custom projection values.
        infer_geometry (bool, optional): Whether to compute and add geometry information.
        infer_datetime (str, optional): Method to infer datetime:
            - "no": Do not infer datetime.
            - "midpoint": Use the midpoint of the datetime range.
            - "unique": Use the unique datetime value (raises an error if not unique).
            - "range": Add `start_datetime` and `end_datetime` based on the range.
        datetime_column (str, optional): Name of the column containing datetime values.
            Required if `infer_datetime` is not "no".
        data_created (str, optional): ISO 8601 timestamp for when the data was created.
        metadata_created (str, optional): ISO 8601 timestamp for when metadata was created.
        count_rows (bool, optional): Add the row count to `table:row_count`.
        asset_href (str, optional): Custom URI for the asset (overrides `uri` if provided).
        asset_key (str, optional): Key for the asset in the STAC item (default: "data").
        asset_title (str, optional): Title of the asset.
        asset_description (str, optional): Description of the asset.
        asset_media_type (str, optional): Media type of the asset (e.g., Parquet MIME type).
        asset_roles (str or list, optional): Roles for the asset (default: "data").
        asset_extra_fields (dict, optional): Additional metadata fields for the asset.
        storage_options (dict, optional): fsspec storage options for accessing the dataset.
        validate (bool, optional): Validate the generated STAC item (default: True).

    Returns:
        pystac.Item: A STAC item populated with metadata and assets.
    """

    if asset_roles is None:
        asset_roles = ["data"]

    if proj is None or proj is False:
        proj = {}

    # NOTE: Consider if its better to create from template or from scratch
    item = copy.deepcopy(template)

    data = None
    storage_options = storage_options or {}
    ds = parquet_dataset_from_url(uri, storage_options)

    if (
        infer_bbox
        or infer_geometry
        or infer_datetime != InferDatetimeOptions.no
        or proj is True
    ):
        # NOTE: gather spatial partitions has been temporarily disabled
        data = dask_geopandas.read_parquet(
            uri, storage_options=storage_options, gather_spatial_partitions=False
        )
        data.calculate_spatial_partitions()

    columns = get_columns(ds.schema)
    item.properties["table:columns"] = columns

    if proj is True:
        proj = get_proj(data)
    proj = proj or {}

    # TODO: Add schema when published
    if SCHEMA_URI not in item.stac_extensions:
        item.stac_extensions.append(SCHEMA_URI)
    if proj and projection.SCHEMA_URI not in item.stac_extensions:
        item.stac_extensions.append(projection.SCHEMA_URI)

    extra_proj = {}
    if infer_bbox:
        spatial_partitions = data.spatial_partitions  # type: ignore

        if spatial_partitions is None:
            msg = "No spatial partitions found in the dataset."
            raise ValueError(msg)

        src_crs = spatial_partitions.crs.to_epsg()
        tf = pyproj.Transformer.from_crs(src_crs, 4326, always_xy=True)

        bbox = spatial_partitions.unary_union.bounds
        # NOTE: bbox of unary union will be stored under proj extension as projected
        extra_proj["proj:bbox"] = bbox

        # NOTE: bbox will be stored in pystsac.Item.bbox in EPSG:4326
        bbox = transform(tf.transform, shapely.geometry.box(*bbox))
        item.bbox = list(bbox.bounds)

    if infer_geometry:
        # NOTE: geom  under proj extension as projected
        geometry = data.unary_union.compute()  # type: ignore
        extra_proj["proj:geometry"] = shapely.geometry.mapping(geometry)

        # NOTE: geometry will be stored in pystsac.Item.geometry in EPSG:4326
        src_crs = data.spatial_partitions.crs.to_epsg()  # type: ignore
        tf = pyproj.Transformer.from_crs(src_crs, 4326, always_xy=True)
        geometry = transform(tf.transform, geometry)
        item.geometry = shapely.geometry.mapping(geometry)

    if infer_bbox and item.geometry is None:
        # If bbox is set then geometry must be set as well.
        item.geometry = shapely.geometry.mapping(
            shapely.geometry.box(*item.bbox, ccw=True)  # type: ignore
        )

    if infer_geometry and item.bbox is None:
        item.bbox = shapely.geometry.shape(item.geometry).bounds  # type: ignore

    if proj or extra_proj:
        item.properties.update(**extra_proj, **proj)

    if infer_datetime == InferDatetimeOptions.no:
        if datetime is None:
            msg = "Must specify 'datetime_data' when 'infer_datetime == no'."
            raise ValueError(msg)
        if datetime_column is not None:
            msg = "Leave 'datetime_column' empty when 'infer_datetime == no'."
            raise ValueError(msg)
    else:
        if datetime_column is None:
            msg = "Must specify 'datetime_column' when 'infer_datetime != no'."
            raise ValueError(msg)
        if datetime is not None:
            msg = "Leave 'datetime_data' empty when inferring datetime."
            raise ValueError(msg)

    if metadata_created is not None:
        item.common_metadata.created = pd.Timestamp(metadata_created).to_pydatetime()
    else:
        item.common_metadata.created = pystac.utils.now_in_utc()

    if datetime is not None:
        item.datetime = datetime

    if infer_datetime == InferDatetimeOptions.midpoint:
        values = dask.compute(data[datetime_column].min(), data[datetime_column].max())  # type: ignore
        item.datetime = pd.Timestamp(pd.Series(values).mean()).to_pydatetime()

    if infer_datetime == InferDatetimeOptions.unique and datetime is not None:
        values = data[datetime_column].unique().compute()  # type: ignore
        n = len(values)
        if n > 1:
            msg = f"infer_datetime='unique', but {n} unique values found."
            raise ValueError(msg)
        item.datetime = values[0].to_pydatetime()

    if infer_datetime == InferDatetimeOptions.range:
        values = dask.compute(data[datetime_column].min(), data[datetime_column].max())  # type: ignore
        values = pd.Series(values).dt.to_pydatetime().tolist()
        item.common_metadata.start_datetime = values[0]
        item.common_metadata.end_datetime = values[1]
        # NOTE: consider if its good practice to set datetime to midpoint when range is set
        item.datetime = pd.Timestamp(pd.Series(values).mean()).to_pydatetime()

    if count_rows:
        item.properties["table:row_count"] = sum(x.count_rows() for x in ds.fragments)

    if asset_key:
        href = asset_href if asset_href is not None else uri
        asset = Asset(
            href,
            title=asset_title,
            description=asset_description,
            media_type=asset_media_type,
            roles=asset_roles,
            extra_fields=asset_extra_fields,
        )

        item.add_asset(asset_key, asset)

    if validate:
        item.validate()

    return item


def get_proj(ds) -> dict:
    """
    Read projection information from the dataset.
    """
    # Use geopandas to get the proj info
    proj = {}
    maybe_crs = ds.geometry.crs
    if maybe_crs:
        maybe_epsg = ds.geometry.crs.to_epsg()
        if maybe_epsg:
            proj["proj:epsg"] = maybe_epsg
        else:
            proj["proj:wkt2"] = ds.geometry.crs.to_wkt()

    return proj


def get_columns(schema: pa.Schema, prefix: str = "") -> list:
    columns = []
    for field in schema:
        if field.name == "__null_dask_index__":
            continue
        # Check if the field is nested
        if pa.types.is_struct(field.type):
            # For nested fields, recurse into the structure
            nested_columns = get_columns(field.type, prefix=field.name + ".")
            columns.extend(nested_columns)
        elif field.name == "geometry":
            column = {"name": "geometry", "type": "WKB"}
            columns.append(column)

        else:
            # Handle non-nested fields
            column = {"name": prefix + field.name, "type": str(field.type).lower()}
            if field.metadata is not None:
                column["metadata"] = field.metadata
            columns.append(column)
    return columns


def parquet_dataset_from_url(url: str, storage_options: dict):
    """
    Load a Parquet dataset from a URL, handling both `https://` and `az://` protocols.

    Args:
        url (str): The URL of the dataset (e.g., `az://path/to/file.parquet` or `https://...`).
        storage_options (dict): Options for cloud storage (e.g., account name, SAS tokens).

    Returns:
        pyarrow.parquet.ParquetDataset: A ParquetDataset object for the given URL.
    """
    protocol = url.split("://")[0]
    _storage_options = {} if protocol == "https" else storage_options or {}
    fs, _, _ = fsspec.get_fs_token_paths(url, storage_options=_storage_options)
    pa_fs = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(fs))
    url2 = url.split("://", 1)[-1]
    ds = pa.parquet.ParquetDataset(url2, filesystem=pa_fs)
    return ds


# def parquet_dataset_from_url(url: str, storage_options):
#     fs, _, _ = fsspec.get_fs_token_paths(url, storage_options=storage_options)
#     pa_fs = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(fs))
#     url2 = url.split("://", 1)[-1]  # pyarrow doesn't auto-strip the prefix.
#     ds = pa.parquet.ParquetDataset(url2, filesystem=pa_fs)
#     return ds
