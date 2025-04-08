import logging
import warnings
from collections.abc import Callable
from typing import Any, Self

import geopandas as gpd
import numpy as np
import odc.geo
import odc.geo.cog
import odc.geo.geobox
import odc.geo.geom
import odc.stac
import pystac
import pystac_client
import pystac_client.warnings
import stac_geoparquet
import xarray as xr

from coastpy.eo.indices import calculate_indices
from coastpy.eo.mask import (
    SceneClassification,
    apply_mask,
    geometry_mask,
    keep_rio_attrs,
    nodata_mask,
    numeric_mask,
    scl_mask,
)
from coastpy.eo.utils import data_extent_from_stac_items, geobox_from_data_extent
from coastpy.io.utils import get_datetimes, update_time_coord
from coastpy.stac.utils import read_snapshot
from coastpy.utils.xarray_utils import scale, unscale

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseCollection:
    """
    Base class for managing STAC-based collections. Supports search, load,
    masking, compositing, spectral indices, and additional operations.
    """

    default_stac_cfg = {}

    def __init__(self, catalog_url: str, collection: str, stac_cfg: dict | None = None):
        self.catalog_url = catalog_url
        self.collection = collection

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=pystac_client.client.NoConformsTo,  # type: ignore
            )
            self.catalog = pystac_client.Client.open(self.catalog_url)

        self.stac_cfg = stac_cfg or self.default_stac_cfg

        # State variables
        self.roi: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.search_params: dict = {}
        self.odc_stac_load_params: dict = {}
        self.items: list | None = None
        self.add_metadata_from_stac: bool = True
        self.dataset: xr.Dataset | None = None
        self.data_extent: gpd.GeoDataFrame | None = None

        # Mask and scale settings
        self.mask_nodata: bool = False
        self.mask_geometry: Any | None = None
        self.mask_values: list[int] | None = None
        self.scale: bool = False
        self.scale_factor: float | None = None
        self.add_offset: float | None = None
        self.scale_vars_to_skip: list[str] | None = None

        # Composite settings
        self.composite_method: str | None = None
        self.percentile: int | None = None
        self.custom_composite_func: Callable[[xr.Dataset], xr.Dataset] | None = None

        # Spectral indices
        self.spectral_indices: list[str] | None = None

        # Hook for custom post-process function
        self.postprocess_function: Callable[[xr.Dataset], xr.Dataset] | None = None

        # Storage optimization
        self.storage_scale_factor: int | float | None = None
        self.storage_add_offset: int | float | None = None
        self.storage_nodata: int | float | None = None
        self.storage_squeeze_singleton: bool = False
        self.storage_scale_vars_to_skip: list[str] | None = None

    # --- Search ---
    def search(
        self: Self,
        roi: gpd.GeoDataFrame,
        date_range: str | None = None,
        query: dict | None = None,
        filter_function: Callable[[list], list] | None = None,
        use_geoparquet_fallback: bool = True,
    ) -> Self:
        """
        Perform a STAC search using either the API or fallback to STAC GeoParquet if necessary.

        Args:
            roi (gpd.GeoDataFrame): Region of interest for the search.
            date_range (str, optional): Date range for the search (ISO8601 format). Defaults to None.
            query (dict, optional): Additional query parameters. Defaults to None.
            filter_function (Callable, optional): Function to filter the resulting items. Defaults to None.
            use_geoparquet_fallback (bool): Whether to fallback to STAC GeoParquet if API search fails. Defaults to True.

        Returns:
            Updated collection instance with search results.
        """

        self.roi = roi
        self.search_params = {
            "collections": self.collection,
            "intersects": self.roi.to_crs(4326).geometry.item(),
            "datetime": date_range,
            "query": query,
        }

        try:
            # Attempt STAC API search
            search = self.catalog.search(
                **{k: v for k, v in self.search_params.items() if v is not None}
            )
            self.items = list(search.items())

            if not self.items:
                raise ValueError(
                    f"No items found for collection {self} the given search parameters."
                )

        except Exception as e:
            if not use_geoparquet_fallback:
                raise RuntimeError(
                    f"STAC API search for {self} failed and fallback is disabled: {e}"
                ) from e

            # Fallback to GeoParquet
            self.items = self._fallback_to_stac_geoparquet(roi, date_range)

        # Apply filter function if provided
        if filter_function:
            try:
                self.items = filter_function(self.items)
                if not self.items:
                    raise ValueError("Filter function returned no items.")
            except Exception as e:
                raise RuntimeError(f"Error in filter_function: {e}") from e

        # Store data extent
        self.data_extent = self._compute_data_extent(self.items)
        return self

    def _fallback_to_stac_geoparquet(
        self, roi: gpd.GeoDataFrame, date_range: str | None = None
    ) -> list:
        """
        Fallback to STAC GeoParquet for searching items.

        Args:
            roi (gpd.GeoDataFrame): Region of interest for the search.
            date_range (str, optional): Date range for filtering. Defaults to None.

        Returns:
            list: STAC items that match the region of interest and date range.
        """
        collection = self.catalog.get_child(self.collection)
        snapshot = read_snapshot(
            collection,
            columns=None,
            storage_options=None,
        )

        # Spatial filter using sjoin
        snapshot = gpd.sjoin(snapshot, roi[["geometry"]].to_crs(snapshot.crs)).drop(  # type: ignore
            columns="index_right"
        )

        # Date range filter if provided
        if date_range:
            start_date, end_date = date_range.split("/")
            snapshot = snapshot[
                (snapshot["datetime"] >= start_date)
                & (snapshot["datetime"] <= end_date)
            ]

        if snapshot.empty:
            raise ValueError("No items found using GeoParquet fallback.")

        return list(stac_geoparquet.to_item_collection(snapshot))

    def _compute_data_extent(self, items: list) -> gpd.GeoDataFrame:
        """
        Compute the data extent from STAC items.
        """
        return data_extent_from_stac_items(items)

    # --- Load ---
    def load(
        self: Self,
        bands: list[str] | None = None,
        add_metadata_from_stac: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Configure loading parameters.
        """

        resolution = kwargs.pop("resolution", None)
        crs = kwargs.pop("crs", "utm")
        geobox = kwargs.pop("geobox", None)

        if geobox is None and resolution:
            if self.data_extent is None:
                raise ValueError("No data extent found. Perform a search first.")

            geobox = geobox_from_data_extent(
                region=self.roi,
                data_extent=self.data_extent,
                crs=crs,
                resolution=resolution,
            )
            self.odc_stac_load_params["geobox"] = geobox

        self.odc_stac_load_params.update(kwargs)

        if "stac_cfg" not in self.odc_stac_load_params:
            self.odc_stac_load_params["stac_cfg"] = self.stac_cfg

        if bands:
            self.odc_stac_load_params["bands"] = bands

        self.add_metadata_from_stac = add_metadata_from_stac

        return self

    def _load(self) -> xr.Dataset:
        """
        Internal: Load data using odc.stac.load.
        """

        if not self.items:
            raise ValueError("No STAC items found. Perform a search first.")

        if self.percentile:
            self.odc_stac_load_params["groupby"] = "id"

        if not any(
            k in self.odc_stac_load_params
            for k in ["geobox", "bbox", "geopolygon", "like"]
        ):
            bbox = tuple(self.search_params["intersects"].bounds)
            bbox = tuple(self.roi.total_bounds)
            self.odc_stac_load_params["bbox"] = bbox

        ds = odc.stac.load(self.items, **self.odc_stac_load_params)

        if (
            self.add_metadata_from_stac
            and "time" in ds.dims
            and ds.sizes["time"] == len(self.items)
        ):
            ds = self._add_metadata_from_stac(self.items, ds)

        return ds

    @classmethod
    def _add_metadata_from_stac(
        cls, items: list[pystac.Item], ds: xr.Dataset
    ) -> xr.Dataset:
        """
        Attach metadata from STAC items to the dataset as coordinates.
        """
        if len(items) != ds.sizes["time"]:
            raise ValueError("Mismatch between STAC items and dataset time dimension.")

        stac_ids = [i.id for i in items]
        ds = ds.assign_coords({"stac_id": ("time", stac_ids)})
        return ds

    # --- Masking ---
    def mask_and_scale(
        self: Self,
        mask_geometry: odc.geo.geom.Geometry | None = None,
        mask_nodata: bool = True,
        mask_values: list[int] | None = None,
        scale: bool = False,
        scale_factor: float | None = None,
        add_offset: float | None = None,
        scale_vars_to_skip: list[str] | None = None,
    ) -> Self:
        """
        Applies masking and scaling transformations to an Xarray dataset or data array.

        Masking:
        - If `mask_geometry` is provided, masks data outside the given geometry.
        - If `mask_nodata` is True, masks values in `mask_values` (if provided).

        Scaling:
        - If `scale` is True, applies scaling using `scale_factor` and `add_offset`.
        - Variables listed in `scale_vars_to_skip` are excluded from scaling.

        Args:
            mask_geometry (odc.geo.geom.Geometry, optional):
                Geometry to mask the dataset against. If None, no geometric mask is applied.
            mask_nodata (bool, optional):
                If True, masks nodata values. Defaults to True.
            mask_values (list[int], optional):
                List of values to mask as nodata. Applied if `mask_nodata` is True.
            scale (bool, optional):
                If True, applies scaling using `scale_factor` and `add_offset`.
            scale_factor (float, optional):
                Factor by which to scale the data. Defaults to None.
            add_offset (float, optional):
                Offset to add during scaling. Defaults to None.
            scale_vars_to_skip (list[str], optional):
                List of variable names to exclude from scaling. Defaults to None.

        Returns BaseCollection with set params:
        """

        self.mask_geometry = mask_geometry
        self.mask_nodata = mask_nodata
        self.mask_values = mask_values
        self.scale = scale

        # Scale options
        self.scale_factor = scale_factor
        self.add_offset = add_offset
        self.scale_vars_to_skip = scale_vars_to_skip or []

        return self

    def _apply_masks_and_scale(self, ds: xr.Dataset) -> xr.Dataset:
        # Apply pre-load masks
        if self.mask_geometry:
            crs = ds.rio.crs
            if not crs:
                msg = "Dataset must have a CRS to apply geometry mask."
                raise ValueError(msg)

            geometry = self.mask_geometry.to_crs(crs)
            ds = geometry_mask(ds, geometry)  # type: ignore

        if self.mask_nodata:
            mask = nodata_mask(ds)
            apply_mask_with_attrs = keep_rio_attrs(
                exclude_attrs={"nodata": [], "encoding": ["_FillValue"]}
            )(apply_mask)
            ds = apply_mask_with_attrs(ds, mask)  # type: ignore
            # NOTE: we used to set the nodata value to np.nan before, but I think this is bad practice
            # ds = set_nodata(ds, np.nan)  # type: ignore

        if self.mask_values:
            mask = numeric_mask(ds, self.mask_values)
            # ds = apply_mask(ds, mask)  # type: ignore

            apply_mask_with_attrs = keep_rio_attrs()(apply_mask)
            ds = apply_mask_with_attrs(ds, mask)  # type: ignore

        if self.scale:
            ds = unscale(
                ds,
                self.scale_factor,
                self.add_offset,
                keep_attrs=True,
                variables_to_ignore=self.scale_vars_to_skip,
            )  # type: ignore

        return ds

    # --- Compositing ---
    def composite(
        self: Self,
        method: str = "simple",
        percentile: int = 50,
        custom_func: Callable[[xr.Dataset], xr.Dataset] | None = None,
        filter_function: Callable[[list], list] | None = None,
    ) -> Self:
        """
        Configure compositing options.
        """
        if not (0 <= percentile <= 100):
            msg = "Percentile must be between 0 and 100."
            raise ValueError(msg)

        if not self.items:
            raise ValueError("No STAC items found. Perform a search first.")

        self.composite_method = method
        self.percentile = percentile
        self.custom_composite_func = custom_func

        if filter_function:
            try:
                self.items = filter_function(self.items)

                if not self.items:
                    raise ValueError("Filter function returned no items.")

            except Exception as e:
                msg = f"Error in filter_function: {e}"
                raise RuntimeError(msg) from e

        return self

    @classmethod
    def _extract_composite_metadata(cls, data: xr.Dataset | xr.DataArray) -> dict:
        """
        Extract composite metadata from the dataset.
        """
        datetimes = data.time.to_series().sort_values()
        start_datetime = datetimes.min().isoformat()
        end_datetime = datetimes.max().isoformat()
        avg_interval = datetimes.diff().mean()
        n_obs = len(datetimes)
        return {
            "datetime": start_datetime,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "eo:cloud_cover": str(int(data["eo:cloud_cover"].mean().item())),
            "composite:avg_interval": avg_interval,
            "composite:n_obs": n_obs,
            "composite:stac_ids": str([str(i) for i in data.stac_id.values]),
        }

    @classmethod
    def _simple_composite(
        cls, ds: xr.DataArray | xr.Dataset
    ) -> xr.DataArray | xr.Dataset:
        """
        Generate a simple median composite dataset.
        """
        try:
            composite = ds.median(dim="time", skipna=True, keep_attrs=True)
            metadata = cls._extract_composite_metadata(ds)

            # NOTE: we could consider to use the starting point of the datetime range
            # as the time. That would make it easier to make homogeneous composites
            # in the next processing step.
            composite = composite.assign_coords(
                {
                    "time": metadata["datetime"],
                    "start_datetime": metadata["start_datetime"],
                    "end_datetime": metadata["end_datetime"],
                }
            )

            composite.attrs.update(metadata)
            composite.attrs.update(
                {
                    "composite:determination_method": "Simple Median Composite",
                    "composite:summary": (
                        "Composite dataset created by taking the median value of each pixel "
                        "across all time steps."
                    ),
                }
            )
            return composite
        except Exception as e:
            raise RuntimeError(f"Failed to generate simple composite: {e}") from e

    def _apply_composite(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Internal: Apply the specified composite method.
        """
        composite_map = {
            "simple": self._simple_composite,
            "percentile": lambda ds: ds.reduce(
                np.percentile, q=self.percentile, dim="time"
            ),
            "custom": self.custom_composite_func,
        }
        if (
            self.composite_method not in composite_map
            or not composite_map[self.composite_method]
        ):
            raise ValueError(f"Unsupported composite method: {self.composite_method}")
        return composite_map[self.composite_method](ds)

    # --- Spectral Indices ---
    def add_spectral_indices(self: Self, indices: list[str]) -> Self:
        """
        Add spectral indices to compute.
        """
        self.spectral_indices = indices
        return self

    @classmethod
    def _compute_spectral_indices(cls, ds: xr.Dataset, spectral_indices) -> xr.Dataset:
        """
        Internal: Compute spectral indices.
        """
        if isinstance(ds, xr.DataArray):
            try:
                ds = ds.to_dataset("band")
            except Exception as e:
                msg = "Cannot convert DataArray to Dataset: {e}"
                raise ValueError(msg) from e

        calculate_indices_with_attrs = keep_rio_attrs()(calculate_indices)
        ds = calculate_indices_with_attrs(ds, spectral_indices, normalize=False)

        return ds

    def postprocess(self: Self, function: Callable[[xr.Dataset], xr.Dataset]) -> Self:
        """
        Set a custom postprocessing function.

        Args:
            function (Callable[[xr.Dataset], xr.Dataset]): A function to postprocess the dataset.

        Returns:
            BaseCollection: The updated instance with the postprocessing function set.
        """
        self.postprocess_function = function
        return self

    @classmethod
    def _postprocess(
        cls, ds: xr.Dataset, function: Callable[[xr.Dataset], xr.Dataset]
    ) -> xr.Dataset:
        """
        Apply the custom postprocessing function to the dataset.

        Args:
            ds (xr.Dataset): The dataset to be processed.
            function (Callable[[xr.Dataset], xr.Dataset]): The postprocessing function.

        Returns:
            xr.Dataset: The processed dataset.
        """
        try:
            return function(ds)
        except Exception as e:
            raise RuntimeError(f"Postprocessing function failed: {e}") from e

    def optimize_for_storage(
        self: Self,
        scale_factor: int | float | None = None,
        add_offset: int | float | None = None,
        nodata: int | float | None = None,
        squeeze_singleton: bool = False,
        scale_vars_to_skip: list[str] | None = None,
    ) -> Self:
        """
        Configure storage optimization options for the dataset.

        Args:
            scale_factor (int | float | None): Scale factor for storage optimization.
            add_offset (int | float | None): Add offset for storage optimization.
            nodata (int | float | None): Nodata value for storage optimization.
            squeeze_singleton (bool): Whether to remove singleton dimensions.
            scale_vars_to_skip (list[str] | None): Variables to ignore when scaling.

        Returns:
            BaseCollection: The updated collection instance.
        """
        if (scale_factor or add_offset) and nodata is None:
            raise ValueError("Nodata value must be provided when scaling data.")

        self.storage_scale_factor = scale_factor
        self.storage_add_offset = add_offset
        self.storage_nodata = nodata
        self.storage_squeeze_singleton = squeeze_singleton
        self.storage_scale_vars_to_skip = scale_vars_to_skip
        return self

    @classmethod
    def _optimize_for_storage(
        cls,
        ds: xr.Dataset,
        scale_factor: int | float | None = None,
        add_offset: int | float | None = None,
        nodata: int | float | None = None,
        squeeze_singleton: bool = False,
        scale_vars_to_skip: list[str] | None = None,
    ) -> xr.Dataset:
        """
        Optimize the dataset for storage by applying scaling and removing singleton dimensions.

        Args:
            ds (xr.Dataset): Input dataset to optimize.
            scale_factor (int | float | None): Scale factor for storage optimization.
            add_offset (int | float | None): Add offset for storage optimization.
            nodata (int | float | None): Nodata value for storage optimization.
            squeeze_singleton (bool): Whether to remove singleton dimensions.
            scale_vars_to_skip (list[str] | None): Variables to ignore when scaling.

        Returns:
            xr.Dataset: Optimized dataset.
        """
        # Squeeze singleton dimensions if requested
        if squeeze_singleton:
            ds = ds.squeeze(drop=True)  # Drop singleton dimensions explicitly

        # Apply scaling if parameters are provided
        if scale_factor is not None:
            if nodata is None:
                raise ValueError("Nodata value must be provided when scaling data.")

            scale_vars_to_skip = scale_vars_to_skip or []

            ds = scale(
                ds,
                scale_factor=scale_factor,
                add_offset=add_offset,
                nodata=nodata,
                keep_attrs=True,
                scale_vars_to_skip=scale_vars_to_skip,
            )  # type: ignore

        # Return the optimized dataset
        return ds

    # --- Execution ---
    def execute(self, compute: bool = False) -> xr.Dataset:
        """
        Execute the workflow: load, mask, composite, and compute indices.
        """
        if not self.items:
            self.search(self.roi, self.search_params.get("datetime", ""))

        self.dataset = self._load()

        if any([self.mask_geometry, self.mask_nodata, self.mask_values]):
            self.dataset = self._apply_masks_and_scale(self.dataset)

        if self.composite_method:
            self.dataset = self._apply_composite(self.dataset)

        if self.spectral_indices:
            self.dataset = self._compute_spectral_indices(
                self.dataset, self.spectral_indices
            )

        if self.postprocess_function:
            self.dataset = self._postprocess(self.dataset, self.postprocess_function)

        if self.storage_scale_factor or self.storage_squeeze_singleton:
            self.dataset = self._optimize_for_storage(
                ds=self.dataset,
                scale_factor=self.storage_scale_factor,
                add_offset=self.storage_add_offset,
                nodata=self.storage_nodata,
                squeeze_singleton=self.storage_squeeze_singleton,
                scale_vars_to_skip=self.storage_scale_vars_to_skip,
            )

        if compute:
            self.dataset = self.dataset.compute()

        return self.dataset


class S2Collection(BaseCollection):
    """
    A class to manage Sentinel-2 collections from the Planetary Computer catalog.
    """

    default_stac_cfg = {
        "sentinel-2-l2a": {
            "assets": {
                "*": {"data_type": "float32", "nodata": np.nan},
                "SCL": {"data_type": "float32", "nodata": np.nan},
                "visual": {"data_type": None, "nodata": np.nan},
            },
        },
        "*": {"warnings": "ignore"},
    }

    def __init__(
        self,
        catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection: str = "sentinel-2-l2a",
        stac_cfg: dict | None = None,
    ):
        stac_cfg = stac_cfg or self.default_stac_cfg
        super().__init__(catalog_url, collection, stac_cfg)

    def search(
        self: Self,
        roi: gpd.GeoDataFrame,
        date_range: str | None = None,
        query: dict | None = None,
        item_limit: int = 300,
        filter_function: Callable[[list], list] | None = None,
    ) -> Self:
        """
        Perform a STAC search using the API, retrieving items sorted by ascending cloud cover
        instead of filtering by a strict cloud cover threshold.

        Args:
            roi (gpd.GeoDataFrame): Region of interest.
            date_range (str, optional): Date range for the search (ISO8601 format).
            filter_function (Callable, optional): Function to filter the resulting items.
            use_geoparquet_fallback (bool): Whether to fallback to STAC GeoParquet.
            item_limit (int, optional): Maximum number of items to retrieve. Default is 300.

        Returns:
            Updated collection instance with search results.
        """

        self.roi = roi

        self.search_params = {
            "collections": self.collection,
            "intersects": self.roi.to_crs(4326).geometry.item(),
            "datetime": date_range,
            "query": query,
            "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
            "limit": item_limit,
        }

        # Attempt STAC API search
        search = self.catalog.search(
            **{k: v for k, v in self.search_params.items() if v is not None}
        )
        self.items = list(search.items())

        if not self.items:
            raise ValueError("No items found for the given search parameters.")

        # Apply filter function if provided
        if filter_function:
            try:
                self.items = filter_function(self.items)
                if not self.items:
                    raise ValueError("Filter function returned no items.")
            except Exception as e:
                raise RuntimeError(f"Error in filter_function: {e}") from e

        # Store data extent
        self.data_extent = self._compute_data_extent(self.items)
        return self

    @classmethod
    def _add_metadata_from_stac(
        cls, items: list[pystac.Item], ds: xr.Dataset
    ) -> xr.Dataset:
        """
        Attach metadata from STAC items to the dataset as coordinates.
        """
        if len(items) != ds.sizes["time"]:
            raise ValueError("Mismatch between STAC items and dataset time dimension.")

        mgrs_tiles = [i.properties["s2:mgrs_tile"] for i in items]
        cloud_cover = [i.properties["eo:cloud_cover"] for i in items]
        rel_orbits = [i.properties["sat:relative_orbit"] for i in items]
        stac_ids = [i.id for i in items]

        ds = ds.assign_coords({"stac_id": ("time", stac_ids)})
        ds = ds.assign_coords({"s2:mgrs_tile": ("time", mgrs_tiles)})
        ds = ds.assign_coords({"eo:cloud_cover": ("time", cloud_cover)})
        ds = ds.assign_coords({"sat:relative_orbit": ("time", rel_orbits)})
        return ds

    # --- Masking ---
    def mask_and_scale(
        self: Self,
        mask_geometry: odc.geo.geom.Geometry | None = None,
        mask_nodata: bool = True,
        mask_values: list[int] | None = None,
        scale: bool = False,
        scale_factor: float | None = None,
        add_offset: float | None = None,
        scale_vars_to_skip: list[str] | None = None,
        mask_scl: list[str | SceneClassification | int] | None = None,
    ) -> Self:
        """
        Applies masking and scaling transformations to a Sentinel-2 dataset.

        Masking:
        - If `mask_geometry` is provided, masks data outside the given geometry.
        - If `mask_nodata` is True, masks values in `mask_values` (if provided).
        - If `mask_scl` is provided, masks specific Sentinel-2 Scene Classification Layer (SCL) values.

        Scaling:
        - If `scale` is True, applies scaling using `scale_factor` and `add_offset`.
        - Variables listed in `scale_vars_to_skip` are excluded from scaling.

        Args:
            mask_geometry (odc.geo.geom.Geometry, optional):
                Geometry to mask the dataset against. If None, no geometric mask is applied.
            mask_nodata (bool, optional):
                If True, masks nodata values. Defaults to True.
            mask_values (list[int], optional):
                List of values to mask as nodata. Applied if `mask_nodata` is True.
            scale (bool, optional):
                If True, applies scaling transformations.
            scale_factor (float, optional):
                Factor by which to scale the data. Defaults to None.
            add_offset (float, optional):
                Offset to add during scaling. Defaults to None.
            scale_vars_to_skip (list[str], optional):
                List of variable names to exclude from scaling. Defaults to None.
            mask_scl (list[str | SceneClassification | int], optional):
                List of Sentinel-2 Scene Classification Layer (SCL) class names or numeric values to mask. Defaults to None.

        Returns:
            S2Collection: The updated instance with masking and scaling applied.

        Valid SCL classes:
            - NO_DATA
            - SATURATED_DEFECTIVE
            - DARK_AREA_PIXELS
            - CLOUD_SHADOWS
            - VEGETATION
            - BARE_SOILS
            - WATER
            - CLOUDS_LOW_PROBABILITY
            - CLOUDS_MEDIUM_PROBABILITY
            - CLOUDS_HIGH_PROBABILITY
            - CIRRUS
            - SNOW_ICE
        """
        super().mask_and_scale(
            mask_geometry=mask_geometry,
            mask_nodata=mask_nodata,
            mask_values=mask_values,
            scale=scale,
            scale_factor=scale_factor,
            add_offset=add_offset,
            scale_vars_to_skip=scale_vars_to_skip,
        )
        self.mask_scl = mask_scl
        return self

    def _apply_masks_and_scale(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply masks to the dataset.
        """
        ds = super()._apply_masks_and_scale(ds)

        if self.mask_scl:
            # NOTE/BUG: it is possible that the mask will upgrade the data type to float32 and with
            # that the nodata value will be changed to np.nan, while the rio attribute will remain the same.
            # This is for now not an issue, but is noted here because it might eventually happen with a very
            # specific use case. Then the fix would be to use the keep_rio_attrs decorator with the right args.
            mask = scl_mask(ds, self.mask_scl)
            ds = apply_mask(ds, mask)  # type: ignore

        return ds


class S2CompositeCollection(BaseCollection):
    """
    A class to manage Sentinel-2 composite collections from a STAC-based catalog.
    """

    default_stac_cfg = {
        "*": {"warnings": "ignore"},
    }

    def __init__(
        self,
        catalog_url: str = "https://coclico.blob.core.windows.net/stac/v1/catalog.json",
        collection: str = "s2-l2a-composite",
        stac_cfg: dict | None = None,
    ):
        """
        Initialize the S2CompositeCollection.

        Args:
            catalog_url (str): URL to the STAC catalog.
            collection (str): Name of the collection in the catalog.
            stac_cfg (dict, optional): Configuration for STAC handling. Defaults to None.
        """
        super().__init__(catalog_url, collection, stac_cfg)
        self.percentile = None  # Disable percentile compositing for this collection.
        self.composite_method = None  # Disable composite method for this collection.
        self.merge = None

    # --- Merge
    def merge_overlapping_tiles(self: Self, merge=True) -> Self:
        self.merge = merge
        return self

    def _merge_overlapping_tiles(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Merge overlapping tiles in the dataset.
        """

        def _median(ds, mask):
            return ds.where(~mask).median("time", skipna=True, keep_attrs=True)

        if not self.merge:
            return ds

        # Get datetimes from the dataset before we lose them in the median computation
        datetimes = get_datetimes(ds)

        mask = nodata_mask(ds)

        apply_median_with_attrs = keep_rio_attrs(
            exclude_attrs={"nodata": [], "encoding": ["_FillValue"]}
        )(_median)
        ds = apply_median_with_attrs(ds, mask)  # type: ignore
        # NOTE: we used to set the nodata value to np.nan before, but I think this is bad practice
        # ds = set_nodata(ds, np.nan)  # type: ignore

        # Update time dimension with the median datetime
        if datetimes:
            ds = update_time_coord(ds, datetimes)

        return ds

    # --- Execute ---
    def execute(self, compute=False) -> xr.Dataset:
        """
        Apply masks to the dataset.
        """
        ds = super().execute(compute=compute)

        if self.merge:
            ds = self._merge_overlapping_tiles(ds)

        return ds


class DeltaDTMCollection(BaseCollection):
    """
    A class to manage DeltaDTM from a STAC-based catalog.
    """

    default_stac_cfg = {
        "*": {"warnings": "ignore"},
    }

    def __init__(
        self,
        catalog_url: str = "https://coclico.blob.core.windows.net/stac/v1/catalog.json",
        collection: str = "deltares-delta-dtm",
        stac_cfg: dict | None = None,
    ):
        """
        Initialize the S2CompositeCollection.

        Args:
            catalog_url (str): URL to the STAC catalog.
            collection (str): Name of the collection in the catalog.
            stac_cfg (dict, optional): Configuration for STAC handling. Defaults to None.
        """
        super().__init__(catalog_url, collection, stac_cfg)
        self.percentile = None  # Disable percentile compositing for this collection.
        self.composite_method = None  # Disable composite method for this collection.
        self.merge = None

    def search(
        self: Self,
        roi: gpd.GeoDataFrame,
        filter_function: Callable[[list], list] | None = None,
    ) -> Self:
        """
        Perform a search specific to the DeltaDTM collection.

        Args:
            roi (GeoDataFrame): Region of interest.
            query (dict, optional): STAC query parameters.
            filter_function (Callable, optional): Filter to apply to search results.

        Returns:
            Self: The updated collection instance with results.
        """
        return super().search(
            roi=roi,
            date_range=None,
            query=None,
            filter_function=filter_function,
            use_geoparquet_fallback=True,
        )

    @staticmethod
    def postprocess_deltadtm(ds: xr.Dataset) -> xr.Dataset:
        # Squeeze time dimension
        # ds = ds.squeeze(drop=True)

        # if "stac_id" in ds:
        #     ds = ds.drop_vars("stac_id")

        # Replace nodata with 0 because its a DTM.
        NEW_NODATA = 0
        mask = nodata_mask(ds)

        def _replace_nodata_with_zero(ds):
            return ds.where(~mask, NEW_NODATA)

        apply_replace_nodata_with_zero_with_attrs = keep_rio_attrs(
            exclude_attrs={"nodata": [], "encoding": ["_FillValue"]}
        )(_replace_nodata_with_zero)

        ds = apply_replace_nodata_with_zero_with_attrs(ds)  # type: ignore

        # NOTE: we don't have to set the nodata value because its actually an elevation.
        # ds = set_nodata(ds, NEW_NODATA)  # type: ignore

        return ds


class CopernicusDEMCollection(BaseCollection):
    """
    A class to manage CopernicusDEM from a STAC-based catalog.
    """

    default_stac_cfg = {
        "*": {"warnings": "ignore"},
    }

    def __init__(
        self,
        catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection: str = "cop-dem-glo-30",
        stac_cfg: dict | None = None,
    ):
        """
        Initialize the CopernicusDEMCollection.

        Args:
            catalog_url (str): URL to the STAC catalog.
            collection (str): Name of the collection in the catalog.
            stac_cfg (dict, optional): Configuration for STAC handling. Defaults to None.
        """
        super().__init__(catalog_url, collection, stac_cfg)
        self.percentile = None  # Disable percentile compositing for this collection.
        self.composite_method = None  # Disable composite method for this collection.
        self.merge = None

    def search(
        self: Self,
        roi: gpd.GeoDataFrame,
        query: dict | None = None,
        filter_function: Callable[[list], list] | None = None,
    ) -> Self:
        """
        Perform a search specific to the CopernicusDEM collection.

        Args:
            roi (GeoDataFrame): Region of interest.
            query (dict, optional): STAC query parameters.
            filter_function (Callable, optional): Filter to apply to search results.

        Returns:
            Self: The updated collection instance with results.
        """
        return super().search(
            roi=roi,
            date_range=None,
            query=query,
            filter_function=filter_function,
            use_geoparquet_fallback=False,
        )

    @staticmethod
    def postprocess_cop_dem30(ds: xr.Dataset) -> xr.Dataset:
        # Squeeze time dimension
        # ds = ds.squeeze(drop=True)
        # if "stac_id" in ds:
        #     ds = ds.drop_vars("stac_id")
        return ds
