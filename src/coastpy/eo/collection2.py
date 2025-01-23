import abc
import logging
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import odc.stac
import pystac
import xarray as xr

from coastpy.eo.indices import calculate_indices
from coastpy.eo.mask import (
    apply_mask,
    geometry_mask,
    nodata_mask,
    numeric_mask,
)
from coastpy.eo.utils import data_extent_from_stac_items

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseCollection(abc.ABC):  # noqa: B024
    """
    Base class for managing STAC-based collections.
    Provides core functionality for searching, loading, masking, and processing datasets.
    """

    def __init__(
        self,
        catalog_url: str,
        collection: str,
        stac_cfg: dict | None = None,
    ):
        self.catalog_url = catalog_url
        self.collection = collection
        self.catalog = odc.stac.Client.open(self.catalog_url)
        self.stac_cfg = stac_cfg or {}

        # Internal state
        self.region_of_interest: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.search_params: dict = {}
        self.odc_load_params: dict = {}
        self.stac_items: list | None = None
        self.dataset: xr.Dataset | None = None

        # Masking options
        self.geometry_mask: Any | None = None
        self.nodata_mask: bool = False
        self.value_mask: list[int] | None = None

        # Composite options
        self.composite_method: str | None = None
        self.percentile: int | None = None
        self.custom_composite_func: Callable[[xr.Dataset], xr.Dataset] | None = None

    def search(
        self,
        region_of_interest: gpd.GeoDataFrame,
        datetime_range: str,
        query: dict | None = None,
        filter_function: Callable[[list[pystac.Item]], list[pystac.Item]] | None = None,
    ) -> "BaseCollection":
        """
        Perform a generic STAC search and store the results.

        Args:
            region_of_interest (gpd.GeoDataFrame): Region of interest.
            datetime_range (str): Temporal range in 'YYYY-MM-DD/YYYY-MM-DD' format.
            query (dict, optional): Additional query parameters.

        Returns:
            BaseCollection: Updated instance with search results.
        """
        self.region_of_interest = region_of_interest
        self.search_params = {
            "collections": self.collection,
            "intersects": self.region_of_interest.to_crs(4326).geometry.item(),
            "datetime": datetime_range,
            "query": query,
        }

        search = self.catalog.search(**self.search_params)
        self.stac_items = list(search.items())

        if not self.stac_items:
            raise ValueError("No items found for the given search parameters.")

        if filter_function:
            try:
                logging.info("Applying custom filter function.")
                self.stac_items = filter_function(self.stac_items)
                if not self.stac_items:
                    raise ValueError("Filter function returned no items.")

            except Exception as e:
                msg = f"Error in filter_function: {e}"
                raise RuntimeError(msg)  # noqa: B904

        self.data_extent = data_extent_from_stac_items(self.stac_items)

        return self

    def load(
        self,
        bands: list[str] | None = None,
        percentile: int | None = None,
        spectral_indices: list[str] | None = None,
        mask_nodata: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> "BaseCollection":
        """
        Configure parameters for loading data via odc.stac.load.

        Args:
            bands (list[str], optional): Bands to load.
            percentile (int | None, optional): Percentile for compositing (e.g., 50 for median).
            spectral_indices (list[str] | None, optional): List of spectral indices to compute.
            mask_nodata (bool, optional): Mask no-data values. Defaults to True.
            normalize (bool, optional): Normalize data. Defaults to True.
            **kwargs: Additional parameters passed to odc.stac.load.

        Returns:
            BaseCollection: Updated instance with load parameters configured.
        """
        if percentile is not None and not (0 <= percentile <= 100):
            raise ValueError("`percentile` must be between 0 and 100.")

        self.odc_load_params.update(kwargs)

        if bands is not None:
            self.odc_load_params["bands"] = bands
        if percentile is not None:
            self.percentile = percentile
        if spectral_indices is not None:
            self.spectral_indices = spectral_indices
        self.mask_nodata = mask_nodata
        self.normalize = normalize

        return self

    def _load(self) -> xr.Dataset:
        """
        Load data using odc.stac.load.

        Returns:
            xr.Dataset: Loaded dataset.
        """
        if not self.stac_items:
            raise ValueError("No STAC items found. Perform a search first.")

        # Adjust groupby for percentile-based compositing
        if self.percentile:
            self.odc_load_params["groupby"] = "id"

        # Fallback to bbox if no spatial bounds are provided
        if (
            not self.odc_load_params.get("geobox")
            and not self.odc_load_params.get("bbox")
            and not self.odc_load_params.get("geopolygon")
            and not self.odc_load_params.get("like")
        ):
            bbox = tuple(self.search_params["intersects"].bounds)
            self.odc_load_params["bbox"] = bbox

        ds = odc.stac.load(
            self.stac_items,
            **self.odc_load_params,
        )

        return ds  # type: ignore

    def mask(
        self,
        geometry: Any | None = None,
        nodata: bool = True,
        values: list[int] | None = None,
    ) -> "BaseCollection":
        """
        Configure masking options.

        Args:
            geometry (Any, optional): Geometry to mask data within.
            nodata (bool): Whether to apply a nodata mask.
            values (list[int], optional): Specific values to mask.

        Returns:
            BaseCollection: Updated instance with masking options configured.
        """
        self.geometry_mask = geometry
        self.nodata_mask = nodata
        self.value_mask = values
        return self

    def _apply_masks(self, ds: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        # Apply pre-load masks
        if self.geometry_mask:
            crs = ds.rio.crs
            if not crs:
                msg = "Dataset must have a CRS to apply geometry mask."
                raise ValueError(msg)
            geometry = self.geometry_mask.to_crs(crs)
            ds = geometry_mask(ds, geometry)
            return ds

        if self.nodata_mask:
            mask = nodata_mask(ds)
            ds = apply_mask(ds, mask)

        if self.value_mask:
            mask = numeric_mask(ds, self.value_mask)
            ds = apply_mask(ds, mask)

        return ds

    def composite(
        self,
        method: str = "simple",
        percentile: int = 50,
        custom_composite_func: Callable[[xr.Dataset], xr.Dataset] | None = None,
    ) -> "BaseCollection":
        """
        Configure composite options.

        Args:
            method (str): Composite method (simple or custom).
            percentile (int, optional): Percentile for compositing. Defaults to 50.
            custom_composite_func (Callable, optional): Custom composite function.

        Returns:
            BaseCollection: Updated instance with composite options configured.
        """
        self.composite_method = method
        self.percentile = percentile
        self.custom_composite_func = custom_composite_func
        return self

    def _default_composite(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Default composite method: median or percentile over the time dimension.

        Args:
            ds (xr.Dataset): Input dataset.

        Returns:
            xr.Dataset: Composite dataset.
        """
        if self.percentile == 50:
            return ds.median(dim="time")
        else:
            return ds.reduce(np.percentile, q=self.percentile, dim="time")

    def _apply_composite(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply the configured composite method to the dataset.

        Args:
            ds (xr.Dataset): Input dataset.

        Returns:
            xr.Dataset: Composite dataset.
        """
        if self.custom_composite_func:
            return self.custom_composite_func(ds)
        else:
            return self._default_composite(ds)

    def add_spectral_indices(self, indices: list[str]) -> "BaseCollection":
        """
        Add spectral indices to the dataset.

        Args:
            indices (list[str]): List of spectral indices to compute.

        Returns:
            BaseCollection: Updated instance with computed indices.
        """
        self.spectral_indices = indices
        return self

    def _compute_spectral_indices(
        self, ds: xr.Dataset, indices: list[str]
    ) -> xr.Dataset:
        """
        Compute spectral indices for the dataset.

        Args:
            ds (xr.Dataset): Input dataset.
            indices (list[str]): List of spectral indices to compute.

        Returns:
            xr.Dataset: Dataset with spectral indices added.
        """
        for index in indices:
            # Example implementation for NDVI; extend as needed
            if index == "NDVI" and {"nir", "red"}.issubset(ds.data_vars):
                ds[index] = (ds["nir"] - ds["red"]) / (ds["nir"] + ds["red"])
        return ds

    def execute(self, compute: bool = False) -> xr.Dataset:
        """
        Execute the workflow: load, apply masks, and compute indices.

        Args:
            compute (bool): Whether to trigger computation for lazy datasets.

        Returns:
            xr.Dataset: Processed dataset.
        """
        if self.stac_items is None:
            search = self.catalog.search(**self.search_params)
            self.stac_items = list(search.items())

        if self.dataset is None:
            self.dataset = self._load()

        if self.geometry_mask or self.nodata_mask or self.value_mask:
            self.dataset = self._apply_masks(self.dataset)

        if self.composite_method:
            if self.composite_method == "simple":
                self.dataset = self._simple_composite(self.dataset)
            elif self.composite_method == "grouped" and self.percentile:
                self.dataset = self._grouped_composite(self.dataset, self.percentile)
            else:
                raise ValueError(
                    f"Unsupported composite method: {self.composite_method}"
                )

        if self.spectral_indices:
            if isinstance(self.dataset, xr.DataArray):
                try:
                    ds = self.dataset.to_dataset("band")
                    self.dataset = ds
                except Exception as e:
                    msg = "Cannot convert DataArray to Dataset: {e}"
                    raise ValueError(msg) from e
                msg = "Spectral indices not implemented for DataArray."
                raise NotImplementedError(msg)

            self.dataset = calculate_indices(
                self.dataset, self.spectral_indices, normalize=False
            )

        if compute:
            self.dataset = self.dataset.compute()

        return self.dataset

        return self.dataset
