import abc
import logging
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import odc.stac
import pystac
import xarray as xr

from coastpy.eo.utils import data_extent_from_stac_items

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseCollection(abc.ABC):
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
        self.search_params: dict = {}
        self.odc_load_params: dict = {}
        self.items: list | None = None
        self.dataset: xr.Dataset | None = None

        # Masking options
        self.geometry_mask: Any | None = None
        self.nodata_mask: bool = False
        self.value_mask: list[int] | None = None

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
        geom = region_of_interest.to_crs(4326).geometry.item()
        self.search_params = {
            "collections": self.collection,
            "intersects": geom,
            "datetime": datetime_range,
            "query": query,
        }

        logger.info(f"Searching with parameters: {self.search_params}")
        search = self.catalog.search(**self.search_params)
        self.items = list(search.items())

        if not self.items:
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

        logger.info(f"Found {len(self.items)} items.")

        self.data_extent = data_extent_from_stac_items(self.items)

        return self

    def load(
        self,
        bands: list[str],
        **kwargs,
    ) -> xr.Dataset:
        """
        Load data from the STAC items using odc.stac.load.

        Args:
            bands (list[str]): Bands to load.
            kwargs: Additional parameters for odc.stac.load.

        Returns:
            xr.Dataset: Loaded dataset.
        """
        if not self.items:
            raise ValueError("No items found. Perform a search first.")

        self.odc_load_params.update(kwargs)
        self.dataset = odc.stac.load(
            self.items,
            bands=bands,
            **self.odc_load_params,
        )
        return self.dataset

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

    def _apply_masks(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply configured masks to the dataset.

        Args:
            ds (xr.Dataset): Dataset to mask.

        Returns:
            xr.Dataset: Masked dataset.
        """
        if self.geometry_mask:
            crs = ds.rio.crs
            if not crs:
                raise ValueError("Dataset must have a CRS to apply geometry mask.")
            ds = ds.rio.clip(self.geometry_mask.geometry, crs=crs)

        if self.nodata_mask:
            ds = ds.where(ds != ds.attrs.get("nodata", np.nan))

        if self.value_mask:
            for value in self.value_mask:
                ds = ds.where(ds != value)

        return ds

    def add_spectral_indices(self, indices: list[str]) -> "BaseCollection":
        """
        Add spectral indices to the dataset.

        Args:
            indices (list[str]): List of spectral indices to compute.

        Returns:
            BaseCollection: Updated instance with computed indices.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Perform `load` first.")
        self.dataset = self._compute_spectral_indices(self.dataset, indices)
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
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Perform `load` first.")

        self.dataset = self._apply_masks(self.dataset)

        if compute:
            self.dataset = self.dataset.compute()

        return self._post_process(self.dataset)

    @abc.abstractmethod
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Abstract method for custom post-processing. Must be implemented by subclasses.

        Args:
            ds (xr.Dataset): Loaded dataset.

        Returns:
            xr.Dataset: Processed dataset.
        """
