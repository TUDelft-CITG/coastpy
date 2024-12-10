import abc
import logging
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import odc.geo.geom
import odc.stac
import pyproj
import pystac
import pystac_client
import rioxarray  # noqa
import stac_geoparquet
import xarray as xr

from coastpy.eo.indices import calculate_indices
from coastpy.stac.utils import read_snapshot
from coastpy.utils.xarray import get_nodata, set_nodata


class ImageCollection:
    """
    A generic class to manage image collections from a STAC-based catalog.
    """

    def __init__(
        self,
        catalog_url: str,
        collection: str,
        stac_cfg: dict | None = None,
    ):
        self.catalog_url = catalog_url
        self.collection = collection
        self.catalog = pystac_client.Client.open(self.catalog_url)
        self.clip = None

        # Configuration
        self.search_params = {}
        self.bands = []
        self.spectral_indices = []
        self.percentile = None
        self.dst_crs = None
        self.load_params = {}
        self.stac_cfg = stac_cfg or {}

        # Internal state
        self.items = None
        self.dataset = None

    def search(
        self,
        roi: gpd.GeoDataFrame,
        datetime_range: str,
        query: dict | None = None,
        filter_function: Callable[[list[pystac.Item]], list[pystac.Item]] | None = None,
    ) -> "ImageCollection":
        """
        Search the catalog for items and optionally apply a filter function.

        Args:
            roi (gpd.GeoDataFrame): Region of interest.
            datetime_range (str): Temporal range in 'YYYY-MM-DD/YYYY-MM-DD'.
            query (dict, optional): Additional query parameters for search.
            filter_function (Callable, optional): A function to filter/sort items.
                Accepts and returns a list of pystac.Items.

        Returns:
            ImageCollection: Updated instance with items populated.
        """
        geom = roi.to_crs(4326).geometry.item()
        self.search_params = {
            "collections": self.collection,
            "intersects": geom,
            "datetime": datetime_range,
            "query": query,
        }
        self.geometry = odc.geo.geom.Geometry(geom)

        # Perform the actual search
        logging.info(f"Executing search with params: {self.search_params}")
        search = self.catalog.search(**self.search_params)
        self.items = list(search.items())

        # Check if items were found
        if not self.items:
            msg = "No items found for the given search parameters."
            raise ValueError(msg)

        # Apply the filter function if provided
        if filter_function:
            try:
                logging.info("Applying custom filter function.")
                self.items = filter_function(self.items)
            except Exception as e:
                msg = f"Error in filter_function: {e}"
                raise RuntimeError(msg)  # noqa: B904

        return self

    def load(
        self,
        bands: list[str],
        percentile: int | None = None,
        spectral_indices: list[str] | None = None,
        chunks: dict[str, int | str] | None = None,
        groupby: str = "solar_day",
        resampling: str | dict[str, str] | None = None,
        dtype: np.dtype | str | None = None,
        crs: str | int | None = None,
        resolution: float | int | None = None,
        pool: int | None = None,
        preserve_original_order: bool = False,
        progress: bool | None = None,
        fail_on_error: bool = True,
        geobox: dict | None = None,
        like: xr.Dataset | None = None,
        patch_url: str | None = None,
        dst_crs: Any | None = None,
        clip: bool | None = None,
    ) -> "ImageCollection":
        """
        Configure loading parameters.

        Args:
            bands (List[str]): Bands to load.
            percentile (int | None): Percentile value for compositing (e.g., 50 for median).
            spectral_indices (List[str]): Spectral indices to calculate.
            Additional args: Parameters for odc.stac.load.

        Returns:
            ImageCollection: Updated instance.

        """
        if percentile is not None and not (0 <= percentile <= 100):
            msg = "Composite percentile must be between 0 and 100."
            raise ValueError(msg)

        self.bands = bands
        self.spectral_indices = spectral_indices
        self.percentile = percentile
        self.dst_crs = dst_crs
        self.clip = clip

        # ODC StaC load parameters
        self.load_params = {
            "chunks": chunks or {},
            "groupby": groupby,
            "resampling": resampling,
            "stac_cfg": self.stac_cfg,
            "dtype": dtype,
            "crs": crs,
            "resolution": resolution,
            "pool": pool,
            "preserve_original_order": preserve_original_order,
            "progress": progress,
            "fail_on_error": fail_on_error,
            "geobox": geobox,
            "like": like,
            "patch_url": patch_url,
        }
        return self

    def _load(self) -> xr.Dataset:
        """
        Internal method to load data using odc.stac.
        """
        if not self.items:
            msg = "No items found. Perform a search first."
            raise ValueError(msg)

        bbox = tuple(self.search_params["intersects"].bounds)

        ds = odc.stac.load(
            self.items,
            bands=self.bands,
            bbox=bbox,
            **self.load_params,
        )

        for band in self.bands:
            nodata_value = get_nodata(ds[band])
            ds[band] = ds[band].where(ds[band] != nodata_value)
            ds[band] = set_nodata(ds[band], np.nan)

        if self.dst_crs and (
            pyproj.CRS.from_user_input(self.dst_crs).to_epsg() != ds.rio.crs.to_epsg()
        ):
            ds = ds.rio.reproject(self.dst_crs)
            ds = ds.odc.reproject(self.dst_crs, resampling="bilinear", nodata=np.nan)

            for band in self.bands:
                ds[band] = set_nodata(ds[band], np.nan)

        return ds

    def add_spectral_indices(
        self, indices: list[str], nodata: float | int | None = None
    ) -> xr.Dataset:
        """
        Add spectral indices to the current dataset.

        Args:
            indices (List[str]): Spectral indices to calculate.

        Returns:
            xr.Dataset: Updated dataset with spectral indices.
        """
        if self.dataset is None:
            msg = "No dataset loaded. Perform `execute` first."
            raise ValueError(msg)

        self.dataset = calculate_indices(self.dataset, indices)

        # Set nodata value for the new spectral indices
        if nodata is not None:
            for index in indices:
                self.dataset[index] = set_nodata(self.dataset[index], nodata)

        return self.dataset

    def composite(
        self, percentile: int = 50, nodata: float | int | None = None
    ) -> xr.Dataset:
        """
        Apply a composite operation to the dataset based on the given percentile.

        Args:
            percentile (int): Percentile to calculate (e.g., 50 for median).
                            Values range between 0 and 100.
            nodata (float | int | None): Value to assign for nodata pixels in the resulting composite.

        Returns:
            xr.Dataset: Composited dataset.
        """
        if self.dataset is None:
            msg = "No dataset loaded. Perform `execute` first."
            raise ValueError(msg)

        if not (0 <= percentile <= 100):
            msg = "Percentile must be between 0 and 100."
            raise ValueError(msg)

        logging.info(f"Applying {percentile}th percentile composite.")

        # Use median() if percentile is 50, otherwise quantile()
        if percentile == 50:
            composite = self.dataset.median(dim="time", skipna=True, keep_attrs=True)
            logging.info("Using median() for composite.")
        else:
            composite = self.dataset.quantile(
                percentile / 100, dim="time", skipna=True, keep_attrs=True
            )
            logging.info("Using quantile() for composite.")

        # Set nodata values for each band if provided
        if nodata is not None:

            def apply_nodata(da):
                return set_nodata(da, nodata)

            composite = composite.map(apply_nodata)

        self.dataset = composite
        return self.dataset

    def execute(self) -> xr.Dataset:
        """
        Trigger the search and load process and return the dataset.
        """
        # Perform search if not already done
        if self.items is None:
            logging.info(f"Executing search with params: {self.search_params}")
            search = self.catalog.search(**self.search_params)
            self.items = list(search.items())

        # Perform load if not already done
        if self.dataset is None:
            logging.info("Loading dataset...")
            self.dataset = self._load()

        if self.percentile:
            logging.info("Compositing dataset...")
            self.dataset = self.composite(percentile=self.percentile, nodata=np.nan)

        if self.spectral_indices:
            logging.info(f"Calculating spectral indices: {self.spectral_indices}")
            self.dataset = self.add_spectral_indices(
                self.spectral_indices, nodata=np.nan
            )

        return self.dataset


class S2Collection(ImageCollection):
    """
    A class to manage Sentinel-2 collections from the Planetary Computer catalog.
    """

    def __init__(
        self,
        catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection: str = "sentinel-2-l2a",
    ):
        stac_cfg = {
            "sentinel-2-l2a": {
                "assets": {
                    "*": {"data_type": None, "nodata": np.nan},
                    "SCL": {"data_type": None, "nodata": np.nan},
                    "visual": {"data_type": None, "nodata": np.nan},
                },
            },
            "*": {"warnings": "ignore"},
        }

        super().__init__(catalog_url, collection, stac_cfg)


class TileCollection:
    """
    A generic class to manage tile collections from a STAC-based catalog.
    """

    def __init__(
        self,
        catalog_url: str,
        collection: str,
        stac_cfg: dict | None = None,
    ):
        self.catalog_url = catalog_url
        self.collection = collection
        self.catalog = pystac_client.Client.open(self.catalog_url)

        # Configuration
        self.search_params = {}
        self.bands = []
        self.dst_crs = None
        self.load_params = {}
        self.stac_cfg = stac_cfg or {}

        # Internal state
        self.items = None
        self.dataset = None

    @abc.abstractmethod
    def search(self, roi: gpd.GeoDataFrame) -> "TileCollection":
        """
        Search for DeltaDTM items based on a region of interest.
        """

    def load(
        self,
        chunks: dict[str, int | str] | None = None,
        resampling: str | dict[str, str] | None = None,
        dtype: np.dtype | str | None = None,
        crs: str | int | None = None,
        resolution: float | int | None = None,
        pool: int | None = None,
        preserve_original_order: bool = False,
        progress: bool | None = None,
        fail_on_error: bool = True,
        geobox: dict | None = None,
        like: xr.Dataset | None = None,
        patch_url: str | None = None,
        dst_crs: Any | None = None,
    ) -> "TileCollection":
        """
        Configure loading parameters.

        Args:
            Additional args: Parameters for odc.stac.load.

        Returns:
            DeltaDTMCollection: Updated instance.
        """
        self.dst_crs = dst_crs

        self.load_params = {
            "chunks": chunks or {},
            "resampling": resampling,
            "dtype": dtype,
            "crs": crs,
            "resolution": resolution,
            "pool": pool,
            "preserve_original_order": preserve_original_order,
            "progress": progress,
            "fail_on_error": fail_on_error,
            "geobox": geobox,
            "like": like,
            "patch_url": patch_url,
        }
        return self

    def _load(self) -> xr.Dataset:
        """
        Internal method to load data using odc.stac.
        """
        if not self.items:
            msg = "No items found. Perform a search first."
            raise ValueError(msg)

        bbox = tuple(self.search_params["intersects"].bounds)

        ds = odc.stac.load(
            self.items,
            bbox=bbox,
            **self.load_params,
        ).squeeze()

        if self.dst_crs and (
            pyproj.CRS.from_user_input(self.dst_crs).to_epsg() != ds.rio.crs.to_epsg()
        ):
            ds = ds.rio.reproject(self.dst_crs)
            ds = ds.odc.reproject(self.dst_crs, resampling="cubic", nodata=np.nan)

        return ds

    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Post-process the dataset."""
        return ds

    def execute(self) -> xr.Dataset:
        """
        Trigger the search and load process and return the dataset.
        """
        # Perform search if not already done
        if self.items is None:
            msg = "No items found. Perform a search first."
            raise ValueError(msg)

        # Perform load if not already done
        if self.dataset is None:
            logging.info("Loading dataset...")
            self.dataset = self._load()
            self.dataset = self._post_process(self.dataset)

        return self.dataset


class DeltaDTMCollection(TileCollection):
    """
    A class to manage Delta DTM collections from the CoCliCo catalog.
    """

    def __init__(
        self,
        catalog_url: str = "https://coclico.blob.core.windows.net/stac/v1/catalog.json",
        collection: str = "deltares-delta-dtm",
    ):
        super().__init__(catalog_url, collection)

    def search(self, roi: gpd.GeoDataFrame) -> "DeltaDTMCollection":
        """
        Search for DeltaDTM items based on a region of interest.
        """

        self.search_params = {
            "collections": self.collection,
            "intersects": roi.to_crs(4326).geometry.item(),
        }

        col = self.catalog.get_collection(self.collection)
        storage_options = col.extra_fields["item_assets"]["data"][
            "xarray:storage_options"
        ]
        ddtm_extents = read_snapshot(
            col,
            columns=None,
            storage_options=storage_options,
        )
        r = gpd.sjoin(ddtm_extents, roi.to_crs(ddtm_extents.crs)).drop(
            columns="index_right"
        )
        self.items = list(stac_geoparquet.to_item_collection(r))

        # Check if items were found
        if not self.items:
            msg = "No items found for the given search parameters."
            raise ValueError(msg)

        return self

    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Post-process the dataset."""
        ds["data"] = ds["data"].where(ds["data"] != ds["data"].attrs["nodata"], 0)
        # NOTE: Idk if this is good practice
        ds["data"].attrs["nodata"] = np.nan
        return ds


class CopernicusDEMCollection(TileCollection):
    """
    A class to manage Copernicus DEM collections from the Planetary Computer catalog.
    """

    def __init__(
        self,
        catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        collection: str = "cop-dem-glo-30",
    ):
        stac_cfg = {
            "cop-dem-glo-30": {
                "assets": {
                    "*": {"data_type": "int16", "nodata": -32768},
                },
                "*": {"warnings": "ignore"},
            }
        }

        super().__init__(catalog_url, collection, stac_cfg)

    def search(self, roi: gpd.GeoDataFrame) -> "CopernicusDEMCollection":
        """
        Search for Copernicus DEM items based on a region of interest.
        """
        self.search_params = {
            "collections": self.collection,
            "intersects": roi.to_crs(4326).geometry.item(),
        }

        # Perform the search
        logging.info(f"Executing search with params: {self.search_params}")
        search = self.catalog.search(**self.search_params)
        self.items = list(search.items())

        # Check if items were found
        if not self.items:
            msg = "No items found for the given search parameters."
            raise ValueError(msg)

        return self


if __name__ == "__main__":

    def filter_and_sort_stac_items(
        items: list[pystac.Item],
        max_items: int,
        group_by: str,
        sort_by: str,
    ) -> list[pystac.Item]:
        """
        Filter and sort STAC items by grouping and ranking within each group.

        Args:
            items (list[pystac.Item]): List of STAC items to process.
            max_items (int): Maximum number of items to return per group.
            group_by (str): Property to group by (e.g., 's2:mgrs_tile').
            sort_by (str): Property to sort by within each group (e.g., 'eo:cloud_cover').

        Returns:
            list[pystac.Item]: Filtered and sorted list of STAC items.
        """
        try:
            # Convert STAC items to a DataFrame
            df = (
                stac_geoparquet.arrow.parse_stac_items_to_arrow(items)
                .read_all()
                .to_pandas()
            )

            # Group by the specified property and sort within groups
            df = (
                df.groupby(group_by, group_keys=False)
                .apply(lambda group: group.sort_values(sort_by).head(max_items))
                .reset_index(drop=True)
            )

            # Reconstruct the filtered list of items from indices
            return [items[idx] for idx in df.index]

        except Exception as err:
            logging.error(f"Error filtering and sorting items: {err}")
            return []

    import planetary_computer as pc
    import shapely

    west, south, east, north = (
        -1.4987754821777346,
        46.328320550966765,
        -1.446976661682129,
        46.352022707044455,
    )
    roi = gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(west, south, east, north)], crs=4326
    )

    def filter_function(items):
        return filter_and_sort_stac_items(
            items, max_items=10, group_by="s2:mgrs_tile", sort_by="eo:cloud_cover"
        )

    s2 = (
        S2Collection()
        .search(
            roi,
            datetime_range="2023-01-01/2023-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
            filter_function=filter_function,
        )
        .load(
            bands=["blue", "green", "red", "nir", "swir16"],
            percentile=50,
            spectral_indices=["NDWI", "NDVI"],
            chunks={"x": 256, "y": 256},
            patch_url=pc.sign,
        )
        .execute()
    )

    s2 = s2.compute()
    deltadtm = DeltaDTMCollection().search(roi).load().execute()
    cop_dem = CopernicusDEMCollection().search(roi).load().execute()
    print("done")
