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
from coastpy.eo.mask import apply_mask, geometry_mask, nodata_mask, numeric_mask
from coastpy.stac.utils import read_snapshot

# NOTE: currently all NODATA management is removed because we mask nodata after loading it by default.


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

        # Configuration
        self.search_params = {}
        self.bands = []
        self.spectral_indices = []
        self.percentile = None
        self.dst_crs = None
        self.load_params = {}
        self.stac_cfg = stac_cfg or {}

        # Masking options
        self.geometry_mask = None
        self.nodata_mask = False
        self.value_mask = None

        # Internal state
        self.stac_items = None
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
        self.stac_items = list(search.items())

        # Check if items were found
        if not self.stac_items:
            msg = "No items found for the given search parameters."
            raise ValueError(msg)

        # Apply the filter function if provided
        if filter_function:
            try:
                logging.info("Applying custom filter function.")
                self.stac_items = filter_function(self.stac_items)
            except Exception as e:
                msg = f"Error in filter_function: {e}"
                raise RuntimeError(msg)  # noqa: B904

        return self

    def load(
        self,
        bands: list[str],
        percentile: int | None = None,
        spectral_indices: list[str] | None = None,
        mask_nodata: bool = True,
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
        self.mask_nodata = mask_nodata

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

    def mask(
        self,
        geometry: odc.geo.geom.Geometry | None = None,
        nodata: bool = True,
        values: list[int] | None = None,
    ) -> "ImageCollection":
        """
        Configure masking options.

        Args:
            geometry (odc.geo.geom.Geometry | None): Geometry to mask data within.
            nodata (bool): Whether to apply a nodata mask.
            values (List[float | int] | None): Specific values to mask.

        Returns:
            ImageCollection: Updated instance with masking options configured.
        """
        self.geometry_mask = geometry
        self.nodata_mask = nodata
        self.value_mask = values
        return self

    def _load(self) -> xr.Dataset:
        if not self.stac_items:
            msg = "No items found. Perform a search first."
            raise ValueError(msg)

        bbox = tuple(self.search_params["intersects"].bounds)

        # Load the data
        ds = odc.stac.load(
            self.stac_items,
            bands=self.bands,
            bbox=bbox,
            **self.load_params,
        )

        return ds

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

    def _reproject(
        self, ds: xr.DataArray | xr.Dataset, dst_crs
    ) -> xr.DataArray | xr.Dataset:
        """
        Reproject the dataset to a new CRS.
        """
        ds = ds.odc.reproject(dst_crs, resampling="bilinear", nodata=np.nan)
        return ds

    def add_spectral_indices(self, indices: list[str]) -> "ImageCollection":
        """
        Add spectral indices to the current dataset.

        Args:
            indices (List[str]): Spectral indices to calculate.

        Returns:
            ImageCollection: Updated ImageCollection with spectral indices.
        """
        self.spectral_indices = indices
        return self

    def _composite(
        self,
        ds: xr.DataArray | xr.Dataset,
        percentile: int,
        determination_times: list[str] | None = None,
        stac_ids: list[str] | None = None,
        cloud_cover: float | None = None,
    ) -> xr.DataArray | xr.Dataset:
        # Use median() if percentile is 50, otherwise quantile()
        if percentile == 50:
            composite = ds.median(dim="time", skipna=True, keep_attrs=True)
            logging.info("Using median() for composite.")
        else:
            composite = ds.quantile(
                percentile / 100, dim="time", skipna=True, keep_attrs=True
            )
            composite.attrs["determination_method"] = "percentile"
            logging.info("Using quantile() for composite.")

        composite.attrs["composite:determination_method"] = "percentile"
        composite.attrs["composite:percentile"] = percentile

        if determination_times is not None:
            composite.attrs["composite:determination_datetimes"] = determination_times

        if stac_ids is not None:
            composite.attrs["composite:stac_ids"] = stac_ids

        if cloud_cover is not None:
            composite.attrs["composite:cloud_cover"] = cloud_cover

        return composite

    def composite(self, percentile: int = 50) -> "ImageCollection":
        """
        Apply a composite operation to the dataset based on the given percentile.

        Args:
            percentile (int): Percentile to calculate (e.g., 50 for median).
                            Values range between 0 and 100.
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

        self.percentile = percentile
        return self

    def execute(self) -> xr.DataArray | xr.Dataset:
        if self.stac_items is None:
            search = self.catalog.search(**self.search_params)
            self.stac_items = list(search.items())

        if self.dataset is None:
            self.dataset = self._load()

        if self.geometry_mask or self.nodata_mask or self.value_mask:
            self.dataset = self._apply_masks(self.dataset)

        if self.dst_crs and (
            pyproj.CRS.from_user_input(self.dst_crs).to_epsg()
            != self.dataset.rio.crs.to_epsg()
        ):
            self.dataset = self._reproject(self.dataset, self.dst_crs)

        if self.percentile:
            determination_times = (
                self.dataset.time.to_series().apply(lambda x: x.isoformat()).unique()
            )

            stac_ids = [i.id for i in self.stac_items]

            cloud_cover = float(
                np.mean(
                    [float(i.properties["eo:cloud_cover"]) for i in self.stac_items]
                )
            )

            self.dataset = self._composite(
                ds=self.dataset,
                percentile=self.percentile,
                determination_times=determination_times,
                stac_ids=stac_ids,
                cloud_cover=cloud_cover,
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

            self.dataset = calculate_indices(self.dataset, self.spectral_indices)

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
            ds = ds.odc.reproject(self.dst_crs, resampling="cubic")

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
    import logging
    import os
    from typing import Any, Literal

    import dotenv
    import fsspec
    import geopandas as gpd
    import odc
    import odc.geo.geom
    import odc.stac
    import planetary_computer as pc
    import pystac
    import shapely
    import stac_geoparquet
    import xarray as xr
    from odc.stac import configure_rio

    # from coastpy.eo.collection import S2Collection
    from coastpy.eo.filter import filter_and_sort_stac_items
    from coastpy.stac.utils import read_snapshot
    from coastpy.utils.config import configure_instance

    configure_rio(cloud_defaults=True)
    instance_type = configure_instance()

    dotenv.load_dotenv()
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}

    west, south, east, north = (4.796, 53.108, 5.229, 53.272)

    roi = gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(west, south, east, north)], crs=4326
    )

    def get_coastal_zone(coastal_grid, region_of_interest):
        df = gpd.sjoin(coastal_grid, region_of_interest).drop(columns=["index_right"])
        coastal_zone = df.union_all()
        return odc.geo.geom.Geometry(coastal_zone, crs=df.crs)

    def filter_function(items):
        r = filter_and_sort_stac_items(
            items, max_items=10, group_by="s2:mgrs_tile", sort_by="eo:cloud_cover"
        )
        return r

    def read_coastal_grid(
        zoom: Literal[5, 6, 7, 8, 9, 10],
        buffer_size: Literal["500m", "1000m", "2000m", "5000m", "10000m", "15000m"],
        storage_options,
    ):
        """
        Load the coastal zone data layer for a specific buffer size.
        """
        coclico_catalog = pystac.Catalog.from_file(
            "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
        )
        coastal_zone_collection = coclico_catalog.get_child("coastal-grid")
        if coastal_zone_collection is None:
            msg = "Coastal zone collection not found"
            raise ValueError(msg)
        item = coastal_zone_collection.get_item(f"coastal_grid_z{zoom}_{buffer_size}")
        if item is None:
            msg = f"Coastal zone item for zoom {zoom} with {buffer_size} not found"
            raise ValueError(msg)
        href = item.assets["data"].href
        with fsspec.open(href, mode="rb", **storage_options) as f:
            coastal_zone = gpd.read_parquet(f)
        return coastal_zone

    coastal_grid = read_coastal_grid(
        zoom=10, buffer_size="5000m", storage_options=storage_options
    )

    coastal_zone = get_coastal_zone(coastal_grid, roi)

    s2 = (
        S2Collection()
        .search(
            roi,
            datetime_range="2022-01-01/2023-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
            filter_function=filter_function,
        )
        .load(
            bands=["blue", "green", "red", "nir", "swir16"],
            percentile=50,
            spectral_indices=["NDWI", "NDVI", "MNDWI", "NDMI"],
            mask_nodata=True,
            chunks={},
            patch_url=pc.sign,
        )
        .mask(geometry=coastal_zone, nodata=True)
        .execute()
    )

    print("Done")
    s2 = s2.compute()
