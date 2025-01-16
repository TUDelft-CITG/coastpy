import abc
import logging
from collections.abc import Callable
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import odc.geo
import odc.geo.cog
import odc.geo.geobox
import odc.geo.geom
import odc.stac
import pandas as pd
import pyproj
import pystac
import pystac.item
import pystac_client
import stac_geoparquet
import xarray as xr

from coastpy.eo.indices import calculate_indices
from coastpy.eo.mask import (
    SceneClassification,
    apply_mask,
    geometry_mask,
    nodata_mask,
    numeric_mask,
    scl_mask,
)
from coastpy.eo.utils import geobox_from_data_extent
from coastpy.stac.utils import read_snapshot
from coastpy.utils.xarray import combine_by_first, unscale

# NOTE: currently all NODATA management is removed because we mask nodata after loading it by default.
# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.load_params = {}
        self.bands = []
        self.normalize = True
        self.spectral_indices = []
        self.percentile = None
        self.composite_method = None
        self.stac_cfg = stac_cfg or {}

        # Masking options
        self.geometry_mask = None
        self.nodata_mask = False
        self.value_mask = None

        # Internal state
        self.geometry = None
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
            filter(Callable, optional): A custom function to filter/sort items.
                Accepts and returns a list of pystac.Items.

        Returns:
            ImageCollection: Updated instance with items populated.
        """
        self.region_of_interest = roi

        geom = roi.to_crs(4326).geometry.item()
        self.search_params = {
            "collections": self.collection,
            "intersects": geom,
            "datetime": datetime_range,
            "query": query,
        }

        # Perform the actual search
        logging.info(f"Executing search with params: {self.search_params}")
        search = self.catalog.search(**self.search_params)
        self.stac_items = list(search.items())

        # Check if items were found
        if not self.stac_items:
            msg = "No items found for the given search parameters."
            raise ValueError(msg)

        # Log the number of STAC items found
        logger.info(f"Number of STAC items found: {len(self.stac_items)}")

        # Log the number of unique MGRS grid tiles
        unique_mgrs_tiles = len(
            {item.properties["s2:mgrs_tile"] for item in self.stac_items}
        )
        logger.info(f"Number of unique MGRS grid tiles: {unique_mgrs_tiles}")

        # Log the number of unique relative orbits
        unique_relative_orbits = len(
            {item.properties["sat:relative_orbit"] for item in self.stac_items}
        )
        logger.info(f"Number of unique relative orbits: {unique_relative_orbits}")

        # Apply the filter function if provided
        # move to composite
        if filter_function:
            try:
                logging.info("Applying custom filter function.")
                self.stac_items = filter_function(self.stac_items)
            except Exception as e:
                msg = f"Error in filter_function: {e}"
                raise RuntimeError(msg)  # noqa: B904

        items_as_json = [item.to_dict() for item in self.stac_items]
        self.data_extent = (
            stac_geoparquet.to_geodataframe(items_as_json, dtype_backend="pyarrow")
            .dissolve("s2:mgrs_tile")[["geometry"]]
            .dissolve()
        )

        return self

    def load(
        self,
        bands: list[str],
        percentile: int | None = None,
        spectral_indices: list[str] | None = None,
        mask_nodata: bool = True,
        normalize: bool = True,
        chunks: dict[str, int | str] | None = None,
        groupby: str = "solar_day",
        resampling: str | dict[str, str] | None = None,
        dtype: np.dtype | str | None = None,
        crs: str | int = "utm",
        resolution: float | int | None = None,
        pool: int | None = None,
        preserve_original_order: bool = False,
        progress: bool | None = None,
        fail_on_error: bool = True,
        geobox: odc.geo.geobox.GeoBox | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        geopolygon: dict | None = None,
        lon: tuple[float, float] | None = None,
        lat: tuple[float, float] | None = None,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        like: xr.Dataset | None = None,
        patch_url: Callable | None = None,
        stac_cfg: dict | None = None,
        anchor: str | None = None,
    ) -> "ImageCollection":
        """
        Configure parameters for loading data via odc.stac.load.

        Args:
            bands (list[str]): Bands to load (required).
            percentile (int | None): Percentile for compositing (e.g., 50 for median).
            spectral_indices (list[str] | None): List of spectral indices to compute.
            mask_nodata (bool): Mask no-data values. Defaults to True.
            normalize (bool): Normalize data. Defaults to True.
            resolution (float | int | None): Pixel resolution in CRS units. Defaults to 10.
            crs (str | int): Coordinate reference system. Defaults to 'utm'.
            geobox (GeoBox | None): Exact region, resolution, and CRS to load. Overrides other extent parameters.
            bbox, geopolygon, lon, lat, x, y: Optional parameters to define extent.
            Additional args: Parameters passed to odc.stac.load.

        Returns:
            ImageCollection: Updated instance.
        """
        if not bands:
            raise ValueError("Argument `bands` is required.")

        if percentile is not None and not (0 <= percentile <= 100):
            raise ValueError("`percentile` must be between 0 and 100.")

        self.bands = bands
        self.normalize = normalize
        self.spectral_indices = spectral_indices
        self.percentile = percentile
        self.mask_nodata = mask_nodata

        # Geobox creation
        if geobox is None and resolution:
            geobox = geobox_from_data_extent(
                region=self.region_of_interest,
                data_extent=self.data_extent,
                crs=crs,
                resolution=resolution,
            )
            resolution = None  # Let geobox handle resolution

        # Assemble load parameters
        self.load_params = {
            "chunks": chunks,
            "groupby": groupby,
            "resampling": resampling,
            "dtype": dtype,
            "resolution": resolution,
            "pool": pool,
            "preserve_original_order": preserve_original_order,
            "progress": progress,
            "fail_on_error": fail_on_error,
            "geobox": geobox,
            "bbox": bbox,
            "geopolygon": geopolygon,
            "lon": lon,
            "lat": lat,
            "x": x,
            "y": y,
            "like": like,
            "patch_url": patch_url,
            "stac_cfg": self.stac_cfg or stac_cfg,
            "anchor": anchor,
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
        """
        Load data using odc.stac.load.

        Returns:
            xr.Dataset: Loaded dataset.
        """
        if not self.stac_items:
            raise ValueError("No STAC items found. Perform a search first.")

        # Adjust groupby for percentile-based compositing
        if self.percentile:
            self.load_params["groupby"] = "id"

        # Fallback to bbox if no spatial bounds are provided
        if (
            not self.load_params.get("geobox")
            and not self.load_params.get("bbox")
            and not self.load_params.get("geopolygon")
            and not self.load_params.get("like")
        ):
            bbox = tuple(self.search_params["intersects"].bounds)
            self.load_params["bbox"] = bbox

        # Call odc.stac.load
        # del self.load_params["chunks"]
        # del self.load_params["dtype"]
        ds = odc.stac.load(
            self.stac_items,
            bands=self.bands,
            **self.load_params,
            # chunks = {"time": "auto", "x": "auto", "y":"auto"},
            # dtype="float32"
        )

        # Add metadata if time matches item count
        if ds.sizes["time"] == len(self.stac_items):
            ds = self._add_metadata_from_stac(self.stac_items, ds)

        if self.normalize:
            ds = unscale(ds, scale_factor=10000, variables_to_ignore=["SCL"])

        return ds  # type: ignore

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

        # Assign metadata as coordinates
        ds = ds.assign_coords({"stac_id": ("time", stac_ids)})
        ds = ds.assign_coords({"s2:mgrs_tile": ("time", mgrs_tiles)})
        ds = ds.assign_coords({"eo:cloud_cover": ("time", cloud_cover)})
        ds = ds.assign_coords({"sat:relative_orbit": ("time", rel_orbits)})

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

    @classmethod
    def _extract_composite_metadata(cls, data: xr.Dataset | xr.DataArray) -> dict:
        metadata = {}
        datetimes = data.time.to_series().sort_values()
        start_datetime = datetimes.min().isoformat()
        end_datetime = datetimes.max().isoformat()
        avg_interval = datetimes.diff().mean()
        n_obs = len(datetimes)
        metadata = {
            "datetime": start_datetime,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "eo:cloud_cover": str(int(data["eo:cloud_cover"].mean().item())),
            "composite:avg_interval": avg_interval,
            "composite:n_obs": n_obs,
            "composite:stac_ids": str([str(i) for i in data.stac_id.values]),
        }
        return metadata

    @classmethod
    def _grouped_composite(
        cls,
        ds: xr.DataArray | xr.Dataset,
        percentile: int,
    ) -> xr.DataArray | xr.Dataset:
        """
        Generate a composite dataset using median or percentile methods,
        respecting metadata and sampling constraints.

        Args:
            ds (xr.DataArray | xr.Dataset): The input dataset.
            percentile (int): Percentile value (50 for median).

        Returns:
            xr.DataArray | xr.Dataset: Composite dataset with a time dimension.
        """
        try:
            # Step 1: Create a combined group key
            ds = ds.assign(
                group_key=(
                    xr.apply_ufunc(
                        lambda tile, orbit: tile + "_" + str(orbit),
                        ds["s2:mgrs_tile"],
                        ds["sat:relative_orbit"],
                        vectorize=True,
                    )
                )
            )
            # target_chunks = {k: v for k, v in ds.chunks.items() if k != "time"}  # type: ignore

            # Remove SCL band if present (composite method for categorical data not implemented)
            if "SCL" in ds:
                ds = ds.drop_vars("SCL")

            # Step 2: Sort by cloud coverage
            ds_sorted = ds.sortby("eo:cloud_cover")

            # Step 3: Group by the combined key
            grouped = ds_sorted.groupby("group_key")

            # Step 4: Sample and compute the composite
            def aggregate(group):
                if percentile == 50:
                    return group.median(dim="time", skipna=True, keep_attrs=True)
                else:
                    return group.quantile(
                        percentile / 100, dim="time", skipna=True, keep_attrs=True
                    )

            composite = grouped.map(aggregate)
            datasets = [
                composite.isel(group_key=i) for i in range(composite.sizes["group_key"])
            ]
            collapsed = combine_by_first(datasets)

            # collapsed = collapsed.chunk(target_chunks)

            group_metadata_list = []
            for group, group_data in grouped:
                metadata = cls._extract_composite_metadata(group_data)
                metadata["group_key"] = group
                group_metadata_list.append(metadata)

            # Global metadata aggregation
            datetime = min(item["datetime"] for item in group_metadata_list)
            start_datetime = min(item["start_datetime"] for item in group_metadata_list)
            end_datetime = max(item["end_datetime"] for item in group_metadata_list)
            avg_intervals = [
                item["composite:avg_interval"] for item in group_metadata_list
            ]
            avg_interval = f"{pd.Series(avg_intervals).mean().days} days"  # type: ignore
            avg_obs = np.mean([item["composite:n_obs"] for item in group_metadata_list])

            collapsed = collapsed.assign_coords(
                {
                    "time": datetime,
                    "start_datetime": start_datetime,
                    "end_datetime": end_datetime,
                }
            )

            # Update global attributes for composite metadata
            collapsed.attrs.update(
                {
                    "datetime": datetime,
                    "start_datetime": start_datetime,
                    "end_datetime": end_datetime,
                    "eo:cloud_cover": str(
                        int(ds_sorted["eo:cloud_cover"].mean().item())
                    ),
                    "composite:determination_method": "CoastPy Grouped Median"
                    if percentile == 50
                    else "grouped_percentile",
                    "composite:percentile": str(int(percentile)),
                    "composite:groups": str(
                        list(ds_sorted.group_key.to_series().unique())
                    ),
                    "composite:avg_obs": str(round(float(avg_obs), 2)),
                    "composite:stac_ids": str(
                        [str(i) for i in ds_sorted.stac_id.values]
                    ),
                    "composite:avg_interval": avg_interval,
                    "composite:summary": (
                        f"Composite dataset created by grouping on ['s2:mgrs_tile', 'sat:relative_orbit'], using a "
                        f"{'median' if percentile == 50 else f'{percentile}th percentile'} method, "
                        f"sorted by 'eo:cloud_cover' with an average of {avg_obs} images per group."
                    ),
                }
            )

            return collapsed

        except Exception as e:
            raise RuntimeError(f"Failed to generate composite: {e}") from e

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
                    "composite:determination_method": "CoastPy Simple Median",
                    "composite:summary": (
                        "Composite dataset created by taking the median value of each pixel "
                        "across all time steps."
                    ),
                }
            )
            return composite
        except Exception as e:
            raise RuntimeError(f"Failed to generate simple composite: {e}") from e

    def composite(
        self,
        method: Literal["simple", "grouped"] = "simple",
        percentile: int = 50,
        filter_function: Callable[[list[pystac.Item]], list[pystac.Item]] | None = None,
    ) -> "ImageCollection":
        """
        Apply a composite operation to the dataset based on the given percentile.

        Args:
            composite_method (str): Composite method to apply. Options: 'simple', 'grouped'.
            percentile (int): Percentile to calculate (e.g., 50 for median).
                            Values range between 0 and 100.
            filter_function (Callable, optional): A custom function to filter/sort items.
                                                Accepts and returns a list of pystac.Items.

        Returns:
            ImageCollection: Composited dataset.

        Raises:
            ValueError: If percentile is not between 0 and 100 or if no STAC items are found.
            RuntimeError: If an error occurs in the filter_function.
        """
        if not (0 <= percentile <= 100):
            msg = "Percentile must be between 0 and 100."
            raise ValueError(msg)
        logging.info(f"Applying {percentile}th percentile composite.")

        if not self.stac_items:
            raise ValueError("No STAC items found. Perform a search first.")

        if filter_function:
            try:
                logging.info("Applying custom filter function.")
                self.stac_items = filter_function(self.stac_items)

                # Log the number of STAC items found
                logger.info(f"Number of STAC items remaining: {len(self.stac_items)}")

                # Log the number of unique MGRS grid tiles
                unique_mgrs_tiles = len(
                    {item.properties["s2:mgrs_tile"] for item in self.stac_items}
                )
                logger.info(f"Number of unique MGRS grid tiles: {unique_mgrs_tiles}")

                # Log the number of unique relative orbits
                unique_relative_orbits = len(
                    {item.properties["sat:relative_orbit"] for item in self.stac_items}
                )
                logger.info(
                    f"Number of unique relative orbits: {unique_relative_orbits}"
                )

            except Exception as e:
                msg = f"Error in filter_function: {e}"
                raise RuntimeError(msg) from e

        self.composite_method = method
        self.percentile = percentile
        return self

    def execute(self, compute=False) -> xr.DataArray | xr.Dataset:
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
                    "*": {"data_type": "float32", "nodata": np.nan},
                    "SCL": {"data_type": "float32", "nodata": np.nan},
                    "visual": {"data_type": None, "nodata": np.nan},
                },
            },
            "*": {"warnings": "ignore"},
        }

        super().__init__(catalog_url, collection, stac_cfg)

    def mask(
        self,
        geometry: odc.geo.geom.Geometry | None = None,
        nodata: bool = True,
        values: list[int] | None = None,
        scl_classes: list[str | SceneClassification | int] | None = None,
    ) -> "S2Collection":
        """
        Mask the dataset based on geometry, nodata, specific values, or the SCL band.

        Args:
            geometry (odc.geo.geom.Geometry | None, optional): Geometry to mask data within. Defaults to None.
            nodata (bool, optional): Whether to apply a nodata mask. Defaults to True.
            values (list[int] | None, optional): Specific values to mask. Defaults to None.
            scl_classes (list[SceneClassification | int] | None, optional): List of SCL class names or numeric values to mask. Defaults to None.

        Returns:
            S2Collection: Updated instance with masking options configured.

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
        super().mask(geometry, nodata, values)
        self.scl_classes = scl_classes
        return self

    def _apply_masks(self, ds: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        """
        Apply masks to the dataset.
        """
        ds = super()._apply_masks(ds)

        if self.scl_classes:
            mask = scl_mask(ds, self.scl_classes)
            ds = apply_mask(ds, mask)

        return ds


class S2CompositeCollection(ImageCollection):
    """
    A class to manage Sentinel-2 composite collections from a STAC-based catalog.
    """

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

    def search(self, roi: gpd.GeoDataFrame) -> "S2CompositeCollection":
        """
        Search for S2 composite items based on a region of interest.

        Args:
            roi (gpd.GeoDataFrame): Region of interest.

        Returns:
            S2CompositeCollection: Updated instance with search results.
        """
        # Define the search parameters
        self.search_params = {
            "collections": self.collection,
            "intersects": roi.to_crs(4326).geometry.item(),
        }

        # Access the STAC collection
        col = self.catalog.get_collection(self.collection)
        # Load spatial extents of the collection using `read_snapshot`
        composite_extents = read_snapshot(
            col,
            columns=None,
            storage_options=None,
        )

        # Perform spatial join with the region of interest
        matched_items = gpd.sjoin(
            composite_extents, roi.to_crs(composite_extents.crs)
        ).drop(columns="index_right")
        self.stac_items = list(stac_geoparquet.to_item_collection(matched_items))

        # Check if items were found
        if not self.stac_items:
            raise ValueError("No items found for the given search parameters.")

        return self

    def load(
        self,
        bands: list[str],
        spectral_indices: list[str] | None = None,
        chunks: dict[str, int | str | Literal["auto"]] | None = None,
        resampling: str | dict[str, str] | None = None,
        dtype: np.dtype | str | None = None,
        crs: str | int = "utm",
        resolution: float | int | None = None,
        pool: int | None = None,
        preserve_original_order: bool = False,
        progress: bool | None = None,
        fail_on_error: bool = True,
        geobox: odc.geo.geobox.GeoBox | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        geopolygon: dict | None = None,
        lon: tuple[float, float] | None = None,
        lat: tuple[float, float] | None = None,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        like: xr.Dataset | None = None,
        patch_url: Callable | None = None,
        stac_cfg: dict | None = None,
        anchor: str | None = None,
    ) -> "S2CompositeCollection":
        """
        Configure parameters for loading data via odc.stac.load.

        Args:
            bands (list[str]): Bands to load (required).
            chunks (dict, optional): Chunk size for dask. Defaults to None.
            resampling (str | dict, optional): Resampling strategy. Defaults to None.
            dtype (np.dtype | str, optional): Data type for the output. Defaults to None.
            crs (str | int, optional): Coordinate reference system. Defaults to "utm".
            resolution (float | int, optional): Pixel resolution in CRS units. Defaults to 10.
            Additional args: Passed to odc.stac.load.

        Returns:
            S2CompositeCollection: Updated instance with loading parameters.
        """
        if not bands:
            raise ValueError("Argument `bands` is required.")

        self.bands = bands
        self.spectral_indices = spectral_indices

        # Geobox creation
        if geobox is None and resolution:
            geobox = geobox_from_data_extent(
                region=self.region_of_interest,
                data_extent=self.data_extent,
                crs=crs,
                resolution=resolution,
            )
            resolution = None  # Let geobox handle resolution

        # Assemble load parameters
        self.load_params = {
            "chunks": chunks,
            "resampling": resampling,
            "dtype": dtype,
            "resolution": resolution,
            "pool": pool,
            "preserve_original_order": preserve_original_order,
            "progress": progress,
            "fail_on_error": fail_on_error,
            "geobox": geobox,
            "bbox": bbox,
            "geopolygon": geopolygon,
            "lon": lon,
            "lat": lat,
            "x": x,
            "y": y,
            "like": like,
            "patch_url": patch_url,
            "stac_cfg": self.stac_cfg or stac_cfg,
            "anchor": anchor,
        }

        return self

    def _load(self) -> xr.Dataset:
        """
        Load data using odc.stac.load.

        Returns:
            xr.Dataset: Loaded dataset.
        """
        if not self.stac_items:
            raise ValueError("No STAC items found. Perform a search first.")

        # Fallback to bbox if no spatial bounds are provided
        if (
            not self.load_params.get("geobox")
            and not self.load_params.get("bbox")
            and not self.load_params.get("geopolygon")
            and not self.load_params.get("like")
        ):
            bbox = tuple(self.search_params["intersects"].bounds)
            self.load_params["bbox"] = bbox

        # Call odc.stac.load
        # TODO: chunks = "auto"
        ds = odc.stac.load(
            self.stac_items,
            bands=self.bands,
            **self.load_params,
        )

        # # Add metadata if time matches item count
        # if ds.sizes["time"] == len(self.stac_items):
        #     ds = self._add_metadata_from_stac(self.stac_items, ds)

        # if self.normalize:
        #     ds = unscale(ds, scale_factor=10000, variables_to_ignore=["SCL"])

        return ds  # type: ignore

    # def execute(self, compute=False) -> xr.DataArray | xr.Dataset:
    #     """
    #     Execute the data loading process and return the dataset.

    #     Args:
    #         compute (bool): Whether to trigger computation for lazy datasets. Defaults to False.

    #     Returns:
    #         xr.DataArray | xr.Dataset: The loaded dataset.
    #     """
    #     if self.stac_items is None:
    #         raise ValueError("No STAC items found. Perform a search first.")

    #     if self.dataset is None:
    #         self.dataset = self._load()

    #     # Mask application is skipped as composite datasets are pre-processed
    #     if compute:
    #         self.dataset = self.dataset.compute()

    #     return self.dataset


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
            TileCollection: Updated instance.
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
