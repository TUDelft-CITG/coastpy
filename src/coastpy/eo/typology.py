import warnings
from typing import Literal

import dask.bag as db
import geopandas as gpd
import odc.geo
import odc.geo.cog
import odc.geo.geobox
import odc.geo.geom
import odc.stac
import planetary_computer as pc
import shapely
import stac_geoparquet
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from rasterio.enums import Resampling
from shapely.geometry import Point

from coastpy.eo.collection2 import (
    CopernicusDEMCollection,
    DeltaDTMCollection,
    S2CompositeCollection,
)
from coastpy.geo.geoms import create_offset_rectangle
from coastpy.geo.ops import get_rotation_angle
from coastpy.io.utils import get_datetimes, merge_time_attrs, update_time_coord
from coastpy.utils.xarray_utils import interpolate_raster, trim_outer_nans


def transect_region(transect, offset_distance):
    roi = gpd.GeoDataFrame(
        transect.drop(columns=["geometry"]),
        geometry=[
            create_offset_rectangle(
                transect.to_crs(transect.utm_epsg.item()).geometry.item(),
                offset_distance,
            )
        ],
        crs=transect.utm_epsg.item(),
    )
    return roi.to_crs(transect.crs)


def get_rotation_angle_from_transect(
    transect: gpd.GeoDataFrame,
    target_axis: Literal[
        "closest", "vertical", "horizontal", "horizontal-right-aligned"
    ] = "horizontal-right-aligned",
) -> float:
    utm_epsg = transect.utm_epsg.item()
    p1, p2 = list(map(Point, transect.to_crs(utm_epsg).geometry.item().coords))
    rotation_angle = get_rotation_angle(p1, p2, target_axis=target_axis)
    return rotation_angle


class TypologyCollection:
    """
    A class that integrates Sentinel-2 composites, Delta DTM, and Copernicus DEM
    into a unified geospatial analysis cube.
    """

    def __init__(self):
        """
        Initialize the TypologyCollection.
        """
        # State variables
        self.roi: gpd.GeoDataFrame | None = None
        self.coastal_zone: odc.geo.geom.Geometry | None = None
        self.dataset: xr.Dataset | None = None
        self.sas_token: str | None = None

        # Default configurations
        self.S2_DEFAULTS = {
            "bands": ["blue", "green", "red", "nir", "swir16", "swir22"],
            "spectral_indices": ["NDWI", "NDVI", "MNDWI", "NDMI"],
            "dtype": "float32",
            "resolution": 10,
            "crs": "utm",
            "mask_nodata": True,
            "add_metadata_from_stac": False,
            # "scale": True,
            # "scale_factor": 0.0001,
            "patch_url": lambda url: f"{url}?{self.sas_token}",
            "chunks": {"time": "auto", "y": "auto", "x": "auto"},
        }

        self.DELTADTM_DEFAULTS = {
            "dtype": "float32",
            "mask_nodata": True,
            "add_metadata_from_stac": False,
            "patch_url": lambda url: f"{url}?{self.sas_token}",
            "chunks": {"y": "auto", "x": "auto"},
        }
        self.COP_DEM_DEFAULTS = {
            "dtype": "float32",
            "mask_nodata": True,
            "add_metadata_from_stac": False,
            "patch_url": pc.sign,
            "chunks": {"y": "auto", "x": "auto"},
        }

    def search(
        self,
        roi: gpd.GeoDataFrame,
        coastal_zone: odc.geo.geom.Geometry | None = None,
        sas_token: str | None = None,
    ) -> "TypologyCollection":
        """
        Configure the region of interest and coastal zone.

        Args:
            roi (gpd.GeoDataFrame): The region of interest.
            coastal_zone (odc.geo.geom.Geometry): Coastal geometry for masking.

        Returns:
            TypologyCollection: Updated instance.
        """
        self.roi = roi
        self.coastal_zone = coastal_zone
        self.sas_token = sas_token
        return self

    def load(
        self,
        s2_config: dict | None = None,
        deltadtm_config: dict | None = None,
        cop_dem_config: dict | None = None,
        compute: bool = False,
    ) -> "TypologyCollection":
        """
        Load and assemble data from multiple collections.

        Args:
            s2_config (dict, optional): Configuration for Sentinel-2 composites.
            delta_dtm_config (dict, optional): Configuration for Delta DTM.
            cop_dem_config (dict, optional): Configuration for Copernicus DEM.
            compute (bool): Whether to compute the final dataset.

        Returns:
            TypologyCollection: Updated instance with loaded data.
        """
        if self.roi is None:
            raise ValueError(
                "The region of interest (`roi`) must be set via `search()` before loading."
            )

        # Merge user-specified configurations with defaults
        s2_config = {**self.S2_DEFAULTS, **(s2_config or {})}
        deltadtm_config = {**self.DELTADTM_DEFAULTS, **(deltadtm_config or {})}
        cop_dem_config = {**self.COP_DEM_DEFAULTS, **(cop_dem_config or {})}

        # --- Load Sentinel-2 Composite ---
        s2_nodata = s2_config.get("mask_nodata", False)
        s2_scale = s2_config.get("scale", False)
        s2_scale_factor = s2_config.get("scale_factor", False)
        s2_spectral_indices = s2_config.get("spectral_indices", [])

        cube = (
            S2CompositeCollection(stac_cfg={})
            .search(self.roi)
            .load(
                **s2_config,
            )
            .mask_and_scale(
                mask_geometry=self.coastal_zone,
                mask_nodata=s2_nodata,
                scale=s2_scale,
                scale_factor=s2_scale_factor,
            )
            .add_spectral_indices(s2_spectral_indices)
            .merge_overlapping_tiles()  # type: ignore
            .execute(compute=compute)
        )
        s2_datetimes = get_datetimes(cube)

        geobox = cube.odc.geobox

        # --- Load Delta DTM ---
        mask_ddtm_nodata = deltadtm_config.pop("mask_nodata", False)
        ddtm = (
            DeltaDTMCollection()
            .search(self.roi)
            .load(like=geobox, **deltadtm_config)
            .mask_and_scale(
                mask_geometry=self.coastal_zone, mask_nodata=mask_ddtm_nodata
            )
            # NOTE: this would set the nodata values to 0, which make kind of sense, but
            # we decide to do it at the end.
            # .postprocess(DeltaDTMCollection.postprocess_deltadtm)
            .execute(compute=compute)
        ).squeeze()

        ddtm_datetimes = get_datetimes(ddtm)

        # --- Load Copernicus DEM ---
        mask_cop_dem30_nodata = cop_dem_config.pop("mask_nodata", False)
        cop = (
            CopernicusDEMCollection()
            .search(self.roi)
            .load(
                like=geobox,
                **cop_dem_config,
            )
            .mask_and_scale(
                mask_geometry=self.coastal_zone, mask_nodata=mask_cop_dem30_nodata
            )
            # .postprocess(CopernicusDEMCollection.postprocess_cop_dem30)
            .execute()
        ).squeeze()

        cop_datetimes = get_datetimes(cop)

        # --- Merge into a Single Dataset ---
        cube["deltadtm"] = ddtm["data"]
        cube["cop_dem_glo_30"] = cop["data"]

        # --- Update Time Dimension ---
        new_datetime = merge_time_attrs([s2_datetimes, ddtm_datetimes, cop_datetimes])
        cube = update_time_coord(cube, new_datetime)

        self.dataset = cube
        return self

    def execute(self, compute: bool = False) -> xr.Dataset:
        """
        Execute the full workflow and return the final dataset.

        Args:
            compute (bool): Whether to compute the dataset.

        Returns:
            xr.Dataset: The assembled geospatial analysis cube.
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call `load()` before executing.")
        return self.dataset.compute() if compute else self.dataset

    @staticmethod
    def chip_from_transect(
        dataset: xr.Dataset,
        transect: gpd.GeoDataFrame,
        y_shape: int,
        x_shape: int,
        resampling: Resampling = Resampling.cubic,
        rotate: bool = False,
        offset_distance: int | None = None,
        target_axis: Literal[
            "closest", "horizontal", "vertical", "horizontal-right-aligned"
        ]
        | None = None,
        resolution: int | None = None,
    ) -> xr.Dataset:
        """
        Rotate, clip, and interpolate the dataset to align with the transect's orientation.

        Args:
            dataset (xr.Dataset): The dataset to be rotated and clipped.
            transect (gpd.GeoDataFrame): GeoDataFrame containing the transect.
            offset_distance (Optional[int]): Distance for creating an offset rectangle for clipping (required if rotate=True).
            target_axis (Optional[Literal]): Method for determining rotation angle (required if rotate=True).
            y_shape (Optional[int]): The y shape of the dataset (required if rotate=True).
            x_shape (Optional[int]): The x shape of the dataset (required if rotate=True).
            resampling (Optional[Resampling]): Resampling method for interpolation during rotation (required if rotate=True).
            resolution (Optional[int]): The resolution of the dataset (required if rotate=True).
            rotate (bool): Whether to rotate the dataset to align with the transect.

        Returns:
            xr.Dataset: The rotated, clipped, and interpolated dataset.
        """
        TRANSECT_LENGTH = 2000
        # Altough its currently hard-coded here below, the rotation margin can be
        # computed like: np.sqrt(OFFSET_DISTANCE**2 + (TRANSECT_LENGTH / 2) ** 2)
        MAX_TRANSECT_RADIUS = 1100  # ()

        if transect.empty:
            raise ValueError("The input transect GeoDataFrame is empty.")

        if dataset.rio.crs is None:
            raise ValueError("The dataset does not have a valid CRS.")

        if rotate:
            if target_axis is None:
                raise ValueError("When rotate=True, target_axis is required.")

            if offset_distance is None:
                raise ValueError("When rotate=True, offset_distance is required.")

            if resolution is None:
                raise ValueError("When rotate=True, resolution is required.")

            # Create ROI from transect
            roi = transect_region(transect, offset_distance)
            roi_geom = Geometry(roi.geometry.item(), crs=roi.crs).to_crs(
                dataset.rio.crs
            )

            # Buffer transect and clip before rotating
            roi_buffered = (
                gpd.GeoSeries.from_xy(transect.lon, transect.lat, crs=4326)
                .to_crs(transect.utm_epsg.item())
                .buffer(MAX_TRANSECT_RADIUS)
                .to_frame("geometry")
                .to_crs(4326)
            )
            roi_buffered = gpd.GeoDataFrame(
                geometry=[shapely.box(*roi_buffered.total_bounds)], crs=roi_buffered.crs
            )
            roi_buffered_geom = Geometry(
                roi_buffered.geometry.item(), crs=roi_buffered.crs
            ).to_crs(dataset.rio.crs)

            dataset = dataset.odc.crop(roi_buffered_geom)

            # Rotate
            gbox = GeoBox.from_geopolygon(
                roi_buffered_geom, resolution=resolution
            ).flipy()
            rotation_angle = get_rotation_angle_from_transect(
                transect, target_axis=target_axis
            )
            dataset = dataset.odc.reproject(
                gbox.rotate(-rotation_angle), resampling=resampling
            )

            # Clip and trim outer NaNs
            # Although it might be more intuitive to use dataset.odc.crop(roi_geom) here,
            # we cannot do that because the raster is already rotated. If we use crop we
            # get the wrong transformation matrix. Therefore we use mask and then manually
            # trim the outer nan values.
            dataset = dataset.odc.mask(roi_geom)
            dataset = trim_outer_nans(dataset)  # type: ignore

        else:
            # Validate required arguments for non-rotation
            if None in (y_shape, x_shape):
                raise ValueError("When rotate=False, y_shape and x_shape are required.")

            if y_shape != x_shape:
                raise ValueError(
                    "For rotate=False, y_shape and x_shape must be equal to define a square ROI."
                )

            # Create bounding box for non-rotated clipping
            bbox = (
                gpd.GeoSeries.from_xy(transect.lon, transect.lat, crs=4326)
                .to_crs(transect.utm_epsg.item())
                .buffer(TRANSECT_LENGTH / 2)
                .to_crs(4326)
                .total_bounds
            )
            roi = gpd.GeoDataFrame(geometry=[shapely.box(*bbox)], crs=4326)
            roi_geom = Geometry(roi.geometry.item(), crs=roi.crs).to_crs(
                dataset.rio.crs
            )

            dataset = dataset.odc.crop(roi_geom, apply_mask=False)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Transform that is non-rectilinear or with rotation found. Unable to recalculate.",
            )

            # Interpolate dataset to the desired shape
            dataset = interpolate_raster(
                dataset, y_shape=y_shape, x_shape=x_shape, resampling=resampling
            )

        return dataset


def process_row(sample: gpd.GeoDataFrame) -> xr.Dataset | None:
    """Converts a STAC GeoParquet row into an Xarray dataset."""
    try:
        ds = odc.stac.load(stac_geoparquet.to_item_collection(sample)).squeeze(
            drop=True
        )
        ds = ds.drop_vars("spatial_ref", errors="ignore")

        # These are STAC GeoParquet that cannot be added as coordinates to the dataset
        sample = sample.drop(
            columns=[
                "type",
                "stac_version",
                "stac_extensions",
                "bbox",
                "links",
                "assets",
                "collection",
                "created",
                "datetime",
            ]
        )
        # Add the remaining metadata as coordinates
        meta = sample.iloc[0].to_dict()
        meta["stac_id"] = meta.pop("id", None)
        coords = {k: (("uuid",), [v]) for k, v in meta.items()}
        coords["uuid"] = (("uuid",), [meta["uuid"]])
        return ds.assign_coords(**coords)  # type: ignore

    except Exception as e:
        print(f"Error processing {sample.get('id', 'UNKNOWN')}: {e}")
        return None


def load_stac_xr(df: gpd.GeoDataFrame, use_dask=False) -> xr.Dataset:
    """Loads STAC GeoParquet training items into an Xarray dataset, optionally using Dask Bag for efficiency."""

    if use_dask:
        bag = db.from_sequence([df.iloc[[i]] for i in range(len(df))])
        delayed_datasets = bag.map(process_row)
        datasets = delayed_datasets.compute()

    else:
        datasets = []
        for i in range(len(df)):
            sample = df.iloc[[i]]
            ds = process_row(sample)
            datasets.append(ds)

    if not datasets:
        raise ValueError("No valid datasets found.")

    return xr.concat(
        datasets, dim="uuid", join="override", combine_attrs="no_conflicts"
    )
