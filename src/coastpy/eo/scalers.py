import numpy as np
import xarray as xr

from coastpy.eo.band_statistics import scalers_dict


class BaseScaler:
    """Base class for all scalers ensuring consistency in transformation."""

    def __init__(self, stats: dict, **kwargs):  # noqa
        """
        Initialize a scaler with precomputed statistics.

        Args:
            stats (dict): Dictionary containing necessary statistics for transformation.
            kwargs: Additional parameters passed from the user.
        """
        self.required_stats = self.get_required_stats()
        missing_stats = [s for s in self.required_stats if s not in stats]
        if missing_stats:
            raise ValueError(f"Missing required statistics: {missing_stats}")
        self.stats = stats

    @classmethod
    def get_required_stats(cls):
        """Return the list of required statistics for the scaler."""
        return []

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Apply transformation (implemented in subclasses)."""
        raise NotImplementedError("Each scaler must implement the transform method.")


class MinMaxScaler(BaseScaler):
    """Min-Max Scaling to range [0,1]."""

    @classmethod
    def get_required_stats(cls):
        return ["min", "max"]

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Scale values to the range [0,1] using precomputed min/max."""
        if np.isnan(self.stats["min"]) or np.isnan(self.stats["max"]):
            raise ValueError("MinMaxScaler: `min` and `max` must not be NaN.")

        return (
            (data - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
        ).clip(0, 1)


class RobustMinMaxScaler(BaseScaler):
    """Min-Max Scaling using robust percentiles to handle outliers."""

    @classmethod
    def get_required_stats(cls):
        return ["clip_min", "clip_max"]

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Scale values using precomputed robust min/max (percentiles)."""
        if np.isnan(self.stats["clip_min"]) or np.isnan(self.stats["clip_max"]):
            raise ValueError(
                "RobustMinMaxScaler: `clip_min` and `clip_max` must not be NaN."
            )

        return (
            (data - self.stats["clip_min"])
            / (self.stats["clip_max"] - self.stats["clip_min"])
        ).clip(0, 1)


class StandardScaler(BaseScaler):
    """Standard scaling using mean and standard deviation."""

    @classmethod
    def get_required_stats(cls):
        return ["mean", "std_dev"]

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Apply standard normalization using precomputed mean and std."""
        if np.isnan(self.stats["mean"]) or np.isnan(self.stats["std_dev"]):
            raise ValueError("StandardScaler: `mean` and `std_dev` must not be NaN.")

        return (data - self.stats["mean"]) / self.stats["std_dev"]


class ScalerFactory:
    """Factory class for creating scaler instances."""

    _registry: dict[str, type[BaseScaler]] = {
        "min_max": MinMaxScaler,
        "robust_min_max": RobustMinMaxScaler,
        "standard": StandardScaler,
    }

    @staticmethod
    def create(
        scaler_type: str | type[BaseScaler], stats: dict, **kwargs
    ) -> BaseScaler:
        """
        Creates an instance of the requested scaler.

        Args:
            scaler_type (str | Type[BaseScaler]): The type of scaler to create.
            stats (dict): Precomputed statistics required for the scaler.
            kwargs: Additional parameters for the scaler.

        Returns:
            BaseScaler: An instance of the requested scaler.

        Raises:
            ValueError: If an invalid scaler type is provided.
        """
        if isinstance(scaler_type, str):
            scaler_type = scaler_type.lower()
            if scaler_type not in ScalerFactory._registry:
                raise ValueError(
                    f"Invalid scaler type '{scaler_type}'. Available options: "
                    f"{', '.join(ScalerFactory._registry.keys())}"
                )
            scaler_class = ScalerFactory._registry[scaler_type]
        else:
            scaler_class = scaler_type

        return scaler_class(stats=stats, **kwargs)


def create_scalers(
    band_scaler_mapping: dict[str, str | type[BaseScaler]],
) -> dict[str, BaseScaler]:
    """
    Creates scalers based on band-to-scaler mapping.

    Args:
        band_scaler_mapping (Dict[str, Union[str, Type[BaseScaler]]]): Dictionary mapping bands to scaler types.

    Returns:
        Dict[str, BaseScaler]: Dictionary of instantiated scalers.

    Raises:
        ValueError: If a required statistic is missing for a scaler.
    """
    scalers = {}
    for band, scaler_type in band_scaler_mapping.items():
        if band not in scalers_dict:
            raise ValueError(f"No statistics available for band: {band}")
        scalers[band] = ScalerFactory.create(scaler_type, stats=scalers_dict[band])
    return scalers


def apply_scaling(
    data: xr.Dataset | xr.DataArray,
    scalers: dict[str, BaseScaler],
    keep_attrs: bool = True,
    inplace: bool = True,
) -> xr.Dataset | xr.DataArray:
    """
    Apply scaling to an Xarray Dataset or DataArray, with the option to keep attributes and modify in-place.

    Args:
        data (xr.Dataset | xr.DataArray): The input dataset or data array.
        scalers (dict[str, BaseScaler]): Dictionary of scalers to apply per variable.
        keep_attrs (bool, optional): Whether to retain attributes in the output. Defaults to True.
        inplace (bool, optional): Whether to modify the input dataset in-place. Defaults to True.

    Returns:
        xr.Dataset | xr.DataArray: The dataset with scaled variables added.
    """

    def _apply_scaler(scaler: BaseScaler, da: xr.DataArray) -> xr.DataArray:
        """Apply scaling to a single DataArray."""
        scaled = scaler.transform(da)
        if keep_attrs:
            scaled.attrs = da.attrs  # Preserve attributes
        return scaled

    if isinstance(data, xr.DataArray):
        return scalers.get(data.name, lambda x: x)(data)  # type: ignore

    elif isinstance(data, xr.Dataset):
        scaled_vars = {
            var: _apply_scaler(scalers[var], da)
            for var, da in data.data_vars.items()
            if var in scalers
        }

        if inplace:
            return data.assign(**scaled_vars)  # type: ignore
        else:
            return xr.Dataset(
                scaled_vars, coords=data.coords, attrs=data.attrs if keep_attrs else {}
            )

    else:
        raise TypeError("Input must be an Xarray Dataset or DataArray.")
