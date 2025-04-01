from collections.abc import Callable
from typing import cast

import geopandas as gpd
import xarray as xr
from shapely.wkt import loads as wkt_loads

from coastpy.eo.mask import nodata_mask
from coastpy.eo.typology import encode_region_of_interest


class BaseTransform:
    """
    Base class for transform transformations on xarray objects.

    Attributes:
        variables (List[str]): List of variable names to apply the transformation to.
        group_dim (Optional[str]): Dimension over which to apply the transformation. If None, apply globally.
        suffix (str): Suffix to append to transformed variable names.
    """

    suffix: str = ""  # Default suffix, subclasses should override
    is_dataset_transform: bool = False

    def __init__(self, variables: list[str], group_dim: str | None = None):
        """
        Initializes the BaseTransform.

        Args:
            variables (List[str]): Variables to apply transformations to.
            group_dim (Optional[str]): Dimension over which to apply the transformation. If None, apply globally.
        """
        self.variables = variables
        self.group_dim = group_dim

    def _apply_function(self, data: xr.DataArray, func: Callable) -> xr.DataArray:
        """
        Applies a function to the data, optionally grouping by the specified dimension.

        Args:
            data (xr.DataArray): The input data array.
            func (Callable): The function to apply.

        Returns:
            xr.DataArray: The transformed data array.
        """
        if self.group_dim and self.group_dim in data.dims:
            return data.groupby(self.group_dim).map(func)
        else:
            return func(data)

    def transform(self, data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        Applies the transformation to the specified variables in the dataset or directly to a DataArray.

        Args:
            data (Union[xr.Dataset, xr.DataArray]): The input dataset or data array.

        Returns:
            Union[xr.Dataset, xr.DataArray]: The transformed dataset or data array.
        """
        if isinstance(data, xr.Dataset):
            transformed_vars = {}
            for var in self.variables:
                if var in data.data_vars:
                    transformed_vars[f"{var}{self.suffix}"] = self._apply_function(
                        data[var], self._transformation
                    )
                else:
                    raise ValueError(f"Variable '{var}' not found in the dataset.")
            return xr.Dataset(transformed_vars, coords=data.coords)

        elif isinstance(data, xr.DataArray):
            if data.name in self.variables:
                return self._apply_function(data, self._transformation)
            else:
                raise ValueError(
                    f"DataArray name '{data.name}' not in specified variables {self.variables}."
                )
        else:
            raise TypeError("Input must be an xarray Dataset or DataArray.")

    def _transformation(self, data: xr.DataArray) -> xr.DataArray:
        """
        The transformation function to be implemented by subclasses.

        Args:
            data (xr.DataArray): The input data array.

        Returns:
            xr.DataArray: The transformed data array.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class NoDataMaskTransform(BaseTransform):
    """Transform transformation to compute NoData masks."""

    suffix = "_mask_nodata"

    def _transformation(self, data: xr.DataArray) -> xr.DataArray:
        """Computes a NoData mask for the input data array."""
        da = cast(xr.DataArray, nodata_mask(data))
        da = da.astype("int16")
        return da


class RelativeTransform(BaseTransform):
    """Transform transformation to compute relative values per group."""

    suffix = "_rel"

    def __init__(self, variables: list[str], group_dim: str | None = None):
        """
        Initializes the relative transformation.

        Args:
            variables (list[str]): Variables to apply transformations to.
            group_dim (Optional[str]): The dimension over which to compute relative values.
                                       Defaults to None (global transformation).
        """
        super().__init__(variables, group_dim)

    def _transformation(self, data: xr.DataArray) -> xr.DataArray:
        """Computes relative values by subtracting the local minimum."""
        return data - data.min()


class RegionOfInterestTransform(BaseTransform):
    """
    Compute region of interest (ROI) mask from transect geometries per chip.
    """

    is_dataset_transform = True
    suffix = "region_of_interest"

    def __init__(self, group_dim: str = "uuid"):
        """
        Initializes the transform.

        Args:
            group_dim (str): Dimension over which to apply ROI logic (e.g., "uuid").
        """
        super().__init__(variables=[], group_dim=group_dim)

    def transform(self, data: xr.Dataset) -> xr.DataArray:
        """
        Applies the ROI transform using groupby map, using a reference variable for spatial dims.

        Args:
            data (xr.Dataset): Dataset containing the chips and transect geometry.

        Returns:
            xr.DataArray: Binary region of interest mask.
        """
        if self.group_dim not in data.dims:
            msg = f"Dataset must contain dimension '{self.group_dim}'."
            raise ValueError(msg)

        # Use the first data variable as the spatial container
        first_var = next(iter(data.data_vars))
        ref_array = data[first_var]

        def compute_mask_from_array(ref_chip: xr.DataArray) -> xr.DataArray:
            # Get corresponding dataset slice
            sel_dict = {self.group_dim: ref_chip[self.group_dim].item()}
            chip = data.sel(sel_dict)

            # Extract transect geometry
            wkt_str = chip.transect_geometry.item()
            geometry = wkt_loads(wkt_str)
            transect = gpd.GeoDataFrame(geometry=[geometry], crs=4326)

            mask = encode_region_of_interest(transect=transect, ds=chip)
            return mask

        # Apply the ROI computation grouped by chip
        roi_mask = ref_array.groupby(self.group_dim).map(compute_mask_from_array)

        # Set name so it can be inserted into the dataset later
        roi_mask.name = "region_of_interest"
        return roi_mask


class TransformFactory:
    """Factory class for creating transform instances with optional parameters."""

    _registry: dict[str, type[BaseTransform]] = {
        "nodata_mask": NoDataMaskTransform,
        "relative": RelativeTransform,
        "region_of_interest": RegionOfInterestTransform,
    }

    @staticmethod
    def create(transform_type: str | type[BaseTransform], **kwargs) -> BaseTransform:
        """
        Creates an instance of the requested transform.

        Args:
            transform_type (str | Type[BaseTransform]): The type of transform to create.
            kwargs: Additional parameters for the transform (e.g., `group_dim="uuid"`).

        Returns:
            BaseTransform: An instance of the requested transform.

        Raises:
            ValueError: If an invalid transform type is provided.
        """
        if isinstance(transform_type, str):
            transform_type = transform_type.lower()
            if transform_type not in TransformFactory._registry:
                raise ValueError(
                    f"Invalid transform type '{transform_type}'. Available options: "
                    f"{', '.join(TransformFactory._registry.keys())}"
                )
            transform_class = TransformFactory._registry[transform_type]
        else:
            transform_class = transform_type

        return transform_class(**kwargs)


def create_transforms(
    transform_mapping: dict[str, list[dict]],
) -> dict[str, list[BaseTransform]]:
    """
    Creates transforms based on the parsed transform mapping.

    Returns:
        dict: Mapping from variable name or '__dataset__' to list of transform objects.
    """
    transforms: dict[str, list[BaseTransform]] = {}

    for key, config_list in transform_mapping.items():
        transforms[key] = []
        for cfg in config_list:
            cfg_ = cfg.copy()
            transform_type = cfg_.pop("type")

            # Only pass variables if present
            transforms[key].append(TransformFactory.create(transform_type, **cfg))

    return transforms


def apply_transforms(
    data: xr.Dataset | xr.DataArray,
    transforms: dict[str, list[BaseTransform]],
    keep_attrs: bool = True,
    merge: bool = True,
) -> xr.Dataset | xr.DataArray:
    def _apply_transform(transform: BaseTransform, da: xr.DataArray) -> xr.DataArray:
        transformed = transform.transform(da)
        if keep_attrs:
            transformed.attrs = da.attrs  # Preserve metadata
        return cast(xr.DataArray, transformed)

    if isinstance(data, xr.DataArray):
        return transforms.get(data.name, lambda x: x)(data)  # type: ignore

    elif isinstance(data, xr.Dataset):
        transformed_ds = xr.Dataset(
            coords=data.coords, attrs=data.attrs if keep_attrs else {}
        )

        for var, transform_list in transforms.items():
            for transform in transform_list:
                if transform.is_dataset_transform:
                    transformed_ds[transform.suffix] = transform.transform(data)
                elif var in data.data_vars:
                    transformed = transform.transform(data[var])
                    transformed_ds[f"{var}{transform.suffix}"] = transformed

        return data.merge(transformed_ds) if merge else transformed_ds

    else:
        raise TypeError("Input must be an xarray Dataset or DataArray.")
