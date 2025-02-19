from collections.abc import Callable
from typing import cast

import xarray as xr

from coastpy.eo.mask import nodata_mask


class BaseTransform:
    """
    Base class for transform transformations on xarray objects.

    Attributes:
        variables (List[str]): List of variable names to apply the transformation to.
        group_dim (Optional[str]): Dimension over which to apply the transformation. If None, apply globally.
        suffix (str): Suffix to append to transformed variable names.
    """

    suffix: str = ""  # Default suffix, subclasses should override

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


class TransformFactory:
    """Factory class for creating transform instances with optional parameters."""

    _registry: dict[str, type[BaseTransform]] = {
        "nodata_mask": NoDataMaskTransform,
        "relative": RelativeTransform,
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
    transform_mapping: dict[str, str | type[BaseTransform] | dict | list],
) -> dict[str, list[BaseTransform]]:
    """
    Creates transform transformations based on a band-to-transform mapping.

    Args:
        transform_mapping (Dict[str, Union[str, Type[BaseTransform], dict, list]]):
            - Dictionary mapping variables to transform types.
            - Each value can be:
              - A **string** (e.g., `"relative"`)
              - A **transform class** (e.g., `RelativeTransform`)
              - A **dictionary** with optional parameters (e.g., `{"type": "relative", "group_dim": "uuid"}`)
              - A **list** of any of the above for multiple transformations per variable.

    Returns:
        Dict[str, list[BaseTransform]]: Dictionary mapping variables to lists of instantiated transforms.
    """
    transforms = {}

    for band, config in transform_mapping.items():
        # Ensure config is always a list (to handle single or multiple transforms uniformly)
        config_list = config if isinstance(config, list) else [config]
        transforms[band] = []

        for item in config_list:
            # If a dictionary, extract the transform type safely
            if isinstance(item, dict):
                item_copy = item.copy()  # 🛠 Copy to prevent modifying original dict
                if "type" not in item_copy:
                    raise ValueError(
                        f"Missing 'type' key in transform config for '{band}': {item}"
                    )
                transform_type = item_copy.pop("type")
                transforms[band].append(
                    TransformFactory.create(
                        transform_type, variables=[band], **item_copy
                    )
                )
            else:
                # If it's a string or class, create the transform directly
                transforms[band].append(TransformFactory.create(item, variables=[band]))

    return transforms


def apply_transforms(
    data: xr.Dataset | xr.DataArray,
    transforms: dict[str, list[BaseTransform]],
    keep_attrs: bool = True,
    merge: bool = True,
) -> xr.Dataset | xr.DataArray:
    """
    Apply transformations to an Xarray Dataset or DataArray.

    Args:
        data (xr.Dataset | xr.DataArray): The input dataset or data array.
        transforms (Dict[str, list[BaseTransform]]): Dictionary mapping variables to lists of transformations.
        keep_attrs (bool, optional): Whether to retain attributes in the output. Defaults to True.
        merge (bool, optional): If True, merges transformed variables with the original dataset.

    Returns:
        xr.Dataset | xr.DataArray: Transformed dataset or data array.
    """

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
            if var in data.data_vars:
                for transform in transform_list:
                    transformed_ds[f"{var}{transform.suffix}"] = _apply_transform(
                        transform, data[var]
                    )

        return data.merge(transformed_ds) if merge else transformed_ds

    else:
        raise TypeError("Input must be an xarray Dataset or DataArray.")
