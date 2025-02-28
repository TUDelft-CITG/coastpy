from enum import Enum
from functools import wraps
from typing import Literal

import numpy as np
import odc.geo
import odc.stac  # noqa
import rioxarray  # noqa
import xarray as xr
from odc.geo import geom


class SceneClassification(Enum):
    """Enum for Sentinel-2 Scene Classification Classes."""

    NO_DATA = 0
    SATURATED_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    BARE_SOILS = 5
    WATER = 6
    CLOUDS_LOW_PROBABILITY = 7
    CLOUDS_MEDIUM_PROBABILITY = 8
    CLOUDS_HIGH_PROBABILITY = 9
    CIRRUS = 10
    SNOW_ICE = 11

    @classmethod
    def from_string(cls, name: str):
        """Get numeric value from class name."""
        for item in cls:
            if item.name.lower() == name.lower().replace(" ", "_"):
                return item.value
        msg = f"Invalid class name '{name}'. Must be one of {[e.name for e in cls]}."
        raise ValueError(msg)

    @classmethod
    def all_classes(cls):
        """List all available class names."""
        return [item.name for item in cls]


def keep_rio_attrs(exclude_attrs: dict[str, list[str]] | None = None):
    """
    Preserve rioxarray-specific geospatial attributes (CRS, nodata, encoding)
    during xarray operations, with an option to exclude specific attributes or keys.

    Args:
        exclude_attrs (dict[str, list[str]], optional): Attributes to exclude from restoration.
            Keys are attribute names (e.g., "nodata", "encoding"), and values are lists of
            specific keys to exclude for nested attributes (e.g., ["scale_factor", "_FillValue"]).
            Defaults to None.

    Returns:
        Callable: The decorator to wrap a function with geospatial attribute preservation.
    """
    exclude_attrs = exclude_attrs or {}

    def decorator(func):
        @wraps(func)
        def wrapper(data: xr.Dataset | xr.DataArray, *args, **kwargs):
            if not isinstance(data, (xr.Dataset | xr.DataArray)):
                raise TypeError(
                    f"Input must be an xarray Dataset or DataArray, not {type(data)}"
                )

            # Preserve CRS and metadata
            crs = getattr(data.rio, "crs", None)
            if isinstance(data, xr.Dataset):
                metadata = {
                    var_name: {
                        "nodata": getattr(data[var_name].rio, "nodata", None),
                        "encoding": data[var_name].encoding.copy(),
                    }
                    for var_name in data.data_vars
                }
            else:  # Handle DataArray
                metadata = {
                    "nodata": getattr(data.rio, "nodata", None),
                    "encoding": data.encoding.copy(),
                }

            # Apply the wrapped function
            result = func(data, *args, **kwargs)

            # Remove attributes excluded from restoration
            def remove_excluded_attrs(variable, attrs_to_exclude):
                """
                Remove excluded attributes from the variable's rio or encoding metadata.
                """
                if "nodata" in attrs_to_exclude:
                    variable.attrs.pop(
                        "_FillValue", None
                    )  # Remove _FillValue if present
                    variable.attrs.pop("nodata", None)  # Remove nodata if present
                if "encoding" in attrs_to_exclude:
                    for key in attrs_to_exclude["encoding"]:
                        variable.encoding.pop(key, None)

            # Restore CRS and metadata, excluding specific attributes
            if isinstance(result, xr.Dataset):
                if crs:
                    result = result.rio.write_crs(crs)
                for var_name, meta in metadata.items():
                    if var_name in result.data_vars:
                        if "nodata" not in exclude_attrs and meta["nodata"] is not None:
                            result[var_name] = result[var_name].rio.write_nodata(
                                meta["nodata"]
                            )
                        elif "nodata" in exclude_attrs:
                            remove_excluded_attrs(result[var_name], exclude_attrs)

                        if "encoding" in exclude_attrs:
                            encoding_to_restore = {
                                k: v
                                for k, v in meta["encoding"].items()
                                if k not in exclude_attrs["encoding"]
                            }
                            result[var_name].encoding.update(encoding_to_restore)
                        else:
                            result[var_name].encoding.update(meta["encoding"])
            elif isinstance(result, xr.DataArray):
                if crs:
                    result = result.rio.write_crs(crs)
                if "nodata" not in exclude_attrs and metadata["nodata"] is not None:
                    result = result.rio.write_nodata(metadata["nodata"])
                elif "nodata" in exclude_attrs:
                    remove_excluded_attrs(result, exclude_attrs)

                if "encoding" in exclude_attrs:
                    encoding_to_restore = {
                        k: v
                        for k, v in metadata["encoding"].items()
                        if k not in exclude_attrs["encoding"]
                    }
                    result.encoding.update(encoding_to_restore)
                else:
                    result.encoding.update(metadata["encoding"])

            return result

        return wrapper

    return decorator


def apply_mask(
    data: xr.Dataset | xr.DataArray,
    mask: xr.Dataset | xr.DataArray,
    keep_attrs: bool = True,
) -> xr.Dataset | xr.DataArray:
    """
    Apply a binary mask to an xarray Dataset or DataArray.

    Args:
        data (xarray.Dataset or xarray.DataArray): The data to mask.
        mask (xarray.Dataset or xarray.DataArray): Binary mask (`True` for masked pixels, `False` otherwise).
        keep_attrs (bool, optional): Whether to retain attributes in the output. Defaults to True.

    Returns:
        xarray.Dataset or xarray.DataArray: The masked data.

    Raises:
        ValueError: If `data` and `mask` are datasets and their variables do not match.
    """
    if isinstance(data, xr.Dataset) and isinstance(mask, xr.Dataset):
        if set(data.data_vars) != set(mask.data_vars):
            msg = "The variables in `data` and `mask` datasets must match."
            raise ValueError(msg)
        return data.map(lambda da: apply_mask(da, mask[da.name], keep_attrs=keep_attrs))

    if isinstance(data, xr.Dataset) and isinstance(mask, xr.DataArray):
        return data.map(lambda da: apply_mask(da, mask, keep_attrs=keep_attrs))

    if isinstance(data, xr.DataArray) and isinstance(mask, xr.DataArray):
        masked = data.where(~mask, other=np.nan)
        if keep_attrs:
            masked.attrs = data.attrs
        return masked

    msg = f"Unsupported combination of input types: {type(data)} and {type(mask)}"
    raise TypeError(msg)


def nodata_mask(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Generate a binary mask for nodata values, NaNs, and infinities in a Dataset or DataArray.

    Args:
        data (xarray.Dataset or xarray.DataArray): Input data.

    Returns:
        xarray.Dataset or xarray.DataArray: Binary mask (`True` for nodata pixels, `False` otherwise).
    """
    # NOTE: Adding masks for nulls, invalids and zero's is a a bit experimental, so leave
    # the comment here.
    # mask = ds.isnull()  # catch nan, np.nan, None
    # mask = mask + ds.isin([np.inf, -np.inf])  # catch np.inf and -np.inf
    # mask = mask + ds == 0  # catch zeros to avoid division by zero error

    if isinstance(data, xr.Dataset):
        return data.map(nodata_mask)

    if isinstance(data, xr.DataArray):
        nodata = get_nodata(data)

        mask = data.isnull()  # Catch NaN values
        mask |= data.isin([np.inf, -np.inf])  # Catch positive/negative infinity

        if nodata is not None:
            mask |= data == nodata

        return mask

    raise TypeError(f"Unsupported input type: {type(data)}")


def numeric_mask(
    data: xr.Dataset | xr.DataArray, values: list[int]
) -> xr.Dataset | xr.DataArray:
    """
    Generate a binary mask for specific numeric values in a Dataset or DataArray.

    Args:
        data (xarray.Dataset or xarray.DataArray): Input data.
        values (list[int]): List of numeric values to mask.

    Returns:
        xarray.Dataset or xarray.DataArray: Binary mask (`True` for specified values, `False` otherwise).
    """
    if isinstance(data, xr.Dataset):
        return data.map(lambda da: numeric_mask(da, values))

    if isinstance(data, xr.DataArray):
        return data.isin(values)

    msg = f"Unsupported input type: {type(data)}"
    raise TypeError(msg)


@keep_rio_attrs()
def geometry_mask(
    data: xr.Dataset | xr.DataArray,
    geometry: geom.Geometry,
    invert: bool = False,
    all_touched: bool = True,
) -> xr.Dataset | xr.DataArray:
    """
    Generate a mask for an xarray Dataset or DataArray based on a geometry.

    Args:
        data (xr.Dataset | xr.DataArray): The input data to mask.
        geometry (odc.geo.geom.Geometry): The geometry to use as the mask.
        invert (bool, optional): Whether to invert the mask (mask inside instead of outside).
            Defaults to False.
        all_touched (bool, optional): Whether to include all pixels touched by the geometry.
            Defaults to True.
        keep_attrs (bool, optional): Whether to retain the attributes of the input data.
            Defaults to True.

    Returns:
        xr.Dataset | xr.DataArray: The masked data.

    Raises:
        TypeError: If the input data is not an xarray Dataset or DataArray.
    """

    if isinstance(data, xr.Dataset):
        return data.map(lambda da: geometry_mask(da, geometry, invert, all_touched))

    if isinstance(data, xr.DataArray):
        return data.odc.mask(geometry, invert=invert, all_touched=all_touched)

    msg = f"Unsupported input type: {type(data)}"
    raise TypeError(msg)


def scl_mask(
    data: xr.DataArray | xr.Dataset,
    to_mask: list[str | int | SceneClassification],
) -> xr.DataArray:
    """
    Generate a binary mask for Sentinel-2 Scene Classification (SCL) values or classes.

    Args:
        data (xarray.Dataset or xarray.DataArray): Input data that contians or is the Sentinel 2 SCL layer.
        to_mask (list[str | int]): List of class names or numeric values to mask.

    Returns:
        xarray.DataArray: Binary mask (`True` for specified SCL values, `False` otherwise).
    """
    if isinstance(data, xr.Dataset):
        if "SCL" not in data.data_vars:
            msg = "Dataset must contain an 'SCL' variable."
            raise ValueError(msg)
        data = data["SCL"]

    numeric_to_mask = []
    for item in to_mask:
        if isinstance(item, int):
            if item not in [cls.value for cls in SceneClassification]:
                msg = f"Invalid numeric value '{item}' in `to_mask`."
                raise ValueError(msg)
            numeric_to_mask.append(item)
        elif isinstance(item, str):
            numeric_to_mask.append(SceneClassification.from_string(item))
        else:
            msg = f"Invalid type '{type(item)}' in `to_mask`. Must be str or int."
            raise ValueError(msg)

    return numeric_mask(data, numeric_to_mask)  # type: ignore


def get_nodata(
    da: xr.DataArray | xr.Dataset,
) -> float | int | None | dict[str, float | int | None]:
    """
    Get the nodata value(s) from an xarray DataArray or Dataset.

    Args:
        da (xr.DataArray | xr.Dataset): Input data.

    Returns:
        float | int | None: The nodata value for a DataArray.
        dict[str, float | int | None]: A dictionary of nodata values for each variable in a Dataset.

    Raises:
        ValueError: If `band` is specified but not found in the Dataset.
        TypeError: If the input is not an xarray DataArray or Dataset.
    """

    def _get_nodata(dataarray: xr.DataArray) -> float | int | None:
        """
        Extract the nodata value for a single DataArray.

        The priority is:
        1. `nodata` attribute (rioxarray standard)
        2. `_FillValue` attribute (NetCDF standard)
        3. `rio.nodata` property (rioxarray integration with rasterio)
        """
        return (
            dataarray.attrs.get("nodata")
            or dataarray.attrs.get("_FillValue")
            or getattr(dataarray.rio, "nodata", None)
        )

    if isinstance(da, xr.DataArray):
        # For DataArray, return a single nodata value
        return _get_nodata(da)

    if isinstance(da, xr.Dataset):
        # For Dataset, return a dictionary of nodata values for each variable
        nodata_values = {
            var_name: _get_nodata(da[var_name]) for var_name in da.data_vars
        }
        return nodata_values  # type: ignore

    raise TypeError("Input must be an xarray DataArray or Dataset.")


def set_nodata(
    data: xr.DataArray | xr.Dataset,
    nodata: float | int | dict[str, float | int | None] | None,
    nodata_attr: Literal["nodata", "_FillValue"] = "nodata",
) -> xr.DataArray | xr.Dataset:
    """
    Set the nodata value(s) for an Xarray DataArray or Dataset.

    Args:
        data (xr.DataArray | xr.Dataset): Input Xarray object to modify.
        nodata (float | int | dict[str, float | int | None] | None):
            - For a DataArray: A single nodata value.
            - For a Dataset: Either a single nodata value (applied to all variables) or
              a dictionary of {variable_name: nodata_value}.
        target (str, optional): Target attribute to set, either 'nodata' (default) or '_FillValue'.

    Returns:
        xr.DataArray | xr.Dataset: The modified DataArray or Dataset.

    Raises:
        ValueError: If nodata is a dictionary and contains variables not in the Dataset.
        TypeError: If the input is not an xarray DataArray or Dataset.
    """
    if nodata_attr not in ["nodata", "_FillValue"]:
        raise ValueError(
            "The target parameter must be either 'nodata' or '_FillValue'."
        )

    def _set_nodata_for_var(
        var: xr.DataArray, nodata_value: float | int | None
    ) -> xr.DataArray:
        """Set the nodata value for a single DataArray."""
        if nodata_value is not None:
            var.attrs[nodata_attr] = nodata_value
        else:
            var.attrs.pop(nodata_attr, None)  # Remove the attribute if nodata is None

        # Sync with rioxarray if available
        try:
            if nodata_value is not None:
                var.rio.write_nodata(nodata_value, inplace=True)
            else:
                var.rio.update_attrs({"nodata": None}, inplace=True)
        except AttributeError:
            pass  # rioxarray is not available or not in use

        return var

    if isinstance(data, xr.DataArray):
        if isinstance(nodata, dict):
            raise TypeError(
                "For DataArray, nodata must be a single value, not a dictionary."
            )
        return _set_nodata_for_var(data, nodata)

    elif isinstance(data, xr.Dataset):
        if isinstance(nodata, dict):
            # Ensure all keys in the nodata dictionary exist in the Dataset
            invalid_vars = [var for var in nodata if var not in data.data_vars]
            if invalid_vars:
                raise ValueError(f"Variables {invalid_vars} not found in Dataset.")

            # Apply nodata values per variable
            for var_name, nodata_value in nodata.items():
                data[var_name] = _set_nodata_for_var(data[var_name], nodata_value)
        else:
            # Apply the same nodata value to all variables
            for var_name in data.data_vars:
                data[var_name] = _set_nodata_for_var(data[var_name], nodata)

        return data

    else:
        raise TypeError("Input must be an xarray DataArray or Dataset.")
