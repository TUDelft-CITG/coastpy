from enum import Enum

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
    Generate a binary mask for nodata values in a Dataset or DataArray.

    Args:
        data (xarray.Dataset or xarray.DataArray): Input data.

    Returns:
        xarray.Dataset or xarray.DataArray: Binary mask (`True` for nodata pixels, `False` otherwise).
    """
    # NOTE: In the old version we also checked for these
    # mask = ds.isnull()  # catch nan, np.nan, None
    # mask = mask + ds.isin([np.inf, -np.inf])  # catch np.inf and -np.inf
    # mask = mask + ds == 0  # catch zeros to avoid division by zero error

    if isinstance(data, xr.Dataset):
        return data.map(nodata_mask)

    if isinstance(data, xr.DataArray):
        nodata = None
        try:  # noqa: SIM105
            nodata = data.rio.nodata  # rioxarray nodata
        except AttributeError:
            pass

        if nodata is None:
            nodata = data.attrs.get("nodata", data.attrs.get("_FillValue", None))

        if nodata is None:
            return xr.zeros_like(data, dtype=bool)  # No nodata value defined

        return data == nodata

    msg = f"Unsupported input type: {type(data)}"
    raise TypeError(msg)


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
        # NOTE: consider using crop?
        # return data.odc.crop(geometry, apply_mask=True)
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


if __name__ == "__main__":
    import numpy as np
    import xarray as xr

    # Create an example dataset
    data = xr.Dataset(
        {
            "var1": xr.DataArray(
                np.array([[1, 2, 3], [4, 5, -999]]),  # Includes nodata value (-999)
                dims=["x", "y"],
                name="var1",
                attrs={"nodata": -999},
            ),
            "var2": xr.DataArray(
                np.array([[10, 20, 30], [40, 50, 60]]),
                dims=["x", "y"],
                name="var2",
            ),
            "SCL": xr.DataArray(
                np.array(
                    [[0, 4, 6], [9, 10, 11]]
                ),  # SCL layer with Scene Classification values
                dims=["x", "y"],
                name="SCL",
            ),
        }
    )

    # 1. Apply Nodata Mask
    nodata = nodata_mask(data)
    print("Nodata Mask:")
    print(nodata)

    # Apply the nodata mask to the dataset
    masked_nodata = apply_mask(data, nodata)
    print("\nDataset with Nodata Mask Applied:")
    print(masked_nodata)

    # 2. Apply Numeric Mask
    values_to_mask = [5, 30, 60]  # Mask specific values
    numeric = numeric_mask(data, values_to_mask)
    print("\nNumeric Mask:")
    print(numeric)

    # Apply the numeric mask to the dataset
    masked_numeric = apply_mask(data, numeric)
    print("\nDataset with Numeric Mask Applied:")
    print(masked_numeric)

    # 3. Apply SCL Mask
    scl_classes_to_mask = ["Vegetation", 11]  # Mask specific SCL classes
    scl = scl_mask(data["SCL"], scl_classes_to_mask)
    print("\nSCL Mask:")
    print(scl)

    # Apply the SCL mask to the entire dataset
    masked_scl = apply_mask(data, scl)
    print("\nDataset with SCL Mask Applied:")
    print(masked_scl)
