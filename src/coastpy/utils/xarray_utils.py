import warnings

import numpy as np
import rioxarray  # noqa
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from shapely import Polygon

from coastpy.eo.mask import get_nodata, nodata_mask, set_nodata


def scale(
    data: xr.Dataset | xr.DataArray,
    scale_factor: float,
    nodata: int | float,
    add_offset: int | float | None = None,
    keep_attrs: bool = True,
    scale_vars_to_skip: list[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Scale an xarray Dataset or DataArray to integer format for storage efficiency.

    Args:
        data (xr.Dataset | xr.DataArray): The dataset or data array to scale.
        scale_factor (float): Factor to scale the data.
        nodata (int | float): Nodata value to apply.
        add_offset (float, optional): Offset to add during scaling. Defaults to 0.0.
        keep_attrs (bool, optional): Whether to retain attributes in the output. Defaults to True.
        scale_vars_to_skip (list[str], optional): Variables to exclude from scaling. Defaults to None.

    Returns:
        xr.Dataset | xr.DataArray: Scaled dataset or data array.
    """
    add_offset = add_offset or 0.0
    scale_vars_to_skip = scale_vars_to_skip or []

    def _scale_var(var, keep_attrs=keep_attrs):
        """Scale a single variable."""
        mask = nodata_mask(var)

        # Scale the values
        scaled_var = ((var.where(~mask) - add_offset) * scale_factor).round()

        # Replace invalid values with the new nodata value
        scaled_var = scaled_var.where(~mask, nodata)

        if keep_attrs:
            scaled_var.attrs.update(var.attrs)

        # Set new nodata value in metadata and rio attributes
        scaled_var = set_nodata(scaled_var, nodata)

        scaled_var.attrs.update(
            {
                "scale_factor": 1 / scale_factor,
                "add_offset": add_offset,
                "data_type": "int16",
            }
        )

        # Cast to integer type
        return scaled_var.astype("int16")

    if isinstance(data, xr.Dataset):
        scaled_data = data.copy()
        for var_name, var in data.data_vars.items():
            if var_name not in scale_vars_to_skip:
                scaled_data[var_name] = _scale_var(var)

        # Cast all variables (including ignored ones) to a storage-efficient type
        for var_name in scaled_data.data_vars:
            if var_name in scale_vars_to_skip:
                scaled_data[var_name] = scaled_data[var_name].astype("int16")

        if keep_attrs:
            scaled_data.attrs.update(data.attrs)
        return scaled_data

    if isinstance(data, xr.DataArray):
        if scale_vars_to_skip:
            raise ValueError("scale_vars_to_skip is not applicable for a DataArray.")
        return _scale_var(data)

    raise TypeError(f"Unsupported input type: {type(data)}")


def unscale(
    data: xr.Dataset | xr.DataArray,
    scale_factor: float | None = None,
    add_offset: float | None = None,
    nodata: int | float | None = None,
    keep_attrs: bool = True,
    variables_to_ignore: list[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Unscale an xarray Dataset or DataArray to its original range, ignoring specified variables.

    Args:
        data (xr.Dataset | xr.DataArray): The dataset or data array to unscale.
        scale_factor (float, optional): Factor used to scale the data. If None, attempts to use
                                        metadata attributes (e.g., `raster:scale`, `scale_factor`).
        add_offset (float, optional): Offset used during scaling. Defaults to 0 if not provided.
        nodata (int | float, optional): Nodata value to apply. If None, attempts to use metadata attributes.
        keep_attrs (bool, optional): Whether to retain attributes in the output. Defaults to True.
        variables_to_ignore (list[str], optional): List of variable names to ignore during unscale.
                                                   Defaults to an empty list.

    Returns:
        xr.Dataset | xr.DataArray: Unscaled data.
    """
    if variables_to_ignore is None:
        variables_to_ignore = []

    def _unscale_var(var: xr.DataArray) -> xr.DataArray:
        """Unscale a single variable."""
        # Retrieve scaling factors from metadata
        sf = scale_factor or var.attrs.pop("scale_factor", 1.0)
        offset = add_offset or var.attrs.pop("add_offset", 0.0)
        nodata_ = nodata or get_nodata(var)

        # Mask and scale
        mask = nodata_mask(var)
        unscaled_var = (var.where(~mask) * sf) + offset

        unscaled_var = unscaled_var.where(~mask, nodata_)
        unscaled_var = set_nodata(unscaled_var, nodata_)

        # Retain attributes
        if keep_attrs:
            unscaled_var.attrs.update(var.attrs)

        unscaled_var.attrs.update(
            {
                "scale_factor": 1.0,
                "add_offset": 0.0,
            }
        )

        return unscaled_var  # type: ignore

    if isinstance(data, xr.Dataset):
        unscaled_data = data.copy()
        for var_name, var in data.data_vars.items():
            if var_name not in variables_to_ignore:
                unscaled_data[var_name] = _unscale_var(var)
        if keep_attrs:
            unscaled_data.attrs.update(data.attrs)
        return unscaled_data

    if isinstance(data, xr.DataArray):
        return _unscale_var(data)

    raise TypeError(f"Unsupported input type: {type(data)}")


def make_template(data: xr.DataArray) -> xr.DataArray:
    """
    Create a template DataArray with the same structure as `data` but filled with object data type.

    Args:
        data (xr.DataArray): The input data array for which the template will be created.

    Returns:
        xr.DataArray: A template data array filled with the object data type.

    Raises:
        ValueError: If the input data is not chunked.

    Example:
        >>> arr = xr.DataArray(np.random.rand(4, 3)).chunk((2, 2))
        >>> template = make_template(arr)
        >>> print(template)
        <xarray.DataArray (dim_0: 2, dim_1: 2)>
        dask.array<astype, shape=(2, 2), dtype=object, chunksize=(2, 2), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
    """

    if data.chunks is None:
        msg = "The input data array must be chunked."
        raise ValueError(msg)

    offsets = dict(
        zip(
            data.dims,
            [np.hstack([np.array(0), np.cumsum(x)[:-1]]) for x in data.chunks],
            strict=True,
        )
    )
    template = data.isel(**offsets).astype(object)  # type: ignore

    return template


def extract_and_set_nodata(
    ds: xr.Dataset, vars_with_nodata: list[str], vars_without_nodata: list[str]
) -> tuple[xr.Dataset, float | int]:
    """
    Extracts unique nodata values from the dataset and sets it for the given data variables.

    Args:
        ds (xr.Dataset): Dataset to extract nodata from and set nodata for.
        vars_with_nodata (list[str]): List of data variables that already have nodata values set.
        vars_without_nodata (list[str]): List of data variables to which the common nodata value should be applied.

    Returns:
        tuple[xr.Dataset, float | int]:
            - ds: Dataset with nodata values set for the given data variables.
            - common_nodata_value: The common nodata value set for the data variables.

    Raises:
        ValueError: If no nodata values are found or multiple unique nodata values exist among the specified variables.

    Example:
        >>> ds = xr.Dataset({
        ...    "a": (["x", "y"], np.random.rand(4, 3)),
        ...    "b": (["x", "y"], np.random.rand(4, 3))
        ... })
        >>> ds['a'].rio.write_nodata(-999)
        >>> ds_updated, nodata_val = extract_and_set_nodata(ds, ["a"], ["b"])
        >>> print(nodata_val)
        -999
    """

    nodata_values = [ds[var].rio.nodata for var in vars_with_nodata]
    nodata_values = [val for val in nodata_values if val is not None]
    unique_nodata_values = np.unique(nodata_values)

    # Check if the nodata values are consistent
    if len(unique_nodata_values) == 0:
        msg = "No nodata value found for the specified variables."
        raise ValueError(msg)
    elif len(unique_nodata_values) > 1:
        msg = "Multiple nodata values found. Ensure consistent nodata values."
        raise ValueError(msg)

    common_nodata_value = unique_nodata_values[0]
    for var in vars_without_nodata:
        ds[var] = ds[var].rio.write_nodata(common_nodata_value)

    return ds, common_nodata_value


def raster_center(ds: xr.Dataset) -> tuple[float, float]:
    """
    Calculate the center of a raster dataset.

    Args:
        ds (xr.Dataset): The raster dataset.

    Returns:
        Tuple[float, float]: A tuple representing the (x, y) coordinates of the raster's center.

    Example:
        >>> ds = xr.Dataset(...)
        >>> raster_center(ds)
        (150.0, 100.0)
    """
    width, height = ds.rio.width, ds.rio.height
    x_middle = width / 2
    y_middle = height / 2
    return (x_middle, y_middle)


def rotate_raster(
    ds: xr.Dataset,
    rotation_angle: float,
    resampling: Resampling,
    pivot: tuple[float, float] | None = None,
) -> xr.Dataset:
    """
    Rotate a raster dataset around a pivot point or its center.

    Args:
        ds (xr.Dataset): Raster dataset to be rotated.
        rotation_angle (float): Angle to rotate the raster, in degrees. Positive values represent counterclockwise rotation.
        resampling (Resampling): Resampling method to use during reprojection.
        pivot (Optional[Tuple[float, float]]): (x, y) coordinates of the pivot point. If not provided, the raster's center is used.

    Returns:
        xr.Dataset: Rotated raster dataset.

    Raises:
        UserWarning: If the absolute rotation angle is 45 degrees, which may result in a raster that is not of the expected shape, with a clipped view because the axis should also be swapped.
    """
    if abs(rotation_angle) > 45:
        msg = "The absolute rotation angle larger than 45 degrees, which may result in a raster that clipped. Consider adjusting the rotation in the other direction."
        warnings.warn(
            msg,
            UserWarning,
            stacklevel=2,
        )

    src_transform = ds.rio.transform()
    rotation = Affine.rotation(rotation_angle, pivot=pivot)

    # TODO: Compute the scaling factors for the new grid
    # dst_transform = src_transform * Affine.scale(x_scale, y_scale)

    dst_transform = src_transform * rotation

    # Rescale the y-axis to correct the inversion
    rescale_y = Affine(1, 0, 0, 0, -1, ds.rio.height)
    dst_transform = dst_transform * rescale_y

    ds = ds.rio.reproject(
        dst_crs=ds.rio.crs, transform=dst_transform, resampling=resampling
    )
    ds = ds.rio.write_transform(dst_transform)
    ds = ds.assign_coords(
        {"y": ("y", range(ds.sizes["y"])), "x": ("x", range(ds.sizes["x"]))}
    )
    return ds


def interpolate_raster(
    ds: xr.Dataset,
    y_shape: int,
    x_shape: int,
    resampling: Resampling,
) -> xr.Dataset:
    """
    Interpolates a given raster (xarray Dataset) to a specified shape using Rasterio resampling methods.

    Args:
        ds (xr.Dataset): The input raster to interpolate.
        y_shape (int): Desired number of grid points along the y dimension.
        x_shape (int): Desired number of grid points along the x dimension.
        resampling: rasterio.enums.Resampling: The interpolation method to use.

    Returns:
        xr.Dataset: Interpolated raster with updated geospatial metadata.
    """

    # Compute the target transformation based on the desired shape

    transform = ds.rio.transform()

    # Define the target transform for the new resolution
    target_transform = transform * transform.scale(
        (ds.sizes["x"] / x_shape), (ds.sizes["y"] / y_shape)
    )

    # Create a template for the new shape
    out_shape = (
        (ds.sizes["band"], y_shape, x_shape)
        if "band" in ds.dims
        else (y_shape, x_shape)
    )

    # Reproject the dataset to the new shape using rasterio
    interpolated = ds.rio.reproject(
        ds.rio.crs,
        shape=out_shape,
        transform=target_transform,
        resampling=resampling,
        nodata=np.nan,
    )

    interpolated = interpolated.rio.write_transform(target_transform)

    return interpolated


import xarray as xr


def trim_outer_nans(
    data: xr.DataArray | xr.Dataset,
    nodata: float | int | None = None,
    crop_size: int = 0,
) -> xr.DataArray | xr.Dataset:
    """
    Trim the outer nodata or NaN values from an xarray DataArray or Dataset, returning a bounding box around the data.

    Args:
        data (xr.DataArray | xr.Dataset): Input DataArray or Dataset with potential outer NaN or nodata values.
        nodata (float | int | None): Optional no-data value to use for trimming. Defaults to None, which uses NaN.
        crop_size (int, optional): The number of pixels to crop from the outer edges of the data after trimming.
                                   Defaults to 0 (no additional cropping).

    Returns:
        (xr.DataArray | xr.Dataset): A DataArray or Dataset trimmed of its outer NaN or nodata values, with optional
                                     additional cropping applied.
    """

    # Determine the representative DataArray for NaN or nodata calculation
    ref_data_array = (
        data[next(iter(data.data_vars.keys()))]
        if isinstance(data, xr.Dataset)
        else data
    )
    transform = ref_data_array.rio.transform()

    mask = ref_data_array != nodata if nodata is not None else ~np.isnan(ref_data_array)
    y_valid, x_valid = np.where(mask)

    # If no valid data exists, return the input as-is
    if not y_valid.size or not x_valid.size:
        return data

    # Compute bounding indices with optional additional cropping
    y_min, y_max = (
        max(y_valid.min() + crop_size, 0),
        min(y_valid.max() - crop_size, ref_data_array.shape[0] - 1),
    )
    x_min, x_max = (
        max(x_valid.min() + crop_size, 0),
        min(x_valid.max() - crop_size, ref_data_array.shape[1] - 1),
    )

    # In Python slicing is end-exclusive, so we need to add 1 to the max indices
    trimmed_data = data.isel(y=slice(y_min, y_max + 1), x=slice(x_min, x_max + 1))

    # Adjust the x,y offsets, taking into account the rotation and translation
    new_c = transform.c + x_min * transform.a + y_min * transform.b
    new_f = transform.f + x_min * transform.d + y_min * transform.e

    # Create the new transformation matrix
    new_transform = Affine(
        transform.a,
        transform.b,
        new_c,
        transform.d,
        transform.e,
        new_f,
    )

    # Apply the new transformation to the trimmed data
    trimmed_data = trimmed_data.rio.write_transform(new_transform)

    return trimmed_data


def extract_data_inside_polygon(rotated: xr.Dataset, polygon: Polygon) -> xr.Dataset:
    """
    Extracts the data inside the bounding box of a given polygon from a rotated xarray dataset.

    Args:
        rotated (xr.Dataset): The rotated dataset to extract data from.
        polygon (Polygon): The polygon defining the region of interest.

    Returns:
        xr.Dataset: The subset of the rotated dataset inside the polygon's bounding box.

    Example:
        >>> rotated_ds = <Your rotated xarray dataset>
        >>> polygon = Polygon([(x1, y1), (x2, y2), ...])
        >>> subset = extract_data_inside_polygon(rotated_ds, polygon)
        >>> print(subset)
    """

    x_coords, y_coords = zip(*polygon.exterior.coords, strict=True)
    x_indices, y_indices = [], []

    for x_val, y_val in zip(x_coords, y_coords, strict=True):
        # Compute distance to every grid point for x and y coordinates separately
        distance_x = abs(rotated.xc - x_val)
        distance_y = abs(rotated.yc - y_val)

        # Use np.unravel_index to get the indices of the minimum distance
        y_idx, x_idx = np.unravel_index(
            (distance_x + distance_y).argmin().values, distance_x.shape
        )

        x_indices.append(x_idx)
        y_indices.append(y_idx)

    # Determine the slice bounds
    x_slice = slice(min(x_indices), max(x_indices) + 1)
    y_slice = slice(min(y_indices), max(y_indices) + 1)

    # Slice the dataset
    subset = rotated.isel(x=x_slice, y=y_slice)

    return subset


def apply_mask(da: xr.DataArray, values: list[int]) -> xr.DataArray:
    """
    Mask specific pixel values in the input DataArray.

    Args:
        da (xr.DataArray): The input DataArray to be masked.
        values (list[int]): List of numeric values to mask.

    Returns:
        xr.DataArray: The input DataArray with pixels matching the values set to NaN.
    """
    return da.where(~da.isin(values))


def combine_by_first(
    datasets: list[xr.Dataset | xr.DataArray],
) -> xr.Dataset | xr.DataArray:
    """
    Combine multiple xarray datasets or data arrays, ignoring NaN values.

    Args:
        datasets (list[xr.Dataset | xr.DataArray]): List of datasets/data arrays to combine.

    Returns:
        xr.Dataset | xr.DataArray: Combined dataset or data array.
    """
    if not datasets:
        msg = "The input list of datasets is empty."
        raise ValueError(msg)

    # Ensure all inputs are of the same type
    first = datasets[0]
    if not all(isinstance(ds, type(first)) for ds in datasets):
        msg = "All datasets must be of the same type (either Dataset or DataArray)."
        raise TypeError(msg)

    # Combine using combine_first in a loop
    combined = datasets[0]
    for ds in datasets[1:]:
        combined = combined.combine_first(ds)  # type: ignore

    return combined


def to_array_with_attrs(
    ds: xr.Dataset, dim: str = "band", name: str | None = None
) -> xr.DataArray:
    """
    Convert an xarray Dataset to a DataArray while preserving both global and variable attributes,
    including `rioxarray` metadata like `nodata`.

    Args:
        ds (xr.Dataset): Input Dataset with attributes to preserve.
        dim (str): Name of the dimension for stacking (default: "band").
        name (str, optional): Name for the resulting DataArray. Defaults to None.

    Returns:
        xr.DataArray: DataArray with preserved global and variable attributes.
    """
    # Preserve global attributes
    attrs = ds.attrs.copy()
    nodata = get_nodata(ds)

    # Convert Dataset to DataArray
    da = ds.to_array(dim=dim, name=name)

    # Restore global attributes
    da.attrs.update(attrs)

    da = set_nodata(da, nodata)

    return da  # type: ignore
