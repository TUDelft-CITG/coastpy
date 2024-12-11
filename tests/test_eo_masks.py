import numpy as np
import pytest
import xarray as xr

from coastpy.eo.mask import (
    apply_mask,
    nodata_mask,
    numeric_mask,
    scl_mask,
)


@pytest.fixture()
def example_data():
    """Fixture to create an example dataset."""
    return xr.Dataset(
        {
            "var1": xr.DataArray(
                np.array([[1, 2, 3], [4, 5, -999]]),
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
                np.array([[0, 4, 6], [9, 10, 11]]),
                dims=["x", "y"],
                name="SCL",
            ),
        }
    )


def test_nodata_mask(example_data):
    """Test nodata_mask."""
    expected_mask = xr.Dataset(
        {
            "var1": xr.DataArray(
                [[False, False, False], [False, False, True]],
                dims=["x", "y"],
            ),
            "var2": xr.DataArray(
                [[False, False, False], [False, False, False]],
                dims=["x", "y"],
            ),
            "SCL": xr.DataArray(
                [[False, False, False], [False, False, False]],
                dims=["x", "y"],
            ),
        }
    )
    mask = nodata_mask(example_data)
    xr.testing.assert_equal(mask, expected_mask)


def test_numeric_mask(example_data):
    """Test numeric_mask."""
    values_to_mask = [5, 30, 60]
    expected_mask = xr.Dataset(
        {
            "var1": xr.DataArray(
                [[False, False, False], [False, True, False]],
                dims=["x", "y"],
            ),
            "var2": xr.DataArray(
                [[False, False, True], [False, False, True]],
                dims=["x", "y"],
            ),
            "SCL": xr.DataArray(
                [[False, False, False], [False, False, False]],
                dims=["x", "y"],
            ),
        }
    )
    mask = numeric_mask(example_data, values_to_mask)
    xr.testing.assert_equal(mask, expected_mask)


def test_scl_mask(example_data):
    """Test scl_mask."""
    scl_classes_to_mask = ["Vegetation", 11]
    expected_mask = xr.DataArray(
        [[False, True, False], [False, False, True]],
        dims=["x", "y"],
    )
    mask = scl_mask(example_data["SCL"], scl_classes_to_mask)
    xr.testing.assert_equal(mask, expected_mask)


def test_apply_mask(example_data):
    """Test apply_mask with nodata_mask."""
    expected_masked = xr.Dataset(
        {
            "var1": xr.DataArray(
                np.array([[1, 2, 3], [4, 5, np.nan]]),
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
                np.array([[0, 4, 6], [9, 10, 11]]),
                dims=["x", "y"],
                name="SCL",
            ),
        }
    )
    nodata = nodata_mask(example_data)
    masked = apply_mask(example_data, nodata)
    xr.testing.assert_equal(masked, expected_masked)
