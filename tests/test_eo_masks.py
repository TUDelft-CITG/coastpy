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


# TODO: ALMOST WORKS
# def test_geometry_mask():
#     """Test geometry_mask to extract a 3x3 subarray."""
#     # Create GeoBox and DataArray
#     geobox = GeoBox((10, 10), affine=Affine(0.2, 0.0, 4.0, 0.0, -0.2, 54.0), crs="EPSG:4326")
#     data = xr_zeros(geobox, dtype="float32")

#     # Populate known values
#     data.values[1:4, 1:4] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#     # Define the polygon geometry
#     polygon = geom.polygon(
#         [(4.3, 53.7), (4.9, 53.7), (4.9, 53.1), (4.3, 53.1), (4.3, 53.7)],
#         crs="EPSG:4326",
#     )

#     # Apply the geometry_mask with all_touched=True
#     masked_data = geometry_mask(data, polygon, invert=False, all_touched=True)

#     # Define the expected result
#     expected_values = np.full(data.shape, np.nan, dtype=data.dtype)  # Default to NaN
#     expected_values[1:4, 1:4] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Expected region

#     expected = data.copy()
#     expected.values = expected_values

#     # Assert the masked data matches the expected result
#     xr.testing.assert_equal(masked_data, expected)
