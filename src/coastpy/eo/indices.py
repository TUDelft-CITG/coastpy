import inspect
import logging
import re
from collections.abc import Callable

import numpy as np
import xarray as xr

from coastpy.utils.xarray_utils import set_nodata

# Define logging
logger = logging.getLogger(__name__)

INDEX_DICT = {
    "NDVI": {
        "formula": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
        "description": "Normalized Difference Vegetation Index, Rouse 1973",
    },
    "kNDVI": {
        "formula": lambda ds: np.tanh(((ds.nir - ds.red) / (ds.nir + ds.red)) ** 2),
        "description": "Non-linear Normalized Difference Vegetation Index, Camps-Valls et al. 2021",
    },
    "EVI": {
        "formula": lambda ds: (
            (2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1)
        ),
        "description": "Enhanced Vegetation Index, Huete 2002",
    },
    "LAI": {
        "formula": lambda ds: (
            3.618
            * ((2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
            - 0.118
        ),
        "description": "Leaf Area Index, Boegh 2002",
    },
    "SAVI": {
        "formula": lambda ds: ((1.5 * (ds.nir - ds.red)) / (ds.nir + ds.red + 0.5)),
        "description": "Soil Adjusted Vegetation Index, Huete 1988",
    },
    "MSAVI": {
        "formula": lambda ds: (
            (2 * ds.nir + 1 - ((2 * ds.nir + 1) ** 2 - 8 * (ds.nir - ds.red)) ** 0.5)
            / 2
        ),
        "description": "Modified Soil Adjusted Vegetation Index, Qi et al. 1994",
    },
    "NDMI": {
        "formula": lambda ds: (ds.nir - ds.swir16) / (ds.nir + ds.swir16),
        "description": "Normalized Difference Moisture Index, Gao 1996",
    },
    "NBR": {
        "formula": lambda ds: (ds.nir - ds.swir22) / (ds.nir + ds.swir22),
        "description": "Normalized Burn Ratio, Lopez Garcia 1991",
    },
    "BAI": {
        "formula": lambda ds: (1.0 / ((0.10 - ds.red) ** 2 + (0.06 - ds.nir) ** 2)),
        "description": "Burn Area Index, Martin 1998",
    },
    "NDCI": {
        "formula": lambda ds: (ds.rededge1 - ds.red) / (ds.rededge1 + ds.red),
        "description": "Normalized Difference Chlorophyll Index, Mishra & Mishra, 2012",
    },
    "NDSI": {
        "formula": lambda ds: (ds.green - ds.swir16) / (ds.green + ds.swir16),
        "description": "Normalized Difference Snow Index, Hall 1995",
    },
    "NDTI": {
        "formula": lambda ds: (ds.swir16 - ds.swir22) / (ds.swir16 + ds.swir22),
        "description": "Normalized Difference Tillage Index, Van Deventer et al. 1997",
    },
    "NDWI": {
        "formula": lambda ds: (ds.green - ds.nir) / (ds.green + ds.nir),
        "description": "Normalized Difference Water Index, McFeeters 1996",
    },
    "MNDWI": {
        "formula": lambda ds: (ds.green - ds.swir16) / (ds.green + ds.swir16),
        "description": "Modified Normalized Difference Water Index, Xu 2006",
    },
    "NDBI": {
        "formula": lambda ds: (ds.swir16 - ds.nir) / (ds.swir16 + ds.nir),
        "description": "Normalized Difference Built-Up Index, Zha 2003",
    },
    "BUI": {
        "formula": lambda ds: ((ds.swir16 - ds.nir) / (ds.swir16 + ds.nir))
        - ((ds.nir - ds.red) / (ds.nir + ds.red)),
        "description": "Built-Up Index, He et al. 2010",
    },
    "BAEI": {
        "formula": lambda ds: (ds.red + 0.3) / (ds.green + ds.swir16),
        "description": "Built-Up Area Extraction Index, Bouzekri et al. 2015",
    },
    "NBI": {
        "formula": lambda ds: (ds.swir16 + ds.red) / ds.nir,
        "description": "New Built-Up Index, Jieli et al. 2010",
    },
    "BSI": {
        "formula": lambda ds: ((ds.swir16 + ds.red) - (ds.nir + ds.blue))
        / ((ds.swir16 + ds.red) + (ds.nir + ds.blue)),
        "description": "Bare Soil Index, Rikimaru et al. 2002",
    },
    "AWEI_ns": {
        "formula": lambda ds: (
            4 * (ds.green - ds.swir16) - (0.25 * ds.nir + 2.75 * ds.swir22)
        ),
        "description": "Automated Water Extraction Index (no shadows), Feyisa 2014",
    },
    "AWEI_sh": {
        "formula": lambda ds: (
            ds.blue + 2.5 * ds.green - 1.5 * (ds.nir + ds.swir16) - 0.25 * ds.swir22
        ),
        "description": "Automated Water Extraction Index (shadows), Feyisa 2014",
    },
    "WI": {
        "formula": lambda ds: (
            1.7204
            + 171 * ds.green
            + 3 * ds.red
            - 70 * ds.nir
            - 45 * ds.swir16
            - 71 * ds.swir22
        ),
        "description": "Water Index, Fisher 2016",
    },
    "TCW": {
        "formula": lambda ds: (
            0.0315 * ds.blue
            + 0.2021 * ds.green
            + 0.3102 * ds.red
            + 0.1594 * ds.nir
            + -0.6806 * ds.swir16
            + -0.6109 * ds.swir22
        ),
        "description": "Tasseled Cap Wetness, Crist 1985",
    },
    "TCG": {
        "formula": lambda ds: (
            -0.1603 * ds.blue
            + -0.2819 * ds.green
            + -0.4934 * ds.red
            + 0.7940 * ds.nir
            + -0.0002 * ds.swir16
            + -0.1446 * ds.swir22
        ),
        "description": "Tasseled Cap Greeness, Crist 1985",
    },
    "TCB": {
        "formula": lambda ds: (
            0.2043 * ds.blue
            + 0.4158 * ds.green
            + 0.5524 * ds.red
            + 0.5741 * ds.nir
            + 0.3124 * ds.swir16
            + -0.2303 * ds.swir22
        ),
        "description": "Tasseled Cap Brightness, Crist 1985",
    },
    "CMR": {
        "formula": lambda ds: (ds.swir16 / ds.swir22),
        "description": "Clay Minerals Ratio, Drury 1987",
    },
    "FMR": {
        "formula": lambda ds: (ds.swir16 / ds.nir),
        "description": "Ferrous Minerals Ratio, Segal 1982",
    },
    "IOR": {
        "formula": lambda ds: (ds.red / ds.blue),
        "description": "Iron Oxide Ratio, Segal 1982",
    },
    "BR": {
        "formula": lambda ds: (ds.blue - ds.red) / (ds.blue + ds.red),
        "description": "Blue-Red Index, CoastSat Classifier",
    },
}


def _get_fargs(func: Callable) -> set[str]:
    """Returns a set of variables used in a provided function by inspecting its source code."""
    source_code = inspect.getsource(func)
    vars_used = re.findall(r"ds\.([a-zA-Z0-9]+)", source_code)
    return set(vars_used)


def calculate_indices(
    ds: xr.Dataset,
    index: str | list[str],
    normalize: bool = True,
    drop: bool = False,
    nodata: float | None = None,
) -> xr.Dataset:
    """
    Calculate spectral indices for an xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the spectral bands required for index calculation.
    index : str or list of str
        The name(s) of the index or indices to calculate.
    normalize : bool, optional
        If True, normalize data by dividing by 10000. Defaults to True.
    drop : bool, optional
        If True, drop the original bands from the dataset. Defaults to False.
    nodata: float or None, optional
        If provided, replace nodata values with this value. Defaults to np.nan.

    Returns
    -------
    xr.Dataset
        A dataset with the calculated indices added as new variables.
    """
    # Ensure index is a list for consistent processing
    indices = [index] if isinstance(index, str) else index

    # Validate indices
    invalid_indices = [idx for idx in indices if idx not in INDEX_DICT]
    if invalid_indices:
        msg = (
            f"Invalid index/indices: {invalid_indices}. "
            f"Valid options are: {list(INDEX_DICT.keys())}."
        )
        raise ValueError(msg)

    # Normalize dataset if requested
    if normalize:
        ds = ds / 10000.0

    # Compute indices
    for idx in indices:
        index_info = INDEX_DICT[idx]
        formula = index_info["formula"]
        required_bands = _get_fargs(formula)

        # Check required bands are present in the dataset
        missing_bands = required_bands - set(ds.data_vars)
        if missing_bands:
            msg = f"Dataset is missing required bands for '{idx}': {missing_bands}."
            raise ValueError(msg)

        # Calculate index and add to dataset
        ds[idx] = formula(ds)

        if nodata is not None:
            ds[idx] = set_nodata(ds[idx], nodata)

    # Drop original bands if requested
    if drop:
        ds = ds.drop_vars([var for var in ds.data_vars if var not in indices])

    return ds
