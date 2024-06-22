# NOTE: this is copied from the dea-notebooks repository due because that package cannot
# be installed on MSPC due to dependencies. Try to keep this code in sync with the original
# and remove this note when the package can be installed. Also add a reference to dea-notebooks if
# you use this code.
## dea_spatialtools.py
import inspect
import re
import warnings
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString, MultiLineString
from skimage.measure import find_contours


def calculate_indices(
    ds: xr.Dataset,
    index: str | list[str] | None = None,
    bands_to_rename: dict[str, str] | None = None,
    collection: str | None = None,
    custom_varname: str | None = None,
    normalise: bool = True,
    drop: bool = False,
    inplace: bool = False,
) -> xr.Dataset:
    """
    Takes an xarray dataset containing spectral bands, calculates one of
    a set of remote sensing indices, and adds the resulting array as a
    new variable in the original dataset.

    Note: by default, this function will create a new copy of the data
    in memory. This can be a memory-expensive operation, so to avoid
    this, set `inplace=True`.

    Last modified: July 2023

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array with containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculate the selected water index.
    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'AWEI_ns'`` (Automated Water Extraction Index,
                  no shadows, Feyisa 2014)
        * ``'AWEI_sh'`` (Automated Water Extraction Index,
                   shadows, Feyisa 2014)
        * ``'BAEI'`` (Built-Up Area Extraction Index, Bouzekri et al. 2015)
        * ``'BAI'`` (Burn Area Index, Martin 1998)
        * ``'BSI'`` (Bare Soil Index, Rikimaru et al. 2002)
        * ``'BUI'`` (Built-Up Index, He et al. 2010)
        * ``'CMR'`` (Clay Minerals Ratio, Drury 1987)
        * ``'EVI'`` (Enhanced Vegetation Index, Huete 2002)
        * ``'FMR'`` (Ferrous Minerals Ratio, Segal 1982)
        * ``'IOR'`` (Iron Oxide Ratio, Segal 1982)
        * ``'LAI'`` (Leaf Area Index, Boegh 2002)
        * ``'MNDWI'`` (Modified Normalised Difference Water Index, Xu 1996)
        * ``'MSAVI'`` (Modified Soil Adjusted Vegetation Index,
                 Qi et al. 1994)
        * ``'NBI'`` (New Built-Up Index, Jieli et al. 2010)
        * ``'NBR'`` (Normalised Burn Ratio, Lopez Garcia 1991)
        * ``'NDBI'`` (Normalised Difference Built-Up Index, Zha 2003)
        * ``'NDCI'`` (Normalised Difference Chlorophyll Index,
                Mishra & Mishra, 2012)
        * ``'NDMI'`` (Normalised Difference Moisture Index, Gao 1996)
        * ``'NDSI'`` (Normalised Difference Snow Index, Hall 1995)
        * ``'NDTI'`` (Normalise Difference Tillage Index,
                Van Deventeret et al. 1997)
        * ``'NDVI'`` (Normalised Difference Vegetation Index, Rouse 1973)
        * ``'NDWI'`` (Normalised Difference Water Index, McFeeters 1996)
        * ``'SAVI'`` (Soil Adjusted Vegetation Index, Huete 1988)
        * ``'TCB'`` (Tasseled Cap Brightness, Crist 1985)
        * ``'TCG'`` (Tasseled Cap Greeness, Crist 1985)
        * ``'TCW'`` (Tasseled Cap Wetness, Crist 1985)
        * ``'TCB_GSO'`` (Tasseled Cap Brightness, Nedkov 2017)
        * ``'TCG_GSO'`` (Tasseled Cap Greeness, Nedkov 2017)
        * ``'TCW_GSO'`` (Tasseled Cap Wetness, Nedkov 2017)
        * ``'WI'`` (Water Index, Fisher 2016)
        * ``'kNDVI'`` (Non-linear Normalised Difference Vegation Index,
                 Camps-Valls et al. 2021)
    bandnames_mapper : Optional[Dict[str, str]], default is None
        A dictionary that maps the original band names in the dataset to the standard band names
        that the indices formulas use. For example, if the red band in the dataset is named 'B4',
        the dictionary should include the entry 'B4': 'red'.
    collection : str
        An string that tells the function what data collection is
        being used to calculate the index. This is necessary because
        different collections use different names for bands covering
        a similar spectra.

        Valid options are:

        * ``'ga_ls_2'`` (for GA Landsat Collection 2)
        * ``'ga_ls_3'`` (for GA Landsat Collection 3)
        * ``'ga_s2_1'`` (for GA Sentinel 2 Collection 1)
        * ``'ga_s2_3'`` (for GA Sentinel 2 Collection 3)
        * ``'pc_s2_2'`` (for PC Sentinel 2 Collection 2)
    custom_varname : str, optional
        By default, the original dataset will be returned with
        a new index variable named after `index` (e.g. 'NDVI'). To
        specify a custom name instead, you can supply e.g.
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable.
    normalise : bool, optional
        Some coefficient-based indices (e.g. ``'WI'``, ``'BAEI'``,
        ``'AWEI_ns'``, ``'AWEI_sh'``, ``'TCW'``, ``'TCG'``, ``'TCB'``,
        ``'TCW_GSO'``, ``'TCG_GSO'``, ``'TCB_GSO'``, ``'EVI'``,
        ``'LAI'``, ``'SAVI'``, ``'MSAVI'``) produce different results if
        surface reflectance values are not scaled between 0.0 and 1.0
        prior to calculating the index. Setting `normalise=True` first
        scales values to a 0.0-1.0 range by dividing by 10000.0.
        Defaults to True.
    drop : bool, optional
        Provides the option to drop the original input data, thus saving
        space. if drop = True, returns only the index and its values.
    inplace: bool, optional
        If `inplace=True`, calculate_indices will modify the original
        array in-place, adding bands to the input dataset. The default
        is `inplace=False`, which will instead make a new copy of the
        original data (and use twice the memory).

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the
        original Dataset.
    """

    if bands_to_rename is None:
        bands_to_rename = {}

    def get_fargs(func: Callable) -> set[str]:
        """Returns a set of variables used in a provided function by inspecting its source code."""
        source_code = inspect.getsource(func)
        vars_used = re.findall(r"ds\.([a-zA-Z0-9]+)", source_code)
        return set(vars_used)

    # Set ds equal to a copy of itself in order to prevent the function
    # from editing the input dataset. This can prevent unexpected
    # behaviour though it uses twice as much memory.
    if not inplace:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        print(f"Dropping bands {bands_to_drop}")

    index_dict = {
        # Dictionary containing remote sensing index band recipes
        # Normalised Difference Vegation Index, Rouse 1973
        "NDVI": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
        # Non-linear Normalised Difference Vegation Index,
        # Camps-Valls et al. 2021
        "kNDVI": lambda ds: np.tanh(((ds.nir - ds.red) / (ds.nir + ds.red)) ** 2),
        # Enhanced Vegetation Index, Huete 2002
        "EVI": lambda ds: (
            (2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1)
        ),
        # Leaf Area Index, Boegh 2002
        "LAI": lambda ds: (
            3.618
            * ((2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
            - 0.118
        ),
        # Soil Adjusted Vegetation Index, Huete 1988
        "SAVI": lambda ds: ((1.5 * (ds.nir - ds.red)) / (ds.nir + ds.red + 0.5)),
        # Mod. Soil Adjusted Vegetation Index, Qi et al. 1994
        "MSAVI": lambda ds: (
            (2 * ds.nir + 1 - ((2 * ds.nir + 1) ** 2 - 8 * (ds.nir - ds.red)) ** 0.5)
            / 2
        ),
        # Normalised Difference Moisture Index, Gao 1996
        "NDMI": lambda ds: (ds.nir - ds.swir1) / (ds.nir + ds.swir1),
        # Normalised Burn Ratio, Lopez Garcia 1991
        "NBR": lambda ds: (ds.nir - ds.swir2) / (ds.nir + ds.swir2),
        # Burn Area Index, Martin 1998
        "BAI": lambda ds: (1.0 / ((0.10 - ds.red) ** 2 + (0.06 - ds.nir) ** 2)),
        # Normalised Difference Chlorophyll Index,
        # (Mishra & Mishra, 2012)
        "NDCI": lambda ds: (ds.red_edge_1 - ds.red) / (ds.red_edge_1 + ds.red),
        # Normalised Difference Snow Index, Hall 1995
        "NDSI": lambda ds: (ds.green - ds.swir1) / (ds.green + ds.swir1),
        # Normalised Difference Tillage Index,
        # Van Deventer et al. 1997
        "NDTI": lambda ds: (ds.swir1 - ds.swir2) / (ds.swir1 + ds.swir2),
        # Normalised Difference Water Index, McFeeters 1996
        "NDWI": lambda ds: (ds.green - ds.nir) / (ds.green + ds.nir),
        # Modified Normalised Difference Water Index, Xu 2006
        "MNDWI": lambda ds: (ds.green - ds.swir1) / (ds.green + ds.swir1),
        # Normalised Difference Built-Up Index, Zha 2003
        "NDBI": lambda ds: (ds.swir1 - ds.nir) / (ds.swir1 + ds.nir),
        # Built-Up Index, He et al. 2010
        "BUI": lambda ds: ((ds.swir1 - ds.nir) / (ds.swir1 + ds.nir))
        - ((ds.nir - ds.red) / (ds.nir + ds.red)),
        # Built-up Area Extraction Index, Bouzekri et al. 2015
        "BAEI": lambda ds: (ds.red + 0.3) / (ds.green + ds.swir1),
        # New Built-up Index, Jieli et al. 2010
        "NBI": lambda ds: (ds.swir1 + ds.red) / ds.nir,
        # Bare Soil Index, Rikimaru et al. 2002
        "BSI": lambda ds: ((ds.swir1 + ds.red) - (ds.nir + ds.blue))
        / ((ds.swir1 + ds.red) + (ds.nir + ds.blue)),
        # Automated Water Extraction Index (no shadows), Feyisa 2014
        "AWEI_ns": lambda ds: (
            4 * (ds.green - ds.swir1) - (0.25 * ds.nir * +2.75 * ds.swir2)
        ),
        # Automated Water Extraction Index (shadows), Feyisa 2014
        "AWEI_sh": lambda ds: (
            ds.blue + 2.5 * ds.green - 1.5 * (ds.nir + ds.swir1) - 0.25 * ds.swir2
        ),
        # Water Index, Fisher 2016
        "WI": lambda ds: (
            1.7204
            + 171 * ds.green
            + 3 * ds.red
            - 70 * ds.nir
            - 45 * ds.swir1
            - 71 * ds.swir2
        ),
        # Tasseled Cap Wetness, Crist 1985
        "TCW": lambda ds: (
            0.0315 * ds.blue
            + 0.2021 * ds.green
            + 0.3102 * ds.red
            + 0.1594 * ds.nir
            + -0.6806 * ds.swir1
            + -0.6109 * ds.swir2
        ),
        # Tasseled Cap Greeness, Crist 1985
        "TCG": lambda ds: (
            -0.1603 * ds.blue
            + -0.2819 * ds.green
            + -0.4934 * ds.red
            + 0.7940 * ds.nir
            + -0.0002 * ds.swir1
            + -0.1446 * ds.swir2
        ),
        # Tasseled Cap Brightness, Crist 1985
        "TCB": lambda ds: (
            0.2043 * ds.blue
            + 0.4158 * ds.green
            + 0.5524 * ds.red
            + 0.5741 * ds.nir
            + 0.3124 * ds.swir1
            + -0.2303 * ds.swir2
        ),
        # Tasseled Cap Transformations with Sentinel-2 coefficients
        # after Nedkov 2017 using Gram-Schmidt orthogonalization (GSO)
        # Tasseled Cap Wetness, Nedkov 2017
        "TCW_GSO": lambda ds: (
            0.0649 * ds.blue
            + 0.2802 * ds.green
            + 0.3072 * ds.red
            + -0.0807 * ds.nir
            + -0.4064 * ds.swir1
            + -0.5602 * ds.swir2
        ),
        # Tasseled Cap Greeness, Nedkov 2017
        "TCG_GSO": lambda ds: (
            -0.0635 * ds.blue
            + -0.168 * ds.green
            + -0.348 * ds.red
            + 0.3895 * ds.nir
            + -0.4587 * ds.swir1
            + -0.4064 * ds.swir2
        ),
        # Tasseled Cap Brightness, Nedkov 2017
        "TCB_GSO": lambda ds: (
            0.0822 * ds.blue
            + 0.136 * ds.green
            + 0.2611 * ds.red
            + 0.5741 * ds.nir
            + 0.3882 * ds.swir1
            + 0.1366 * ds.swir2
        ),
        # Clay Minerals Ratio, Drury 1987
        "CMR": lambda ds: (ds.swir1 / ds.swir2),
        # Ferrous Minerals Ratio, Segal 1982
        "FMR": lambda ds: (ds.swir1 / ds.nir),
        # Iron Oxide Ratio, Segal 1982
        "IOR": lambda ds: (ds.red / ds.blue),
        # Add blue/red for CoastSat classifier
        "BR": lambda ds: (ds.blue - ds.red) / (ds.blue + ds.red),
    }

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # calculate for each index in the list of indices supplied (indexes)
    for index in indices:
        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an
        # invalid option being provided, raise an exception informing user to
        # choose from the list of valid options
        if index is None:
            msg = (
                "No remote sensing `index` was provided. Please refer to the function"
                " \ndocumentation for a full list of valid options for `index` (e.g."
                " 'NDVI')"
            )
            raise ValueError(msg)

        elif (
            index in ["WI", "BAEI", "AWEI_ns", "AWEI_sh", "EVI", "LAI", "SAVI", "MSAVI"]
            and not normalise
        ):
            warnings.warn(
                f"\nA coefficient-based index ('{index}') normally "
                "applied to surface reflectance values in the \n"
                "0.0-1.0 range was applied to values in the 0-10000 "
                "range. This can produce unexpected results; \nif "
                "required, resolve this by setting `normalise=True`"
            )

        elif index_func is None:
            msg = (
                f"The selected index '{index}' is not one of the valid remote sensing"
                " index options. \nPlease refer to the function documentation for a"
                " full list of valid options for `index`"
            )
            raise ValueError(msg)

        if collection == "ga_ls_3":
            # Dictionary mapping full data names to simpler 'red' alias names
            bandnames_dict = {
                "nbart_nir": "nir",
                "nbart_red": "red",
                "nbart_green": "green",
                "nbart_blue": "blue",
                "nbart_swir_1": "swir1",
                "nbart_swir_2": "swir2",
                "nbar_red": "red",
                "nbar_green": "green",
                "nbar_blue": "blue",
                "nbar_nir": "nir",
                "nbar_swir_1": "swir1",
                "nbar_swir_2": "swir2",
            }

            bands_to_rename.update(bandnames_dict)

        elif (collection == "ga_s2_1") | (collection == "ga_s2_3"):
            # Dictionary mapping full data names to simpler 'red' alias names
            bandnames_dict = {
                "nbart_red": "red",
                "nbart_green": "green",
                "nbart_blue": "blue",
                "nbart_nir_1": "nir",
                "nbart_red_edge_1": "red_edge_1",
                "nbart_red_edge_2": "red_edge_2",
                "nbart_swir_2": "swir1",
                "nbart_swir_3": "swir2",
                "nbar_red": "red",
                "nbar_green": "green",
                "nbar_blue": "blue",
                "nbar_nir_1": "nir",
                "nbar_red_edge_1": "red_edge_1",
                "nbar_red_edge_2": "red_edge_2",
                "nbar_swir_2": "swir1",
                "nbar_swir_3": "swir2",
            }

            bands_to_rename.update(bandnames_dict)

        elif collection == "pc_s2_2":
            bandnames_dict = {"swir16": "swir1", "swir22": "swir2"}
            bands_to_rename.update(bandnames_dict)

        bands_to_rename = {
            a: b for a, b in bands_to_rename.items() if a in ds.variables
        }

        if not get_fargs(index_func).issubset(
            set(ds.rename(bands_to_rename).variables)
        ):
            unique_used_vars = set.union(*[get_fargs(f) for f in index_dict.values()])

            missing_bands = set(get_fargs(index_func)) - set(
                ds.rename(bands_to_rename).variables
            )

            error_message = (
                "The dataset is missing some required bands for the index"
                f" '{index}'.\nRequired bands for index '{index}':"
                f" {sorted(get_fargs(index_func))}.\nPresent bands in the dataset:"
                f" {sorted(ds.drop(['x', 'y']).variables)}.\n\nMissing bands:"
                f" {sorted(missing_bands)}.\n\nPlease ensure that"
                " all required bands are present in the dataset `ds`.\nIf the bands in"
                " `ds` use different names, consider using a custom `bandname_mapper`"
                " to rename them to their common names.\n\nFor your reference, here"
                " are all the bands that are used by the available index functions:"
                f" {unique_used_vars}."
            )

            raise ValueError(error_message)

        try:
            # If normalised=True, divide data by 10,000 before applying func
            mult = 10000.0 if normalise else 1.0
            index_array = index_func(ds.rename(bands_to_rename) / mult)
        except AttributeError:
            msg = (
                f"Please verify that all bands required to compute {index} are present"
                " in `ds`. \nThese bands may vary depending on the `collection` (e.g."
                " the Landsat `nbart_nir` band \nis equivelent to `nbart_nir_1` for"
                " Sentinel 2)"
            )
            raise ValueError(msg)

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if inplace=False
    if drop and not inplace:
        ds = ds.drop(bands_to_drop)

    # If inplace == True, delete bands in-place instead of using drop
    if drop and inplace:
        for band_to_drop in bands_to_drop:
            del ds[band_to_drop]

    # Return input dataset with added water index variable
    return ds


def subpixel_contours(
    da,
    z_values=None,
    crs=None,
    affine=None,
    attribute_df=None,
    output_path=None,
    min_vertices=2,
    dim="time",
    errors="ignore",
    verbose=False,
):
    """
    Uses `skimage.measure.find_contours` to extract multiple z-value
    contour lines from a two-dimensional array (e.g. multiple elevations
    from a single DEM), or one z-value for each array along a specified
    dimension of a multi-dimensional array (e.g. to map waterlines
    across time by extracting a 0 NDWI contour from each individual
    timestep in an xarray timeseries).

    Contours are returned as a geopandas.GeoDataFrame with one row per
    z-value or one row per array along a specified dimension. The
    `attribute_df` parameter can be used to pass custom attributes
    to the output contour features.

    Last modified: November 2020

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array from which
        contours are extracted. If a two-dimensional array is provided,
        the analysis will run in 'single array, multiple z-values' mode
        which allows you to specify multiple `z_values` to be extracted.
        If a multi-dimensional array is provided, the analysis will run
        in 'single z-value, multiple arrays' mode allowing you to
        extract contours for each array along the dimension specified
        by the `dim` parameter.
    z_values : int, float or list of ints, floats
        An individual z-value or list of multiple z-values to extract
        from the array. If operating in 'single z-value, multiple
        arrays' mode specify only a single z-value.
    crs : string or CRS object, optional
        An EPSG string giving the coordinate system of the array
        (e.g. 'EPSG:3577'). If none is provided, the function will
        attempt to extract a CRS from the xarray object's `crs`
        attribute.
    affine : affine.Affine object, optional
        An affine.Affine object (e.g. `from affine import Affine;
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0) giving the
        affine transformation used to convert raster coordinates
        (e.g. [0, 0]) to geographic coordinates. If none is provided,
        the function will attempt to obtain an affine transformation
        from the xarray object (e.g. either at `da.transform` or
        `da.geobox.transform`).
    output_path : string, optional
        The path and filename for the output shapefile.
    attribute_df : pandas.Dataframe, optional
        A pandas.Dataframe containing attributes to pass to the output
        contour features. The dataframe must contain either the same
        number of rows as supplied `z_values` (in 'multiple z-value,
        single array' mode), or the same number of rows as the number
        of arrays along the `dim` dimension ('single z-value, multiple
        arrays mode').
    min_vertices : int, optional
        The minimum number of vertices required for a contour to be
        extracted. The default (and minimum) value is 2, which is the
        smallest number required to produce a contour line (i.e. a start
        and end point). Higher values remove smaller contours,
        potentially removing noise from the output dataset.
    dim : string, optional
        The name of the dimension along which to extract contours when
        operating in 'single z-value, multiple arrays' mode. The default
        is 'time', which extracts contours for each array along the time
        dimension.
    errors : string, optional
        If 'raise', then any failed contours will raise an exception.
        If 'ignore' (the default), a list of failed contours will be
        printed. If no contours are returned, an exception will always
        be raised.
    verbose : bool, optional
        Print debugging messages. Default False.

    Returns
    -------
    output_gdf : geopandas geodataframe
        A geopandas geodataframe object with one feature per z-value
        ('single array, multiple z-values' mode), or one row per array
        along the dimension specified by the `dim` parameter ('single
        z-value, multiple arrays' mode). If `attribute_df` was
        provided, these values will be included in the shapefile's
        attribute table.
    """

    if z_values is None:
        z_values = [0.0]

    def contours_to_multiline(da_i, z_value, min_vertices=2):
        """
        Helper function to apply marching squares contour extraction
        to an array and return a data as a shapely MultiLineString.
        The `min_vertices` parameter allows you to drop small contours
        with less than X vertices.
        """

        # Extracts contours from array, and converts each discrete
        # contour into a Shapely LineString feature. If the function
        # returns a KeyError, this may be due to an unresolved issue in
        # scikit-image: https://github.com/scikit-image/scikit-image/issues/4830
        # line_features = [
        #     LineString(i[:, [1, 0]])
        #     for i in find_contours(da_i.to_numpy(), z_value)
        #     if i.shape[0] > min_vertices
        # ]

        line_features = [
            LineString(i[:, [1, 0]])
            for i in find_contours(da_i, z_value)
            if i.shape[0] > min_vertices
        ]

        # Output resulting lines into a single combined MultiLineString
        return MultiLineString(line_features)

    # Check if CRS is provided as a xarray.DataArray attribute.
    # If not, require supplied CRS
    try:
        crs = da.crs
    except:
        if crs is None:
            msg = (
                "Please add a `crs` attribute to the xarray.DataArray, or provide a CRS"
                " using the function's `crs` parameter (e.g. 'EPSG:3577')"
            )
            raise ValueError(msg)

    # Check if Affine transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    try:
        affine = da.transform
    except:
        if affine is None:
            msg = (
                "Please provide an Affine object using the `affine` parameter (e.g."
                " `from affine import Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0,"
                " 6886890.0)`"
            )
            raise TypeError(msg)

    # If z_values is supplied is not a list, convert to list:
    # z_values = z_values if (isinstance(z_values, list | np.ndarray)) else [z_values]

    def ensure_list(z_values):
        # If z_values is a single number (not in a list or array), convert it to a list
        if isinstance(z_values, int | float):
            return [z_values]

        # If z_values is a NumPy array
        elif isinstance(z_values, np.ndarray):
            # Convert single-value arrays to a list
            if z_values.size == 1:
                return [z_values.tolist()]
            # Return the array as is if it has more than one element
            return z_values

        # If z_values is already a list, return as is
        elif isinstance(z_values, list):
            return z_values

        # Handle other data types, if necessary
        else:
            # You can raise an error or handle other types differently
            raise ValueError("z_values must be a number, list, or numpy array")

    z_values = ensure_list(z_values)

    dim = "z_value"
    # contour_arrays = {
    #     str(i.to_numpy())[0:10]: contours_to_multiline(da, i, min_vertices)
    #     for i in z_values
    # }
    contour_arrays = {
        str(i)[0:10]: contours_to_multiline(da, i, min_vertices) for i in z_values
    }

    attribute_df = list(contour_arrays.keys())

    # Convert output contours to a geopandas.GeoDataFrame
    contours_gdf = gpd.GeoDataFrame(
        data=attribute_df,
        geometry=list(contour_arrays.values()),
        crs=crs,
    )

    # Define affine and use to convert array coords to geographic coords.
    # We need to add 0.5 x pixel size to the x and y to obtain the centre
    # point of our pixels, rather than the top-left corner
    shapely_affine = [
        affine.a,
        affine.b,
        affine.d,
        affine.e,
        affine.xoff + affine.a / 2.0,
        affine.yoff + affine.e / 2.0,
    ]
    contours_gdf["geometry"] = contours_gdf.affine_transform(shapely_affine)

    # Rename the data column to match the dimension
    contours_gdf = contours_gdf.rename({0: dim}, axis=1)

    # Drop empty timesteps
    empty_contours = contours_gdf.geometry.is_empty
    ", ".join(map(str, contours_gdf[empty_contours][dim].to_list()))
    contours_gdf = contours_gdf[~empty_contours]
    contours_gdf = contours_gdf.explode(index_parts=False).reset_index(drop=True)
    return contours_gdf


if __name__ == "__main__":
    import numpy as np
    import xarray as xr

    # Define the size of the arrays
    n_x, n_y = 10, 10

    # Create the coordinate arrays
    x = np.arange(n_x)
    y = np.arange(n_y)

    # Create some dummy data for each band
    nir = np.random.rand(n_y, n_x)
    red = np.random.rand(n_y, n_x)
    swir1 = np.random.rand(n_y, n_x)
    blue = np.random.rand(n_y, n_x)
    green = np.random.rand(n_y, n_x)

    # Create the xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "nir": (("y", "x"), nir),
            "red": (("y", "x"), red),
            "swir1": (("y", "x"), swir1),
            "blue": (("y", "x"), blue),
            "green": (("y", "x"), green),
        },
        coords={"x": x, "y": y},
    )

    ds2 = calculate_indices(ds, ["NBR"])
    ds2 = calculate_indices(ds, ["NBR"])

    print(ds)
