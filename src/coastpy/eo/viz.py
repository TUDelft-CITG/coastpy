import holoviews as hv
import numpy as np
import xarray as xr
import xrspatial.multispectral as ms


def view_rgb(
    ds: xr.Dataset,
    crs: str = "EPSG:4326",
    add_title: bool = True,
    **kwargs,
) -> hv.DynamicMap:
    """
    Generate an RGB plot from an xarray Dataset with 'red', 'green', and 'blue' bands.

    Args:
        ds (xr.Dataset): Input dataset containing 'red', 'green', and 'blue' data variables.
        crs (str): CRS to reproject the dataset to (default: "EPSG:4326").
        add_title (bool): Whether to add a title displaying the timestamp (for single time steps).
        **kwargs: Additional keyword arguments passed to `hvplot.rgb`.

    Returns:
        holoviews.DynamicMap: An interactive RGB plot, with a slider if a time dimension exists.
    """
    # Check for a time dimension
    if "time" in ds.dims:
        if ds.time.size > 1:
            # Multiple time steps: Add a slider
            da = (
                ds[["red", "green", "blue"]]
                .to_array("band")
                .rio.reproject(crs, nodata=np.nan)
            )
            tc = ms.true_color(*da).rio.write_crs(crs)
            title = "RGB View with time slider"
            return tc.hvplot.rgb(
                x="x",
                y="y",
                bands="band",
                geo=True,
                tiles="EsriImagery",
                **kwargs,
                title=title,
            )
        else:
            # Single time step: Flatten and add timestamp
            ds = ds.isel(time=0)
            timestamp = str(ds.time.values)  # Access scalar value directly

    # No time or single time step
    da = ds[["red", "green", "blue"]].to_array("band").rio.reproject(crs, nodata=np.nan)
    tc = ms.true_color(*da).rio.write_crs(crs)
    plot = tc.hvplot.rgb(
        x="x", y="y", bands="band", geo=True, tiles="EsriImagery", **kwargs
    )
    if add_title and "time" in ds.coords:
        timestamp = str(ds.time.values)  # Access scalar value directly
        plot = plot.opts(title=f"RGB View - {timestamp}")
    return plot
