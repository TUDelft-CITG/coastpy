import geopandas as gpd
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_ols_fit(x, y) -> pd.Series:
    """
    Performs ordinary least squares (OLS) regression using scikit-learn.

    Parameters:
        x (np.ndarray): Array of years.
        y (np.ndarray): Array of shoreline positions.

    Returns:
        pd.Series: A pandas Series containing the intercept, slope, and R-squared.
    """
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x.reshape(-1, 1), y)

    return pd.Series({"intercept": intercept, "slope": slope, "r_squared": r_squared})


def compute_ols_trend(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """
    Compute Ordinary Least Squares (OLS) regression trend for each transect_id group.
    Only includes the transect_id, geometry (lon/lat as Point), and sds:change_rate.

    Args:
        df (pd.DataFrame): Input DataFrame with 'transect_id', 'obs_is_primary', 'datetime', and 'chainage'.
        x (str): Name of the independent variable column (e.g., 'datetime').
        y (str): Name of the dependent variable column (e.g., 'chainage').

    Returns:
        pd.DataFrame: DataFrame with 'transect_id', 'sds:change_rate', and 'geometry' (origin Point).
    """
    # Filter for primary observations
    df = df[df["obs_is_primary"]]

    if df.empty:
        return pd.DataFrame()

    # Compute OLS regression for each transect_id
    def ols_fit(group: pd.DataFrame) -> pd.Series:
        x_vals = group[x].dt.year.values
        y_vals = group[y].values
        result = compute_ols_fit(x_vals, y_vals)
        return pd.Series({"sds:change_rate": result["slope"]})

    trends = df.groupby("transect_id").apply(ols_fit).reset_index()

    # Extract unique lon/lat per transect_id and create Point geometry
    origins = df[["transect_id", "transect_lon", "transect_lat"]].drop_duplicates(
        "transect_id"
    )

    origins = origins.assign(
        geometry=gpd.GeoSeries.from_xy(
            origins["transect_lon"], origins["transect_lat"], crs="EPSG:4326"
        )
    )

    # Merge trends with origins (geometry)
    result = trends.merge(
        origins[["geometry", "transect_id"]], on="transect_id", how="left"
    )
    result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")
    result = result.set_index("transect_id")

    return result[["sds:change_rate", "geometry"]]
