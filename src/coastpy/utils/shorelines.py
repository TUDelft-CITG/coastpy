import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_ols_fit(x: np.ndarray, y: np.ndarray) -> pd.Series:
    """
    Perform Ordinary Least Squares (OLS) linear regression using scikit-learn and compute
    key statistical descriptors: intercept, slope, standard error of the slope, and R².

    Parameters:
        x (np.ndarray): 1D array of years (independent variable).
        y (np.ndarray): 1D array of shoreline positions (dependent variable).

    Returns:
        pd.Series: {
            "change_intercept": float,
            "change_rate": float,
            "change_rate_std_err": float,
            "r_squared": float
        }
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Must have at least 3 points to compute standard error (dof > 0)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        return pd.Series(
            {
                "change_intercept": np.nan,
                "change_rate": np.nan,
                "change_rate_std_err": np.nan,
                "r_squared": np.nan,
            }
        )

    # Reshape x for sklearn
    x_reshaped = x.reshape(-1, 1)
    model = LinearRegression().fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)

    residuals = y - y_pred
    dof = len(y) - 2  # 2 parameters: slope and intercept

    # Variance of residuals and denominator for std_err
    x_centered = x - np.mean(x)
    x_var_sum = np.sum(x_centered**2)

    if dof > 0 and x_var_sum > 0:
        residual_var = np.sum(residuals**2) / dof
        std_err = np.sqrt(residual_var / x_var_sum)
    else:
        std_err = np.nan

    # R² computation
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - ss_res / ss_total if ss_total > 0 else np.nan

    return pd.Series(
        {
            "change_intercept": model.intercept_,
            "change_rate": model.coef_[0],
            "change_rate_std_err": std_err,
            "r_squared": r_squared,
        }
    )


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
