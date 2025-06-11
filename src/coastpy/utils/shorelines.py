import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_ols_fit(x: np.ndarray, y: np.ndarray) -> pd.Series:
    if len(x) < 3:  # need at least 3 points for dof > 0
        return pd.Series(
            {
                "change_intercept": np.nan,
                "change_rate": np.nan,
                "change_rate_std_err": np.nan,
                "r_squared": np.nan,
            }
        )

    x = x.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    residuals = y - y_pred
    dof = len(y) - 2

    if dof <= 0 or np.sum((x - np.mean(x)) ** 2) == 0:
        std_err = np.nan
    else:
        residual_var = np.sum(residuals**2) / dof
        std_err = np.sqrt(residual_var / np.sum((x - np.mean(x)) ** 2))

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - ss_res / ss_total if ss_total != 0 else np.nan

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
