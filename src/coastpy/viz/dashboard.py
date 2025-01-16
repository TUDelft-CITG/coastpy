import colorcet as cc
import geopandas as gpd
import geoviews as gv
import geoviews.tile_sources as gvts
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
from holoviews import streams
from shapely.geometry import Point

from coastpy.utils.shorelines import compute_ols_fit, compute_ols_trend

pn.extension("hvplot")


class ShorelineSeriesApp:
    def __init__(self, time_series: gpd.GeoDataFrame):
        self.time_series = time_series
        self.setup()

        self.tiles = gvts.EsriImagery()
        self.point_draw = gv.Points([]).opts(
            size=10, color="red", tools=["hover"], width=800, height=450
        )
        self.default_transect = np.random.choice(
            self.time_series["transect_id"].unique()
        )

        self.setup_ui()
        self.view = self.create_view()

    def setup(self):
        """Preprocess the dataset and compute trends."""
        self.origins = gpd.GeoDataFrame(
            self.time_series[["transect_id"]],
            geometry=gpd.points_from_xy(
                self.time_series.lon, self.time_series.lat, crs="EPSG:4326"
            ),
        ).to_crs("EPSG:3857")
        self.ac = compute_ols_trend(self.time_series, x="datetime", y="chainage")

    def get_nearest_transect(self, x, y):
        point = Point(x, y)
        point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs(
            "EPSG:3857"
        )
        transect_id = gpd.sjoin_nearest(point_gdf, self.origins).transect_id.iloc[0]
        return transect_id

    def setup_ui(self):
        self.point_draw_stream = streams.PointDraw(
            source=self.point_draw, num_objects=1
        )
        self.point_draw_stream.add_subscriber(self.on_point_draw)

    def plot_ambient_change(self):
        # Calculate the 2.5th and 97.5th percentiles to exclude extreme outliers
        lower_quantile = self.ac["sds:change_rate"].quantile(0.025)
        upper_quantile = self.ac["sds:change_rate"].quantile(0.975)

        # Set the plot range considering the quantiles
        min_range = max(lower_quantile, -15)
        max_range = min(upper_quantile, 15)

        # Ensure that the color limits are symmetric around zero
        max_abs_range = max(abs(min_range), abs(max_range))
        clim = (-max_abs_range, max_abs_range)

        # Plot using the calculated range
        ac_plot = self.ac.hvplot(
            geo=True,
            tiles=None,
            color="sds:change_rate",
            frame_width=650,
            frame_height=500,
            colorbar=True,
            cnorm="linear",
            cmap=cc.CET_D3[::-1],
            clim=clim,
        )  # type: ignore
        return (ac_plot * self.tiles * self.point_draw).opts(
            title="Ambient shoreline-change"
        )

    def plot_time_series(self, transect_name):
        sample = self.time_series[
            self.time_series["transect_id"] == transect_name
        ].sort_values("datetime")

        # Plot by obs_is_primary with additional attributes
        plot_obs_is_primary = sample[sample["obs_is_primary"]][
            ["datetime", "chainage"]
        ].hvplot.scatter(
            x="datetime",
            y="chainage",
            color="green",
            marker="o",
            label="Primary observation",
        )

        # Plot by obs_is_primary with additional attributes
        plot_obs_is_not_primary = sample[~sample["obs_is_primary"]][
            ["datetime", "chainage"]
        ].hvplot.scatter(
            x="datetime",
            y="chainage",
            color="red",
            marker="o",
            label="Non-primary observation",
        )

        plot_step_changes = sample[sample["obs_is_step_change"]].hvplot.scatter(
            x="datetime",
            y="chainage",
            color="orange",
            marker="x",
            size=200,
            label="Obs > max. step change",
        )

        plot_outliers = sample[sample["obs_is_outlier"]].hvplot.scatter(
            x="datetime",
            y="chainage",
            color="purple",
            marker="x",
            size=200,
            label="Obs is outlier",
        )

        # Overlay plot with series group as outer fill
        plot_subseries_id = sample[
            ["datetime", "chainage", "subseries_id"]
        ].hvplot.scatter(
            x="datetime",
            y="chainage",
            by="subseries_id",
            label="Series group",
            fill_alpha=0,
            line_alpha=1,
            line_width=2,
        )

        return (
            plot_obs_is_primary
            * plot_obs_is_not_primary
            * plot_subseries_id
            * plot_step_changes
            * plot_outliers
        ).opts(title=f"All observations for {transect_name}")

    def plot_time_series2(self, transect_name):
        data = self.time_series[
            self.time_series["transect_id"] == transect_name
        ].sort_values("datetime")

        # Compute the trend line for primary observations
        primary_obs = data[data["obs_is_primary"]]
        ols_fit = compute_ols_fit(
            primary_obs["datetime"].dt.year.values,
            primary_obs.shoreline_position.values,
        )

        # Generate trend line values
        x_values = data["datetime"].dt.year.values
        y_values = ols_fit["intercept"] + ols_fit["slope"] * x_values

        trend_line = pd.DataFrame(
            {"datetime": pd.to_datetime(x_values, format="%Y"), "trend": y_values}
        )

        plot_obs_is_primary = data[data["obs_is_primary"]].hvplot.scatter(
            x="datetime",
            y="shoreline_position",
            color="green",
            marker="o",
            label="Primary observation",
        )

        plot_obs_is_outlier = data[data["obs_is_outlier"]].hvplot.scatter(
            x="datetime",
            y="shoreline_position",
            color="red",
            marker="o",
            label="MAD Outlier",
        )
        # # Plot trend line
        plot_trend_line = trend_line.hvplot.line(
            x="datetime",
            y="trend",
            color="gray",
            line_width=2,
            label="Trend Line",
            line_alpha=0.5,
        )

        return (plot_trend_line * plot_obs_is_primary * plot_obs_is_outlier).opts(
            title=f"Trend for {transect_name}: {round(ols_fit.slope, 2)} m/yr"
        )

    def plot_observations(self, transect_name):
        sample = self.time_series[
            self.time_series["transect_id"] == transect_name
        ].sort_values("datetime")

        # Base observation plot
        observations_plot = sample.assign(
            year=pd.to_datetime(sample["datetime"]).dt.year
        ).hvplot.points(
            geo=True,
            tiles=None,
            width=600,
            height=500,
            color="year",
            cmap=cc.bmy,
            title=f"Observations for {transect_name}",
        )

        # Create the circles for primary and non-primary observations
        circles_primary = gv.Points(
            sample[sample["obs_is_primary"]][["geometry"]]
        ).opts(size=5, fill_alpha=0, line_color="green", line_width=2)

        circles_non_primary = gv.Points(
            sample[~sample["obs_is_primary"]][["geometry"]]
        ).opts(size=5, fill_alpha=0, line_color="red", line_width=2)

        # Create markers for outliers and step changes
        markers_outliers = gv.Points(
            sample[sample["obs_is_outlier"]][["geometry"]]
        ).opts(size=8, marker="x", color="purple", line_width=2)

        markers_step_changes = gv.Points(
            sample[sample["obs_is_step_change"]][["geometry"]]
        ).opts(size=8, marker="x", color="orange", line_width=2)

        return (
            observations_plot
            * circles_primary
            * circles_non_primary
            * markers_outliers
            * markers_step_changes
            * self.tiles
        )

    def on_point_draw(self, data):
        if data:
            x, y = data["Longitude"][0], data["Latitude"][0]
            self.update_view(x, y)

    def update_view(self, x, y):
        try:
            transect_name = self.get_nearest_transect(x, y)
            new_all_obs_plot = self.plot_time_series(transect_name)
            new_primary_obs_plot = self.plot_time_series2(transect_name)
            new_observations_plot = self.plot_observations(transect_name)
            self.view[0, 1].object = new_observations_plot
            self.view[1, 0].object = new_all_obs_plot
            self.view[1, 1].object = new_primary_obs_plot
        except Exception:
            ambient_change_plot = self.plot_ambient_change()
            all_obs_plot = self.plot_time_series(self.default_transect)
            primary_obs_plot = self.plot_time_series2(self.default_transect)
            observations_plot = self.plot_observations(self.default_transect)

            self.view[0, 0] = pn.pane.HoloViews(
                ambient_change_plot, sizing_mode="stretch_both"
            )
            self.view[0, 1] = pn.pane.HoloViews(
                observations_plot, sizing_mode="stretch_both"
            )
            self.view[1, 0] = pn.pane.HoloViews(
                all_obs_plot, sizing_mode="stretch_both"
            )
            self.view[1, 1] = pn.pane.HoloViews(
                primary_obs_plot, sizing_mode="stretch_both"
            )

    def create_view(self):
        ambient_change_plot = self.plot_ambient_change()
        all_obs_plot = self.plot_time_series(self.default_transect)
        primary_obs_plot = self.plot_time_series2(self.default_transect)
        observations_plot = self.plot_observations(self.default_transect)
        self.view = pn.GridSpec(ncols=2, nrows=2, width=1600, height=1000)
        self.view[0, 0] = pn.pane.HoloViews(
            ambient_change_plot, sizing_mode="stretch_both"
        )
        self.view[0, 1] = pn.pane.HoloViews(
            observations_plot, sizing_mode="stretch_both"
        )
        self.view[1, 0] = pn.pane.HoloViews(all_obs_plot, sizing_mode="stretch_both")
        self.view[1, 1] = pn.pane.HoloViews(
            primary_obs_plot, sizing_mode="stretch_both"
        )
        return self.view

    def show(self):
        """Display the dashboard in the browser."""
        self.create_view().show()
