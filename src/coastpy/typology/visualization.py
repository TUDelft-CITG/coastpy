import cartopy.crs as ccrs
import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot.pandas  # noqa: F401
import panel as pn

from coastpy.typology.encoding import (
    encode_categorical,
    get_encoding_and_colors_from_gdf,
    get_ordered_color_list,
)


def plot_coastal_typology(
    gdf: gpd.GeoDataFrame,
    column: str,
    projection: ccrs.Projection | None = None,
    tiles: str | None = "ESRI",
    frame_width: int = 800,
    grid: bool = False,
    colorbar: bool = False,
    hover_cols: list[str] | None = None,
    **kwargs,
) -> hv.Overlay:
    """Plot geospatial features colored by encoded class values using hvplot.

    This function automatically encodes class values, applies a consistent colormap,
    and generates an interactive Holoviews plot.

    Args:
        gdf: GeoDataFrame with geometries and class column (e.g. 'class:shore_type').
        column: Name of class column to visualize.
        projection: Optional Cartopy projection (e.g. ccrs.Robinson()).
        tiles: Basemap tile source (default is 'ESRI').
        frame_width: Width of the plot in pixels (default is 800).
        grid: Whether to overlay a cartographic grid (default is False).
        colorbar: Whether to show a colorbar (default is False).
        hover_cols: Columns to display on hover (default is [column]).
        **kwargs: Additional keyword arguments passed to hvplot.

    Returns:
        Holoviews Overlay with colored geometries and optional grid.
    """
    encoding, color_map = get_encoding_and_colors_from_gdf(gdf)
    gdf = encode_categorical(gdf.copy(), column, encoding)

    code_column = f"{column}_code"
    cmap = get_ordered_color_list(column, encoding, color_map)
    hover_cols = hover_cols or [column]

    plot = gdf.hvplot(
        geo=True,
        color=code_column,
        cmap=cmap,
        tiles=tiles,
        frame_width=frame_width,
        colorbar=colorbar,
        hover_cols=hover_cols,
        projection=projection,
        **kwargs,
    )  # type: ignore

    if grid:
        grid_layer = gv.feature.grid(
            projection=projection, color="gray", title="", fill_color="none"
        )
        return plot * grid_layer

    return plot


def build_legend_html(
    gdf: gpd.GeoDataFrame,
    column: str,
) -> pn.pane.HTML:
    """Build an HTML categorical legend for a given column.

    Args:
        gdf: GeoDataFrame with geometries and class column (e.g. 'class:shore_type').
        column: Name of class column to visualize.

    Returns:
        Panel HTML pane with rendered color legend.
    """

    encoding, color_map = get_encoding_and_colors_from_gdf(gdf)

    encoded = encoding[column]
    colors = color_map[column]

    # Sort by encoding index
    sorted_classes = sorted(encoded.items(), key=lambda x: x[1])

    html = f"<b>{column}</b><br>"
    for class_name, _ in sorted_classes:
        hex_color = colors.get(class_name.lower(), "#CCCCCC")
        html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 15px; height: 15px; background: {hex_color};
                        margin-right: 6px; border: 1px solid #aaa;"></div>
            <div>{class_name}</div>
        </div>
        """
    return pn.pane.HTML(html, width=250)


class CoastalTypologyDashboard:
    def __init__(self, gdf):
        self.gdf = gdf
        self.columns = [
            "class:shore_type",
            "class:coastal_type",
            "class:is_built_environment",
            "class:has_defense",
        ]
        self.app = self._build_layout()

    def _build_layout(self):
        rows = []
        for col in self.columns:
            plot = plot_coastal_typology(self.gdf, col)
            legend = build_legend_html(self.gdf, col)
            rows.append(pn.Row(plot, legend))

        return pn.Column(*rows)

    def show(self):
        return self.app.show()

    def panel(self):
        return self.app
