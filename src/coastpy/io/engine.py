import os
from typing import Literal

import duckdb
import geopandas as gpd
import pandas as pd
import pystac
import shapely
from shapely.wkb import loads as wkb_loads

from coastpy.stac.utils import read_snapshot


class BaseQueryEngine:
    """
    Base class for querying geospatial data using DuckDB.

    Attributes:
        storage_backend (Literal["azure", "aws"]): The cloud storage backend.
        con (duckdb.DuckDBPyConnection): DuckDB connection object.
    """

    def __init__(self, storage_backend: Literal["azure", "aws"] = "azure") -> None:
        self.storage_backend = storage_backend
        self.con = duckdb.connect(database=":memory:", read_only=False)
        self._initialize_spatial_extension()
        self._configure_storage_backend()

    def _initialize_spatial_extension(self) -> None:
        """Initializes the spatial extension in DuckDB."""
        self.con.execute("INSTALL spatial;")
        self.con.execute("LOAD spatial;")

    def _configure_storage_backend(self) -> None:
        """Configures the storage backend for DuckDB."""
        if self.storage_backend == "azure":
            self.con.execute("INSTALL azure;")
            self.con.execute("LOAD azure;")

        elif self.storage_backend == "aws":
            self.con.execute("INSTALL httpfs;")
            self.con.execute("LOAD httpfs;")
            self.con.execute(f"SET s3_region = '{os.getenv('AWS_REGION')}';")
            self.con.execute(
                f"SET s3_access_key_id = '{os.getenv('AWS_ACCESS_KEY_ID')}';"
            )
            self.con.execute(
                f"SET s3_secret_access_key = '{os.getenv('AWS_SECRET_ACCESS_KEY')}';"
            )

    def _get_token(self) -> str:
        """
        Retrieve the storage SAS token based on the storage backend.

        Returns:
            str: The SAS token if the storage backend is Azure and the token is available, otherwise an empty string.
        """
        if self.storage_backend == "azure":
            sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
            return sas_token if sas_token else ""
        return ""

    def execute_query(self, query: str) -> gpd.GeoDataFrame | pd.DataFrame:
        """
        Executes a SQL query and returns the result as a GeoDataFrame if geometries exist.

        Args:
            query (str): The SQL query to execute.

        Returns:
            gpd.GeoDataFrame | pd.DataFrame: GeoDataFrame if a geometry column exists, otherwise a regular DataFrame.
        """
        df = self.con.execute(query).fetchdf()

        if df.empty:
            return pd.DataFrame()

        if "geometry" in df.columns:
            # Safely convert WKB to Shapely geometries, handling nulls
            df["geometry"] = df["geometry"].apply(
                lambda x: wkb_loads(bytes(x)) if pd.notnull(x) else None  # type: ignore
            )

            # Convert to GeoDataFrame with a default CRS (EPSG:4326) because of bbox search
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            return gdf

        return df


class STACQueryEngine(BaseQueryEngine):
    """
    A query engine optimized by STAC for querying geospatial data.

    Inherits from BaseQueryEngine.

    Attributes:
        quadtiles (gpd.GeoDataFrame): GeoDataFrame of quadtiles from the STAC collection.
        proj_epsg (int): EPSG code of the projection used in the quadtiles.
    """

    def __init__(
        self,
        stac_collection: pystac.Collection,
        storage_backend: Literal["azure", "aws"] = "azure",
        columns: list[str] | None = None,
    ) -> None:
        super().__init__(storage_backend=storage_backend)
        self.extents = read_snapshot(
            stac_collection,
        )
        try:
            self.proj_epsg = self.extents["proj:code"].unique().item()
        except KeyError:
            self.proj_epsg = self.extents["proj:epsg"].unique().item()

        if columns is None or not columns:
            # NOTE: before we used a wildcard, but that was tricky.. now we will rely on STAC? Also a bit
            # tricky if the STAC is not complete..
            # self.columns = ["*"]
            self.columns = [
                i["name"] for i in self.extents.assets.iloc[0]["data"]["table:columns"]
            ]
        else:
            self.columns = columns

    def get_data_within_bbox(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        sas_token: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Retrieve data within the specified bounding box.

        Args:
            minx (float): Minimum X coordinate of the bounding box.
            miny (float): Minimum Y coordinate of the bounding box.
            maxx (float): Maximum X coordinate of the bounding box.
            maxy (float): Maximum Y coordinate of the bounding box.

        Returns:
            pd.DataFrame: The queried data.
        """

        # Helper function to escape column names
        def escape_column(col):
            if col == "geometry":
                return "ST_AsWKB(ST_Transform(geometry, 'EPSG:4326', 'EPSG:4326')) AS geometry"
            return f'"{col}"'

        bbox = shapely.box(minx, miny, maxx, maxy)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

        # Perform spatial join to get all overlapping HREFs
        overlapping_hrefs = gpd.sjoin(self.extents, bbox_gdf).href.tolist()

        # Sign each HREF with the SAS token if the storage backend is Azure
        sas_token = self._get_token() if sas_token is None else sas_token
        if self.storage_backend == "azure":
            signed_hrefs = []
            for href in overlapping_hrefs:
                signed_href = href.replace(
                    "az://", "https://coclico.blob.core.windows.net/"
                )
                signed_href = signed_href + "?" + sas_token
                signed_hrefs.append(signed_href)
        else:
            signed_hrefs = overlapping_hrefs

        # Join the hrefs into a single string
        hrefs_str = ", ".join(f'"{href}"' for href in signed_hrefs)

        columns_str = ", ".join(escape_column(col) for col in self.columns)

        query = f"""
            SELECT {columns_str}
            FROM read_parquet([{hrefs_str}])
            WHERE
                bbox.xmin <= {maxx} AND
                bbox.ymin <= {maxy} AND
                bbox.xmax >= {minx} AND
                bbox.ymax >= {miny};
        """
        return self.execute_query(query)


class HREFQueryEngine(BaseQueryEngine):
    """
    A query engine for directly querying geospatial data using an HREF.

    Inherits from BaseQueryEngine.

    Attributes:
        href (str): The direct link (HREF) to the geospatial data.
    """

    def __init__(
        self, href: str, storage_backend: Literal["azure", "aws"] = "azure"
    ) -> None:
        super().__init__(storage_backend=storage_backend)
        self.href = href

    def get_data_within_bbox(
        self, minx: float, miny: float, maxx: float, maxy: float
    ) -> gpd.GeoDataFrame:
        """
        Queries data within a specified bounding box from a direct HREF.

        Args:
            minx (float): Minimum X coordinate (longitude) of the bounding box.
            miny (float): Minimum Y coordinate (latitude) of the bounding box.
            maxx (float): Maximum X coordinate (longitude) of the bounding box.
            maxy (float): Maximum Y coordinate (latitude) of the bounding box.

        Returns:
            gpd.GeoDataFrame: Queried data within the bounding box as a GeoDataFrame.
        """
        query = f"""
        SELECT *
        FROM read_parquet('{self.href}', hive_partitioning = false)
        WHERE
            bbox.xmin <= {maxx} AND
            bbox.ymin <= {maxy} AND
            bbox.xmax >= {minx} AND
            bbox.ymax >= {miny};
        """
        return self.execute_query(query)
