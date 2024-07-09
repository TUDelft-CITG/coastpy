import os
from typing import Literal

import dotenv
import duckdb
import geopandas as gpd
import pystac
import shapely
from shapely.wkb import loads as wkb_loads

from coastpy.io.utils import read_items_extent

dotenv.load_dotenv(override=True)


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

            if duckdb.__version__ > "0.10.0":
                try:
                    self.con.execute(
                        f"""
                                CREATE SECRET secret1 (
                                TYPE AZURE,
                                CONNECTION_STRING '{os.getenv('CLIENT_AZURE_STORAGE_CONNECTION_STRING')}');
                    """
                    )
                except Exception as e:
                    print(f"{e}")

            else:
                self.con.execute(
                    f"SET AZURE_STORAGE_CONNECTION_STRING = '{os.getenv('CLIENT_AZURE_STORAGE_CONNECTION_STRING')}';"
                )

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

    def execute_query(self, query: str) -> gpd.GeoDataFrame:
        """
        Executes a SQL query and returns the result as a GeoDataFrame.

        Args:
            query (str): The SQL query to execute.

        Returns:
            gpd.GeoDataFrame: The query result as a GeoDataFrame.
        """
        df = self.con.execute(query).fetchdf()
        if not df.empty:
            df["geometry"] = df.geometry.map(lambda b: wkb_loads(bytes(b)))
            return gpd.GeoDataFrame(df, crs="EPSG:4326")
        return gpd.GeoDataFrame()


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
        self.extents = read_items_extent(
            stac_collection, columns=["geometry", "assets", "proj:epsg"]
        )
        self.proj_epsg = self.extents["proj:epsg"].unique().item()

        if columns is None or not columns:
            self.columns = ["*"]
        else:
            self.columns = columns

    def get_data_within_bbox(self, minx, miny, maxx, maxy):
        bbox = shapely.box(minx, miny, maxx, maxy)
        bbox = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

        # Perform spatial join to get all overlapping HREFs
        overlapping_hrefs = gpd.sjoin(self.extents, bbox).href.tolist()

        href = str(overlapping_hrefs).replace("'", '"')

        # Construct and execute the query
        columns_str = ", ".join(self.columns)
        query = f"""
        SELECT {columns_str}
        FROM read_parquet({href})
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
