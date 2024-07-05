from pathlib import Path

import dask_geopandas
import fsspec
import mercantile

from coastpy.geo.quadtiles_utils import add_geo_columns
from coastpy.geo.size import estimate_memory_usage_per_row
from coastpy.io.utils import name_data
from coastpy.utils.size_utils import size_to_bytes


class EqualSizePartitioner:
    def __init__(
        self,
        df,
        out_dir,
        max_size,
        sort_by="quadkey",
        quadkey_zoom_level=12,
        geo_columns=None,
        column_order=None,
        dtypes=None,
    ):
        if geo_columns is None:
            geo_columns = ["bbox", "quadkey", "bounding_quadkey"]

        self.df = df
        self.out_dir = Path(out_dir)
        self.max_size_bytes = size_to_bytes(max_size)
        self.sort_by = sort_by
        self.quadkey_zoom_level = quadkey_zoom_level
        self.geo_columns = geo_columns
        self.column_order = column_order
        self.dtypes = dtypes

        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Set the naming function for the output files
        self.naming_function = name_data

    def preprocess(self):
        """
        Preprocess the dataframe by estimating row size and cumulative sum.
        Also, ensure necessary columns like 'quadkey', 'bbox', and 'bounding_quadkey' are present.
        """
        df = self.df.copy()  # Work on a copy to avoid modifying the original dataframe
        df["size"] = estimate_memory_usage_per_row(df)

        if not all(column in df.columns for column in self.geo_columns):
            df = add_geo_columns(
                df,
                geo_columns=self.geo_columns,
                quadkey_zoom_level=self.quadkey_zoom_level,
            )

        df = df.sort_values(self.sort_by).reset_index(drop=True)
        df["size_cumsum"] = df["size"].cumsum()

        return df

    def process(self):
        """
        Process each group of data based on 'quadkey'.
        """
        df = self.preprocess()
        while not df.empty and df["size_cumsum"].max() > self.max_size_bytes:
            split_index = df[df["size_cumsum"] > self.max_size_bytes].index.min() + 1
            partition_df = df.iloc[:split_index].reset_index(drop=True)
            df = df.iloc[split_index:].reset_index(drop=True)
            df["size_cumsum"] = df["size"].cumsum()

            if not partition_df.empty:
                self.write_data(partition_df)

        if not df.empty:
            self.write_data(df)

    def write_data(self, partition_df, column_order=None):
        """
        Write the processed dataframe partition to a parquet file using fsspec
        to support both local and cloud storage.
        """
        if not partition_df.empty:
            partition_df = partition_df[column_order] if column_order else partition_df
            # Generate the output path using the naming function
            outpath = self.naming_function(partition_df, prefix=str(self.out_dir))

            # Ensure the output directory exists
            fs = fsspec.open(outpath, "wb").fs
            fs.makedirs(fs._parent(outpath), exist_ok=True)

            if self.dtypes:
                partition_df = partition_df.astype(self.dtypes)

            if self.column_order:
                partition_df = partition_df[self.column_order]

            # Use fsspec to write the DataFrame to parquet
            with fs.open(outpath, "wb") as f:
                partition_df.to_parquet(f, index=False)


class QuadKeyEqualSizePartitioner(EqualSizePartitioner):
    def __init__(
        self,
        df,
        out_dir,
        max_size,
        min_quadkey_zoom,
        sort_by,
        quadkey_zoom_level=12,
        geo_columns=None,
        column_order=None,
        dtypes=None,
    ):
        super().__init__(
            df,
            out_dir,
            max_size,
            sort_by,
            quadkey_zoom_level,
            geo_columns,
            column_order,
            dtypes,
        )
        self.min_quadkey_zoom = min_quadkey_zoom
        self.quadkey_grouper = f"quadkey_z{min_quadkey_zoom}"
        self.column_order = column_order
        self.dtypes = dtypes

    def add_quadkey_group(self):
        """
        Adds a column to the DataFrame with quadkey based on the specified zoom level.
        """
        self.df[self.quadkey_grouper] = self.df.apply(
            lambda r: mercantile.quadkey(
                mercantile.tile(r.lon, r.lat, self.min_quadkey_zoom)
            ),
            axis=1,
        )

    def process(self):
        """
        Override the process method to handle quadkey-based partitioning.
        """
        self.add_quadkey_group()
        grouped = self.df.groupby(self.quadkey_grouper)

        for _, group in grouped:
            group2 = group.drop(columns=[self.quadkey_grouper], errors="ignore")
            partitioner = EqualSizePartitioner(
                group2,
                self.out_dir,
                self.max_size_bytes,
                self.sort_by,
                self.quadkey_zoom_level,
                self.geo_columns,
                self.column_order,
                self.dtypes,
            )
            partitioner.process()

    def write_data(self, partition_df):
        """
        Writes each partition to a separate parquet file.
        """
        super().write_data(partition_df)


class HivePartitionEqualSizePartitioner:
    def __init__(self, data_dir, out_dir, max_size, dtypes):
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.max_size = max_size
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dtypes = dtypes

    def read_hive_partitions(self):
        """
        Read data from Hive-partitioned directories and aggregate into a single DataFrame.
        """
        pattern = str(self.data_dir / "*=*/" / "*.parquet")
        df = dask_geopandas.read_parquet(pattern, engine="pyarrow")
        return df

    def process(self):
        """
        Process data from Hive-partitioned directories for equal-size partitioning.
        """
        df = self.read_hive_partitions().compute()
        partitioner = EqualSizePartitioner(df, self.out_dir, self.max_size, self.dtypes)
        partitioner.process()
