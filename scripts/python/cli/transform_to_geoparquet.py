import os
import pathlib
import sys

src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
sys.path.append(src_dir)


import dask_geopandas

if __name__ == "__main__":
    # TODO: refactor example below to CLI tool

    coastlines_fp = pathlib.Path.home().joinpath(
        "data", "src", "coastlines_vector", "lines.shp"
    )

    outdir = pathlib.Path.home().joinpath("data", "prc", "coastlines", "osm_vector2")

    ddf = dask_geopandas.read_file(coastlines_fp, npartitions=20)
    ddf = ddf.to_crs("EPSG:3857")
    ddf.calculate_spatial_partitions()

    # name_function = lambda x: f"coastlines{x}.parquet"
    # transects_dir = local_dirs["temp"].joinpath("transects6")
    print(f"Writing transects to {outdir}...")
    ddf.to_parquet(outdir, write_index=True)
    print("Done!")

    # # take a sample but keep same amount of partitions. Recalculating spatial partitions is required.
    # ddf = ddf.sample(frac=0.01)
    # ddf.calculate_spatial_partitions()

    # print("Indexing data")
    # bbox = build_bbox(-180, -60, 180, 60, src_crs="epsg:4326", dst_crs="epsg:3857")
    # ddf = ddf.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    # ddf.calculate_spatial_partitions()

    # name_function = lambda x: f"coastlines{x}.gpq"
    # outdir = coastlines_dir.joinpath("osm_z8_sample")
    # print(f"Writing transects to {outdir}...")
    # ddf.to_parquet(outdir, name_function=name_function, write_index=True)
    # print("Done!")
