import dask.bag as db
import geopandas as gpd
import odc.stac
import stac_geoparquet
import xarray as xr


def process_row(sample: gpd.GeoDataFrame) -> xr.Dataset | None:
    """Converts a STAC GeoParquet row into an Xarray dataset."""
    try:
        ds = odc.stac.load(stac_geoparquet.to_item_collection(sample)).squeeze(
            drop=True
        )
        ds = ds.drop_vars("spatial_ref", errors="ignore")

        # These are STAC GeoParquet that cannot be added as coordinates to the dataset
        sample = sample.drop(
            columns=[
                "type",
                "stac_version",
                "stac_extensions",
                "bbox",
                "links",
                "assets",
                "collection",
                "created",
                "datetime",
            ]
        )
        # Add the remaining metadata as coordinates
        meta = sample.iloc[0].to_dict()
        meta["stac_id"] = meta.pop("id", None)
        coords = {k: (("uuid",), [v]) for k, v in meta.items()}
        coords["uuid"] = (("uuid",), [meta["uuid"]])
        return ds.assign_coords(**coords)  # type: ignore

    except Exception as e:
        print(f"Error processing {sample.get('id', 'UNKNOWN')}: {e}")
        return None


def load_stac_xr(df: gpd.GeoDataFrame, use_dask=False) -> xr.Dataset:
    """Loads STAC GeoParquet training items into an Xarray dataset, optionally using Dask Bag for efficiency."""

    if use_dask:
        bag = db.from_sequence([df.iloc[[i]] for i in range(len(df))])
        delayed_datasets = bag.map(process_row)
        datasets = delayed_datasets.compute()

    else:
        datasets = []
        for i in range(len(df)):
            sample = df.iloc[[i]]
            ds = process_row(sample)
            datasets.append(ds)

    if not datasets:
        raise ValueError("No valid datasets found.")

    return xr.concat(
        datasets, dim="uuid", join="override", combine_attrs="no_conflicts"
    )
