# CoastPy

Python tools for cloud-native coastal analytics.

## Installation

```bash
$ pip install coastpy
```

## Usage
More extensive installation and usage instructions will be made available by
documentation. For now you can clone the repository, install the environment, that comes
with jupyterlab and then launch jupyter lab to explore the tutorials.

```bash
git clone https://TUDelft-CITG/coastpy.git
cd coastpy
mamba env create -f environment.yml
jupyter lab
```

## Data

The data that is produced with this software can be directly accessed via the cloud using
tools like
[DuckDB](https://duckdb.org/docs/installation/?version=stable&environment=cli&platform=macos&download_method=package_manager);
see the [tutorials](./tutorials/) for other methods and latest instructions.

### Global Coastal Transect System (GCTS)

Cross-shore coastal transects are essential to coastal monitoring, offering a consistent
reference line to measure coastal change, while providing a robust foundation to map
coastal characteristics and derive coastal statistics thereof.

The Global Coastal Transect System consists of more than 11 million cross-shore coastal
transects uniformly spaced at 100-m intervals alongshore, for all OpenStreetMap
coastlines that are longer than 5 kilometers.


```bash
# Download all transects in the United States.
duckdb -c "COPY (SELECT * FROM 'az://coclico.blob.core.windows.net/gcts/release/2024-08-02/*.parquet' AS gcts WHERE gcts.country = 'US') TO 'United_States.parquet' (FORMAT 'PARQUET')"
```

```bash
# Or, download the data in bulk using AZ CLI
az storage blob download-batch \
    --destination "./" \
    --source "gcts" \
    --pattern "release/2024-08-02/*.parquet" \
    --account-name coclico
```

## Citation:

```latex
@article{CALKOEN2024106257,
  title   = {Enabling coastal analytics at planetary scale},
  author  = {Floris Reinier Calkoen and Arjen Pieter Luijendijk and Kilian Vos and Etiënne Kras and Fedor Baart},
  journal = {Environmental Modelling \& Software},
  year    = {2024},
  pages   = {106257},
  issn    = {1364-8152},
  doi     = {10.1016/j.envsoft.2024.106257},
  url     = {https://www.sciencedirect.com/science/article/pii/S1364815224003189}
}
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`coastpy` was created by Floris Calkoen. The software is licensed under the terms of the
MIT license. Data licenses can be found in the respective STAC collection.

## Credits

Initial template of `coastpy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
