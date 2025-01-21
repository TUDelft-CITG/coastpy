# CoastPy

Python tools for cloud-native coastal analytics.

## Installation

You can install coastpy with pip in a Python environment where [GDAL](https://pypi.org/project/GDAL/) and [pyproj](https://pypi.org/project/pyproj/) are already installed.

```bash
pip install coastpy
```

However, if you start from scratch, it's probably easier to install with conda:

```bash
conda env create -f https://raw.githubusercontent.com/TUDelft-CITG/coastpy/refs/heads/main/environment.yaml
```

## Data

The data that is produced with this software can be directly accessed via the cloud using
tools like
[DuckDB](https://duckdb.org/docs/installation/?version=stable&environment=cli&platform=macos&download_method=package_manager);
see the [tutorials](./tutorials/) and [analytics](./analytics/) other for other access
methods (Python) and latest usage instructions.

### Global Coastal Transect System (GCTS)

Cross-shore coastal transects are essential to coastal monitoring, offering a consistent
reference line to measure coastal change, while providing a robust foundation to map
coastal characteristics and derive coastal statistics thereof.

The Global Coastal Transect System consists of more than 11 million cross-shore coastal
transects uniformly spaced at 100-m intervals alongshore, for all OpenStreetMap
coastlines that are longer than 5 kilometers.

```bash
# Download all transects located in the United States.
duckdb -c "COPY (SELECT * FROM 'az://coclico.blob.core.windows.net/gcts/release/2024-08-02/*.parquet' AS gcts WHERE gcts.country = 'US') TO 'United_States.parquet' (FORMAT 'PARQUET')"
```

```bash
# Download transects by bounding box.
duckdb -c "COPY (SELECT * FROM 'az://coclico.blob.core.windows.net/gcts/release/2024-08-02/*.parquet' AS gcts WHERE bbox.xmin <= 14.58 AND bbox.ymin <= -22.77 AND bbox.xmax >= 14.27 AND bbox.ymax >= -23.57) TO area_of_interest.parquet (FORMAT 'PARQUET')"
```

```bash
# Or, download the data in bulk using AZ CLI
az storage blob download-batch \
    --destination "./" \
    --source "gcts" \
    --pattern "release/2024-08-02/*.parquet" \
    --account-name coclico
```

### Coastal Grid

The Coastal Grid dataset provides a global tiling system for coastal analytics. It
supports scalable data processing workflows by offering coastal tiles at varying zoom
levels (5, 6, 7, 8, 9, 10) and buffer sizes (500 m, 1000 m, 2000 m, 5000 m, 10000 m, 15000 m).

## Usage instructions

Better installation and usage instructions will come when we build the documentation. For
now, to run the tutorials, analytics or scripts proceed as follows.

### Installation

1. Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) or [GitHub Desktop](https://github.com/apps/desktop)
2. Clone [CoastPy](https://github.com/TUDelft-CITG/coastpy)
3. Install [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install)
4. Open a Miniforge prompt (finder/spotlight)
5. Run the following commands:

```bash
# Make sure to update if you already had miniforge installed
conda update --all --yes

# Create the software environment
conda env create -f https://raw.githubusercontent.com/TUDelft-CITG/coastpy/refs/heads/main/environment.yaml
```

### Usage

1. Open Miniforge
2. Change to the directory where CoastPy was cloned by using `cd /path/to/coastpy`
3. Activate the software environment by `mamba activate coastal`
4. Launch Jupyter lab by `jupyter lab`
5. Navigate to the [tutorials](https://github.com/TUDelft-CITG/coastpy/tree/main/tutorials) folder in Jupyter lab


## Citation:

```latex
@article{CALKOEN2025106257,
  title     = {Enabling coastal analytics at planetary scale},
  journal   = {Environmental Modelling & Software},
  volume    = {183},
  pages     = {106257},
  year      = {2025},
  issn      = {1364-8152},
  doi       = {https://doi.org/10.1016/j.envsoft.2024.106257},
  url       = {https://www.sciencedirect.com/science/article/pii/S1364815224003189},
}
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`coastpy` was created by Floris Calkoen. The software is licensed under the terms of the
MIT license. Data licenses are typically CC-BY-4.0, and can be found in the respective
STAC collection.
