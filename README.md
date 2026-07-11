# UnumLocalia

**UnumLocalia** is an open-source Python toolkit accompanying multimodal spatial biology datasets.
It provides a simple, reproducible environment for visualisation, exploration,
segmentation benchmarking, and basic single-cell analysis — all without modifying
your original data.

---

## Quickstart

**Three steps for non‑programmers**

1. **Install Miniforge and mamba**  
   - Install Miniforge from https://github.com/conda-forge/miniforge and follow the installer for your OS.  
   - Open Terminal and run:
   ```bash
   conda install -n base -c conda-forge mamba
   ```

2. **Create the environment and install UnumLocalia**
```bash
# from the repository root
mamba env create -f environment.yml
eval "$(mamba shell hook --shell zsh)"   # follow printed instructions for your shell
mamba activate unumlocalia
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e .
```

3. **Download the dataset and launch the GUI**
    - Download the dataset ZIP from the DOI listed below and extract it to a folder, for example ~/UnumLocalia-dataset.
    - Launch UnumLocalia:
    ```bash
    mamba activate unumlocalia
    unumlocalia
    ```
    - In the app, click File → Load dataset and select the dataset root folder.

---

## Features

| Module | Functionality |
|---|---|
| **Viewer** | Interactive napari viewer for H&E, COMET, Xenium transcripts, and cell boundaries |
| **Segmentation** | Load label masks (GeoJSON files) |
| **Cell Quanfication** | Measure COMET intensities, assign transcripts |
| **Export** | Figures (PNG), cell quantification (CSV) |

---

## Supported data modalities

UnumLocalia expects a dataset folder containing:

```
dataset/
    core01_hcc/
        comet/
            comet_thresholding.csv
            core01_comet.ome.zarr
            keypoints_comet.csv
            matrix_comet.csv
        he/
            core01_he.ome.zarr
            keypoints_he.csv
            matrix_he.csv
        xenium/
            cell_boundaries_comet_space.geojson
            cell_boundaries.parquet
            cells.csv
            nucleus_boundaries.parquet
            transcripts.parquet
        
    core02_non_tumour/
        (same as core01)
    core03_tonsil/
        (same as core01)
    core04_hca/
        (same as core01)
    dataset_manifest.csv
```

All files are auto-detected — no manual configuration required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Felixillion/UnumLocalia.git
cd UnumLocalia
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate unumlocalia
```

### 3. Verify the installation

```bash
unumlocalia --version   # prints 1.0.0
```

---

## Quick start

### Launch the GUI

```bash
conda activate unumlocalia
unumlocalia
```

A napari window will open with the UnumLocalia panel docked on the right.
Go to the **Settings** tab, select your dataset folder, and click **Load dataset**.

Alternatively, launch from Python:

```python
from unumlocalia.widgets import launch
launch()
```

### Programmatic use

```python
from unumlocalia import DatasetLoader

# Load dataset
loader = DatasetLoader("/path/to/dataset")
loader.load()

# Inspect
print(loader.manifest.summary())
print(f"Genes: {len(loader.genes)}   Proteins: {len(loader.proteins)}")

# Run analysis
from unumlocalia import run_pca, run_umap, run_leiden, plot_umap

adata = loader.anndata_ref
run_pca(adata, modality="genes")
run_umap(adata, modality="genes")
run_leiden(adata, modality="genes", resolution=0.5)

fig = plot_umap(adata, color="leiden_genes", modality="genes")
fig.savefig("umap_clusters.png", dpi=150, bbox_inches="tight")
```

### Benchmark a custom segmentation

```python
from unumlocalia import (
    DatasetLoader,
    load_segmentation,
    measure_comet_intensities,
    assign_xenium_transcripts,
    build_anndata,
    compare_segmentations,
)

loader = DatasetLoader("/path/to/dataset").load()

# Load user segmentation (label mask, GeoJSON, or QuPath JSON)
labels = load_segmentation("/path/to/my_segmentation.tif")

# Measure protein intensities
comet_df = measure_comet_intensities(labels, loader.comet_arrays)

# Assign transcripts
tx_df = assign_xenium_transcripts(labels, loader.transcripts_df)

# Build fresh AnnData
adata_user = build_anndata(labels, comet_df, tx_df, loader.cells_df)

# Compare clusterings
from unumlocalia import compare_clusterings, plot_contingency
results = compare_clusterings(
    adata_original.obs["leiden_genes"],
    adata_user.obs["leiden_genes"],
)
print(f"ARI: {results['ari']:.4f}   NMI: {results['nmi']:.4f}")
```

---

## Repository structure

```
UnumLocalia/
├── unumlocalia/
│   ├── __init__.py        ← Public API + version
│   ├── io.py              ← Auto-detection, lazy loading, alignment
│   ├── viewer.py          ← napari layer management + cell inspector
│   ├── segmentation.py    ← Segmentation loading + AnnData construction
│   ├── analysis.py        ← PCA, UMAP, Leiden, Scanpy wrappers
│   ├── benchmark.py       ← ARI, NMI, Sankey, marker correlation
│   ├── widgets.py         ← PyQt GUI panels
│   └── utils.py           ← Shared math + export utilities
├── examples/              ← Worked example scripts
├── docs/                  ← Documentation
├── tests/                 ← pytest test suite
├── environment.yml        ← Conda environment
├── setup.py
└── pyproject.toml
```

---

## Design philosophy

> Prioritise robustness, usability, and maintainability over the number of features.

- **Never modifies raw data** — everything happens in memory.
- **Lazy loading** — large TIFF images are memory-mapped; only requested tiles are read into RAM.
- **Modular** — each module is independently importable and testable.
- **Reproducible** — all analysis results can be exported to CSV, H5AD, or figures.

---

## Citing UnumLocalia

If you use UnumLocalia in your research, please cite:

> *UnumLocalia: an open-source toolkit for multimodal spatial biology benchmarking.*
> (manuscript in preparation)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
