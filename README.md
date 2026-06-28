# SpatialBench

**SpatialBench** is an open-source Python toolkit accompanying multimodal spatial biology datasets.
It provides a simple, reproducible environment for visualisation, exploration,
segmentation benchmarking, and basic single-cell analysis — all without modifying
your original data.

---

## Features

| Module | Functionality |
|---|---|
| **Viewer** | Interactive napari viewer for H&E, COMET, Xenium transcripts, and cell/nucleus boundaries |
| **Segmentation** | Load label masks, GeoJSON, or QuPath objects; measure COMET intensities; assign transcripts |
| **Analysis** | PCA, UMAP, Leiden clustering, heatmaps, dot plots, spatial plots via Scanpy |
| **Benchmark** | ARI, NMI, contingency heatmaps, Sankey diagrams, per-marker scatter + correlations |
| **Export** | Figures (PNG, SVG, PDF), tables (CSV), AnnData (H5AD) |

---

## Supported data modalities

SpatialBench expects a dataset folder containing:

```
dataset/
    cells.csv                       ← Xenium cell metadata
    cell_boundaries.parquet         ← Xenium cell polygons
    nucleus_boundaries.parquet      ← Xenium nucleus polygons
    transcripts.parquet             ← Xenium transcript coordinates + gene names
    matrix.csv                      ← 3×3 affine alignment matrix
    he.tif                          ← Aligned H&E whole-slide image
    comet/
        CK8.ome.tiff
        CD45.ome.tiff
        ...                         ← One OME-TIFF per protein marker
    anndata.h5ad                    ← Reference integrated AnnData
```

All files are auto-detected — no manual configuration required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SpatialBench/SpatialBench.git
cd SpatialBench
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate spatialbench
```

### 3. Verify the installation

```bash
spatialbench --version   # prints 1.0.0
```

---

## Quick start

### Launch the GUI

```bash
conda activate spatialbench
spatialbench
```

A napari window will open with the SpatialBench panel docked on the right.
Go to the **Settings** tab, select your dataset folder, and click **Load dataset**.

Alternatively, launch from Python:

```python
from spatialbench.widgets import launch
launch()
```

### Programmatic use

```python
from spatialbench import DatasetLoader

# Load dataset
loader = DatasetLoader("/path/to/dataset")
loader.load()

# Inspect
print(loader.manifest.summary())
print(f"Genes: {len(loader.genes)}   Proteins: {len(loader.proteins)}")

# Run analysis
from spatialbench import run_pca, run_umap, run_leiden, plot_umap

adata = loader.anndata_ref
run_pca(adata, modality="genes")
run_umap(adata, modality="genes")
run_leiden(adata, modality="genes", resolution=0.5)

fig = plot_umap(adata, color="leiden_genes", modality="genes")
fig.savefig("umap_clusters.png", dpi=150, bbox_inches="tight")
```

### Benchmark a custom segmentation

```python
from spatialbench import (
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
from spatialbench import compare_clusterings, plot_contingency
results = compare_clusterings(
    adata_original.obs["leiden_genes"],
    adata_user.obs["leiden_genes"],
)
print(f"ARI: {results['ari']:.4f}   NMI: {results['nmi']:.4f}")
```

---

## Repository structure

```
SpatialBench/
├── spatialbench/
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

## Citing SpatialBench

If you use SpatialBench in your research, please cite:

> *SpatialBench: an open-source toolkit for multimodal spatial biology benchmarking.*
> (manuscript in preparation)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
