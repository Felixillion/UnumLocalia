# UnumLocalia Downstream Analysis

This directory contains optional downstream analysis workflows for data exported from UnumLocalia.

These scripts are intentionally separated from the main UnumLocalia application so that the viewer and quantification tools remain lightweight and easy to install.

---

## Clustering Workflow

`clustering.py` performs unsupervised clustering of quantified cells using:

- Xenium transcript counts
- COMET protein intensities
- Combined transcript + protein features

The workflow generates dimensionality reductions, Leiden clusters, marker analyses, and comparisons between clustering modalities.

---

## Input Files

The workflow expects:

### Cell Quantification CSV

Generated from:

```text
UnumLocalia
→ Cell Quantification
→ Export Quantification CSV
```

Example:

```text
core01_hcc_quantification_cells.csv
```

Only the core identifier needs to be specified in `clustering.py`:

```python
CORE_NAME = "core01"
```

### Protein Threshold JSON (optional)

Generated from:

```text
UnumLocalia
→ Cell Quantification
→ Export Thresholds JSON
```

Example:

```text
core01_thresholds.json
```

If supplied, protein measurements below the threshold are set to zero prior to analysis.

---

## Installation

The recommended installation method uses Mamba, which is substantially faster than Conda for resolving scientific Python dependencies.

If Mamba is not already installed:

```bash
conda install -n base -c conda-forge mamba
```

Create the analysis environment:

```bash
mamba env create -f environment.yml
```

Activate the environment:

```bash
mamba activate unumlocalia_analysis
```

The environment definition is provided in `environment.yml` and includes all packages required for clustering, dimensionality reduction, visualisation, marker analysis, and cluster comparison.

---

## Configuration

Open:

```python
clustering.py
```

and modify the user settings near the top of the file:

```python
CORE_NAME = "core01"
```

The workflow automatically searches the current directory for:

```text
core01*_quantification_cells.csv
core01*_thresholds.json
```

and loads the matching files.

The remaining parameters control quality filtering, dimensionality reduction, neighborhood construction, and clustering resolution.

---

## Quality Control

The workflow applies default transcript-based quality control:

```python
MIN_GENES = 5
MIN_TRANSCRIPTS = 20
```

Cells failing either threshold are excluded from transcript-based analyses.

Protein analyses additionally require at least one non-zero protein measurement.

Filtered cell tables are saved in:

```text
gene/cells_used.csv
protein/cells_used.csv
combined/cells_used.csv
```

## Notes

- Xenium control probes and non-biological features (`NegControlProbe`, `NegControlCodeword`, `DeprecatedCodeword`, `Intergenic`, and related features) are excluded during cell quantification export.
- DAPI is excluded from protein clustering by default.
- Protein thresholds exported from UnumLocalia can be applied automatically.
- All clustering analyses are deterministic when using the default random seed.
- Generated `.h5ad` files can be loaded directly into Scanpy for additional downstream analysis.

---

## Running the Workflow

```bash
python clustering.py
```

---

## Analysis Performed

### Gene Analysis

The workflow:

1. Filters low-quality cells
2. Normalizes transcript counts
3. Identifies highly variable genes
4. Performs PCA
5. Computes UMAP and PaCMAP embeddings
6. Runs Leiden clustering
7. Identifies cluster marker genes

Outputs:

```text
results_coreXX/gene/
```

---

### Protein Analysis

The workflow:

1. Applies optional thresholds
2. Performs arcsinh transformation
3. Z-score normalizes proteins
4. Performs PCA
5. Computes UMAP and PaCMAP embeddings
6. Runs Leiden clustering

Outputs:

```text
results_coreXX/protein/
```

---

### Combined Analysis

The workflow combines:

```text
Gene PCA components
+
Protein expression features
```

to generate integrated clusters.

Outputs:

```text
results_coreXX/combined/
```

---

### Cluster Comparison

The workflow compares:

- Gene clusters
- Protein clusters
- Combined clusters

using:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Contingency heatmaps
- Cluster agreement maps
- Sankey diagrams

Outputs:

```text
results_coreXX/comparison/
```

---

## Output Structure

```text
results_coreXX/

├── gene/
│   ├── gene_analysis.h5ad
│   ├── cluster_assignments.csv
│   ├── cluster_sizes.csv
│   ├── umap_clusters.png
│   ├── pacmap_clusters.png
│   ├── spatial_clusters.png
│   ├── gene_heatmap_per_cluster.png
│   └── dotplot_cluster_markers.png
│
├── protein/
│   ├── protein_analysis.h5ad
│   ├── cluster_assignments.csv
│   ├── cluster_sizes.csv
│   ├── umap_clusters.png
│   ├── pacmap_clusters.png
│   ├── spatial_clusters.png
│   └── protein_heatmap_per_cluster.png
│
├── combined/
│   ├── combined_analysis.h5ad
│   ├── cluster_assignments.csv
│   ├── cluster_sizes.csv
│   ├── umap_combined.png
│   ├── pacmap_combined.png
│   ├── spatial_combined.png
│   ├── heatmap_combined_gene_markers.png
│   └── dotplot_combined_gene_markers.png
│
└── comparison/
    ├── ARI_NMI_scores.csv
    ├── heatmaps/
    ├── mapped/
    ├── merged/
    └── sankey/
```

---

## Relationship to UnumLocalia

UnumLocalia itself focuses on:

- Data visualisation
- Transcript visualisation
- Protein visualisation
- Segmentation import
- Cell quantification
- Session management

The analysis scripts in this directory provide optional downstream processing of exported quantification results.
