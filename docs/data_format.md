# Dataset Format

UnumLocalia uses an automated directory loader. You do not need to manually select individual files; instead, select the **root directory** of your dataset.

UnumLocalia will recursively search (up to two levels deep) to discover all relevant modalities.

## Required and Optional Files
The auto-detection looks for specific file names and extensions:

### 1. Xenium Data
* `cells.csv`: Cell metadata.
* `cell_boundaries.parquet`: Cell boundaries (polygons).
* `nucleus_boundaries.parquet`: Nucleus boundaries (polygons).
* `transcripts.parquet`: Transcript coordinates and names.

### 2. COMET Data (Proteins)
* `*.ome.tif` or `*.ome.tiff`: Multiplexed immunofluorescence images. 
* UnumLocalia uses the filename (minus the extension) as the protein marker name (e.g., `CD45.ome.tiff` becomes `CD45`).
* *Tip: It helps to place these inside a folder named `comet/` or `proteins/` to help the auto-detector prioritise them.*

### 3. H&E Image
* `*.tif` or `*.tiff`: The aligned H&E whole-slide image. If there are multiple TIFFs, UnumLocalia assumes the largest one is the H&E slide.

### 4. Alignment
* `matrix.csv`: A 3x3 affine transformation matrix (9 values). This is used to align COMET coordinates with Xenium coordinates.

### 5. AnnData Reference
* `*.h5ad`: An integrated `anndata` object acting as a read-only baseline for single-cell clustering.

## Example Directory Structure
```
my_dataset/
├── cells.csv
├── cell_boundaries.parquet
├── nucleus_boundaries.parquet
├── transcripts.parquet
├── matrix.csv
├── tissue_HE.tif
├── anndata.h5ad
└── comet/
    ├── CD45.ome.tiff
    ├── CK8.ome.tiff
    └── FOXP3.ome.tiff
```
