# Installation

SpatialBench runs locally to handle very large images and datasets. It is written in Python and is highly cross-platform.

## Requirements
* Python 3.9, 3.10, or 3.11.
* Conda or Miniconda is highly recommended for dependency management.

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Felixillion/SpatialBench.git
   cd SpatialBench
   ```

2. **Create the environment:**
   We provide an `environment.yml` that pins all dependencies including `napari`, `scanpy`, and UI frameworks.
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate and Verify:**
   ```bash
   conda activate spatialbench
   spatialbench --version
   ```

## Starting SpatialBench
You can start the GUI from the terminal:
```bash
conda activate spatialbench
spatialbench
```

This will open a Napari window with SpatialBench docked on the side. Navigate to the **Settings** tab to load your dataset folder.
