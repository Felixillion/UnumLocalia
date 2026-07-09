# Installation

UnumLocalia runs locally to handle very large images and datasets. It is written in Python and is highly cross-platform.

## Requirements
* Python 3.9, 3.10, or 3.11.
* Conda or Miniconda is highly recommended for dependency management.

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Felixillion/UnumLocalia.git
   cd UnumLocalia
   ```

2. **Create the environment:**
   We provide an `environment.yml` that pins all dependencies including `napari`, `scanpy`, and UI frameworks.
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate and Verify:**
   ```bash
   conda activate unumlocalia
   unumlocalia --version
   ```

## Starting UnumLocalia
You can start the GUI from the terminal:
```bash
conda activate unumlocalia
unumlocalia
```

This will open a Napari window with UnumLocalia docked on the side. Navigate to the **Settings** tab to load your dataset folder.
