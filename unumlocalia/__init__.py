"""
UnumLocalia
============
A multimodal spatial biology toolkit for visualisation and single-cell analysis
of paired Xenium, COMET, and H&E datasets.

Version 1.0 — initial release.
"""

from importlib.metadata import PackageNotFoundError, version as _version

__author__ = "UnumLocalia Contributors"
__email__ = ""
__license__ = "MIT"

try:
    __version__ = _version("unumlocalia")
except PackageNotFoundError:
    __version__ = "1.0.0-dev"

# Public API re-exports
from unumlocalia.io import DatasetLoader, DatasetManifest, detect_files  # noqa: F401
from unumlocalia.viewer import SpatialViewer  # noqa: F401
from unumlocalia.analysis import (  # noqa: F401
    prepare_modality,
    run_pca,
    run_neighbors,
    run_umap,
    run_leiden,
    plot_umap,
    plot_heatmap,
    plot_dotplot,
    plot_spatial_clusters,
    measure_comet_intensities,
    assign_xenium_transcripts,
    build_anndata,
)

__all__ = [
    # io
    "DatasetLoader",
    "DatasetManifest",
    "detect_files",
    # viewer
    "SpatialViewer",
    # analysis
    "prepare_modality",
    "run_pca",
    "run_neighbors",
    "run_umap",
    "run_leiden",
    "plot_umap",
    "plot_heatmap",
    "plot_dotplot",
    "plot_spatial_clusters",
    "measure_comet_intensities",
    "assign_xenium_transcripts",
    "build_anndata",
]
