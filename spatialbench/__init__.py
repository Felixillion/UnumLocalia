"""
SpatialBench
============
A multimodal spatial biology toolkit for visualisation, segmentation
benchmarking, and single-cell analysis of paired Xenium, COMET, and H&E
datasets.

Version 1.0 — initial release.
"""

from importlib.metadata import PackageNotFoundError, version as _version

__author__ = "SpatialBench Contributors"
__email__ = ""
__license__ = "MIT"

try:
    __version__ = _version("spatialbench")
except PackageNotFoundError:
    __version__ = "1.0.0-dev"

# Public API re-exports
from spatialbench.io import DatasetLoader, DatasetManifest, detect_files  # noqa: F401
from spatialbench.viewer import SpatialViewer  # noqa: F401
from spatialbench.segmentation import (  # noqa: F401
    load_segmentation,
    measure_comet_intensities,
    assign_xenium_transcripts,
    build_anndata,
    compare_segmentations,
)
from spatialbench.analysis import (  # noqa: F401
    prepare_modality,
    run_pca,
    run_neighbors,
    run_umap,
    run_leiden,
    plot_umap,
    plot_heatmap,
    plot_dotplot,
    plot_spatial_clusters,
)
from spatialbench.benchmark import (  # noqa: F401
    compare_clusterings,
    compare_markers,
    plot_contingency,
    plot_sankey,
    plot_marker_scatter,
)

__all__ = [
    # io
    "DatasetLoader",
    "DatasetManifest",
    "detect_files",
    # viewer
    "SpatialViewer",
    # segmentation
    "load_segmentation",
    "measure_comet_intensities",
    "assign_xenium_transcripts",
    "build_anndata",
    "compare_segmentations",
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
    # benchmark
    "compare_clusterings",
    "compare_markers",
    "plot_contingency",
    "plot_sankey",
    "plot_marker_scatter",
]
