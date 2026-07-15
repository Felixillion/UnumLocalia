"""
UnumLocalia
============
A multimodal spatial biology toolkit for visualisation and single-cell analysis
of paired Xenium, COMET, and H&E datasets.

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

__all__ = [
    # io
    "DatasetLoader",
    "DatasetManifest",
    "detect_files",
    # viewer
    "SpatialViewer",
]
