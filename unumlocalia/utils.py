"""
unumlocalia.utils
==================
Shared utility functions used across all UnumLocalia modules.

All functions are pure/stateless and operate without modifying any input data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; widgets switch to Qt canvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def affine_transform_coords(
    coords: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply a 3×3 affine transformation matrix to 2-D coordinates.

    Parameters
    ----------
    coords:
        Array of shape ``(N, 2)`` with columns ``[x, y]``.
    matrix:
        3×3 affine matrix (row-major, homogeneous coordinates).

    Returns
    -------
    np.ndarray
        Transformed coordinates, shape ``(N, 2)``.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {coords.shape}")
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must be (3, 3), got {matrix.shape}")

    # Homogeneous coordinates: append column of ones
    n = coords.shape[0]
    hom = np.ones((n, 3), dtype=np.float64)
    hom[:, :2] = coords

    transformed = (matrix @ hom.T).T  # (3, N) -> (N, 3)
    return transformed[:, :2]


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a 3×3 affine matrix.

    Parameters
    ----------
    matrix:
        3×3 affine matrix.

    Returns
    -------
    np.ndarray
        Inverse matrix, shape (3, 3).
    """
    return np.linalg.inv(matrix)


# ---------------------------------------------------------------------------
# Normalisation / transform
# ---------------------------------------------------------------------------

def arcsinh_transform(x: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    """Arcsinh transform commonly used for cytometry / COMET protein data.

    Parameters
    ----------
    x:
        Raw intensity values.
    cofactor:
        Scaling factor (default 5, standard for CyTOF; use 150 for flow).

    Returns
    -------
    np.ndarray
        Transformed values.
    """
    return np.arcsinh(x / cofactor)


def log1p_norm(
    counts: np.ndarray,
    target_sum: float = 1e4,
) -> np.ndarray:
    """Per-cell library-size normalisation followed by log1p.

    Parameters
    ----------
    counts:
        Dense count matrix, shape ``(cells, genes)``.
    target_sum:
        Each cell is scaled so its total equals this value before log1p.

    Returns
    -------
    np.ndarray
        Normalised matrix.
    """
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero for empty cells
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normed = counts / row_sums * target_sum
    return np.log1p(normed)


# ---------------------------------------------------------------------------
# Napari helpers
# ---------------------------------------------------------------------------

def shapes_to_napari(
    boundaries_df: pd.DataFrame,
    x_col: str = "vertex_x",
    y_col: str = "vertex_y",
    cell_col: str = "cell_id",
) -> List[np.ndarray]:
    """Convert a boundaries parquet DataFrame to a list of polygon arrays
    suitable for a napari ``Shapes`` layer.

    Napari Shapes expects shapes as a list of ``(N, 2)`` arrays in
    ``[row, col]`` i.e. ``[y, x]`` order.

    Parameters
    ----------
    boundaries_df:
        DataFrame with one row per vertex, columns including x, y and cell ID.
    x_col, y_col:
        Column names for x and y coordinates.
    cell_col:
        Column name identifying which cell each vertex belongs to.

    Returns
    -------
    list of np.ndarray
        One array per cell polygon, each shape ``(N, 2)`` in [y, x] order.
    """
    shapes = []
    for _, group in boundaries_df.groupby(cell_col, sort=False):
        xy = group[[y_col, x_col]].to_numpy(dtype=np.float32)
        shapes.append(xy)
    return shapes


def label_mask_to_shapes(labels: np.ndarray) -> List[np.ndarray]:
    """Extract polygon contours from a label mask for use in a napari
    ``Shapes`` layer.

    Parameters
    ----------
    labels:
        2-D integer label mask; 0 = background, N > 0 = cell N.

    Returns
    -------
    list of np.ndarray
        One polygon per unique label, each shape ``(N, 2)`` in [row, col]
        (i.e. [y, x]) order.
    """
    try:
        from skimage.measure import find_contours
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for label_mask_to_shapes. "
            "Install it with: conda install scikit-image"
        ) from exc

    shapes = []
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        mask = (labels == lbl).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if contours:
            # Take the longest contour if there are multiple
            poly = max(contours, key=len)
            shapes.append(poly.astype(np.float32))
    return shapes


def colormap_from_name(name: str) -> str:
    """Return a napari-compatible colormap name.

    Falls back to ``'green'`` if the requested name is not recognised.

    Parameters
    ----------
    name:
        Colormap name string (e.g. ``'red'``, ``'cyan'``, ``'magma'``).

    Returns
    -------
    str
        Validated colormap name.
    """
    _NAPARI_COLORMAPS = {
        "red", "green", "blue", "cyan", "magenta", "yellow",
        "gray", "grays", "hot", "cool", "magma", "inferno",
        "plasma", "viridis", "turbo", "hsv",
    }
    if name.lower() in _NAPARI_COLORMAPS:
        return name.lower()
    logger.warning("Unknown colormap '%s'; falling back to 'green'.", name)
    return "green"


# ---------------------------------------------------------------------------
# Figure export
# ---------------------------------------------------------------------------

def export_figure(
    fig: "matplotlib.figure.Figure",
    path: Union[str, Path],
    fmt: str = "png",
    dpi: int = 150,
) -> Path:
    """Save a matplotlib Figure to disk.

    Parameters
    ----------
    fig:
        The matplotlib figure to export.
    path:
        Destination file path (extension will be replaced by *fmt* if needed).
    fmt:
        Output format: ``'png'``, ``'svg'``, or ``'pdf'``.
    dpi:
        Resolution for raster formats (ignored for SVG).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(path)
    fmt = fmt.lower().lstrip(".")
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError(f"Unsupported format '{fmt}'. Use png, svg, or pdf.")

    # Ensure the file extension matches the format
    save_path = path.with_suffix(f".{fmt}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path,
        format=fmt,
        dpi=dpi if fmt == "png" else None,
        bbox_inches="tight",
    )
    logger.info("Figure saved to %s", save_path)
    return save_path


def export_dataframe(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Save a DataFrame to CSV.

    Parameters
    ----------
    df:
        DataFrame to export.
    path:
        Destination file path.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Table saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def safe_read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a CSV file with informative error messages.

    Parameters
    ----------
    path:
        Path to the CSV file.
    **kwargs:
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def safe_read_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a Parquet file with informative error messages.

    Parameters
    ----------
    path:
        Path to the Parquet file.
    **kwargs:
        Additional keyword arguments forwarded to ``pd.read_parquet``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, **kwargs)


def centroid_from_bounds(
    boundaries_df: pd.DataFrame,
    x_col: str = "vertex_x",
    y_col: str = "vertex_y",
    cell_col: str = "cell_id",
) -> pd.DataFrame:
    """Compute polygon centroids from a boundaries DataFrame.

    Parameters
    ----------
    boundaries_df:
        Vertex-per-row DataFrame.
    x_col, y_col:
        Coordinate column names.
    cell_col:
        Cell identifier column.

    Returns
    -------
    pd.DataFrame
        Columns: ``cell_id``, ``centroid_x``, ``centroid_y``.
    """
    grp = boundaries_df.groupby(cell_col)
    cx = grp[x_col].mean().rename("centroid_x")
    cy = grp[y_col].mean().rename("centroid_y")
    return pd.concat([cx, cy], axis=1).reset_index()
