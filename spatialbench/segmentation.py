"""
spatialbench.segmentation
=========================
Loading user-supplied segmentations and deriving new AnnData objects.

Supported input formats
-----------------------
* **Label mask** — 2-D integer TIFF / PNG where pixel value = cell ID.
* **GeoJSON** — FeatureCollection of Polygon features (e.g. from QuPath export).
* **QuPath objects JSON** — ``objects.json`` exported from a QuPath project.

All segmentations are returned as a 2-D ``np.ndarray`` label mask so that
downstream functions can operate on a single unified representation.

Key outputs
-----------
* Per-cell COMET protein intensities (mean intensity within each label).
* Per-cell Xenium transcript counts (point-in-polygon assignment).
* A fresh ``anndata.AnnData`` object combining both data types.

Nothing written to disk unless the user explicitly requests an export.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation loading
# ---------------------------------------------------------------------------

def load_segmentation(
    path: Union[str, Path],
    shape: Optional[Tuple[int, int]] = None,
    invert_y: bool = False,
) -> np.ndarray:
    """Load a segmentation from file and return a 2-D integer label mask.

    Supported formats are detected from the file extension.

    Parameters
    ----------
    path:
        Path to the segmentation file.
    shape:
        ``(height, width)`` of the output mask.  Required when loading from
        GeoJSON or QuPath objects (rasterisation needs a canvas size).  If
        ``None``, inferred from the image extent.
    invert_y:
        Flip the y-axis after rasterisation (useful when GeoJSON/QuPath
        coordinates have a bottom-left origin).

    Returns
    -------
    np.ndarray
        2-D integer array, shape ``(H, W)``, dtype ``int32``.
        Background = 0, cell N = N.

    Raises
    ------
    ValueError
        If the file format is not recognised.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {path}")

    suffix = path.suffix.lower()
    stem_lower = path.stem.lower()

    if suffix in (".tif", ".tiff", ".png"):
        return _load_label_tiff(path)

    if suffix == ".geojson" or (suffix == ".json" and "geo" in stem_lower):
        return _load_geojson(path, shape=shape, invert_y=invert_y)

    if suffix == ".json":
        # Assume QuPath objects.json
        return _load_qupath_objects(path, shape=shape, invert_y=invert_y)

    raise ValueError(
        f"Unsupported segmentation format '{suffix}'. "
        "Supported: .tif/.tiff/.png (label mask), .geojson, .json (QuPath)."
    )


def _load_label_tiff(path: Path) -> np.ndarray:
    """Load a 2-D label mask from a TIFF or PNG file."""
    import tifffile

    arr = tifffile.imread(path).squeeze()
    if arr.ndim != 2:
        raise ValueError(
            f"Label mask must be 2-D after squeezing, got shape {arr.shape}"
        )
    return arr.astype(np.int32)


def _load_geojson(
    path: Path,
    shape: Optional[Tuple[int, int]],
    invert_y: bool,
) -> np.ndarray:
    """Rasterise a GeoJSON FeatureCollection of polygon annotations."""
    with open(path) as fh:
        data = json.load(fh)

    features = data.get("features", [])
    if not features:
        raise ValueError(f"No features found in GeoJSON file: {path}")

    polygons = []
    for feat in features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]  # outer ring
            polygons.append(np.array(coords, dtype=np.float64)[:, :2])
        elif geom.get("type") == "MultiPolygon":
            for ring_group in geom["coordinates"]:
                polygons.append(np.array(ring_group[0], dtype=np.float64)[:, :2])

    if shape is None:
        all_pts = np.vstack(polygons)
        shape = (
            int(np.ceil(all_pts[:, 1].max())) + 1,
            int(np.ceil(all_pts[:, 0].max())) + 1,
        )

    return _rasterise_polygons(polygons, shape, invert_y=invert_y)


def _load_qupath_objects(
    path: Path,
    shape: Optional[Tuple[int, int]],
    invert_y: bool,
) -> np.ndarray:
    """Rasterise a QuPath ``objects.json`` export.

    QuPath exports either a GeoJSON-style FeatureCollection or a list of
    annotation objects, each with a ``roi`` or ``geometry`` key.
    """
    with open(path) as fh:
        data = json.load(fh)

    # Try FeatureCollection first
    if isinstance(data, dict) and "features" in data:
        return _load_geojson(path, shape=shape, invert_y=invert_y)

    # Handle list of objects
    objects = data if isinstance(data, list) else data.get("objects", [])
    polygons = []
    for obj in objects:
        roi = obj.get("roi", obj.get("geometry", {}))
        if not roi:
            continue
        roi_type = roi.get("type", "")
        coords_list = roi.get("coordinates", roi.get("points", []))
        if roi_type == "Polygon" and coords_list:
            polygons.append(np.array(coords_list[0], dtype=np.float64)[:, :2])
        elif roi_type == "MultiPolygon":
            for ring_group in coords_list:
                polygons.append(np.array(ring_group[0], dtype=np.float64)[:, :2])

    if not polygons:
        raise ValueError(f"Could not extract any polygons from QuPath file: {path}")

    if shape is None:
        all_pts = np.vstack(polygons)
        shape = (
            int(np.ceil(all_pts[:, 1].max())) + 1,
            int(np.ceil(all_pts[:, 0].max())) + 1,
        )

    return _rasterise_polygons(polygons, shape, invert_y=invert_y)


def _rasterise_polygons(
    polygons: List[np.ndarray],
    shape: Tuple[int, int],
    invert_y: bool = False,
) -> np.ndarray:
    """Draw filled polygons into an integer label mask.

    Parameters
    ----------
    polygons:
        List of ``(N, 2)`` arrays with columns ``[x, y]`` (image space).
    shape:
        ``(H, W)`` of the output canvas.
    invert_y:
        Flip y-axis (for coordinate systems with bottom-left origin).

    Returns
    -------
    np.ndarray
        Label mask, dtype ``int32``.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required to rasterise polygons. "
            "Install with: conda install opencv"
        ) from exc

    mask = np.zeros(shape, dtype=np.int32)
    H = shape[0]

    for cell_id, poly in enumerate(polygons, start=1):
        pts = poly[:, :2].astype(np.float32)
        if invert_y:
            pts[:, 1] = H - pts[:, 1]
        # Convert [x, y] → [col, row] for OpenCV
        pts_rc = pts[:, [0, 1]].reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts_rc], color=cell_id)

    logger.info(
        "Rasterised %d polygons into label mask, shape=%s", len(polygons), shape
    )
    return mask


# ---------------------------------------------------------------------------
# Intensity measurement
# ---------------------------------------------------------------------------

def measure_comet_intensities(
    labels: np.ndarray,
    comet_arrays: Dict[str, np.ndarray],
    stat: str = "mean",
) -> pd.DataFrame:
    """Measure per-cell COMET protein intensities from a label mask.

    Parameters
    ----------
    labels:
        2-D integer label mask.
    comet_arrays:
        ``{marker_name: 2-D image array}`` dict as returned by
        :func:`spatialbench.io.get_comet_arrays`.
    stat:
        Summary statistic: ``'mean'``, ``'median'``, or ``'sum'``.

    Returns
    -------
    pd.DataFrame
        One row per cell, columns: ``cell_id``, one column per marker.

    Notes
    -----
    Only cells present in both *labels* and each COMET image are included.
    Background (label 0) is excluded.
    """
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        logger.warning("No cell labels found in label mask.")
        return pd.DataFrame()

    stat_fn = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
    }.get(stat)
    if stat_fn is None:
        raise ValueError(f"Unsupported stat '{stat}'. Use 'mean', 'median', or 'sum'.")

    records: List[Dict] = []

    for cell_id in unique_ids:
        cell_mask = labels == cell_id
        row: Dict = {"cell_id": int(cell_id)}
        for marker, arr in comet_arrays.items():
            # Align shapes if COMET image is different size to label mask
            if arr.shape != labels.shape:
                arr_aligned = _resize_array(arr, labels.shape)
            else:
                arr_aligned = arr
            pixel_vals = arr_aligned[cell_mask].astype(np.float64)
            row[marker] = float(stat_fn(pixel_vals)) if len(pixel_vals) > 0 else np.nan
        records.append(row)

    df = pd.DataFrame(records)
    logger.info(
        "Measured COMET intensities: %d cells × %d markers",
        len(df), len(comet_arrays),
    )
    return df


def _resize_array(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array to *target_shape* using nearest-neighbour resampling."""
    try:
        import cv2
        return cv2.resize(
            arr.astype(np.float32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    except ImportError:
        # Fallback: use numpy index scaling
        h_scale = target_shape[0] / arr.shape[0]
        w_scale = target_shape[1] / arr.shape[1]
        rows = (np.arange(target_shape[0]) / h_scale).astype(int)
        cols = (np.arange(target_shape[1]) / w_scale).astype(int)
        return arr[np.ix_(rows, cols)]


# ---------------------------------------------------------------------------
# Transcript assignment
# ---------------------------------------------------------------------------

def assign_xenium_transcripts(
    labels: np.ndarray,
    transcripts_df: pd.DataFrame,
    x_col: str = "x_location",
    y_col: str = "y_location",
    gene_col: str = "feature_name",
    pixel_size: float = 1.0,
) -> pd.DataFrame:
    """Assign Xenium transcripts to cells in a label mask.

    Each transcript is assigned to the cell whose label overlies its
    pixel coordinate (fast raster lookup, no polygon operations needed).

    Parameters
    ----------
    labels:
        2-D integer label mask.
    transcripts_df:
        Transcript DataFrame with x/y coordinates and gene names.
    x_col, y_col:
        Column names for transcript coordinates (in the same space as the
        label mask).
    gene_col:
        Column name for gene/feature names.
    pixel_size:
        Scaling factor from transcript coordinate units to label mask pixels
        (e.g. if transcripts are in µm and the label mask is in µm/pixel).

    Returns
    -------
    pd.DataFrame
        Transcripts with an additional ``cell_id`` column (0 = unassigned).
    """
    df = transcripts_df.copy()

    # Convert coordinates to integer pixel indices
    col_idx = (df[x_col].to_numpy() / pixel_size).astype(int)
    row_idx = (df[y_col].to_numpy() / pixel_size).astype(int)

    H, W = labels.shape

    # Clamp to valid range
    row_idx = np.clip(row_idx, 0, H - 1)
    col_idx = np.clip(col_idx, 0, W - 1)

    df["cell_id"] = labels[row_idx, col_idx].astype(int)

    n_assigned = (df["cell_id"] > 0).sum()
    logger.info(
        "Assigned %d / %d transcripts to cells (%d unique cells)",
        n_assigned, len(df),
        df.loc[df["cell_id"] > 0, "cell_id"].nunique(),
    )
    return df


def pivot_transcript_counts(
    assigned_df: pd.DataFrame,
    gene_col: str = "feature_name",
    cell_col: str = "cell_id",
) -> pd.DataFrame:
    """Pivot assigned transcripts to a cell × gene count matrix.

    Parameters
    ----------
    assigned_df:
        Output of :func:`assign_xenium_transcripts`.
    gene_col:
        Column containing gene names.
    cell_col:
        Column containing cell IDs.

    Returns
    -------
    pd.DataFrame
        Index = cell_id, columns = gene names, values = integer counts.
        Background (cell_id == 0) is excluded.
    """
    df = assigned_df[assigned_df[cell_col] > 0]
    counts = (
        df.groupby([cell_col, gene_col])
        .size()
        .unstack(fill_value=0)
    )
    logger.info(
        "Transcript count matrix: %d cells × %d genes", *counts.shape
    )
    return counts


# ---------------------------------------------------------------------------
# AnnData construction
# ---------------------------------------------------------------------------

def build_anndata(
    labels: np.ndarray,
    comet_intensities: Optional[pd.DataFrame] = None,
    transcripts_df: Optional[pd.DataFrame] = None,
    cells_metadata: Optional[pd.DataFrame] = None,
    gene_col: str = "feature_name",
    cell_col: str = "cell_id",
    x_centroid_col: str = "x_centroid",
    y_centroid_col: str = "y_centroid",
) -> "anndata.AnnData":
    """Build a fresh AnnData object from a user-supplied segmentation.

    Parameters
    ----------
    labels:
        2-D integer label mask (used to enumerate cells).
    comet_intensities:
        Per-cell protein intensities (from :func:`measure_comet_intensities`).
        Must have a ``cell_id`` column.
    transcripts_df:
        Assigned transcript DataFrame (from :func:`assign_xenium_transcripts`).
        Must have ``cell_id`` and gene name columns.
    cells_metadata:
        Optional metadata DataFrame with additional per-cell columns
        (e.g. area, centroid coordinates).
    gene_col:
        Column name for gene names in *transcripts_df*.
    cell_col:
        Column name for cell identifiers (must match across all DataFrames).
    x_centroid_col, y_centroid_col:
        Column names for spatial coordinates in *cells_metadata*.

    Returns
    -------
    anndata.AnnData
        AnnData with:
        - ``.X`` — gene count matrix (cells × genes).
        - ``.obsm['X_protein']`` — protein intensity matrix.
        - ``.obs`` — cell metadata (area, centroid, etc.).
        - ``.obsm['spatial']`` — spatial coordinates array ``(N, 2)``.
        - ``.layers['raw_counts']`` — copy of raw integer gene counts.
        - ``.uns['source']`` — ``'user_segmentation'``.

    Raises
    ------
    ImportError
        If ``anndata`` is not installed.
    ValueError
        If no cells can be derived from *labels*.
    """
    try:
        import anndata
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "anndata and scipy are required to build AnnData objects. "
            "Install with: conda install anndata scipy"
        ) from exc

    # Determine cell IDs from label mask
    cell_ids = np.unique(labels)
    cell_ids = sorted(int(c) for c in cell_ids if c != 0)

    if not cell_ids:
        raise ValueError("Label mask contains no cell labels (all background).")

    # ---- Gene count matrix ------------------------------------------------
    if transcripts_df is not None:
        gene_counts = pivot_transcript_counts(
            transcripts_df, gene_col=gene_col, cell_col=cell_col
        )
        # Reindex to all cells in label mask (fill missing with 0)
        gene_counts = gene_counts.reindex(cell_ids, fill_value=0)
        X = sp.csr_matrix(gene_counts.to_numpy(dtype=np.float32))
        var = pd.DataFrame(index=gene_counts.columns.astype(str))
    else:
        X = sp.csr_matrix((len(cell_ids), 0), dtype=np.float32)
        var = pd.DataFrame()

    # ---- Observation metadata ---------------------------------------------
    obs = pd.DataFrame(index=[str(c) for c in cell_ids])
    obs.index.name = "cell_id"

    if cells_metadata is not None and cell_col in cells_metadata.columns:
        meta_indexed = cells_metadata.set_index(cell_col)
        meta_indexed.index = meta_indexed.index.astype(str)
        obs = obs.join(meta_indexed, how="left")

    # ---- Build AnnData ----------------------------------------------------
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Raw counts layer
    adata.layers["raw_counts"] = X.copy()

    # ---- Protein layer ----------------------------------------------------
    if comet_intensities is not None and not comet_intensities.empty:
        prot_indexed = comet_intensities.set_index(cell_col)
        prot_indexed.index = prot_indexed.index.astype(str)
        prot_aligned = prot_indexed.reindex(
            [str(c) for c in cell_ids], fill_value=np.nan
        )
        adata.obsm["X_protein"] = prot_aligned.to_numpy(dtype=np.float32)
        adata.uns["protein_names"] = list(prot_aligned.columns)

    # ---- Spatial coordinates ---------------------------------------------
    if (
        cells_metadata is not None
        and x_centroid_col in cells_metadata.columns
        and y_centroid_col in cells_metadata.columns
    ):
        coord_df = (
            cells_metadata.set_index(cell_col)[[x_centroid_col, y_centroid_col]]
            .reindex(cell_ids)
        )
        adata.obsm["spatial"] = coord_df.to_numpy(dtype=np.float32)

    # ---- Compute cell areas from label mask if not in metadata -----------
    if "cell_area" not in adata.obs.columns:
        area_dict = _compute_label_areas(labels)
        adata.obs["cell_area"] = [
            area_dict.get(int(c), np.nan)
            for c in adata.obs.index.astype(int, errors="ignore")
        ]

    adata.uns["source"] = "user_segmentation"
    logger.info(
        "Built AnnData: %d cells × %d genes, %d protein channels",
        adata.n_obs, adata.n_vars,
        adata.obsm.get("X_protein", np.empty((0, 0))).shape[1],
    )
    return adata


def _compute_label_areas(labels: np.ndarray) -> Dict[int, int]:
    """Return a dict of {cell_id: pixel_count} for each label."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts) if u != 0}


# ---------------------------------------------------------------------------
# Segmentation comparison
# ---------------------------------------------------------------------------

@dataclass
class SegmentationMetrics:
    """Quantitative comparison between two segmentations.

    Fields
    ------
    n_cells_a, n_cells_b:
        Cell counts in each segmentation.
    mean_area_a, mean_area_b:
        Mean cell area (pixels) in each segmentation.
    overlap_coefficient:
        Fraction of pixels with a non-background label in both masks that
        are assigned to the same cell ID.  Ranges [0, 1].
    mean_centroid_distance:
        Mean Euclidean distance between matched centroids (Hungarian matching
        not performed at this scale — we use nearest-neighbour matching).
    protein_pearson:
        Dict ``{marker: Pearson r}`` for per-cell protein intensities.
    protein_spearman:
        Dict ``{marker: Spearman r}`` for per-cell protein intensities.
    transcript_pearson:
        Pearson r of total transcript counts per cell.
    transcript_spearman:
        Spearman r of total transcript counts per cell.
    summary_df:
        Tabular summary as a DataFrame.
    """
    n_cells_a: int = 0
    n_cells_b: int = 0
    mean_area_a: float = 0.0
    mean_area_b: float = 0.0
    overlap_coefficient: float = 0.0
    mean_centroid_distance: float = float("nan")
    protein_pearson: Dict[str, float] = field(default_factory=dict)
    protein_spearman: Dict[str, float] = field(default_factory=dict)
    transcript_pearson: float = float("nan")
    transcript_spearman: float = float("nan")

    @property
    def summary_df(self) -> pd.DataFrame:
        """Return a concise summary table."""
        rows = [
            ("Cell count (A)", self.n_cells_a),
            ("Cell count (B)", self.n_cells_b),
            ("Mean area (A, px²)", round(self.mean_area_a, 1)),
            ("Mean area (B, px²)", round(self.mean_area_b, 1)),
            ("Overlap coefficient", round(self.overlap_coefficient, 4)),
            ("Mean centroid distance (px)", round(self.mean_centroid_distance, 2)),
            ("Transcript Pearson r", round(self.transcript_pearson, 4)),
            ("Transcript Spearman r", round(self.transcript_spearman, 4)),
        ]
        for marker in self.protein_pearson:
            rows.append((
                f"Protein Pearson r ({marker})",
                round(self.protein_pearson[marker], 4),
            ))
        return pd.DataFrame(rows, columns=["Metric", "Value"])


def compare_segmentations(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    comet_a: Optional[pd.DataFrame] = None,
    comet_b: Optional[pd.DataFrame] = None,
    counts_a: Optional[pd.DataFrame] = None,
    counts_b: Optional[pd.DataFrame] = None,
) -> SegmentationMetrics:
    """Compare two label masks and compute overlap + correlation metrics.

    Parameters
    ----------
    labels_a, labels_b:
        2-D integer label masks to compare.  Must be the same shape.
    comet_a, comet_b:
        Optional per-cell COMET intensity DataFrames for each segmentation.
        Must have a ``cell_id`` column.
    counts_a, counts_b:
        Optional per-cell transcript count DataFrames for each segmentation.
        Index should be cell IDs, columns are gene names.

    Returns
    -------
    SegmentationMetrics
    """
    from scipy.stats import pearsonr, spearmanr  # type: ignore

    if labels_a.shape != labels_b.shape:
        raise ValueError(
            f"Label masks must have the same shape. "
            f"Got {labels_a.shape} and {labels_b.shape}."
        )

    metrics = SegmentationMetrics()

    # ---- Cell counts and areas -------------------------------------------
    areas_a = _compute_label_areas(labels_a)
    areas_b = _compute_label_areas(labels_b)
    metrics.n_cells_a = len(areas_a)
    metrics.n_cells_b = len(areas_b)
    metrics.mean_area_a = float(np.mean(list(areas_a.values()))) if areas_a else 0.0
    metrics.mean_area_b = float(np.mean(list(areas_b.values()))) if areas_b else 0.0

    # ---- Pixel-level overlap (intersection of non-background) ------------
    fg_a = labels_a > 0
    fg_b = labels_b > 0
    both_fg = fg_a & fg_b
    if both_fg.sum() > 0:
        agree = (labels_a[both_fg] == labels_b[both_fg]).sum()
        metrics.overlap_coefficient = float(agree) / float(both_fg.sum())

    # ---- Centroid distance (nearest-neighbour match) ---------------------
    centroids_a = _compute_centroids(labels_a, areas_a)
    centroids_b = _compute_centroids(labels_b, areas_b)
    if centroids_a and centroids_b:
        metrics.mean_centroid_distance = _mean_nn_distance(centroids_a, centroids_b)

    # ---- Protein correlations -------------------------------------------
    if comet_a is not None and comet_b is not None:
        markers = [c for c in comet_a.columns if c != "cell_id"]
        for marker in markers:
            if marker not in comet_b.columns:
                continue
            vals_a = comet_a[marker].dropna().to_numpy()
            vals_b = comet_b[marker].dropna().to_numpy()
            n = min(len(vals_a), len(vals_b))
            if n < 3:
                continue
            r_p, _ = pearsonr(vals_a[:n], vals_b[:n])
            r_s, _ = spearmanr(vals_a[:n], vals_b[:n])
            metrics.protein_pearson[marker] = float(r_p)
            metrics.protein_spearman[marker] = float(r_s)

    # ---- Transcript correlations ----------------------------------------
    if counts_a is not None and counts_b is not None:
        total_a = counts_a.sum(axis=1).sort_index()
        total_b = counts_b.sum(axis=1).sort_index()
        shared = total_a.index.intersection(total_b.index)
        if len(shared) >= 3:
            r_p, _ = pearsonr(total_a[shared].to_numpy(), total_b[shared].to_numpy())
            r_s, _ = spearmanr(total_a[shared].to_numpy(), total_b[shared].to_numpy())
            metrics.transcript_pearson = float(r_p)
            metrics.transcript_spearman = float(r_s)

    logger.info(
        "Segmentation comparison: A=%d cells, B=%d cells, overlap=%.3f",
        metrics.n_cells_a, metrics.n_cells_b, metrics.overlap_coefficient,
    )
    return metrics


def _compute_centroids(
    labels: np.ndarray,
    areas: Dict[int, int],
) -> Dict[int, Tuple[float, float]]:
    """Compute centroids of each label by averaging pixel coordinates."""
    centroids: Dict[int, Tuple[float, float]] = {}
    for cell_id in areas:
        rows, cols = np.where(labels == cell_id)
        centroids[cell_id] = (float(cols.mean()), float(rows.mean()))
    return centroids


def _mean_nn_distance(
    centroids_a: Dict[int, Tuple[float, float]],
    centroids_b: Dict[int, Tuple[float, float]],
) -> float:
    """Compute mean nearest-neighbour distance from A centroids to B centroids."""
    pts_a = np.array(list(centroids_a.values()))
    pts_b = np.array(list(centroids_b.values()))

    dists = []
    for pt in pts_a:
        d = np.hypot(pts_b[:, 0] - pt[0], pts_b[:, 1] - pt[1])
        dists.append(float(d.min()))

    return float(np.mean(dists)) if dists else float("nan")
