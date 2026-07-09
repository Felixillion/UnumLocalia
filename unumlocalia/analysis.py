"""
unumlocalia.analysis
=====================
Single-cell analysis pipeline built on top of Scanpy.

All functions operate on ``anndata.AnnData`` objects and return either
a modified AnnData or a ``matplotlib.Figure``.  No ``plt.show()`` calls
are made — display is the responsibility of the calling widget or script.

Supported modalities
--------------------
* ``'genes'``   — log1p-normalised gene count matrix (``.X``).
* ``'proteins'``— arcsinh-transformed protein intensity matrix
                  (``.obsm['X_protein']``).
* ``'combined'``— concatenated and scaled gene + protein matrix.

Each modality is stored in ``.obsm`` under a key like ``'X_pca_genes'``
so that multiple modality analyses can co-exist in the same AnnData.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Key names used in obsm / obs
_MODALITY_KEYS = {
    "genes": "X_pca_genes",
    "proteins": "X_pca_proteins",
    "combined": "X_pca_combined",
}
_UMAP_KEYS = {
    "genes": "X_umap_genes",
    "proteins": "X_umap_proteins",
    "combined": "X_umap_combined",
}


# ---------------------------------------------------------------------------
# Modality preparation
# ---------------------------------------------------------------------------

def prepare_modality(
    adata: "anndata.AnnData",
    modality: Literal["genes", "proteins", "combined"] = "genes",
    cofactor: float = 5.0,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
) -> np.ndarray:
    """Return a normalised feature matrix for the requested modality.

    The matrix is *not* stored in the AnnData — it is returned for
    immediate use and discarded after analysis.

    Parameters
    ----------
    adata:
        AnnData object.
    modality:
        Which data modality to prepare.
    cofactor:
        Arcsinh cofactor for protein normalisation.
    target_sum:
        Library-size normalisation target for gene counts.
    n_top_genes:
        Number of highly variable genes to select (genes modality only).

    Returns
    -------
    np.ndarray
        Dense 2-D matrix of shape ``(n_cells, n_features)``.

    Raises
    ------
    ValueError
        If the requested modality data is not present in *adata*.
    """
    import scanpy as sc

    if modality == "genes":
        return _prepare_genes(adata, target_sum=target_sum, n_top_genes=n_top_genes)

    elif modality == "proteins":
        return _prepare_proteins(adata, cofactor=cofactor)

    elif modality == "combined":
        gene_mat = _prepare_genes(adata, target_sum=target_sum, n_top_genes=n_top_genes)
        prot_mat = _prepare_proteins(adata, cofactor=cofactor)
        # Scale each modality to unit variance before concatenation
        from sklearn.preprocessing import StandardScaler
        gene_scaled = StandardScaler().fit_transform(gene_mat)
        prot_scaled = StandardScaler().fit_transform(prot_mat)
        return np.hstack([gene_scaled, prot_scaled])

    else:
        raise ValueError(
            f"Unknown modality '{modality}'. Use 'genes', 'proteins', or 'combined'."
        )


def _prepare_genes(
    adata: "anndata.AnnData",
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
) -> np.ndarray:
    """Normalise gene counts: library-size → log1p → HVG selection."""
    import scanpy as sc

    if adata.n_vars == 0:
        raise ValueError("AnnData has no gene variables (n_vars == 0).")

    # Work on a lightweight copy to avoid modifying the input
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)

    if tmp.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat_v3")
        tmp = tmp[:, tmp.var.highly_variable]

    X = tmp.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)


def _prepare_proteins(
    adata: "anndata.AnnData",
    cofactor: float = 5.0,
) -> np.ndarray:
    """Arcsinh-transform the protein intensity matrix."""
    if "X_protein" not in adata.obsm:
        raise ValueError(
            "No protein data found in adata.obsm['X_protein']. "
            "Run measure_comet_intensities and build_anndata first."
        )
    prot = adata.obsm["X_protein"].astype(np.float64)
    # Replace NaNs with 0 before transform
    prot = np.nan_to_num(prot, nan=0.0)
    return np.arcsinh(prot / cofactor).astype(np.float32)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(
    adata: "anndata.AnnData",
    modality: Literal["genes", "proteins", "combined"] = "genes",
    n_comps: int = 50,
    **prepare_kwargs,
) -> "anndata.AnnData":
    """Run PCA and store result in ``adata.obsm``.

    Parameters
    ----------
    adata:
        AnnData object (modified in place).
    modality:
        Data modality to use.
    n_comps:
        Number of principal components.
    **prepare_kwargs:
        Forwarded to :func:`prepare_modality`.

    Returns
    -------
    anndata.AnnData
        The same *adata*, with ``obsm[X_pca_<modality>]`` and
        ``uns['pca_<modality>']['variance_ratio']`` added.
    """
    from sklearn.decomposition import PCA

    X = prepare_modality(adata, modality=modality, **prepare_kwargs)
    n_comps = min(n_comps, X.shape[1], X.shape[0] - 1)

    pca = PCA(n_components=n_comps, random_state=42)
    embedding = pca.fit_transform(X).astype(np.float32)

    key = _MODALITY_KEYS[modality]
    adata.obsm[key] = embedding
    adata.uns[f"pca_{modality}"] = {
        "variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    logger.info(
        "PCA (%s): %d cells × %d PCs, top-PC variance=%.1f%%",
        modality, *embedding.shape,
        pca.explained_variance_ratio_[0] * 100,
    )
    return adata


# ---------------------------------------------------------------------------
# Neighbours and UMAP
# ---------------------------------------------------------------------------

def run_neighbors(
    adata: "anndata.AnnData",
    modality: Literal["genes", "proteins", "combined"] = "genes",
    n_neighbors: int = 15,
    n_pcs: int = 50,
) -> "anndata.AnnData":
    """Compute the k-nearest-neighbour graph using Scanpy.

    Parameters
    ----------
    adata:
        AnnData object (modified in place).
    modality:
        Which PCA embedding to use as input.
    n_neighbors:
        Number of neighbours.
    n_pcs:
        Number of PCA components to use.

    Returns
    -------
    anndata.AnnData
    """
    import scanpy as sc

    pca_key = _MODALITY_KEYS[modality]
    if pca_key not in adata.obsm:
        logger.info("PCA not found for '%s'; running PCA first.", modality)
        run_pca(adata, modality=modality)

    sc.pp.neighbors(
        adata,
        use_rep=pca_key,
        n_neighbors=n_neighbors,
        n_pcs=min(n_pcs, adata.obsm[pca_key].shape[1]),
        key_added=f"neighbors_{modality}",
    )
    logger.info("Computed neighbours (%s): n=%d", modality, n_neighbors)
    return adata


def run_umap(
    adata: "anndata.AnnData",
    modality: Literal["genes", "proteins", "combined"] = "genes",
    min_dist: float = 0.3,
    spread: float = 1.0,
) -> "anndata.AnnData":
    """Run UMAP and store result in ``adata.obsm[X_umap_<modality>]``.

    Parameters
    ----------
    adata:
        AnnData object (modified in place).
    modality:
        Which neighbour graph to use.
    min_dist, spread:
        UMAP layout parameters.

    Returns
    -------
    anndata.AnnData
    """
    import scanpy as sc

    neighbors_key = f"neighbors_{modality}"
    if neighbors_key not in adata.uns:
        logger.info("Neighbours not found for '%s'; computing first.", modality)
        run_neighbors(adata, modality=modality)

    umap_key = _UMAP_KEYS[modality]
    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        neighbors_key=neighbors_key,
    )
    # Rename the default X_umap key to our modality-specific key
    if "X_umap" in adata.obsm:
        adata.obsm[umap_key] = adata.obsm.pop("X_umap")

    logger.info("UMAP complete (%s).", modality)
    return adata


# ---------------------------------------------------------------------------
# Leiden clustering
# ---------------------------------------------------------------------------

def run_leiden(
    adata: "anndata.AnnData",
    modality: Literal["genes", "proteins", "combined"] = "genes",
    resolution: float = 0.5,
    cluster_key: Optional[str] = None,
) -> "anndata.AnnData":
    """Run Leiden clustering and store labels in ``adata.obs``.

    Parameters
    ----------
    adata:
        AnnData object (modified in place).
    modality:
        Which neighbour graph to use.
    resolution:
        Leiden resolution parameter (higher → more clusters).
    cluster_key:
        Column name in ``adata.obs`` for cluster labels.  Defaults to
        ``'leiden_<modality>'``.

    Returns
    -------
    anndata.AnnData
    """
    import scanpy as sc

    neighbors_key = f"neighbors_{modality}"
    if neighbors_key not in adata.uns:
        logger.info("Neighbours not found for '%s'; computing first.", modality)
        run_neighbors(adata, modality=modality)

    obs_key = cluster_key or f"leiden_{modality}"
    sc.tl.leiden(
        adata,
        resolution=resolution,
        neighbors_key=neighbors_key,
        key_added=obs_key,
    )

    n_clusters = adata.obs[obs_key].nunique()
    logger.info(
        "Leiden clustering (%s, res=%.2f): %d clusters", modality, resolution, n_clusters
    )
    return adata


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap(
    adata: "anndata.AnnData",
    color: Union[str, List[str]],
    modality: Literal["genes", "proteins", "combined"] = "genes",
    figsize: tuple = (6, 5),
    point_size: int = 5,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot a UMAP embedding coloured by a metadata column or gene expression.

    Parameters
    ----------
    adata:
        AnnData object.
    color:
        Column in ``adata.obs`` or gene name to use for colouring.
    modality:
        Which UMAP embedding to use.
    figsize:
        Figure size in inches.
    point_size:
        Scatter point size.
    title:
        Plot title.

    Returns
    -------
    matplotlib.Figure
    """
    import scanpy as sc

    umap_key = _UMAP_KEYS[modality]
    if umap_key not in adata.obsm:
        raise ValueError(
            f"UMAP embedding not found for modality '{modality}'. "
            f"Run run_umap(adata, modality='{modality}') first."
        )

    # Temporarily point the default X_umap to our key
    adata.obsm["X_umap"] = adata.obsm[umap_key]

    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.umap(
        adata,
        color=color,
        size=point_size,
        ax=ax,
        show=False,
        title=title or (color if isinstance(color, str) else ", ".join(color)),
    )
    fig.tight_layout()
    return fig


def plot_heatmap(
    adata: "anndata.AnnData",
    features: List[str],
    groupby: str,
    modality: Literal["genes", "proteins"] = "genes",
    figsize: tuple = (10, 6),
    standard_scale: Optional[str] = "var",
) -> plt.Figure:
    """Plot a heatmap of mean feature expression per cluster.

    Parameters
    ----------
    adata:
        AnnData object.
    features:
        List of gene names (for *genes*) or protein names (for *proteins*).
    groupby:
        Column in ``adata.obs`` to group cells by (e.g. cluster column).
    modality:
        Which data to visualise.
    figsize:
        Figure size.
    standard_scale:
        Scanpy standard_scale option: ``'var'`` (per-gene) or ``'obs'`` or ``None``.

    Returns
    -------
    matplotlib.Figure
    """
    import scanpy as sc

    if modality == "proteins":
        adata = _inject_proteins_as_vars(adata, features)

    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.heatmap(
        adata,
        var_names=[f for f in features if f in adata.var_names],
        groupby=groupby,
        ax=ax,
        show=False,
        standard_scale=standard_scale,
        cmap="RdBu_r",
    )
    fig.tight_layout()
    return fig


def plot_dotplot(
    adata: "anndata.AnnData",
    features: List[str],
    groupby: str,
    modality: Literal["genes", "proteins"] = "genes",
    figsize: tuple = (10, 5),
    standard_scale: Optional[str] = "var",
) -> plt.Figure:
    """Plot a dot plot of mean expression and fraction expressed per cluster.

    Parameters
    ----------
    adata:
        AnnData object.
    features:
        Gene or protein names.
    groupby:
        Grouping column in ``adata.obs``.
    modality:
        Which data modality.
    figsize:
        Figure size.
    standard_scale:
        Scanpy standard_scale option.

    Returns
    -------
    matplotlib.Figure
    """
    import scanpy as sc

    if modality == "proteins":
        adata = _inject_proteins_as_vars(adata, features)

    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.dotplot(
        adata,
        var_names=[f for f in features if f in adata.var_names],
        groupby=groupby,
        ax=ax,
        show=False,
        standard_scale=standard_scale,
    )
    fig.tight_layout()
    return fig


def plot_spatial_clusters(
    adata: "anndata.AnnData",
    color: str,
    coord_key: str = "spatial",
    figsize: tuple = (7, 7),
    point_size: int = 3,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot cells in tissue space coloured by a metadata column.

    Parameters
    ----------
    adata:
        AnnData object with spatial coordinates in ``adata.obsm[coord_key]``.
    color:
        Column in ``adata.obs`` for colouring.
    coord_key:
        Key in ``adata.obsm`` containing ``(N, 2)`` spatial coordinates.
    figsize:
        Figure size.
    point_size:
        Scatter point size.
    title:
        Plot title.

    Returns
    -------
    matplotlib.Figure
    """
    if coord_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates not found in adata.obsm['{coord_key}']. "
            "Ensure build_anndata was called with centroid columns."
        )

    xy = adata.obsm[coord_key]
    cats = adata.obs[color]

    fig, ax = plt.subplots(figsize=figsize)

    if hasattr(cats, "cat"):
        # Categorical — use a colour per category
        categories = cats.cat.categories
        palette = plt.cm.get_cmap("tab20", len(categories))
        for i, cat in enumerate(categories):
            mask = cats == cat
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=point_size, c=[palette(i)], label=cat, alpha=0.7, linewidths=0
            )
        ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize="small")
    else:
        sc_plot = ax.scatter(
            xy[:, 0], xy[:, 1],
            s=point_size, c=cats.to_numpy(), cmap="viridis",
            alpha=0.7, linewidths=0
        )
        plt.colorbar(sc_plot, ax=ax, shrink=0.6)

    ax.set_title(title or color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()  # image convention: y increases downward
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_pca_variance(
    adata: "anndata.AnnData",
    modality: Literal["genes", "proteins", "combined"] = "genes",
    n_comps: int = 20,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    """Scree plot of PCA explained variance.

    Parameters
    ----------
    adata:
        AnnData object with PCA results.
    modality:
        Which PCA result to plot.
    n_comps:
        Number of components to show.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.Figure
    """
    key = f"pca_{modality}"
    if key not in adata.uns or "variance_ratio" not in adata.uns[key]:
        raise ValueError(
            f"PCA results not found for modality '{modality}'. "
            f"Run run_pca(adata, modality='{modality}') first."
        )

    variance_ratio = np.array(adata.uns[key]["variance_ratio"])[:n_comps]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(1, len(variance_ratio) + 1), variance_ratio * 100, color="steelblue")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"PCA Scree — {modality}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inject_proteins_as_vars(
    adata: "anndata.AnnData",
    protein_names: Optional[List[str]] = None,
) -> "anndata.AnnData":
    """Create a temporary AnnData with protein intensities as variables.

    This allows Scanpy plotting functions (which operate on .X / var_names)
    to be used directly for protein data.

    Parameters
    ----------
    adata:
        Source AnnData with ``obsm['X_protein']``.
    protein_names:
        Subset of protein names to include.  If ``None``, include all.

    Returns
    -------
    anndata.AnnData
        New temporary AnnData object (not in-place).
    """
    import anndata as ad
    import scipy.sparse as sp

    if "X_protein" not in adata.obsm:
        raise ValueError("No protein data in adata.obsm['X_protein'].")

    prot_matrix = adata.obsm["X_protein"].astype(np.float32)
    all_proteins = adata.uns.get("protein_names", [
        f"protein_{i}" for i in range(prot_matrix.shape[1])
    ])

    if protein_names is not None:
        indices = [
            all_proteins.index(p) for p in protein_names if p in all_proteins
        ]
        prot_matrix = prot_matrix[:, indices]
        var_names = [all_proteins[i] for i in indices]
    else:
        var_names = all_proteins

    tmp = ad.AnnData(
        X=sp.csr_matrix(prot_matrix),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=var_names),
    )
    return tmp


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
        :func:`unumlocalia.io.get_comet_arrays`.
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
