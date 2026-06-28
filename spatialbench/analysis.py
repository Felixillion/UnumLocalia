"""
spatialbench.analysis
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
