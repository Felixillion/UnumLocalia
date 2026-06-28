"""
spatialbench.benchmark
======================
Clustering and marker comparison benchmarks.

Clustering benchmarks
---------------------
Compare two sets of cluster labels (e.g. Leiden clusters from original
Xenium segmentation vs. from a user-uploaded segmentation) using:

* **Adjusted Rand Index (ARI)** — concordance between two partitions,
  corrected for chance.  Range: [-1, 1]; 1 = perfect agreement.
* **Normalized Mutual Information (NMI)** — information shared between two
  partitions.  Range: [0, 1].

Visualisations
--------------
* **Contingency heatmap** — cell-count overlap between cluster pairs.
* **Sankey diagram** — flow of cells between cluster assignments (plotly).

Marker comparison
-----------------
Given two AnnData objects (or one AnnData and two protein intensity matrices),
compare per-cell intensities for a set of overlapping markers:

* Scatter plot per marker.
* Pearson and Spearman correlations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering comparison metrics
# ---------------------------------------------------------------------------

def compare_clusterings(
    labels_a: Sequence,
    labels_b: Sequence,
    name_a: str = "Segmentation A",
    name_b: str = "Segmentation B",
) -> Dict[str, float]:
    """Compute ARI and NMI between two sets of cluster labels.

    Parameters
    ----------
    labels_a, labels_b:
        Cluster label sequences (e.g. pandas Series, list, or numpy array).
        Must be the same length.
    name_a, name_b:
        Human-readable names for reporting.

    Returns
    -------
    dict
        Keys: ``'ari'``, ``'nmi'``, ``'name_a'``, ``'name_b'``.

    Raises
    ------
    ValueError
        If the label sequences have different lengths.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a = np.asarray(labels_a)
    b = np.asarray(labels_b)

    if len(a) != len(b):
        raise ValueError(
            f"Label sequences must be the same length: got {len(a)} and {len(b)}."
        )

    ari = float(adjusted_rand_score(a, b))
    nmi = float(normalized_mutual_info_score(a, b, average_method="arithmetic"))

    logger.info(
        "Clustering comparison ('%s' vs '%s'): ARI=%.4f  NMI=%.4f",
        name_a, name_b, ari, nmi,
    )
    return {"ari": ari, "nmi": nmi, "name_a": name_a, "name_b": name_b}


def clustering_metrics_table(results: Dict[str, float]) -> pd.DataFrame:
    """Format clustering comparison results as a printable DataFrame.

    Parameters
    ----------
    results:
        Output of :func:`compare_clusterings`.

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        ("Adjusted Rand Index (ARI)", round(results["ari"], 4)),
        ("Normalized Mutual Information (NMI)", round(results["nmi"], 4)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------------------------------------------------------------------------
# Contingency heatmap
# ---------------------------------------------------------------------------

def plot_contingency(
    labels_a: Sequence,
    labels_b: Sequence,
    name_a: str = "Segmentation A clusters",
    name_b: str = "Segmentation B clusters",
    figsize: tuple = (8, 7),
    normalize: bool = True,
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot a heatmap of cell counts shared between cluster pairs.

    Parameters
    ----------
    labels_a, labels_b:
        Cluster label sequences of equal length.
    name_a, name_b:
        Axis labels.
    figsize:
        Figure size.
    normalize:
        If ``True``, normalise each row of the contingency matrix to fractions
        (rows sum to 1).
    cmap:
        Matplotlib colormap.

    Returns
    -------
    matplotlib.Figure
    """
    a = pd.Series(labels_a, name=name_a)
    b = pd.Series(labels_b, name=name_b)

    contingency = pd.crosstab(a, b)

    if normalize:
        row_sums = contingency.sum(axis=1)
        contingency = contingency.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(contingency.to_numpy(), cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(contingency.columns)))
    ax.set_yticks(range(len(contingency.index)))
    ax.set_xticklabels(contingency.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(contingency.index, fontsize=8)
    ax.set_xlabel(name_b)
    ax.set_ylabel(name_a)
    ax.set_title(
        f"Cluster contingency\n({name_a} vs {name_b})"
        + (" — row-normalised" if normalize else "")
    )

    plt.colorbar(im, ax=ax, shrink=0.7, label="Fraction" if normalize else "Count")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sankey diagram
# ---------------------------------------------------------------------------

def plot_sankey(
    labels_a: Sequence,
    labels_b: Sequence,
    name_a: str = "Segmentation A",
    name_b: str = "Segmentation B",
    min_flow: int = 5,
) -> "plotly.graph_objects.Figure":
    """Generate an interactive Sankey diagram showing cell flow between clusters.

    Parameters
    ----------
    labels_a, labels_b:
        Cluster label sequences (same length).
    name_a, name_b:
        Node label prefixes for each side.
    min_flow:
        Minimum number of cells required to draw a flow link (filters noise).

    Returns
    -------
    plotly.graph_objects.Figure
        An interactive Plotly figure that can be displayed in a QWebEngineView.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the Sankey diagram. "
            "Install with: conda install plotly"
        ) from exc

    a = pd.Series(labels_a).astype(str)
    b = pd.Series(labels_b).astype(str)

    # Build contingency table
    contingency = pd.crosstab(a, b)

    # Node labels: left side = A clusters, right side = B clusters
    cats_a = sorted(contingency.index.tolist())
    cats_b = sorted(contingency.columns.tolist())

    nodes_a = [f"{name_a}: {c}" for c in cats_a]
    nodes_b = [f"{name_b}: {c}" for c in cats_b]
    all_nodes = nodes_a + nodes_b
    node_indices = {n: i for i, n in enumerate(all_nodes)}

    # Generate links
    sources, targets, values = [], [], []
    for cat_a in cats_a:
        for cat_b in cats_b:
            count = int(contingency.at[cat_a, cat_b])
            if count >= min_flow:
                sources.append(node_indices[f"{name_a}: {cat_a}"])
                targets.append(node_indices[f"{name_b}: {cat_b}"])
                values.append(count)

    # Colour palette
    import plotly.express as px
    n_colors = max(len(cats_a), len(cats_b), 1)
    palette = px.colors.qualitative.Plotly[:n_colors] * (n_colors // 10 + 1)
    node_colors = [palette[i % len(palette)] for i in range(len(all_nodes))]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=all_nodes,
            color=node_colors,
            pad=15,
            thickness=20,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))
    fig.update_layout(
        title=f"Cell flow: {name_a} → {name_b}",
        font_size=11,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Marker comparison
# ---------------------------------------------------------------------------

def compare_markers(
    values_a: Union[pd.DataFrame, np.ndarray],
    values_b: Union[pd.DataFrame, np.ndarray],
    markers: Optional[List[str]] = None,
    columns_a: Optional[List[str]] = None,
    columns_b: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare per-cell intensities for a set of markers between two datasets.

    Parameters
    ----------
    values_a, values_b:
        Per-cell intensity matrices.  Either DataFrames (with marker column
        names) or 2-D numpy arrays.
    markers:
        Subset of marker names to compare.  If ``None``, compare all columns
        that are present in both datasets.
    columns_a, columns_b:
        Column names for *values_a* / *values_b* when they are numpy arrays.

    Returns
    -------
    pd.DataFrame
        Columns: ``marker``, ``pearson_r``, ``pearson_p``,
        ``spearman_r``, ``spearman_p``, ``n_cells``.
    """
    from scipy.stats import pearsonr, spearmanr

    # Convert arrays to DataFrames
    if isinstance(values_a, np.ndarray):
        if columns_a is None:
            raise ValueError("columns_a must be provided when values_a is an ndarray.")
        values_a = pd.DataFrame(values_a, columns=columns_a)
    if isinstance(values_b, np.ndarray):
        if columns_b is None:
            raise ValueError("columns_b must be provided when values_b is an ndarray.")
        values_b = pd.DataFrame(values_b, columns=columns_b)

    if markers is None:
        markers = sorted(set(values_a.columns) & set(values_b.columns))

    if not markers:
        raise ValueError("No overlapping marker columns found between the two datasets.")

    records = []
    for marker in markers:
        if marker not in values_a.columns or marker not in values_b.columns:
            logger.warning("Marker '%s' not in both datasets; skipping.", marker)
            continue

        a = values_a[marker].dropna().to_numpy()
        b = values_b[marker].dropna().to_numpy()
        n = min(len(a), len(b))
        if n < 3:
            logger.warning("Fewer than 3 values for marker '%s'; skipping.", marker)
            continue

        r_p, p_p = pearsonr(a[:n], b[:n])
        r_s, p_s = spearmanr(a[:n], b[:n])
        records.append({
            "marker": marker,
            "pearson_r": round(float(r_p), 4),
            "pearson_p": round(float(p_p), 4),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": round(float(p_s), 4),
            "n_cells": n,
        })

    df = pd.DataFrame(records)
    logger.info("Marker comparison: %d markers compared.", len(df))
    return df


def plot_marker_scatter(
    values_a: Union[pd.DataFrame, np.ndarray],
    values_b: Union[pd.DataFrame, np.ndarray],
    marker: str,
    name_a: str = "Segmentation A",
    name_b: str = "Segmentation B",
    columns_a: Optional[List[str]] = None,
    columns_b: Optional[List[str]] = None,
    figsize: tuple = (5, 5),
) -> plt.Figure:
    """Scatter plot of per-cell intensities for one marker between two datasets.

    A line of best fit and correlation coefficients are added automatically.

    Parameters
    ----------
    values_a, values_b:
        Per-cell intensity matrices (DataFrame or ndarray).
    marker:
        Marker to plot.
    name_a, name_b:
        Axis labels.
    columns_a, columns_b:
        Column names when passing ndarrays.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.Figure
    """
    from scipy.stats import pearsonr, spearmanr

    if isinstance(values_a, np.ndarray):
        values_a = pd.DataFrame(values_a, columns=columns_a or [])
    if isinstance(values_b, np.ndarray):
        values_b = pd.DataFrame(values_b, columns=columns_b or [])

    a = values_a[marker].dropna().to_numpy()
    b = values_b[marker].dropna().to_numpy()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    r_p, _ = pearsonr(a, b)
    r_s, _ = spearmanr(a, b)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(a, b, alpha=0.4, s=8, color="steelblue", linewidths=0)

    # Line of best fit
    m, c = np.polyfit(a, b, 1)
    x_line = np.linspace(a.min(), a.max(), 200)
    ax.plot(x_line, m * x_line + c, "r-", linewidth=1.5, label="Line of best fit")

    ax.set_xlabel(f"{marker} — {name_a}")
    ax.set_ylabel(f"{marker} — {name_b}")
    ax.set_title(
        f"{marker}\nPearson r = {r_p:.3f}  ·  Spearman r = {r_s:.3f}  (n={n})"
    )
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_marker_correlation_summary(
    comparison_df: pd.DataFrame,
    metric: str = "pearson_r",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Bar chart summarising per-marker correlation coefficients.

    Parameters
    ----------
    comparison_df:
        Output of :func:`compare_markers`.
    metric:
        Column to plot: ``'pearson_r'`` or ``'spearman_r'``.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.Figure
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Column '{metric}' not in comparison DataFrame.")

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["steelblue" if v >= 0 else "salmon" for v in comparison_df[metric]]
    bars = ax.barh(comparison_df["marker"], comparison_df[metric], color=colors)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Marker comparison — {metric.replace('_', ' ').title()}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-1.05, 1.05)

    # Annotate bars with values
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(
            bar.get_width() + 0.02 if val >= 0 else bar.get_width() - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    fig.tight_layout()
    return fig
