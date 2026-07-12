"""
UnumLocalia — Analysis and visualization toolkit for multimodal spatial biology data.
"""

# ------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------

# Create and load installed environment
# mamba env create -f environment.yml
# mamba activate unumlocalia_analysis


# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

from pathlib import Path

CORE_NAME = "core01"

CSV_FILE = next(
    Path(".").glob(
        f"{CORE_NAME}*_quantification_cells.csv"
    )
)

THRESHOLD_FILE = next(
    Path(".").glob(
        f"{CORE_NAME}*_thresholds.json"
    ),
    None,
)

OUTPUT_DIR = f"results_{CORE_NAME}"

MIN_GENES = 5
MIN_TRANSCRIPTS = 20

# Remove DAPI from protein analysis (recommended)
REMOVE_DAPI = True

GENE_PCA_COMPONENTS = 50
GENE_N_PCS = 30
GENE_HVGS = 2000

PROTEIN_N_PCS = 8

GENE_RESOLUTION = 0.5
PROTEIN_RESOLUTION = 0.03
COMBINED_RESOLUTION = 0.5

GENE_N_NEIGHBORS = 15
PROTEIN_N_NEIGHBORS = 30
COMBINED_N_NEIGHBORS = 15

RANDOM_STATE = 42


# ------------------------------------------------------------
# FILES PRODUCED
# ------------------------------------------------------------

# results/

#     gene/
#         gene_analysis.h5ad
#         cluster_assignments.csv
#         umap_clusters.png
#         pacmap_clusters.png
#         spatial_clusters.png
#         marker_plots...

#     protein/
#         protein_analysis.h5ad
#         cluster_assignments.csv
#         umap_clusters.png
#         pacmap_clusters.png
#         spatial_clusters.png

#     combined/
#         combined_analysis.h5ad
#         cluster_assignments.csv
#         umap_combined.png
#         pacmap_combined.png
#         spatial_combined.png

#     comparison/
#         ARI_NMI_scores.csv
#         heatmaps/
#         mapped/
#         merged/
#         sankey/


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd

import scanpy as sc
import pacmap

import seaborn as sns
import matplotlib.pyplot as plt

# Colours
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex

from scipy.stats import zscore

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Sankey diagrams
import plotly.graph_objects as go

from scipy.optimize import linear_sum_assignment

# JSON file
import json

# αSMA characterisation
import html


# ------------------------------------------------------------
# CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for sub in [
    "gene",
    "protein",
    "combined",
    "comparison",
]:
    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            sub,
        ),
        exist_ok=True,
    )


# Create folders for comparison subplots
for sub in [
    "gene",
    "protein",
    "combined",
    "comparison",
    "comparison/heatmaps",
    "comparison/mapped",
    "comparison/merged",
    "comparison/sankey",
]:
    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            sub,
        ),
        exist_ok=True,
    )


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

#
# Spatial plots
#
def spatial_plot(
    adata,
    cluster_key,
    path,
):
    coords = adata.obsm["spatial"]

    clusters = (
        adata.obs[cluster_key]
        .astype("category")
    )

    palette = adata.uns[
        f"{cluster_key}_colors"
    ]

    cmap = ListedColormap(palette)

    plt.figure(figsize=(8, 8))

    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=clusters.cat.codes,
        cmap=cmap,
        s=3,
    )

    plt.gca().invert_yaxis()
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


#
# Consistent cluster palette assignment
#
def assign_cluster_palette(
    adata,
    cluster_key,
):
    cats = list(
        adata.obs[cluster_key]
        .astype("category")
        .cat.categories
    )

    palette = sns.color_palette(
        "tab20",
        n_colors=max(20, len(cats))
    )[:len(cats)]

    palette_hex = [
            to_hex(c)
            for c in palette
    ]

    adata.uns[
        f"{cluster_key}_colors"
    ] = palette_hex

    return palette_hex


#
# Disagreement map
#
def disagreement_map(
    true_labels,
    mapped_labels,
    coords_df,
    outfile,
):

    same = (
        true_labels.astype(str)
        ==
        mapped_labels.astype(str)
    )

    plt.figure(
        figsize=(8,8)
    )

    plt.scatter(
        coords_df["centroid_x"],
        coords_df["centroid_y"],
        c=np.where(
            same,
            "lightgrey",
            "red",
        ),
        s=3,
    )

    plt.gca().invert_yaxis()
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=300,
    )

    plt.close()


#
# Sankey diagram
#
def sankey_from_contingency(
    ct,
    title,
    outfile,
):

    left_nodes = [
        str(x)
        for x in ct.index
    ]

    right_nodes = [
        str(x)
        for x in ct.columns
    ]

    all_nodes = (
        left_nodes
        +
        right_nodes
    )

    node_map = {
        n:i
        for i,n
        in enumerate(all_nodes)
    }

    source = []
    target = []
    value = []

    for r in ct.index:
        for c in ct.columns:

            v = ct.loc[r,c]

            if v == 0:
                continue

            source.append(
                node_map[str(r)]
            )

            target.append(
                node_map[str(c)]
            )

            value.append(
                int(v)
            )

    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=all_nodes
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
            ),
        )
    )

    fig.update_layout(
        title_text=title
    )

    fig.write_image(
        outfile,
        scale=3,
    )


#
# Cluster mapping functions
#
def hungarian_mapping(ct):
    """
    One-to-one mapping between clusterings.
    """

    cost = (
        ct.max().max()
        - ct.values
    )

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {
        str(ct.index[i]): str(ct.columns[j])
        for i, j in zip(
            row_ind,
            col_ind,
        )
    }

    return mapping


def merge_mapping(ct):
    """
    Many-to-one mapping.
    """

    mapping = {}

    for col in ct.columns:

        mapping[col] = (
            ct[col]
            .idxmax()
        )

    mapping = {
        str(k): str(v)
        for k, v in mapping.items()
    }

    return mapping


#
# Agreement metrics
#
def compute_agreement(
    true_labels,
    mapped_labels,
):

    agree = (
        true_labels.astype(str)
        ==
        mapped_labels.astype(str)
    )

    global_agreement = (
        agree.mean()
    )

    per_cluster = (
        pd.DataFrame({
            "cluster": true_labels,
            "agree": agree,
        })
        .groupby("cluster")["agree"]
        .mean()
    )

    return (
        global_agreement,
        per_cluster,
    )


#
# Three-way Sankey diagram
#
def sankey_three_way(prot_labels, gene_labels, comb_labels,
                     title, fname, min_flow=10):
    """
    prot_labels, gene_labels, comb_labels: pd.Series aligned by index (same cells)
    min_flow: minimum cell count to show a link; smaller flows aggregated into 'Other'
    """
    # Build contingency tables
    ct_pg = pd.crosstab(prot_labels, gene_labels)
    ct_gc = pd.crosstab(gene_labels, comb_labels)

    # Node lists
    prot_nodes = [f"P_{p}" for p in ct_pg.index.astype(str)]
    gene_nodes = [f"G_{g}" for g in ct_pg.columns.astype(str)]
    comb_nodes = [f"C_{c}" for c in ct_gc.columns.astype(str)]

    # All nodes
    all_nodes = prot_nodes + gene_nodes + comb_nodes
    node_index = {n:i for i,n in enumerate(all_nodes)}

    sources, targets, values = [], [], []

    # Links: protein -> gene (use ct_pg)
    for p in ct_pg.index:
        for g in ct_pg.columns:
            v = ct_pg.loc[p,g]
            if v >= min_flow:
                sources.append(node_index[f"P_{p}"])
                targets.append(node_index[f"G_{g}"])
                values.append(int(v))
            else:
                # aggregate small flows into P_other->G_{g} or skip; here we skip to keep simple
                pass

    # Links: gene -> combined (use ct_gc)
    for g in ct_gc.index:
        for c in ct_gc.columns:
            v = ct_gc.loc[g,c]
            if v >= min_flow:
                sources.append(node_index[f"G_{g}"])
                targets.append(node_index[f"C_{c}"])
                values.append(int(v))
            else:
                pass

    # Build Sankey
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=10,
            thickness=12,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=["#636EFA"]*len(prot_nodes) + ["#EF553B"]*len(gene_nodes) + ["#00CC96"]*len(comb_nodes)
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    fig.update_layout(title_text=title, font_size=10)
    fig.write_image(fname, scale=3)
    print("Saved:", fname)


#
# αSMA characterisation
#
def normalize_marker_name(name):

    name = html.unescape(
        str(name)
    ).strip()

    name = name.replace(
        "#945;",
        "α"
    )

    name = name.replace(
        "&alpha;",
        "α"
    )

    name = name.replace(
        "αSMA",
        "αSMA"
    )

    name = " ".join(
        name.split()
    )

    return name


# ------------------------------------------------------------
# DETECT COLUMNS AUTOMATICALLY
# ------------------------------------------------------------

df = pd.read_csv(CSV_FILE)

df.columns = [
    str(c).strip()
    for c in df.columns
]

metadata_cols = [
    "label",
    "cell_id",
    "centroid_x",
    "centroid_y",
    "area_px",
]

protein_cols = [
    c
    for c in df.columns
    if c.endswith("_mean")
]

#
# Additional mapping from normalised marker names to original CSV columns for αSMA characterisation
#
marker_to_column = {}

for col in protein_cols:

    base_name = normalize_marker_name(
        col.removesuffix("_mean")
    )

    marker_to_column[
        base_name
    ] = col

#
# Apply protein thresholing
#
if (
    THRESHOLD_FILE is not None
    and os.path.exists(
        THRESHOLD_FILE
    )
):

    print(
        f"Loading thresholds from: {THRESHOLD_FILE}"
    )

    with open(
        THRESHOLD_FILE,
        "r",
    ) as f:

        threshold_data = json.load(f)

    print(
        f"Threshold file core: "
        f"{threshold_data.get('core', 'unknown')}"
    )

    thresholds_applied = 0
    total_cells_thresholded = 0

    for marker, settings in (
        threshold_data["proteins"]
        .items()
    ):
        
        # Normalise marker name to match CSV columns
        marker = normalize_marker_name(
            marker
        )
        
        # Accommodate different naming conventions in CSV vs JSON (namely "_mean" suffix)
        csv_marker = marker_to_column.get(
            marker
        )

        if csv_marker is None:

            print(
                f"Warning: {marker} not present in CSV"
            )

            continue

        #
        # Threshold JSON currently stores:
        #
        # {
        #   "threshold": xxx,
        #   "display_max": xxx,
        #   "method": "manual"
        # }
        #

        threshold = settings.get(
            "threshold"
        )

        if threshold is None:
            continue

        n_below = (
            df[csv_marker] < threshold
        ).sum()

        df.loc[
            df[csv_marker] < threshold,
            csv_marker,
        ] = 0

        # Add a new column indicating whether the protein is positive (above threshold)
        # positive_col = (
        #     marker.replace(
        #         "_mean",
        #         "_positive"
        #     )
        # )
        #
        # df[positive_col] = (
        #     df[csv_marker] >= threshold
        # ).astype(int)

        thresholds_applied += 1
        total_cells_thresholded += n_below

        print(
            f"{marker}: "
            f"{n_below:,} cells clipped below threshold {threshold}"
        )

    # Threshold summary
    print(
        f"Total values thresholded: "
        f"{total_cells_thresholded:,}"
    )

    print(
        f"Applied thresholds to "
        f"{thresholds_applied} proteins"
    )

    # Save thresholded quantification table
    df.to_csv(
        f"{OUTPUT_DIR}/thresholded_quantification.csv",
        index=False,
    )

    with open(
        f"{OUTPUT_DIR}/thresholds_used.json",
        "w",
    ) as f:

        json.dump(
            threshold_data,
            f,
            indent=4,
        )
    
elif THRESHOLD_FILE is not None:

    print(
        f"Threshold file not found: "
        f"{THRESHOLD_FILE}"
    )


if REMOVE_DAPI:
    protein_cols = [
        c
        for c in protein_cols
        if c != "DAPI_mean"
    ]

gene_cols = [
    c
    for c in df.columns
    if c not in metadata_cols
    and c not in protein_cols
]

#
# QC printouts
#
print(
    f"Loaded {len(df):,} cells"
)

print(
    f"{len(gene_cols):,} genes"
)

print(
    f"{len(protein_cols):,} proteins"
)


# ------------------------------------------------------------
# GENE QC
# ------------------------------------------------------------

gene_counts = df[gene_cols]

df["n_genes"] = (
    gene_counts > 0
).sum(axis=1)

df["n_transcripts"] = (
    gene_counts.sum(axis=1)
)

gene_pass = (
    (df["n_genes"] >= MIN_GENES)
    &
    (df["n_transcripts"] >= MIN_TRANSCRIPTS)
)


# ------------------------------------------------------------
# PROTEIN COVERAGE
# ------------------------------------------------------------

protein_matrix = df[protein_cols]

df["protein_sum"] = (
    protein_matrix.sum(axis=1)
)

protein_pass = (
    df["protein_sum"] > 0
)


# ------------------------------------------------------------
# GENE ANALYSIS
# ------------------------------------------------------------

#
# Create
#
gene_df = df.loc[
    gene_pass
].copy()

#
# Build AnnData
#
gene_adata = sc.AnnData(
    X=gene_df[gene_cols].values
)

gene_adata.obs = gene_df[
    [
        "centroid_x",
        "centroid_y",
    ]
].copy()


gene_adata.obs_names = (
    gene_df["cell_id"]
    .astype(str)
)


gene_adata.var_names = gene_cols

gene_adata.obsm["spatial"] = (
    gene_df[
        ["centroid_x", "centroid_y"]
    ].to_numpy()
)

#
# Normalize
#
sc.pp.normalize_total(
    gene_adata,
    target_sum=1e4,
)

sc.pp.log1p(
    gene_adata
)

#
# HVGs
#
n_hvg = min(
    GENE_HVGS,
    gene_adata.n_vars,
)

sc.pp.highly_variable_genes(
    gene_adata,
    n_top_genes=n_hvg,
    flavor="seurat",
)

gene_adata = gene_adata[
    :,
    gene_adata.var["highly_variable"]
].copy()

#
# PCA
#
sc.pp.scale(
    gene_adata,
    max_value=10,
)

gene_pca_comps = min(
    GENE_PCA_COMPONENTS,
    gene_adata.n_vars - 1,
    gene_adata.n_obs - 1,
)

sc.tl.pca(
    gene_adata,
    n_comps=max(
        2,
        gene_pca_comps,
    ),
)

#
# UMAP
#
sc.pp.neighbors(
    gene_adata,
    n_neighbors=GENE_N_NEIGHBORS,
    n_pcs=GENE_N_PCS,
)

sc.tl.umap(
    gene_adata,
    random_state=RANDOM_STATE,
)

plt.close()


#
# PACMAP
#
pac = pacmap.PaCMAP(
    random_state=RANDOM_STATE,
)

gene_adata.obsm["X_pacmap"] = (
    pac.fit_transform(
        gene_adata.obsm["X_pca"]
    )
)

#
# Leiden
#
sc.tl.leiden(
    gene_adata,
    resolution=GENE_RESOLUTION,
    key_added="gene_cluster",
    flavor="igraph",
    n_iterations=2,
    random_state=RANDOM_STATE,
    directed=False,
)

# Print cluster counts
cluster_sizes = (
    gene_adata.obs["gene_cluster"]
    .value_counts()
    .sort_values(ascending=False)
)

print(cluster_sizes)

# Save cluster sizes to CSV
cluster_sizes.to_csv(
        f"{OUTPUT_DIR}/gene/cluster_sizes.csv",
        index=False,
    )

# Assign consistent cluster palette
assign_cluster_palette(
    gene_adata,
    "gene_cluster",
)


# ------------------------------------------------------------
# GENE MARKER ANALYSIS
# ------------------------------------------------------------

if gene_adata.obs["gene_cluster"].nunique() > 1:

    sc.tl.rank_genes_groups(
        gene_adata,
        "gene_cluster",
        method="wilcoxon",
    )

    dp = sc.pl.rank_genes_groups_dotplot(
        gene_adata,
        n_genes=5,
        groupby="gene_cluster",
        return_fig=True,
        show=False,
    )

    dp.savefig(
        f"{OUTPUT_DIR}/gene/dotplot_cluster_markers.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    top_genes = []

    for cl in (
        gene_adata.obs["gene_cluster"]
        .cat.categories
    ):

        genes = (
            sc.get.rank_genes_groups_df(
                gene_adata,
                group=cl,
            )
            .head(5)["names"]
            .tolist()
        )

        top_genes.extend(genes)

    top_genes = list(
        dict.fromkeys(
            top_genes
        )
    )

    sc.pl.heatmap(
        gene_adata,
        var_names=top_genes,
        groupby="gene_cluster",
        show=False,
    )

    plt.savefig(
        f"{OUTPUT_DIR}/gene/gene_heatmap_per_cluster.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ------------------------------------------------------------
# GENE PLOTS
# ------------------------------------------------------------

sc.pl.umap(
    gene_adata,
    color="gene_cluster",
    show=False,
)

plt.savefig(
    f"{OUTPUT_DIR}/gene/umap_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

sc.pl.embedding(
    gene_adata,
    basis="pacmap",
    color="gene_cluster",
    show=False,
)

plt.savefig(
    f"{OUTPUT_DIR}/gene/pacmap_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

spatial_plot(
    gene_adata,
    "gene_cluster",
    f"{OUTPUT_DIR}/gene/spatial_clusters.png",
)


# ------------------------------------------------------------
# PROTEIN ANALYSIS
# ------------------------------------------------------------

protein_df = df.loc[
    protein_pass
].copy()

# Build matrix
prot = protein_df[
    protein_cols
].copy()

prot = np.arcsinh(
    prot / 5.0
)

prot = prot.apply(
    zscore,
    axis=0
).fillna(0)

protein_adata = sc.AnnData(
    X=prot.values
)

protein_adata.obs = protein_df[
    [
        "centroid_x",
        "centroid_y",
    ]
].copy()


protein_adata.obs_names = (
    protein_df["cell_id"]
    .astype(str)
)


protein_adata.var_names = protein_cols

protein_adata.obsm["spatial"] = (
    protein_df[
        ["centroid_x", "centroid_y"]
    ].to_numpy()
)

sc.pp.scale(
    protein_adata,
    max_value=5,
)

protein_pca_comps = min(
    PROTEIN_N_PCS,
    len(protein_cols) - 1,
    protein_adata.n_obs - 1,
)

sc.tl.pca(
    protein_adata,
    n_comps=max(
        2,
        protein_pca_comps,
    ),
)

sc.pp.neighbors(
    protein_adata,
    n_neighbors=PROTEIN_N_NEIGHBORS,
)

sc.tl.umap(
    protein_adata,
    random_state=RANDOM_STATE,
)

pac = pacmap.PaCMAP(
    random_state=RANDOM_STATE,
)

protein_adata.obsm["X_pacmap"] = (
    pac.fit_transform(
        protein_adata.obsm["X_pca"]
    )
)

#
# Leiden
#
sc.tl.leiden(
    protein_adata,
    resolution=PROTEIN_RESOLUTION,
    key_added="protein_cluster",
    flavor="igraph",
    n_iterations=2,
    random_state=RANDOM_STATE,
    directed=False,
)

# Print cluster counts
cluster_sizes = (
    protein_adata.obs["protein_cluster"]
    .value_counts()
    .sort_values(ascending=False)
)

print(cluster_sizes)

# Save cluster sizes to CSV
cluster_sizes.to_csv(
        f"{OUTPUT_DIR}/protein/cluster_sizes.csv",
        index=False,
    )

# Assign consistent cluster palette
assign_cluster_palette(
    protein_adata,
    "protein_cluster",
)


# ------------------------------------------------------------
# PROTEIN HEATMAP
# ------------------------------------------------------------

protein_heat_df = pd.DataFrame(
    protein_adata.X,
    columns=protein_adata.var_names,
)

protein_heat_df["cluster"] = (
    protein_adata.obs["protein_cluster"]
    .astype(int)
    .values
)

mean_per_cluster = (
    protein_heat_df
    .groupby("cluster")
    .mean()
)

plt.figure(
    figsize=(10, 6)
)

sns.heatmap(
    mean_per_cluster,
    cmap="viridis",
)

plt.ylabel(
    "Cluster"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/protein/protein_heatmap_per_cluster.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# PROTEIN PLOTS
# ------------------------------------------------------------

sc.pl.umap(
    protein_adata,
    color="protein_cluster",
    show=False,
)

plt.savefig(
    f"{OUTPUT_DIR}/protein/umap_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

sc.pl.embedding(
    protein_adata,
    basis="pacmap",
    color="protein_cluster",
    show=False,
)

plt.savefig(
    f"{OUTPUT_DIR}/protein/pacmap_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

spatial_plot(
    protein_adata,
    "protein_cluster",
    f"{OUTPUT_DIR}/protein/spatial_clusters.png",
)


# ------------------------------------------------------------
# GENE PCs
# ------------------------------------------------------------

combined_df = df.loc[
    gene_pass
    &
    protein_pass
].copy()

#
# Save filtered cell tables
#
gene_df.to_csv(
    f"{OUTPUT_DIR}/gene/cells_used.csv",
    index=False,
)

protein_df.to_csv(
    f"{OUTPUT_DIR}/protein/cells_used.csv",
    index=False,
)

combined_df.to_csv(
    f"{OUTPUT_DIR}/combined/cells_used.csv",
    index=False,
)


combined_gene_adata = sc.AnnData(
    X=combined_df[gene_cols].values
)

combined_gene_adata.var_names = gene_cols

sc.pp.normalize_total(
    combined_gene_adata,
    target_sum=1e4,
)

sc.pp.log1p(
    combined_gene_adata,
)

#
# HVGs
#
n_hvg = min(
    GENE_HVGS,
    combined_gene_adata.n_vars,
)

sc.pp.highly_variable_genes(
    combined_gene_adata,
    n_top_genes=n_hvg,
    flavor="seurat",
)

combined_gene_adata = (
    combined_gene_adata[
        :,
        combined_gene_adata.var[
            "highly_variable"
        ]
    ]
    .copy()
)

sc.pp.scale(
    combined_gene_adata,
    max_value=10,
)

n_comps = min(
    GENE_N_PCS,
    combined_gene_adata.n_vars - 1,
    combined_gene_adata.n_obs - 1,
)

sc.tl.pca(
    combined_gene_adata,
    n_comps=n_comps,
)


gene_pcs = combined_gene_adata.obsm["X_pca"]


# ------------------------------------------------------------
# PROTEIN MATRIX
# ------------------------------------------------------------

prot = combined_df[
    protein_cols
].copy()

prot = np.arcsinh(
    prot / 5.0
)

prot = prot.apply(
    zscore,
    axis=0
).fillna(0)


# ------------------------------------------------------------
# BUILD COMBINED REPRESENTATION
# ------------------------------------------------------------

combined_matrix = np.hstack(
    [
        gene_pcs,
        prot.values,
    ]
)

combined_adata = sc.AnnData(
    X=combined_matrix
)


combined_adata.obs = combined_df[
    [
        "centroid_x",
        "centroid_y",
    ]
].copy()

combined_adata.obs_names = (
    combined_df["cell_id"]
    .astype(str)
)

combined_adata.obsm["spatial"] = (
    combined_df[
        ["centroid_x", "centroid_y"]
    ].to_numpy()
)


sc.pp.neighbors(
    combined_adata,
    n_neighbors=COMBINED_N_NEIGHBORS,
)

sc.tl.umap(
    combined_adata,
    random_state=RANDOM_STATE,
)

pac = pacmap.PaCMAP(
    random_state=RANDOM_STATE,
)

combined_adata.obsm["X_pacmap"] = (
    pac.fit_transform(
        combined_matrix
    )
)

#
# Leiden
#
sc.tl.leiden(
    combined_adata,
    resolution=COMBINED_RESOLUTION,
    key_added="combined_cluster",
    flavor="igraph",
    n_iterations=2,
    random_state=RANDOM_STATE,
    directed=False,
)

# Print cluster counts
cluster_sizes = (
    combined_adata.obs["combined_cluster"]
    .value_counts()
    .sort_values(ascending=False)
)

print(cluster_sizes)

# Save cluster sizes to CSV
cluster_sizes.to_csv(
        f"{OUTPUT_DIR}/combined/cluster_sizes.csv",
        index=False,
    )

# Assign consistent cluster palette
assign_cluster_palette(
    combined_adata,
    "combined_cluster",
)


# ------------------------------------------------------------
# COMBINED GENE MARKERS
# ------------------------------------------------------------

combined_gene_adata.obs[
    "combined_cluster"
] = (
    combined_adata.obs[
        "combined_cluster"
    ]
    .values
)

if (
    combined_gene_adata.obs[
        "combined_cluster"
    ].nunique() > 1
):
    
    combined_gene_adata.obs[
        "combined_cluster"
    ] = (
        combined_gene_adata.obs[
            "combined_cluster"
        ]
        .astype("category")
    )

    sc.tl.rank_genes_groups(
        combined_gene_adata,
        "combined_cluster",
        method="wilcoxon",
    )

    #
    # Heatmap
    #
    top_genes = []

    for cl in (
        combined_gene_adata.obs[
            "combined_cluster"
        ]
        .cat.categories
    ):

        genes = (
            sc.get.rank_genes_groups_df(
                combined_gene_adata,
                group=cl,
            )
            .head(5)["names"]
            .tolist()
        )

        top_genes.extend(
            genes
        )

    top_genes = list(
        dict.fromkeys(
            top_genes
        )
    )

    sc.pl.heatmap(
        combined_gene_adata,
        var_names=top_genes,
        groupby="combined_cluster",
        show=False,
    )

    plt.savefig(
        f"{OUTPUT_DIR}/combined/heatmap_combined_gene_markers.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    #
    # Dotplot
    #
    dp = sc.pl.rank_genes_groups_dotplot(
        combined_gene_adata,
        n_genes=5,
        groupby="combined_cluster",
        return_fig=True,
        show=False,
    )

    dp.savefig(
        f"{OUTPUT_DIR}/combined/dotplot_combined_gene_markers.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ------------------------------------------------------------
# COMBINED PLOTS
# ------------------------------------------------------------

sc.pl.umap(
    combined_adata,
    color="combined_cluster",
    show=False,
)

plt.savefig(
    f"{OUTPUT_DIR}/combined/umap_combined.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

sc.pl.embedding(
    combined_adata,
    basis="pacmap",
    color="combined_cluster",
    show=False,
)

plt.savefig(
    f"{OUTPUT_DIR}/combined/pacmap_combined.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

spatial_plot(
    combined_adata,
    "combined_cluster",
    f"{OUTPUT_DIR}/combined/spatial_combined.png",
)


# ------------------------------------------------------------
# SAVE CLUSTER ASSIGNMENTS
# ------------------------------------------------------------

#
# Gene
#
gene_clusters = (
    gene_adata.obs[
        ["gene_cluster"]
    ]
    .reset_index(names="cell_id")
)

gene_clusters.to_csv(
    f"{OUTPUT_DIR}/gene/cluster_assignments.csv",
    index=False,
)

#
# Protein
#
protein_clusters = (
    protein_adata.obs[
        ["protein_cluster"]
    ]
    .reset_index(names="cell_id")
)

protein_clusters.to_csv(
    f"{OUTPUT_DIR}/protein/cluster_assignments.csv",
    index=False,
)

#
# Combined
#
combined_clusters = (
    combined_adata.obs[
        ["combined_cluster"]
    ]
    .reset_index(names="cell_id")
)

combined_clusters.to_csv(
    f"{OUTPUT_DIR}/combined/cluster_assignments.csv",
    index=False,
)


#
# Save h5ad data
#
gene_adata.write_h5ad(
    f"{OUTPUT_DIR}/gene/gene_analysis.h5ad"
)

protein_adata.write_h5ad(
    f"{OUTPUT_DIR}/protein/protein_analysis.h5ad"
)

combined_adata.write_h5ad(
    f"{OUTPUT_DIR}/combined/combined_analysis.h5ad"
)


# ------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------

comparison = (
    gene_clusters
    .merge(
        protein_clusters,
        on="cell_id"
    )
    .merge(
        combined_clusters,
        on="cell_id"
    )
)


# ------------------------------------------------------------
# CLUSTER MAPPINGS
# ------------------------------------------------------------

ct_gp = pd.crosstab(
    comparison["gene_cluster"],
    comparison["protein_cluster"],
)

ct_gc = pd.crosstab(
    comparison["gene_cluster"],
    comparison["combined_cluster"],
)

ct_pc = pd.crosstab(
    comparison["protein_cluster"],
    comparison["combined_cluster"],
)

#
# Hungarian one-to-one
#
map_gp = hungarian_mapping(ct_gp)
map_gc = hungarian_mapping(ct_gc)
map_pc = hungarian_mapping(ct_pc)

#
# Many-to-one
#
merge_gp = merge_mapping(ct_gp)
merge_gc = merge_mapping(ct_gc)
merge_pc = merge_mapping(ct_pc)


# ------------------------------------------------------------
# APPLY MAPPINGS
# ------------------------------------------------------------

gene_labels = (
    comparison["gene_cluster"]
    .astype(str)
)

protein_labels = (
    comparison["protein_cluster"]
    .astype(str)
)

combined_labels = (
    comparison["combined_cluster"]
    .astype(str)
)

#
# Hungarian
#
mapped_protein_from_gene = (
    gene_labels.map(map_gp)
)

mapped_combined_from_gene = (
    gene_labels.map(map_gc)
)

mapped_combined_from_protein = (
    protein_labels.map(map_pc)
)

#
# Many-to-one
#
merged_gene_from_protein = (
    protein_labels.map(merge_gp)
)

merged_gene_from_combined = (
    combined_labels.map(merge_gc)
)

merged_protein_from_combined = (
    combined_labels.map(merge_pc)
)


# ------------------------------------------------------------
# HUNGARIAN DISAGREEMENT MAPS
# ------------------------------------------------------------

comparison_coords = (
    combined_df[
        [
            "cell_id",
            "centroid_x",
            "centroid_y",
        ]
    ]
    .drop_duplicates()
)

comparison_plot_df = (
    comparison.merge(
        comparison_coords,
        on="cell_id",
    )
)

disagreement_map(
    gene_labels,
    mapped_protein_from_gene,
    comparison_plot_df,
    f"{OUTPUT_DIR}/comparison/mapped/disagree_hungarian_gene_vs_protein.png",
)

disagreement_map(
    gene_labels,
    mapped_combined_from_gene,
    comparison_plot_df,
    f"{OUTPUT_DIR}/comparison/mapped/disagree_hungarian_gene_vs_combined.png",
)

disagreement_map(
    protein_labels,
    mapped_combined_from_protein,
    comparison_plot_df,
    f"{OUTPUT_DIR}/comparison/mapped/disagree_hungarian_protein_vs_combined.png",
)


# ------------------------------------------------------------
# MERGED DISAGREEMENT MAPS
# ------------------------------------------------------------

disagreement_map(
    gene_labels,
    merged_gene_from_protein,
    comparison_plot_df,
    f"{OUTPUT_DIR}/comparison/merged/disagree_merged_gene_vs_protein.png",
)

disagreement_map(
    gene_labels,
    merged_gene_from_combined,
    comparison_plot_df,
    f"{OUTPUT_DIR}/comparison/merged/disagree_merged_gene_vs_combined.png",
)

disagreement_map(
    protein_labels,
    merged_protein_from_combined,
    comparison_plot_df,
    f"{OUTPUT_DIR}/comparison/merged/disagree_merged_protein_vs_combined.png",
)


# ------------------------------------------------------------
# AGREEMENT METRICS
# ------------------------------------------------------------

global_gp_map, per_gp_map = (
    compute_agreement(
        gene_labels,
        mapped_protein_from_gene,
    )
)

global_gp_merge, per_gp_merge = (
    compute_agreement(
        gene_labels,
        merged_gene_from_protein,
    )
)

per_gp_map.to_csv(
    f"{OUTPUT_DIR}/comparison/mapped/hungarian_gene_vs_protein_agreement.csv"
)

per_gp_merge.to_csv(
    f"{OUTPUT_DIR}/comparison/merged/merged_gene_vs_protein_agreement.csv"
)

with open(
    f"{OUTPUT_DIR}/comparison/mapped/hungarian_global_agreement.txt",
    "w",
) as f:

    f.write(
        str(global_gp_map)
    )

with open(
    f"{OUTPUT_DIR}/comparison/merged/merged_global_agreement.txt",
    "w",
) as f:

    f.write(
        str(global_gp_merge)
    )


# ------------------------------------------------------------
# AGREEMENT BARPLOTS
# ------------------------------------------------------------

plt.figure(
    figsize=(10,5)
)

sns.barplot(
    x=per_gp_map.index,
    y=per_gp_map.values,
    color="steelblue",
)

plt.ylim(0,1)

plt.ylabel(
    "Agreement fraction"
)

plt.title(
    "Hungarian agreement"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/comparison/mapped/hungarian_agreement_barplot.png",
    dpi=300,
)

plt.close()


plt.figure(
    figsize=(10,5)
)

sns.barplot(
    x=per_gp_merge.index,
    y=per_gp_merge.values,
    color="seagreen",
)

plt.ylim(0,1)

plt.ylabel(
    "Agreement fraction"
)

plt.title(
    "Merged agreement"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/comparison/merged/merged_agreement_barplot.png",
    dpi=300,
)

plt.close()


# ------------------------------------------------------------
# ARI/NMI
# ------------------------------------------------------------

scores = pd.DataFrame(
    {
        "Gene vs Protein": [
            adjusted_rand_score(
                comparison["gene_cluster"],
                comparison["protein_cluster"],
            ),
            normalized_mutual_info_score(
                comparison["gene_cluster"],
                comparison["protein_cluster"],
            ),
        ],

        "Gene vs Combined": [
            adjusted_rand_score(
                comparison["gene_cluster"],
                comparison["combined_cluster"],
            ),
            normalized_mutual_info_score(
                comparison["gene_cluster"],
                comparison["combined_cluster"],
            ),
        ],

        "Protein vs Combined": [
            adjusted_rand_score(
                comparison["protein_cluster"],
                comparison["combined_cluster"],
            ),
            normalized_mutual_info_score(
                comparison["protein_cluster"],
                comparison["combined_cluster"],
            ),
        ],
    },
    index=[
        "ARI",
        "NMI",
    ],
)



scores.to_csv(
    f"{OUTPUT_DIR}/comparison/ARI_NMI_scores.csv"
)


# ------------------------------------------------------------
# CONTINGENCY HEATMAPS
# ------------------------------------------------------------

def contingency_plot(
    labels_a,
    labels_b,
    title,
    outfile,
):

    ct = pd.crosstab(
        labels_a,
        labels_b,
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="viridis",
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=300,
    )

    plt.close()


contingency_plot(
    comparison["gene_cluster"],
    comparison["protein_cluster"],
    "Gene vs Protein",
    f"{OUTPUT_DIR}/comparison/heatmaps/gene_vs_protein_heatmap.png",
)

contingency_plot(
    comparison["gene_cluster"],
    comparison["combined_cluster"],
    "Gene vs Combined",
    f"{OUTPUT_DIR}/comparison/heatmaps/gene_vs_combined_heatmap.png",
)

contingency_plot(
    comparison["protein_cluster"],
    comparison["combined_cluster"],
    "Protein vs Combined",
    f"{OUTPUT_DIR}/comparison/heatmaps/protein_vs_combined_heatmap.png",
)


# ------------------------------------------------------------
# HUNGARIAN SANKEY
# ------------------------------------------------------------

map_pg = {
    str(v): str(k)
    for k, v in map_gp.items()
}

mapped_gene_from_protein = (
    protein_labels.map(
        map_pg
    )
)

ct_pg_hungarian = pd.crosstab(
    protein_labels,
    mapped_gene_from_protein,
)

sankey_from_contingency(
    ct_pg_hungarian,
    "Protein → Gene (Hungarian)",
    f"{OUTPUT_DIR}/comparison/mapped/sankey_protein_gene_hungarian.png",
)


# ------------------------------------------------------------
# MERGED SANKEY
# ------------------------------------------------------------

ct_pg_merged = pd.crosstab(
    protein_labels,
    merged_gene_from_protein,
)

sankey_from_contingency(
    ct_pg_merged,
    "Protein → Gene (Merged)",
    f"{OUTPUT_DIR}/comparison/merged/sankey_protein_gene_merged.png",
)


# ------------------------------------------------------------
# THREE-WAY SANKEY
# ------------------------------------------------------------

sankey_three_way(
    protein_labels,
    gene_labels,
    combined_labels,
    "Protein → Gene → Combined",
    f"{OUTPUT_DIR}/comparison/sankey/sankey_protein_gene_combined.png",
)


# ------------------------------------------------------------
# ANALYSIS SUMMARY
# ------------------------------------------------------------

print(
    f"{len(gene_cols)} genes detected"
)

print(
    f"{len(protein_cols)} proteins detected"
)

if THRESHOLD_FILE is not None:
    print(
        f"Threshold file: "
        f"{THRESHOLD_FILE}"
    )

print(
    f"Gene cells: {gene_pass.sum():,}"
)

print(
    f"Protein cells: {protein_pass.sum():,}"
)

print(
    f"Combined cells: {(gene_pass & protein_pass).sum():,}"
)


#
# Analysis complete
#
print("\nAnalysis complete")

print(
    f"Gene cells: {gene_adata.n_obs:,}"
)

print(
    f"Protein cells: {protein_adata.n_obs:,}"
)

print(
    f"Combined cells: {combined_adata.n_obs:,}"
)

print(
    f"\nOutputs written to:\n{OUTPUT_DIR}"
)

