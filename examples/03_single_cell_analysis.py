"""
Example 3: Single-cell analysis — PCA, UMAP, Leiden clustering, and plots.

This example works on the reference AnnData supplied with the dataset,
or on an AnnData built from a user segmentation (see Example 2).

Usage
-----
    conda activate unumlocalia
    python examples/03_single_cell_analysis.py /path/to/dataset
"""

import sys
from pathlib import Path


def main(dataset_folder: str) -> None:
    from unumlocalia.io import DatasetLoader
    from unumlocalia.analysis import (
        run_pca,
        run_neighbors,
        run_umap,
        run_leiden,
        plot_pca_variance,
        plot_umap,
        plot_spatial_clusters,
        plot_heatmap,
        plot_dotplot,
    )
    from unumlocalia.benchmark import (
        compare_clusterings,
        clustering_metrics_table,
        plot_contingency,
    )
    from unumlocalia.utils import export_figure, export_dataframe

    out_dir = Path(dataset_folder) / "analysis_output"
    out_dir.mkdir(exist_ok=True)

    # ---- Load dataset -------------------------------------------------------
    print("Loading dataset…")
    loader = DatasetLoader(dataset_folder)
    loader.load()

    adata = loader.anndata_ref
    if adata is None:
        raise FileNotFoundError(
            "No .h5ad file found. Provide a dataset with a reference AnnData."
        )
    # Back-to-memory (unbacked) for in-memory operations
    adata = adata.to_memory()
    print(adata)

    # ====================================================================
    # GENES modality
    # ====================================================================
    print("\n=== Gene expression analysis ===")

    run_pca(adata, modality="genes", n_comps=50)
    fig_scree = plot_pca_variance(adata, modality="genes", n_comps=20)
    export_figure(fig_scree, out_dir / "pca_scree_genes.png")

    run_neighbors(adata, modality="genes", n_neighbors=15)
    run_umap(adata, modality="genes")
    run_leiden(adata, modality="genes", resolution=0.5)

    fig_umap_genes = plot_umap(adata, color="leiden_genes", modality="genes")
    export_figure(fig_umap_genes, out_dir / "umap_leiden_genes.png")

    # Spatial cluster map
    if "spatial" in adata.obsm:
        fig_spatial = plot_spatial_clusters(adata, color="leiden_genes")
        export_figure(fig_spatial, out_dir / "spatial_leiden_genes.png")

    # Export cluster assignments
    export_dataframe(
        adata.obs[["leiden_genes"]],
        out_dir / "leiden_genes.csv",
    )
    print(f"Gene clusters: {adata.obs['leiden_genes'].nunique()} clusters")

    # ====================================================================
    # PROTEINS modality (if available)
    # ====================================================================
    if "X_protein" in adata.obsm:
        print("\n=== Protein expression analysis ===")

        run_pca(adata, modality="proteins", n_comps=15)
        run_neighbors(adata, modality="proteins", n_neighbors=10)
        run_umap(adata, modality="proteins")
        run_leiden(adata, modality="proteins", resolution=0.5)

        fig_umap_prot = plot_umap(adata, color="leiden_proteins", modality="proteins")
        export_figure(fig_umap_prot, out_dir / "umap_leiden_proteins.png")

        print(f"Protein clusters: {adata.obs['leiden_proteins'].nunique()} clusters")

        # ---- Clustering comparison ----------------------------------------
        print("\n=== Comparing gene vs protein clusters ===")
        results = compare_clusterings(
            adata.obs["leiden_genes"],
            adata.obs["leiden_proteins"],
            name_a="Gene clusters",
            name_b="Protein clusters",
        )
        print(clustering_metrics_table(results).to_string(index=False))

        fig_contingency = plot_contingency(
            adata.obs["leiden_genes"],
            adata.obs["leiden_proteins"],
            name_a="Gene clusters",
            name_b="Protein clusters",
        )
        export_figure(fig_contingency, out_dir / "contingency_genes_vs_proteins.png")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_single_cell_analysis.py /path/to/dataset")
        sys.exit(1)
    main(sys.argv[1])
