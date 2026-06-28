"""
tests/test_analysis.py
======================
Unit tests for spatialbench.analysis
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spatialbench.analysis import (
    prepare_modality,
    run_pca,
    run_neighbors,
    run_umap,
    run_leiden,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_adata() -> ad.AnnData:
    """Create a mock AnnData with both genes and proteins."""
    # Gene counts (10 cells × 20 genes)
    X = sp.csr_matrix(np.random.randint(0, 10, (10, 20)).astype(np.float32))
    
    # Protein intensities (10 cells × 5 proteins)
    prot = np.random.rand(10, 5).astype(np.float32) * 100
    
    # Spatial coordinates
    spatial = np.random.rand(10, 2).astype(np.float32) * 50
    
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
    )
    adata.obsm["X_protein"] = prot
    adata.obsm["spatial"] = spatial
    adata.uns["protein_names"] = [f"prot_{i}" for i in range(5)]
    return adata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrepareModality:
    def test_genes(self, mock_adata):
        X = prepare_modality(mock_adata, modality="genes", n_top_genes=10)
        assert X.shape[0] == 10
        # Highly variable genes might be fewer than n_top_genes in random data,
        # but shouldn't exceed it.
        assert X.shape[1] <= 10

    def test_proteins(self, mock_adata):
        X = prepare_modality(mock_adata, modality="proteins")
        assert X.shape == (10, 5)

    def test_combined(self, mock_adata):
        X = prepare_modality(mock_adata, modality="combined", n_top_genes=10)
        assert X.shape[0] == 10
        # Combines HVGs + proteins
        assert X.shape[1] > 5

    def test_invalid_modality(self, mock_adata):
        with pytest.raises(ValueError, match="Unknown modality"):
            prepare_modality(mock_adata, modality="invalid")


class TestPipeline:
    def test_pca_adds_obsm(self, mock_adata):
        run_pca(mock_adata, modality="genes", n_comps=5)
        assert "X_pca_genes" in mock_adata.obsm
        assert mock_adata.obsm["X_pca_genes"].shape == (10, 5)
        assert "pca_genes" in mock_adata.uns

    def test_neighbors_adds_uns(self, mock_adata):
        # run_neighbors should automatically run PCA if missing
        run_neighbors(mock_adata, modality="proteins", n_neighbors=5, n_pcs=3)
        assert "X_pca_proteins" in mock_adata.obsm
        assert "neighbors_proteins" in mock_adata.uns
        # Scanpy's neighbors also adds connectivities to obsp
        assert "neighbors_proteins_connectivities" in mock_adata.obsp or "connectivities" in mock_adata.obsp

    def test_umap_adds_obsm(self, mock_adata):
        # run_umap should automatically run neighbors if missing
        run_umap(mock_adata, modality="combined")
        assert "X_umap_combined" in mock_adata.obsm
        assert mock_adata.obsm["X_umap_combined"].shape == (10, 2)

    def test_leiden_adds_obs(self, mock_adata):
        run_leiden(mock_adata, modality="genes", resolution=0.5)
        assert "leiden_genes" in mock_adata.obs.columns
        assert mock_adata.obs["leiden_genes"].nunique() > 0
