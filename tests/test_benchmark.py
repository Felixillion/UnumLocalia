"""
tests/test_benchmark.py
=======================
Unit tests for unumlocalia.benchmark
"""

import numpy as np
import pandas as pd
import pytest

from unumlocalia.benchmark import (
    compare_clusterings,
    clustering_metrics_table,
    compare_markers,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClusteringComparison:
    def test_identical_clusters(self):
        labels_a = [0, 0, 1, 1, 2, 2]
        labels_b = [0, 0, 1, 1, 2, 2]
        result = compare_clusterings(labels_a, labels_b)
        
        assert result["ari"] == pytest.approx(1.0, abs=1e-5)
        assert result["nmi"] == pytest.approx(1.0, abs=1e-5)

    def test_different_clusters(self):
        labels_a = [0, 0, 1, 1, 2, 2]
        labels_b = [1, 2, 1, 2, 1, 2]  # Poor overlap
        result = compare_clusterings(labels_a, labels_b)
        
        assert result["ari"] < 0.5
        assert result["nmi"] < 0.5

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            compare_clusterings([1, 2], [1, 2, 3])

    def test_metrics_table(self):
        result = {"ari": 0.8, "nmi": 0.7, "name_a": "A", "name_b": "B"}
        df = clustering_metrics_table(result)
        assert len(df) == 2
        assert "Metric" in df.columns
        assert "Value" in df.columns


class TestMarkerComparison:
    def test_perfect_correlation(self):
        # Create identical dataframes
        df_a = pd.DataFrame({
            "CK8": [10.0, 20.0, 30.0, 40.0],
            "CD45": [5.0, 15.0, 25.0, 35.0]
        })
        df_b = df_a.copy()
        
        result = compare_markers(df_a, df_b)
        assert len(result) == 2
        
        # Pearson and Spearman should both be 1.0
        assert (result["pearson_r"] == 1.0).all()
        assert (result["spearman_r"] == 1.0).all()
        assert (result["n_cells"] == 4).all()

    def test_numpy_arrays(self):
        arr_a = np.array([[10, 5], [20, 15], [30, 25], [40, 35]], dtype=float)
        arr_b = arr_a.copy()
        
        # Should raise error if columns not provided
        with pytest.raises(ValueError, match="columns_a must be provided"):
            compare_markers(arr_a, arr_b)
            
        result = compare_markers(
            arr_a, arr_b,
            columns_a=["CK8", "CD45"],
            columns_b=["CK8", "CD45"]
        )
        assert len(result) == 2
        assert (result["pearson_r"] == 1.0).all()

    def test_subset_markers(self):
        df_a = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        df_b = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "D": [7, 8, 9]})
        
        result = compare_markers(df_a, df_b, markers=["A"])
        assert len(result) == 1
        assert result.iloc[0]["marker"] == "A"
