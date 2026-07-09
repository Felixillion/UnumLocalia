"""
tests/test_segmentation.py
==========================
Unit tests for unumlocalia.segmentation
"""

import numpy as np
import pandas as pd
import pytest

from unumlocalia.segmentation import (
    _compute_label_areas,
    assign_xenium_transcripts,
    pivot_transcript_counts,
    build_anndata,
    compare_segmentations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_labels() -> np.ndarray:
    """4×4 label mask with 3 cells."""
    return np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ], dtype=np.int32)


@pytest.fixture()
def transcripts_df() -> pd.DataFrame:
    return pd.DataFrame({
        "x_location": [0.5, 2.5, 2.5, 0.5],
        "y_location": [0.5, 0.5, 2.5, 2.5],
        "feature_name": ["EPCAM", "CD3E", "EPCAM", "CD3E"],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLabelAreas:
    def test_areas_correct(self, simple_labels):
        areas = _compute_label_areas(simple_labels)
        assert areas[1] == 4
        assert areas[2] == 4
        assert areas[3] == 4
        assert 0 not in areas

    def test_empty_mask(self):
        mask = np.zeros((4, 4), dtype=np.int32)
        areas = _compute_label_areas(mask)
        assert len(areas) == 0


class TestTranscriptAssignment:
    def test_assigns_correctly(self, simple_labels, transcripts_df):
        result = assign_xenium_transcripts(simple_labels, transcripts_df)
        # (0.5, 0.5) → row 0, col 0 → label 0 (background)
        assert result.iloc[0]["cell_id"] == 0
        # (2.5, 0.5) → row 0, col 2 → label 1
        assert result.iloc[1]["cell_id"] == 1
        # (2.5, 2.5) → row 2, col 2 → label 3
        assert result.iloc[2]["cell_id"] == 3
        # (0.5, 2.5) → row 2, col 0 → label 2
        assert result.iloc[3]["cell_id"] == 2

    def test_column_added(self, simple_labels, transcripts_df):
        result = assign_xenium_transcripts(simple_labels, transcripts_df)
        assert "cell_id" in result.columns

    def test_original_unchanged(self, transcripts_df):
        original_len = len(transcripts_df)
        labels = np.zeros((10, 10), dtype=np.int32)
        assign_xenium_transcripts(labels, transcripts_df)
        assert len(transcripts_df) == original_len


class TestPivotCounts:
    def test_shape(self, simple_labels, transcripts_df):
        assigned = assign_xenium_transcripts(simple_labels, transcripts_df)
        counts = pivot_transcript_counts(assigned)
        # Cells with transcripts: 1, 2, 3
        assert counts.shape[1] == 2  # 2 genes

    def test_no_background(self, simple_labels, transcripts_df):
        assigned = assign_xenium_transcripts(simple_labels, transcripts_df)
        counts = pivot_transcript_counts(assigned)
        assert 0 not in counts.index


class TestBuildAnndata:
    def test_creates_anndata(self, simple_labels, transcripts_df):
        assigned = assign_xenium_transcripts(simple_labels, transcripts_df)
        adata = build_anndata(simple_labels, transcripts_df=assigned)
        assert adata.n_obs > 0

    def test_cell_area_computed(self, simple_labels):
        adata = build_anndata(simple_labels)
        assert "cell_area" in adata.obs.columns
        assert (adata.obs["cell_area"] > 0).all()

    def test_source_uns(self, simple_labels):
        adata = build_anndata(simple_labels)
        assert adata.uns.get("source") == "user_segmentation"

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError, match="no cell labels"):
            build_anndata(np.zeros((10, 10), dtype=np.int32))


class TestCompareSegmentations:
    def test_identical_masks(self, simple_labels):
        metrics = compare_segmentations(simple_labels, simple_labels)
        assert metrics.n_cells_a == metrics.n_cells_b == 3
        assert metrics.overlap_coefficient == pytest.approx(1.0, abs=0.01)

    def test_different_shape_raises(self, simple_labels):
        other = np.zeros((8, 8), dtype=np.int32)
        with pytest.raises(ValueError, match="same shape"):
            compare_segmentations(simple_labels, other)

    def test_no_overlap(self, simple_labels):
        # Completely different labelling
        other = np.zeros_like(simple_labels)
        other[0, 0] = 99  # one cell in background of simple_labels
        metrics = compare_segmentations(simple_labels, other)
        assert metrics.overlap_coefficient < 0.5
