"""
tests/test_io.py
================
Unit tests for unumlocalia.io — file detection and manifest building.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unumlocalia.io import detect_files, DatasetManifest, load_alignment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_dataset(tmp_path) -> Path:
    """Create a minimal mock dataset folder structure."""
    # Xenium files
    pd.DataFrame({
        "cell_id": [1, 2, 3],
        "x_centroid": [100.0, 200.0, 300.0],
        "y_centroid": [150.0, 250.0, 350.0],
        "cell_area": [50.0, 60.0, 70.0],
        "transcript_counts": [10, 20, 30],
    }).to_csv(tmp_path / "cells.csv", index=False)

    pd.DataFrame({
        "cell_id": [1, 1, 2, 2],
        "vertex_x": [90.0, 110.0, 190.0, 210.0],
        "vertex_y": [140.0, 160.0, 240.0, 260.0],
    }).to_parquet(tmp_path / "cell_boundaries.parquet")

    pd.DataFrame({
        "cell_id": [1, 2],
        "vertex_x": [95.0, 195.0],
        "vertex_y": [145.0, 245.0],
    }).to_parquet(tmp_path / "nucleus_boundaries.parquet")

    pd.DataFrame({
        "x_location": [100.0, 200.0, 300.0],
        "y_location": [150.0, 250.0, 350.0],
        "feature_name": ["EPCAM", "CD3E", "EPCAM"],
    }).to_parquet(tmp_path / "transcripts.parquet")

    # Alignment matrix (identity)
    matrix = pd.DataFrame(np.eye(3))
    matrix.to_csv(tmp_path / "matrix.csv", index=False, header=False)

    # COMET — minimal valid TIFF
    try:
        import tifffile
        comet_dir = tmp_path / "comet"
        comet_dir.mkdir()
        for marker in ["CK8", "CD45"]:
            arr = np.zeros((64, 64), dtype=np.uint16)
            tifffile.imwrite(comet_dir / f"{marker}.ome.tiff", arr)
    except ImportError:
        pass  # Skip TIFF creation if tifffile not available in test env

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectFiles:
    def test_cells_detected(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        assert manifest.cells is not None
        assert manifest.cells.name == "cells.csv"

    def test_boundaries_detected(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        assert manifest.cell_boundaries is not None
        assert manifest.nucleus_boundaries is not None

    def test_transcripts_detected(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        assert manifest.transcripts is not None

    def test_matrix_detected(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        assert manifest.matrix is not None

    def test_comet_detected(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        # Only test if tifffile created the files
        if (mock_dataset / "comet" / "CK8.ome.tiff").exists():
            assert "CK8" in manifest.comet
            assert "CD45" in manifest.comet

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_files(tmp_path / "nonexistent_folder")

    def test_manifest_summary(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        summary = manifest.summary()
        assert "cells.csv" in summary
        assert "✓" in summary


class TestLoadAlignment:
    def test_identity_matrix(self, mock_dataset):
        manifest = detect_files(mock_dataset)
        matrix = load_alignment(manifest)
        assert matrix is not None
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-6)

    def test_no_matrix_file(self, tmp_path):
        manifest = DatasetManifest(folder=tmp_path)
        result = load_alignment(manifest)
        assert result is None

    def test_flat_9_values(self, tmp_path):
        """matrix.csv written as a single row of 9 values."""
        flat = pd.DataFrame([np.arange(9, dtype=float)])
        flat.to_csv(tmp_path / "matrix.csv", index=False, header=False)
        manifest = DatasetManifest(folder=tmp_path, matrix=tmp_path / "matrix.csv")
        matrix = load_alignment(manifest)
        assert matrix.shape == (3, 3)
        np.testing.assert_array_equal(matrix.flatten(), np.arange(9))
