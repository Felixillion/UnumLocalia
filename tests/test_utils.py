"""
tests/test_utils.py
===================
Unit tests for unumlocalia.utils
"""

import numpy as np
import pytest

from unumlocalia.utils import (
    affine_transform_coords,
    arcsinh_transform,
    log1p_norm,
    colormap_from_name,
    export_figure,
)


class TestAffineTransform:
    def test_identity(self):
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        identity = np.eye(3)
        result = affine_transform_coords(coords, identity)
        np.testing.assert_allclose(result, coords, atol=1e-6)

    def test_translation(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        matrix = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [0.0, 0.0,  1.0],
        ])
        result = affine_transform_coords(coords, matrix)
        expected = np.array([[10.0, 20.0], [11.0, 21.0]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_scaling(self):
        coords = np.array([[2.0, 4.0]])
        matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = affine_transform_coords(coords, matrix)
        expected = np.array([[4.0, 12.0]])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_wrong_coord_shape(self):
        with pytest.raises(ValueError, match="coords must be"):
            affine_transform_coords(np.ones((5, 3)), np.eye(3))

    def test_wrong_matrix_shape(self):
        with pytest.raises(ValueError, match="matrix must be"):
            affine_transform_coords(np.ones((5, 2)), np.eye(4))


class TestNormalisation:
    def test_arcsinh_zero(self):
        result = arcsinh_transform(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_arcsinh_positive(self):
        x = np.array([5.0])
        result = arcsinh_transform(x, cofactor=5.0)
        # arcsinh(5/5) = arcsinh(1) ≈ 0.881
        assert abs(result[0] - np.arcsinh(1.0)) < 1e-6

    def test_log1p_norm_shape(self):
        counts = np.array([[100, 200, 300], [50, 50, 0]], dtype=float)
        result = log1p_norm(counts, target_sum=1e4)
        assert result.shape == counts.shape

    def test_log1p_norm_zero_row(self):
        counts = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        result = log1p_norm(counts, target_sum=1e4)
        # Zero row should remain zero after normalisation
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)


class TestColormap:
    def test_known_colormap(self):
        assert colormap_from_name("red") == "red"

    def test_case_insensitive(self):
        assert colormap_from_name("GREEN") == "green"

    def test_unknown_falls_back(self):
        result = colormap_from_name("not_a_real_colormap_xyz")
        assert result == "green"


class TestExportFigure:
    def test_export_png(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = export_figure(fig, tmp_path / "test", fmt="png")
        assert out.exists()
        assert out.suffix == ".png"
        plt.close(fig)

    def test_invalid_format(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Unsupported format"):
            export_figure(fig, tmp_path / "test", fmt="bmp")
        plt.close(fig)
