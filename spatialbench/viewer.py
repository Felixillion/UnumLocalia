"""
spatialbench.viewer
===================
Napari-based spatial viewer with programmatic layer management.

The :class:`SpatialViewer` wraps a ``napari.Viewer`` instance and provides
typed methods for adding each data layer type.  It is *not* responsible for
any GUI widgets — those live in ``widgets.py``.

All data added to the viewer is passed by reference; no copies are made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default display settings
_DEFAULT_HE_COLORMAP = "gray"
_DEFAULT_TRANSCRIPT_SIZE = 4
_DEFAULT_BOUNDARY_WIDTH = 1


class SpatialViewer:
    """Managed wrapper around a ``napari.Viewer`` for SpatialBench.

    Parameters
    ----------
    title:
        Window title for the napari viewer.

    Examples
    --------
    >>> sv = SpatialViewer()
    >>> sv.add_he_layer(he_array)
    >>> sv.add_comet_layer("CK8", ck8_array)
    >>> sv.add_transcript_layer("EPCAM", coords_array)
    >>> sv.run()
    """

    def __init__(self, title: str = "SpatialBench") -> None:
        import napari  # deferred import so the module loads without napari

        self._viewer = napari.Viewer(title=title, show=False)
        logger.info("napari viewer created.")

    # ------------------------------------------------------------------ #
    # Core napari viewer access
    # ------------------------------------------------------------------ #

    @property
    def viewer(self):
        """The underlying ``napari.Viewer`` instance."""
        return self._viewer

    def show(self) -> None:
        """Make the napari window visible."""
        self._viewer.window.show()

    def run(self) -> None:
        """Enter the napari event loop (blocks until window is closed)."""
        import napari
        napari.run()

    # ------------------------------------------------------------------ #
    # H&E layer
    # ------------------------------------------------------------------ #

    def add_he_layer(
        self,
        array: np.ndarray,
        name: str = "H&E",
        opacity: float = 1.0,
        gamma: float = 1.0,
        brightness: float = 0.0,
    ):
        """Add or replace the H&E image layer.

        Parameters
        ----------
        array:
            2-D (greyscale) or 3-D (H, W, C) RGB image array.
        name:
            Layer name shown in the napari layer list.
        opacity:
            Layer opacity in [0, 1].
        gamma:
            Gamma correction value.
        brightness:
            Additive brightness offset applied as ``contrast_limits``.

        Returns
        -------
        napari.layers.Image
        """
        self._remove_layer_if_exists(name)

        # Determine if RGB
        is_rgb = array.ndim == 3 and array.shape[2] in (3, 4)

        layer = self._viewer.add_image(
            array,
            name=name,
            rgb=is_rgb,
            opacity=opacity,
            gamma=gamma,
            colormap=_DEFAULT_HE_COLORMAP if not is_rgb else None,
            blending="translucent",
        )

        # Apply brightness as a contrast limit shift
        if not is_rgb and brightness != 0.0:
            lo, hi = layer.contrast_limits
            shift = brightness * (hi - lo)
            layer.contrast_limits = (lo - shift, hi - shift)

        logger.info("Added H&E layer: shape=%s", array.shape)
        return layer

    def update_he_settings(
        self,
        opacity: Optional[float] = None,
        gamma: Optional[float] = None,
        brightness: Optional[float] = None,
        name: str = "H&E",
    ) -> None:
        """Update display settings on the H&E layer without re-adding it.

        Parameters
        ----------
        opacity:
            New opacity value.
        gamma:
            New gamma value.
        brightness:
            New brightness offset (relative to current contrast limits).
        name:
            Layer name to locate.
        """
        layer = self._get_layer(name)
        if layer is None:
            logger.warning("H&E layer '%s' not found.", name)
            return
        if opacity is not None:
            layer.opacity = opacity
        if gamma is not None:
            layer.gamma = gamma
        if brightness is not None:
            lo, hi = layer.contrast_limits
            span = hi - lo
            shift = brightness * span
            layer.contrast_limits = (lo - shift, hi - shift)

    # ------------------------------------------------------------------ #
    # COMET layers
    # ------------------------------------------------------------------ #

    def add_comet_layer(
        self,
        marker_name: str,
        array: np.ndarray,
        colormap: str = "green",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        opacity: float = 0.8,
        visible: bool = True,
    ):
        """Add or replace a COMET protein marker image layer.

        Parameters
        ----------
        marker_name:
            Protein name (e.g. ``'CK8'``).  Used as the layer name.
        array:
            2-D intensity image.
        colormap:
            Napari colormap name (e.g. ``'red'``, ``'cyan'``).
        vmin, vmax:
            Contrast limits.  If ``None``, napari auto-computes them.
        opacity:
            Layer opacity in [0, 1].
        visible:
            Whether the layer is initially visible.

        Returns
        -------
        napari.layers.Image
        """
        from spatialbench.utils import colormap_from_name

        layer_name = f"COMET: {marker_name}"
        self._remove_layer_if_exists(layer_name)

        cmap = colormap_from_name(colormap)

        layer = self._viewer.add_image(
            array,
            name=layer_name,
            colormap=cmap,
            opacity=opacity,
            visible=visible,
            blending="additive",
        )

        if vmin is not None or vmax is not None:
            lo = vmin if vmin is not None else float(array.min())
            hi = vmax if vmax is not None else float(array.max())
            layer.contrast_limits = (lo, hi)

        logger.info(
            "Added COMET layer '%s': colormap=%s vmin=%s vmax=%s",
            marker_name, cmap, vmin, vmax,
        )
        return layer

    def update_comet_layer(
        self,
        marker_name: str,
        colormap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        opacity: Optional[float] = None,
        visible: Optional[bool] = None,
    ) -> None:
        """Update display settings on an existing COMET layer.

        Parameters
        ----------
        marker_name:
            Protein name as used when the layer was added.
        colormap:
            New colormap name.
        vmin, vmax:
            New contrast limits.
        opacity:
            New opacity.
        visible:
            New visibility.
        """
        from spatialbench.utils import colormap_from_name

        layer = self._get_layer(f"COMET: {marker_name}")
        if layer is None:
            logger.warning("COMET layer '%s' not found.", marker_name)
            return

        if colormap is not None:
            layer.colormap = colormap_from_name(colormap)
        if vmin is not None or vmax is not None:
            lo = vmin if vmin is not None else layer.contrast_limits[0]
            hi = vmax if vmax is not None else layer.contrast_limits[1]
            layer.contrast_limits = (lo, hi)
        if opacity is not None:
            layer.opacity = opacity
        if visible is not None:
            layer.visible = visible

    # ------------------------------------------------------------------ #
    # Transcript layers
    # ------------------------------------------------------------------ #

    def add_transcript_layer(
        self,
        gene: str,
        coords: np.ndarray,
        color: Union[str, Sequence[float]] = "yellow",
        size: float = _DEFAULT_TRANSCRIPT_SIZE,
        opacity: float = 0.8,
        visible: bool = True,
    ):
        """Add transcript dots for one gene.

        Parameters
        ----------
        gene:
            Gene name, used as the layer name prefix.
        coords:
            Array of shape ``(N, 2)`` with columns ``[x, y]``.
            **Note**: napari Points expects ``[row, col]`` i.e. ``[y, x]``.
        color:
            Dot colour (named colour string or RGBA sequence).
        size:
            Dot diameter in data units.
        opacity:
            Layer opacity in [0, 1].
        visible:
            Whether the layer starts visible.

        Returns
        -------
        napari.layers.Points
        """
        layer_name = f"Transcripts: {gene}"
        self._remove_layer_if_exists(layer_name)

        # Napari expects (row, col) == (y, x)
        yx = coords[:, ::-1] if coords.shape[1] == 2 else coords

        layer = self._viewer.add_points(
            yx,
            name=layer_name,
            face_color=color,
            edge_color="transparent",
            size=size,
            opacity=opacity,
            visible=visible,
            blending="translucent",
        )
        logger.info(
            "Added transcript layer '%s': %d points, colour=%s",
            gene, len(coords), color,
        )
        return layer

    def remove_transcript_layer(self, gene: str) -> None:
        """Remove the transcript layer for a specific gene."""
        self._remove_layer_if_exists(f"Transcripts: {gene}")

    def update_transcript_layer(
        self,
        gene: str,
        color: Optional[Union[str, Sequence[float]]] = None,
        size: Optional[float] = None,
        opacity: Optional[float] = None,
        visible: Optional[bool] = None,
    ) -> None:
        """Update display settings on an existing transcript layer."""
        layer = self._get_layer(f"Transcripts: {gene}")
        if layer is None:
            logger.warning("Transcript layer for gene '%s' not found.", gene)
            return
        if color is not None:
            layer.face_color = color
        if size is not None:
            layer.size = size
        if opacity is not None:
            layer.opacity = opacity
        if visible is not None:
            layer.visible = visible

    # ------------------------------------------------------------------ #
    # Boundary layers
    # ------------------------------------------------------------------ #

    def add_boundary_layer(
        self,
        shapes: List[np.ndarray],
        name: str = "Cell boundaries",
        color: Union[str, Sequence[float]] = "white",
        width: float = _DEFAULT_BOUNDARY_WIDTH,
        opacity: float = 0.7,
        visible: bool = True,
    ):
        """Add polygon boundary shapes (cell or nucleus).

        Parameters
        ----------
        shapes:
            List of ``(N, 2)`` arrays in ``[y, x]`` (row, col) order,
            as returned by :func:`spatialbench.utils.shapes_to_napari`.
        name:
            Layer name.
        color:
            Edge colour.
        width:
            Line width in display units.
        opacity:
            Layer opacity.
        visible:
            Initial visibility.

        Returns
        -------
        napari.layers.Shapes
        """
        self._remove_layer_if_exists(name)

        layer = self._viewer.add_shapes(
            shapes,
            shape_type="polygon",
            name=name,
            edge_color=color,
            face_color="transparent",
            edge_width=width,
            opacity=opacity,
            visible=visible,
        )
        logger.info("Added boundary layer '%s': %d shapes.", name, len(shapes))
        return layer

    def update_boundary_layer(
        self,
        name: str,
        color: Optional[Union[str, Sequence[float]]] = None,
        width: Optional[float] = None,
        opacity: Optional[float] = None,
        visible: Optional[bool] = None,
    ) -> None:
        """Update display settings on a boundary shapes layer."""
        layer = self._get_layer(name)
        if layer is None:
            logger.warning("Boundary layer '%s' not found.", name)
            return
        if color is not None:
            layer.edge_color = color
        if width is not None:
            layer.edge_width = width
        if opacity is not None:
            layer.opacity = opacity
        if visible is not None:
            layer.visible = visible

    # ------------------------------------------------------------------ #
    # Segmentation label layer
    # ------------------------------------------------------------------ #

    def add_segmentation_layer(
        self,
        labels: np.ndarray,
        name: str = "Uploaded segmentation",
        opacity: float = 0.5,
        visible: bool = True,
    ):
        """Add a label mask as a napari Labels layer.

        Parameters
        ----------
        labels:
            2-D integer array; 0 = background, N > 0 = cell N.
        name:
            Layer name.
        opacity:
            Layer opacity.
        visible:
            Initial visibility.

        Returns
        -------
        napari.layers.Labels
        """
        self._remove_layer_if_exists(name)

        layer = self._viewer.add_labels(
            labels,
            name=name,
            opacity=opacity,
            visible=visible,
        )
        logger.info(
            "Added segmentation layer '%s': %d unique labels.",
            name, len(np.unique(labels)) - 1,
        )
        return layer

    # ------------------------------------------------------------------ #
    # Cell inspector
    # ------------------------------------------------------------------ #

    def get_cell_info_at(
        self,
        x: float,
        y: float,
        cells_df: pd.DataFrame,
        transcripts_df: Optional[pd.DataFrame] = None,
        comet_intensities: Optional[pd.DataFrame] = None,
        x_col: str = "x_centroid",
        y_col: str = "y_centroid",
        id_col: str = "cell_id",
        radius: float = 20.0,
    ) -> Optional[Dict[str, Any]]:
        """Find the nearest cell to a clicked coordinate and return its metadata.

        Parameters
        ----------
        x, y:
            Click coordinates in data (image) space.
        cells_df:
            Cell metadata DataFrame (must contain x/y centroid columns).
        transcripts_df:
            Optional transcripts DataFrame to count per-cell transcripts.
        comet_intensities:
            Optional DataFrame with per-cell protein intensities.
        x_col, y_col:
            Column names for centroid coordinates.
        id_col:
            Column name for cell identifier.
        radius:
            Maximum distance in data units to search for a cell.

        Returns
        -------
        dict or None
            Keys: ``cell_id``, ``x``, ``y``, ``area``, ``transcript_count``,
            ``proteins`` (dict), ``cluster`` (if present), or ``None`` if no
            cell is within *radius*.
        """
        if cells_df is None or len(cells_df) == 0:
            return None

        dists = np.hypot(
            cells_df[x_col].to_numpy() - x,
            cells_df[y_col].to_numpy() - y,
        )
        idx = int(np.argmin(dists))
        if dists[idx] > radius:
            return None

        row = cells_df.iloc[idx]
        cell_id = row[id_col]

        info: Dict[str, Any] = {
            "cell_id": cell_id,
            "x": float(row.get(x_col, np.nan)),
            "y": float(row.get(y_col, np.nan)),
            "area": float(row.get("cell_area", row.get("area", np.nan))),
            "transcript_count": int(
                row.get("transcript_counts", row.get("transcript_count", 0))
            ),
            "cluster": row.get("leiden", row.get("cluster", None)),
            "proteins": {},
        }

        # Protein intensities
        if comet_intensities is not None and id_col in comet_intensities.columns:
            match = comet_intensities.loc[
                comet_intensities[id_col] == cell_id
            ]
            if not match.empty:
                prot_row = match.iloc[0]
                info["proteins"] = {
                    col: float(prot_row[col])
                    for col in prot_row.index
                    if col != id_col
                }

        return info

    # ------------------------------------------------------------------ #
    # ROI export
    # ------------------------------------------------------------------ #

    def export_roi_screenshot(
        self,
        output_path: Union[str, Path],
        fmt: str = "png",
        canvas_only: bool = True,
        scale: float = 2.0,
    ) -> Path:
        """Export the current viewer canvas as an image file.

        Parameters
        ----------
        output_path:
            Destination file path.
        fmt:
            Output format: ``'png'``, ``'svg'``, or ``'pdf'``.
        canvas_only:
            If ``True``, export only the canvas (no UI chrome).
        scale:
            Pixel scale factor for raster export (higher = more detail).

        Returns
        -------
        Path
            Saved file path.
        """
        from spatialbench.utils import export_figure

        output_path = Path(output_path)
        fmt = fmt.lower().lstrip(".")

        if fmt == "png":
            screenshot = self._viewer.screenshot(
                canvas_only=canvas_only, scale=scale
            )
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(
                figsize=(screenshot.shape[1] / 72, screenshot.shape[0] / 72),
                dpi=72,
            )
            ax.imshow(screenshot)
            ax.axis("off")
            saved = export_figure(fig, output_path, fmt="png", dpi=150)
            plt.close(fig)
        elif fmt in ("svg", "pdf"):
            # For vector formats, render screenshot into matplotlib and export
            screenshot = self._viewer.screenshot(canvas_only=canvas_only, scale=2.0)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(
                figsize=(8, 8 * screenshot.shape[0] / screenshot.shape[1])
            )
            ax.imshow(screenshot)
            ax.axis("off")
            saved = export_figure(fig, output_path, fmt=fmt)
            plt.close(fig)
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use png, svg, or pdf.")

        logger.info("ROI exported to %s", saved)
        return saved

    # ------------------------------------------------------------------ #
    # Layer management helpers
    # ------------------------------------------------------------------ #

    def _get_layer(self, name: str):
        """Return a napari layer by name, or ``None`` if not found."""
        try:
            return self._viewer.layers[name]
        except KeyError:
            return None

    def _remove_layer_if_exists(self, name: str) -> None:
        """Remove a layer by name if it exists (for replace semantics)."""
        layer = self._get_layer(name)
        if layer is not None:
            self._viewer.layers.remove(layer)
            logger.debug("Removed existing layer '%s'.", name)

    def layer_names(self) -> List[str]:
        """Return the names of all currently loaded layers."""
        return [layer.name for layer in self._viewer.layers]

    def clear_all_layers(self) -> None:
        """Remove all layers from the viewer."""
        self._viewer.layers.clear()
        logger.info("All layers cleared.")

    def reset_view(self) -> None:
        """Reset the camera to fit all visible layers."""
        self._viewer.reset_view()
