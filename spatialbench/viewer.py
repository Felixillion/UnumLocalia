"""
spatialbench.viewer
===================
Napari-based spatial viewer with programmatic layer management for multi-core datasets.

The :class:`SpatialViewer` wraps a ``napari.Viewer`` instance and provides
typed methods for adding each data layer type.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default display settings
_DEFAULT_HE_COLORMAP = "gray"
_DEFAULT_TRANSCRIPT_SIZE = 4
_DEFAULT_BOUNDARY_WIDTH = 1


class SpatialViewer:
    """Managed wrapper around a ``napari.Viewer`` for SpatialBench."""

    def __init__(self, title: str = "SpatialBench") -> None:
        import napari  # deferred import

        self._viewer = napari.Viewer(title=title, show=False)
        self._active_core: Optional[str] = None
        logger.info("napari viewer created.")

    # ------------------------------------------------------------------ #
    # Core napari viewer access
    # ------------------------------------------------------------------ #

    @property
    def viewer(self):
        return self._viewer
        
    @property
    def active_core(self) -> Optional[str]:
        return self._active_core
        
    def set_active_core(self, core: str) -> None:
        """Switch the active core. Hides layers from other cores."""
        self._active_core = core
        for layer in self._viewer.layers:
            layer_core = layer.metadata.get("core")
            if layer_core is None:
                continue
                
            if layer_core != core:
                # Store the visibility state if it was visible
                if layer.visible:
                    layer.metadata["user_visible"] = True
                layer.visible = False
            else:
                # Restore the user's preferred visibility
                layer.visible = layer.metadata.get("user_visible", layer.visible)
                
        # Reset view so it zooms to the new core's data
        self.reset_view()

    def show(self) -> None:
        self._viewer.window.show()

    def run(self) -> None:
        import napari
        napari.run()

    # ------------------------------------------------------------------ #
    # Layer Add/Update Helpers
    # ------------------------------------------------------------------ #

    def _convert_affine(self, affine: Optional[np.ndarray]) -> Any:
        if affine is None:
            return None
        
        # napari affine takes N+1 x N+1 matrix for N dimensions. 2D image = 3x3
        try:
            if affine.shape == (2, 3):
                return np.vstack([affine, [0.0, 0.0, 1.0]])
            if affine.shape == (3, 3):
                return affine
        except Exception as e:
            logger.warning("Could not parse affine matrix: %s", e)
        return None

    # ------------------------------------------------------------------ #
    # H&E layer
    # ------------------------------------------------------------------ #

    def add_he_layer(
        self,
        core: str,
        array: np.ndarray,
        name: Optional[str] = None,
        opacity: float = 1.0,
        gamma: float = 1.0,
        brightness: float = 0.0,
        affine: Optional[np.ndarray] = None,
        visible: bool = True,
    ):
        name = name or f"{core}::he"
        self._remove_layer_if_exists(name)

        is_rgb = array.ndim == 3 and array.shape[2] in (3, 4)
        
        # If this isn't the active core, default to invisible
        is_visible = visible if (self._active_core is None or self._active_core == core) else False

        layer = self._viewer.add_image(
            array,
            name=name,
            rgb=is_rgb,
            opacity=opacity,
            gamma=gamma,
            colormap=_DEFAULT_HE_COLORMAP if not is_rgb else None,
            blending="translucent",
            affine=self._convert_affine(affine),
            visible=is_visible
        )
        # store metadata so other code can find it
        layer.metadata.update({
            "core": core,
            "modality": "he",
            "user_visible": visible
        })

        # --- NEW: persist the affine matrix in metadata so callers can read the exact matrix Napari uses
        try:
            if affine is not None:
                # convert to 3x3 homogeneous if needed
                M = np.asarray(affine, dtype=float)
                if M.shape == (2, 3):
                    M = np.vstack([M, [0.0, 0.0, 1.0]])
                elif M.shape != (3, 3):
                    M = M.reshape(3, 3)
                layer.metadata["affine_matrix"] = M.tolist()
        except Exception:
            # don't crash on metadata write
            pass

        if not is_rgb and brightness != 0.0:
            lo, hi = layer.contrast_limits
            shift = brightness * (hi - lo)
            layer.contrast_limits = (lo - shift, hi - shift)

        return layer

    # ------------------------------------------------------------------ #
    # COMET layers
    # ------------------------------------------------------------------ #

    def add_comet_layer(
        self,
        core: str,
        marker_name: str,
        array: np.ndarray,
        colormap: str = "green",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        opacity: float = 0.8,
        visible: bool = True,
        affine: Optional[np.ndarray] = None,
    ):
        from spatialbench.utils import colormap_from_name

        layer_name = f"{core}::comet::{marker_name}"
        self._remove_layer_if_exists(layer_name)

        cmap = colormap_from_name(colormap)
        is_visible = visible if (self._active_core is None or self._active_core == core) else False

        layer = self._viewer.add_image(
            array,
            name=layer_name,
            colormap=cmap,
            opacity=opacity,
            visible=is_visible,
            blending="additive",
            affine=self._convert_affine(affine),
        )
        
        layer.metadata.update({
            "core": core,
            "modality": "comet",
            "marker": marker_name,
            "user_visible": visible
        })

        # persist the affine matrix in metadata so callers can read the exact matrix Napari uses
        try:
            if affine is not None:
                M = np.asarray(affine, dtype=float)
                if M.shape == (2, 3):
                    M = np.vstack([M, [0.0, 0.0, 1.0]])
                elif M.shape != (3, 3):
                    M = M.reshape(3, 3)
                layer.metadata["affine_matrix"] = M.tolist()
        except Exception:
            pass

        if vmin is not None or vmax is not None:
            lo = vmin if vmin is not None else float(array.min())
            hi = vmax if vmax is not None else float(array.max())
            layer.contrast_limits = (lo, hi)

        return layer

    def update_comet_layer(
        self,
        marker_name: str,
        core: Optional[str] = None,
        colormap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        opacity: Optional[float] = None,
        visible: Optional[bool] = None,
    ) -> None:
        from spatialbench.utils import colormap_from_name
        
        cores_to_update = [core] if core else [lyr.metadata.get("core") for lyr in self._viewer.layers if lyr.metadata.get("marker") == marker_name]

        for c in cores_to_update:
            layer = self._get_layer(f"{c}::comet::{marker_name}")
            if layer is None:
                continue

            if colormap is not None:
                layer.colormap = colormap_from_name(colormap)
            if vmin is not None or vmax is not None:
                lo = vmin if vmin is not None else layer.contrast_limits[0]
                hi = vmax if vmax is not None else layer.contrast_limits[1]
                layer.contrast_limits = (lo, hi)
            if opacity is not None:
                layer.opacity = opacity
            if visible is not None:
                layer.metadata["user_visible"] = visible
                if self._active_core is None or self._active_core == c:
                    layer.visible = visible


    # ------------------------------------------------------------------ #
    # Transcript layers
    # ------------------------------------------------------------------ #

    def add_transcript_layer(
        self,
        core: str,
        gene: str,
        coords: np.ndarray,
        color: Union[str, Sequence[float]] = "yellow",
        size: float = _DEFAULT_TRANSCRIPT_SIZE,
        opacity: float = 0.8,
        visible: bool = True,
        affine: Optional[np.ndarray] = None,
    ):
        layer_name = f"{core}::transcripts::{gene}"
        self._remove_layer_if_exists(layer_name)

        # Napari expects (row, col) == (y, x)
        yx = coords[:, ::-1] if coords.shape[1] == 2 else coords

        layer = self._viewer.add_points(
            yx,
            name=layer_name,
            face_color=color,
            size=size,
            opacity=opacity,
            visible=visible,
            metadata={
                "core": core,
                "marker": gene,
                "modality": "xenium",
                "sb_source": f"transcripts::{gene}",
            },
        )

        if affine is not None:
            napari_affine = self._convert_affine(np.asarray(affine, dtype=float))
            if napari_affine is not None:
                layer.affine = napari_affine

        return layer


    def update_transcript_layer(
        self,
        gene: str,
        core: Optional[str] = None,
        color: Optional[Union[str, Sequence[float]]] = None,
        size: Optional[float] = None,
        opacity: Optional[float] = None,
        visible: Optional[bool] = None,
    ) -> None:
        cores_to_update = [core] if core else [lyr.metadata.get("core") for lyr in self._viewer.layers if lyr.metadata.get("marker") == gene and lyr.metadata.get("modality") == "xenium"]

        for c in cores_to_update:
            layer = self._get_layer(f"{c}::transcripts::{gene}")
            if layer is None:
                continue

            if color is not None:
                layer.face_color = color
            if size is not None:
                layer.size = size
            if opacity is not None:
                layer.opacity = opacity
            if visible is not None:
                layer.metadata["user_visible"] = visible
                if self._active_core is None or self._active_core == c:
                    layer.visible = visible


    def set_transcript_size(self, size: float) -> None:
        """Global update for all transcript layers."""
        for layer in self._viewer.layers:
            if layer.metadata.get("modality") == "xenium":
                layer.size = size


    # ------------------------------------------------------------------ #
    # Boundary layers
    # ------------------------------------------------------------------ #

    def add_boundary_layer(
        self,
        core: str,
        shapes: List[np.ndarray],
        name: str = "boundaries",
        color: Union[str, Sequence[float]] = "white",
        width: float = _DEFAULT_BOUNDARY_WIDTH,
        opacity: float = 0.7,
        visible: bool = True,
        affine: Optional[np.ndarray] = None,
    ):
        layer_name = f"{core}::{name}"
        self._remove_layer_if_exists(layer_name)

        is_visible = visible if (self._active_core is None or self._active_core == core) else False


        ## DEBUG
        print(
            "ADDING BOUNDARY:",
            layer_name,
            "visible=",
            is_visible
        )


        layer = self._viewer.add_shapes(
            shapes,
            shape_type="path",
            name=layer_name,
            edge_color=color,
            # face_color="transparent",
            edge_width=width,
            opacity=opacity,
            visible=is_visible,
            affine=self._convert_affine(affine),
        )


        ## DEBUG
        print(
            "BOUNDARY SHAPES:",
            len(layer.data)
        )

        print(
            "FIRST SHAPE:",
            layer.data[0][:5]
        )

        
        layer.metadata.update({
            "core": core,
            "modality": "boundary",
            "marker": name,
            "user_visible": visible
        })
        
        return layer

    # ------------------------------------------------------------------ #
    # Cell inspector
    # ------------------------------------------------------------------ #

    def add_label_layer(
        self,
        core: str,
        labels: np.ndarray,
        name: Optional[str] = None,
        visible: bool = True,
        affine: Optional[np.ndarray] = None,
        color_map: Optional[str] = "glasbey",
    ):
        """
        Add a labels image (integer mask) for a core. Stores mapping in metadata.
        """
        name = name or f"{core}::masks"
        self._remove_layer_if_exists(name)

        # ensure integer dtype
        labels = np.asarray(labels)
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int32)

        is_visible = visible if (self._active_core is None or self._active_core == core) else False

        layer = self._viewer.add_labels(
            labels,
            name=name,
            visible=is_visible,
            affine=self._convert_affine(affine),
        )

        # store metadata so other code can find it
        layer.metadata.update({
            "core": core,
            "modality": "mask",
            "user_visible": visible,
            "label_dtype": str(labels.dtype),
        })

        # apply a categorical colormap if available
        try:
            from napari.utils.colormaps import Colormap
            # napari has built-in categorical maps; if not, leave default
            # keep this minimal — user can change in UI
        except Exception:
            pass

        return layer

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
    # Layer management helpers
    # ------------------------------------------------------------------ #

    def _get_layer(self, name: str):
        try:
            return self._viewer.layers[name]
        except KeyError:
            return None

    def _remove_layer_if_exists(self, name: str) -> None:
        layer = self._get_layer(name)
        if layer is not None:
            self._viewer.layers.remove(layer)

    def clear_all_layers(self) -> None:
        self._viewer.layers.clear()

    def reset_view(self) -> None:
        self._viewer.reset_view()
