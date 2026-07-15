# unumlocalia/widgets.py
"""
PyQt-based dock widget panels that integrate with the Napari viewer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from unumlocalia.utils import safe_read_parquet, shapes_to_napari

# Save image
from imageio import imwrite

# Scale bar imports
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# CSV export and session saving/loading
from qtpy.QtWidgets import QFileDialog

# JSON export for threshold data
import json

# Suppress warnings
import warnings

logger = logging.getLogger(__name__)

try:
    from qtpy import QtWidgets
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtGui import QFont, QColor
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    _QT_AVAILABLE = True
except Exception:
    _QT_AVAILABLE = False
    logger.warning("Qt not available. UnumLocalia GUI cannot be displayed.")


## Genes for side bar
DEFAULT_FAVOURITE_GENES = [
    "CD3E",
    "CD4",
    "CD8A",
    "MS4A1",
    "EPCAM",
    "MKI67",
    "PECAM1",
]


def _clear_layout(widget: QtWidgets.QWidget) -> None:
    """Remove all children widgets from a container safely."""
    layout = widget.layout()
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()


class CollapsibleGroup(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        # Have tick box for sections
        # self.setCheckable(True)
        # self.setChecked(True)
        # self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        for child in self.findChildren(QWidget):
            if child is not self:
                child.setVisible(checked)


# ---------- Cell mask class ----------
class CellBoundaryRow(QWidget):
    """
    Minimal cell-mask control: checkbox + info label.
    Loads rasterized mask synchronously via loader.load_geojson_mask(core).
    """

    def __init__(self, sv, loader, display_name="Xenium", layer_suffix="cells", parent=None):
        super().__init__(parent)
        self.sv = sv
        self.loader = loader

        main_layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()

        main_layout.addLayout(row1)
        main_layout.addLayout(row2)

        # Row options
        row1.setContentsMargins(0, 2, 0, 2)
        row1.setSpacing(6)

        row2.setContentsMargins(0, 2, 0, 2)
        row2.setSpacing(6)


        self.chk = QCheckBox(display_name)
        self.chk.setChecked(False)
        self.chk.toggled.connect(self._on_toggle)

        self.display_name = display_name
        self.layer_suffix = layer_suffix

        # Cell border colour
        self.color = "white"

        self.color_btn = QPushButton("■")
        self.color_btn.setFixedWidth(30)
        self.color_btn.setStyleSheet(
            f"color: {self.color};"
            "font-weight: bold;"
            "font-size: 16px;"
        )
        self.color_btn.clicked.connect(
            self._pick_color
        )

        self.info_lbl = QLabel("")
        self.info_lbl.setStyleSheet("color: gray;")

        # Boundary controls (row 1)
        row1.addWidget(self.chk)
        row1.addWidget(self.color_btn)
        row1.addWidget(self.info_lbl)
        row1.addStretch()

        # Cell fill
        self.fill_chk = QCheckBox("Fill")
        self.fill_chk.toggled.connect(
            self._update_fill
        )

        # Cell fill opacity slider
        self.opacity_sl = QSlider(Qt.Horizontal)

        self.opacity_sl.setRange(0, 100)
        self.opacity_sl.setValue(20)

        self.opacity_sl.valueChanged.connect(
            self._update_fill
        )

        # Add separate fill colour (to border)
        self.fill_color = "#ffff00"

        self.fill_color_btn = QPushButton("■")
        self.fill_color_btn.setFixedWidth(30)

        self.fill_color_btn.setStyleSheet(
            f"color: {self.fill_color};"
            "font-weight: bold;"
            "font-size: 16px;"
        )

        self.fill_color_btn.clicked.connect(
            self._pick_fill_color
        )

        # Fill controls (row 2)
        row2.addWidget(self.fill_chk)
        row2.addWidget(self.fill_color_btn)
        row2.addWidget(QLabel("α"))
        row2.addWidget(self.opacity_sl)
        row2.addStretch()

        self.fill_chk.setEnabled(False)
        self.opacity_sl.setEnabled(False)
        self.fill_color_btn.setEnabled(False)


    def sync_to_core(self, core: str):
        """Update checkbox state when core changes."""
        if not core:
            self.chk.setEnabled(False)
            self.info_lbl.setText("")
            return

        self.chk.setEnabled(True)
        lname = f"{core}::{self.layer_suffix}"
        layer = None
        try:
            layer = self.sv._get_layer(
                f"{core}::{self.layer_suffix}"
            )
        except Exception:
            layer = None

        if layer is not None:

            self.chk.blockSignals(True)

            self.chk.setChecked(
                bool(layer.visible)
            )

            self.chk.blockSignals(False)

            self.info_lbl.setText(
                f"cells: {len(layer.data)}"
            )

        else:
            self.chk.blockSignals(True)
            self.chk.setChecked(False)
            self.chk.blockSignals(False)

            self.fill_chk.blockSignals(True)
            self.fill_chk.setChecked(False)
            self.fill_chk.blockSignals(False)

            self.fill_chk.setEnabled(False)
            self.opacity_sl.setEnabled(False)
            self.fill_color_btn.setEnabled(False)

            if hasattr(self, "layers_tab"):
                if self.layers_tab.cell_boundaries_visible:
                    self.info_lbl.setText(
                        "saved in session"
                    )
                else:
                    self.info_lbl.setText("")


    def _on_toggle(self, checked: bool):

        core = getattr(self.sv, "active_core", None)

        if not core:
            return

        layer_name = f"{core}::{self.layer_suffix}"

        layer = None

        for l in self.sv.viewer.layers:
            if getattr(l, "name", None) == layer_name:
                layer = l
                break

        if layer is None:
            try:
                self.layers_tab._load_boundary_layer_for_core(
                    core
                )

                layer = self.sv._get_layer(
                    f"{core}::{self.layer_suffix}"
                )

            except Exception:
                layer = None

        if layer is None:
            self.info_lbl.setText("no boundaries")
            return

        layer.visible = checked

        self.fill_chk.setEnabled(checked)
        self.opacity_sl.setEnabled(checked)
        self.fill_color_btn.setEnabled(checked)

        if not checked:

            self.fill_chk.blockSignals(True)
            self.fill_chk.setChecked(False)
            self.fill_chk.blockSignals(False)

            self._update_fill()

        if hasattr(self, "layers_tab"):
            self.layers_tab.cell_boundaries_visible = checked

        self.info_lbl.setText(
            f"cells: {len(layer.data)}"
        )


    ## Cell border colour
    def _pick_color(self):

        c = QColorDialog.getColor()

        if not c.isValid():
            return

        self.color = c.name()

        self.color_btn.setStyleSheet(
            f"color: {self.color};"
            "font-weight: bold;"
            "font-size: 16px;"
        )

        core = getattr(
            self.sv,
            "active_core",
            None
        )

        if core is None:
            return
        
        if hasattr(self, "layers_tab"):
            self.layers_tab.cell_boundary_color = self.color

        try:
            layer = self.sv._get_layer(
                f"{core}::{self.layer_suffix}"
            )

            if layer is not None:
                layer.edge_color = self.color
            
            self._update_fill()

        except Exception:
            pass
        
    ## Cell fill
    def _update_fill(self):
        core = getattr(
            self.sv,
            "active_core",
            None
        )

        if not core:
            return

        try:
            layer = self.sv._get_layer(
                f"{core}::{self.layer_suffix}"
            )
        except Exception:
            layer = None

        if layer is None:
            return

        try:
            layer.edge_color = self.color

            if self.fill_chk.isChecked():
                opacity = (
                    self.opacity_sl.value() / 100.0
                )

                c = QColor(self.fill_color)

                layer.face_color = np.tile(
                    np.array([
                        c.redF(),
                        c.greenF(),
                        c.blueF(),
                        opacity
                    ]),
                    (len(layer.data), 1)
                )
                
            else:
                layer.face_color = np.tile(
                    np.array([0, 0, 0, 0]),
                    (len(layer.data), 1)
                )

            try:
                layer.refresh()
            except Exception:
                pass

        except Exception:
            pass

        if hasattr(self, "layers_tab"):
            self.layers_tab.cell_fill_enabled = (
                self.fill_chk.isChecked()
            )

            self.layers_tab.cell_fill_opacity = (
                self.opacity_sl.value() / 100.0
            )


    ## Cell fill colour
    def _pick_fill_color(self):
        c = QColorDialog.getColor()

        if not c.isValid():
            return

        self.fill_color = c.name()

        if hasattr(self, "layers_tab"):
            self.layers_tab.cell_fill_color = self.fill_color

        self.fill_color_btn.setStyleSheet(
            f"color: {self.fill_color};"
            "font-weight: bold;"
            "font-size: 16px;"
        )

        self._update_fill()


class CometChannelRow(QWidget):
    """Row controls for a single COMET marker across all cores."""

    _COLORMAPS = ["green", "red", "cyan", "magenta", "yellow", "blue", "gray", "hot", "viridis"]

    def __init__(self, marker: str, sv, loader, parent=None):
        super().__init__(parent)
        self.marker = marker
        self.sv = sv
        self.loader = loader

        self.layers_tab = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        self.vis_chk = QCheckBox(marker)
        self.vis_chk.setFixedWidth(150)
        self.vis_chk.setChecked(False)
        self.vis_chk.toggled.connect(self._on_change)
        row.addWidget(self.vis_chk)

        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(self._COLORMAPS)
        self.cmap_cb.setFixedWidth(75)
        self.cmap_cb.currentTextChanged.connect(self._on_change)
        row.addWidget(self.cmap_cb)

        self.vmin_sp = QSpinBox()
        self.vmin_sp.setRange(0, 65535)
        self.vmin_sp.setFixedWidth(60)
        self.vmin_sp.valueChanged.connect(self._on_change)
        row.addWidget(QLabel("Min"))
        row.addWidget(self.vmin_sp)

        self.vmax_sp = QSpinBox()
        self.vmax_sp.setRange(0, 65535)
        self.vmax_sp.setFixedWidth(60)
        self.vmax_sp.valueChanged.connect(self._on_change)
        row.addWidget(QLabel("Max"))
        row.addWidget(self.vmax_sp)

        self.op_sl = QSlider(Qt.Horizontal)
        self.op_sl.setRange(0, 100)
        self.op_sl.setValue(80)
        self.op_sl.setFixedWidth(60)
        self.op_sl.valueChanged.connect(self._on_change)
        row.addWidget(QLabel("α"))
        row.addWidget(self.op_sl)

        self.setLayout(row)
        self._updating_ui = False

    def sync_to_core(self, core: str):
        """When core swaps, load the saved vmin/vmax for this core and apply."""
        self._updating_ui = True
        try:
            thresh = self.loader.comet_thresholds.get(core, {}).get(self.marker, (0.0, 1000.0))
            self.vmin_sp.setValue(int(thresh[0]))
            self.vmax_sp.setValue(int(thresh[1]))
        finally:
            self._updating_ui = False

    def _on_change(self, *args):
        if self._updating_ui:
            return

        core = self.sv.active_core

        if not core:
            return

        vmin = self.vmin_sp.value()
        vmax = self.vmax_sp.value()

        if core in self.loader.comet_thresholds:
            self.loader.comet_thresholds[core][self.marker] = (float(vmin), float(vmax))

        if (
            self.layers_tab is not None
            and not getattr(
                self.layers_tab,
                "_restoring_session",
                False,
            )
        ):
            if self.vis_chk.isChecked():
                self.layers_tab.active_proteins.add(self.marker)
            else:
                self.layers_tab.active_proteins.discard(self.marker)

        self.sv.update_comet_layer(
            self.marker,
            core=core,
            colormap=self.cmap_cb.currentText(),
            vmin=vmin,
            vmax=vmax,
            opacity=self.op_sl.value() / 100.0,
            visible=self.vis_chk.isChecked()
        )

        if self.layers_tab is not None:
            self.layers_tab.protein_settings[
                self.marker
            ] = {
                "colormap": self.cmap_cb.currentText(),
                "opacity": self.op_sl.value(),
                "vmin": self.vmin_sp.value(),
                "vmax": self.vmax_sp.value(),
            }


def _apply_affine_to_coords(M: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Apply a 2x3 or 3x3 affine matrix M to coords (N,2) and return transformed (N,2).
    Accepts M as 2x3, 3x3, or flattened; returns coords in same ordering (x,y).
    """
    if M is None:
        return coords
    M = np.asarray(M, dtype=float)
    if M.shape == (2, 3):
        # convert to 3x3 homogeneous
        M_h = np.vstack([M, [0.0, 0.0, 1.0]])
    elif M.shape == (3, 3):
        M_h = M
    else:
        # try to reshape if possible
        try:
            M_h = M.reshape(3, 3)
        except Exception:
            return coords
    H = np.hstack([coords, np.ones((coords.shape[0], 1), dtype=float)])
    out = (H @ M_h.T)[:, :2]
    return out


class TranscriptChannelRow(QWidget):
    """Row controls for a single Gene transcript across all cores.

    Behavior:
    - Always pass COMET-pixel coords (x,y) to the viewer helper (do not pre-warp).
    - Compute a single authoritative 3x3 matrix M_h (COMET px -> viewer/world)
      by preferring the image layer's metadata/transform, then loader.normalized
      matrix, then loader.raw matrix (zeroing translation).
    - Attach a Napari Affine(matrix=M_h) to the created points layer exactly once,
      and only if the existing transform differs.
    - Minimal, defensive error handling and optional terminal diagnostics via
      UNUMLOCALIA_DEBUG environment variable.
    """
    def __init__(self, gene: str, sv, loader, parent=None):
        super().__init__(parent)
        self.gene = gene
        self.sv = sv
        self.loader = loader

        self.layers_tab = None

        self.color = "yellow"

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(4)

        self.vis_chk = QCheckBox(gene)
        self.vis_chk.setMinimumWidth(100)
        self.vis_chk.setMaximumWidth(150)
        self.vis_chk.setChecked(False)
        self.vis_chk.toggled.connect(self._on_change)
        row.addWidget(self.vis_chk)

        self.color_btn = QPushButton("■")
        self.color_btn.setStyleSheet(f"color: {self.color}; font-weight: bold; font-size: 16px;")
        self.color_btn.setFixedWidth(30)
        self.color_btn.clicked.connect(self._pick_color)
        row.addWidget(self.color_btn)
        row.addSpacing(10)
        row.setAlignment(Qt.AlignLeft)
        self.setLayout(row)

    def _pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.color = c.name()

            if self.layers_tab is not None:
                self.layers_tab.active_gene_colors[self.gene] = self.color

            try:
                self.layers_tab._refresh_gene_panels()
            except Exception:
                pass

            self.color_btn.setStyleSheet(f"color: {self.color}; font-weight: bold; font-size: 16px;")
            try:
                core = getattr(self.sv, "active_core", None)
                try:
                    # prefer viewer API
                    self.sv.update_transcript_layer(self.gene, core=core, color=self.color)
                    return
                except Exception:
                    pass
                # fallback: set on layer object directly
                layer = None
                try:
                    layer = self.sv._get_layer(f"{core}::transcripts::{self.gene}")
                except Exception:
                    layer = None
                if layer is None and hasattr(self.sv, "viewer"):
                    for l in self.sv.viewer.layers:
                        try:
                            if getattr(l, "name", "") == f"{core}::transcripts::{self.gene}" or (hasattr(l, "metadata") and l.metadata.get("canonical_name") == f"{core}::transcripts::{self.gene}"):
                                layer = l
                                break
                        except Exception:
                            pass
                if layer is not None:
                    try:
                        layer.face_color = self.color
                    except Exception:
                        try:
                            layer.properties["color"] = self.color
                        except Exception:
                            pass
            except Exception:
                pass

    def _remove_layer_by_name(self, name: str):
        """Robust removal: try helper, try viewer.layers, try metadata tag, and try viewer-specific removal API."""
        # 1) try viewer helper
        try:
            layer = None
            try:
                layer = self.sv._get_layer(name)
            except Exception:
                layer = None
            if layer is not None:
                try:
                    layer.visible = False
                except Exception:
                    pass
                try:
                    try:
                        self.sv.remove_layer(layer)
                    except Exception:
                        if hasattr(self.sv, "viewer") and hasattr(self.sv.viewer, "layers"):
                            try:
                                self.sv.viewer.layers.remove(layer)
                            except Exception:
                                pass
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # 2) fallback: search viewer.layers for matching names or metadata canonical_name
        try:
            if hasattr(self.sv, "viewer") and hasattr(self.sv.viewer, "layers"):
                for l in list(self.sv.viewer.layers):
                    try:
                        lname = getattr(l, "name", "") or ""
                        meta = getattr(l, "metadata", {}) or {}
                        if lname == name or meta.get("canonical_name") == name or f"::transcripts::{self.gene}" in lname:
                            try:
                                l.visible = False
                            except Exception:
                                pass
                            try:
                                try:
                                    self.sv.remove_layer(l)
                                except Exception:
                                    try:
                                        self.sv.viewer.layers.remove(l)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) last resort: try viewer-specific transcript removal helpers if present
        try:
            try:
                self.sv.remove_transcript_layer(name)
                return
            except Exception:
                pass
            try:
                self.sv.remove_transcript_layer(self.sv.active_core, self.gene)
                return
            except Exception:
                pass
        except Exception:
            pass

    def _compute_authoritative_matrix(self, core: str):
        """Return a 3x3 numpy matrix M_h (COMET px -> viewer/world) or None."""
        M_h = None
        # 1) Prefer image layer metadata/transform (most authoritative)
        img_layer = None
        try:
            img_layer = self.sv._get_layer(f"{core}::comet::DAPI")
        except Exception:
            img_layer = None
        if img_layer is None and hasattr(self.sv, "viewer"):
            for l in self.sv.viewer.layers:
                try:
                    meta = getattr(l, "metadata", {}) or {}
                    if meta.get("core") == core and meta.get("modality") == "comet":
                        img_layer = l
                        break
                except Exception:
                    pass

        if img_layer is not None:
            meta = getattr(img_layer, "metadata", {}) or {}
            if "affine_matrix" in meta:
                try:
                    M_h = np.asarray(meta["affine_matrix"], dtype=float).reshape(3, 3)
                    return M_h
                except Exception:
                    M_h = None

            # Defensive handling: a_img.matrix may be many shapes across napari versions.
            a_img = getattr(img_layer, "affine", None) or getattr(img_layer, "transform", None)
            if a_img is not None and hasattr(a_img, "matrix"):
                try:
                    arr = np.asarray(a_img.matrix, dtype=float)

                    # Accept common shapes: (3,3), (2,3), flattened 6 or 9
                    if arr.ndim == 2 and arr.shape == (3, 3):
                        M_h = arr
                        return M_h
                    if arr.ndim == 2 and arr.shape == (2, 3):
                        M_h = np.vstack([arr, [0.0, 0.0, 1.0]])
                        return M_h
                    if arr.ndim == 1 and arr.size in (6, 9):
                        if arr.size == 6:
                            M_h = arr.reshape(2, 3)
                            M_h = np.vstack([M_h, [0.0, 0.0, 1.0]])
                            return M_h
                        else:
                            M_h = arr.reshape(3, 3)
                            return M_h
                    # If arr is unexpected (e.g., 1D length 3), ignore and fallback to loader matrices
                    self._debug("unexpected image transform shape, ignoring image transform:", arr.shape, arr if arr.size <= 9 else "<large>")
                except Exception as _:
                    M_h = None

        # 2) Fallback: build from loader matrices (prefer normalized then raw; zero translation for raw)
        try:
            M_com = None
            try:
                M_com = self.loader.alignment_matrices_comet.get(core)
            except Exception:
                M_com = None
            if M_com is None:
                try:
                    M_com = self.loader.alignment_matrices_comet_raw.get(core)
                except Exception:
                    M_com = None

            if M_com is not None:
                M_arr = np.asarray(M_com, dtype=float)
                if M_arr.shape == (2, 3):
                    M_h = np.vstack([M_arr, [0.0, 0.0, 1.0]])
                elif M_arr.shape == (3, 3):
                    M_h = M_arr.copy()
                    try:
                        # zero translation if this is a raw exported matrix (safe if already zero)
                        M_h[0, 2] = 0.0
                        M_h[1, 2] = 0.0
                    except Exception:
                        pass
                else:
                    try:
                        if M_arr.size == 6:
                            M_h = M_arr.reshape(2, 3)
                            M_h = np.vstack([M_h, [0.0, 0.0, 1.0]])
                        elif M_arr.size == 9:
                            M_h = M_arr.reshape(3, 3)
                            M_h[0, 2] = 0.0
                            M_h[1, 2] = 0.0
                    except Exception:
                        M_h = None
        except Exception:
            M_h = None

        return M_h

    def _attach_affine_if_needed(self, created_layer, M_h):
        """Attach Napari Affine(matrix=M_h) to created_layer if it differs from existing transform.
        Defensive: coerce flattened arrays, validate shape, and skip with debug if invalid.
        """
        if created_layer is None or M_h is None:
            return

        try:
            # Coerce to numpy array and inspect shape
            M = np.asarray(M_h, dtype=float)
            # Accept 1D flattened arrays of length 6 or 9
            if M.ndim == 1:
                if M.size == 6:
                    M = M.reshape(2, 3)
                elif M.size == 9:
                    M = M.reshape(3, 3)
            # Accept 2x3 -> convert to 3x3 homogeneous
            if M.ndim == 2 and M.shape == (2, 3):
                M = np.vstack([M, [0.0, 0.0, 1.0]])
            # Final check: must be 3x3
            if not (M.ndim == 2 and M.shape == (3, 3)):
                # helpful debug output
                self._debug("refusing to attach affine: not 3x3 after coercion; shape:", M.shape, "value:", M.flatten()[:9].tolist())
                # persist raw for inspection
                try:
                    created_layer.metadata = getattr(created_layer, "metadata", {}) or {}
                    created_layer.metadata["affine_matrix"] = M.tolist()
                except Exception:
                    pass
                return

            # Compare to existing transform if present
            existing_aff = getattr(created_layer, "affine", None) or getattr(created_layer, "transform", None)
            need_set = True
            try:
                if existing_aff is not None and hasattr(existing_aff, "matrix"):
                    existing_mat = np.asarray(existing_aff.matrix, dtype=float).reshape(3, 3)
                    if np.allclose(existing_mat, M, atol=1e-6, rtol=1e-6):
                        need_set = False
            except Exception:
                need_set = True

            if not need_set:
                self._debug("existing transform matches authoritative matrix; not overwriting")
                return

            # Construct Napari Affine defensively
            aff_obj = None
            try:
                from napari.utils.transforms import Affine
                try:
                    aff_obj = Affine(M)   # positional constructor preferred
                except TypeError:
                    aff_obj = Affine(matrix=M)
            except Exception as e:
                aff_obj = None
                self._debug("could not construct Napari Affine object:", e)

            if aff_obj is not None:
                # attach using transform attribute first (widely supported)
                try:
                    created_layer.transform = aff_obj
                except Exception:
                    try:
                        created_layer.affine = aff_obj
                    except Exception:
                        self._debug("failed to set aff_obj on layer via transform/affine attribute")

            # persist numeric matrix for diagnostics
            try:
                created_layer.metadata = getattr(created_layer, "metadata", {}) or {}
                created_layer.metadata["affine_matrix"] = M.tolist()
            except Exception:
                pass

            self._debug("attached authoritative affine to transcript layer")
        except Exception as e:
            self._debug("unexpected error in _attach_affine_if_needed:", e)
            try:
                created_layer.metadata = getattr(created_layer, "metadata", {}) or {}
                created_layer.metadata["affine_matrix"] = np.asarray(M_h, dtype=float).tolist()
            except Exception:
                pass

    def _on_change(self, *args):
        core = getattr(self.sv, "active_core", None)

        if self.layers_tab is not None:
            if self.vis_chk.isChecked():
                self.layers_tab.active_genes.add(self.gene)

                # Add genes to recent that aren't listed as a favourite gene
                if self.gene not in DEFAULT_FAVOURITE_GENES:
                    if self.gene in self.layers_tab.recent_genes:
                        self.layers_tab.recent_genes.remove(
                            self.gene
                        )

                    self.layers_tab.recent_genes.insert(
                        0,
                        self.gene,
                    )

                    self.layers_tab.recent_genes = (
                        self.layers_tab.recent_genes[:15]
                    )


                self.layers_tab.active_gene_colors[self.gene] = self.color

                try:
                    self.layers_tab._refresh_gene_panels()
                except Exception:
                    pass
                
            else:
                self.layers_tab.active_genes.discard(self.gene)

                try:
                    self.layers_tab._refresh_gene_panels()
                except Exception:
                    pass
                
        if not core:
            return

        layer_name = f"{core}::transcripts::{self.gene}"

        # Force loader pixel size to the known-correct value (temporary override)
        self.loader.xenium_pixel_size_um = 0.2125

        # If unchecked: remove/hide existing layer
        if not self.vis_chk.isChecked():
            self._remove_layer_by_name(layer_name)
            return

        core_manifest = self.loader.manifest.cores.get(core)
        if core_manifest is None or core_manifest.transcripts is None:
            return

        try:
            df_gene = safe_read_parquet(
                core_manifest.transcripts,
                filters=[("feature_name", "==", self.gene)],
                columns=["x_location", "y_location"],
            )
        except Exception as e:
            logger.warning("Failed to read transcripts for %s / %s: %s", core, self.gene, e)
            return

        coords = df_gene[["x_location", "y_location"]].to_numpy(dtype=np.float64)
        if coords.size == 0:
            return

        # --- Force mapping using fitted transcript_affine_by_core (µm -> COMET px) ---
        coords_mapped = None
        try:
            M_fit = self.loader.transcript_affine_by_core.get(core)
            if M_fit is not None:
                M_f = np.asarray(M_fit, dtype=float)
                if M_f.shape == (2, 3):
                    M_f = np.vstack([M_f, [0.0, 0.0, 1.0]])
                H3 = np.hstack([coords, np.ones((coords.shape[0], 1))])   # coords are in µm
                coords_mapped = (H3 @ M_f.T)[:, :2]
            else:
                # fallback to exported inverse if fitted not present
                px_um = float(getattr(self.loader, "xenium_pixel_size_um", 0.2125))
                M_exported = self.loader.alignment_matrices_comet_raw.get(core)
                if M_exported is not None:
                    M_e = np.asarray(M_exported, dtype=float)
                    if M_e.shape == (2, 3):
                        M_e = np.vstack([M_e, [0.0, 0.0, 1.0]])

                    coords_pix = coords / px_um
                    H = np.hstack([coords_pix, np.ones((coords_pix.shape[0], 1))])
                    try:
                        coords_mapped = (H @ np.linalg.inv(M_e).T)[:, :2]
                    except Exception:
                        coords_mapped = coords_pix.copy()
                else:
                    coords_mapped = coords / (self.loader.xenium_pixel_size_um or 0.2125)
        except Exception:
            coords_mapped = coords.copy()


        # --- Autoscale guard: prevent upscaling transcript cloud (only allow shrinking) ---
        try:
            img = self.loader.he_arrays.get(core)
            if img is not None:
                img_h = img.shape[0]
                img_w = img.shape[1] if img.ndim >= 2 else img.shape[-1]
                coord_min = coords_mapped.min(axis=0)
                coord_max = coords_mapped.max(axis=0)
                coord_range = coord_max - coord_min
                coord_max_dim = max(coord_range[0], coord_range[1], 1.0)
                img_max_dim = max(img_h, img_w, 1.0)
                suggested_scale = img_max_dim / coord_max_dim
                # Prevent upscaling: cap suggested_scale at 1.0 so we never enlarge the transcript cloud.
                suggested_scale = min(suggested_scale, 1.0)
                # Apply only if a meaningful downscale is required
                if suggested_scale < 0.99:
                    center = (coord_min + coord_max) / 2.0
                    coords_mapped = (coords_mapped - center) * suggested_scale + center
        except Exception:
            pass


        # --- Choose authoritative 3x3 COMET->viewer matrix (M_h) for diagnostics ---
        M_h = None
        img_layer = None
        try:
            try:
                img_layer = self.sv._get_layer(f"{core}::comet::DAPI")
            except Exception:
                img_layer = None
            if img_layer is None and hasattr(self.sv, "viewer"):
                for l in self.sv.viewer.layers:
                    try:
                        meta = getattr(l, "metadata", {}) or {}
                        if meta.get("core") == core and meta.get("modality") == "comet":
                            img_layer = l
                            break
                    except Exception:
                        pass

            if img_layer is not None:
                meta = getattr(img_layer, "metadata", {}) or {}
                if "affine_matrix" in meta:
                    try:
                        M_h = np.asarray(meta["affine_matrix"], dtype=float).reshape(3, 3)
                    except Exception:
                        M_h = None
                else:
                    a_img = getattr(img_layer, "affine", None) or getattr(img_layer, "transform", None)
                    if a_img is not None and hasattr(a_img, "matrix"):
                        try:
                            arr = np.asarray(a_img.matrix, dtype=float)
                            if arr.ndim == 2 and arr.shape == (3, 3):
                                M_h = arr
                            elif arr.ndim == 2 and arr.shape == (2, 3):
                                M_h = np.vstack([arr, [0.0, 0.0, 1.0]])
                        except Exception:
                            M_h = None

            if M_h is None:
                M_com = self.loader.alignment_matrices_comet.get(core) or self.loader.alignment_matrices_comet_raw.get(core)
                if M_com is not None:
                    M_arr = np.asarray(M_com, dtype=float)
                    if M_arr.shape == (2, 3):
                        M_h = np.vstack([M_arr, [0.0, 0.0, 1.0]])
                    elif M_arr.shape == (3, 3):
                        M_h = M_arr.copy()
                    else:
                        try:
                            M_h = M_arr.reshape(3, 3)
                        except Exception:
                            M_h = None
        except Exception:
            M_h = None

        # Remove any existing layer first to avoid duplicates
        self._remove_layer_by_name(layer_name)

        # --- Pre-warp into viewer/world coords and add points layer (avoid Napari Affine fragility) ---
        try:
            if M_h is not None:
                H_pts = np.hstack([coords_mapped, np.ones((coords_mapped.shape[0], 1))])
                coords_world = coords_mapped
            else:
                coords_world = coords_mapped.copy()
        except Exception:
            coords_world = coords_mapped.copy()


        # Compute authoritative matrix M_h (COMET px -> viewer/world)
        M_h = self._compute_authoritative_matrix(core)

        # Napari expects (row, col) ordering -> reverse x,y to y,x
        # pts_for_napari = coords_world[:, ::-1]
        pts_for_napari = coords_world


        try:
            created_layer = self.sv.add_transcript_layer(
                core=core,
                gene=self.gene,
                coords=coords_world,
                color=self.color,
                visible=True,
                affine=M_h,
            )
        except Exception:
            # fallback if viewer helper fails
            try:
                created_layer = self.sv.viewer.add_points(
                    pts_for_napari[:, ::-1],  # manual (y,x) swap for napari
                    name=layer_name,
                    size=10,
                    face_color=self.color,
                    visible=True,
                )
            except Exception as e:
                logger.exception(
                    "Failed to add transcript layer for %s / %s: %s",
                    core,
                    self.gene,
                    e,
                )
                return


        # Persist the authoritative matrix for diagnostics
        try:
            created_layer.metadata = getattr(created_layer, "metadata", {}) or {}
            created_layer.metadata["affine_matrix"] = np.asarray(M_h, dtype=float).tolist() if M_h is not None else None
        except Exception:
            pass

        # Ensure visible/color
        try:
            created_layer.visible = True
        except Exception:
            pass
        try:
            created_layer.face_color = self.color
        except Exception:
            try:
                created_layer.properties["color"] = self.color
            except Exception:
                pass

        # Set size and refresh as before
        size_val = 25.0

        try:
            if (
                self.layers_tab is not None
                and hasattr(self.layers_tab, "tx_size")
            ):
                size_val = float(
                    self.layers_tab.tx_size.value()
                )
        except Exception:
            pass

        try:
            try:
                self.sv.update_transcript_layer(self.gene, core=core, color=self.color, size=size_val, visible=True)
            except Exception:
                layer = None
                try:
                    layer = self.sv._get_layer(layer_name)
                except Exception:
                    layer = None
                if layer is None and hasattr(self.sv, "viewer"):
                    for l in self.sv.viewer.layers:
                        try:
                            if getattr(l, "name", "") == layer_name or (hasattr(l, "metadata") and l.metadata.get("canonical_name") == layer_name):
                                layer = l
                                break
                        except Exception:
                            pass
                if layer is not None:
                    try:
                        layer.face_color = self.color
                    except Exception:
                        try:
                            layer.properties["color"] = self.color
                        except Exception:
                            pass
                    try:
                        layer.size = size_val
                    except Exception:
                        try:
                            layer.properties["size"] = size_val
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            if hasattr(self.sv, "refresh_transcript_layers"):
                self.sv.refresh_transcript_layers()
        except Exception:
            pass


## Tab with data selection/loading
class DataTab(QWidget):
    dataset_loaded = Signal(object)  # emits DatasetLoader

    # Session save/load
    session_save_requested = Signal()
    session_load_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        group = QGroupBox("Dataset Loading")
        g_layout = QVBoxLayout(group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select UnumLocalia dataset folder...")

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)

        h = QHBoxLayout()
        h.addWidget(self.path_edit)
        h.addWidget(browse_btn)
        g_layout.addLayout(h)

        self.load_btn = QPushButton("Load Dataset (Manifest)")
        self.load_btn.clicked.connect(self._load)
        g_layout.addWidget(self.load_btn)

        self.log_area = QLabel("Select a folder with a dataset_manifest.csv")
        self.log_area.setWordWrap(True)
        g_layout.addWidget(self.log_area)

        # Session save/load
        session_group = QGroupBox("Session")
        session_layout = QVBoxLayout(session_group)
        self.save_session_btn = QPushButton("Save Session")
        self.load_session_btn = QPushButton("Load Session")
        session_layout.addWidget(self.save_session_btn)
        session_layout.addWidget(self.load_session_btn)
        layout.addWidget(session_group)

        self.save_session_btn.clicked.connect(self.session_save_requested)
        self.load_session_btn.clicked.connect(self.session_load_requested)

        # Keep at end of this section
        layout.addWidget(group)
        layout.addStretch()

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.path_edit.setText(folder)

    def _load(self):
        path = self.path_edit.text()
        if not path:
            return

        from unumlocalia.io import DatasetLoader
        try:
            self.log_area.setText("Loading manifest and scanning files...")
            QApplication.processEvents()

            loader = DatasetLoader(path).load(
                do_load_transcripts=False,    # keep genes list only
                load_boundaries=True,         # keep for loading functions, but have cores load individually
                load_he=True,                 # H&E
                load_comet=True,              # COMET
            )

            self.log_area.setText(loader.manifest.summary())
            self.dataset_loaded.emit(loader)

        except Exception as e:
            self.log_area.setText(f"Error loading dataset:\n{e}")
            logger.exception("Error loading dataset")


## Layers tab
class LayersTab(QWidget):
    def __init__(self, sv, parent=None):
        super().__init__(parent)
        self.sv = sv
        self.loader = None

        self.comet_rows: List[CometChannelRow] = []
        self.gene_rows: List[TranscriptChannelRow] = []

        # Persisted layer state across core swaps
        self.active_genes = set()
        self.active_proteins = set()

        self.recent_genes = []
        self.protein_settings = {}
        self.active_gene_colors = {}

        self.he_visible = True #H&E shown between cores
        self.cell_masks_visible = False
        self.cell_boundaries_visible = False

        layout = QVBoxLayout(self)

        # 1. Core Selection
        core_group = QGroupBox("Active Core")
        c_layout = QHBoxLayout(core_group)
        self.core_combo = QComboBox()
        self.core_combo.currentTextChanged.connect(self._on_core_swapped)
        c_layout.addWidget(self.core_combo)
        layout.addWidget(core_group)

        # 2. H&E Control
        self.he_group = CollapsibleGroup("H&&E Base")
        he_layout = QHBoxLayout(self.he_group)
        self.he_vis = QCheckBox()
        self.he_vis.setChecked(True)
        self.he_vis.toggled.connect(self._update_he)
        self.he_op = QSlider(Qt.Horizontal)
        self.he_op.setRange(0, 100)
        self.he_op.setValue(100)
        self.he_op.valueChanged.connect(self._update_he)
        layout.addWidget(self.he_group)
        he_layout.addWidget(QLabel("Visible"))
        he_layout.addWidget(self.he_vis)
        he_layout.addSpacing(15)
        he_layout.addWidget(QLabel("Opacity"))
        he_layout.addWidget(self.he_op)

        # 3. Scrollable Markers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.scroll_content = QWidget()

        self.scroll_layout = QVBoxLayout(self.scroll_content)

        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(4)

        scroll.setWidget(self.scroll_content)

        layout.addWidget(scroll)

        # Cell border colour
        self.cell_boundary_color = "white"

        # Cell fill
        self.cell_fill_enabled = False
        self.cell_fill_opacity = 0.2

        self.cell_fill_color = "#ffff00"

        # Export image
        export_group = QGroupBox("Export PNG")
        export_layout = QVBoxLayout(export_group)

        btn_row = QHBoxLayout()

        self.export_btn = QPushButton("Export PNG")

        btn_row.addWidget(self.export_btn)
        export_layout.addLayout(btn_row)

        self.export_btn.clicked.connect(self._export_image)

        layout.addWidget(export_group)

        # Refresh for gene search panel
        self._refreshing_gene_panels = False

        # Scale bar (for image export)
        self.scalebar_chk = QCheckBox("Include scale bar")
        self.scalebar_chk.setChecked(True)
        export_layout.addWidget(self.scalebar_chk)

        self.segmentation_rows = {}
        

    def _on_tx_size_changed(self, v):
        if hasattr(self, "tx_size_label"):
            self.tx_size_label.setText(
                str(int(v))
            )
        
        try:
            self.sv.set_transcript_size(float(v))
        except Exception:
            pass
        # also update existing layers immediately
        try:
            for l in list(self.sv.viewer.layers):
                try:
                    meta = getattr(l, "metadata", {}) or {}
                    if meta.get("modality") == "xenium" or "::transcripts::" in getattr(l, "name", ""):
                        try:
                            l.size = float(v)
                        except Exception:
                            try:
                                l.properties["size"] = float(v)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        

    ## Export settings
    def export_state(self):
        return {
            "he_visible":
                self.he_vis.isChecked(),

            "he_opacity":
                self.he_op.value(),

            "active_core":
                self.core_combo.currentText(),

            "protein_settings": {
                row.marker: {
                    "visible":
                        row.vis_chk.isChecked(),
                    "colormap":
                        row.cmap_cb.currentText(),
                    "opacity":
                        row.op_sl.value(),
                    "vmin":
                        row.vmin_sp.value(),
                    "vmax":
                        row.vmax_sp.value(),
                }
                for row in self.comet_rows
            },

            "recent_genes":
                self.recent_genes,

            "active_genes":
                list(self.active_genes),

            "active_gene_colors":
                self.active_gene_colors,

            "cell_boundary_color":
                self.cell_boundary_color,

            "cell_fill_enabled":
                self.cell_fill_enabled,

            "cell_fill_opacity_slider":
                int(
                    self.cell_fill_opacity * 100
                ),

            "cell_fill_color":
                self.cell_fill_color,

            "transcript_size":
                self.tx_size.value()
                if hasattr(self, "tx_size")
                else 25.0,

            # Is a cell mask active?
            "cell_boundaries_visible":
                self.cell_boundaries_visible,
        }
    

    ## Import settings
    def import_state(self, state):
        self._restoring_session = True

        if "he_visible" in state:
            self.he_vis.setChecked(
                state["he_visible"]
            )

        if "he_opacity" in state:
            self.he_op.setValue(
                int(state["he_opacity"])
            )
        
        self.protein_settings = state.get(
            "protein_settings",
            {}
        )

        self.active_proteins = {
            marker
            for marker, settings
            in self.protein_settings.items()
            if settings.get("visible", False)
        }

        self.recent_genes = state.get(
            "recent_genes",
            [],
        )

        self.active_genes = set(
            state.get(
                "active_genes",
                []
            )
        )

        self.active_gene_colors = state.get(
            "active_gene_colors",
            {}
        )

        self.cell_boundary_color = state.get(
            "cell_boundary_color",
            "white",
        )

        self.cell_fill_enabled = state.get(
            "cell_fill_enabled",
            False,
        )

        self.cell_fill_opacity = (
            state.get(
                "cell_fill_opacity_slider",
                20,
            ) / 100
        )

        self.cell_fill_color = state.get(
            "cell_fill_color",
            "#ffff00",
        )

        if (
            hasattr(self, "tx_size")
            and "transcript_size" in state
        ):
            self.tx_size.setValue(
                int(
                    state["transcript_size"]
                )
            )

        self.cell_boundaries_visible = state.get(
            "cell_boundaries_visible",
            False,
        )

        core = state.get(
            "active_core"
        )

        if core:
            self.core_combo.setCurrentText(
                core
            )

        self._on_core_swapped(
            self.core_combo.currentText()
        )

        try:
            self._refresh_gene_panels()
        except Exception:
            pass

        self._restoring_session = False
    

    def populate(self, loader):
        """Called when dataset is loaded."""
        self.loader = loader
        self.core_combo.blockSignals(True)
        self.core_combo.clear()

        cores = list(loader.manifest.cores.keys())
        if not cores:
            return

        self.core_combo.addItems(cores)
        self.core_combo.blockSignals(False)

        _clear_layout(self.scroll_content)
        self.comet_rows = []
        self.gene_rows = []

        # --- CELL MASKS ---
        # Add Cell Mask control row (single global row that operates on active core)
        mask_group = CollapsibleGroup("Cell Boundaries")
        mask_layout = QVBoxLayout(mask_group)

        self.seg_rows_container = QWidget()

        self.seg_rows_layout = QVBoxLayout(
            self.seg_rows_container
        )

        self.seg_rows_layout.setSpacing(2)

        seg_scroll = QScrollArea()

        seg_scroll.setWidgetResizable(True)

        seg_scroll.setMinimumHeight(120)
        seg_scroll.setMaximumHeight(220)

        seg_scroll.setWidget(
            self.seg_rows_container
        )

        mask_layout.addWidget(
            seg_scroll
        )

        self.import_seg_btn = QPushButton(
            "Import Segmentation..."
        )

        self.import_seg_btn.clicked.connect(
            self._import_segmentation
        )

        mask_layout.addWidget(
            self.import_seg_btn
        )


        mask_layout.setContentsMargins(6, 6, 6, 6)

        self.scroll_layout.addWidget(mask_group)

        # --- COMET PROTEINS ---
        # COMET controls
        if loader.proteins:
            comet_group = CollapsibleGroup("COMET Proteins")
            comet_layout = QVBoxLayout(comet_group)
            
            comet_scroll = QScrollArea()
            comet_scroll.setWidgetResizable(True)

            comet_widget = QWidget()
            comet_inner_layout = QVBoxLayout(comet_widget)

            comet_scroll.setWidget(comet_widget)

            comet_layout.addWidget(comet_scroll)

            comet_group.setFont(QFont("", 9, QFont.Bold))

            for p in loader.proteins:
                row = CometChannelRow(p, self.sv, self.loader)
                row.layers_tab = self
                self.comet_rows.append(row)
                comet_inner_layout.addWidget(row)

            self.scroll_layout.addWidget(comet_group)
            comet_layout.setContentsMargins(6, 6, 6, 6)

        # --- XENIUM GENES ---
        # Gene controls
        if loader.genes:
            gene_group = CollapsibleGroup("Xenium Genes")
            gene_layout = QVBoxLayout(gene_group)
            gene_group.setFont(QFont("", 9, QFont.Bold))
            gene_group.setStyleSheet("margin-top: 10px;")

            # Search box
            search_layout = QHBoxLayout()

            self.gene_search = QLineEdit()
            self.gene_search.setPlaceholderText("Search genes...")
            self.gene_search.textChanged.connect(self._filter_genes)

            self.gene_clear_btn = QPushButton("✕")
            self.gene_clear_btn.setFixedWidth(30)
            self.gene_clear_btn.clicked.connect(self.gene_search.clear)

            search_layout.addWidget(self.gene_search)
            search_layout.addWidget(self.gene_clear_btn)

            gene_layout.addLayout(search_layout)

            # Gene dot sizes
            tx_size_layout = QHBoxLayout()

            tx_size_layout.addWidget(QLabel("Global Dot Size"))

            # Slider
            self.tx_size = QSlider(Qt.Horizontal)

            self.tx_size.setRange(1, 100)
            self.tx_size.setValue(25)

            self.tx_size.valueChanged.connect(
                self._on_tx_size_changed
            )

            self.tx_size_label = QLabel("25")
            self.tx_size_label.setStyleSheet(
                "font-weight: bold;"
            )
            self.tx_size_label.setMinimumWidth(35)

            tx_size_layout.addWidget(
                self.tx_size
            )

            tx_size_layout.addWidget(
                self.tx_size_label
            )

            gene_layout.addLayout(tx_size_layout)

            # Unselect genes
            self.clear_gene_layers_btn = QPushButton("Unselect All Genes")
            self.clear_gene_layers_btn.clicked.connect(
                self._unselect_all_genes
            )

            gene_layout.addWidget(self.clear_gene_layers_btn)

            # Store full gene list
            self.all_genes = sorted(loader.genes)
            self.favourite_genes_widget = QWidget()
            self.favourite_genes_layout = QVBoxLayout(self.favourite_genes_widget)

            self.active_genes_widget = QWidget()
            self.active_genes_layout = QVBoxLayout(self.active_genes_widget)

            self.recent_genes_widget = QWidget()
            self.recent_genes_layout = QVBoxLayout(self.recent_genes_widget)

            # Container where matching rows will appear
            gene_scroll = QScrollArea()
            gene_scroll.setWidgetResizable(True)
            gene_scroll.setMaximumHeight(120)

            self.gene_rows_widget = QWidget()
            self.gene_rows_layout = QVBoxLayout(self.gene_rows_widget)

            self.gene_rows_layout.setContentsMargins(0, 0, 0, 0)
            self.gene_rows_layout.setSpacing(1)

            gene_scroll.setWidget(self.gene_rows_widget)

            gene_layout.addWidget(gene_scroll)

            # Gene order
            gene_layout.addWidget(QLabel("Active Genes"))
            gene_layout.addWidget(self.active_genes_widget)

            gene_layout.addWidget(QLabel("Recent Genes"))
            gene_layout.addWidget(self.recent_genes_widget)

            gene_layout.addWidget(QLabel("Favourite Genes"))
            gene_layout.addWidget(self.favourite_genes_widget)

            gene_layout.setContentsMargins(6, 6, 6, 6)

            # Populate initial view
            self._filter_genes("")
            self._refresh_gene_panels()

            self.scroll_layout.addWidget(gene_group)

        self.scroll_layout.addStretch()

        # Add everything to viewer
        self._add_all_to_viewer()

        # Set active core
        self.core_combo.setCurrentText(cores[0])
        self._on_core_swapped(cores[0])
        self.sv.reset_view()

        # Dynamic rows for user-defined segmentation
        xenium_row = CellBoundaryRow(
            self.sv,
            loader,
            display_name="Xenium",
            layer_suffix="cells",
        )

        xenium_row.layers_tab = self

        self.seg_rows_layout.addWidget(
            xenium_row
        )

        self.segmentation_rows.setdefault(
            "__xenium__",
            {}
        )["cells"] = xenium_row


    def _clear_qt_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)

            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _adjust_matrix(self, M_raw: Optional[np.ndarray], core_id: str) -> Optional[np.ndarray]:
        """Subtract the core's reference translation from the matrix so images stay centered."""
        if M_raw is None:
            return None

        T_ref = np.array([0.0, 0.0])
        M_com = self.loader.alignment_matrices_comet_raw.get(core_id)
        if M_com is not None:
            T_ref = np.array([M_com[0, 2], M_com[1, 2]])
        elif self.loader.alignment_matrices_he_raw.get(core_id) is not None:
            M_he = self.loader.alignment_matrices_he_raw.get(core_id)
            T_ref = np.array([M_he[0, 2], M_he[1, 2]])

        M = M_raw.copy()
        if M.ndim == 2 and M.shape == (2, 3):
            M[0, 2] -= T_ref[0]
            M[1, 2] -= T_ref[1]
        elif M.ndim == 2 and M.shape == (3, 3):
            M[0, 2] -= T_ref[0]
            M[1, 2] -= T_ref[1]
        return M

    def _remove_all_transcript_layers(self):
        """Remove any transcript layers from the viewer (robust)."""
        try:
            if not hasattr(self.sv, "viewer"):
                return
            for l in list(self.sv.viewer.layers):
                try:
                    meta = getattr(l, "metadata", {}) or {}
                    name = getattr(l, "name", "") or ""
                    # canonical transcript naming: "<core>::transcripts::<gene>"
                    if "::transcripts::" in name or meta.get("sb_source", "").startswith("transcripts::") or meta.get("modality") == "xenium":
                        try:
                            # prefer viewer API removal
                            try:
                                self.sv.remove_layer(l)
                            except Exception:
                                self.sv.viewer.layers.remove(l)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _add_all_to_viewer(self):
        self.sv.clear_all_layers()
        loader = self.loader
        for core_id, core_man in loader.manifest.cores.items():

            # H&E
            if core_id in loader.he_arrays:
                M_he = self._adjust_matrix(loader.alignment_matrices_he_raw.get(core_id), core_id)
                he = loader.he_arrays[core_id]
                self.sv.add_he_layer(core_id, he, affine=M_he, visible=False)

            # COMET (lazy placeholders)
            if core_id in loader.comet_paths_by_core:
                comet_path = loader.comet_paths_by_core[core_id]
                markers = loader.comet_markers.get(core_id, [])
                M_com = self._adjust_matrix(loader.alignment_matrices_comet_raw.get(core_id), core_id)

                if markers:
                    for i, marker in enumerate(markers):
                        placeholder = np.zeros((2, 2), dtype=np.uint8)
                        layer = self.sv.add_comet_layer(core_id, marker, placeholder, affine=M_com, visible=False)
                        layer.metadata["lazy_path"] = str(comet_path)
                        layer.metadata["lazy_channel_index"] = int(i)
                        try:
                            def _make_on_visible(l=layer, cid=core_id, idx=i, marker_name=marker, M_com_local=M_com):
                                def _on_visible(event=None):
                                    try:
                                        if not getattr(l, "visible", False):
                                            return

                                        data = getattr(l, "data", None)
                                        if data is not None and getattr(data, "shape", ()) != (2, 2):
                                            return

                                        try:
                                            orig_name = getattr(l, "name", "")
                                            orig_opacity = getattr(l, "opacity", 1.0)
                                            l.metadata["_loading_orig_name"] = orig_name
                                            l.metadata["_loading_orig_opacity"] = orig_opacity
                                            l.name = f"{marker_name} (loading...)"
                                            try:
                                                l.opacity = max(0.15, orig_opacity * 0.6)
                                            except Exception:
                                                pass
                                            try:
                                                from qtpy.QtWidgets import QApplication
                                                QApplication.processEvents()
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass

                                        arr = loader.get_comet_channel(cid, idx)

                                        try:
                                            if hasattr(arr, "compute") and callable(arr.compute):
                                                arr = arr.compute()
                                        except Exception:
                                            pass

                                        try:
                                            l.data = arr
                                        except Exception:
                                            import numpy as _np
                                            l.data = _np.array(arr)

                                        if M_com_local is not None:
                                            try:
                                                l.metadata["affine_matrix"] = (
                                                    M_com_local.tolist()
                                                    if isinstance(M_com_local, np.ndarray)
                                                    else M_com_local
                                                )
                                            except Exception:
                                                pass

                                        try:
                                            orig_name = l.metadata.pop("_loading_orig_name", None)
                                            orig_opacity = l.metadata.pop("_loading_orig_opacity", None)
                                            if orig_name:
                                                l.name = orig_name
                                            if orig_opacity is not None:
                                                try:
                                                    l.opacity = orig_opacity
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass

                                    except Exception:
                                        try:
                                            l.name = f"{marker_name} (load failed)"
                                            l.metadata["load_failed"] = True
                                        except Exception:
                                            pass
                                        logger.exception("Failed to load lazy COMET channel for %s / %s", cid, marker_name)
                                return _on_visible
                            if hasattr(layer, "events") and hasattr(layer.events, "visible"):
                                layer.events.visible.connect(_make_on_visible())
                        except Exception:
                            logger.debug("Could not connect visible event for lazy COMET layer")
                else:
                    placeholder = np.zeros((2, 2), dtype=np.uint8)
                    layer = self.sv.add_comet_layer(core_id, "Channel_0", placeholder, affine=M_com, visible=False)
                    layer.metadata["lazy_path"] = str(comet_path)
                    layer.metadata["lazy_channel_index"] = 0
                    try:
                        if hasattr(layer, "events") and hasattr(layer.events, "visible"):
                            def _on_visible_single(event=None, l=layer, cid=core_id):
                                if not getattr(l, "visible", False):
                                    return
                                try:
                                    arr = loader.get_comet_channel(cid, 0)
                                    l.data = arr
                                except Exception:
                                    logger.exception("Failed to load lazy COMET channel (single)")
                            layer.events.visible.connect(_on_visible_single)
                    except Exception:
                        logger.debug("Could not connect visible event for lazy COMET layer (single)")


    ## Create Nepari boundaries
    def _load_boundary_layer_for_core(self, core_id):

        if self.sv._get_layer(f"{core_id}::cells") is not None:
            return

        if core_id not in self.loader.cell_boundaries_df:
            return

        df_cb = self.loader.cell_boundaries_df[core_id]

        if "vertex_x" not in df_cb.columns:
            return

        M_fit = self.loader.transcript_affine_by_core.get(core_id)

        shapes_napari = []

        for _, group in df_cb.groupby(
            "cell_id",
            sort=False
        ):

            xy = group[
                ["vertex_x", "vertex_y"]
            ].to_numpy(dtype=float)

            if M_fit is not None:

                H = np.hstack([
                    xy,
                    np.ones((len(xy), 1))
                ])

                xy = (
                    H @ np.asarray(M_fit).T
                )[:, :2]

            shapes_napari.append(
                xy[:, ::-1]
            )

        M_com = self._adjust_matrix(
            self.loader.alignment_matrices_comet_raw.get(core_id),
            core_id
        )

        self.sv.add_boundary_layer(
            core_id,
            shapes_napari,
            name="cells",
            color=self.cell_boundary_color,
            visible=False,
            affine=M_com,
        )
    

    def _on_core_swapped(self, core: str):
        """Called when the active core selection changes."""

        try:
            # update viewer state via sv helper if available
            try:
                self.sv.set_active_core(core)
            except Exception:
                pass
        except Exception:
            pass

        # sync comet rows and gene rows if present
        try:
            for r in self.comet_rows:
                try:
                    r.sync_to_core(core)
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            for r in self.gene_rows:
                try:
                    r.sync_to_core(core)
                except Exception:
                    pass
        except Exception:
            pass

        # sync mask row
        try:
            for core_rows in self.segmentation_rows.values():
                for row in core_rows.values():
                    row.sync_to_core(core)
        except Exception:
            pass
        

        # Restore H&E
        try:
            self._update_he()
        except Exception:
            pass
        
        # Restore proteins
        for row in self.comet_rows:
            settings = (
                self.protein_settings.get(
                    row.marker,
                    {}
                )
            )

            row.vis_chk.blockSignals(True)
            row.cmap_cb.blockSignals(True)
            row.op_sl.blockSignals(True)

            row.vis_chk.setChecked(
                row.marker in self.active_proteins
            )

            row.cmap_cb.setCurrentText(
                settings.get(
                    "colormap",
                    "green"
                )
            )

            row.op_sl.setValue(
                settings.get(
                    "opacity",
                    80
                )
            )

            row.vis_chk.blockSignals(False)
            row.cmap_cb.blockSignals(False)
            row.op_sl.blockSignals(False)

            row._on_change()
        
        # Restore genes
        for gene in self.active_genes:
            try:

                row = TranscriptChannelRow(
                    gene,
                    self.sv,
                    self.loader,
                )

                row.layers_tab = self

                self._apply_saved_gene_color(row, gene,)

                row.vis_chk.setChecked(True)

                row._on_change()

            except Exception:
                pass

        try:
            self._on_tx_size_changed(
                self.tx_size.value()
            )
        except Exception:
            pass

        # Restore cell fill colour
        try:
            for core_rows in self.segmentation_rows.values():
                for row in core_rows.values():
                    row.color = self.cell_boundary_color

                    row.color_btn.setStyleSheet(
                        f"color: {self.cell_boundary_color};"
                        "font-weight: bold;"
                        "font-size: 16px;"
                    )

                    row.fill_chk.blockSignals(True)

                    if row.chk.isChecked():
                        row.fill_chk.setChecked(
                            self.cell_fill_enabled
                        )
                    else:
                        row.fill_chk.setChecked(False)

                    row.fill_chk.blockSignals(False)

                    row.opacity_sl.blockSignals(True)

                    row.opacity_sl.setValue(
                        int(self.cell_fill_opacity * 100)
                    )

                    row.opacity_sl.blockSignals(False)

                    row.fill_color = self.cell_fill_color

                    row.fill_color_btn.setStyleSheet(
                        f"color: {self.cell_fill_color};"
                        "font-weight: bold;"
                        "font-size: 16px;"
                    )

                    row._update_fill()

        except Exception:
            pass
        
        # User-defined segmentation
        for method_name in (
            self.loader
            .custom_segmentations
            .get(core, {})
        ):
            try:

                lname = (
                    f"{core}::segmentation::{method_name}"
                )

                layer = self.sv._get_layer(lname)

                if layer is not None:
                    layer.visible = True

            except Exception:
                pass
            

    def _update_he(self):
        core = self.core_combo.currentText()
        if not core:
            return

        try:
            layer = self.sv._get_layer(f"{core}::he")
            if layer:
                layer.metadata["user_visible"] = self.he_vis.isChecked()
                layer.visible = self.he_vis.isChecked()
                layer.opacity = self.he_op.value() / 100.0
        except Exception:
            pass

    # Filter genes for searching
    def _filter_genes(self, text):
        """
        Live gene filtering.

        Creates widgets only for matching genes instead of
        making thousands of widgets visible/invisible.
        """

        text = text.strip().lower()

        self._clear_qt_layout(
            self.gene_rows_layout
        )

        if not text:
            self.gene_rows_widget.hide()
            self.scroll_content.update()
            return

        self.gene_rows_widget.show()        

        matches = [
            g
            for g in self.all_genes
            if text in g.lower()
        ]

        self.gene_rows_widget.setUpdatesEnabled(False)

        self._clear_qt_layout(self.gene_rows_layout)

        # Prevent huge result sets from generating thousands of widgets
        MAX_VISIBLE_ROWS = 25

        for gene in matches[:MAX_VISIBLE_ROWS]:
            row = TranscriptChannelRow(
                gene,
                self.sv,
                self.loader,
            )

            row.layers_tab = self

            self._apply_saved_gene_color(row, gene,)

            if gene in self.active_genes:
                row.vis_chk.setChecked(True)

            self.gene_rows_layout.addWidget(row)

        self.gene_rows_widget.setUpdatesEnabled(True)

        self.scroll_content.update()

    # Unselect all genes
    def _unselect_all_genes(self):
        self._remove_all_transcript_layers()

        for row in self.gene_rows_layout.parent().findChildren(
            TranscriptChannelRow
        ):
            try:
                row.vis_chk.setChecked(False)
            except Exception:
                pass


    ## Add export method
    def _export_image(self, checked=False):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            f"{self.sv.active_core}_screenshot.png",
            "PNG (*.png)"
        )

        if not filename:
            return

        try:

            current_size = float(
                self.tx_size.value()
            )

            for layer in self.sv.viewer.layers:

                try:

                    if "::transcripts::" in layer.name:

                        layer.size = current_size

                except Exception:
                    pass

            # Suppress deprecation warning from Napari regarding qt_viewer access
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Public access to Window.qt_viewer is deprecated.*",
                    category=FutureWarning,
                )

                img = self.sv.viewer.window.qt_viewer.canvas.native.grabFramebuffer()

            img = img.convertToFormat(img.Format_RGBA8888)

            width = img.width()
            height = img.height()

            ptr = img.bits()
            ptr.setsize(height * width * 4)

            img = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

            # Upscale image for high-res export
            im = Image.fromarray(img)

            target_width = 4000

            if im.width < target_width:
                scale_factor = target_width / im.width

                im = im.resize(
                    (
                        int(im.width * scale_factor),
                        int(im.height * scale_factor),
                    ),
                    Image.Resampling.LANCZOS,
                )

            img = np.asarray(im)

            if self.scalebar_chk.isChecked():

                img = self._add_scalebar(
                    img,
                    scale=1
                )

            imwrite(
                filename,
                img
            )

        except Exception as e:

            print(
                "EXPORT FAILED:",
                e
            )


    ## Scale bar (for image export)
    def _add_scalebar(
        self,
        img,
        scale=1,
    ):
        """
        Draw publication-style scale bar
        in bottom-right corner.
        """

        try:

            im = Image.fromarray(img)

            draw = ImageDraw.Draw(im)

            width, height = im.size

            margin = int(30 * scale)

            # Default scale-bar lengths
            zoom = self.sv.viewer.camera.zoom

            # Suppress deprecation warning from Napari regarding qt_viewer access
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Public access to Window.qt_viewer is deprecated.*",
                    category=FutureWarning,
                )

                canvas_width = (
                    self.sv.viewer.window.qt_viewer.canvas.size[0]
                )

            pixel_size_um = 0.2125

            core = self.sv.active_core

            layer = self.sv._get_layer(
                f"{core}::he"
            )

            M = np.asarray(
                layer.metadata["affine_matrix"],
                dtype=float
            )

            scale_factor = np.sqrt(
                M[0,0]**2 +
                M[1,0]**2
            )

            effective_um_per_world = (
                pixel_size_um / scale_factor
            )
            
            visible_width_um = (
                (canvas_width / zoom)
                * effective_um_per_world
            )

            bar_um = self._nice_scalebar_length_um(
                visible_width_um
            )

            bar_px = int(
                width * (bar_um / visible_width_um)
            )

            label = f"{bar_um:g} um"

            x2 = width - margin
            x1 = x2 - bar_px

            y = height - 80

            # black outline
            outline_width = 20

            inner_width = 10

            draw.line(
                [(x1, y), (x2, y)],
                fill="black",
                width=outline_width,
            )

            draw.line(
                [(x1, y), (x2, y)],
                fill="white",
                width=inner_width,
            )

            # white interior
            draw.line(
                [(x1, y), (x2, y)],
                fill="white",
                width=max(2, int(2 * scale))
            )

            try:
                try:
                    font = ImageFont.truetype(
                        "Arial.ttf",
                        60
                    )
                except Exception:
                    try:
                        font = ImageFont.truetype(
                            "/System/Library/Fonts/Supplemental/Arial.ttf",
                            60
                        )
                    except Exception:
                        font = ImageFont.load_default()
            except Exception:
                font = None

            text_x = x1

            text_y = y - 100

            # black outline text
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):

                    draw.text(
                        (
                            text_x + dx,
                            text_y + dy,
                        ),
                        label,
                        fill="black",
                        font=font,
                    )

            # white text
            draw.text(
                (
                    text_x,
                    text_y,
                ),
                label,
                fill="white",
                font=font,
            )

            return np.asarray(im)

        except Exception as e:
            print("SCALEBAR ERROR:", e)
            return img


## Scale bar calculations
    def _nice_scalebar_length_um(
        self,
        visible_width_um,
    ):
        """
        Choose a sensible scalebar length
        based on the current field of view.
        """

        target = visible_width_um * 0.20

        choices = [
            10,
            20,
            25,
            50,
            75,
            100,
            200,
            250,
            500,
            1000,
            2000,
            5000
        ]

        for c in choices:
            if c >= target:
                return c

        return choices[-1]
    

    ## User-defined segmentation import
    def _import_segmentation(self):
        core = self.core_combo.currentText()

        if not core:
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select segmentation",
            "",
            "GeoJSON (*.geojson *.json)"
        )

        if not path:
            return

        method_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Segmentation Name",
            "Method name:"
        )

        if not ok or not method_name:
            return

        try:
            shapes = self.loader.load_custom_geojson(
                core,
                method_name,
                path,
                coordinate_space="COMET",
            )

            if hasattr(self, "cell_quant_tab"):
                self.cell_quant_tab.refresh()

            self._add_custom_segmentation_layer(
                core,
                method_name,
            )

            row = CellBoundaryRow(
                self.sv,
                self.loader,
                display_name=method_name,
                layer_suffix=f"segmentation::{method_name}",
            )

            row.color = "cyan"

            row.color_btn.setStyleSheet(
                "color: cyan;"
                "font-weight: bold;"
                "font-size: 16px;"
            )

            row.layers_tab = self

            self.seg_rows_layout.addWidget(
                row
            )

            self.segmentation_rows.setdefault(
                core,
                {}
            )[method_name] = row

            row.chk.setChecked(True)

        except Exception as e:
            logger.exception(
                "Failed to import segmentation"
            )

            QtWidgets.QMessageBox.warning(
                self,
                "Import failed",
                str(e)
            )


    ## User-defined segmentation helper
    def _add_custom_segmentation_layer(
        self,
        core_id,
        method_name,
    ):
    
        seg = (
            self.loader
            .custom_segmentations
            .get(core_id, {})
            .get(method_name)
        )

        if seg is None:
            return

        M_com = self._adjust_matrix(
            self.loader.alignment_matrices_comet_raw.get(
                core_id
            ),
            core_id,
        )

        self.sv.add_boundary_layer(
            core_id,
            seg["shapes"],
            name=f"segmentation::{method_name}",
            color="cyan",
            visible=True,
            affine=M_com,
        )


    ## Refreshing gene panels (favourite, active, recent)
    def _apply_saved_gene_color(
        self,
        row,
        gene,
    ):
        """
        Restore the saved colour for a gene row.
        """

        if gene not in self.active_gene_colors:
            return

        row.color = self.active_gene_colors[gene]

        row.color_btn.setStyleSheet(
            f"color: {row.color};"
            "font-weight: bold;"
            "font-size: 16px;"
        )


    ## Refresh function (repopulates layout)
    def _refresh_gene_panels(self):
        if self._refreshing_gene_panels:
            return
        
        self._refreshing_gene_panels = True

        # Favourite genes panel
        self._clear_qt_layout(
            self.favourite_genes_layout
        )

        try:
            for gene in DEFAULT_FAVOURITE_GENES:

                if gene not in self.all_genes:
                    continue

                row = TranscriptChannelRow(
                    gene,
                    self.sv,
                    self.loader,
                )

                row.layers_tab = self

                if gene in self.active_genes:
                    row.vis_chk.blockSignals(True)
                    row.vis_chk.setChecked(True)
                    row.vis_chk.blockSignals(False)

                self._apply_saved_gene_color(row, gene,)

                self.favourite_genes_layout.addWidget(row)

            # Active genes panel
            self._clear_qt_layout(
                self.active_genes_layout
            )

            for gene in sorted(self.active_genes):

                row = TranscriptChannelRow(
                    gene,
                    self.sv,
                    self.loader,
                )

                row.layers_tab = self

                self._apply_saved_gene_color(row, gene,)

                row.vis_chk.blockSignals(True)
                row.vis_chk.setChecked(True)
                row.vis_chk.blockSignals(False)

                self.active_genes_layout.addWidget(
                    row
                )

            # Recent genes panel
            self._clear_qt_layout(
                self.recent_genes_layout
            )

            # Ensures favourite genes don't clutter "recent"
            for gene in self.recent_genes:

                if gene in DEFAULT_FAVOURITE_GENES:
                    continue

                row = TranscriptChannelRow(
                    gene,
                    self.sv,
                    self.loader,
                )

                row.layers_tab = self

                self._apply_saved_gene_color(row, gene,)

                self.recent_genes_layout.addWidget(row)
        finally:
            self._refreshing_gene_panels = False


# Minimal helper tabs (kept for completeness)
class CellQuantificationTab(QWidget):
    def __init__(self, sv, parent=None):
        super().__init__(parent)

        self.sv = sv
        self.loader = None

        layout = QVBoxLayout(self)

        # Instructions
        self.help_label = QLabel(
            "Workflow:\n"
            "1. Go to the Layers tab.\n"
            "2. Enable Xenium cell boundaries or import a custom segmentation.\n"
            "3. Return here and select the core and segmentation.\n"
            "4. Run cell quantification (proteins + transcripts)."
        )

        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

        # Core dropdown
        layout.addWidget(QLabel("Core"))
        self.core_combo = QComboBox()

        self.core_combo.currentTextChanged.connect(
            lambda _: self.refresh()
        )

        layout.addWidget(self.core_combo)

        # Segmentation
        layout.addWidget(QLabel("Segmentation"))
        self.seg_combo = QComboBox()
        layout.addWidget(self.seg_combo)

        # Run protein quantification
        self.run_btn = QPushButton("Run Cell Quantification")
        layout.addWidget(self.run_btn)
        self.run_btn.clicked.connect(self._run_quantification)

        self.result_label = QLabel("No quantification run")
        layout.addWidget(self.result_label)

        # Export CSV
        self.export_btn = QPushButton("Export Quantification CSV")
        layout.addWidget(self.export_btn)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)

        # Export Thresholds
        self.export_thresholds_btn = QPushButton("Export Thresholds JSON")
        layout.addWidget(self.export_thresholds_btn)
        self.export_thresholds_btn.setEnabled(True)
        self.export_thresholds_btn.clicked.connect(self._export_thresholds)


        self.last_core = None
        self.last_method = None

        # Stretch (keep at end)
        layout.addStretch()


    ## Give access to the loader
    def set_loader(self, loader):
        self.loader = loader
        self.core_combo.clear()
        self.core_combo.addItems(
                sorted(
                    loader.manifest.cores.keys()
                )
            )
        self.refresh()


    def _run_quantification(self):
        core = self.core_combo.currentText()

        method_name = self.seg_combo.currentData()

        if method_name is None:
            method_name = self.seg_combo.currentText()

        if not method_name:
                self.result_label.setText(
                    "No segmentation selected"
                )
                return

        try:
            df = self.loader.quantify_comet_segmentation(
                core,
                method_name,
            )

            self.last_core = core
            self.last_method = method_name

            self.result_label.setText(
                        f"{len(df):,} cells quantified in {core}"
                    )

            # button becomes available only after quantification has been run
            self.export_btn.setEnabled(True)

        except Exception as e:

            self.result_label.setText(
                f"Error: {e}"
            )


    ## Export CSV file
    def _export_csv(self):
        if self.last_core is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Quantification",
            f"{self.last_core}_quantification_cells.csv",
            "CSV (*.csv)"
        )

        if not path:
            return

        self.loader.export_segmentation_quantification(
            self.last_core,
            self.last_method,
            path,
        )


    ## Export thresholds as JSON file
    def _export_thresholds(self):

        core = self.core_combo.currentText()

        if not core:
            return

        thresholds = (
            self.loader.comet_thresholds.get(
                core,
                {}
            )
        )

        output = {
            "core": core,
            "version": "1.0",
            "proteins": {},
        }

        for marker, values in thresholds.items():

            vmin, vmax = values

            output["proteins"][marker] = {
                "threshold": float(vmin),
                "display_max": float(vmax),
                "method": "manual",
            }

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Thresholds",
            f"{core}_thresholds.json",
            "JSON (*.json)"
        )

        if not path:
            return

        with open(path, "w") as f:

            json.dump(
                output,
                f,
                indent=4,
            )

        self.result_label.setText(
            f"Exported thresholds for {core}"
        )


    def refresh(self):

        if self.loader is None:
            return

        core = self.core_combo.currentText()

        self.seg_combo.clear()

        self.seg_combo.clear()

        for name in sorted(
            self.loader
            .custom_segmentations
            .get(core, {})
        ):
            # Change display name for segmentation calculation
            display_name = (
                "Xenium (default)"
                if name == "cells"
                else name
            )

            self.seg_combo.addItem(
                display_name,
                userData=name,
            )

        # Ensure "Run Cell Quantification" only when a segmentation is available
        has_segmentation = (
            self.seg_combo.count() > 0
        )

        self.run_btn.setEnabled(
            has_segmentation
        )

        if not has_segmentation:
            self.result_label.setText(
                "No segmentation available"
            )


# ---------------------------------------------------------------------------
# Launch helper (exposed symbol)
# ---------------------------------------------------------------------------

def launch():
    """
    Create the UnumLocalia control tabs and attach them to the viewer.
    This function is intentionally minimal: it expects a `SpatialViewer` class
    available at unumlocalia.viewer.SpatialViewer with the methods used below.
    """
    try:
        from unumlocalia.viewer import SpatialViewer
    except Exception as e:
        logger.exception("Could not import SpatialViewer: %s", e)
        raise

    sv = SpatialViewer("UnumLocalia")

    data_tab = DataTab()
    layers_tab = LayersTab(sv)
    cell_quant_tab = CellQuantificationTab(sv)

    layers_tab.cell_quant_tab = cell_quant_tab

    # wire dataset_loaded signal to layers_tab.populate
    data_tab.dataset_loaded.connect(layers_tab.populate)
    data_tab.dataset_loaded.connect(cell_quant_tab.set_loader)

    ## Save session
    def save_session():
        if layers_tab.loader is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Session",
            "analysis.ulproj",
            "UnumLocalia Project (*.ulproj)"
        )

        if not path:
            return

        ui_state = layers_tab.export_state()

        layers_tab.loader.save_session(
            path,
            ui_state,
        )


    ## Load session
    def load_session():

        path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Session",
                "",
                "UnumLocalia Project (*.ulproj)"
            )


        if not path:
            return

        import json

        with open(path, "r") as f:
            session = json.load(f)

        dataset_folder = session.get(
            "dataset_folder"
        )

        # Automatically reload dataset
        from unumlocalia.io import DatasetLoader

        # Add loading message
        import os

        data_tab.log_area.setText(
            f"Loading session: {os.path.basename(path)}"
        )

        QApplication.processEvents()

        loader = DatasetLoader(
            dataset_folder
        ).load(
            do_load_transcripts=False,
            load_boundaries=True,
            load_he=True,
            load_comet=True,
            load_adata=False,
        )

        loader.comet_thresholds = session.get(
            "comet_thresholds",
            {}
        )

        layers_tab.populate(loader)
        cell_quant_tab.set_loader(loader)
        layers_tab.loader = loader

        #
        # Restore imported segmentations
        #
        for core, methods in (
            session.get(
                "segmentations",
                {}
            ).items()
        ):

            for method_name, seg_path in (
                methods.items()
            ):

                if (
                    core not in
                    layers_tab.loader.custom_segmentations
                    or
                    method_name not in
                    layers_tab.loader.custom_segmentations.get(
                        core,
                        {}
                    )
                ):

                    try:

                        layers_tab.loader.load_custom_geojson(
                            core,
                            method_name,
                            seg_path,
                        )

                        layers_tab._add_custom_segmentation_layer(
                            core,
                            method_name,
                        )

                        row = CellBoundaryRow(
                            sv,
                            layers_tab.loader,
                            display_name=method_name,
                            layer_suffix=f"segmentation::{method_name}",
                        )

                        row.layers_tab = layers_tab

                        layers_tab.seg_rows_layout.addWidget(
                            row
                        )

                        layers_tab.segmentation_rows.setdefault(
                            core,
                            {}
                        )[method_name] = row

                        row.chk.setChecked(True)

                    except Exception:
                        logger.exception(
                            "Failed restoring segmentation"
                        )

        layers_tab.import_state(
            session.get(
                "ui",
                {}
            )
        )

        # Add message once session is loaded (stating what was loaded)
        data_tab.log_area.setText(
            loader.manifest.summary()
        )

        layers_tab._on_core_swapped(layers_tab.core_combo.currentText())

    data_tab.session_save_requested.connect(save_session)
    data_tab.session_load_requested.connect(load_session)

    tabs = QTabWidget()
    tabs.addTab(data_tab, "Data")
    tabs.addTab(layers_tab, "Layers")
    tabs.addTab(cell_quant_tab, "Cell Quantification")

    # attach to viewer window (viewer implementation must provide add_dock_widget)
    try:
        sv.viewer.window.add_dock_widget(tabs, name="UnumLocalia Controls", area="right")
    except Exception:
        # fallback: try to add via viewer API if available
        try:
            sv.viewer.add_dock_widget(tabs, name="UnumLocalia Controls", area="right")
        except Exception:
            logger.debug("Could not attach dock widget via known APIs; continuing.")

    sv.show()
    sv.run()