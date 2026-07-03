# spatialbench/widgets.py
"""
PyQt-based dock widget panels that integrate with the Napari viewer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from spatialbench.utils import safe_read_parquet, shapes_to_napari

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
    logger.warning("Qt not available. SpatialBench GUI cannot be displayed.")


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
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        for child in self.findChildren(QWidget):
            if child is not self:
                child.setVisible(checked)


# ---------- Cell mask class ----------
class CellMaskRow(QWidget):
    """
    Minimal cell-mask control: checkbox + info label.
    Loads rasterized mask synchronously via loader.load_geojson_mask(core).
    """

    def __init__(self, sv, loader, parent=None):
        super().__init__(parent)
        self.sv = sv
        self.loader = loader

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)

        self.chk = QCheckBox("Cell masks")
        self.chk.setChecked(False)
        self.chk.toggled.connect(self._on_toggle)
        row.addWidget(self.chk)

        self.info_lbl = QLabel("")
        self.info_lbl.setStyleSheet("color: gray;")
        row.addWidget(self.info_lbl)

        row.addStretch()
        self.setLayout(row)

    def sync_to_core(self, core: str):
        """Update checkbox state when core changes."""
        if not core:
            self.chk.setEnabled(False)
            self.info_lbl.setText("")
            return

        self.chk.setEnabled(True)
        lname = f"{core}::cell_mask"
        layer = None
        try:
            layer = self.sv.viewer.layers.get(lname)
        except Exception:
            layer = None

        if layer is not None:
            self.chk.blockSignals(True)
            self.chk.setChecked(bool(layer.visible))
            self.chk.blockSignals(False)
            meta = getattr(layer, "metadata", {}) or {}
            nlabels = meta.get("n_labels")
            if nlabels is not None:
                self.info_lbl.setText(f"labels: {nlabels}")
            else:
                self.info_lbl.setText("")
        else:
            self.chk.blockSignals(True)
            self.chk.setChecked(False)
            self.chk.blockSignals(False)
            self.info_lbl.setText("")

    def _on_toggle(self, checked: bool):
        core = getattr(self.sv, "active_core", None)
        if not core:
            return

        lname = f"{core}::cell_mask"

        # turning off: hide if present
        if not checked:
            try:
                layer = self.sv.viewer.layers.get(lname)
            except Exception:
                layer = None
            if layer is not None:
                try:
                    layer.visible = False
                except Exception:
                    pass
            return

        # turning on: load mask and add labels layer
        try:
            self.info_lbl.setText("loading masks...")
            QApplication.processEvents()
        except Exception:
            pass

        try:
            mask, id_map = self.loader.load_geojson_mask(core, overwrite=False)
        except Exception as e:
            logger.exception("Failed to load geojson mask for %s: %s", core, e)
            self.chk.blockSignals(True)
            self.chk.setChecked(False)
            self.chk.blockSignals(False)
            self.info_lbl.setText("mask load failed")
            return

        if mask is None:
            self.info_lbl.setText("no mask")
            self.chk.blockSignals(True)
            self.chk.setChecked(False)
            self.chk.blockSignals(False)
            return

        try:
            try:
                # Prefer normalized (translation-zeroed) COMET matrix if present, else raw
                M_com_raw = None
                try:
                    M_com_raw = self.loader.alignment_matrices_comet.get(core)
                except Exception:
                    M_com_raw = None
                if M_com_raw is None:
                    try:
                        M_com_raw = self.loader.alignment_matrices_comet_raw.get(core)
                    except Exception:
                        M_com_raw = None

                # Convert to viewer / napari affine if helper exists
                M_for_viewer = None
                if M_com_raw is not None and hasattr(self.sv, "_convert_affine"):
                    try:
                        M_for_viewer = self.sv._convert_affine(M_com_raw)
                    except Exception:
                        M_for_viewer = None

                # If no conversion helper, try to coerce a 2x3 affine for napari
                if M_for_viewer is None and M_com_raw is not None:
                    try:
                        M_arr = np.asarray(M_com_raw, dtype=float)
                        if M_arr.shape == (3, 3):
                            # napari accepts 3x3 or 2x3; convert to 2x3
                            M_for_viewer = M_arr[:2, :]
                        elif M_arr.shape == (2, 3):
                            M_for_viewer = M_arr
                        else:
                            # try reshape fallback
                            M_for_viewer = M_arr.reshape(3, 3)[:2, :]
                    except Exception:
                        M_for_viewer = None

                # Optional: sanity-check shapes against COMET channel (if available)
                try:
                    # prefer channel 0 as reference; don't raise if missing
                    if hasattr(self.loader, "get_comet_channel"):
                        ref = None
                        try:
                            ref = self.loader.get_comet_channel(core, channel_index=0, use_cache=True)
                        except Exception:
                            ref = None
                        if ref is not None:
                            # if shapes differ, log a warning (no crash)
                            if getattr(ref, "shape", None) != getattr(mask, "shape", None):
                                logger.warning("Mask shape %s != COMET shape %s for core %s", getattr(mask, "shape", None), getattr(ref, "shape", None), core)
                except Exception:
                    pass

                # Add labels layer using viewer helper if available, passing affine if we computed one
                if hasattr(self.sv, "add_label_layer"):
                    try:
                        lbl_layer = self.sv.add_label_layer(core=core, labels=mask, name=lname, affine=M_for_viewer, visible=True)
                    except Exception:
                        # fallback to napari API; pass affine if available
                        try:
                            lbl_layer = self.sv.viewer.add_labels(mask, name=lname, visible=True, affine=M_for_viewer)
                        except Exception:
                            lbl_layer = self.sv.viewer.add_labels(mask, name=lname, visible=True)
                else:
                    # no helper: use napari directly
                    try:
                        lbl_layer = self.sv.viewer.add_labels(mask, name=lname, visible=True, affine=M_for_viewer)
                    except Exception:
                        lbl_layer = self.sv.viewer.add_labels(mask, name=lname, visible=True)

                # metadata and UI update
                lbl_layer.metadata = getattr(lbl_layer, "metadata", {}) or {}
                lbl_layer.metadata["core"] = core
                lbl_layer.metadata["modality"] = "cell_mask"
                lbl_layer.metadata["sb_source"] = "geojson_mask"
                lbl_layer.metadata["label_to_cell_id"] = id_map
                lbl_layer.metadata["n_labels"] = int(len(id_map)) if id_map is not None else 0
                self.info_lbl.setText(f"labels: {lbl_layer.metadata.get('n_labels', 0)}")
            except Exception as e:
                logger.exception("Failed to add labels layer for %s: %s", core, e)
                self.chk.blockSignals(True)
                self.chk.setChecked(False)
                self.chk.blockSignals(False)
                self.info_lbl.setText("add failed")
        except Exception as e:
            logger.exception("Failed to add labels layer for %s: %s", core, e)
            self.chk.blockSignals(True)
            self.chk.setChecked(False)
            self.chk.blockSignals(False)
            self.info_lbl.setText("add failed")


class CometChannelRow(QWidget):
    """Row controls for a single COMET marker across all cores."""

    _COLORMAPS = ["green", "red", "cyan", "magenta", "yellow", "blue", "gray", "hot", "viridis"]

    def __init__(self, marker: str, sv, loader, parent=None):
        super().__init__(parent)
        self.marker = marker
        self.sv = sv
        self.loader = loader

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(4)

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
            self._on_change()

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

        self.sv.update_comet_layer(
            self.marker,
            core=core,
            colormap=self.cmap_cb.currentText(),
            vmin=vmin,
            vmax=vmax,
            opacity=self.op_sl.value() / 100.0,
            visible=self.vis_chk.isChecked()
        )


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
      SPATIALBENCH_DEBUG environment variable.
    """
    def __init__(self, gene: str, sv, loader, parent=None):
        super().__init__(parent)
        self.gene = gene
        self.sv = sv
        self.loader = loader
        self.color = "yellow"

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(4)

        self.vis_chk = QCheckBox(gene)
        self.vis_chk.setFixedWidth(200)
        self.vis_chk.setChecked(False)
        self.vis_chk.toggled.connect(self._on_change)
        row.addWidget(self.vis_chk)

        self.color_btn = QPushButton("■")
        self.color_btn.setStyleSheet(f"color: {self.color}; font-weight: bold; font-size: 16px;")
        self.color_btn.setFixedWidth(30)
        self.color_btn.clicked.connect(self._pick_color)
        row.addWidget(self.color_btn)
        row.addStretch()
        self.setLayout(row)

    # Backwards-compatible debug helpers (both names supported)
    def _debug(self, *args):
        try:
            import os
            if os.environ.get("SPATIALBENCH_DEBUG"):
                print("TRANSCRIPT_DBG:", *args, flush=True)
        except Exception:
            pass

    def _debug_print(self, *args):
        # keep the older name used elsewhere; delegate to _debug
        self._debug(*args)

    def _pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.color = c.name()
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


                    self._debug_print("IMG_TRANSFORM_DBG:", core, "arr_shape", getattr(arr, "shape", None), "arr", arr.tolist(), "img_layer_shape", getattr(img_layer, "data", getattr(img_layer, "shape", None)))



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

                    
                    # DEBUG: show pixel size and raw coords used by the widget
                    try:
                        print("DBG_PX: widget px_um:", getattr(self.loader, "xenium_pixel_size_um", None))
                        print("DBG_PX: coords raw sample (first 5) [X,Y]:", coords[:5].tolist())
                    except Exception as _e:
                        print("DBG_PX: failed to print px_um/coords:", _e)


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





            # DEBUG: widget sees mapped coords and exported matrix
        try:
            import os, numpy as _np
            if os.environ.get("SPATIALBENCH_DEBUG"):
                print("WIDGET_DBG: coords_mapped_sample", coords_mapped[:5].tolist(), flush=True)
                print("WIDGET_DBG: exported_matrix_M_e", _np.asarray(M_exported if 'M_exported' in locals() else self.loader.alignment_matrices_comet_raw.get(core)).tolist(), flush=True)
        except Exception:
            pass
        



        # WIDGET DEBUG: confirm coords_mapped and matrix used
        try:
            import os, numpy as _np
            if os.environ.get("SPATIALBENCH_DEBUG"):
                print("WIDGET_DBG: core", core, "coords_mapped_sample:", coords_mapped[:5].tolist(), flush=True)
                # show which matrix branch was used (fitted vs exported)
                if 'M_f' in locals():
                    print("WIDGET_DBG: used_fitted_matrix:", _np.asarray(M_f).tolist(), flush=True)
                if 'M_e' in locals():
                    print("WIDGET_DBG: used_exported_matrix:", _np.asarray(M_e).tolist(), flush=True)
        except Exception:
            pass




        # DEBUG: inspect exported-matrix branch internals (paste immediately after coords_mapped is set)
        try:
            import numpy as _np
            # coords_pix and H should be in scope where coords_mapped was computed
            print("DBG_INT: coords_pix sample:", None if 'coords_pix' not in locals() else _np.asarray(coords_pix)[:5].tolist())
            if 'M_e' in locals() and M_e is not None:
                Me = _np.asarray(M_e, dtype=float)
                print("DBG_INT: M_e shape:", Me.shape)
                print("DBG_INT: M_e (first 3x3):", Me.reshape(3,3).tolist() if Me.size in (6,9) or Me.shape==(3,3) else Me.tolist())
                try:
                    Minv = _np.linalg.inv(Me)
                    print("DBG_INT: M_inv (first 3x3):", Minv[:3,:3].tolist())
                except Exception as _e:
                    print("DBG_INT: M_inv compute failed:", _e)
            else:
                print("DBG_INT: no M_e in locals or M_e is None")
            print("DBG_INT: coords_mapped (widget) sample:", _np.asarray(coords_mapped)[:5].tolist())
            # Compare to the exported-inverse mapping computed in REPL (if you ran it earlier)
            try:
                coords_by_export_inv_local = (H @ _np.linalg.inv(Me).T)[:, :2]
                print("DBG_INT: coords_by_export_inv sample:", coords_by_export_inv_local[:5].tolist())
                print("DBG_INT: coords_mapped equals exported-inv (allclose):", _np.allclose(_np.asarray(coords_mapped)[:5], coords_by_export_inv_local[:5], atol=1e-6))
            except Exception as _e:
                print("DBG_INT: compare to exported-inv failed:", _e)
        except Exception as _e:
            print("DBG_INT: debug block failed:", _e)



        # --- DEBUG: record which mapping branch and the matrix used ---
        try:
            import numpy as _np
            used_exported = 'M_e' in locals() and ('M_inv' in locals() or ('M_e' in locals() and M_e is not None))
            used_fitted = 'M_use' in locals() and (M_use is not None)
            used_micron2px = not (used_exported or used_fitted)
            print("DBG_CHOICE: used_exported_inv:", bool(used_exported))
            print("DBG_CHOICE: used_fitted:", bool(used_fitted))
            print("DBG_CHOICE: used_micron2px:", bool(used_micron2px))
            print("DBG_CHOICE: M_e (exported) shape/vals:", None if 'M_e' not in locals() or M_e is None else _np.asarray(M_e, dtype=float).shape, None if 'M_e' not in locals() or M_e is None else _np.asarray(M_e, dtype=float).tolist()[:9])
            print("DBG_CHOICE: M_use (fitted) shape/vals:", None if 'M_use' not in locals() or M_use is None else _np.asarray(M_use, dtype=float).shape, None if 'M_use' not in locals() or M_use is None else _np.asarray(M_use, dtype=float).tolist()[:9])
            print("DBG_CHOICE: coords_mapped sample (first 5):", coords_mapped[:5].tolist())
        except Exception as _dbg_e:
            print("DBG_CHOICE: debug print failed:", _dbg_e)




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

        self._debug_print("core", core, "coords_mapped[:3]", coords_mapped[:3].tolist() if coords_mapped.size else None)
        self._debug_print("img_layer present", img_layer is not None, "M_h present", M_h is not None)

        # Remove any existing layer first to avoid duplicates
        self._remove_layer_by_name(layer_name)

        # --- Pre-warp into viewer/world coords and add points layer (avoid Napari Affine fragility) ---
        try:
            if M_h is not None:
                H_pts = np.hstack([coords_mapped, np.ones((coords_mapped.shape[0], 1))])
                coords_world = (H_pts @ np.asarray(M_h, dtype=float).T)[:, :2]
            else:
                coords_world = coords_mapped.copy()
        except Exception:
            coords_world = coords_mapped.copy()

        # Napari expects (row, col) ordering -> reverse x,y to y,x
        pts_for_napari = coords_world[:, ::-1]



        # DEBUG: final transform and points passed to Napari
        try:
            import os, numpy as _np
            if os.environ.get("SPATIALBENCH_DEBUG"):
                # M_img is the Napari image-layer matrix if available; M_h is the authoritative matrix computed earlier
                print("FINAL_DBG: M_img", None if 'M_img' not in locals() else (_np.asarray(M_img).tolist()), flush=True)
                print("FINAL_DBG: M_h", None if 'M_h' not in locals() else (_np.asarray(M_h).tolist()), flush=True)
                # sample of points passed to viewer (either coords_mapped or pts_for_napari / pts_world)
                if 'pts_for_napari' in locals():
                    print("FINAL_DBG: pts_for_napari_sample", pts_for_napari[:5].tolist(), flush=True)
                elif 'coords_mapped' in locals():
                    print("FINAL_DBG: coords_mapped_sample", coords_mapped[:5].tolist(), flush=True)
                elif 'pts_world' in locals():
                    print("FINAL_DBG: pts_world_sample", pts_world[:5].tolist(), flush=True)
        except Exception:
            pass



        # WIDGET DEBUG: final points and candidate affines
        try:
            import os, numpy as _np
            if os.environ.get("SPATIALBENCH_DEBUG"):
                print("FINAL_DBG: coords_mapped[:5]", coords_mapped[:5].tolist(), flush=True)
                # M_h is the authoritative 3x3 matrix computed by _compute_authoritative_matrix
                print("FINAL_DBG: M_h", None if 'M_h' not in locals() else _np.asarray(M_h).tolist(), flush=True)
                # M_img is the Napari image-layer matrix if discovered
                print("FINAL_DBG: M_img", None if 'M_img' not in locals() else _np.asarray(M_img).tolist(), flush=True)
        except Exception:
            pass





        try:
            created_layer = self.sv.add_transcript_layer(core, self.gene, pts_for_napari, self.color, True)
        except Exception:
            try:
                created_layer = self.sv.viewer.add_points(pts_for_napari, name=layer_name, size=10, face_color=self.color, visible=True)
            except Exception as e:
                logger.exception("Failed to add transcript layer for %s / %s: %s", core, self.gene, e)
                return

        if created_layer is None:
            return



        # WIDGET DEBUG: layer metadata after creation
        try:
            import os, numpy as _np
            if os.environ.get("SPATIALBENCH_DEBUG"):
                try:
                    meta = getattr(created_layer, "metadata", {}) or {}
                    print("FINAL_DBG: created_layer.metadata.affine_matrix", meta.get("affine_matrix"), flush=True)
                except Exception:
                    print("FINAL_DBG: created_layer metadata unavailable", flush=True)
        except Exception:
            pass




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
        size_val = 6.0
        try:
            if hasattr(self.sv, "_transcript_size"):
                size_val = float(self.sv._transcript_size)
            elif hasattr(self.sv, "transcript_size"):
                size_val = float(getattr(self.sv, "transcript_size"))
            else:
                size_val = 25.0
        except Exception:
            size_val = 25.0

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
        try:
            if hasattr(self.sv, "reset_view"):
                self.sv.reset_view()
        except Exception:
            pass











class DataTab(QWidget):
    dataset_loaded = Signal(object)  # emits DatasetLoader

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        group = QGroupBox("Dataset Loading")
        g_layout = QVBoxLayout(group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select SpatialBench dataset folder...")

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

        from spatialbench.io import DatasetLoader
        try:
            self.log_area.setText("Loading manifest and scanning files...")
            QApplication.processEvents()
            loader = DatasetLoader(path).load(
                do_load_transcripts=False,    # keep genes list only
                load_boundaries=False,        # skip boundaries initially
                load_he=True,                 # H&E
                load_comet=True,              # COMET
                load_adata=False,             # skip AnnData
            )
            self.log_area.setText(loader.manifest.summary())
            self.dataset_loaded.emit(loader)
        except Exception as e:
            self.log_area.setText(f"Error loading dataset:\n{e}")
            logger.exception("Error loading dataset")


class LayersTab(QWidget):
    def __init__(self, sv, parent=None):
        super().__init__(parent)
        self.sv = sv
        self.loader = None

        self.comet_rows: List[CometChannelRow] = []
        self.gene_rows: List[TranscriptChannelRow] = []

        layout = QVBoxLayout(self)

        # 1. Core Selection
        core_group = QGroupBox("Active Core")
        c_layout = QHBoxLayout(core_group)
        self.core_combo = QComboBox()
        self.core_combo.currentTextChanged.connect(self._on_core_swapped)
        c_layout.addWidget(self.core_combo)
        layout.addWidget(core_group)

        # 2. H&E Control
        self.he_group = CollapsibleGroup("H&E Base")
        he_layout = QFormLayout(self.he_group)
        self.he_vis = QCheckBox()
        self.he_vis.setChecked(True)
        self.he_vis.toggled.connect(self._update_he)
        self.he_op = QSlider(Qt.Horizontal)
        self.he_op.setRange(0, 100)
        self.he_op.setValue(100)
        self.he_op.valueChanged.connect(self._update_he)
        he_layout.addRow("Visible", self.he_vis)
        he_layout.addRow("Opacity", self.he_op)
        layout.addWidget(self.he_group)

        # 3. Transcript Size Global
        tx_size_layout = QHBoxLayout()
        tx_size_layout.addWidget(QLabel("Global Dot Size:"))
        self.tx_size = QDoubleSpinBox()
        self.tx_size.setRange(1.0, 100.0)
        self.tx_size.setValue(25.0)
        self.tx_size.setSingleStep(1.0)
        # connect to handler that updates viewer and existing layers
        self.tx_size.valueChanged.connect(self._on_tx_size_changed)
        tx_size_layout.addWidget(self.tx_size)
        tx_size_layout.addStretch()
        layout.addLayout(tx_size_layout)

        # 4. Scrollable Markers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll)

    def _on_tx_size_changed(self, v):
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

        # COMET controls
        if loader.proteins:
            lbl = QLabel("COMET Proteins")
            lbl.setFont(QFont("", 9, QFont.Bold))
            self.scroll_layout.addWidget(lbl)

            for p in loader.proteins:
                row = CometChannelRow(p, self.sv, self.loader)
                self.comet_rows.append(row)
                self.scroll_layout.addWidget(row)

        # Gene controls
        if loader.genes:
            lbl2 = QLabel("Xenium Genes")
            lbl2.setFont(QFont("", 9, QFont.Bold))
            lbl2.setStyleSheet("margin-top: 10px;")
            self.scroll_layout.addWidget(lbl2)

            for g in loader.genes:
                row = TranscriptChannelRow(g, self.sv, self.loader)
                self.gene_rows.append(row)
                self.scroll_layout.addWidget(row)

        # Add Cell Mask control row (single global row that operates on active core)
        self.mask_row = CellMaskRow(self.sv, loader)
        self.scroll_layout.addWidget(self.mask_row)

        self.scroll_layout.addStretch()

        # Add everything to viewer
        self._add_all_to_viewer()

        # Set active core
        self.core_combo.setCurrentText(cores[0])
        self._on_core_swapped(cores[0])
        self.sv.reset_view()

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

            # Boundaries (optional, if preloaded)
            if core_id in loader.cell_boundaries_df:
                df_cb = loader.cell_boundaries_df[core_id]
                if "vertex_x" in df_cb.columns:
                    shapes = shapes_to_napari(df_cb)
                    self.sv.add_boundary_layer(core_id, shapes, name="cells", color="white", visible=False)

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
            if getattr(self, "mask_row", None) is not None:
                try:
                    self.mask_row.sync_to_core(core)
                except Exception:
                    pass
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


# Minimal helper tabs (kept for completeness)
class AnalysisTab(QWidget):
    def __init__(self, sv, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Analysis controls will go here."))
        layout.addStretch()


class BenchmarkTab(QWidget):
    def __init__(self, sv, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Benchmarking controls will go here."))
        layout.addStretch()


# ---------------------------------------------------------------------------
# Launch helper (exposed symbol)
# ---------------------------------------------------------------------------

def launch():
    """
    Create the SpatialBench control tabs and attach them to the viewer.
    This function is intentionally minimal: it expects a `SpatialViewer` class
    available at spatialbench.viewer.SpatialViewer with the methods used below.
    """
    try:
        from spatialbench.viewer import SpatialViewer
    except Exception as e:
        logger.exception("Could not import SpatialViewer: %s", e)
        raise

    sv = SpatialViewer("SpatialBench")

    data_tab = DataTab()
    layers_tab = LayersTab(sv)
    analysis_tab = AnalysisTab(sv)
    benchmark_tab = BenchmarkTab(sv)

    # wire dataset_loaded signal to layers_tab.populate
    data_tab.dataset_loaded.connect(layers_tab.populate)

    tabs = QTabWidget()
    tabs.addTab(data_tab, "Data")
    tabs.addTab(layers_tab, "Layers")
    tabs.addTab(analysis_tab, "Analysis")
    tabs.addTab(benchmark_tab, "Benchmark")

    # attach to viewer window (viewer implementation must provide add_dock_widget)
    try:
        sv.viewer.window.add_dock_widget(tabs, name="SpatialBench Controls", area="right")
    except Exception:
        # fallback: try to add via viewer API if available
        try:
            sv.viewer.add_dock_widget(tabs, name="SpatialBench Controls", area="right")
        except Exception:
            logger.debug("Could not attach dock widget via known APIs; continuing.")

    sv.show()
    sv.run()