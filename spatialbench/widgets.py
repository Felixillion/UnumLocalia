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
            # add labels layer (use viewer helper if available)
            try:
                lbl_layer = self.sv.add_label_layer(core=core, labels=mask, name=lname, visible=True)
            except Exception:
                lbl_layer = self.sv.viewer.add_labels(mask, name=lname, visible=True)
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
    """Row controls for a single Gene transcript across all cores."""
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

    def _pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.color = c.name()
            self.color_btn.setStyleSheet(f"color: {self.color}; font-weight: bold; font-size: 16px;")
            # update existing layer color live via viewer API
            try:
                core = getattr(self.sv, "active_core", None)
                # prefer viewer API
                try:
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
                    # prefer viewer API if available
                    try:
                        self.sv.remove_layer(layer)
                    except Exception:
                        # fallback to viewer.layers removal
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
                                # try viewer API removal first
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
            # Some viewers may expose a remove_transcript_layer API
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


    def _on_change(self, *args):
        core = getattr(self.sv, "active_core", None)
        if not core:
            return

        layer_name = f"{core}::transcripts::{self.gene}"

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

        # read raw coords (x,y) in Xenium µm
        coords = df_gene[["x_location", "y_location"]].to_numpy(dtype=np.float64)
        if coords.size == 0:
            return

        # Prefer the precomputed transcript affine (Xenium µm -> COMET pixels) if available
        M_use = self.loader.transcript_affine_by_core.get(core)
        if M_use is not None:
            try:
                coords_mapped = _apply_affine_to_coords(M_use, coords)  # coords_mapped are COMET pixels (x,y)
            except Exception:
                logger.debug("Affine transform failed for transcripts; using raw coords")
                coords_mapped = coords.copy()
        else:
            # Fallback: micron->Xenium pixels -> COMET pixels using exported matrix
            try:
                px_um = self.loader.xenium_pixel_size_um if self.loader.xenium_pixel_size_um is not None else 0.2125
                M_exported = self.loader.alignment_matrices_comet_raw.get(core)
                if M_exported is not None:
                    # ensure 3x3
                    M_e = np.asarray(M_exported, dtype=float)
                    if M_e.shape == (2, 3):
                        M_e = np.vstack([M_e, [0.0, 0.0, 1.0]])
                    # micron -> Xenium pixels
                    coords_pix = coords / px_um
                    H = np.hstack([coords_pix, np.ones((coords_pix.shape[0], 1))])
                    try:
                        M_inv = np.linalg.inv(M_e)
                        coords_mapped = (H @ M_inv.T)[:, :2]
                    except Exception:
                        coords_mapped = coords_pix.copy()
                else:
                    # last resort: convert µm -> Xenium pixels and use those as-is
                    px_um = px_um if px_um is not None else 0.2125
                    coords_mapped = coords / px_um
            except Exception:
                coords_mapped = coords.copy()

        # Napari expects (row, col) == (y, x) for Points; we keep coords_mapped as (x,y)
        # AUTOSCALE coords only if absolutely necessary (avoid changing scale if affine exists)
        try:
            img = self.loader.he_arrays.get(core)
            if img is not None and self.loader.transcript_affine_by_core.get(core) is None:
                img_h = img.shape[0]
                img_w = img.shape[1] if img.ndim >= 2 else img.shape[-1]
                coord_min = coords_mapped.min(axis=0)
                coord_max = coords_mapped.max(axis=0)
                coord_range = coord_max - coord_min
                coord_max_dim = max(coord_range[0], coord_range[1], 1.0)
                img_max_dim = max(img_h, img_w, 1.0)
                suggested_scale = img_max_dim / coord_max_dim
                # Apply conservative autoscale only when clearly needed
                if suggested_scale > 1.25:
                    center = (coord_min + coord_max) / 2.0
                    coords_mapped = (coords_mapped - center) * suggested_scale + center
        except Exception:
            pass

        # Remove any existing layer first to avoid duplicates
        self._remove_layer_by_name(layer_name)

        # Convert to viewer ordering (row, col) = (y, x)
        pts_for_viewer = coords_mapped[:, ::-1]

        # Add transcript layer using viewer API (viewer converts (x,y) -> (row,col))
        created_layer = None
        try:
            try:
                created_layer = self.sv.add_transcript_layer(layer_name, pts_for_viewer, color=self.color, visible=True)
            except TypeError:
                created_layer = self.sv.add_transcript_layer(core, self.gene, pts_for_viewer, color=self.color, visible=True)
        except Exception as e:
            logger.exception("Failed to add transcript layer for %s / %s: %s", core, self.gene, e)
            return

        # Ensure created layer is discoverable and set metadata and size
        try:
            if created_layer is not None:
                try:
                    created_layer.name = layer_name
                except Exception:
                    pass
                try:
                    created_layer.metadata = getattr(created_layer, "metadata", {}) or {}
                    created_layer.metadata["canonical_name"] = layer_name
                    created_layer.metadata["sb_source"] = f"transcripts::{self.gene}"
                    created_layer.metadata["modality"] = "xenium"
                except Exception:
                    pass
        except Exception:
            pass

        # Apply global dot size (prefer viewer stored value, else 25)
        size_val = 25.0
        try:
            if hasattr(self.sv, "_transcript_size"):
                size_val = float(self.sv._transcript_size)
            elif hasattr(self.sv, "transcript_size"):
                size_val = float(getattr(self.sv, "transcript_size"))
            else:
                size_val = 25.0
        except Exception:
            size_val = 25.0

        # Set size explicitly on the created layer so live updates can find it
        try:
            try:
                self.sv.update_transcript_layer(self.gene, core=core, color=self.color, size=size_val, visible=True)
            except Exception:
                # fallback: set properties directly on the layer object
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

        # Refresh viewer if available
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