"""
spatialbench.widgets
====================
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
    from qtpy.QtCore import Qt, QThread, Signal, QObject
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
        QListWidget,
        QListWidgetItem,
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
            # update existing layer color if present
            try:
                layer = self.sv._get_layer(f"{self.sv.active_core}::tx::{self.gene}")
                if layer is not None:
                    try:
                        layer.properties["color"] = self.color
                    except Exception:
                        try:
                            layer.color = self.color
                        except Exception:
                            pass
            except Exception:
                pass

    def _remove_layer_by_name(self, name: str):
        """Try several removal/hide strategies to be robust across viewer implementations."""
        try:
            layer = self.sv._get_layer(name)
            if layer is not None:
                try:
                    layer.visible = False
                except Exception:
                    pass
                try:
                    self.sv.remove_layer(layer)
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # fallback: search viewer.layers and remove by name
        try:
            if hasattr(self.sv, "viewer") and hasattr(self.sv.viewer, "layers"):
                for l in list(self.sv.viewer.layers):
                    try:
                        if getattr(l, "name", "") == name:
                            try:
                                self.sv.viewer.layers.remove(l)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_change(self, *args):
        core = getattr(self.sv, "active_core", None)
        if not core:
            return

        layer_name = f"{core}::tx::{self.gene}"

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

        # Apply COMET or H&E affine if available
        M_com = self.loader.alignment_matrices_comet.get(core)
        M_he = self.loader.alignment_matrices_he.get(core)
        M_use = M_com if M_com is not None else M_he
        if M_use is not None:
            try:
                coords = _apply_affine_to_coords(M_use, coords)
            except Exception:
                logger.debug("Affine transform failed for transcripts; using raw coords")

        # Remove any existing layer first to avoid duplicates
        self._remove_layer_by_name(layer_name)

        # Add transcript layer (try preferred signature, then fallback)
        added = False
        try:
            # preferred: (core, gene, coords, color=..., visible=...)
            self.sv.add_transcript_layer(core, self.gene, coords, color=self.color, visible=True)
            added = True
        except TypeError:
            try:
                # fallback: (name, coords, color=..., visible=...)
                self.sv.add_transcript_layer(layer_name, coords, color=self.color, visible=True)
                added = True
            except Exception as e:
                logger.exception("Failed to add transcript layer for %s / %s: %s", core, self.gene, e)
                return
        except Exception as e:
            logger.exception("Failed to add transcript layer for %s / %s: %s", core, self.gene, e)
            return

        if not added:
            return

        # Immediately apply global dot size (prefer viewer stored value, else 25)
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

        try:
            if hasattr(self.sv, "set_transcript_size"):
                self.sv.set_transcript_size(size_val)
        except Exception:
            pass

        # Set size on the layer if possible
        try:
            layer = self.sv._get_layer(layer_name)
            if layer is None and hasattr(self.sv, "viewer") and hasattr(self.sv.viewer, "layers"):
                for l in self.sv.viewer.layers:
                    if getattr(l, "name", "") == layer_name:
                        layer = l
                        break
            if layer is not None:
                try:
                    layer.properties["size"] = size_val
                except Exception:
                    try:
                        layer.size = size_val
                    except Exception:
                        try:
                            layer.metadata["size"] = size_val
                        except Exception:
                            pass
        except Exception:
            pass

        # Force a viewer refresh if available
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
        self.tx_size.valueChanged.connect(lambda v: self.sv.set_transcript_size(v))
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
                
        self.scroll_layout.addStretch()
        
        # Add everything to viewer
        self._add_all_to_viewer()
        
        # Set active core
        self.core_combo.setCurrentText(cores[0])
        self._on_core_swapped(cores[0])
        self.sv.reset_view()

    def _add_all_to_viewer(self):
        self.sv.clear_all_layers()
        loader = self.loader
        for core_id, core_man in loader.manifest.cores.items():
            
            # H&E
            if core_id in loader.he_arrays:
                M_he = loader.alignment_matrices_he.get(core_id)
                he = loader.he_arrays[core_id]
                self.sv.add_he_layer(core_id, he, affine=M_he, visible=False)
                
            # COMET (lazy placeholders)
            if core_id in loader.comet_paths_by_core:
                comet_path = loader.comet_paths_by_core[core_id]
                markers = loader.comet_markers.get(core_id, [])
                M_com = loader.alignment_matrices_comet.get(core_id)

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
        if not core:
            return
        self.sv.set_active_core(core)
        for row in self.comet_rows:
            row.sync_to_core(core)
        self._update_he()
            
    def _update_he(self):
        core = self.core_combo.currentText()
        if not core:
            return
        
        layer = self.sv._get_layer(f"{core}::he")
        if layer:
            layer.metadata["user_visible"] = self.he_vis.isChecked()
            layer.visible = self.he_vis.isChecked()
            layer.opacity = self.he_op.value() / 100.0


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


def launch():
    from spatialbench.viewer import SpatialViewer
    sv = SpatialViewer("SpatialBench v1.0")
    
    data_tab = DataTab()
    layers_tab = LayersTab(sv)
    analysis_tab = AnalysisTab(sv)
    benchmark_tab = BenchmarkTab(sv)
    
    data_tab.dataset_loaded.connect(layers_tab.populate)
    
    tabs = QTabWidget()
    tabs.addTab(data_tab, "Data")
    tabs.addTab(layers_tab, "Layers")
    tabs.addTab(analysis_tab, "Analysis")
    tabs.addTab(benchmark_tab, "Benchmark")
    
    sv.viewer.window.add_dock_widget(tabs, name="SpatialBench Controls", area="right")
    sv.show()
    sv.run()