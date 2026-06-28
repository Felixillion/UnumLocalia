"""
spatialbench.widgets
====================
PyQt-based dock widget panels that integrate with the Napari viewer.

Architecture
------------
SpatialBench launches as a standard napari application.  All custom UI lives
in a single dock widget (``SpatialBenchPanel``) that contains a ``QTabWidget``
with four tabs:

1. **Layers** — Layer controls for H&E, COMET, Transcripts, Boundaries + ROI
   export + Cell Inspector.
2. **Analysis** — Segmentation / modality selection, PCA / UMAP / Leiden
   buttons, inline matplotlib plot display.
3. **Benchmark** — Segmentation comparison, clustering metrics, Sankey, marker
   scatter.
4. **Settings** — Dataset path, reload, display preferences.

No analysis logic lives here; widgets delegate to the core modules
(``io``, ``viewer``, ``segmentation``, ``analysis``, ``benchmark``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qt imports — wrapped so module can be imported without a display for testing
# ---------------------------------------------------------------------------

try:
    from qtpy.QtCore import Qt, QThread, Signal, QObject, QTimer
    from qtpy.QtGui import QColor, QFont
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    logger.warning(
        "Qt not available. SpatialBench GUI cannot be displayed. "
        "Install via: conda install pyqt"
    )


# ---------------------------------------------------------------------------
# Worker thread for long-running computations
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class AnalysisWorker(QObject):
        """Runs analysis in a background thread to keep the UI responsive."""
        finished = Signal(object)  # emits result
        error = Signal(str)        # emits error message

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def run(self):
            try:
                result = self._fn(*self._args, **self._kwargs)
                self.finished.emit(result)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Analysis error")
                self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Matplotlib canvas widget
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class MatplotlibCanvas(QWidget):
        """Embed a matplotlib Figure in a Qt widget."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._fig: Optional[Figure] = None
            self._canvas: Optional[FigureCanvas] = None

            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)

            self._placeholder = QLabel("No plot yet.")
            self._placeholder.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(self._placeholder)

        def update_figure(self, fig: Figure) -> None:
            """Replace the displayed figure."""
            if self._canvas is not None:
                self._layout.removeWidget(self._canvas)
                self._canvas.close()
                self._canvas = None

            self._placeholder.hide()
            self._fig = fig
            self._canvas = FigureCanvas(fig)
            self._canvas.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            self._layout.addWidget(self._canvas)
            self._canvas.draw()

        def clear(self) -> None:
            """Remove current figure."""
            if self._canvas is not None:
                self._layout.removeWidget(self._canvas)
                self._canvas.close()
                self._canvas = None
            self._placeholder.show()
            self._fig = None


# ---------------------------------------------------------------------------
# Collapsible group box
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class CollapsibleGroup(QGroupBox):
        """A QGroupBox that can be collapsed/expanded by clicking its title."""

        def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
            super().__init__(title, parent)
            self.setCheckable(True)
            self.setChecked(True)
            self.toggled.connect(self._on_toggle)

        def _on_toggle(self, checked: bool) -> None:
            for child in self.findChildren(QWidget):
                if child is not self:
                    child.setVisible(checked)


# ---------------------------------------------------------------------------
# Gene / Protein search widget
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class SearchListWidget(QWidget):
        """Filterable list widget backed by a list of strings."""

        def __init__(
            self,
            items: List[str],
            title: str = "Search",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self._all_items = items

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            self._search_box = QLineEdit()
            self._search_box.setPlaceholderText(f"Search {title}…")
            self._search_box.textChanged.connect(self._filter)
            layout.addWidget(self._search_box)

            self._list = QListWidget()
            self._list.setMaximumHeight(160)
            self._populate(items)
            layout.addWidget(self._list)

        def _populate(self, items: List[str]) -> None:
            self._list.clear()
            for item in items:
                self._list.addItem(QListWidgetItem(item))

        def _filter(self, text: str) -> None:
            filtered = [i for i in self._all_items
                        if text.lower() in i.lower()]
            self._populate(filtered)

        def update_items(self, items: List[str]) -> None:
            self._all_items = items
            self._filter(self._search_box.text())

        @property
        def selected(self) -> Optional[str]:
            items = self._list.selectedItems()
            return items[0].text() if items else None

        @property
        def list_widget(self) -> "QListWidget":
            return self._list


# ---------------------------------------------------------------------------
# Cell inspector panel
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class CellInspectorPanel(QWidget):
        """Read-only panel showing metadata for the last clicked cell."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            header = QLabel("Cell Inspector")
            header.setFont(QFont("", 10, QFont.Bold))
            layout.addWidget(header)

            self._table = QTableWidget(0, 2)
            self._table.setHorizontalHeaderLabels(["Field", "Value"])
            self._table.horizontalHeader().setStretchLastSection(True)
            self._table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._table.setSelectionMode(QTableWidget.SingleSelection)
            self._table.setAlternatingRowColors(True)
            layout.addWidget(self._table)

        def update_cell(self, info: Optional[Dict[str, Any]]) -> None:
            """Populate the table with cell metadata.

            Parameters
            ----------
            info:
                Dictionary as returned by
                :meth:`spatialbench.viewer.SpatialViewer.get_cell_info_at`.
                Pass ``None`` to clear the panel.
            """
            self._table.setRowCount(0)
            if info is None:
                return

            rows = [
                ("Cell ID", str(info.get("cell_id", ""))),
                ("X (centroid)", f"{info.get('x', 0.0):.2f}"),
                ("Y (centroid)", f"{info.get('y', 0.0):.2f}"),
                ("Area (px²)", f"{info.get('area', 0.0):.1f}"),
                ("Transcript count", str(info.get("transcript_count", 0))),
                ("Cluster", str(info.get("cluster", "—"))),
            ]

            proteins = info.get("proteins", {})
            for marker, val in proteins.items():
                rows.append((f"  {marker}", f"{val:.3f}"))

            self._table.setRowCount(len(rows))
            for r, (field, val) in enumerate(rows):
                self._table.setItem(r, 0, QTableWidgetItem(str(field)))
                self._table.setItem(r, 1, QTableWidgetItem(str(val)))
            self._table.resizeColumnsToContents()


# ---------------------------------------------------------------------------
# Layer controls — H&E
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class HEControlPanel(CollapsibleGroup):
        """Controls for the H&E image layer."""

        def __init__(
            self,
            sv: "spatialbench.viewer.SpatialViewer",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__("H&E", parent)
            self._sv = sv

            form = QFormLayout()
            self.setLayout(form)

            # Visibility
            self._vis_chk = QCheckBox()
            self._vis_chk.setChecked(True)
            self._vis_chk.toggled.connect(self._update)
            form.addRow("Visible", self._vis_chk)

            # Opacity
            self._opacity_sl = self._make_slider(0, 100, 100)
            form.addRow("Opacity", self._opacity_sl)

            # Gamma
            self._gamma_sl = self._make_slider(10, 300, 100)
            form.addRow("Gamma (×0.01)", self._gamma_sl)

            # Brightness
            self._brightness_sl = self._make_slider(-100, 100, 0)
            form.addRow("Brightness", self._brightness_sl)

        def _make_slider(self, lo: int, hi: int, val: int) -> QSlider:
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            sl.valueChanged.connect(self._update)
            return sl

        def _update(self) -> None:
            try:
                layer = self._sv._get_layer("H&E")
                if layer is None:
                    return
                layer.visible = self._vis_chk.isChecked()
                layer.opacity = self._opacity_sl.value() / 100.0
                layer.gamma = self._gamma_sl.value() / 100.0
                # Brightness applied as contrast limit shift
                brt = self._brightness_sl.value() / 100.0
                lo, hi = layer.contrast_limits
                span = hi - lo
                layer.contrast_limits = (lo - brt * span, hi - brt * span)
            except Exception:  # pylint: disable=broad-except
                pass


# ---------------------------------------------------------------------------
# Layer controls — COMET
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class CometChannelRow(QWidget):
        """One row of controls for a single COMET marker channel."""

        _COLORMAPS = ["green", "red", "cyan", "magenta", "yellow", "blue",
                      "gray", "hot", "viridis"]

        def __init__(
            self,
            marker: str,
            sv: "spatialbench.viewer.SpatialViewer",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self._marker = marker
            self._sv = sv

            row = QHBoxLayout(self)
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(4)

            # Visibility
            self._vis = QCheckBox(marker)
            self._vis.setChecked(True)
            self._vis.setFixedWidth(90)
            self._vis.toggled.connect(self._update)
            row.addWidget(self._vis)

            # Colour
            self._cmap = QComboBox()
            self._cmap.addItems(self._COLORMAPS)
            self._cmap.setFixedWidth(72)
            self._cmap.currentTextChanged.connect(self._update)
            row.addWidget(self._cmap)

            # Min
            self._vmin = QSpinBox()
            self._vmin.setRange(0, 65535)
            self._vmin.setValue(0)
            self._vmin.setFixedWidth(60)
            self._vmin.valueChanged.connect(self._update)
            row.addWidget(QLabel("Min"))
            row.addWidget(self._vmin)

            # Max
            self._vmax = QSpinBox()
            self._vmax.setRange(0, 65535)
            self._vmax.setValue(1000)
            self._vmax.setFixedWidth(60)
            self._vmax.valueChanged.connect(self._update)
            row.addWidget(QLabel("Max"))
            row.addWidget(self._vmax)

            # Opacity
            self._opacity = QSlider(Qt.Horizontal)
            self._opacity.setRange(0, 100)
            self._opacity.setValue(80)
            self._opacity.setFixedWidth(60)
            self._opacity.valueChanged.connect(self._update)
            row.addWidget(QLabel("α"))
            row.addWidget(self._opacity)

        def _update(self) -> None:
            try:
                self._sv.update_comet_layer(
                    self._marker,
                    colormap=self._cmap.currentText(),
                    vmin=self._vmin.value(),
                    vmax=self._vmax.value(),
                    opacity=self._opacity.value() / 100.0,
                    visible=self._vis.isChecked(),
                )
            except Exception:  # pylint: disable=broad-except
                pass

    class CometControlPanel(CollapsibleGroup):
        """Scrollable panel with one :class:`CometChannelRow` per marker."""

        def __init__(
            self,
            markers: List[str],
            sv: "spatialbench.viewer.SpatialViewer",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__("COMET Channels", parent)
            self._sv = sv

            outer = QVBoxLayout(self)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(240)

            inner = QWidget()
            self._rows_layout = QVBoxLayout(inner)
            self._rows_layout.setContentsMargins(0, 0, 0, 0)
            self._rows_layout.setSpacing(0)

            for marker in markers:
                self._rows_layout.addWidget(
                    CometChannelRow(marker, sv)
                )

            self._rows_layout.addStretch()
            scroll.setWidget(inner)
            outer.addWidget(scroll)


# ---------------------------------------------------------------------------
# Layer controls — Transcripts
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class TranscriptControlPanel(CollapsibleGroup):
        """Gene search + transcript display controls."""

        def __init__(
            self,
            genes: List[str],
            sv: "spatialbench.viewer.SpatialViewer",
            transcripts_df,  # pd.DataFrame | None
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__("Transcripts", parent)
            self._sv = sv
            self._transcripts_df = transcripts_df

            layout = QVBoxLayout(self)

            # Gene search
            self._gene_search = SearchListWidget(genes, title="gene")
            layout.addWidget(self._gene_search)

            # Display button
            btn_row = QHBoxLayout()
            self._btn_show = QPushButton("Show gene")
            self._btn_show.clicked.connect(self._show_gene)
            btn_row.addWidget(self._btn_show)
            self._btn_hide = QPushButton("Hide gene")
            self._btn_hide.clicked.connect(self._hide_gene)
            btn_row.addWidget(self._btn_hide)
            layout.addLayout(btn_row)

            # Style controls
            form = QFormLayout()
            self._colour_btn = QPushButton()
            self._colour_btn.setStyleSheet("background-color: yellow;")
            self._colour_btn.setFixedWidth(40)
            self._colour_btn.clicked.connect(self._pick_colour)
            self._current_colour = (1.0, 1.0, 0.0, 1.0)
            form.addRow("Colour", self._colour_btn)

            self._size_spin = QDoubleSpinBox()
            self._size_spin.setRange(1.0, 50.0)
            self._size_spin.setValue(4.0)
            self._size_spin.setSingleStep(0.5)
            form.addRow("Dot size", self._size_spin)

            self._opacity_sl = QSlider(Qt.Horizontal)
            self._opacity_sl.setRange(0, 100)
            self._opacity_sl.setValue(80)
            form.addRow("Opacity", self._opacity_sl)

            layout.addLayout(form)

        def _show_gene(self) -> None:
            gene = self._gene_search.selected
            if gene is None or self._transcripts_df is None:
                return
            gene_df = self._transcripts_df[
                self._transcripts_df.get("feature_name",
                self._transcripts_df.iloc[:, 2]) == gene
            ]
            if len(gene_df) == 0:
                QMessageBox.information(self, "No transcripts",
                    f"No transcripts found for gene '{gene}'.")
                return
            x_col = next(
                (c for c in ["x_location", "x"] if c in gene_df.columns), None
            )
            y_col = next(
                (c for c in ["y_location", "y"] if c in gene_df.columns), None
            )
            if x_col is None or y_col is None:
                return
            coords = gene_df[[x_col, y_col]].to_numpy()
            self._sv.add_transcript_layer(
                gene,
                coords,
                color=self._current_colour,
                size=self._size_spin.value(),
                opacity=self._opacity_sl.value() / 100.0,
            )

        def _hide_gene(self) -> None:
            gene = self._gene_search.selected
            if gene:
                self._sv.remove_transcript_layer(gene)

        def _pick_colour(self) -> None:
            col = QColorDialog.getColor(parent=self)
            if col.isValid():
                self._current_colour = (
                    col.red() / 255.0,
                    col.green() / 255.0,
                    col.blue() / 255.0,
                    1.0,
                )
                self._colour_btn.setStyleSheet(
                    f"background-color: {col.name()};"
                )


# ---------------------------------------------------------------------------
# Layer controls — Boundaries
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class BoundaryControlPanel(CollapsibleGroup):
        """Controls for cell and nucleus boundary layers."""

        def __init__(
            self,
            sv: "spatialbench.viewer.SpatialViewer",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__("Boundaries", parent)
            self._sv = sv

            form = QFormLayout(self)

            # Cell boundaries
            self._cell_vis = QCheckBox("Show cell boundaries")
            self._cell_vis.setChecked(True)
            self._cell_vis.toggled.connect(
                lambda v: self._sv.update_boundary_layer("Cell boundaries", visible=v)
            )
            form.addRow(self._cell_vis)

            self._cell_colour_btn = self._make_colour_btn("white")
            self._cell_colour_btn.clicked.connect(
                lambda: self._pick_colour("Cell boundaries", self._cell_colour_btn)
            )
            form.addRow("Cell colour", self._cell_colour_btn)

            self._cell_width = QDoubleSpinBox()
            self._cell_width.setRange(0.1, 10.0)
            self._cell_width.setValue(1.0)
            self._cell_width.valueChanged.connect(
                lambda v: self._sv.update_boundary_layer("Cell boundaries", width=v)
            )
            form.addRow("Cell line width", self._cell_width)

            # Nucleus boundaries
            self._nuc_vis = QCheckBox("Show nucleus boundaries")
            self._nuc_vis.setChecked(True)
            self._nuc_vis.toggled.connect(
                lambda v: self._sv.update_boundary_layer("Nucleus boundaries", visible=v)
            )
            form.addRow(self._nuc_vis)

            self._nuc_colour_btn = self._make_colour_btn("cyan")
            self._nuc_colour_btn.clicked.connect(
                lambda: self._pick_colour("Nucleus boundaries", self._nuc_colour_btn)
            )
            form.addRow("Nucleus colour", self._nuc_colour_btn)

            self._nuc_width = QDoubleSpinBox()
            self._nuc_width.setRange(0.1, 10.0)
            self._nuc_width.setValue(1.0)
            self._nuc_width.valueChanged.connect(
                lambda v: self._sv.update_boundary_layer("Nucleus boundaries", width=v)
            )
            form.addRow("Nucleus line width", self._nuc_width)

        def _make_colour_btn(self, name: str) -> QPushButton:
            btn = QPushButton()
            btn.setFixedWidth(40)
            btn.setStyleSheet(f"background-color: {name};")
            return btn

        def _pick_colour(self, layer_name: str, btn: QPushButton) -> None:
            col = QColorDialog.getColor(parent=self)
            if col.isValid():
                btn.setStyleSheet(f"background-color: {col.name()};")
                self._sv.update_boundary_layer(layer_name, color=col.name())


# ---------------------------------------------------------------------------
# ROI export panel
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class ROIExportPanel(CollapsibleGroup):
        """Panel for exporting the current viewer canvas."""

        def __init__(
            self,
            sv: "spatialbench.viewer.SpatialViewer",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__("Export ROI / Screenshot", parent)
            self._sv = sv

            layout = QFormLayout(self)

            self._fmt_combo = QComboBox()
            self._fmt_combo.addItems(["png", "svg", "pdf"])
            layout.addRow("Format", self._fmt_combo)

            self._scale_spin = QDoubleSpinBox()
            self._scale_spin.setRange(1.0, 4.0)
            self._scale_spin.setValue(2.0)
            self._scale_spin.setSingleStep(0.5)
            layout.addRow("Scale (PNG)", self._scale_spin)

            btn = QPushButton("Save screenshot…")
            btn.clicked.connect(self._export)
            layout.addRow(btn)

        def _export(self) -> None:
            fmt = self._fmt_combo.currentText()
            path, _ = QFileDialog.getSaveFileName(
                self, "Save screenshot",
                os.path.expanduser("~"),
                f"{fmt.upper()} files (*.{fmt})",
            )
            if not path:
                return
            try:
                saved = self._sv.export_roi_screenshot(
                    path, fmt=fmt, scale=self._scale_spin.value()
                )
                QMessageBox.information(self, "Saved", f"Screenshot saved to:\n{saved}")
            except Exception as exc:  # pylint: disable=broad-except
                QMessageBox.critical(self, "Export error", str(exc))


# ---------------------------------------------------------------------------
# Layers tab
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class LayersTab(QScrollArea):
        """Viewer tab: all layer controls + cell inspector."""

        def __init__(
            self,
            sv: "spatialbench.viewer.SpatialViewer",
            genes: List[str],
            proteins: List[str],
            transcripts_df,
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self.setWidgetResizable(True)

            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setSpacing(6)

            # H&E controls
            self._he_panel = HEControlPanel(sv)
            layout.addWidget(self._he_panel)

            # COMET controls
            if proteins:
                self._comet_panel = CometControlPanel(proteins, sv)
                layout.addWidget(self._comet_panel)

            # Transcript controls
            self._transcript_panel = TranscriptControlPanel(
                genes, sv, transcripts_df
            )
            layout.addWidget(self._transcript_panel)

            # Boundary controls
            self._boundary_panel = BoundaryControlPanel(sv)
            layout.addWidget(self._boundary_panel)

            # ROI export
            self._roi_panel = ROIExportPanel(sv)
            layout.addWidget(self._roi_panel)

            # Cell inspector
            self._inspector = CellInspectorPanel()
            layout.addWidget(self._inspector)

            layout.addStretch()
            self.setWidget(container)

        @property
        def cell_inspector(self) -> CellInspectorPanel:
            return self._inspector


# ---------------------------------------------------------------------------
# Analysis tab
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class AnalysisTab(QWidget):
        """Tab for running single-cell analysis on an AnnData object."""

        def __init__(
            self,
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self._adata = None  # set by main panel after segmentation

            layout = QVBoxLayout(self)
            layout.setSpacing(6)

            # Segmentation selector
            seg_grp = QGroupBox("Segmentation")
            seg_layout = QFormLayout(seg_grp)
            self._seg_combo = QComboBox()
            self._seg_combo.addItems(["Original (Xenium)", "Uploaded segmentation"])
            seg_layout.addRow("Use", self._seg_combo)
            layout.addWidget(seg_grp)

            # Modality selector
            mod_grp = QGroupBox("Modality")
            mod_layout = QFormLayout(mod_grp)
            self._mod_combo = QComboBox()
            self._mod_combo.addItems(["Genes", "Proteins", "Combined"])
            mod_layout.addRow("Analyse", self._mod_combo)
            layout.addWidget(mod_grp)

            # Analysis buttons
            btn_grp = QGroupBox("Run analysis")
            btn_layout = QVBoxLayout(btn_grp)

            self._pca_btn = QPushButton("1. Run PCA")
            self._pca_btn.clicked.connect(self._run_pca)
            btn_layout.addWidget(self._pca_btn)

            self._umap_btn = QPushButton("2. Run UMAP")
            self._umap_btn.clicked.connect(self._run_umap)
            btn_layout.addWidget(self._umap_btn)

            # Leiden resolution
            leiden_row = QHBoxLayout()
            leiden_row.addWidget(QLabel("Leiden resolution:"))
            self._res_spin = QDoubleSpinBox()
            self._res_spin.setRange(0.05, 5.0)
            self._res_spin.setValue(0.5)
            self._res_spin.setSingleStep(0.05)
            leiden_row.addWidget(self._res_spin)
            btn_layout.addLayout(leiden_row)

            self._leiden_btn = QPushButton("3. Run Leiden clustering")
            self._leiden_btn.clicked.connect(self._run_leiden)
            btn_layout.addWidget(self._leiden_btn)
            layout.addWidget(btn_grp)

            # Plot type selector
            plot_grp = QGroupBox("Visualise")
            plot_layout = QFormLayout(plot_grp)
            self._plot_combo = QComboBox()
            self._plot_combo.addItems([
                "UMAP (clusters)",
                "UMAP (gene expression)",
                "Spatial clusters",
                "PCA variance",
            ])
            plot_layout.addRow("Plot type", self._plot_combo)

            self._colour_edit = QLineEdit()
            self._colour_edit.setPlaceholderText("Gene name or obs column")
            plot_layout.addRow("Colour by", self._colour_edit)

            self._plot_btn = QPushButton("Generate plot")
            self._plot_btn.clicked.connect(self._plot)
            plot_layout.addRow(self._plot_btn)
            layout.addWidget(plot_grp)

            # Export
            export_row = QHBoxLayout()
            self._export_fig_btn = QPushButton("Export figure…")
            self._export_fig_btn.clicked.connect(self._export_figure)
            export_row.addWidget(self._export_fig_btn)
            self._export_table_btn = QPushButton("Export metadata CSV…")
            self._export_table_btn.clicked.connect(self._export_table)
            export_row.addWidget(self._export_table_btn)
            layout.addLayout(export_row)

            # Canvas
            self._canvas = MatplotlibCanvas()
            self._canvas.setMinimumHeight(300)
            layout.addWidget(self._canvas)

            self._last_figure = None

        # ---- AnnData injection -------------------------------------------

        def set_anndata(self, adata) -> None:
            """Set the AnnData object to use for analysis."""
            self._adata = adata

        # ---- Analysis actions --------------------------------------------

        def _modality(self) -> str:
            return self._mod_combo.currentText().lower()

        def _run_pca(self) -> None:
            if self._adata is None:
                QMessageBox.warning(self, "No data", "Load a dataset first.")
                return
            try:
                from spatialbench import analysis
                analysis.run_pca(self._adata, modality=self._modality())
                QMessageBox.information(self, "Done", "PCA complete.")
            except Exception as exc:
                QMessageBox.critical(self, "PCA error", str(exc))

        def _run_umap(self) -> None:
            if self._adata is None:
                return
            try:
                from spatialbench import analysis
                analysis.run_neighbors(self._adata, modality=self._modality())
                analysis.run_umap(self._adata, modality=self._modality())
                QMessageBox.information(self, "Done", "UMAP complete.")
            except Exception as exc:
                QMessageBox.critical(self, "UMAP error", str(exc))

        def _run_leiden(self) -> None:
            if self._adata is None:
                return
            try:
                from spatialbench import analysis
                analysis.run_leiden(
                    self._adata,
                    modality=self._modality(),
                    resolution=self._res_spin.value(),
                )
                QMessageBox.information(self, "Done", "Leiden clustering complete.")
            except Exception as exc:
                QMessageBox.critical(self, "Leiden error", str(exc))

        def _plot(self) -> None:
            if self._adata is None:
                return
            try:
                from spatialbench import analysis
                plot_type = self._plot_combo.currentText()
                colour = self._colour_edit.text().strip() or f"leiden_{self._modality()}"
                mod = self._modality()

                if "UMAP" in plot_type:
                    fig = analysis.plot_umap(self._adata, color=colour, modality=mod)
                elif "Spatial" in plot_type:
                    fig = analysis.plot_spatial_clusters(self._adata, color=colour)
                else:
                    fig = analysis.plot_pca_variance(self._adata, modality=mod)

                self._canvas.update_figure(fig)
                self._last_figure = fig
            except Exception as exc:
                QMessageBox.critical(self, "Plot error", str(exc))

        def _export_figure(self) -> None:
            if self._last_figure is None:
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save figure", os.path.expanduser("~"),
                "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
            )
            if path:
                from spatialbench.utils import export_figure
                export_figure(self._last_figure, path,
                              fmt=Path(path).suffix.lstrip("."))

        def _export_table(self) -> None:
            if self._adata is None:
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save metadata CSV", os.path.expanduser("~"),
                "CSV (*.csv)"
            )
            if path:
                self._adata.obs.to_csv(path)


# ---------------------------------------------------------------------------
# Benchmark tab
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class BenchmarkTab(QWidget):
        """Tab for segmentation + clustering + marker benchmarking."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._adata_a = None
            self._adata_b = None

            layout = QVBoxLayout(self)
            layout.setSpacing(6)

            # Segmentation comparison
            seg_grp = QGroupBox("Segmentation comparison")
            seg_layout = QVBoxLayout(seg_grp)

            load_row = QHBoxLayout()
            self._load_seg_btn = QPushButton("Load segmentation…")
            self._load_seg_btn.clicked.connect(self._load_segmentation)
            load_row.addWidget(self._load_seg_btn)
            seg_layout.addLayout(load_row)

            self._compare_seg_btn = QPushButton("Compare segmentations")
            self._compare_seg_btn.clicked.connect(self._compare_segmentations)
            seg_layout.addWidget(self._compare_seg_btn)

            self._seg_table = QTableWidget(0, 2)
            self._seg_table.setHorizontalHeaderLabels(["Metric", "Value"])
            self._seg_table.horizontalHeader().setStretchLastSection(True)
            self._seg_table.setMaximumHeight(180)
            seg_layout.addWidget(self._seg_table)

            self._export_seg_btn = QPushButton("Export segmentation comparison CSV…")
            self._export_seg_btn.clicked.connect(self._export_seg_comparison)
            seg_layout.addWidget(self._export_seg_btn)
            layout.addWidget(seg_grp)

            # Clustering comparison
            clust_grp = QGroupBox("Clustering comparison")
            clust_layout = QVBoxLayout(clust_grp)

            key_row = QFormLayout()
            self._key_a_edit = QLineEdit("leiden_genes")
            self._key_b_edit = QLineEdit("leiden_genes")
            key_row.addRow("Cluster key A", self._key_a_edit)
            key_row.addRow("Cluster key B", self._key_b_edit)
            clust_layout.addLayout(key_row)

            self._clust_btn = QPushButton("Compare clusterings")
            self._clust_btn.clicked.connect(self._compare_clusterings)
            clust_layout.addWidget(self._clust_btn)

            self._clust_result = QLabel("ARI: —   NMI: —")
            self._clust_result.setFont(QFont("", 10, QFont.Bold))
            clust_layout.addWidget(self._clust_result)

            plot_row = QHBoxLayout()
            self._contingency_btn = QPushButton("Contingency heatmap")
            self._contingency_btn.clicked.connect(self._plot_contingency)
            plot_row.addWidget(self._contingency_btn)

            self._sankey_btn = QPushButton("Sankey diagram")
            self._sankey_btn.clicked.connect(self._plot_sankey)
            plot_row.addWidget(self._sankey_btn)
            clust_layout.addLayout(plot_row)
            layout.addWidget(clust_grp)

            # Marker comparison
            marker_grp = QGroupBox("Marker comparison")
            marker_layout = QVBoxLayout(marker_grp)

            self._marker_edit = QLineEdit()
            self._marker_edit.setPlaceholderText("Comma-separated markers, e.g. CK8,CD45")
            marker_layout.addWidget(QLabel("Markers to compare:"))
            marker_layout.addWidget(self._marker_edit)

            self._marker_btn = QPushButton("Compare markers")
            self._marker_btn.clicked.connect(self._compare_markers)
            marker_layout.addWidget(self._marker_btn)

            self._marker_table = QTableWidget(0, 5)
            self._marker_table.setHorizontalHeaderLabels(
                ["Marker", "Pearson r", "p", "Spearman r", "n cells"]
            )
            self._marker_table.setMaximumHeight(150)
            marker_layout.addWidget(self._marker_table)
            layout.addWidget(marker_grp)

            # Canvas for plots
            self._canvas = MatplotlibCanvas()
            self._canvas.setMinimumHeight(280)
            layout.addWidget(self._canvas)

            self._last_figure = None
            self._seg_metrics = None
            self._labels_b = None

        # ---- Data injection ---------------------------------------------

        def set_adata_pair(self, adata_a, adata_b) -> None:
            self._adata_a = adata_a
            self._adata_b = adata_b

        # ---- Actions ----------------------------------------------------

        def _load_segmentation(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load segmentation",
                os.path.expanduser("~"),
                "Segmentation files (*.tif *.tiff *.geojson *.json)",
            )
            if not path:
                return
            try:
                from spatialbench.segmentation import load_segmentation
                self._labels_b = load_segmentation(path)
                QMessageBox.information(
                    self, "Loaded",
                    f"Segmentation loaded: {Path(path).name}\n"
                    f"Shape: {self._labels_b.shape}"
                )
            except Exception as exc:
                QMessageBox.critical(self, "Load error", str(exc))

        def _compare_segmentations(self) -> None:
            if self._labels_b is None:
                QMessageBox.warning(self, "No segmentation",
                    "Load a segmentation file first.")
                return
            QMessageBox.information(
                self, "Note",
                "Full segmentation comparison requires the original label mask. "
                "Feature in next update."
            )

        def _export_seg_comparison(self) -> None:
            if self._seg_metrics is None:
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save comparison CSV", os.path.expanduser("~"),
                "CSV (*.csv)"
            )
            if path:
                from spatialbench.utils import export_dataframe
                export_dataframe(self._seg_metrics.summary_df, path)

        def _get_label_pair(self):
            if self._adata_a is None or self._adata_b is None:
                QMessageBox.warning(self, "No data",
                    "Both AnnData objects must be loaded.")
                return None, None
            key_a = self._key_a_edit.text().strip()
            key_b = self._key_b_edit.text().strip()
            if key_a not in self._adata_a.obs.columns:
                QMessageBox.warning(self, "Missing column",
                    f"Column '{key_a}' not found in AnnData A.")
                return None, None
            if key_b not in self._adata_b.obs.columns:
                QMessageBox.warning(self, "Missing column",
                    f"Column '{key_b}' not found in AnnData B.")
                return None, None
            return (
                self._adata_a.obs[key_a].tolist(),
                self._adata_b.obs[key_b].tolist(),
            )

        def _compare_clusterings(self) -> None:
            labels_a, labels_b = self._get_label_pair()
            if labels_a is None:
                return
            try:
                from spatialbench.benchmark import compare_clusterings
                result = compare_clusterings(labels_a, labels_b)
                self._clust_result.setText(
                    f"ARI: {result['ari']:.4f}   NMI: {result['nmi']:.4f}"
                )
                self._last_labels = (labels_a, labels_b)
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))

        def _plot_contingency(self) -> None:
            if not hasattr(self, "_last_labels"):
                QMessageBox.warning(self, "No results",
                    "Run 'Compare clusterings' first.")
                return
            try:
                from spatialbench.benchmark import plot_contingency
                fig = plot_contingency(*self._last_labels,
                    name_a=self._key_a_edit.text(),
                    name_b=self._key_b_edit.text())
                self._canvas.update_figure(fig)
                self._last_figure = fig
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))

        def _plot_sankey(self) -> None:
            if not hasattr(self, "_last_labels"):
                QMessageBox.warning(self, "No results",
                    "Run 'Compare clusterings' first.")
                return
            try:
                from spatialbench.benchmark import plot_sankey
                fig = plot_sankey(*self._last_labels)
                # Show plotly figure in browser
                import tempfile, webbrowser
                with tempfile.NamedTemporaryFile(
                    suffix=".html", delete=False, mode="w"
                ) as f:
                    f.write(fig.to_html())
                    webbrowser.open(f.name)
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))

        def _compare_markers(self) -> None:
            if self._adata_a is None or self._adata_b is None:
                QMessageBox.warning(self, "No data",
                    "Both AnnData objects must be loaded.")
                return
            markers_text = self._marker_edit.text().strip()
            markers = [m.strip() for m in markers_text.split(",") if m.strip()]
            if not markers:
                QMessageBox.warning(self, "No markers",
                    "Enter comma-separated marker names.")
                return
            try:
                from spatialbench.benchmark import compare_markers

                prot_a = self._adata_a.obsm.get("X_protein")
                prot_b = self._adata_b.obsm.get("X_protein")
                if prot_a is None or prot_b is None:
                    QMessageBox.warning(self, "No protein data",
                        "Protein data not found in one or both AnnData objects.")
                    return

                names_a = self._adata_a.uns.get("protein_names", [])
                names_b = self._adata_b.uns.get("protein_names", [])

                result_df = compare_markers(
                    prot_a, prot_b,
                    markers=markers,
                    columns_a=names_a,
                    columns_b=names_b,
                )
                self._populate_marker_table(result_df)
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))

        def _populate_marker_table(self, df) -> None:
            self._marker_table.setRowCount(len(df))
            for r, row in df.iterrows():
                self._marker_table.setItem(r, 0, QTableWidgetItem(str(row["marker"])))
                self._marker_table.setItem(r, 1, QTableWidgetItem(f"{row['pearson_r']:.4f}"))
                self._marker_table.setItem(r, 2, QTableWidgetItem(f"{row['pearson_p']:.4f}"))
                self._marker_table.setItem(r, 3, QTableWidgetItem(f"{row['spearman_r']:.4f}"))
                self._marker_table.setItem(r, 4, QTableWidgetItem(str(row["n_cells"])))
            self._marker_table.resizeColumnsToContents()


# ---------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class SettingsTab(QWidget):
        """Dataset path selection and global preferences."""

        dataset_changed = Signal(str)  # emitted when a new dataset path is set

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setSpacing(8)

            # Dataset path
            ds_grp = QGroupBox("Dataset")
            ds_layout = QFormLayout(ds_grp)

            path_row = QHBoxLayout()
            self._path_edit = QLineEdit()
            self._path_edit.setPlaceholderText("/path/to/dataset/folder")
            path_row.addWidget(self._path_edit)

            browse_btn = QPushButton("Browse…")
            browse_btn.clicked.connect(self._browse)
            path_row.addWidget(browse_btn)
            ds_layout.addRow("Dataset folder", path_row)

            load_btn = QPushButton("Load dataset")
            load_btn.clicked.connect(self._emit_load)
            ds_layout.addRow(load_btn)
            layout.addWidget(ds_grp)

            # Display preferences
            disp_grp = QGroupBox("Display preferences")
            disp_layout = QFormLayout(disp_grp)

            self._bg_combo = QComboBox()
            self._bg_combo.addItems(["black", "white", "dark gray"])
            disp_layout.addRow("Viewer background", self._bg_combo)

            layout.addWidget(disp_grp)

            # About
            about = QLabel(
                "<b>SpatialBench v1.0</b><br>"
                "A multimodal spatial biology toolkit.<br>"
                "<a href='https://github.com/SpatialBench/SpatialBench'>"
                "github.com/SpatialBench</a>"
            )
            about.setOpenExternalLinks(True)
            about.setWordWrap(True)
            layout.addWidget(about)

            layout.addStretch()

        def _browse(self) -> None:
            folder = QFileDialog.getExistingDirectory(
                self, "Select dataset folder", os.path.expanduser("~")
            )
            if folder:
                self._path_edit.setText(folder)

        def _emit_load(self) -> None:
            path = self._path_edit.text().strip()
            if path:
                self.dataset_changed.emit(path)

        @property
        def dataset_path(self) -> str:
            return self._path_edit.text().strip()


# ---------------------------------------------------------------------------
# Main SpatialBench dock panel
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    class SpatialBenchPanel(QWidget):
        """Top-level panel added as a napari dock widget.

        This widget wires together all tabs and responds to dataset load
        requests from the Settings tab.
        """

        def __init__(
            self,
            sv: "spatialbench.viewer.SpatialViewer",
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self._sv = sv
            self._loader = None
            self._adata_original = None
            self._adata_user = None

            self.setMinimumWidth(360)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            # Tab widget
            self._tabs = QTabWidget()
            layout.addWidget(self._tabs)

            # Placeholder layers tab (will be replaced after dataset load)
            self._layers_placeholder = QLabel(
                "Load a dataset to enable layer controls.\n\n"
                "Go to the Settings tab and select a dataset folder."
            )
            self._layers_placeholder.setAlignment(Qt.AlignCenter)
            self._layers_placeholder.setWordWrap(True)
            self._tabs.addTab(self._layers_placeholder, "Layers")

            # Analysis tab
            self._analysis_tab = AnalysisTab()
            self._tabs.addTab(self._analysis_tab, "Analysis")

            # Benchmark tab
            self._benchmark_tab = BenchmarkTab()
            self._tabs.addTab(self._benchmark_tab, "Benchmark")

            # Settings tab
            self._settings_tab = SettingsTab()
            self._settings_tab.dataset_changed.connect(self._load_dataset)
            self._tabs.addTab(self._settings_tab, "Settings")

            # Navigate to settings by default
            self._tabs.setCurrentIndex(3)

        # ---- Dataset loading --------------------------------------------

        def _load_dataset(self, folder: str) -> None:
            """Detect and load all data from *folder*."""
            from spatialbench.io import DatasetLoader
            from spatialbench.utils import shapes_to_napari

            try:
                loader = DatasetLoader(folder)
                loader.load()
                self._loader = loader
            except Exception as exc:
                QMessageBox.critical(self, "Load error", str(exc))
                return

            # ---- Add napari layers ----------------------------------------
            sv = self._sv

            if loader.he_array is not None:
                sv.add_he_layer(loader.he_array)

            for marker, arr in loader.comet_arrays.items():
                sv.add_comet_layer(marker, arr)

            if loader.cell_boundaries_df is not None:
                shapes = shapes_to_napari(loader.cell_boundaries_df)
                sv.add_boundary_layer(shapes, name="Cell boundaries")

            if loader.nucleus_boundaries_df is not None:
                shapes = shapes_to_napari(loader.nucleus_boundaries_df)
                sv.add_boundary_layer(shapes, name="Nucleus boundaries",
                                      color="cyan")

            sv.reset_view()

            # ---- Replace layers tab with real controls ---------------------
            layers_tab = LayersTab(
                sv,
                genes=loader.genes,
                proteins=loader.proteins,
                transcripts_df=loader.transcripts_df,
            )
            # Wire cell-click to inspector
            self._connect_click_inspector(layers_tab)

            self._tabs.removeTab(0)
            self._tabs.insertTab(0, layers_tab, "Layers")
            self._tabs.setCurrentIndex(0)
            self._layers_tab = layers_tab

            # ---- Pass AnnData to analysis tab ------------------------------
            if loader.anndata_ref is not None:
                self._adata_original = loader.anndata_ref
                self._analysis_tab.set_anndata(self._adata_original)

            QMessageBox.information(
                self, "Dataset loaded",
                f"Dataset loaded from:\n{folder}\n\n"
                + loader.manifest.summary()
            )

        def _connect_click_inspector(self, layers_tab: "LayersTab") -> None:
            """Wire napari mouse click → cell inspector update."""
            inspector = layers_tab.cell_inspector
            loader = self._loader

            @self._sv.viewer.mouse_drag_callbacks.append
            def _on_click(viewer, event):
                try:
                    pos = event.position
                    if len(pos) >= 2:
                        y, x = pos[-2], pos[-1]
                        info = self._sv.get_cell_info_at(
                            x, y,
                            cells_df=loader.cells_df,
                        )
                        inspector.update_cell(info)
                except Exception:  # pylint: disable=broad-except
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch() -> None:
    """Launch the SpatialBench GUI.

    This is the function called by the ``spatialbench`` CLI command.
    It creates a napari viewer, adds the SpatialBench dock panel, and
    enters the event loop.
    """
    if not _QT_AVAILABLE:
        raise RuntimeError(
            "Qt is not available. Install napari and PyQt/PySide with:\n"
            "  conda install napari pyqt -c conda-forge"
        )

    import napari
    from spatialbench.viewer import SpatialViewer

    sv = SpatialViewer(title="SpatialBench")
    panel = SpatialBenchPanel(sv)

    sv.viewer.window.add_dock_widget(
        panel,
        name="SpatialBench",
        area="right",
        allowed_areas=["right", "left"],
    )
    sv.show()
    napari.run()
