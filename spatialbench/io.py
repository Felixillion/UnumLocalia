"""
spatialbench.io
===============
Dataset discovery, manifest building, and lazy data loading.

Philosophy
----------
* Auto-detect all modalities in a dataset folder — no manual file selection.
* Use memory-mapped access for large TIFF files (OME-TIFF, H&E) so that only
  the requested tiles are read into RAM.
* Never modify any file on disk.

Expected dataset layout (files may be in sub-folders or at root)::

    dataset/
        cells.csv
        cell_boundaries.parquet
        nucleus_boundaries.parquet
        transcripts.parquet
        matrix.csv                  ← 3×3 affine alignment matrix
        he.tif  (or *.tiff)         ← aligned H&E whole-slide image
        comet/                      ← one OME-TIFF per protein marker
            CK8.ome.tiff
            CD45.ome.tiff
            ...
        anndata.h5ad  (or *.h5ad)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

from spatialbench.utils import safe_read_csv, safe_read_parquet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-name patterns for auto-detection
# ---------------------------------------------------------------------------

_PATTERN_CELLS = re.compile(r"^cells\.csv$", re.IGNORECASE)
_PATTERN_CELL_BOUNDS = re.compile(r"^cell_boundaries\.parquet$", re.IGNORECASE)
_PATTERN_NUC_BOUNDS = re.compile(r"^nucleus_boundaries\.parquet$", re.IGNORECASE)
_PATTERN_TRANSCRIPTS = re.compile(r"^transcripts\.parquet$", re.IGNORECASE)
_PATTERN_MATRIX = re.compile(r"^matrix\.csv$", re.IGNORECASE)
_PATTERN_HE = re.compile(r".*\.(tif|tiff)$", re.IGNORECASE)
_PATTERN_COMET = re.compile(r".*\.ome\.(tif|tiff)$", re.IGNORECASE)
_PATTERN_ANNDATA = re.compile(r".*\.h5ad$", re.IGNORECASE)

# Sub-folder names associated with COMET images
_COMET_FOLDER_HINTS = {"comet", "comet_images", "proteins", "multiplex"}


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    """Paths to all detected files in a SpatialBench dataset.

    Any field may be ``None`` if the corresponding file was not found.
    The manifest is built by :func:`detect_files` and is immutable.
    """

    folder: Path

    # Xenium
    cells: Optional[Path] = None
    cell_boundaries: Optional[Path] = None
    nucleus_boundaries: Optional[Path] = None
    transcripts: Optional[Path] = None

    # Alignment
    matrix: Optional[Path] = None

    # H&E
    he: Optional[Path] = None

    # COMET — mapping from marker name → file path
    comet: Dict[str, Path] = field(default_factory=dict)

    # AnnData
    anndata: Optional[Path] = None

    # ----------------------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable summary of detected files."""
        lines = [f"Dataset folder : {self.folder}"]
        lines.append(f"  cells.csv    : {'✓' if self.cells else '✗'}")
        lines.append(f"  cell bounds  : {'✓' if self.cell_boundaries else '✗'}")
        lines.append(f"  nuc  bounds  : {'✓' if self.nucleus_boundaries else '✗'}")
        lines.append(f"  transcripts  : {'✓' if self.transcripts else '✗'}")
        lines.append(f"  matrix.csv   : {'✓' if self.matrix else '✗'}")
        lines.append(f"  H&E          : {'✓ ' + str(self.he.name) if self.he else '✗'}")
        lines.append(f"  COMET ({len(self.comet):>2} ch): "
                     + (", ".join(self.comet) if self.comet else "—"))
        lines.append(f"  AnnData      : {'✓' if self.anndata else '✗'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File detection
# ---------------------------------------------------------------------------

def detect_files(folder: Path) -> DatasetManifest:
    """Recursively scan *folder* and build a :class:`DatasetManifest`.

    The function walks the directory tree up to two levels deep to support
    datasets where COMET images are stored in a sub-folder.

    Parameters
    ----------
    folder:
        Root directory of the dataset.

    Returns
    -------
    DatasetManifest
        Populated manifest (fields are ``None`` when not found).

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    manifest = DatasetManifest(folder=folder)
    comet_candidates: List[Path] = []
    he_candidates: List[Path] = []

    # Collect all files within two levels
    all_files: List[Path] = []
    for depth in range(3):  # 0, 1, 2 levels deep
        pattern = ("*/" * depth) + "*"
        all_files.extend(folder.glob(pattern))

    # Remove duplicate paths (glob can return duplicates)
    all_files = sorted(set(p for p in all_files if p.is_file()))

    for fp in all_files:
        name = fp.name

        if _PATTERN_CELLS.match(name):
            manifest.cells = fp

        elif _PATTERN_CELL_BOUNDS.match(name):
            manifest.cell_boundaries = fp

        elif _PATTERN_NUC_BOUNDS.match(name):
            manifest.nucleus_boundaries = fp

        elif _PATTERN_TRANSCRIPTS.match(name):
            manifest.transcripts = fp

        elif _PATTERN_MATRIX.match(name):
            manifest.matrix = fp

        elif _PATTERN_ANNDATA.match(name):
            if manifest.anndata is None:
                manifest.anndata = fp

        elif _PATTERN_COMET.match(name):
            # OME-TIFFs (*.ome.tif / *.ome.tiff) → COMET markers
            comet_candidates.append(fp)

        elif _PATTERN_HE.match(name):
            # Plain TIFFs that are not OME-TIFFs → H&E candidates
            if not _PATTERN_COMET.match(name):
                he_candidates.append(fp)

    # ---- Assign COMET markers -----------------------------------------
    # Prefer files inside a folder whose name suggests COMET
    prioritised_comet = [
        fp for fp in comet_candidates
        if fp.parent.name.lower() in _COMET_FOLDER_HINTS
    ]
    if not prioritised_comet:
        prioritised_comet = comet_candidates

    for fp in sorted(prioritised_comet):
        marker_name = _extract_marker_name(fp)
        manifest.comet[marker_name] = fp

    # ---- Assign H&E -------------------------------------------------------
    # Prefer the largest file (H&E slides are big) if multiple candidates
    if he_candidates:
        he_sorted = sorted(he_candidates, key=lambda p: p.stat().st_size, reverse=True)
        manifest.he = he_sorted[0]
        if len(he_sorted) > 1:
            logger.info(
                "Multiple TIFF candidates for H&E; selected largest: %s",
                manifest.he.name,
            )

    logger.info("Dataset manifest built:\n%s", manifest.summary())
    return manifest


def _extract_marker_name(path: Path) -> str:
    """Derive a clean protein marker name from a file path.

    Strips common suffixes like ``.ome``, ``.tif``, ``.tiff``.

    Parameters
    ----------
    path:
        File path of the COMET image.

    Returns
    -------
    str
        Clean marker name (e.g. ``'CK8'``, ``'CD45'``).
    """
    name = path.name
    # Remove extensions
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


# ---------------------------------------------------------------------------
# Alignment matrix
# ---------------------------------------------------------------------------

def load_alignment(manifest: DatasetManifest) -> Optional[np.ndarray]:
    """Load the 3×3 affine alignment matrix from ``matrix.csv``.

    Parameters
    ----------
    manifest:
        Dataset manifest containing the matrix file path.

    Returns
    -------
    np.ndarray or None
        Shape ``(3, 3)`` float64 matrix, or ``None`` if no matrix file found.
    """
    if manifest.matrix is None:
        logger.info("No matrix.csv found; skipping alignment.")
        return None

    raw = pd.read_csv(manifest.matrix, header=None).to_numpy(dtype=np.float64)

    if raw.shape == (3, 3):
        matrix = raw
    elif raw.size == 9:
        matrix = raw.reshape(3, 3)
    elif raw.shape[0] == 1 and raw.size == 9:
        matrix = raw.reshape(3, 3)
    else:
        raise ValueError(
            f"matrix.csv must contain 9 values in a 3×3 layout, got shape {raw.shape}"
        )

    logger.info("Loaded 3×3 alignment matrix from %s", manifest.matrix.name)
    return matrix


# ---------------------------------------------------------------------------
# Xenium data
# ---------------------------------------------------------------------------

def load_cells(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Load Xenium cell-level metadata.

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    pd.DataFrame or None
        One row per cell with columns such as ``cell_id``, ``x_centroid``,
        ``y_centroid``, ``transcript_counts``, ``cell_area``, etc.
    """
    if manifest.cells is None:
        logger.warning("cells.csv not found in manifest.")
        return None
    df = safe_read_csv(manifest.cells)
    logger.info("Loaded cells.csv: %d cells, %d columns", len(df), df.shape[1])
    return df


def load_transcripts(
    manifest: DatasetManifest,
    columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Load Xenium transcript coordinates.

    Parameters
    ----------
    manifest:
        Dataset manifest.
    columns:
        Subset of columns to load (saves memory). If ``None``, load all.
        Typically: ``['x_location', 'y_location', 'feature_name']``.

    Returns
    -------
    pd.DataFrame or None
        Transcript table.
    """
    if manifest.transcripts is None:
        logger.warning("transcripts.parquet not found in manifest.")
        return None
    df = safe_read_parquet(manifest.transcripts, columns=columns)
    logger.info("Loaded transcripts: %d rows", len(df))
    return df


def load_cell_boundaries(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Load Xenium cell boundary polygons.

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    pd.DataFrame or None
        Vertex-per-row DataFrame.
    """
    if manifest.cell_boundaries is None:
        logger.warning("cell_boundaries.parquet not found in manifest.")
        return None
    df = safe_read_parquet(manifest.cell_boundaries)
    logger.info("Loaded cell boundaries: %d vertices", len(df))
    return df


def load_nucleus_boundaries(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Load Xenium nucleus boundary polygons.

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    pd.DataFrame or None
        Vertex-per-row DataFrame.
    """
    if manifest.nucleus_boundaries is None:
        logger.warning("nucleus_boundaries.parquet not found in manifest.")
        return None
    df = safe_read_parquet(manifest.nucleus_boundaries)
    logger.info("Loaded nucleus boundaries: %d vertices", len(df))
    return df


# ---------------------------------------------------------------------------
# Image loading (lazy / memory-mapped)
# ---------------------------------------------------------------------------

def load_he(manifest: DatasetManifest) -> Optional[tifffile.TiffFile]:
    """Open the H&E TIFF as a memory-mapped file (lazy).

    The image data is NOT read into RAM; only the file handle is opened.
    Call :func:`he_to_array` to materialise a region.

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    tifffile.TiffFile or None
        Open TIFF file handle, or ``None`` if not found.
    """
    if manifest.he is None:
        return None
    tif = tifffile.TiffFile(manifest.he)
    logger.info("Opened H&E TIFF (lazy): %s", manifest.he.name)
    return tif


def he_to_array(tif: tifffile.TiffFile) -> np.ndarray:
    """Read the first series/page of a TiffFile into a numpy array.

    For very large images this may use significant RAM.  For napari display,
    consider using :func:`he_to_memmap` instead.

    Parameters
    ----------
    tif:
        Open TiffFile handle (from :func:`load_he`).

    Returns
    -------
    np.ndarray
        Image array, typically shape ``(H, W, 3)`` for RGB or ``(H, W)`` for
        greyscale.
    """
    return tif.asarray()


def he_to_memmap(manifest: DatasetManifest) -> Optional[np.ndarray]:
    """Return a memory-mapped numpy view of the H&E image.

    This is the preferred approach for large images: only the requested
    tiles/slices are paged into RAM by the OS.

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    np.ndarray or None
        Memory-mapped array backed by the TIFF file, or ``None`` if not found.
    """
    if manifest.he is None:
        return None
    try:
        arr = tifffile.memmap(manifest.he)
        logger.info(
            "Memory-mapped H&E: shape=%s dtype=%s", arr.shape, arr.dtype
        )
        return arr
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Could not memory-map H&E (%s); falling back to full read.", exc
        )
        return tifffile.imread(manifest.he)


def load_comet_channel(
    path: Path,
    series: int = 0,
    level: int = 0,
) -> np.ndarray:
    """Load a single COMET OME-TIFF channel as a memory-mapped array.

    Parameters
    ----------
    path:
        Path to the OME-TIFF file for one marker.
    series:
        OME series index (usually 0 for single-channel files).
    level:
        Pyramid level (0 = full resolution).

    Returns
    -------
    np.ndarray
        2-D (or 3-D with singleton dims) image array.
    """
    try:
        arr = tifffile.memmap(path, series=series, level=level)
        # Squeeze singleton dimensions (e.g. (1, 1, H, W) → (H, W))
        arr = arr.squeeze()
        logger.info(
            "Loaded COMET channel %s: shape=%s dtype=%s",
            path.stem, arr.shape, arr.dtype,
        )
        return arr
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "memmap failed for %s (%s); falling back to imread.", path.name, exc
        )
        arr = tifffile.imread(path).squeeze()
        return arr


def get_comet_arrays(
    manifest: DatasetManifest,
) -> Dict[str, np.ndarray]:
    """Return a dict of ``{marker_name: array}`` for all COMET channels.

    Each array is memory-mapped (lazy).

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    dict[str, np.ndarray]
    """
    arrays = {}
    for marker, path in manifest.comet.items():
        arrays[marker] = load_comet_channel(path)
    return arrays


# ---------------------------------------------------------------------------
# AnnData
# ---------------------------------------------------------------------------

def load_anndata(manifest: DatasetManifest) -> Optional["anndata.AnnData"]:
    """Load the reference AnnData object (read-only).

    Parameters
    ----------
    manifest:
        Dataset manifest.

    Returns
    -------
    anndata.AnnData or None
        The AnnData object opened in backed (read-only) mode, or ``None`` if
        no ``.h5ad`` file was found.
    """
    if manifest.anndata is None:
        logger.info("No .h5ad file found in manifest.")
        return None

    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "anndata is required to load .h5ad files. "
            "Install it with: conda install anndata"
        ) from exc

    # Open in backed read-only mode to avoid loading all data into RAM
    adata = anndata.read_h5ad(manifest.anndata, backed="r")
    logger.info(
        "Loaded AnnData (backed, read-only): %d cells × %d vars",
        adata.n_obs, adata.n_vars,
    )
    return adata


# ---------------------------------------------------------------------------
# Gene / protein list generation
# ---------------------------------------------------------------------------

def generate_gene_protein_lists(
    manifest: DatasetManifest,
    transcripts_df: Optional[pd.DataFrame] = None,
    gene_col: str = "feature_name",
    output_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Generate sorted gene and protein lists and write them to CSV.

    The CSVs are written alongside the dataset folder (or to *output_dir*)
    and are used by the GUI search widgets.

    Parameters
    ----------
    manifest:
        Dataset manifest.
    transcripts_df:
        Pre-loaded transcripts DataFrame. If ``None``, the transcripts file
        is read from the manifest (only the gene name column).
    gene_col:
        Column in the transcripts DataFrame containing gene names.
    output_dir:
        Directory where ``genes.csv`` and ``proteins.csv`` are saved.
        Defaults to the dataset folder.

    Returns
    -------
    (genes, proteins) : tuple[list[str], list[str]]
        Sorted unique gene names and sorted protein marker names.
    """
    out_dir = Path(output_dir) if output_dir else manifest.folder

    # ---- Genes ------------------------------------------------------------
    if transcripts_df is None and manifest.transcripts is not None:
        transcripts_df = safe_read_parquet(
            manifest.transcripts, columns=[gene_col]
        )

    genes: List[str] = []
    if transcripts_df is not None and gene_col in transcripts_df.columns:
        genes = sorted(transcripts_df[gene_col].dropna().unique().tolist())
        genes_csv = out_dir / "genes.csv"
        pd.DataFrame({"gene": genes}).to_csv(genes_csv, index=False)
        logger.info("Wrote %d genes to %s", len(genes), genes_csv)

    # ---- Proteins ---------------------------------------------------------
    proteins = sorted(manifest.comet.keys())
    if proteins:
        proteins_csv = out_dir / "proteins.csv"
        pd.DataFrame({"protein": proteins}).to_csv(proteins_csv, index=False)
        logger.info("Wrote %d proteins to %s", len(proteins), proteins_csv)

    return genes, proteins


# ---------------------------------------------------------------------------
# High-level convenience loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """High-level loader that wraps manifest detection and all lazy loaders.

    Usage
    -----
    >>> loader = DatasetLoader("/path/to/dataset")
    >>> loader.load()
    >>> cells = loader.cells_df
    >>> he    = loader.he_array

    Attributes set after :meth:`load`
    ----------------------------------
    manifest : DatasetManifest
    cells_df : pd.DataFrame | None
    transcripts_df : pd.DataFrame | None
    cell_boundaries_df : pd.DataFrame | None
    nucleus_boundaries_df : pd.DataFrame | None
    alignment_matrix : np.ndarray | None
    he_array : np.ndarray | None   (memory-mapped)
    comet_arrays : dict[str, np.ndarray]
    anndata_ref : anndata.AnnData | None
    genes : list[str]
    proteins : list[str]
    """

    def __init__(self, folder: Union[str, Path]) -> None:
        self.folder = Path(folder).resolve()
        self.manifest: Optional[DatasetManifest] = None

        self.cells_df: Optional[pd.DataFrame] = None
        self.transcripts_df: Optional[pd.DataFrame] = None
        self.cell_boundaries_df: Optional[pd.DataFrame] = None
        self.nucleus_boundaries_df: Optional[pd.DataFrame] = None
        self.alignment_matrix: Optional[np.ndarray] = None
        self.he_array: Optional[np.ndarray] = None
        self.comet_arrays: Dict[str, np.ndarray] = {}
        self.anndata_ref = None

        self.genes: List[str] = []
        self.proteins: List[str] = []

    # ------------------------------------------------------------------
    def load(
        self,
        load_transcripts: bool = True,
        load_boundaries: bool = True,
        load_he: bool = True,
        load_comet: bool = True,
        load_adata: bool = True,
    ) -> "DatasetLoader":
        """Detect files and load all modalities.

        Parameters
        ----------
        load_transcripts:
            Whether to load the transcripts parquet. Can be deferred for
            speed (large files).
        load_boundaries:
            Whether to load boundary parquets.
        load_he:
            Whether to open the H&E image.
        load_comet:
            Whether to open COMET images.
        load_adata:
            Whether to load the AnnData reference.

        Returns
        -------
        self
            Returns ``self`` for method chaining.
        """
        self.manifest = detect_files(self.folder)

        # Always load cells (small file, used everywhere)
        self.cells_df = load_cells(self.manifest)

        if load_transcripts:
            self.transcripts_df = load_transcripts_data(self.manifest)

        if load_boundaries:
            self.cell_boundaries_df = load_cell_boundaries(self.manifest)
            self.nucleus_boundaries_df = load_nucleus_boundaries(self.manifest)

        self.alignment_matrix = load_alignment(self.manifest)

        if load_he:
            self.he_array = he_to_memmap(self.manifest)

        if load_comet:
            self.comet_arrays = get_comet_arrays(self.manifest)

        if load_adata:
            self.anndata_ref = load_anndata(self.manifest)

        # Generate gene/protein lists (also writes CSVs)
        self.genes, self.proteins = generate_gene_protein_lists(
            self.manifest, transcripts_df=self.transcripts_df
        )

        return self

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        loaded = "loaded" if self.manifest else "not loaded"
        return (
            f"DatasetLoader(folder={self.folder}, status={loaded}, "
            f"cells={len(self.cells_df) if self.cells_df is not None else 0}, "
            f"genes={len(self.genes)}, proteins={len(self.proteins)})"
        )


# alias to avoid shadowing the module-level function in DatasetLoader.load
def load_transcripts_data(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Alias for :func:`load_transcripts` used inside :class:`DatasetLoader`."""
    return load_transcripts(manifest)
