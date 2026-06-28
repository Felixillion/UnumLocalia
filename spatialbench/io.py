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
        matrix.csv                  ← 3×3 affine alignment matrix (legacy)
        matrix_comet.csv            ← 2x3 or 3x3 affine matrix for COMET (preferred)
        matrix_he.csv               ← 2x3 or 3x3 affine matrix for H&E (preferred)
        keypoints_comet.csv         ← optional keypoints (src/dst)
        keypoints_he.csv            ← optional keypoints (src/dst)
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
from typing import Dict, List, Optional, Tuple, Union

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
_PATTERN_MATRIX_MOD = re.compile(r"^matrix_(?P<modality>\w+)\.csv$", re.IGNORECASE)
_PATTERN_KEYPOINTS_MOD = re.compile(r"^keypoints_(?P<modality>\w+)\.csv$", re.IGNORECASE)
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

    # Legacy single alignment matrix (3x3)
    matrix: Optional[Path] = None

    # Per-modality alignment matrices (e.g., 'comet' -> Path('.../matrix_comet.csv'))
    alignment: Dict[str, Path] = field(default_factory=dict)

    # Per-modality keypoints (e.g., 'comet' -> Path('.../keypoints_comet.csv'))
    keypoints: Dict[str, Path] = field(default_factory=dict)

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
        lines.append(f"  Alignments    : {', '.join(self.alignment.keys()) if self.alignment else '—'}")
        lines.append(f"  Keypoints     : {', '.join(self.keypoints.keys()) if self.keypoints else '—'}")
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

        # modality-specific matrix (matrix_comet.csv, matrix_he.csv, etc.)
        m_mod = _PATTERN_MATRIX_MOD.match(name)
        if m_mod:
            mod = m_mod.group("modality").lower()
            manifest.alignment[mod] = fp
            continue

        # modality-specific keypoints (keypoints_comet.csv, keypoints_he.csv, etc.)
        k_mod = _PATTERN_KEYPOINTS_MOD.match(name)
        if k_mod:
            mod = k_mod.group("modality").lower()
            manifest.keypoints[mod] = fp
            continue

        if _PATTERN_CELLS.match(name):
            manifest.cells = fp

        elif _PATTERN_CELL_BOUNDS.match(name):
            manifest.cell_boundaries = fp

        elif _PATTERN_NUC_BOUNDS.match(name):
            manifest.nucleus_boundaries = fp

        elif _PATTERN_TRANSCRIPTS.match(name):
            manifest.transcripts = fp

        elif _PATTERN_MATRIX.match(name):
            # legacy single matrix.csv
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
# Alignment matrix helpers
# ---------------------------------------------------------------------------

def read_affine_matrix(path: Path) -> np.ndarray:
    """Read a 2x3 or 3x3 affine matrix from CSV and return as numpy array."""
    raw = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    # Accept common shapes: (2,3), (3,3), flattened 6 or 9 values
    if raw.shape == (2, 3):
        return raw
    if raw.shape == (3, 3):
        return raw
    if raw.size == 6:
        return raw.reshape(2, 3)
    if raw.size == 9:
        return raw.reshape(3, 3)
    # Some CSVs may be a single row with 9 values
    if raw.shape[0] == 1 and raw.size == 9:
        return raw.reshape(3, 3)
    raise ValueError(f"Expected 2x3 or 3x3 matrix in {path}, got {raw.shape}")


def read_keypoints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read keypoints CSV and return (src_pts, dst_pts) as float32 arrays.

    Accepts headers:
      - fixedX,fixedY,alignmentX,alignmentY  (Xenium Explorer export)
      - source_x,source_y,target_x,target_y
    Or infers columns by position (first two = src, next two = dst).
    """
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]

    # Xenium Explorer format
    if set(['fixedx', 'fixedy', 'alignmentx', 'alignmenty']).issubset(cols):
        fixedx_i = cols.index('fixedx')
        fixedy_i = cols.index('fixedy')
        alignx_i = cols.index('alignmentx')
        aligny_i = cols.index('alignmenty')
        dst_x = df.iloc[:, fixedx_i].to_numpy(dtype=np.float32)
        dst_y = df.iloc[:, fixedy_i].to_numpy(dtype=np.float32)
        src_x = df.iloc[:, alignx_i].to_numpy(dtype=np.float32)
        src_y = df.iloc[:, aligny_i].to_numpy(dtype=np.float32)
    # Generic source/target names
    elif set(['source_x', 'source_y', 'target_x', 'target_y']).issubset(cols):
        src_x = df.iloc[:, cols.index('source_x')].to_numpy(dtype=np.float32)
        src_y = df.iloc[:, cols.index('source_y')].to_numpy(dtype=np.float32)
        dst_x = df.iloc[:, cols.index('target_x')].to_numpy(dtype=np.float32)
        dst_y = df.iloc[:, cols.index('target_y')].to_numpy(dtype=np.float32)
    else:
        # Try positional inference: first two columns = src, next two = dst
        arr = df.to_numpy(dtype=np.float32)
        if arr.shape[1] >= 4:
            src_x, src_y, dst_x, dst_y = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        else:
            raise ValueError(f"Unrecognized keypoints format in {path}")
    src = np.vstack([src_x, src_y]).T
    dst = np.vstack([dst_x, dst_y]).T
    return src, dst


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply affine (2x3 or 3x3) to Nx2 points and return Nx2 array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    if matrix.shape == (2, 3):
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homo = np.hstack([pts, ones])  # (N,3)
        out = homo @ matrix.T
        return out
    if matrix.shape == (3, 3):
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homo = np.hstack([pts, ones])
        out_h = homo @ matrix.T
        return out_h[:, :2]
    raise ValueError("Matrix must be 2x3 or 3x3")


def apply_affine_to_image(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]):
    """Warp image using OpenCV. output_shape is (height, width)."""
    try:
        import cv2
    except Exception as exc:
        raise ImportError("OpenCV (cv2) is required for image warping. Install via conda-forge opencv.") from exc

    if matrix.shape == (3, 3):
        M = matrix[:2, :]
    else:
        M = matrix
    h, w = output_shape
    warped = cv2.warpAffine(image, M.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR)
    return warped


def load_alignments(manifest: DatasetManifest) -> Dict[str, np.ndarray]:
    """Load modality-specific alignment matrices and return dict {modality: matrix}.

    Populates matrices for modalities found in manifest.alignment. If no
    modality-specific matrices are found but a legacy matrix.csv exists, the
    legacy matrix is returned under the key 'global'.
    """
    matrices: Dict[str, np.ndarray] = {}

    # modality-specific first
    for mod, path in manifest.alignment.items():
        try:
            matrices[mod] = read_affine_matrix(path)
            logger.info("Loaded alignment for %s from %s", mod, path.name)
        except Exception as exc:
            logger.warning("Failed to read alignment for %s: %s", mod, exc)

    # fallback: legacy matrix.csv applies to all modalities if present and no per-modality found
    if manifest.matrix is not None and not matrices:
        try:
            matrices['global'] = read_affine_matrix(manifest.matrix)
            logger.info("Loaded legacy matrix.csv as global alignment")
        except Exception as exc:
            logger.warning("Failed to read legacy matrix.csv: %s", exc)

    # Note: keypoints are kept on manifest.keypoints (paths). Reading keypoints
    # is done on demand via read_keypoints() when needed (viewer, recompute).
    return matrices


# ---------------------------------------------------------------------------
# Xenium data
# ---------------------------------------------------------------------------

def load_cells(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Load Xenium cell-level metadata."""
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
    """Load Xenium transcript coordinates."""
    if manifest.transcripts is None:
        logger.warning("transcripts.parquet not found in manifest.")
        return None
    df = safe_read_parquet(manifest.transcripts, columns=columns)
    logger.info("Loaded transcripts: %d rows", len(df))
    return df


def load_cell_boundaries(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Load Xenium cell boundary polygons."""
    if manifest.cell_boundaries is None:
        logger.warning("cell_boundaries.parquet not found in manifest.")
        return None
    df = safe_read_parquet(manifest.cell_boundaries)
    logger.info("Loaded cell boundaries: %d vertices", len(df))
    return df


def load_nucleus_boundaries(manifest: DatasetManifest) -> Optional[pd.DataFrame]:
    """Load Xenium nucleus boundary polygons."""
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
    """Open the H&E TIFF as a memory-mapped file (lazy)."""
    if manifest.he is None:
        return None
    tif = tifffile.TiffFile(manifest.he)
    logger.info("Opened H&E TIFF (lazy): %s", manifest.he.name)
    return tif


def he_to_array(tif: tifffile.TiffFile) -> np.ndarray:
    """Read the first series/page of a TiffFile into a numpy array."""
    return tif.asarray()


def he_to_memmap(manifest: DatasetManifest) -> Optional[np.ndarray]:
    """Return a memory-mapped numpy view of the H&E image."""
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
    """Load a single COMET OME-TIFF channel as a memory-mapped array."""
    try:
        arr = tifffile.memmap(path, series=series, level=level)
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
    """Return a dict of ``{marker_name: array}`` for all COMET channels."""
    arrays = {}
    for marker, path in manifest.comet.items():
        arrays[marker] = load_comet_channel(path)
    return arrays


# ---------------------------------------------------------------------------
# AnnData
# ---------------------------------------------------------------------------

def load_anndata(manifest: DatasetManifest) -> Optional["anndata.AnnData"]:
    """Load the reference AnnData object (read-only)."""
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
    """Generate sorted gene and protein lists and write them to CSV."""
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
    alignment_matrix : np.ndarray | None   # legacy single matrix
    alignment_matrices : dict[str, np.ndarray]  # per-modality matrices
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
        self.alignment_matrix: Optional[np.ndarray] = None  # legacy single matrix
        self.alignment_matrices: Dict[str, np.ndarray] = {}  # per-modality
        self.he_array: Optional[np.ndarray] = None
        self.comet_arrays: Dict[str, np.ndarray] = {}
        self.anndata_ref = None

        self.genes: List[str] = []
        self.proteins: List[str] = []

    # ------------------------------------------------------------------
    def load(
        self,
        do_load_transcripts: bool = True,
        load_boundaries: bool = True,
        load_he: bool = True,
        load_comet: bool = True,
        load_adata: bool = True,
    ) -> "DatasetLoader":
        """Detect files and load all modalities."""
        self.manifest = detect_files(self.folder)

        # Always load cells (small file, used everywhere)
        self.cells_df = load_cells(self.manifest)

        if do_load_transcripts:
            self.transcripts_df = load_transcripts(self.manifest)

        if load_boundaries:
            self.cell_boundaries_df = load_cell_boundaries(self.manifest)
            self.nucleus_boundaries_df = load_nucleus_boundaries(self.manifest)

        # Load per-modality alignments (and fallback to legacy matrix.csv)
        self.alignment_matrices = load_alignments(self.manifest)
        # Keep legacy single matrix for backward compatibility
        self.alignment_matrix = self.alignment_matrices.get('global', None)

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
