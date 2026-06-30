"""
spatialbench.io
===============
Dataset discovery, manifest building, and lazy data loading for multi-core datasets.

Philosophy
----------
* Uses `dataset_manifest.csv` to map out per-core files.
* Use memory-mapped access for large TIFF files (OME-TIFF, H&E) so that only
  the requested tiles are read into RAM.
* Never modify any file on disk.
"""

from __future__ import annotations

import logging
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import tifffile
import zarr

from spatialbench.utils import safe_read_csv, safe_read_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoreManifest & DatasetManifest
# ---------------------------------------------------------------------------

@dataclass
class CoreManifest:
    """Paths to all detected files for a single core."""
    core_id: str
    core_folder: Path
    
    comet_file: Optional[Path] = None
    comet_thresholds: Optional[Path] = None
    he_file: Optional[Path] = None
    anndata_file: Optional[Path] = None
    xenium_folder: Optional[Path] = None
    
    alignment_comet: Optional[Path] = None
    alignment_he: Optional[Path] = None

    # Xenium specific files (resolved from xenium_folder)
    cells: Optional[Path] = None
    cell_boundaries: Optional[Path] = None
    nucleus_boundaries: Optional[Path] = None
    transcripts: Optional[Path] = None


@dataclass
class DatasetManifest:
    """Paths to all cores in a SpatialBench dataset."""
    folder: Path
    cores: Dict[str, CoreManifest] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of detected cores."""
        lines = [f"Dataset folder : {self.folder}"]
        lines.append(f"Found {len(self.cores)} cores:")
        for cid, core in self.cores.items():
            lines.append(f"  - {cid}")
            lines.append(f"      H&E       : {'✓' if core.he_file else '✗'}")
            lines.append(f"      COMET     : {'✓' if core.comet_file else '✗'}")
            lines.append(f"      Xenium    : {'✓' if core.xenium_folder else '✗'}")
            lines.append(f"      AnnData   : {'✓' if core.anndata_file else '✗'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File detection
# ---------------------------------------------------------------------------

def detect_files(folder: Path) -> DatasetManifest:
    """Reads dataset_manifest.csv in folder and builds DatasetManifest."""
    folder = Path(folder).resolve()
    manifest_csv = folder / "dataset_manifest.csv"
    
    if not manifest_csv.exists():
        raise FileNotFoundError(f"dataset_manifest.csv not found in {folder}")

    df = pd.read_csv(manifest_csv)
    manifest = DatasetManifest(folder=folder)

    for _, row in df.iterrows():
        if pd.isna(row.get('core_id')):
            continue
            
        cid = str(row['core_id'])
        
        # Resolve core folder if it exists
        core_folder_str = str(row.get('core_folder', ''))
        cf = folder / core_folder_str if core_folder_str else folder
        
        core = CoreManifest(core_id=cid, core_folder=cf)

        if 'comet_file' in row and pd.notna(row['comet_file']):
            core.comet_file = (folder / str(row['comet_file'])).resolve()
            thresh = core.comet_file.parent / "comet_thresholding.csv"
            if thresh.exists():
                core.comet_thresholds = thresh

        if 'he_file' in row and pd.notna(row['he_file']):
            core.he_file = (folder / str(row['he_file'])).resolve()
        
        if 'anndata_file' in row and pd.notna(row['anndata_file']):
            core.anndata_file = (folder / str(row['anndata_file'])).resolve()

        if 'alignment_comet' in row and pd.notna(row['alignment_comet']):
            core.alignment_comet = (folder / str(row['alignment_comet'])).resolve()

        if 'alignment_he' in row and pd.notna(row['alignment_he']):
            core.alignment_he = (folder / str(row['alignment_he'])).resolve()
        
        if 'xenium_folder' in row and pd.notna(row['xenium_folder']):
            xf = (folder / str(row['xenium_folder'])).resolve()
            core.xenium_folder = xf
            # Automatically find the canonical Xenium files inside this folder
            cells = xf / "cells.csv"
            if cells.exists(): core.cells = cells
            
            cb = xf / "cell_boundaries.parquet"
            if cb.exists(): core.cell_boundaries = cb
            
            nb = xf / "nucleus_boundaries.parquet"
            if nb.exists(): core.nucleus_boundaries = nb
            
            tx = xf / "transcripts.parquet"
            if tx.exists(): core.transcripts = tx
            
        manifest.cores[cid] = core
    
    logger.info("Dataset manifest built:\n%s", manifest.summary())
    return manifest


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
    if raw.shape[0] == 1 and raw.size == 9:
        return raw.reshape(3, 3)
    raise ValueError(f"Expected 2x3 or 3x3 matrix in {path}, got {raw.shape}")

def normalize_affine_matrix(M: np.ndarray) -> np.ndarray:
    """Zero out translation so all cores start at same origin."""
    M = np.asarray(M, dtype=np.float64).copy()
    if M.ndim == 2 and M.shape[1] >= 3:
        M[:, 2] = 0.0
    return M


# -------------------------
# Robust image loader (memmap -> imread -> aszarr)
# -------------------------
def load_image_robust(path: Path):
    """Try to load an image robustly. Return numpy array or zarr-like array.

    For .ome.zarr folders:
      - open with zarr and return the full-res level 0 group.

    For TIFF:
      - try memmap, then imread, then aszarr.
    """
    # OME-Zarr case: folder ending with .ome.zarr
    if path.is_dir() and (path.suffix == ".zarr" or path.suffixes == [".ome", ".zarr"]):
        root = zarr.open(path, mode="r")
        # full-res level is "0"
        if "0" in root:
            logger.info("Opened OME-Zarr %s (level 0)", path.name)
            return root["0"]
        logger.info("Opened OME-Zarr %s (no explicit level 0)", path.name)
        return root

    # TIFF case
    try:
        arr = tifffile.memmap(path)
        arr = np.asarray(arr)
        logger.info("memmap OK for %s shape=%s", path.name, getattr(arr, "shape", None))
        return arr
    except Exception as e_mem:
        logger.debug("memmap failed for %s: %s", path.name, e_mem)

    try:
        arr = tifffile.imread(path)
        arr = np.asarray(arr)
        logger.info("imread OK for %s shape=%s", path.name, getattr(arr, "shape", None))
        return arr
    except Exception as e_im:
        logger.debug("imread failed for %s: %s", path.name, e_im)

    try:
        tf = tifffile.TiffFile(path)
        try:
            z = tf.aszarr()
            logger.info("aszarr OK for %s (zarr-like)", path.name)
            return z
        finally:
            tf.close()
    except Exception as e_z:
        logger.debug("aszarr failed for %s: %s", path.name, e_z)

    raise RuntimeError(f"Failed to load image {path.name} with memmap/imread/aszarr.")


# ---------------------------------------------------------------------------
# High-level convenience loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """High-level loader for multi-core datasets."""

    def __init__(self, folder: Union[str, Path]) -> None:
        self.folder = Path(folder).resolve()
        self.manifest: Optional[DatasetManifest] = None

        # Per-core data dictionaries
        self.cells_df: Dict[str, pd.DataFrame] = {}
        self.transcripts_df: Dict[str, pd.DataFrame] = {}
        self.cell_boundaries_df: Dict[str, pd.DataFrame] = {}
        self.nucleus_boundaries_df: Dict[str, pd.DataFrame] = {}
        
        self.alignment_matrices_comet: Dict[str, np.ndarray] = {}
        self.alignment_matrices_he: Dict[str, np.ndarray] = {}
        
        self.he_arrays: Dict[str, np.ndarray] = {}
        self.comet_arrays: Dict[str, np.ndarray] = {}
        self.comet_markers: Dict[str, List[str]] = {}
        self.comet_thresholds: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        self.anndata_refs: Dict[str, Any] = {}

        # Lazy COMET handling: store paths and a small cache
        self.comet_paths_by_core: Dict[str, Path] = {}
        self._comet_cache: Dict[Tuple[str, int], Any] = {}  # (core_id, channel_index) -> array or zarr

        # Global aggregations for UI menus
        self.genes: List[str] = []
        self.proteins: List[str] = []

    def load(self, do_load_transcripts: bool = True, load_boundaries: bool = True,
             load_he: bool = True, load_comet: bool = True, load_adata: bool = True) -> "DatasetLoader":
        
        self.manifest = detect_files(self.folder)
        all_genes = set()
        all_proteins = set()

        for core_id, core in self.manifest.cores.items():
            
            # --- Xenium ---
            if core.cells:
                self.cells_df[core_id] = safe_read_csv(core.cells)
            
            # Always read feature_name metadata to populate gene list,
            # but only keep full DataFrame if do_load_transcripts=True.
            if core.transcripts:
                try:
                    df_tx_meta = safe_read_parquet(core.transcripts, columns=["feature_name"])
                    if "feature_name" in df_tx_meta.columns:
                        all_genes.update(df_tx_meta["feature_name"].dropna().unique().tolist())
                    if do_load_transcripts:
                        df_tx = safe_read_parquet(core.transcripts)
                        self.transcripts_df[core_id] = df_tx
                except Exception as e:
                    logger.warning("Failed to read Xenium transcripts metadata for %s: %s", core_id, e)
                    
            if load_boundaries:
                if core.cell_boundaries:
                    self.cell_boundaries_df[core_id] = safe_read_parquet(core.cell_boundaries)
                if core.nucleus_boundaries:
                    self.nucleus_boundaries_df[core_id] = safe_read_parquet(core.nucleus_boundaries)
                    
            # --- Alignments ---
            if core.alignment_comet:
                try:
                    M = read_affine_matrix(core.alignment_comet)
                    self.alignment_matrices_comet[core_id] = normalize_affine_matrix(M)
                except Exception as e:
                    logger.warning("Failed to load COMET alignment for %s: %s", core_id, e)
            
            if core.alignment_he:
                try:
                    M = read_affine_matrix(core.alignment_he)
                    self.alignment_matrices_he[core_id] = normalize_affine_matrix(M)
                except Exception as e:
                    logger.warning("Failed to load H&E alignment for %s: %s", core_id, e)
            
            # --- H&E Images (Lazy) ---
            if load_he and core.he_file:
                try:
                    arr = load_image_robust(core.he_file)
                    if isinstance(arr, zarr.hierarchy.Group):
                        if "0" in arr:
                            he5d = arr["0"]          # shape (T, C, Z, Y, X)
                            he = he5d[0, :, 0, :, :]   # (3, Y, X)
                            he = np.moveaxis(he, 0, -1)  # → (Y, X, 3)
                            self.he_arrays[core_id] = he
                        else:
                            raise RuntimeError("H&E OME-Zarr missing channel 0")
                    else:
                        self.he_arrays[core_id] = arr
                except Exception as e:
                    logger.warning("Failed to load H&E for %s: %s", core_id, e)
                        
            # --- COMET (lazy) ---
            if load_comet and core.comet_file:
                self.comet_paths_by_core[core_id] = core.comet_file

                markers: List[str] = []
                thresholds: Dict[str, Tuple[float, float]] = {}

                # 1) thresholds CSV: row0=Channel, row1=Min, row2=Max
                if core.comet_thresholds:
                    try:
                        tdf = safe_read_csv(core.comet_thresholds, header=None)
                        if tdf.shape[0] >= 3:
                            names_row = tdf.iloc[0].dropna()
                            mins_row = tdf.iloc[1]
                            maxs_row = tdf.iloc[2]
                            for col_idx, name in names_row.items():
                                name_str = str(name).strip()
                                if not name_str or "Unnamed" in name_str:
                                    continue
                                name_str = html.unescape(name_str)
                                if "#945;SMA" in name_str:
                                    name_str = name_str.replace("#945;SMA", "αSMA")
                                markers.append(name_str)
                                try:
                                    min_val = float(mins_row[col_idx])
                                    max_val = float(maxs_row[col_idx])
                                except Exception:
                                    min_val, max_val = 0.0, 1000.0
                                thresholds[name_str] = (min_val, max_val)
                        else:
                            logger.warning("comet_thresholding.csv for %s has unexpected shape %s",
                                           core_id, tdf.shape)
                    except Exception as e:
                        logger.debug("Failed to read comet_thresholds for %s: %s", core_id, e)

                # 2) OME-XML channel names override
                try:
                    from xml.etree import ElementTree as ET
                    xml_path = core.comet_file / "OME" / "METADATA.ome.xml"
                    if xml_path.exists():
                        tree = ET.parse(xml_path)
                        root_xml = tree.getroot()
                        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

                        xml_markers: List[str] = []
                        for ch in root_xml.findall(".//ome:Channel", ns):
                            name = ch.get("Name")
                            if name:
                                name_str = html.unescape(name).strip()
                                if "#945;SMA" in name_str:
                                    name_str = name_str.replace("#945;SMA", "αSMA")
                                xml_markers.append(name_str)

                        if xml_markers:
                            new_thresholds: Dict[str, Tuple[float, float]] = {}
                            for m in xml_markers:
                                if m in thresholds:
                                    new_thresholds[m] = thresholds[m]
                                else:
                                    new_thresholds[m] = (0.0, 1000.0)
                            markers = xml_markers
                            thresholds = new_thresholds
                except Exception as e:
                    logger.warning("Failed to parse OME-XML for %s: %s", core_id, e)

                self.comet_markers[core_id] = markers
                self.comet_thresholds[core_id] = thresholds
                all_proteins.update(markers)

            # --- AnnData ---
            if load_adata and core.anndata_file:
                try:
                    import anndata
                    self.anndata_refs[core_id] = anndata.read_h5ad(core.anndata_file, backed="r")
                except Exception as e:
                    logger.warning("Failed to load AnnData for %s: %s", core_id, e)
                    
        # Filter genes to remove control/deprecated codewords
        bad_prefixes = (
            "DeprecatedCodeword",
            "NegControlCodeword",
            "UnassignedCodeword",
            "NegControlProbe",
        )
        self.genes = sorted(
            g for g in all_genes
            if isinstance(g, str) and not any(g.startswith(p) for p in bad_prefixes)
        )
        self.proteins = sorted(list(all_proteins))
        
        try:
            if self.genes:
                pd.DataFrame({"gene": self.genes}).to_csv(self.folder / "genes.csv", index=False)
            if self.proteins:
                pd.DataFrame({"protein": self.proteins}).to_csv(self.folder / "proteins.csv", index=False)
        except Exception:
            pass

        return self
    
    def get_comet_channel(self, core_id: str, channel_index: int = 0, use_cache: bool = True):
        """Load a single COMET channel (by index) for a core on demand."""
        key = (core_id, int(channel_index))
        if use_cache and key in self._comet_cache:
            return self._comet_cache[key]

        path = self.comet_paths_by_core.get(core_id)
        if path is None:
            raise FileNotFoundError(f"No COMET file recorded for core {core_id}")

        # OME-Zarr case
        if path.is_dir() and (path.suffix == ".zarr" or path.suffixes == [".ome", ".zarr"]):
            try:
                root = zarr.open(path, mode="r")
                if "0" not in root:
                    raise RuntimeError(f"OME-Zarr {path} missing level 0")

                level0 = root["0"]
                # level0 may be an Array or a group with "0" as full-res
                if isinstance(level0, zarr.core.Array):
                    arr5d = level0[:]          # (T, C, Z, Y, X)
                else:
                    if "0" not in level0:
                        raise RuntimeError(f"OME-Zarr {path}/0 missing full-res array")
                    fullres = level0["0"]
                    arr5d = fullres[:]         # (T, C, Z, Y, X)

                arr2d = arr5d[0, channel_index, 0, :, :]  # (Y, X)

                self._comet_cache[key] = arr2d
                return arr2d

            except Exception as exc:
                logger.exception(
                    "Failed to load COMET channel %s for core %s from OME-Zarr: %s",
                    channel_index, core_id, exc
                )
                raise

        # TIFF fallback
        try:
            tf = tifffile.TiffFile(path)
            try:
                arr_full = tf.asarray()
                arr_full = np.asarray(arr_full)
                if arr_full.ndim == 3:
                    if arr_full.shape[0] <= 64 and arr_full.shape[0] < arr_full.shape[-1]:
                        channel = arr_full[channel_index]
                    else:
                        channel = arr_full[..., channel_index]
                elif arr_full.ndim == 4:
                    channel = arr_full[0, channel_index]
                else:
                    channel = arr_full
                self._comet_cache[key] = channel
                return channel
            finally:
                tf.close()
        except Exception as exc:
            logger.exception("Failed to load COMET channel %s for core %s from TIFF: %s",
                             channel_index, core_id, exc)
            raise

    def __repr__(self) -> str:
        loaded = "loaded" if self.manifest else "not loaded"
        return (
            f"DatasetLoader(folder={self.folder}, status={loaded}, "
            f"cores={len(self.manifest.cores) if self.manifest else 0}, "
            f"genes={len(self.genes)}, proteins={len(self.proteins)})"
        )