# spatialbench/io.py
"""
Dataset discovery, manifest building, and lazy data loading for multi-core datasets.

This version includes robust zarr handling and helpers to rasterize GeoJSON cell
polygons into per-core label masks, compute per-cell COMET statistics, and assign
transcripts to cells using the mask.
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
import json

from shapely.geometry import shape, Polygon
from shapely.validation import make_valid

from spatialbench.utils import safe_read_csv, safe_read_parquet

# PIL for rasterization
from PIL import Image, ImageDraw

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

    # Optional manifest-provided GeoJSON path (relative or absolute)
    xenium_geojson_comet: Optional[str] = None


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

        # Optional GeoJSON path column in manifest
        if 'xenium_geojson_comet' in row and pd.notna(row['xenium_geojson_comet']):
            core.xenium_geojson_comet = str(row['xenium_geojson_comet'])

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
    """Zero out translation so all cores start at same origin while preserving homogeneous row."""
    M = np.asarray(M, dtype=np.float64).copy()
    # 2x3 case: zero translation column (last column)
    if M.ndim == 2 and M.shape == (2, 3):
        M[:, 2] = 0.0
        return M
    # 3x3 case: zero translation entries but keep bottom row [0,0,1]
    if M.ndim == 2 and M.shape == (3, 3):
        M[0, 2] = 0.0
        M[1, 2] = 0.0
        M[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return M
    # fallback: if shape unusual, attempt to zero last column if present
    if M.ndim == 2 and M.shape[1] >= 3:
        M[:, 2] = 0.0
    return M


def _compute_centering_scale_correction(M: np.ndarray, pts_mapped: np.ndarray, img_w: int, img_h: int, target_fraction: float = 0.9) -> np.ndarray:
    """
    Compute a conservative uniform scale + translation correction in image pixel space
    that centers the mapped points and scales them to occupy `target_fraction` of the image max dimension.
    Returns a 3x3 correction matrix C such that M_corrected = C @ M.
    """
    mn = pts_mapped.min(axis=0)
    mx = pts_mapped.max(axis=0)
    rng = mx - mn
    coord_max_dim = max(rng[0], rng[1], 1.0)
    img_max_dim = max(img_w, img_h, 1.0)
    s = (img_max_dim * target_fraction) / coord_max_dim

    center_pts = pts_mapped.mean(axis=0)
    center_img = np.array([img_w / 2.0, img_h / 2.0])

    # translate cloud to origin, scale, translate to image center
    T1 = np.array([[1, 0, -center_pts[0]], [0, 1, -center_pts[1]], [0, 0, 1]], dtype=float)
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=float)
    T2 = np.array([[1, 0, center_img[0]], [0, 1, center_img[1]], [0, 0, 1]], dtype=float)

    C = T2 @ S @ T1
    return C


# -------------------------
# Robust image loader (memmap -> imread -> aszarr)
# -------------------------
def load_image_robust(path: Path):
    """Try to load an image robustly. Return numpy array or zarr-like array.

    For .ome.zarr folders:
      - open with zarr and return the full-res level 0 group or array.

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

        # Alignment
        self.alignment_matrices_comet: Dict[str, np.ndarray] = {}
        self.alignment_matrices_he: Dict[str, np.ndarray] = {}
        self.alignment_matrices_comet_raw: Dict[str, np.ndarray] = {}
        self.alignment_matrices_he_raw: Dict[str, np.ndarray] = {}

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

        # store chosen affine to map transcript (x,y) in microns -> image (x,y) pixels
        # Keys: core_id -> 3x3 numpy array mapping Xenium µm -> COMET pixels
        self.transcript_affine_by_core: Dict[str, np.ndarray] = {}
        # xenium pixel size in microns (set from manifest or OME-XML if available)
        self.xenium_pixel_size_um: Optional[float] = None

        # cell masks
        self.cell_mask_by_core: Dict[str, np.ndarray] = {}
        self.cell_label_to_id_by_core: Dict[str, Dict[int, str]] = {}
        self.comet_cell_stats: Dict[str, Any] = {}

        # User-imported segmentations
        self.custom_segmentations: Dict[str, Dict[str, Dict[str, Any]]] = {}

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
            # COMET alignment
            if core.alignment_comet:
                try:
                    M_raw = read_affine_matrix(core.alignment_comet)
                    # store raw and normalized separately
                    self.alignment_matrices_comet_raw[core_id] = M_raw.copy()
                    self.alignment_matrices_comet[core_id] = normalize_affine_matrix(M_raw)
                except Exception as e:
                    logger.warning("Failed to load COMET alignment for %s: %s", core_id, e)

            # H&E alignment
            if core.alignment_he:
                try:
                    M_raw = read_affine_matrix(core.alignment_he)
                    self.alignment_matrices_he_raw[core_id] = M_raw.copy()
                    self.alignment_matrices_he[core_id] = normalize_affine_matrix(M_raw)
                except Exception as e:
                    logger.warning("Failed to load H&E alignment for %s: %s", core_id, e)

            # --- H&E Images (Lazy) ---
            if load_he and core.he_file:
                try:
                    arr = load_image_robust(core.he_file)
                    # robust check for zarr group/array
                    try:
                        # prefer duck-typing: if object has keys() and '0' in it, treat as group
                        if hasattr(arr, "keys") and "0" in arr:
                            he5d = arr["0"]
                            # he5d expected shape (T, C, Z, Y, X)
                            he = he5d[0, :, 0, :, :]   # (C, Y, X)
                            he = np.moveaxis(he, 0, -1)  # → (Y, X, C)
                            self.he_arrays[core_id] = he
                        else:
                            self.he_arrays[core_id] = np.asarray(arr)
                    except Exception:
                        self.he_arrays[core_id] = np.asarray(arr)
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

                        # Try to read PhysicalSizeX/Y from OME-XML if present (useful for Xenium pixel size)
                        try:
                            pix_node = root_xml.find(".//ome:Pixels", ns)
                            if pix_node is not None:
                                phys_x = pix_node.get("PhysicalSizeX")
                                if phys_x:
                                    try:
                                        px_um = float(phys_x)
                                        if self.xenium_pixel_size_um is None:
                                            self.xenium_pixel_size_um = px_um

                                        # actual assignment (unchanged)
                                        self.xenium_pixel_size_um = px_um

                                        logger.info("Found PhysicalSizeX in OME-XML (µm): %s", phys_x)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
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

            # --- Attempt to fit transcript affine from GeoJSON + Xenium boundaries ---
            # This is done after boundaries and transcripts are loaded for the core.
            try:
                self._fit_affine_from_geojson_and_xenium(core_id, core)
            except Exception:
                # don't fail loading if fitting fails
                logger.debug("GeoJSON affine fit skipped or failed for core %s", core_id)

        # Filter genes to remove control/deprecated codewords
        bad_prefixes = (
            "DeprecatedCodeword",
            "NegControlCodeword",
            "UnassignedCodeword",
            "NegControlProbe",
            "Intergenic"
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

    def _fit_affine_from_geojson_and_xenium(self, core_id: str, core_manifest: CoreManifest) -> None:
        """
        If a GeoJSON of COMET-space cell polygons exists, fit a 3x3 affine A mapping
        Xenium cell centroids (µm) -> COMET pixel centroids and store in self.transcript_affine_by_core.
        """
        # locate geojson: prefer manifest column, else xenium_folder/cell_boundaries_comet_space.geojson
        gj_path: Optional[Path] = None
        if core_manifest.xenium_geojson_comet:
            candidate = Path(core_manifest.xenium_geojson_comet)
            if not candidate.is_absolute():
                candidate = self.folder / candidate
            if candidate.exists():
                gj_path = candidate
        elif core_manifest.xenium_folder:
            candidate = core_manifest.xenium_folder / "cell_boundaries_comet_space.geojson"
            if candidate.exists():
                gj_path = candidate

        if gj_path is None or not gj_path.exists():
            return

        # load xenium cell centroids (µm)
        try:
            df_x = self.cell_boundaries_df.get(core_id)
            if df_x is None or df_x.empty:
                return
            cent_x: Dict[str, Tuple[float, float]] = {}
            for cid, group in df_x.groupby("cell_id"):
                xs = group["vertex_x"].to_numpy(dtype=float)
                ys = group["vertex_y"].to_numpy(dtype=float)
                if len(xs) < 3:
                    continue
                poly = Polygon(np.column_stack([xs, ys]))
                if not poly.is_valid:
                    poly = make_valid(poly)
                    if poly.geom_type == "MultiPolygon":
                        poly = max(poly.geoms, key=lambda p: p.area)
                if poly.is_valid and poly.area > 0:
                    c = poly.centroid
                    cent_x[str(cid)] = (c.x, c.y)
        except Exception:
            logger.debug("Failed to compute Xenium centroids for core %s", core_id)
            return

        # load geojson centroids (COMET pixels)
        try:
            with open(gj_path, "r") as f:
                gj = json.load(f)
        except Exception:
            logger.debug("Failed to read GeoJSON %s for core %s", gj_path, core_id)
            return

        cent_c: Dict[str, Tuple[float, float]] = {}
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            cid = str(props.get("cell_id") or props.get("name"))
            geom = feat.get("geometry")
            if geom is None:
                continue
            poly = shape(geom)
            if not poly.is_valid:
                poly = make_valid(poly)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
            if poly.is_valid and poly.area > 0:
                c = poly.centroid
                cent_c[cid] = (c.x, c.y)

        # build matched arrays
        src = []
        dst = []
        for cid, s in cent_x.items():
            if cid in cent_c:
                src.append(s)      # Xenium µm
                dst.append(cent_c[cid])  # COMET pixels

        if len(src) < 6:
            # not enough matches for a stable fit
            logger.debug("Not enough centroid matches for core %s (found=%d)", core_id, len(src))
            return

        src = np.array(src, dtype=float)
        dst = np.array(dst, dtype=float)

        # fit affine (least squares)
        N = src.shape[0]
        X = np.zeros((2*N, 6), dtype=float)
        y = np.zeros((2*N,), dtype=float)
        for i in range(N):
            x0, y0 = src[i,0], src[i,1]
            u, v = dst[i,0], dst[i,1]
            X[2*i]   = [x0, y0, 1, 0, 0, 0]
            X[2*i+1] = [0, 0, 0, x0, y0, 1]
            y[2*i]   = u
            y[2*i+1] = v
        params, *_ = np.linalg.lstsq(X, y, rcond=None)
        A = np.array([[params[0], params[1], params[2]],
                      [params[3], params[4], params[5]],
                      [0.0,       0.0,       1.0]])
        # store and log residual
        H = np.hstack([src, np.ones((src.shape[0],1))])
        mapped = (H @ A.T)[:, :2]
        rms = np.sqrt(((mapped - dst)**2).sum(axis=1)).mean()
        self.transcript_affine_by_core[core_id] = A

        logger.info("Fitted transcript affine for %s from GeoJSON (matches=%d, rms=%.2f px)", core_id, len(src), rms)


    # -----------------------
    # Robust COMET channel loader (handles zarr versions)
    # -----------------------
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
                # prefer explicit "0" level
                if "0" not in root:
                    # sometimes the array is at root itself
                    # try to detect array-like root
                    if hasattr(root, "shape") and getattr(root, "ndim", None) is not None:
                        level0 = root
                    else:
                        raise RuntimeError(f"OME-Zarr {path} missing level 0")
                else:
                    level0 = root["0"]

                # Determine whether level0 is an array or a group containing a full-res array
                arr5d = None
                # Try to detect zarr Array class robustly across zarr versions
                zarr_array_type = getattr(zarr, "Array", None)
                if zarr_array_type is None:
                    # fallback to core.Array if present
                    core_mod = getattr(zarr, "core", zarr)
                    zarr_array_type = getattr(core_mod, "Array", None)

                # If level0 is an array-like object (zarr array or numpy-like), try to read it
                if zarr_array_type is not None and isinstance(level0, zarr_array_type):
                    arr5d = level0[:]  # (T, C, Z, Y, X)
                else:
                    # If level0 is a group-like mapping, try to find a nested "0" array or a single array inside
                    if hasattr(level0, "keys") and "0" in level0:
                        fullres = level0["0"]
                        arr5d = fullres[:]  # (T, C, Z, Y, X)
                    else:
                        # If level0 exposes shape/ndim, try to read it
                        try:
                            arr5d = np.asarray(level0)
                        except Exception:
                            raise RuntimeError(f"Unable to interpret OME-Zarr structure at {path}")

                # Validate arr5d shape
                if arr5d is None:
                    raise RuntimeError(f"Failed to read OME-Zarr array for {path}")

                # Expect arr5d to be 5D: (T, C, Z, Y, X) or similar
                if arr5d.ndim == 5:
                    arr2d = arr5d[0, channel_index, 0, :, :]  # (Y, X)
                elif arr5d.ndim == 4:
                    # maybe (C, Y, X) or (T, C, Y, X)
                    if arr5d.shape[0] <= 8 and arr5d.shape[0] > arr5d.shape[-1]:
                        # treat first axis as channels
                        arr2d = arr5d[channel_index, :, :]
                    else:
                        # fallback: take first timepoint, channel index
                        arr2d = arr5d[0, channel_index, :, :]
                elif arr5d.ndim == 3:
                    # (C, Y, X) or (Z, Y, X)
                    if arr5d.shape[0] > 1 and arr5d.shape[0] <= 64:
                        arr2d = arr5d[channel_index, :, :]
                    else:
                        arr2d = arr5d
                else:
                    # fallback: try to coerce to 2D
                    arr2d = np.squeeze(arr5d)
                    if arr2d.ndim != 2:
                        raise RuntimeError(f"Unexpected array shape from OME-Zarr: {arr5d.shape}")

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

    # ---------------------------------------------------------------------------
    # Rasterise GeoJSON -> label mask and compute per-cell COMET stats
    # ---------------------------------------------------------------------------
    def load_geojson_mask(self, core_id: str, geojson_path: Optional[Path] = None, overwrite: bool = False):
        """
        Rasterize GeoJSON polygons (assumed COMET pixel coords) into a label mask for core_id.
        Stores mask in self.cell_mask_by_core[core_id] and mapping label->cell_id.
        Returns (mask, label_to_cell_id).
        """
        if core_id in self.cell_mask_by_core and not overwrite:
            return self.cell_mask_by_core[core_id], self.cell_label_to_id_by_core.get(core_id, {})

        if self.manifest is None:
            raise RuntimeError("Manifest not loaded; call load() first.")

        core = self.manifest.cores.get(core_id)
        if core is None:
            raise FileNotFoundError(f"Core {core_id} not found in manifest")

        # find geojson path
        if geojson_path is None:
            if core.xenium_geojson_comet:
                candidate = Path(core.xenium_geojson_comet)
                if not candidate.is_absolute():
                    candidate = self.folder / candidate
                if candidate.exists():
                    geojson_path = candidate
            if geojson_path is None and core.xenium_folder:
                candidate = core.xenium_folder / "cell_boundaries_comet_space.geojson"
                if candidate.exists():
                    geojson_path = candidate
        if geojson_path is None or not Path(geojson_path).exists():
            raise FileNotFoundError(f"GeoJSON not found for core {core_id}")

        # load COMET channel to get shape
        img = self.get_comet_channel(core_id, channel_index=0)
        if img is None:
            raise RuntimeError(f"Could not load COMET image for core {core_id} to determine mask size")
        h = int(img.shape[0])
        w = int(img.shape[1])

        # read geojson
        with open(geojson_path, "r") as f:
            gj = json.load(f)

        # Prepare blank label image (PIL uses (width, height))
        label_img = Image.new("I", (w, h), 0)
        draw = ImageDraw.Draw(label_img)

        label_to_id: Dict[int, str] = {}
        label = 1

        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            cid = str(props.get("cell_id") or props.get("name") or f"cell_{label}")
            geom = feat.get("geometry")
            if geom is None:
                continue
            poly = shape(geom)
            if not poly.is_valid:
                poly = make_valid(poly)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
            if not poly.is_valid or poly.area <= 0:
                continue

            # polygon coordinates are (x,y) in COMET pixels; PIL expects sequence of (x,y)
            try:
                coords = [(float(x), float(y)) for x, y in np.array(poly.exterior.coords)]
                # draw polygon with integer label
                draw.polygon(coords, outline=label, fill=label)
                label_to_id[label] = cid
                label += 1
            except Exception:
                continue

        mask = np.asarray(label_img, dtype=np.int32)
        self.cell_mask_by_core[core_id] = mask
        self.cell_label_to_id_by_core[core_id] = label_to_id

        # Optionally compute per-cell COMET stats (mean, median, sum) for channel 0
        try:
            channel = np.asarray(img, dtype=float)
            labels = mask
            unique_labels = np.unique(labels)
            stats = []
            for lab in unique_labels:
                if lab == 0:
                    continue
                mask_bool = labels == lab
                vals = channel[mask_bool]
                if vals.size == 0:
                    continue
                stats.append({
                    "label": int(lab),
                    "cell_id": label_to_id.get(int(lab), ""),
                    "area_px": int(mask_bool.sum()),
                    "mean": float(np.nanmean(vals)),
                    "median": float(np.nanmedian(vals)),
                    "sum": float(np.nansum(vals)),
                })
            df_stats = pd.DataFrame(stats)
            self.comet_cell_stats[core_id] = df_stats
        except Exception:
            self.comet_cell_stats[core_id] = None

        return mask, label_to_id

    def __repr__(self) -> str:
        loaded = "loaded" if self.manifest else "not loaded"
        return (
            f"DatasetLoader(folder={self.folder}, status={loaded}, "
            f"cores={len(self.manifest.cores) if self.manifest else 0}, "
            f"genes={len(self.genes)}, proteins={len(self.proteins)})"
        )
    

    ## Load user-imported segmentations
    def load_custom_geojson(
        self,
        core_id: str,
        method_name: str,
        geojson_path: Union[str, Path],
        coordinate_space: str = "COMET",
    ):
        """
        Load user-supplied segmentation polygons.
        """

        from shapely.geometry import shape
        from shapely.validation import make_valid

        geojson_path = Path(geojson_path)

        with open(geojson_path, "r") as f:
            gj = json.load(f)

        shapes_napari = []

        for feat in gj.get("features", []):

            geom = feat.get("geometry")

            if geom is None:
                continue

            try:

                poly = shape(geom)

                if not poly.is_valid:
                    poly = make_valid(poly)

                if poly.geom_type == "MultiPolygon":
                    poly = max(
                        poly.geoms,
                        key=lambda p: p.area
                    )

                coords = np.asarray(
                    poly.exterior.coords,
                    dtype=float
                )

                # Napari uses y,x
                shapes_napari.append(
                    coords[:, ::-1]
                )

            except Exception:
                continue

## DEBUG
        n_polygons = len(shapes_napari)

        n_vertices = sum(
            len(p)
            for p in shapes_napari
        )

        print(
            f"Imported {n_polygons:,} polygons"
        )

        print(
            f"Imported {n_vertices:,} vertices"
        )

        print(
            f"Average vertices/cell: "
            f"{n_vertices / max(n_polygons, 1):.1f}"
        )
## ---



        self.custom_segmentations.setdefault(
            core_id,
            {}
        )[method_name] = {
            "space": coordinate_space,
            "path": str(geojson_path),
            "shapes": shapes_napari,
        }

        logger.info(
            "Loaded %d polygons for %s (%s)",
            len(shapes_napari),
            method_name,
            core_id,
        )

        return shapes_napari