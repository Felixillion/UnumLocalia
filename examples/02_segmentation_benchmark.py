"""
Example 2: Load a custom segmentation, build a new AnnData, and compare
           it to the original Xenium segmentation.

Usage
-----
    conda activate unumlocalia
    python examples/02_segmentation_benchmark.py \\
        /path/to/dataset \\
        /path/to/my_segmentation.tif
"""

import sys
from pathlib import Path


def main(dataset_folder: str, segmentation_path: str) -> None:
    from unumlocalia.io import DatasetLoader
    from unumlocalia.segmentation import (
        load_segmentation,
        measure_comet_intensities,
        assign_xenium_transcripts,
        build_anndata,
        compare_segmentations,
    )
    from unumlocalia.utils import export_dataframe

    # ---- Load dataset -------------------------------------------------------
    print(f"Loading dataset from: {dataset_folder}")
    loader = DatasetLoader(dataset_folder)
    loader.load()

    # ---- Load user segmentation ---------------------------------------------
    print(f"\nLoading segmentation from: {segmentation_path}")
    labels_user = load_segmentation(segmentation_path)
    print(f"Label mask shape: {labels_user.shape}  "
          f"Unique labels: {len(set(labels_user.flat)) - 1} cells")

    # ---- Measure COMET intensities in user segmentation -------------------
    print("\nMeasuring COMET intensities…")
    comet_intensities = measure_comet_intensities(labels_user, loader.comet_arrays)
    print(comet_intensities.head())

    # ---- Assign transcripts -----------------------------------------------
    print("\nAssigning Xenium transcripts to user segmentation…")
    tx_assigned = assign_xenium_transcripts(labels_user, loader.transcripts_df)
    n_assigned = (tx_assigned["cell_id"] > 0).sum()
    print(f"Transcripts assigned: {n_assigned} / {len(tx_assigned)}")

    # ---- Build fresh AnnData -----------------------------------------------
    print("\nBuilding AnnData from user segmentation…")
    adata_user = build_anndata(
        labels_user,
        comet_intensities=comet_intensities,
        transcripts_df=tx_assigned,
        cells_metadata=loader.cells_df,
    )
    print(adata_user)

    # ---- Export AnnData ----------------------------------------------------
    out_h5ad = Path(dataset_folder) / "user_segmentation.h5ad"
    adata_user.write_h5ad(out_h5ad)
    print(f"\nAnnData saved to: {out_h5ad}")

    # ---- Compare segmentations (requires original label mask) --------------
    # NOTE: if the original Xenium label mask is available as a TIFF:
    #   labels_orig = load_segmentation("/path/to/xenium_labels.tif")
    #   metrics = compare_segmentations(labels_orig, labels_user, ...)
    #   print(metrics.summary_df)

    # ---- Export COMET intensities ------------------------------------------
    out_csv = Path(dataset_folder) / "user_comet_intensities.csv"
    export_dataframe(comet_intensities, out_csv)
    print(f"COMET intensities exported to: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python 02_segmentation_benchmark.py "
            "/path/to/dataset /path/to/segmentation.tif"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
