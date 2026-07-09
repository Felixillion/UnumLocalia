"""
Example 1: Load a UnumLocalia dataset and launch the interactive viewer.

Usage
-----
    conda activate unumlocalia
    python examples/01_load_and_view.py /path/to/dataset
"""

import sys
from pathlib import Path


def main(dataset_folder: str) -> None:
    from unumlocalia.io import DatasetLoader
    from unumlocalia.viewer import SpatialViewer
    from unumlocalia.utils import shapes_to_napari

    # ---- Load dataset -------------------------------------------------------
    print(f"Loading dataset from: {dataset_folder}")
    loader = DatasetLoader(dataset_folder)
    loader.load()

    print(loader.manifest.summary())
    print(f"\nGenes detected   : {len(loader.genes)}")
    print(f"Proteins detected: {len(loader.proteins)}")

    # ---- Launch viewer ------------------------------------------------------
    sv = SpatialViewer(title="UnumLocalia — Example 1")

    # H&E layer
    if loader.he_array is not None:
        sv.add_he_layer(loader.he_array, opacity=1.0)
        print("Added H&E layer.")

    # COMET layers
    for marker, arr in loader.comet_arrays.items():
        sv.add_comet_layer(marker, arr, colormap="green", opacity=0.8)
    print(f"Added {len(loader.comet_arrays)} COMET channels.")

    # Cell boundaries
    if loader.cell_boundaries_df is not None:
        shapes = shapes_to_napari(loader.cell_boundaries_df)
        sv.add_boundary_layer(shapes, name="Cell boundaries", color="white")
        print(f"Added {len(shapes)} cell boundary polygons.")

    # Nucleus boundaries
    if loader.nucleus_boundaries_df is not None:
        shapes = shapes_to_napari(loader.nucleus_boundaries_df)
        sv.add_boundary_layer(shapes, name="Nucleus boundaries", color="cyan")

    # Example: show transcripts for a specific gene
    if loader.transcripts_df is not None and len(loader.genes) > 0:
        example_gene = loader.genes[0]
        tx = loader.transcripts_df
        gene_tx = tx[tx["feature_name"] == example_gene]
        if len(gene_tx) > 0:
            coords = gene_tx[["x_location", "y_location"]].to_numpy()
            sv.add_transcript_layer(example_gene, coords, color="yellow", size=4)
            print(f"Added transcripts for '{example_gene}': {len(coords)} points.")

    sv.reset_view()
    print("\nViewer ready. Close the napari window to exit.")
    sv.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_load_and_view.py /path/to/dataset")
        sys.exit(1)
    main(sys.argv[1])
