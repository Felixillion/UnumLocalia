# SpatialBench Documentation

Welcome to the documentation for **SpatialBench**, a multimodal spatial biology toolkit.

SpatialBench is designed to accompany publicly available multimodal spatial datasets and provide a simple, reproducible environment for visualisation, exploration, segmentation benchmarking, and basic single-cell analysis.

## Core Principles
* **Never modifies raw data**: All data loaded is read-only and never modified. 
* **Lazy loading**: Datasets are large, so lazy loading is used to only load memory when required (especially for OME-TIFFs and H&E images).
* **Modular Codebase**: Every functionality is modular and distinct across `spatialbench/io.py`, `spatialbench/viewer.py`, `spatialbench/segmentation.py`, `spatialbench/analysis.py`, and `spatialbench/benchmark.py`.

## Getting Started
See the [Installation Guide](installation.md) to set up SpatialBench locally, and read about the [Data Formats](data_format.md) to understand how to structure your spatial datasets.
