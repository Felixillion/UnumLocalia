# UnumLocalia Documentation

Welcome to the documentation for **UnumLocalia**, a multimodal spatial biology toolkit.

UnumLocalia is designed to accompany publicly available multimodal spatial datasets and provide a simple, reproducible environment for visualisation, exploration, segmentation benchmarking, and basic single-cell analysis.

## Core Principles
* **Never modifies raw data**: All data loaded is read-only and never modified. 
* **Lazy loading**: Datasets are large, so lazy loading is used to only load memory when required (especially for OME-TIFFs and H&E images).
* **Modular Codebase**: Every functionality is modular and distinct across `unumlocalia/io.py`, `unumlocalia/viewer.py`, `unumlocalia/segmentation.py`, `unumlocalia/analysis.py`, and `unumlocalia/benchmark.py`.

## Getting Started
See the [Installation Guide](installation.md) to set up UnumLocalia locally, and read about the [Data Formats](data_format.md) to understand how to structure your spatial datasets.
