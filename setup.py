"""
UnumLocalia — setup.py
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="unumlocalia",
    version="1.1.1",
    author="UnumLocalia Contributors",
    description=(
        "A multimodal spatial biology toolkit for visualisation, "
        "segmentation benchmarking, and single-cell analysis."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Felixillion/UnumLocalia",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "napari>=0.4.19",
        "qtpy",
        "numpy>=1.23",
        "pandas>=1.5",
        "pyarrow>=12.0",
        "tifffile>=2023.1",
        "anndata>=0.9",
        "scanpy>=1.9",
        "scikit-learn>=1.2",
        "scikit-image>=0.20",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "plotly>=5.14",
        "opencv-python-headless>=4.7",
        "shapely>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "unumlocalia=unumlocalia.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "spatial transcriptomics", "spatial proteomics", "xenium",
        "napari", "single-cell", "multimodal", "bioinformatics",
    ],
)
