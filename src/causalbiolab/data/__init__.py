"""Data loading, preprocessing, and path management for causal-bio-lab.

This module provides:
- Standardized data path management (paths.py)
- scRNA-seq preprocessing with perturbation support (sc_preprocess.py)
- Bulk RNA-seq preprocessing for treatment effect studies (bulk_preprocess.py)
"""

from causalbiolab.data.paths import (
    DataPaths,
    get_data_paths,
    reset_data_paths,
    setup_data_directories,
    # Convenience functions for common datasets
    perturb_seq_paths,
    replogle_paths,
    norman_paths,
)

from causalbiolab.data.loaders import (
    download_norman,
    load_norman,
    list_perturbations,
    get_perturbation_pairs,
)

__all__ = [
    # Core path management
    "DataPaths",
    "get_data_paths",
    "reset_data_paths",
    "setup_data_directories",
    # Dataset-specific paths
    "perturb_seq_paths",
    "replogle_paths",
    "norman_paths",
    # Data loaders
    "download_norman",
    "load_norman",
    "list_perturbations",
    "get_perturbation_pairs",
]
