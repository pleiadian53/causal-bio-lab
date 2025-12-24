"""Data path management for causal-bio-lab.

This module provides standardized paths for causal inference datasets, ensuring
consistency across the codebase. Adapted from genai-lab's path management.

Directory Structure:
    data/                           # Root data directory (not in git)
    ├── perturbation/               # Perturbation/intervention datasets
    │   ├── replogle_k562/          # Replogle et al. K562 Perturb-seq
    │   │   ├── raw/                # Raw downloaded files
    │   │   └── processed/          # Preprocessed h5ad files
    │   ├── norman_k562/            # Norman et al. combinatorial perturbations
    │   └── dixit_k562/             # Dixit et al. Perturb-seq
    ├── observational/              # Observational (non-interventional) data
    │   ├── gtex/                   # GTEx for treatment effect estimation
    │   ├── tcga/                   # TCGA for drug response
    │   └── gdsc/                   # Genomics of Drug Sensitivity in Cancer
    ├── synthetic/                  # Synthetic datasets for benchmarking
    │   ├── sergio/                 # SERGIO simulator outputs
    │   └── causal_bench/           # CausalBench datasets
    └── models/                     # Trained model checkpoints

Environment Variables:
    CAUSALBIOLAB_DATA_ROOT: Override default data root directory
    CAUSALBIOLAB_PERTURBATION_ROOT: Override perturbation data directory
    CAUSALBIOLAB_OBSERVATIONAL_ROOT: Override observational data directory

Usage:
    from causalbiolab.data.paths import get_data_paths, DataPaths
    
    paths = get_data_paths()
    replogle_processed = paths.perturbation_processed("replogle_k562")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve()
    
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


@dataclass
class DataPaths:
    """Centralized data path management for causal inference.
    
    Attributes:
        root: Root data directory
        perturbation_root: Perturbation/intervention data directory
        observational_root: Observational data directory
        synthetic_root: Synthetic benchmark data directory
        models_root: Model checkpoints directory
    """
    root: Path
    perturbation_root: Path = field(init=False)
    observational_root: Path = field(init=False)
    synthetic_root: Path = field(init=False)
    models_root: Path = field(init=False)
    
    def __post_init__(self):
        self.root = Path(self.root)
        self.perturbation_root = Path(
            os.getenv("CAUSALBIOLAB_PERTURBATION_ROOT", self.root / "perturbation")
        )
        self.observational_root = Path(
            os.getenv("CAUSALBIOLAB_OBSERVATIONAL_ROOT", self.root / "observational")
        )
        self.synthetic_root = self.root / "synthetic"
        self.models_root = self.root / "models"
    
    # =========================================================================
    # Perturbation data paths (Perturb-seq, CRISPR screens)
    # =========================================================================
    
    def perturbation_dataset_dir(self, dataset: str) -> Path:
        """Get directory for a perturbation dataset.
        
        Args:
            dataset: Dataset name (e.g., "replogle_k562", "norman_k562")
        """
        return self.perturbation_root / dataset
    
    def perturbation_raw(self, dataset: str) -> Path:
        """Get raw data directory for a perturbation dataset."""
        return self.perturbation_dataset_dir(dataset) / "raw"
    
    def perturbation_processed(self, dataset: str, filename: str = "counts.h5ad") -> Path:
        """Get processed file path for a perturbation dataset.
        
        Args:
            dataset: Dataset name
            filename: Processed file name (default: counts.h5ad)
        """
        return self.perturbation_dataset_dir(dataset) / "processed" / filename
    
    # =========================================================================
    # Observational data paths (GTEx, TCGA, drug response)
    # =========================================================================
    
    def observational_dataset_dir(self, dataset: str) -> Path:
        """Get directory for an observational dataset.
        
        Args:
            dataset: Dataset name (e.g., "gtex", "tcga", "gdsc")
        """
        return self.observational_root / dataset
    
    def observational_raw(self, dataset: str) -> Path:
        """Get raw data directory for an observational dataset."""
        return self.observational_dataset_dir(dataset) / "raw"
    
    def observational_processed(self, dataset: str, filename: str = "counts.h5ad") -> Path:
        """Get processed file path for an observational dataset.
        
        Args:
            dataset: Dataset name
            filename: Processed file name (default: counts.h5ad)
        """
        return self.observational_dataset_dir(dataset) / "processed" / filename
    
    # =========================================================================
    # Synthetic data paths (SERGIO, CausalBench)
    # =========================================================================
    
    def synthetic_dataset_dir(self, dataset: str) -> Path:
        """Get directory for a synthetic dataset.
        
        Args:
            dataset: Dataset name (e.g., "sergio", "causal_bench")
        """
        return self.synthetic_root / dataset
    
    def synthetic_processed(self, dataset: str, filename: str = "data.h5ad") -> Path:
        """Get processed file path for a synthetic dataset."""
        return self.synthetic_dataset_dir(dataset) / filename
    
    # =========================================================================
    # Model paths
    # =========================================================================
    
    def model_dir(self, model_name: str) -> Path:
        """Get directory for a trained model."""
        return self.models_root / model_name
    
    def model_checkpoint(self, model_name: str, checkpoint: str = "best.pt") -> Path:
        """Get checkpoint path for a trained model."""
        return self.model_dir(model_name) / checkpoint
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def ensure_dirs(
        self, 
        dataset: str, 
        data_type: Literal["perturbation", "observational", "synthetic"] = "perturbation"
    ):
        """Create directory structure for a dataset.
        
        Args:
            dataset: Dataset name
            data_type: "perturbation", "observational", or "synthetic"
        """
        if data_type == "perturbation":
            self.perturbation_raw(dataset).mkdir(parents=True, exist_ok=True)
            self.perturbation_processed(dataset).parent.mkdir(parents=True, exist_ok=True)
        elif data_type == "observational":
            self.observational_raw(dataset).mkdir(parents=True, exist_ok=True)
            self.observational_processed(dataset).parent.mkdir(parents=True, exist_ok=True)
        else:
            self.synthetic_dataset_dir(dataset).mkdir(parents=True, exist_ok=True)
    
    def list_datasets(
        self, 
        data_type: Literal["perturbation", "observational", "synthetic"] = "perturbation"
    ) -> list[str]:
        """List available datasets.
        
        Args:
            data_type: "perturbation", "observational", or "synthetic"
            
        Returns:
            List of dataset names that have processed files
        """
        if data_type == "perturbation":
            root = self.perturbation_root
        elif data_type == "observational":
            root = self.observational_root
        else:
            root = self.synthetic_root
            
        if not root.exists():
            return []
        
        datasets = []
        for d in root.iterdir():
            if d.is_dir():
                # Check for processed files
                if (d / "processed").exists() or any(d.glob("*.h5ad")):
                    datasets.append(d.name)
        return sorted(datasets)
    
    def __repr__(self) -> str:
        return (
            f"DataPaths(\n"
            f"  root={self.root},\n"
            f"  perturbation_root={self.perturbation_root},\n"
            f"  observational_root={self.observational_root},\n"
            f"  synthetic_root={self.synthetic_root},\n"
            f"  models_root={self.models_root}\n"
            f")"
        )


# Global singleton
_data_paths: DataPaths | None = None


def get_data_paths(data_root: str | Path | None = None) -> DataPaths:
    """Get the global DataPaths instance.
    
    Args:
        data_root: Override data root directory. If None, uses:
            1. CAUSALBIOLAB_DATA_ROOT environment variable
            2. <project_root>/data/
            
    Returns:
        DataPaths instance
    """
    global _data_paths
    
    if _data_paths is None or data_root is not None:
        if data_root is None:
            data_root = os.getenv("CAUSALBIOLAB_DATA_ROOT")
        
        if data_root is None:
            project_root = _find_project_root()
            data_root = project_root / "data"
        
        _data_paths = DataPaths(root=Path(data_root))
    
    return _data_paths


def reset_data_paths():
    """Reset the global DataPaths instance (useful for testing)."""
    global _data_paths
    _data_paths = None


# ============================================================================
# Convenience functions for common perturbation datasets
# ============================================================================

def perturb_seq_paths(dataset: str = "replogle_k562") -> dict[str, Path]:
    """Get paths for a Perturb-seq dataset.
    
    Args:
        dataset: Dataset name (default: replogle_k562)
    
    Returns:
        Dict with 'raw', 'processed', and 'counts' paths
    """
    paths = get_data_paths()
    return {
        "raw": paths.perturbation_raw(dataset),
        "processed": paths.perturbation_processed(dataset).parent,
        "counts": paths.perturbation_processed(dataset, "counts.h5ad"),
    }


def replogle_paths() -> dict[str, Path]:
    """Get paths for Replogle et al. K562 Perturb-seq dataset.
    
    This is a large-scale CRISPR screen with ~2.5M cells.
    """
    return perturb_seq_paths("replogle_k562")


def norman_paths() -> dict[str, Path]:
    """Get paths for Norman et al. combinatorial perturbation dataset.
    
    This dataset includes single and double gene knockouts.
    """
    return perturb_seq_paths("norman_k562")


# ============================================================================
# CLI for setup
# ============================================================================

def setup_data_directories():
    """Create the standard data directory structure."""
    paths = get_data_paths()
    
    # Create main directories
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.perturbation_root.mkdir(parents=True, exist_ok=True)
    paths.observational_root.mkdir(parents=True, exist_ok=True)
    paths.synthetic_root.mkdir(parents=True, exist_ok=True)
    paths.models_root.mkdir(parents=True, exist_ok=True)
    
    # Create perturbation dataset directories
    for dataset in ["replogle_k562", "norman_k562", "dixit_k562"]:
        paths.ensure_dirs(dataset, "perturbation")
    
    # Create observational dataset directories
    for dataset in ["gtex", "tcga", "gdsc"]:
        paths.ensure_dirs(dataset, "observational")
    
    # Create synthetic dataset directories
    for dataset in ["sergio", "causal_bench"]:
        paths.ensure_dirs(dataset, "synthetic")
    
    # Create .gitignore in data directory
    gitignore = paths.root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Ignore all data files\n*\n!.gitignore\n!README.md\n")
    
    # Create README
    readme = paths.root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Data Directory\n\n"
            "This directory contains datasets for causal inference in computational biology.\n\n"
            "## Structure\n\n"
            "```\n"
            "data/\n"
            "├── perturbation/        # Perturb-seq, CRISPR screens\n"
            "│   ├── replogle_k562/   # Replogle et al. genome-wide screen\n"
            "│   ├── norman_k562/     # Norman et al. combinatorial perturbations\n"
            "│   └── dixit_k562/      # Dixit et al. Perturb-seq\n"
            "├── observational/       # Non-interventional data\n"
            "│   ├── gtex/            # GTEx tissue expression\n"
            "│   ├── tcga/            # TCGA cancer data\n"
            "│   └── gdsc/            # Drug sensitivity data\n"
            "├── synthetic/           # Benchmark datasets\n"
            "│   ├── sergio/          # SERGIO simulator\n"
            "│   └── causal_bench/    # CausalBench\n"
            "└── models/              # Trained model checkpoints\n"
            "```\n\n"
            "## Usage\n\n"
            "```python\n"
            "from causalbiolab.data.paths import get_data_paths\n\n"
            "paths = get_data_paths()\n"
            "replogle = paths.perturbation_processed('replogle_k562')\n"
            "```\n"
        )
    
    print(f"Created data directory structure at: {paths.root}")
    print(paths)


if __name__ == "__main__":
    setup_data_directories()
