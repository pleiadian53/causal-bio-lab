"""Data loaders for perturbation datasets.

This module provides utilities to download and load standard perturbation
datasets used in causal inference for computational biology.

Supported datasets:
- Norman et al. 2019: Combinatorial CRISPR perturbations in K562 cells
- Replogle et al. 2022: Genome-wide Perturb-seq in K562/RPE1 cells

Usage:
    from causalbiolab.data.loaders import load_norman, load_replogle
    
    adata = load_norman()  # Downloads if not present
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scanpy as sc


def download_norman(
    output_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download Norman et al. 2019 combinatorial perturbation dataset.
    
    This dataset contains single and double gene knockouts in K562 cells,
    making it ideal for studying combinatorial perturbation effects.
    
    Source: https://github.com/snap-stanford/GEARS (preprocessed version)
    
    Args:
        output_dir: Directory to save the data. If None, uses default paths.
        force: If True, re-download even if file exists.
        
    Returns:
        Path to the downloaded h5ad file
    """
    import urllib.request
    import gzip
    import shutil
    
    from causalbiolab.data.paths import get_data_paths
    
    # Determine output path
    if output_dir is None:
        paths = get_data_paths()
        paths.ensure_dirs("norman_k562", "perturbation")
        output_path = paths.perturbation_processed("norman_k562", "perturb_processed.h5ad")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "perturb_processed.h5ad"
    
    if output_path.exists() and not force:
        print(f"Dataset already exists at {output_path}")
        return output_path
    
    # URL for preprocessed Norman dataset (from GEARS)
    # This is the version used in GEARS paper, preprocessed and ready to use
    url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
    
    print(f"Downloading Norman dataset...")
    print(f"  Source: Harvard Dataverse (GEARS preprocessed)")
    print(f"  Target: {output_path}")
    
    # Download
    temp_path = output_path.with_suffix(".h5ad.tmp")
    try:
        urllib.request.urlretrieve(url, temp_path)
        shutil.move(temp_path, output_path)
        print(f"  ✓ Downloaded successfully ({output_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download Norman dataset: {e}")
    
    return output_path


def load_norman(
    processed: bool = True,
    download: bool = True,
) -> "sc.AnnData":
    """Load Norman et al. 2019 combinatorial perturbation dataset.
    
    Args:
        processed: If True, load preprocessed version. If False, load raw.
        download: If True, download if not present.
        
    Returns:
        AnnData object with perturbation annotations
        
    Key annotations in adata.obs:
        - condition: Perturbation condition (e.g., "ctrl", "GENE1+GENE2")
        - condition_name: Human-readable condition name
        - cell_type: Cell type (K562)
        - control: Boolean indicating control cells
    """
    import scanpy as sc
    from causalbiolab.data.paths import get_data_paths
    
    paths = get_data_paths()
    
    # Check for processed file
    processed_path = paths.perturbation_processed("norman_k562", "perturb_processed.h5ad")
    
    if not processed_path.exists():
        if download:
            print("Norman dataset not found locally. Downloading...")
            download_norman()
        else:
            raise FileNotFoundError(
                f"Norman dataset not found at {processed_path}. "
                "Set download=True to download automatically."
            )
    
    print(f"Loading Norman dataset from {processed_path}...")
    adata = sc.read_h5ad(processed_path)
    
    # Standardize perturbation annotations
    adata = _standardize_norman_annotations(adata)
    
    print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    _print_perturbation_summary(adata)
    
    return adata


def _standardize_norman_annotations(adata: "sc.AnnData") -> "sc.AnnData":
    """Standardize Norman dataset annotations for our pipeline.
    
    Creates consistent column names across datasets:
    - perturbation: The perturbation label
    - is_control: Boolean for control cells
    - n_perturbations: Number of genes perturbed
    """
    import pandas as pd
    
    # The GEARS-preprocessed Norman dataset uses 'condition' column
    if "condition" in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs["condition"].copy()
    elif "gene" in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs["gene"].copy()
    
    # Identify control cells
    if "perturbation" in adata.obs.columns:
        ctrl_patterns = ["ctrl", "control", "non-targeting", "nt"]
        adata.obs["is_control"] = adata.obs["perturbation"].str.lower().isin(ctrl_patterns)
        
        # Count perturbations (for combinatorial)
        def count_perturb(x):
            if pd.isna(x) or x.lower() in ctrl_patterns:
                return 0
            # Norman uses "+" for combinations
            if "+" in str(x):
                return len(str(x).split("+"))
            return 1
        
        adata.obs["n_perturbations"] = adata.obs["perturbation"].apply(count_perturb)
    
    # Ensure library size is computed
    if "library_size" not in adata.obs.columns:
        import numpy as np
        adata.obs["library_size"] = np.array(adata.X.sum(axis=1)).ravel()
    
    return adata


def _print_perturbation_summary(adata: "sc.AnnData") -> None:
    """Print summary of perturbation annotations."""
    if "is_control" not in adata.obs.columns:
        return
    
    n_control = adata.obs["is_control"].sum()
    n_perturbed = (~adata.obs["is_control"]).sum()
    n_unique = adata.obs["perturbation"].nunique()
    
    print(f"  Perturbation summary:")
    print(f"    Control cells: {n_control}")
    print(f"    Perturbed cells: {n_perturbed}")
    print(f"    Unique perturbations: {n_unique}")
    
    if "n_perturbations" in adata.obs.columns:
        max_perturb = adata.obs["n_perturbations"].max()
        if max_perturb > 1:
            n_combo = (adata.obs["n_perturbations"] > 1).sum()
            print(f"    Combinatorial perturbations: {n_combo} cells")


def list_perturbations(
    adata: "sc.AnnData",
    min_cells: int = 10,
    perturbation_col: str = "perturbation",
) -> "pd.DataFrame":
    """List all perturbations with cell counts.
    
    Args:
        adata: AnnData with perturbation annotations
        min_cells: Minimum cells to include a perturbation
        perturbation_col: Column containing perturbation labels
        
    Returns:
        DataFrame with perturbation names and cell counts
    """
    import pandas as pd
    
    counts = adata.obs[perturbation_col].value_counts()
    counts = counts[counts >= min_cells]
    
    df = pd.DataFrame({
        "perturbation": counts.index,
        "n_cells": counts.values,
    })
    
    # Add n_genes column if available
    if "n_perturbations" in adata.obs.columns:
        n_genes = adata.obs.groupby(perturbation_col)["n_perturbations"].first()
        df["n_genes"] = df["perturbation"].map(n_genes)
    
    return df.reset_index(drop=True)


def get_perturbation_pairs(
    adata: "sc.AnnData",
    target_perturbation: str,
    control_value: str = "ctrl",
    perturbation_col: str = "perturbation",
) -> "sc.AnnData":
    """Extract control and target perturbation cells for comparison.
    
    Args:
        adata: Full AnnData object
        target_perturbation: The perturbation to study
        control_value: Value indicating control cells
        perturbation_col: Column with perturbation labels
        
    Returns:
        AnnData subset with only control and target cells
    """
    mask = (
        (adata.obs[perturbation_col] == target_perturbation) |
        (adata.obs[perturbation_col].str.lower() == control_value.lower())
    )
    
    adata_subset = adata[mask].copy()
    
    n_ctrl = (adata_subset.obs[perturbation_col].str.lower() == control_value.lower()).sum()
    n_treat = (adata_subset.obs[perturbation_col] == target_perturbation).sum()
    
    print(f"Extracted: {n_ctrl} control, {n_treat} {target_perturbation}")
    
    return adata_subset
