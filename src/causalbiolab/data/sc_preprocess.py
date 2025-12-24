"""scRNA-seq preprocessing for causal inference with perturbation support.

This script preprocesses scRNA-seq data, particularly Perturb-seq datasets,
while preserving raw counts and extracting perturbation metadata.

Key features:
- Preserves raw counts for NB/ZINB models
- Extracts perturbation labels (gene knockouts, treatments)
- Computes library size for normalization in models
- Supports control vs. perturbed cell identification

Usage:
    # Download and preprocess Replogle K562 dataset
    python -m causalbiolab.data.sc_preprocess --dataset replogle_k562
    
    # From local h5ad with perturbation annotations
    python -m causalbiolab.data.sc_preprocess -i perturb_seq.h5ad -o processed.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import scanpy as sc


def load_from_h5ad(path: str | Path) -> "sc.AnnData":
    """Load AnnData from h5ad file.
    
    Args:
        path: Path to h5ad file
    """
    import scanpy as sc
    
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    return adata


def load_10x_mtx(path: str | Path) -> "sc.AnnData":
    """Load 10x Genomics MTX format (filtered_feature_bc_matrix folder).
    
    Args:
        path: Path to the folder containing matrix.mtx, barcodes.tsv, genes.tsv
    """
    import scanpy as sc
    
    adata = sc.read_10x_mtx(
        path,
        var_names="gene_symbols",
        cache=True,
    )
    adata.var_names_make_unique()
    return adata


def compute_qc_metrics(
    adata: "sc.AnnData",
    mito_prefix: str = "MT-",
) -> "sc.AnnData":
    """Compute QC metrics including library size and mitochondrial content.
    
    Args:
        adata: AnnData object with raw counts
        mito_prefix: Prefix for mitochondrial genes ("MT-" for human, "mt-" for mouse)
    """
    import scanpy as sc
    
    # Library size (total counts per cell) - critical for NB models
    adata.obs["library_size"] = np.array(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes"] = np.array((adata.X > 0).sum(axis=1)).ravel()
    
    # Mitochondrial content
    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    
    return adata


def filter_cells_and_genes(
    adata: "sc.AnnData",
    min_genes: int = 200,
    max_genes: int | None = None,
    min_cells: int = 3,
    max_mito_pct: float = 20.0,
) -> "sc.AnnData":
    """Filter low-quality cells and lowly-expressed genes.
    
    Args:
        adata: AnnData object with QC metrics computed
        min_genes: Minimum genes per cell
        max_genes: Maximum genes per cell (None = no upper limit)
        min_cells: Minimum cells per gene
        max_mito_pct: Maximum mitochondrial percentage
    """
    import scanpy as sc
    
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        adata = adata[adata.obs.n_genes < max_genes].copy()
    adata = adata[adata.obs.pct_counts_mt < max_mito_pct].copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Filtered: {n_cells_before} -> {adata.n_obs} cells, "
          f"{n_genes_before} -> {adata.n_vars} genes")
    
    return adata


def extract_perturbation_metadata(
    adata: "sc.AnnData",
    perturbation_col: str = "perturbation",
    control_value: str = "control",
) -> "sc.AnnData":
    """Extract and standardize perturbation metadata.
    
    This function identifies:
    - Control cells (no perturbation)
    - Perturbed cells (with gene knockout/knockdown)
    - Perturbation type (single gene, combination, etc.)
    
    Args:
        adata: AnnData object with perturbation annotations
        perturbation_col: Column name containing perturbation info
        control_value: Value indicating control cells
        
    Returns:
        AnnData with standardized perturbation columns:
        - is_control: Boolean indicating control cells
        - perturbation: Standardized perturbation label
        - n_perturbations: Number of genes perturbed (0 for control)
    """
    if perturbation_col not in adata.obs.columns:
        print(f"Warning: '{perturbation_col}' not found in obs. "
              "Skipping perturbation extraction.")
        return adata
    
    # Identify control cells
    adata.obs["is_control"] = (
        adata.obs[perturbation_col].str.lower() == control_value.lower()
    ) | (
        adata.obs[perturbation_col].isna()
    ) | (
        adata.obs[perturbation_col] == ""
    )
    
    # Count perturbations (for combinatorial screens)
    def count_perturbations(x):
        if pd.isna(x) or x == "" or x.lower() == control_value.lower():
            return 0
        # Handle common separators: +, _, ,
        for sep in ["+", "_", ","]:
            if sep in str(x):
                return len(str(x).split(sep))
        return 1
    
    adata.obs["n_perturbations"] = adata.obs[perturbation_col].apply(count_perturbations)
    
    # Summary statistics
    n_control = adata.obs["is_control"].sum()
    n_perturbed = (~adata.obs["is_control"]).sum()
    n_unique = adata.obs[perturbation_col].nunique()
    
    print(f"Perturbation summary:")
    print(f"  Control cells: {n_control}")
    print(f"  Perturbed cells: {n_perturbed}")
    print(f"  Unique perturbations: {n_unique}")
    
    # Check for combinatorial perturbations
    max_perturb = adata.obs["n_perturbations"].max()
    if max_perturb > 1:
        print(f"  Max perturbations per cell: {max_perturb} (combinatorial screen)")
    
    return adata


def add_treatment_covariates(
    adata: "sc.AnnData",
    dose_col: str | None = None,
    time_col: str | None = None,
    batch_col: str | None = None,
) -> "sc.AnnData":
    """Add treatment-related covariates for causal inference.
    
    Args:
        adata: AnnData object
        dose_col: Column containing dose information
        time_col: Column containing time point information
        batch_col: Column containing batch information (potential confounder)
        
    Returns:
        AnnData with standardized covariate columns
    """
    # Standardize dose if present
    if dose_col is not None and dose_col in adata.obs.columns:
        adata.obs["dose"] = pd.to_numeric(adata.obs[dose_col], errors="coerce")
        print(f"Dose range: [{adata.obs['dose'].min()}, {adata.obs['dose'].max()}]")
    
    # Standardize time if present
    if time_col is not None and time_col in adata.obs.columns:
        adata.obs["time_point"] = adata.obs[time_col]
        print(f"Time points: {adata.obs['time_point'].unique()}")
    
    # Mark batch as potential confounder
    if batch_col is not None and batch_col in adata.obs.columns:
        adata.obs["batch"] = adata.obs[batch_col].astype("category")
        print(f"Batches: {adata.obs['batch'].nunique()}")
    
    return adata


def select_highly_variable_genes(
    adata: "sc.AnnData",
    n_top_genes: int = 2000,
    subset: bool = False,
) -> "sc.AnnData":
    """Identify highly variable genes (HVGs) for dimensionality reduction.
    
    Note: This uses a temporary log-normalization for HVG selection only.
    The raw counts are preserved for NB/ZINB modeling.
    
    Args:
        adata: AnnData object with raw counts
        n_top_genes: Number of HVGs to select
        subset: If True, subset to HVGs only; if False, just mark them
    """
    import scanpy as sc
    from scipy import sparse
    
    # Store raw counts
    adata.layers["counts"] = adata.X.copy()
    
    # Temporary normalization for HVG selection
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    # Find HVGs
    sc.pp.highly_variable_genes(
        adata_norm,
        n_top_genes=n_top_genes,
        flavor="seurat_v3" if sparse.issparse(adata.X) else "seurat",
    )
    
    # Transfer HVG annotations to original
    adata.var["highly_variable"] = adata_norm.var["highly_variable"]
    
    if subset:
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"Subset to {adata.n_vars} highly variable genes")
    
    return adata


def preprocess_for_causal(
    adata: "sc.AnnData",
    output_path: str | Path = "counts_qc.h5ad",
) -> "sc.AnnData":
    """Final preparation for causal inference: ensure counts and metadata are ready.
    
    Args:
        adata: Preprocessed AnnData
        output_path: Path to save the h5ad file
    """
    # Ensure counts are stored (not normalized)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    
    # Ensure library size is computed
    if "library_size" not in adata.obs:
        adata.obs["library_size"] = np.array(adata.X.sum(axis=1)).ravel()
    
    # Save
    adata.write_h5ad(output_path)
    print(f"Saved to {output_path}")
    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Library size range: [{adata.obs.library_size.min():.0f}, "
          f"{adata.obs.library_size.max():.0f}]")
    
    # Report perturbation info if available
    if "is_control" in adata.obs.columns:
        n_control = adata.obs["is_control"].sum()
        print(f"  Control cells: {n_control} ({100*n_control/adata.n_obs:.1f}%)")
    
    return adata


def main():
    """Command-line interface for scRNA-seq preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess scRNA-seq/Perturb-seq data for causal inference"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        choices=["replogle_k562", "norman_k562", "dixit_k562"],
        help="Dataset to download and preprocess (uses standard paths)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input h5ad or 10x MTX folder",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output h5ad file path",
    )
    parser.add_argument(
        "--perturbation-col",
        type=str,
        default="perturbation",
        help="Column name for perturbation labels",
    )
    parser.add_argument(
        "--control-value",
        type=str,
        default="control",
        help="Value indicating control cells",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="Minimum genes per cell",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=3,
        help="Minimum cells per gene",
    )
    parser.add_argument(
        "--max-mito",
        type=float,
        default=20.0,
        help="Maximum mitochondrial percentage",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="Number of highly variable genes to select (optional)",
    )
    parser.add_argument(
        "--setup-dirs",
        action="store_true",
        help="Create data directory structure and exit",
    )
    
    args = parser.parse_args()
    
    # Setup directories mode
    if args.setup_dirs:
        from causalbiolab.data.paths import setup_data_directories
        setup_data_directories()
        return
    
    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    elif args.dataset is not None:
        from causalbiolab.data.paths import get_data_paths
        paths = get_data_paths()
        paths.ensure_dirs(args.dataset, "perturbation")
        output_path = paths.perturbation_processed(args.dataset, "counts.h5ad")
    else:
        output_path = Path("counts_qc.h5ad")
    
    # Load data
    if args.input is not None:
        input_path = Path(args.input)
        print(f"Loading from {input_path}...")
        if input_path.suffix == ".h5ad":
            adata = load_from_h5ad(input_path)
        else:
            adata = load_10x_mtx(input_path)
    else:
        raise ValueError("Please provide --input or --dataset")
    
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # QC metrics
    adata = compute_qc_metrics(adata)
    
    # Filter
    adata = filter_cells_and_genes(
        adata,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        max_mito_pct=args.max_mito,
    )
    
    # Extract perturbation metadata
    adata = extract_perturbation_metadata(
        adata,
        perturbation_col=args.perturbation_col,
        control_value=args.control_value,
    )
    
    # HVG selection (optional)
    if args.n_hvg is not None:
        adata = select_highly_variable_genes(adata, n_top_genes=args.n_hvg)
    
    # Save
    preprocess_for_causal(adata, output_path=output_path)


if __name__ == "__main__":
    main()
