"""Bulk RNA-seq preprocessing for causal inference and treatment effect estimation.

This script provides preprocessing for bulk RNA-seq data used in:
- Treatment effect estimation (drug response, perturbation effects)
- Causal discovery on gene regulatory networks
- Observational studies with confounders

Key features:
- Preserves raw counts for statistical modeling
- Extracts treatment/condition labels
- Identifies potential confounders (batch, donor, etc.)
- Supports matched/paired experimental designs

Usage:
    # From CSV with treatment labels
    python -m causalbiolab.data.bulk_preprocess csv -c counts.csv -m metadata.csv -o processed.h5ad
    
    # From GEO
    python -m causalbiolab.data.bulk_preprocess geo -g GSE12345 -o processed.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def load_from_csv(
    counts_path: str | Path,
    metadata_path: str | Path | None = None,
    genes_as_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load counts matrix and optional metadata from CSV files.
    
    Args:
        counts_path: Path to counts CSV (genes x samples or samples x genes)
        metadata_path: Optional path to sample metadata CSV
        genes_as_rows: If True, counts are genes x samples; if False, samples x genes
        
    Returns:
        Tuple of (counts DataFrame with genes as rows, metadata DataFrame or None)
    """
    counts = pd.read_csv(counts_path, index_col=0)
    
    if not genes_as_rows:
        counts = counts.T
    
    print(f"Loaded counts: {counts.shape[0]} genes x {counts.shape[1]} samples")
    
    metadata = None
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path, index_col=0)
        print(f"Loaded metadata: {metadata.shape[0]} samples x {metadata.shape[1]} columns")
    
    return counts, metadata


def load_from_geo(
    geo_id: str,
    output_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and load a GEO dataset using GEOparse.
    
    Args:
        geo_id: GEO accession (e.g., "GSE12345")
        output_dir: Directory to cache downloaded files
        
    Returns:
        Tuple of (counts DataFrame, metadata DataFrame)
    """
    try:
        import GEOparse
    except ImportError:
        raise ImportError("GEOparse not installed. Run: pip install GEOparse")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {geo_id}...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir=str(output_dir))
    
    # Extract expression data
    pivot_samples = gse.pivot_samples("VALUE")
    counts = pivot_samples.T
    
    # Extract sample metadata
    metadata_list = []
    for gsm_name, gsm in gse.gsms.items():
        meta = {"sample_id": gsm_name}
        meta.update(gsm.metadata)
        # Flatten list values
        for k, v in meta.items():
            if isinstance(v, list) and len(v) == 1:
                meta[k] = v[0]
        metadata_list.append(meta)
    
    metadata = pd.DataFrame(metadata_list).set_index("sample_id")
    
    print(f"Loaded from GEO: {counts.shape[0]} genes x {counts.shape[1]} samples")
    
    return counts, metadata


def compute_library_size(counts: pd.DataFrame) -> pd.Series:
    """Compute library size (total counts per sample).
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        
    Returns:
        Series with library size per sample
    """
    library_size = counts.sum(axis=0)
    print(f"Library size range: [{library_size.min():.0f}, {library_size.max():.0f}]")
    return library_size


def filter_genes(
    counts: pd.DataFrame,
    min_samples: int = 10,
    min_counts: int = 1,
) -> pd.DataFrame:
    """Filter lowly-expressed genes.
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        min_samples: Minimum number of samples where gene must be expressed
        min_counts: Minimum count threshold for "expressed"
        
    Returns:
        Filtered counts DataFrame
    """
    n_genes_before = counts.shape[0]
    
    # Keep genes expressed in at least min_samples
    expressed = (counts >= min_counts).sum(axis=1)
    keep = expressed >= min_samples
    counts_filtered = counts.loc[keep]
    
    print(f"Filtered genes: {n_genes_before} -> {counts_filtered.shape[0]}")
    
    return counts_filtered


def extract_treatment_metadata(
    metadata: pd.DataFrame,
    treatment_col: str = "treatment",
    control_value: str = "control",
    dose_col: str | None = None,
    time_col: str | None = None,
    batch_col: str | None = None,
    donor_col: str | None = None,
) -> pd.DataFrame:
    """Extract and standardize treatment metadata for causal inference.
    
    Args:
        metadata: Sample metadata DataFrame
        treatment_col: Column containing treatment/condition labels
        control_value: Value indicating control samples
        dose_col: Column containing dose information
        time_col: Column containing time point information
        batch_col: Column containing batch (potential confounder)
        donor_col: Column containing donor/patient ID (for paired designs)
        
    Returns:
        Metadata with standardized columns for causal inference
    """
    result = metadata.copy()
    
    # Treatment indicator
    if treatment_col in metadata.columns:
        result["is_control"] = (
            metadata[treatment_col].str.lower() == control_value.lower()
        ) | (
            metadata[treatment_col].isna()
        )
        result["treatment"] = metadata[treatment_col]
        
        n_control = result["is_control"].sum()
        n_treated = (~result["is_control"]).sum()
        print(f"Treatment groups: {n_control} control, {n_treated} treated")
        print(f"  Unique treatments: {metadata[treatment_col].nunique()}")
    
    # Dose (for dose-response analysis)
    if dose_col is not None and dose_col in metadata.columns:
        result["dose"] = pd.to_numeric(metadata[dose_col], errors="coerce")
        print(f"Dose range: [{result['dose'].min()}, {result['dose'].max()}]")
    
    # Time point (for longitudinal studies)
    if time_col is not None and time_col in metadata.columns:
        result["time_point"] = metadata[time_col]
        print(f"Time points: {result['time_point'].unique()}")
    
    # Batch (potential confounder)
    if batch_col is not None and batch_col in metadata.columns:
        result["batch"] = metadata[batch_col].astype("category")
        print(f"Batches: {result['batch'].nunique()} (potential confounder)")
    
    # Donor/patient ID (for paired designs)
    if donor_col is not None and donor_col in metadata.columns:
        result["donor_id"] = metadata[donor_col]
        print(f"Donors: {result['donor_id'].nunique()} (paired design)")
    
    return result


def to_anndata(
    counts: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    library_size: pd.Series | None = None,
) -> "ad.AnnData":
    """Convert counts and metadata to AnnData format.
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        metadata: Optional sample metadata DataFrame
        library_size: Optional precomputed library sizes
        
    Returns:
        AnnData object with samples as obs and genes as var
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata not installed. Run: pip install anndata")
    
    # AnnData expects samples x genes
    X = counts.T.values
    
    # Create obs (sample metadata)
    obs = pd.DataFrame(index=counts.columns)
    if metadata is not None:
        # Align metadata with counts columns
        obs = metadata.loc[counts.columns].copy()
    
    # Add library size
    if library_size is None:
        library_size = counts.sum(axis=0)
    obs["library_size"] = library_size.loc[counts.columns].values
    
    # Create var (gene metadata)
    var = pd.DataFrame(index=counts.index)
    var["n_samples"] = (counts > 0).sum(axis=1).values
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Store raw counts in layers
    adata.layers["counts"] = X.copy()
    
    print(f"Created AnnData: {adata.n_obs} samples x {adata.n_vars} genes")
    
    return adata


def preprocess_bulk_for_causal(
    counts: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    treatment_col: str = "treatment",
    control_value: str = "control",
    dose_col: str | None = None,
    time_col: str | None = None,
    batch_col: str | None = None,
    donor_col: str | None = None,
    min_samples: int = 10,
    output_path: str | Path | None = None,
) -> "ad.AnnData":
    """Full preprocessing pipeline for bulk RNA-seq causal inference.
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        metadata: Optional sample metadata
        treatment_col: Column for treatment labels
        control_value: Value indicating control
        dose_col: Column for dose information
        time_col: Column for time points
        batch_col: Column for batch (confounder)
        donor_col: Column for donor ID (paired design)
        min_samples: Minimum samples per gene for filtering
        output_path: Optional path to save h5ad file
        
    Returns:
        Preprocessed AnnData object
    """
    # Compute library size BEFORE filtering
    library_size = compute_library_size(counts)
    
    # Filter genes
    counts_filtered = filter_genes(counts, min_samples=min_samples)
    
    # Extract treatment metadata if available
    if metadata is not None:
        metadata = extract_treatment_metadata(
            metadata,
            treatment_col=treatment_col,
            control_value=control_value,
            dose_col=dose_col,
            time_col=time_col,
            batch_col=batch_col,
            donor_col=donor_col,
        )
    
    # Convert to AnnData
    adata = to_anndata(counts_filtered, metadata, library_size)
    
    # Save if requested
    if output_path is not None:
        adata.write_h5ad(output_path)
        print(f"Saved to {output_path}")
    
    return adata


def main():
    """Command-line interface for bulk RNA-seq preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess bulk RNA-seq data for causal inference"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Data source")
    
    # CSV subcommand
    csv_parser = subparsers.add_parser("csv", help="Load from CSV files")
    csv_parser.add_argument(
        "--counts", "-c",
        type=str,
        required=True,
        help="Path to counts CSV (genes x samples)",
    )
    csv_parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=None,
        help="Path to metadata CSV",
    )
    csv_parser.add_argument(
        "--samples-as-rows",
        action="store_true",
        help="If set, counts CSV has samples as rows",
    )
    
    # GEO subcommand
    geo_parser = subparsers.add_parser("geo", help="Download from GEO")
    geo_parser.add_argument(
        "--geo-id", "-g",
        type=str,
        required=True,
        help="GEO accession (e.g., GSE12345)",
    )
    geo_parser.add_argument(
        "--cache-dir",
        type=str,
        default="./geo_cache",
        help="Directory to cache downloaded files",
    )
    
    # Common arguments for both subparsers
    for subparser in [csv_parser, geo_parser]:
        subparser.add_argument(
            "--output", "-o",
            type=str,
            default="bulk_counts.h5ad",
            help="Output h5ad file path",
        )
        subparser.add_argument(
            "--min-samples",
            type=int,
            default=10,
            help="Minimum samples per gene",
        )
        subparser.add_argument(
            "--treatment-col",
            type=str,
            default="treatment",
            help="Column name for treatment labels",
        )
        subparser.add_argument(
            "--control-value",
            type=str,
            default="control",
            help="Value indicating control samples",
        )
        subparser.add_argument(
            "--dose-col",
            type=str,
            default=None,
            help="Column name for dose information",
        )
        subparser.add_argument(
            "--time-col",
            type=str,
            default=None,
            help="Column name for time points",
        )
        subparser.add_argument(
            "--batch-col",
            type=str,
            default=None,
            help="Column name for batch (potential confounder)",
        )
        subparser.add_argument(
            "--donor-col",
            type=str,
            default=None,
            help="Column name for donor ID (paired design)",
        )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Load data based on source
    if args.command == "csv":
        counts, metadata = load_from_csv(
            args.counts,
            args.metadata,
            genes_as_rows=not args.samples_as_rows,
        )
    elif args.command == "geo":
        counts, metadata = load_from_geo(args.geo_id, args.cache_dir)
    else:
        parser.print_help()
        return
    
    # Preprocess and save
    preprocess_bulk_for_causal(
        counts,
        metadata,
        treatment_col=args.treatment_col,
        control_value=args.control_value,
        dose_col=args.dose_col,
        time_col=args.time_col,
        batch_col=args.batch_col,
        donor_col=args.donor_col,
        min_samples=args.min_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
