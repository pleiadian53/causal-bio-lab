"""Biological outcome definitions for treatment effect estimation.

This module provides functions to compute biologically meaningful outcomes
from gene expression data, moving beyond pure ML metrics to outcomes that
matter for drug discovery.

Key outcome types:
- Pathway activity scores (e.g., fibrosis, inflammation)
- Gene module scores (e.g., cell cycle, stress response)
- Differential expression magnitude
- Phenotype proxies

Usage:
    from causalbiolab.estimation.outcomes import (
        compute_pathway_score,
        compute_module_score,
    )
    
    # Compute fibrosis-related pathway activity
    fibrosis_score = compute_pathway_score(adata, pathway="fibrosis")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import scanpy as sc
    from numpy.typing import NDArray


# Example gene sets for liver disease (Ochre Bio-relevant)
LIVER_PATHWAYS = {
    "fibrosis": [
        "COL1A1", "COL1A2", "COL3A1", "ACTA2", "TGFB1", "TGFB2",
        "CTGF", "LOX", "LOXL2", "TIMP1", "MMP2", "MMP9",
    ],
    "inflammation": [
        "IL6", "IL1B", "TNF", "CCL2", "CXCL8", "NFKB1",
        "STAT3", "IL10", "TGFB1", "CD68",
    ],
    "lipid_metabolism": [
        "PPARA", "PPARG", "SREBF1", "FASN", "ACACA", "SCD",
        "DGAT1", "DGAT2", "PNPLA3", "MTTP",
    ],
    "oxidative_stress": [
        "NFE2L2", "KEAP1", "NQO1", "HMOX1", "SOD1", "SOD2",
        "CAT", "GPX1", "GSR", "TXNRD1",
    ],
    "apoptosis": [
        "BCL2", "BAX", "CASP3", "CASP8", "CASP9", "TP53",
        "FAS", "FASLG", "CYCS", "APAF1",
    ],
    "cell_cycle": [
        "MKI67", "PCNA", "CDK1", "CDK2", "CCNA2", "CCNB1",
        "CCND1", "CCNE1", "E2F1", "RB1",
    ],
}


def compute_pathway_score(
    adata: "sc.AnnData",
    pathway: str | list[str],
    method: Literal["mean", "median", "pca"] = "mean",
    layer: str | None = None,
) -> NDArray[np.floating]:
    """Compute pathway activity score for each cell.
    
    Args:
        adata: AnnData object with gene expression
        pathway: Either a pathway name from LIVER_PATHWAYS or a list of gene names
        method: Aggregation method ("mean", "median", or "pca" for first PC)
        layer: Layer to use (None = .X)
        
    Returns:
        Array of pathway scores (n_cells,)
    """
    from typing import Literal
    
    # Get gene list
    if isinstance(pathway, str):
        if pathway not in LIVER_PATHWAYS:
            raise ValueError(
                f"Unknown pathway '{pathway}'. "
                f"Available: {list(LIVER_PATHWAYS.keys())}"
            )
        genes = LIVER_PATHWAYS[pathway]
    else:
        genes = pathway
    
    # Find genes present in data
    genes_present = [g for g in genes if g in adata.var_names]
    if len(genes_present) == 0:
        raise ValueError(f"No genes from pathway found in data. Tried: {genes[:5]}...")
    
    if len(genes_present) < len(genes):
        missing = set(genes) - set(genes_present)
        print(f"Warning: {len(missing)} genes not found: {list(missing)[:5]}...")
    
    # Extract expression matrix
    if layer is not None:
        X = adata[:, genes_present].layers[layer]
    else:
        X = adata[:, genes_present].X
    
    # Convert sparse if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Compute score
    if method == "mean":
        scores = X.mean(axis=1)
    elif method == "median":
        scores = np.median(X, axis=1)
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X).ravel()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.asarray(scores).ravel()


def compute_module_score(
    adata: "sc.AnnData",
    gene_list: list[str],
    ctrl_size: int = 50,
    n_bins: int = 25,
    layer: str | None = None,
) -> NDArray[np.floating]:
    """Compute gene module score using scanpy's score_genes.
    
    This is similar to Seurat's AddModuleScore - it computes the average
    expression of a gene set, controlling for expression level by subtracting
    the average expression of a control gene set.
    
    Args:
        adata: AnnData object
        gene_list: List of genes in the module
        ctrl_size: Number of control genes per expression bin
        n_bins: Number of expression bins for control gene selection
        layer: Layer to use
        
    Returns:
        Module scores (n_cells,)
    """
    import scanpy as sc
    
    # Use scanpy's implementation
    sc.tl.score_genes(
        adata,
        gene_list=gene_list,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        score_name="_temp_module_score",
        use_raw=False,
    )
    
    scores = adata.obs["_temp_module_score"].values.copy()
    del adata.obs["_temp_module_score"]
    
    return scores


def compute_perturbation_effect(
    adata: "sc.AnnData",
    perturbation_col: str = "perturbation",
    control_value: str = "control",
    outcome_genes: list[str] | None = None,
    method: Literal["mean_diff", "log2fc", "zscore"] = "mean_diff",
) -> NDArray[np.floating]:
    """Compute perturbation effect as outcome variable.
    
    For each perturbed cell, computes how different it is from control cells
    in terms of expression of outcome genes.
    
    Args:
        adata: AnnData with perturbation annotations
        perturbation_col: Column containing perturbation labels
        control_value: Value indicating control cells
        outcome_genes: Genes to use for outcome (None = all HVGs)
        method: How to compute effect
        
    Returns:
        Perturbation effect scores (n_cells,)
    """
    from typing import Literal
    
    # Identify control cells
    is_control = adata.obs[perturbation_col].str.lower() == control_value.lower()
    
    # Select genes
    if outcome_genes is not None:
        genes = [g for g in outcome_genes if g in adata.var_names]
    elif "highly_variable" in adata.var.columns:
        genes = adata.var_names[adata.var["highly_variable"]].tolist()
    else:
        genes = adata.var_names.tolist()
    
    # Get expression
    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Control mean and std
    X_ctrl = X[is_control]
    ctrl_mean = X_ctrl.mean(axis=0)
    ctrl_std = X_ctrl.std(axis=0) + 1e-8
    
    if method == "mean_diff":
        # Average absolute difference from control mean
        effects = np.abs(X - ctrl_mean).mean(axis=1)
    elif method == "log2fc":
        # Average log2 fold change
        effects = np.abs(np.log2((X + 1) / (ctrl_mean + 1))).mean(axis=1)
    elif method == "zscore":
        # Average z-score magnitude
        effects = np.abs((X - ctrl_mean) / ctrl_std).mean(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return effects


def define_treatment_outcome(
    adata: "sc.AnnData",
    target_perturbation: str,
    perturbation_col: str = "perturbation",
    control_value: str = "control",
    outcome_type: str = "pathway",
    outcome_spec: str | list[str] = "fibrosis",
) -> tuple[NDArray, NDArray, NDArray]:
    """Define treatment and outcome for ATE estimation.
    
    This is a convenience function that extracts:
    - X: Covariates (e.g., library size, batch)
    - T: Binary treatment (1 = target perturbation, 0 = control)
    - Y: Outcome variable
    
    Args:
        adata: AnnData with perturbation data
        target_perturbation: The perturbation to study (e.g., "TGFB1")
        perturbation_col: Column with perturbation labels
        control_value: Value for control cells
        outcome_type: "pathway", "module", or "effect"
        outcome_spec: Pathway name, gene list, or method for outcome
        
    Returns:
        Tuple of (X, T, Y) arrays ready for ATE estimation
    """
    from typing import Literal
    
    # Filter to target perturbation and control only
    mask = (
        (adata.obs[perturbation_col] == target_perturbation) |
        (adata.obs[perturbation_col].str.lower() == control_value.lower())
    )
    adata_subset = adata[mask].copy()
    
    # Define treatment
    T = (adata_subset.obs[perturbation_col] == target_perturbation).astype(int).values
    
    # Define covariates (potential confounders)
    covariate_cols = []
    if "library_size" in adata_subset.obs.columns:
        covariate_cols.append("library_size")
    if "batch" in adata_subset.obs.columns:
        # One-hot encode batch
        batch_dummies = pd.get_dummies(
            adata_subset.obs["batch"], prefix="batch", drop_first=True
        )
        X_batch = batch_dummies.values
    else:
        X_batch = None
    
    # Build covariate matrix
    X_parts = []
    if "library_size" in adata_subset.obs.columns:
        X_parts.append(adata_subset.obs["library_size"].values.reshape(-1, 1))
    if "n_genes" in adata_subset.obs.columns:
        X_parts.append(adata_subset.obs["n_genes"].values.reshape(-1, 1))
    if X_batch is not None:
        X_parts.append(X_batch)
    
    if X_parts:
        X = np.hstack(X_parts)
    else:
        # Fallback: use library size computed from data
        lib_size = np.array(adata_subset.X.sum(axis=1)).ravel()
        X = lib_size.reshape(-1, 1)
    
    # Define outcome
    if outcome_type == "pathway":
        Y = compute_pathway_score(adata_subset, pathway=outcome_spec)
    elif outcome_type == "module":
        Y = compute_module_score(adata_subset, gene_list=outcome_spec)
    elif outcome_type == "effect":
        Y = compute_perturbation_effect(
            adata_subset,
            perturbation_col=perturbation_col,
            control_value=control_value,
            method=outcome_spec if isinstance(outcome_spec, str) else "mean_diff",
        )
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")
    
    return X, T, Y


# Need pandas for one-hot encoding
import pandas as pd
from typing import Literal
