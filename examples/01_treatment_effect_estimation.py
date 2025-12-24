"""
Treatment Effect Estimation on Perturb-seq Data
================================================

This example demonstrates the core workflow for estimating causal treatment
effects from perturbation experiments, following Phase 2 of the roadmap.

Workflow:
1. Load Perturb-seq data (or generate synthetic data for demo)
2. Define a biologically meaningful outcome (pathway activity)
3. Estimate ATE using multiple methods
4. Compare estimators and interpret results

This mirrors how Ochre Bio and similar companies would approach target
prioritization: identifying perturbations with large, robust causal effects
on disease-relevant outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Check if we have real data or need synthetic
USE_SYNTHETIC = True  # Set to False if you have Norman dataset


def generate_synthetic_perturbation_data(
    n_cells: int = 2000,
    n_genes: int = 500,
    n_perturbations: int = 5,
    effect_sizes: list[float] | None = None,
    confounding_strength: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate synthetic Perturb-seq-like data with known ground truth.
    
    This creates a scenario where:
    - Cells are assigned to perturbations (not fully random - some confounding)
    - Each perturbation has a known causal effect on an outcome
    - Library size acts as a confounder (affects both treatment and outcome)
    
    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        n_perturbations: Number of different perturbations
        effect_sizes: True causal effect of each perturbation (None = random)
        confounding_strength: How much library size affects treatment assignment
        seed: Random seed
        
    Returns:
        X: Covariates (n_cells, n_features)
        T: Treatment assignment (n_cells,) - which perturbation (0 = control)
        Y: Outcome (n_cells,)
        metadata: Dict with ground truth and additional info
    """
    rng = np.random.default_rng(seed)
    
    # Generate library sizes (log-normal, realistic for scRNA-seq)
    log_lib_size = rng.normal(10, 0.5, n_cells)  # ~10k-50k counts
    library_size = np.exp(log_lib_size)
    
    # Confounded treatment assignment
    # Higher library size -> more likely to be in certain perturbation groups
    # This simulates batch effects or technical confounding
    treatment_probs = np.zeros((n_cells, n_perturbations + 1))
    treatment_probs[:, 0] = 0.5  # Base probability of control
    
    for i in range(n_perturbations):
        # Each perturbation has different relationship with library size
        treatment_probs[:, i + 1] = 0.1 + confounding_strength * (
            (log_lib_size - log_lib_size.mean()) / log_lib_size.std()
        ) * ((-1) ** i)
    
    # Normalize to valid probabilities
    treatment_probs = np.clip(treatment_probs, 0.01, 0.99)
    treatment_probs = treatment_probs / treatment_probs.sum(axis=1, keepdims=True)
    
    # Assign treatments
    T = np.array([
        rng.choice(n_perturbations + 1, p=treatment_probs[i])
        for i in range(n_cells)
    ])
    
    # True causal effects
    if effect_sizes is None:
        # Random effects: some positive, some negative, some near zero
        effect_sizes = rng.uniform(-2, 2, n_perturbations)
    effect_sizes = np.array(effect_sizes)
    
    # Generate outcome
    # Y = baseline + treatment_effect + confounding + noise
    baseline = 5.0
    
    # Confounding: library size affects outcome directly
    confounding_effect = 0.5 * (log_lib_size - log_lib_size.mean())
    
    # Treatment effect
    treatment_effect = np.zeros(n_cells)
    for i in range(n_perturbations):
        treatment_effect[T == (i + 1)] = effect_sizes[i]
    
    # Noise
    noise = rng.normal(0, 1, n_cells)
    
    Y = baseline + treatment_effect + confounding_effect + noise
    
    # Covariates
    X = np.column_stack([
        library_size,
        log_lib_size,
    ])
    
    metadata = {
        "true_effects": effect_sizes,
        "perturbation_names": [f"Gene_{i+1}" for i in range(n_perturbations)],
        "confounding_strength": confounding_strength,
        "n_per_group": {
            "control": (T == 0).sum(),
            **{f"Gene_{i+1}": (T == i+1).sum() for i in range(n_perturbations)}
        },
    }
    
    return X, T, Y, metadata


def run_ate_comparison(
    X: np.ndarray,
    T_binary: np.ndarray,
    Y: np.ndarray,
    true_effect: float | None = None,
) -> dict:
    """Run ATE estimation with multiple methods and compare.
    
    Args:
        X: Covariates
        T_binary: Binary treatment (0/1)
        Y: Outcome
        true_effect: Ground truth effect (if known)
        
    Returns:
        Dictionary of results
    """
    from causalbiolab.estimation import (
        NaiveATE,
        PropensityATE,
        DoublyRobustATE,
        compare_estimators,
        print_comparison,
    )
    
    results = compare_estimators(X, T_binary, Y)
    print_comparison(results)
    
    if true_effect is not None:
        print(f"\nTrue causal effect: {true_effect:.4f}")
        print("\nBias (estimate - truth):")
        for method, result in results.items():
            bias = result.ate - true_effect
            print(f"  {method}: {bias:+.4f}")
    
    return results


def plot_ate_comparison(
    results: dict,
    true_effect: float | None = None,
    title: str = "ATE Estimation Comparison",
) -> plt.Figure:
    """Visualize ATE estimates with confidence intervals.
    
    Args:
        results: Dictionary of ATEResult objects
        true_effect: Ground truth (if known)
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    ates = [results[m].ate for m in methods]
    ci_lowers = [results[m].ci_lower for m in methods]
    ci_uppers = [results[m].ci_upper for m in methods]
    
    y_pos = np.arange(len(methods))
    
    # Plot estimates with error bars
    ax.errorbar(
        ates, y_pos,
        xerr=[np.array(ates) - np.array(ci_lowers),
              np.array(ci_uppers) - np.array(ates)],
        fmt='o', capsize=5, capthick=2, markersize=10,
        color='steelblue', ecolor='steelblue',
    )
    
    # Add true effect line if known
    if true_effect is not None:
        ax.axvline(true_effect, color='red', linestyle='--', linewidth=2,
                   label=f'True effect = {true_effect:.3f}')
        ax.legend(loc='upper right')
    
    # Add zero line
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Average Treatment Effect (ATE)')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main example workflow."""
    print("=" * 70)
    print("Treatment Effect Estimation Example")
    print("=" * 70)
    
    if USE_SYNTHETIC:
        print("\n📊 Generating synthetic Perturb-seq data...")
        print("   (Set USE_SYNTHETIC=False to use real Norman dataset)\n")
        
        # Generate data with known ground truth
        X, T, Y, metadata = generate_synthetic_perturbation_data(
            n_cells=3000,
            n_perturbations=5,
            effect_sizes=[2.0, -1.5, 0.5, 0.0, -2.5],  # Known effects
            confounding_strength=0.4,  # Moderate confounding
        )
        
        print("Ground truth effects:")
        for name, effect in zip(metadata["perturbation_names"], metadata["true_effects"]):
            print(f"  {name}: {effect:+.2f}")
        
        print("\nCells per group:")
        for group, count in metadata["n_per_group"].items():
            print(f"  {group}: {count}")
        
        # Analyze one perturbation: Gene_1 (true effect = 2.0)
        print("\n" + "=" * 70)
        print("Analyzing Gene_1 (true effect = 2.0)")
        print("=" * 70)
        
        # Create binary treatment: Gene_1 vs control
        mask = (T == 0) | (T == 1)  # Control or Gene_1
        X_sub = X[mask]
        T_binary = (T[mask] == 1).astype(int)
        Y_sub = Y[mask]
        
        results_gene1 = run_ate_comparison(
            X_sub, T_binary, Y_sub,
            true_effect=metadata["true_effects"][0],
        )
        
        # Analyze Gene_4 (true effect = 0.0, null effect)
        print("\n" + "=" * 70)
        print("Analyzing Gene_4 (true effect = 0.0, null)")
        print("=" * 70)
        
        mask = (T == 0) | (T == 4)
        X_sub = X[mask]
        T_binary = (T[mask] == 4).astype(int)
        Y_sub = Y[mask]
        
        results_gene4 = run_ate_comparison(
            X_sub, T_binary, Y_sub,
            true_effect=metadata["true_effects"][3],
        )
        
        # Visualize
        print("\n📈 Creating visualization...")
        fig = plot_ate_comparison(
            results_gene1,
            true_effect=metadata["true_effects"][0],
            title="ATE Estimation: Gene_1 Knockout (True Effect = 2.0)",
        )
        
        # Save figure
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "ate_comparison.png", dpi=150, bbox_inches='tight')
        print(f"   Saved to {output_dir / 'ate_comparison.png'}")
        
        # Key insights
        print("\n" + "=" * 70)
        print("KEY INSIGHTS")
        print("=" * 70)
        print("""
1. NAIVE ESTIMATOR is biased when confounding exists
   - In this example, library size affects both treatment assignment
     and outcome, creating confounding bias.

2. PROPENSITY WEIGHTING corrects for confounding
   - By modeling P(T|X), we can reweight observations to remove
     confounding bias.

3. DOUBLY ROBUST is most reliable
   - Combines outcome modeling with propensity weighting.
   - Consistent if EITHER model is correct.
   - Generally recommended for observational data.

4. FOR PERTURB-SEQ:
   - If cells are truly randomly assigned to perturbations,
     naive estimator is valid.
   - But batch effects, library size variation, and cell quality
     can introduce confounding.
   - Always compare methods as a sensitivity check.
""")
        
    else:
        # Real data workflow
        print("\n📊 Loading Norman dataset...")
        from causalbiolab.data import norman_paths, get_data_paths
        from causalbiolab.estimation import define_treatment_outcome
        
        paths = norman_paths()
        if not paths["counts"].exists():
            print(f"❌ Norman dataset not found at {paths['counts']}")
            print("   Please download and preprocess the data first:")
            print("   python -m causalbiolab.data.sc_preprocess --dataset norman_k562")
            return
        
        import scanpy as sc
        adata = sc.read_h5ad(paths["counts"])
        print(f"   Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # TODO: Implement real data workflow
        print("\n⚠️  Real data workflow not yet implemented.")
        print("   Run with USE_SYNTHETIC=True for now.")


if __name__ == "__main__":
    main()
