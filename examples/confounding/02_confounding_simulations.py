#!/usr/bin/env python
"""
Example: Confounding Simulations for Causal Inference Education

This script demonstrates various confounding scenarios using the
causalbiolab.simulation module. It shows how confounders create
spurious correlations that can mislead causal conclusions.

Inspired by Chapter 01 of "Causal Inference and Discovery in Python"
by Aleksander Molak.

Usage:
    python examples/confounding/02_confounding_simulations.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our simulation module
from causalbiolab.simulation import (
    simulate_basic_confounding,
    simulate_cell_cycle_confounding,
    simulate_batch_effect_confounding,
    simulate_disease_severity_confounding,
    simulate_treatment_effect_with_confounding,
    compare_naive_vs_adjusted,
    plot_confounding_scatter,
)


def demo_basic_confounding():
    """Demonstrate basic confounding with no true causal effect."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Confounding (No True Causal Effect)")
    print("=" * 60)
    
    # Simulate: Z causes both C and X, but C does NOT cause X
    result = simulate_basic_confounding(
        n_samples=1000,
        confounder_effect_on_treatment=1.0,
        confounder_effect_on_outcome=1.0,
        true_causal_effect=0.0,  # No causal effect!
        seed=42,
    )
    
    print(result.summary())
    
    # Key insight
    print("KEY INSIGHT:")
    print(f"  - True causal effect: {result.true_effect:.4f}")
    print(f"  - Naive regression coefficient: {result.naive_regression_coef:.4f}")
    print(f"  - Bias: {abs(result.naive_regression_coef - result.true_effect):.4f}")
    print(f"  - If we intervened on C, X would NOT change!")
    print(f"  - But the naive estimate suggests a strong relationship.")
    
    # Plot
    plot_confounding_scatter(result)
    
    return result


def demo_cell_cycle_confounding():
    """Demonstrate cell cycle confounding in gene expression."""
    print("\n" + "=" * 60)
    print("DEMO 2: Cell Cycle Confounding (MYC vs Ribosomal Genes)")
    print("=" * 60)
    
    result = simulate_cell_cycle_confounding(n_cells=1000, seed=42)
    
    print(result.summary())
    
    # Biological interpretation
    print("BIOLOGICAL INTERPRETATION:")
    print("  - Cell cycle phase drives both MYC and ribosomal gene expression")
    print("  - Observationally: high MYC ↔ high ribosomal genes")
    print("  - But knocking down MYC may have much smaller effect than expected")
    print("  - The correlation is mostly driven by shared cell cycle regulation")
    
    # Show the data
    data = result.data
    print(f"\nData preview:")
    print(data[['cell_cycle_phase', 'MYC', 'ribosomal_genes']].describe())
    
    plot_confounding_scatter(result)
    
    return result


def demo_batch_effect_confounding():
    """Demonstrate batch effect confounding in scRNA-seq."""
    print("\n" + "=" * 60)
    print("DEMO 3: Batch Effect Confounding (Gene A vs Gene B)")
    print("=" * 60)
    
    result = simulate_batch_effect_confounding(
        n_samples=1000,
        n_batches=3,
        seed=42,
    )
    
    print(result.summary())
    
    # Technical interpretation
    print("TECHNICAL INTERPRETATION:")
    print("  - Gene A and Gene B are CAUSALLY INDEPENDENT")
    print("  - But batch effects create spurious correlation")
    print("  - High-efficiency batches → both genes appear higher")
    print("  - This is why batch correction is critical before co-expression analysis")
    
    # Show batch effects
    data = result.data
    print(f"\nMean expression by batch:")
    print(data.groupby('batch')[['gene_A', 'gene_B']].mean())
    
    plot_confounding_scatter(result)
    
    return result


def demo_disease_severity_confounding():
    """Demonstrate disease severity confounding (the protective trap)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Disease Severity Confounding (HIF1A vs Cell Death)")
    print("=" * 60)
    
    result = simulate_disease_severity_confounding(n_patients=500, seed=42)
    
    print(result.summary())
    
    # Critical clinical interpretation
    print("CRITICAL CLINICAL INTERPRETATION:")
    print("  - HIF1A is PROTECTIVE (true effect is NEGATIVE)")
    print("  - But it CORRELATES POSITIVELY with cell death!")
    print("  - Why? Severe disease activates HIF1A AND causes death")
    print("  - Inhibiting HIF1A would likely INCREASE death, not decrease it")
    print("  - This is a classic 'marker vs cause' confusion")
    
    plot_confounding_scatter(result)
    
    return result


def demo_treatment_effect_estimation():
    """Demonstrate biased treatment effect estimation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Treatment Effect Estimation with Confounding")
    print("=" * 60)
    
    # Simulate training data with confounding
    data, true_ate = simulate_treatment_effect_with_confounding(
        n_samples=1000,
        true_ate=10.0,  # True effect: $10,000 increase
        seed=42,
    )
    
    print(f"True Average Treatment Effect: ${true_ate:,.0f}")
    print(f"\nData shape: {data.shape}")
    print(data.head())
    
    # Compare naive vs adjusted estimates
    estimates = compare_naive_vs_adjusted(data)
    
    print(f"\n--- Estimation Results ---")
    print(f"Naive ATE estimate:    ${estimates['naive_ate']:,.0f}")
    print(f"Adjusted ATE estimate: ${estimates['adjusted_ate']:,.0f}")
    print(f"True ATE:              ${true_ate:,.0f}")
    
    print(f"\n--- Why the bias? ---")
    print(f"Treated group mean age:  {estimates['treated_confounder_mean']:.3f}")
    print(f"Control group mean age:  {estimates['control_confounder_mean']:.3f}")
    print("Younger people are more likely to take training,")
    print("but older people have higher baseline earnings.")
    print("This creates NEGATIVE bias in the naive estimate.")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Age distribution by treatment
    treated = data[data['treatment'] == 1]
    control = data[data['treatment'] == 0]
    
    axes[0].hist(control['age'], bins=30, alpha=0.5, label='Control', color='steelblue')
    axes[0].hist(treated['age'], bins=30, alpha=0.5, label='Treated', color='coral')
    axes[0].set_xlabel('Age (normalized)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Age Distribution by Treatment\n(Younger people more likely to be treated)')
    axes[0].legend()
    
    # Outcome vs age, colored by treatment
    axes[1].scatter(control['age'], control['outcome'], alpha=0.5, 
                    label='Control', color='steelblue', s=20)
    axes[1].scatter(treated['age'], treated['outcome'], alpha=0.5,
                    label='Treated', color='coral', s=20)
    axes[1].set_xlabel('Age (normalized)')
    axes[1].set_ylabel('Outcome (earnings)')
    axes[1].set_title('Outcome vs Age by Treatment\n(Age confounds the treatment effect)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return data, estimates


def main():
    """Run all confounding demonstrations."""
    print("=" * 60)
    print("CONFOUNDING SIMULATIONS FOR CAUSAL INFERENCE")
    print("=" * 60)
    print("\nThis script demonstrates how confounders create spurious")
    print("correlations that can mislead causal conclusions.")
    print("\nKey takeaway: Correlation ≠ Causation, and confounding")
    print("is the primary reason why.")
    
    # Run demos
    demo_basic_confounding()
    demo_cell_cycle_confounding()
    demo_batch_effect_confounding()
    demo_disease_severity_confounding()
    demo_treatment_effect_estimation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
In all these examples, we saw:

1. BASIC CONFOUNDING: Z → C and Z → X creates C ↔ X correlation
   even when C has no causal effect on X.

2. CELL CYCLE: Cell cycle drives both MYC and ribosomal genes,
   making them appear correlated even if MYC doesn't regulate ribosomes.

3. BATCH EFFECTS: Technical variation inflates correlations between
   causally independent genes.

4. DISEASE SEVERITY: Protective genes can correlate with bad outcomes
   because severity drives both. Inhibiting them makes things worse!

5. TREATMENT EFFECTS: Naive comparison of treated vs control is biased
   when confounders affect treatment assignment.

SOLUTIONS:
- Measure and adjust for confounders (when possible)
- Use perturbation data (Perturb-seq, CRISPR screens)
- Apply causal inference methods (propensity scores, doubly robust)
- Be honest about what observational data can and cannot tell us
""")


if __name__ == "__main__":
    main()
