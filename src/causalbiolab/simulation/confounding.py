"""
Confounding simulation module for causal inference education.

This module provides data generating processes (DGPs) that demonstrate
confounding effects, inspired by the "Causal Inference and Discovery in Python"
book by Aleksander Molak.

The key insight: when a confounder Z affects both treatment C and outcome X,
we observe a spurious correlation between C and X that does not reflect
the true causal effect.

References:
    - Molak, A. "Causal Inference and Discovery in Python" (Packt, 2023)
    - Pearl, J. "Causality" (Cambridge, 2009)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ConfoundingResult:
    """Results from a confounding simulation.
    
    Attributes:
        data: DataFrame with confounder (Z), treatment (C), and outcome (X)
        true_effect: The true causal effect of C on X (regression coefficient)
        naive_regression_coef: Naive regression coefficient from regressing X on C
        observed_correlation: The observed correlation between C and X
        confounder_treatment_corr: Correlation between Z and C
        confounder_outcome_corr: Correlation between Z and X
        description: Human-readable description of the DGP
    """
    data: pd.DataFrame
    true_effect: float
    naive_regression_coef: float
    observed_correlation: float
    confounder_treatment_corr: float
    confounder_outcome_corr: float
    description: str
    
    def summary(self) -> str:
        """Return a summary of the confounding simulation."""
        bias = abs(self.naive_regression_coef - self.true_effect)
        return f"""
Confounding Simulation Summary
==============================
{self.description}

True causal effect of C → X: {self.true_effect:.4f}
Naive regression coefficient: {self.naive_regression_coef:.4f}
Bias due to confounding:      {bias:.4f}

Additional diagnostics:
  Observed correlation (C, X): {self.observed_correlation:.4f}
  Z → C correlation: {self.confounder_treatment_corr:.4f}
  Z → X correlation: {self.confounder_outcome_corr:.4f}
"""


def simulate_basic_confounding(
    n_samples: int = 1000,
    confounder_effect_on_treatment: float = 1.0,
    confounder_effect_on_outcome: float = 1.0,
    true_causal_effect: float = 0.0,
    noise_treatment: float = 0.1,
    noise_outcome: float = 0.3,
    seed: Optional[int] = None,
) -> ConfoundingResult:
    """
    Simulate basic confounding: Z → C and Z → X, with optional C → X.
    
    The DAG is:
        Z (confounder)
       / \\
      ↓   ↓
      C   X
       \\?/
        
    Where C → X is controlled by true_causal_effect.
    
    Args:
        n_samples: Number of samples to generate
        confounder_effect_on_treatment: Strength of Z → C
        confounder_effect_on_outcome: Strength of Z → X
        true_causal_effect: Actual causal effect of C → X (default 0 = no effect)
        noise_treatment: Noise added to treatment
        noise_outcome: Noise added to outcome
        seed: Random seed for reproducibility
        
    Returns:
        ConfoundingResult with data and diagnostics
        
    Example:
        >>> result = simulate_basic_confounding(seed=42)
        >>> print(result.summary())
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate confounder
    Z = np.random.rand(n_samples)
    
    # Treatment is caused by confounder + noise
    C = confounder_effect_on_treatment * Z + noise_treatment * np.random.rand(n_samples)
    
    # Outcome is caused by confounder + true effect of treatment + noise
    X = (confounder_effect_on_outcome * Z + 
         true_causal_effect * C + 
         noise_outcome * np.random.rand(n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Z': Z,  # Confounder
        'C': C,  # Treatment
        'X': X,  # Outcome
    })
    
    # Compute correlations
    observed_corr, _ = stats.pearsonr(C, X)
    z_c_corr, _ = stats.pearsonr(Z, C)
    z_x_corr, _ = stats.pearsonr(Z, X)
    
    # Compute naive regression coefficient (slope from regressing X on C)
    naive_coef, _ = np.polyfit(C, X, 1)
    
    description = (
        f"Basic confounding simulation with {n_samples} samples.\n"
        f"Z causes both C (effect={confounder_effect_on_treatment}) "
        f"and X (effect={confounder_effect_on_outcome}).\n"
        f"True causal effect of C on X: {true_causal_effect}"
    )
    
    return ConfoundingResult(
        data=data,
        true_effect=true_causal_effect,
        naive_regression_coef=naive_coef,
        observed_correlation=observed_corr,
        confounder_treatment_corr=z_c_corr,
        confounder_outcome_corr=z_x_corr,
        description=description,
    )


def simulate_cell_cycle_confounding(
    n_cells: int = 1000,
    seed: Optional[int] = None,
) -> ConfoundingResult:
    """
    Simulate cell cycle confounding in gene expression.
    
    Biological scenario:
        - Z: Cell cycle phase (continuous proxy, 0=G1, 1=M)
        - C: MYC expression (high in S/G2/M)
        - X: Ribosomal gene expression (high in S/G2/M)
        
    The cell cycle drives both MYC and ribosomal genes, creating
    a spurious correlation even though MYC may not directly regulate
    ribosomal genes.
    
    Args:
        n_cells: Number of cells to simulate
        seed: Random seed
        
    Returns:
        ConfoundingResult with biological interpretation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Cell cycle phase (continuous, 0=G1, 1=M)
    # Use beta distribution to get more cells in G1 (realistic)
    cell_cycle = np.random.beta(2, 5, n_cells)
    
    # MYC expression: peaks in S/G2
    # Higher cell cycle → higher MYC
    myc_expression = (
        2.0 * cell_cycle +  # Cell cycle effect
        0.5 * np.random.randn(n_cells)  # Biological noise
    )
    myc_expression = np.clip(myc_expression, 0, None)  # Expression >= 0
    
    # Ribosomal gene expression: also peaks in S/G2
    # Cell cycle drives this, MYC has small direct effect
    true_myc_effect = 0.1  # Small direct effect
    ribosomal_expression = (
        3.0 * cell_cycle +  # Cell cycle effect (strong)
        true_myc_effect * myc_expression +  # Small MYC effect
        0.8 * np.random.randn(n_cells)  # Biological noise
    )
    ribosomal_expression = np.clip(ribosomal_expression, 0, None)
    
    data = pd.DataFrame({
        'Z': cell_cycle,
        'C': myc_expression,
        'X': ribosomal_expression,
        'cell_cycle_phase': cell_cycle,
        'MYC': myc_expression,
        'ribosomal_genes': ribosomal_expression,
    })
    
    observed_corr, _ = stats.pearsonr(myc_expression, ribosomal_expression)
    z_c_corr, _ = stats.pearsonr(cell_cycle, myc_expression)
    z_x_corr, _ = stats.pearsonr(cell_cycle, ribosomal_expression)
    
    # Compute naive regression coefficient (slope from regressing X on C)
    naive_coef, _ = np.polyfit(myc_expression, ribosomal_expression, 1)
    
    description = (
        f"Cell cycle confounding simulation with {n_cells} cells.\n"
        f"Cell cycle phase drives both MYC and ribosomal gene expression.\n"
        f"True causal effect of MYC on ribosomal genes: {true_myc_effect}\n"
        f"But naive regression coefficient is inflated due to confounding."
    )
    
    return ConfoundingResult(
        data=data,
        true_effect=true_myc_effect,
        naive_regression_coef=naive_coef,
        observed_correlation=observed_corr,
        confounder_treatment_corr=z_c_corr,
        confounder_outcome_corr=z_x_corr,
        description=description,
    )


def simulate_batch_effect_confounding(
    n_samples: int = 1000,
    n_batches: int = 3,
    seed: Optional[int] = None,
) -> ConfoundingResult:
    """
    Simulate batch effect confounding in scRNA-seq.
    
    Scenario:
        - Z: Batch (technical confounder)
        - C: Gene A expression
        - X: Gene B expression
        
    Batch effects inflate/deflate all genes together, creating
    spurious correlations between unrelated genes.
    
    Args:
        n_samples: Total number of cells
        n_batches: Number of batches
        seed: Random seed
        
    Returns:
        ConfoundingResult showing batch-driven correlation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Assign cells to batches
    batch = np.random.choice(n_batches, n_samples)
    
    # Batch effects (different capture efficiency per batch)
    batch_effects = np.random.uniform(0.5, 2.0, n_batches)
    cell_batch_effect = batch_effects[batch]
    
    # Gene A: baseline + batch effect + noise
    # Gene A and B are CAUSALLY INDEPENDENT
    gene_a_baseline = np.random.exponential(2, n_samples)
    gene_a = gene_a_baseline * cell_batch_effect + 0.5 * np.random.randn(n_samples)
    gene_a = np.clip(gene_a, 0, None)
    
    # Gene B: baseline + batch effect + noise (independent of gene A)
    gene_b_baseline = np.random.exponential(3, n_samples)
    gene_b = gene_b_baseline * cell_batch_effect + 0.5 * np.random.randn(n_samples)
    gene_b = np.clip(gene_b, 0, None)
    
    data = pd.DataFrame({
        'Z': cell_batch_effect,
        'C': gene_a,
        'X': gene_b,
        'batch': batch,
        'batch_effect': cell_batch_effect,
        'gene_A': gene_a,
        'gene_B': gene_b,
    })
    
    observed_corr, _ = stats.pearsonr(gene_a, gene_b)
    z_c_corr, _ = stats.pearsonr(cell_batch_effect, gene_a)
    z_x_corr, _ = stats.pearsonr(cell_batch_effect, gene_b)
    
    # Compute naive regression coefficient (slope from regressing X on C)
    naive_coef, _ = np.polyfit(gene_a, gene_b, 1)
    
    description = (
        f"Batch effect confounding with {n_samples} cells across {n_batches} batches.\n"
        f"Gene A and Gene B are causally INDEPENDENT.\n"
        f"But batch effects create spurious correlation and regression coefficient.\n"
        f"True causal effect: 0.0"
    )
    
    return ConfoundingResult(
        data=data,
        true_effect=0.0,  # No causal relationship
        naive_regression_coef=naive_coef,
        observed_correlation=observed_corr,
        confounder_treatment_corr=z_c_corr,
        confounder_outcome_corr=z_x_corr,
        description=description,
    )


def simulate_disease_severity_confounding(
    n_patients: int = 500,
    seed: Optional[int] = None,
) -> ConfoundingResult:
    """
    Simulate disease severity as a confounder.
    
    Scenario:
        - Z: Disease severity (underlying pathology)
        - C: Stress response gene (e.g., HIF1A)
        - X: Cell death rate
        
    Severe disease activates stress responses AND causes cell death.
    The stress response may actually be protective, but it correlates
    with death because both are driven by severity.
    
    Args:
        n_patients: Number of patients/samples
        seed: Random seed
        
    Returns:
        ConfoundingResult with clinical interpretation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Disease severity (0 = healthy, 1 = severe)
    severity = np.random.beta(2, 3, n_patients)
    
    # Stress response gene (activated by severity)
    # HIF1A is actually PROTECTIVE (negative true effect on death)
    stress_gene = (
        2.0 * severity +  # Severity activates stress response
        0.3 * np.random.randn(n_patients)
    )
    stress_gene = np.clip(stress_gene, 0, None)
    
    # Cell death: driven by severity, REDUCED by stress response
    true_protective_effect = -0.3  # Stress response is protective!
    cell_death = (
        3.0 * severity +  # Severity causes death
        true_protective_effect * stress_gene +  # Stress response protects
        0.5 * np.random.randn(n_patients)
    )
    cell_death = np.clip(cell_death, 0, None)
    
    data = pd.DataFrame({
        'Z': severity,
        'C': stress_gene,
        'X': cell_death,
        'disease_severity': severity,
        'HIF1A': stress_gene,
        'cell_death_rate': cell_death,
    })
    
    observed_corr, _ = stats.pearsonr(stress_gene, cell_death)
    z_c_corr, _ = stats.pearsonr(severity, stress_gene)
    z_x_corr, _ = stats.pearsonr(severity, cell_death)
    
    # Compute naive regression coefficient (slope from regressing X on C)
    naive_coef, _ = np.polyfit(stress_gene, cell_death, 1)
    
    description = (
        f"Disease severity confounding with {n_patients} patients.\n"
        f"Severity drives both stress response (HIF1A) and cell death.\n"
        f"True causal effect of HIF1A on death: {true_protective_effect} (PROTECTIVE!)\n"
        f"But naive regression coefficient is POSITIVE due to confounding.\n"
        f"Inhibiting HIF1A would likely INCREASE death, not decrease it."
    )
    
    return ConfoundingResult(
        data=data,
        true_effect=true_protective_effect,
        naive_regression_coef=naive_coef,
        observed_correlation=observed_corr,
        confounder_treatment_corr=z_c_corr,
        confounder_outcome_corr=z_x_corr,
        description=description,
    )


def simulate_treatment_effect_with_confounding(
    n_samples: int = 1000,
    true_ate: float = 10.0,
    confounder_treatment_effect: float = -0.5,
    confounder_outcome_effect: float = 50.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Simulate treatment effect estimation with confounding.
    
    Based on the earnings example from Molak's book:
        - Age (confounder) affects both treatment probability and outcome
        - Younger people more likely to take training
        - Older people have higher baseline earnings
        
    This creates bias in naive ATE estimation.
    
    Args:
        n_samples: Number of samples
        true_ate: True average treatment effect
        confounder_treatment_effect: How confounder affects treatment probability
        confounder_outcome_effect: How confounder affects outcome
        seed: Random seed
        
    Returns:
        Tuple of (DataFrame, true_ate)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Age as confounder (scaled 0-1)
    age = stats.halfnorm.rvs(loc=0, scale=0.3, size=n_samples)
    age = np.clip(age, 0, 1)
    
    # Treatment probability decreases with age
    treatment_prob = 1 / (1 + np.exp(5 * (age - 0.3)))  # Sigmoid
    treatment = np.random.binomial(1, treatment_prob)
    
    # Outcome: baseline + age effect + treatment effect + noise
    baseline = 50000
    outcome = (
        baseline +
        confounder_outcome_effect * 1000 * age +  # Age increases earnings
        true_ate * 1000 * treatment +  # True treatment effect
        np.random.randn(n_samples) * 5000  # Noise
    )
    
    data = pd.DataFrame({
        'age': age,
        'treatment': treatment,
        'outcome': outcome,
    })
    
    return data, true_ate * 1000


def compare_naive_vs_adjusted(
    data: pd.DataFrame,
    treatment_col: str = 'treatment',
    outcome_col: str = 'outcome',
    confounder_col: str = 'age',
) -> Dict[str, float]:
    """
    Compare naive ATE estimate vs adjusted estimate.
    
    Args:
        data: DataFrame with treatment, outcome, and confounder
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_col: Name of confounder column
        
    Returns:
        Dictionary with naive and adjusted estimates
    """
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    
    # Naive estimate (biased)
    naive_ate = treated[outcome_col].mean() - control[outcome_col].mean()
    
    # Simple adjusted estimate using regression
    from sklearn.linear_model import LinearRegression
    
    X = data[[confounder_col, treatment_col]].values
    y = data[outcome_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Treatment coefficient is adjusted ATE
    adjusted_ate = model.coef_[1]  # Coefficient for treatment
    
    return {
        'naive_ate': naive_ate,
        'adjusted_ate': adjusted_ate,
        'treated_mean': treated[outcome_col].mean(),
        'control_mean': control[outcome_col].mean(),
        'treated_confounder_mean': treated[confounder_col].mean(),
        'control_confounder_mean': control[confounder_col].mean(),
    }


# Visualization utilities
def plot_confounding_scatter(
    result: ConfoundingResult,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot pairwise relationships showing confounding structure.
    
    Creates three scatter plots:
        1. Z vs C (confounder → treatment)
        2. Z vs X (confounder → outcome)
        3. C vs X (observed correlation, potentially spurious)
    """
    import matplotlib.pyplot as plt
    
    data = result.data
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Z vs C
    axes[0].scatter(data['Z'], data['C'], alpha=0.5, c='steelblue')
    axes[0].set_xlabel('Z (Confounder)')
    axes[0].set_ylabel('C (Treatment)')
    axes[0].set_title(f'Z → C (r={result.confounder_treatment_corr:.3f})')
    
    # Z vs X
    axes[1].scatter(data['Z'], data['X'], alpha=0.5, c='steelblue')
    axes[1].set_xlabel('Z (Confounder)')
    axes[1].set_ylabel('X (Outcome)')
    axes[1].set_title(f'Z → X (r={result.confounder_outcome_corr:.3f})')
    
    # C vs X (the spurious correlation)
    axes[2].scatter(data['C'], data['X'], alpha=0.5, c='coral')
    axes[2].set_xlabel('C (Treatment)')
    axes[2].set_ylabel('X (Outcome)')
    axes[2].set_title(f'C vs X (r={result.observed_correlation:.3f})\nTrue effect: {result.true_effect:.3f}')
    
    plt.suptitle('Confounding Structure', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_confounding_dag() -> None:
    """Plot the basic confounding DAG using networkx."""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    G = nx.DiGraph()
    G.add_edges_from([('Z', 'C'), ('Z', 'X'), ('C', 'X')])
    
    pos = {
        'Z': (0.5, 1),
        'C': (0, 0),
        'X': (1, 0),
    }
    
    plt.figure(figsize=(6, 4))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    
    # Draw edges with different colors
    nx.draw_networkx_edges(G, pos, edgelist=[('Z', 'C'), ('Z', 'X')],
                           edge_color='steelblue', width=2, arrows=True,
                           arrowsize=20, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, edgelist=[('C', 'X')],
                           edge_color='coral', width=2, style='dashed',
                           arrows=True, arrowsize=20)
    
    plt.title('Confounding DAG\nBlue: confounding paths, Orange: causal path (may be 0)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
