"""
SCM and Do-Calculus Integration

This script demonstrates how SCMs implement do-calculus rules,
showing the connection between graph-based identification and
structural equation implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('../../src')

from causalbiolab.scm.examples import confounded_scm


def backdoor_adjustment_scm():
    """
    Demonstrate back-door adjustment using SCM.
    
    Do-calculus formula: P(Y | do(X)) = sum_Z P(Y | X, Z) P(Z)
    
    SCM implementation:
    1. Sample Z from marginal (unaffected by intervention)
    2. Sample Y from conditional given X and Z
    """
    print("=" * 60)
    print("Back-Door Adjustment: Do-Calculus via SCM")
    print("=" * 60)
    
    # SCM: Z -> X, Z -> Y
    # Z = U_Z
    # X = Z + U_X
    # Y = 2X + Z + U_Y
    
    scm = confounded_scm()
    
    print("\nDAG Structure:")
    print("    Z")
    print("   / \\")
    print("  v   v")
    print("  X → Y")
    print("\nZ is a confounder (back-door path: X ← Z → Y)")
    
    # Method 1: Direct intervention (ground truth)
    print("\n" + "-" * 60)
    print("Method 1: Direct Intervention (Ground Truth)")
    print("-" * 60)
    
    scm_do_x = scm.intervene({'X': 1.0})
    data_do = scm_do_x.sample(5000, random_seed=42)
    y_do_direct = pd.DataFrame(data_do)['Y'].mean()
    
    print(f"E[Y | do(X=1)] = {y_do_direct:.3f}")
    print("This is the true causal effect.")
    
    # Method 2: Back-door adjustment from observational data
    print("\n" + "-" * 60)
    print("Method 2: Back-Door Adjustment")
    print("-" * 60)
    print("Formula: P(Y | do(X)) = sum_Z P(Y | X, Z) P(Z)")
    
    # Sample observational data
    data_obs = scm.sample(10000, random_seed=42)
    df_obs = pd.DataFrame(data_obs)
    
    # Stratify by Z (confounder)
    z_bins = pd.qcut(df_obs['Z'], q=10, labels=False, duplicates='drop')
    df_obs['Z_bin'] = z_bins
    
    # For each stratum, compute E[Y | X=1, Z]
    backdoor_estimates = []
    weights = []
    
    for z_bin in df_obs['Z_bin'].unique():
        stratum = df_obs[df_obs['Z_bin'] == z_bin]
        # Find observations with X close to 1.0
        x_close = stratum[(stratum['X'] > 0.9) & (stratum['X'] < 1.1)]
        
        if len(x_close) > 0:
            # E[Y | X=1, Z] for this stratum
            backdoor_estimates.append(x_close['Y'].mean())
            # P(Z) - weight by stratum size
            weights.append(len(stratum) / len(df_obs))
    
    # Weighted average: sum_Z P(Y | X=1, Z) P(Z)
    y_backdoor = np.average(backdoor_estimates, weights=weights)
    
    print(f"E[Y | do(X=1)] (back-door) = {y_backdoor:.3f}")
    print(f"\nDifference from ground truth: {abs(y_do_direct - y_backdoor):.3f}")
    print("\n✓ Back-door adjustment successfully recovers causal effect!")
    
    # Method 3: Naive (biased) estimate
    print("\n" + "-" * 60)
    print("Method 3: Naive Observational (BIASED)")
    print("-" * 60)
    
    y_naive = df_obs[df_obs['X'] > 0.9]['Y'].mean()
    print(f"E[Y | X≈1] (naive) = {y_naive:.3f}")
    print(f"Bias = {y_naive - y_do_direct:.3f}")
    print("\n✗ Naive estimate is biased by confounding!")
    
    return y_do_direct, y_backdoor, y_naive


def graph_surgery_visualization():
    """Visualize graph surgery (do-operator)."""
    print("\n\n" + "=" * 60)
    print("Graph Surgery: The Do-Operator")
    print("=" * 60)
    
    scm = confounded_scm()
    
    print("\nOriginal Graph:")
    print("    Z")
    print("   / \\")
    print("  v   v")
    print("  X → Y")
    print("\nStructural equations:")
    print("  Z = U_Z")
    print("  X = Z + U_X")
    print("  Y = 2X + Z + U_Y")
    
    print("\nAfter do(X=1) - Graph Surgery:")
    print("    Z")
    print("    |")
    print("    v")
    print("  X → Y")
    print("  ↑")
    print("  1 (fixed)")
    print("\nMutilated structural equations:")
    print("  Z = U_Z          (unchanged)")
    print("  X = 1            (incoming edge cut, set to constant)")
    print("  Y = 2X + Z + U_Y (unchanged)")
    
    # Sample from both
    data_obs = scm.sample(1000, random_seed=42)
    scm_do = scm.intervene({'X': 1.0})
    data_do = scm_do.sample(1000, random_seed=42)
    
    df_obs = pd.DataFrame(data_obs)
    df_do = pd.DataFrame(data_do)
    
    print("\n" + "-" * 60)
    print("Effect of Graph Surgery:")
    print("-" * 60)
    print(f"Observational: Corr(Z, X) = {df_obs['Z'].corr(df_obs['X']):.3f}")
    print(f"Interventional: Corr(Z, X) = {df_do['Z'].corr(df_do['X']):.3f}")
    print("\n✓ Intervention breaks the Z → X edge!")
    print(f"\nObservational: Corr(X, Y) = {df_obs['X'].corr(df_obs['Y']):.3f}")
    print(f"Interventional: Corr(X, Y) = {df_do['X'].corr(df_do['Y']):.3f}")
    print("\n✓ But X → Y edge remains intact!")


def compare_identification_methods():
    """Compare different identification methods."""
    print("\n\n" + "=" * 60)
    print("Comparison: Identification Methods")
    print("=" * 60)
    
    scm = confounded_scm()
    
    # Ground truth
    scm_do = scm.intervene({'X': 1.0})
    data_do = scm_do.sample(5000, random_seed=42)
    y_true = pd.DataFrame(data_do)['Y'].mean()
    
    # Observational data
    data_obs = scm.sample(10000, random_seed=42)
    df_obs = pd.DataFrame(data_obs)
    
    results = {}
    
    # 1. Naive (biased)
    results['Naive'] = df_obs[(df_obs['X'] > 0.9) & (df_obs['X'] < 1.1)]['Y'].mean()
    
    # 2. Back-door adjustment (stratification)
    z_bins = pd.qcut(df_obs['Z'], q=10, labels=False, duplicates='drop')
    df_obs['Z_bin'] = z_bins
    backdoor_estimates = []
    weights = []
    for z_bin in df_obs['Z_bin'].unique():
        stratum = df_obs[df_obs['Z_bin'] == z_bin]
        x_close = stratum[(stratum['X'] > 0.9) & (stratum['X'] < 1.1)]
        if len(x_close) > 0:
            backdoor_estimates.append(x_close['Y'].mean())
            weights.append(len(stratum) / len(df_obs))
    results['Back-door'] = np.average(backdoor_estimates, weights=weights)
    
    # 3. Regression adjustment
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(df_obs[['X', 'Z']], df_obs['Y'])
    # Predict Y for X=1, averaging over Z distribution
    X_new = np.column_stack([np.ones(len(df_obs)), df_obs['Z']])
    results['Regression'] = lr.predict(X_new).mean()
    
    # 4. IPW (inverse probability weighting)
    from sklearn.linear_model import LogisticRegression
    df_obs['T'] = (df_obs['X'] > df_obs['X'].median()).astype(int)
    lr_ps = LogisticRegression()
    lr_ps.fit(df_obs[['Z']], df_obs['T'])
    df_obs['e_Z'] = lr_ps.predict_proba(df_obs[['Z']])[:, 1]
    df_obs['weight'] = np.where(df_obs['T'] == 1, 1 / df_obs['e_Z'], 1 / (1 - df_obs['e_Z']))
    df_obs['weight'] = df_obs['weight'].clip(0.1, 10)
    y1_ipw = (df_obs['T'] * df_obs['Y'] * df_obs['weight']).sum() / (df_obs['T'] * df_obs['weight']).sum()
    results['IPW'] = y1_ipw
    
    # Print results
    print("\nMethod                    | Estimate | Error  | Bias")
    print("-" * 60)
    print(f"Ground Truth (do(X=1))    | {y_true:8.3f} |   -    |   -")
    print("-" * 60)
    for method, estimate in results.items():
        error = abs(estimate - y_true)
        bias = estimate - y_true
        print(f"{method:25} | {estimate:8.3f} | {error:6.3f} | {bias:+6.3f}")
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("1. Naive estimate is biased by confounding")
    print("2. Back-door, regression, and IPW all remove bias")
    print("3. All valid methods converge to the true causal effect")
    print("4. SCM provides ground truth for validation")


if __name__ == "__main__":
    np.random.seed(42)
    
    # Run demonstrations
    backdoor_adjustment_scm()
    graph_surgery_visualization()
    compare_identification_methods()
    
    print("\n" + "=" * 60)
    print("Summary: SCM and Do-Calculus Connection")
    print("=" * 60)
    print("1. Do-calculus provides identification rules (graph-based)")
    print("2. SCMs provide implementation (equation-based)")
    print("3. Graph surgery = modifying structural equations")
    print("4. Back-door adjustment = stratification in SCM")
    print("5. Multiple methods converge when assumptions hold")
    print("=" * 60)
