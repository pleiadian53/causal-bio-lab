"""
Basic SCM Examples: Three Levels of Causation

This script demonstrates the three levels of Pearl's causal hierarchy:
1. Association (seeing)
2. Intervention (doing)
3. Counterfactual (imagining)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('../../src')

from causalbiolab.scm.examples import simple_linear_scm, confounded_scm
from causalbiolab.scm.counterfactuals import LinearSCM


def demonstrate_three_levels():
    """Demonstrate the three levels of causation."""
    print("=" * 60)
    print("Three Levels of Causation")
    print("=" * 60)
    
    # Create simple linear SCM: X -> Y with Y = 2X + U_Y
    scm = simple_linear_scm()
    
    # Level 1: Association (Observational)
    print("\n1. ASSOCIATION (Seeing) - P(Y | X)")
    print("-" * 60)
    data_obs = scm.sample(n_samples=1000, random_seed=42)
    df_obs = pd.DataFrame(data_obs)
    
    # Conditional expectation
    high_x = df_obs[df_obs['X'] > 0]
    low_x = df_obs[df_obs['X'] <= 0]
    
    print(f"E[Y | X > 0] = {high_x['Y'].mean():.3f}")
    print(f"E[Y | X ≤ 0] = {low_x['Y'].mean():.3f}")
    print(f"Observational difference = {high_x['Y'].mean() - low_x['Y'].mean():.3f}")
    print(f"Correlation(X, Y) = {df_obs['X'].corr(df_obs['Y']):.3f}")
    
    # Level 2: Intervention (Causal)
    print("\n2. INTERVENTION (Doing) - P(Y | do(X))")
    print("-" * 60)
    scm_do_x1 = scm.intervene({'X': 1.0})
    scm_do_x0 = scm.intervene({'X': 0.0})
    
    data_do_x1 = scm_do_x1.sample(n_samples=1000, random_seed=42)
    data_do_x0 = scm_do_x0.sample(n_samples=1000, random_seed=42)
    
    y_do_x1 = pd.DataFrame(data_do_x1)['Y'].mean()
    y_do_x0 = pd.DataFrame(data_do_x0)['Y'].mean()
    
    print(f"E[Y | do(X=1)] = {y_do_x1:.3f}")
    print(f"E[Y | do(X=0)] = {y_do_x0:.3f}")
    print(f"Causal effect = {y_do_x1 - y_do_x0:.3f}")
    print(f"True causal effect (from equation Y = 2X + U_Y) = 2.0")
    
    # Level 3: Counterfactual (Individual)
    print("\n3. COUNTERFACTUAL (Imagining) - Y_x for individual")
    print("-" * 60)
    
    # Use LinearSCM for efficient counterfactual computation
    scm_linear = LinearSCM(
        coefficients={'Y': {'X': 2.0}},
        noise_distributions={'X': stats.norm(0, 1), 'Y': stats.norm(0, 0.5)}
    )
    
    # Individual with X=1, Y=3
    observed = {'X': 1.0, 'Y': 3.0}
    
    # Abduction: infer U_Y
    exogenous = scm_linear.abduct(observed)
    print(f"Observed: X={observed['X']}, Y={observed['Y']}")
    print(f"Inferred U_Y = Y - 2*X = {observed['Y']} - 2*{observed['X']} = {exogenous['Y']:.3f}")
    
    # Counterfactual: what if X had been 2?
    y_cf = scm_linear.counterfactual(
        observed=observed,
        intervention={'X': 2.0},
        query='Y'
    )
    
    print(f"\nCounterfactual: Y_{{X=2}} = 2*2 + U_Y = 4 + {exogenous['Y']:.3f} = {y_cf:.3f}")
    print(f"Interpretation: If this person had X=2 instead of X=1,")
    print(f"                their Y would be {y_cf:.3f} instead of {observed['Y']:.3f}")
    
    print("\n" + "=" * 60)
    print("Key Insight: Each level requires increasingly strong assumptions")
    print("=" * 60)
    print("Level 1 (Association):    Data only")
    print("Level 2 (Intervention):   Data + DAG structure")
    print("Level 3 (Counterfactual): Data + Full SCM (structural equations)")


def demonstrate_confounding():
    """Demonstrate why P(Y|X) ≠ P(Y|do(X)) with confounding."""
    print("\n\n" + "=" * 60)
    print("Confounding: Why Observational ≠ Causal")
    print("=" * 60)
    
    # Create confounded SCM: Z -> X, Z -> Y
    scm_conf = confounded_scm()
    
    # Observational data
    data_obs = scm_conf.sample(n_samples=2000, random_seed=42)
    df_obs = pd.DataFrame(data_obs)
    
    # Observational effect (biased)
    high_x = df_obs[df_obs['X'] > df_obs['X'].median()]
    low_x = df_obs[df_obs['X'] <= df_obs['X'].median()]
    obs_effect = high_x['Y'].mean() - low_x['Y'].mean()
    
    print("\nObservational Analysis (BIASED by confounder Z):")
    print(f"E[Y | X > median] = {high_x['Y'].mean():.3f}")
    print(f"E[Y | X ≤ median] = {low_x['Y'].mean():.3f}")
    print(f"Observational effect = {obs_effect:.3f}")
    
    # Interventional effect (unbiased)
    x_high = df_obs['X'].quantile(0.75)
    x_low = df_obs['X'].quantile(0.25)
    
    scm_do_high = scm_conf.intervene({'X': x_high})
    scm_do_low = scm_conf.intervene({'X': x_low})
    
    data_do_high = scm_do_high.sample(n_samples=1000, random_seed=42)
    data_do_low = scm_do_low.sample(n_samples=1000, random_seed=42)
    
    causal_effect = pd.DataFrame(data_do_high)['Y'].mean() - pd.DataFrame(data_do_low)['Y'].mean()
    
    print("\nInterventional Analysis (UNBIASED):")
    print(f"E[Y | do(X={x_high:.2f})] = {pd.DataFrame(data_do_high)['Y'].mean():.3f}")
    print(f"E[Y | do(X={x_low:.2f})] = {pd.DataFrame(data_do_low)['Y'].mean():.3f}")
    print(f"Causal effect = {causal_effect:.3f}")
    
    print(f"\nBias = {obs_effect - causal_effect:.3f}")
    print(f"True causal coefficient (from Y = 2X + Z + U_Y) = 2.0")
    print(f"Estimated from intervention = {causal_effect / (x_high - x_low):.3f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_three_levels()
    demonstrate_confounding()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("1. SCMs formalize the data-generating process")
    print("2. Interventions break incoming causal arrows")
    print("3. Counterfactuals require individual-specific noise terms")
    print("4. Confounding creates bias in observational estimates")
    print("5. Interventions (or adjustment) remove confounding bias")
    print("=" * 60)
