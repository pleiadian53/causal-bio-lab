"""
Counterfactual reasoning utilities for SCMs.

This module implements Pearl's three-step counterfactual computation:
1. Abduction: Infer exogenous variables U from observed data
2. Action: Apply intervention to create mutilated SCM
3. Prediction: Compute query variable under intervention with inferred U
"""

from typing import Dict, Any, Optional, Callable
import numpy as np


def compute_counterfactual(
    scm: 'StructuralCausalModel',
    observed: Dict[str, float],
    intervention: Dict[str, Any],
    query: str
) -> float:
    """
    Compute counterfactual query using abduction-action-prediction.
    
    This is a convenience wrapper around SCM.counterfactual().
    
    Parameters
    ----------
    scm : StructuralCausalModel
        The structural causal model
    observed : Dict[str, float]
        Observed values in the actual world
    intervention : Dict[str, Any]
        Intervention to apply in the counterfactual world
    query : str
        Variable to query in the counterfactual world
    
    Returns
    -------
    float
        Counterfactual value of the query variable
    
    Examples
    --------
    >>> # "What would Y be if we had set X=2, given we observed X=1, Y=3?"
    >>> y_cf = compute_counterfactual(
    ...     scm,
    ...     observed={'X': 1, 'Y': 3},
    ...     intervention={'X': 2},
    ...     query='Y'
    ... )
    """
    return scm.counterfactual(observed, intervention, query)


class LinearSCM:
    """
    Specialized SCM for linear structural equations.
    
    Linear SCMs have the form:
        X_i = sum_j(a_ij * X_j) + U_i
    
    This allows for efficient abduction (solving for U) and
    counterfactual computation.
    
    Parameters
    ----------
    coefficients : Dict[str, Dict[str, float]]
        Nested dict: coefficients[child][parent] = coefficient
    noise_distributions : Dict[str, Any]
        Distribution for each exogenous variable
    
    Examples
    --------
    >>> # X -> Y with Y = 2*X + U_Y
    >>> scm = LinearSCM(
    ...     coefficients={'Y': {'X': 2.0}},
    ...     noise_distributions={'X': stats.norm(0, 1), 'Y': stats.norm(0, 0.5)}
    ... )
    """
    
    def __init__(
        self,
        coefficients: Dict[str, Dict[str, float]],
        noise_distributions: Dict[str, Any]
    ):
        self.coefficients = coefficients
        self.noise_distributions = noise_distributions
        self.variables = self._get_variables()
        self.topological_order = self._infer_order()
    
    def _get_variables(self):
        """Get all variables (roots and children)."""
        variables = set(self.noise_distributions.keys())
        for child in self.coefficients:
            variables.add(child)
        return sorted(variables)
    
    def _infer_order(self):
        """Infer topological order."""
        # Variables with no parents come first
        roots = [v for v in self.variables if v not in self.coefficients]
        children = [v for v in self.variables if v in self.coefficients]
        return roots + children
    
    def sample(self, n_samples: int = 1000, random_seed: Optional[int] = None):
        """Generate samples from the linear SCM."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        samples = {}
        
        for var in self.topological_order:
            # Sample noise
            u = self.noise_distributions[var].rvs(n_samples)
            
            # Compute linear combination of parents
            if var in self.coefficients:
                value = np.zeros(n_samples)
                for parent, coef in self.coefficients[var].items():
                    value += coef * samples[parent]
                samples[var] = value + u
            else:
                # Root variable
                samples[var] = u
        
        return samples
    
    def abduct(self, observed: Dict[str, float]) -> Dict[str, float]:
        """
        Infer exogenous variables from observations.
        
        For linear SCMs, this is straightforward:
            U_i = X_i - sum_j(a_ij * X_j)
        """
        exogenous = {}
        
        for var in self.topological_order:
            if var not in observed:
                continue
            
            if var in self.coefficients:
                # Compute residual
                predicted = 0.0
                for parent, coef in self.coefficients[var].items():
                    if parent in observed:
                        predicted += coef * observed[parent]
                    else:
                        raise ValueError(f"Parent {parent} not observed")
                exogenous[var] = observed[var] - predicted
            else:
                # Root variable
                exogenous[var] = observed[var]
        
        return exogenous
    
    def intervene(self, interventions: Dict[str, float]) -> 'LinearSCM':
        """Apply interventions to create mutilated SCM."""
        new_coefficients = {}
        
        for child, parents in self.coefficients.items():
            if child not in interventions:
                # Keep original equation
                new_coefficients[child] = parents.copy()
        
        # Intervened variables become roots (no parents)
        # They will be set to constant values in prediction
        
        return LinearSCM(new_coefficients, self.noise_distributions)
    
    def predict_with_exogenous(
        self,
        exogenous: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict all variables given exogenous values and interventions.
        """
        values = {}
        
        for var in self.topological_order:
            if var in interventions:
                # Intervened variable
                values[var] = interventions[var]
            elif var in self.coefficients:
                # Compute from parents and exogenous
                value = exogenous.get(var, 0.0)
                for parent, coef in self.coefficients[var].items():
                    value += coef * values[parent]
                values[var] = value
            else:
                # Root variable
                values[var] = exogenous.get(var, 0.0)
        
        return values
    
    def counterfactual(
        self,
        observed: Dict[str, float],
        intervention: Dict[str, float],
        query: str
    ) -> float:
        """
        Compute counterfactual using abduction-action-prediction.
        
        Examples
        --------
        >>> # Observed: X=1, Y=3. Counterfactual: what if X=2?
        >>> y_cf = scm.counterfactual({'X': 1, 'Y': 3}, {'X': 2}, 'Y')
        """
        # Step 1: Abduction
        exogenous = self.abduct(observed)
        
        # Step 2: Action (implicit in predict_with_exogenous)
        # Step 3: Prediction
        values = self.predict_with_exogenous(exogenous, intervention)
        
        return values[query]


def effect_of_treatment_on_treated(
    scm: 'StructuralCausalModel',
    treatment_var: str,
    outcome_var: str,
    observed_data: Dict[str, np.ndarray],
    treatment_value: float = 1.0,
    control_value: float = 0.0
) -> float:
    """
    Compute Effect of Treatment on the Treated (ETT) using counterfactuals.
    
    ETT = E[Y(1) - Y(0) | T=1]
    
    This requires counterfactual reasoning: for each treated unit,
    compute what their outcome would have been under control.
    
    Parameters
    ----------
    scm : StructuralCausalModel
        The structural causal model
    treatment_var : str
        Name of treatment variable
    outcome_var : str
        Name of outcome variable
    observed_data : Dict[str, np.ndarray]
        Observed data for all variables
    treatment_value : float
        Value indicating treatment
    control_value : float
        Value indicating control
    
    Returns
    -------
    float
        Estimated ETT
    
    Examples
    --------
    >>> data = scm.sample(1000)
    >>> ett = effect_of_treatment_on_treated(scm, 'T', 'Y', data)
    """
    # Find treated units
    treated_mask = observed_data[treatment_var] == treatment_value
    n_treated = treated_mask.sum()
    
    if n_treated == 0:
        raise ValueError("No treated units found")
    
    # For each treated unit, compute counterfactual under control
    counterfactual_outcomes = []
    
    for i in np.where(treated_mask)[0]:
        # Get observed values for this unit
        observed = {var: data[i] for var, data in observed_data.items()}
        
        # Compute counterfactual: what if they were not treated?
        y_cf = scm.counterfactual(
            observed=observed,
            intervention={treatment_var: control_value},
            query=outcome_var
        )
        
        counterfactual_outcomes.append(y_cf)
    
    # ETT = average difference between factual and counterfactual
    factual_outcomes = observed_data[outcome_var][treated_mask]
    ett = np.mean(factual_outcomes - np.array(counterfactual_outcomes))
    
    return ett
