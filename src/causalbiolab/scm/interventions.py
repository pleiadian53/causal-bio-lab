"""
Intervention utilities for SCMs.

This module provides helper functions for applying interventions
(do-operator) to structural causal models.
"""

from typing import Dict, Any
import numpy as np


def intervene(scm: 'StructuralCausalModel', **interventions) -> 'StructuralCausalModel':
    """
    Convenience function to apply interventions to an SCM.
    
    Parameters
    ----------
    scm : StructuralCausalModel
        The SCM to intervene on
    **interventions
        Keyword arguments specifying interventions (variable=value)
    
    Returns
    -------
    StructuralCausalModel
        New SCM with interventions applied
    
    Examples
    --------
    >>> scm_do_x = intervene(scm, X=1.5)
    >>> scm_do_xy = intervene(scm, X=1.5, Y=2.0)
    """
    return scm.intervene(interventions)


def soft_intervention(
    scm: 'StructuralCausalModel',
    variable: str,
    noise_scale: float = 0.1
) -> 'StructuralCausalModel':
    """
    Apply a "soft" intervention that adds noise rather than fixing a value.
    
    Soft interventions are useful for modeling imperfect interventions
    or stochastic policies.
    
    Parameters
    ----------
    scm : StructuralCausalModel
        The SCM to intervene on
    variable : str
        Variable to apply soft intervention to
    noise_scale : float
        Scale of additional noise to add
    
    Returns
    -------
    StructuralCausalModel
        New SCM with soft intervention applied
    
    Examples
    --------
    >>> # Add noise to X instead of fixing it
    >>> scm_soft = soft_intervention(scm, 'X', noise_scale=0.5)
    """
    from scipy import stats
    from .base import SCMVariable
    
    var = scm.variables[variable]
    
    # Create new equation that adds noise
    def soft_equation(*args):
        original_value = var.equation(*args)
        noise = stats.norm(0, noise_scale).rvs(len(original_value) if hasattr(original_value, '__len__') else 1)
        return original_value + noise
    
    new_variables = dict(scm.variables)
    new_variables[variable] = SCMVariable(
        name=variable,
        equation=soft_equation,
        parents=var.parents,
        noise_dist=var.noise_dist
    )
    
    from .base import StructuralCausalModel
    return StructuralCausalModel(new_variables, scm.topological_order)


def conditional_intervention(
    scm: 'StructuralCausalModel',
    variable: str,
    condition: callable,
    intervention_value: Any
) -> 'StructuralCausalModel':
    """
    Apply intervention only when a condition is met.
    
    This is useful for modeling treatment assignment rules or
    conditional policies.
    
    Parameters
    ----------
    scm : StructuralCausalModel
        The SCM to intervene on
    variable : str
        Variable to potentially intervene on
    condition : callable
        Function that takes parent values and returns boolean
    intervention_value : Any
        Value to set when condition is True
    
    Returns
    -------
    StructuralCausalModel
        New SCM with conditional intervention
    
    Examples
    --------
    >>> # Intervene on treatment only if severity > 0.5
    >>> scm_cond = conditional_intervention(
    ...     scm,
    ...     'Treatment',
    ...     lambda severity: severity > 0.5,
    ...     intervention_value=1.0
    ... )
    """
    from .base import SCMVariable
    
    var = scm.variables[variable]
    
    def conditional_equation(*args):
        # Separate parent values from noise
        parent_values = args[:-1]
        u = args[-1]
        
        # Check condition
        if condition(*parent_values):
            return np.full_like(u, intervention_value)
        else:
            return var.equation(*args)
    
    new_variables = dict(scm.variables)
    new_variables[variable] = SCMVariable(
        name=variable,
        equation=conditional_equation,
        parents=var.parents,
        noise_dist=var.noise_dist
    )
    
    from .base import StructuralCausalModel
    return StructuralCausalModel(new_variables, scm.topological_order)
