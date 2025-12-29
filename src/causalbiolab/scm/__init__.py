"""
Structural Causal Models (SCMs) module.

This module provides tools for defining, simulating, and reasoning with
structural causal models, including:
- Observational sampling
- Interventions (do-operator)
- Counterfactual reasoning (abduction-action-prediction)
"""

from .base import StructuralCausalModel
from .interventions import intervene
from .counterfactuals import compute_counterfactual

__all__ = [
    'StructuralCausalModel',
    'intervene',
    'compute_counterfactual',
]
