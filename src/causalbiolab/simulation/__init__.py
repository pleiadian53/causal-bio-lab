"""
Simulation module for causal inference education and benchmarking.

This module provides data generating processes (DGPs) for:
- Confounding demonstrations
- Treatment effect estimation
- Causal discovery benchmarks

Inspired by examples from "Causal Inference and Discovery in Python" by Aleksander Molak.
"""

from causalbiolab.simulation.confounding import (
    ConfoundingResult,
    simulate_basic_confounding,
    simulate_cell_cycle_confounding,
    simulate_batch_effect_confounding,
    simulate_disease_severity_confounding,
    simulate_treatment_effect_with_confounding,
    compare_naive_vs_adjusted,
    plot_confounding_scatter,
    plot_confounding_dag,
)

__all__ = [
    # Result classes
    "ConfoundingResult",
    # Basic simulations
    "simulate_basic_confounding",
    # Biological confounding examples
    "simulate_cell_cycle_confounding",
    "simulate_batch_effect_confounding", 
    "simulate_disease_severity_confounding",
    # Treatment effect with confounding
    "simulate_treatment_effect_with_confounding",
    "compare_naive_vs_adjusted",
    # Visualization
    "plot_confounding_scatter",
    "plot_confounding_dag",
]
