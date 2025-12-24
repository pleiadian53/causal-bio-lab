"""
Decision layer for drug target prioritization.

This module provides tools for ranking and selecting drug targets based on
causal effect estimates, moving from model outputs to actionable decisions.

Includes:
- Target ranking based on effect size, robustness, and specificity
- Uncertainty calibration for decision-making
- Go/no-go decision rules
"""

from causalbiolab.decision.target_ranking import (
    TargetScore,
    TargetRanker,
    rank_targets,
    apply_go_nogo_rules,
)

__all__ = [
    "TargetScore",
    "TargetRanker",
    "rank_targets",
    "apply_go_nogo_rules",
]
