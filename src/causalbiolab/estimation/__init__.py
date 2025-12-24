"""
Treatment effect estimation methods.

Includes:
- Average Treatment Effect (ATE)
- Individual Treatment Effect (ITE)
- Conditional Average Treatment Effect (CATE)
- Propensity score methods
- Doubly robust estimators
"""

from causalbiolab.estimation.ate import (
    ATEEstimator,
    ATEResult,
    NaiveATE,
    PropensityATE,
    DoublyRobustATE,
    compare_estimators,
    print_comparison,
)

from causalbiolab.estimation.outcomes import (
    LIVER_PATHWAYS,
    compute_pathway_score,
    compute_module_score,
    compute_perturbation_effect,
    define_treatment_outcome,
)

from causalbiolab.estimation.cate import (
    CATEEstimator,
    CATEResult,
    SLearner,
    TLearner,
    XLearner,
    DoublyRobustCATE,
    compare_cate_estimators,
    print_cate_comparison,
)

__all__ = [
    # ATE estimators
    "ATEEstimator",
    "ATEResult",
    "NaiveATE",
    "PropensityATE",
    "DoublyRobustATE",
    "compare_estimators",
    "print_comparison",
    # CATE estimators
    "CATEEstimator",
    "CATEResult",
    "SLearner",
    "TLearner",
    "XLearner",
    "DoublyRobustCATE",
    "compare_cate_estimators",
    "print_cate_comparison",
    # Outcome definitions
    "LIVER_PATHWAYS",
    "compute_pathway_score",
    "compute_module_score",
    "compute_perturbation_effect",
    "define_treatment_outcome",
]
