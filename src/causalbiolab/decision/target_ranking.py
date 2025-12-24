"""Target ranking and prioritization for drug discovery.

This module implements the decision layer that transforms causal effect
estimates into actionable target rankings. This is where models meet
decisions—the critical step that industry platforms like Ochre Bio's
Late Validation Platform emphasize.

Key principle: Models don't choose targets—decision rules do.

Components:
- TargetScore: Multi-criteria scoring for targets
- TargetRanker: Rank targets by composite score
- Go/No-Go rules: Threshold-based decision criteria

Usage:
    from causalbiolab.decision import TargetRanker, apply_go_nogo_rules
    
    ranker = TargetRanker()
    rankings = ranker.rank(ate_results, cate_results)
    decisions = apply_go_nogo_rules(rankings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from causalbiolab.estimation.ate import ATEResult
    from causalbiolab.estimation.cate import CATEResult


@dataclass
class TargetScore:
    """Multi-criteria score for a drug target candidate.
    
    Attributes:
        name: Target name (e.g., gene symbol)
        effect_size: Magnitude of causal effect (ATE)
        effect_direction: Sign of effect (+1 beneficial, -1 harmful)
        robustness: Confidence in the estimate (1/SE or similar)
        heterogeneity: Degree of effect variation across subgroups
        specificity: How specific is the effect (optional)
        composite_score: Weighted combination of criteria
        rank: Final rank (1 = best)
        decision: Go/No-Go/Maybe decision
    """
    name: str
    effect_size: float
    effect_direction: int
    robustness: float
    heterogeneity: float
    specificity: float = 1.0
    composite_score: float = 0.0
    rank: int = 0
    decision: Literal["go", "no-go", "maybe"] = "maybe"
    
    # Additional metadata
    ate: float = 0.0
    ate_ci_lower: float = 0.0
    ate_ci_upper: float = 0.0
    n_treated: int = 0
    n_control: int = 0
    
    def __repr__(self) -> str:
        return (
            f"TargetScore(name={self.name}, "
            f"effect={self.effect_size:.3f}, "
            f"score={self.composite_score:.3f}, "
            f"rank={self.rank}, "
            f"decision={self.decision})"
        )


@dataclass
class RankingConfig:
    """Configuration for target ranking.
    
    Attributes:
        effect_weight: Weight for effect size in composite score
        robustness_weight: Weight for robustness (1/SE)
        heterogeneity_weight: Weight for low heterogeneity (negative = prefer homogeneous)
        specificity_weight: Weight for specificity
        min_effect_size: Minimum effect size for "go" decision
        max_ci_width: Maximum CI width for "go" decision
        min_n_treated: Minimum treated samples for "go" decision
        beneficial_direction: Expected direction of beneficial effect (+1 or -1)
    """
    effect_weight: float = 0.4
    robustness_weight: float = 0.3
    heterogeneity_weight: float = -0.1  # Negative: prefer homogeneous effects
    specificity_weight: float = 0.2
    
    # Go/No-Go thresholds
    min_effect_size: float = 0.5
    max_ci_width: float = 2.0
    min_n_treated: int = 50
    beneficial_direction: int = 1  # +1 = higher is better, -1 = lower is better
    
    # Score normalization
    normalize_scores: bool = True


class TargetRanker:
    """Rank drug targets based on causal effect estimates.
    
    This class implements a multi-criteria decision framework for
    prioritizing drug targets based on:
    
    1. Effect size: How large is the causal effect?
    2. Robustness: How confident are we in the estimate?
    3. Heterogeneity: Is the effect consistent across subgroups?
    4. Specificity: Is the effect specific to the target?
    
    The composite score is a weighted combination of these criteria.
    """
    
    def __init__(self, config: RankingConfig | None = None):
        """Initialize ranker with configuration.
        
        Args:
            config: Ranking configuration. If None, uses defaults.
        """
        self.config = config or RankingConfig()
    
    def score_target(
        self,
        name: str,
        ate_result: "ATEResult",
        cate_result: "CATEResult | None" = None,
        specificity: float = 1.0,
    ) -> TargetScore:
        """Compute multi-criteria score for a single target.
        
        Args:
            name: Target name
            ate_result: ATE estimation result
            cate_result: Optional CATE result for heterogeneity
            specificity: Specificity score (0-1, higher = more specific)
            
        Returns:
            TargetScore with all criteria computed
        """
        # Effect size (absolute value)
        effect_size = abs(ate_result.ate)
        effect_direction = 1 if ate_result.ate > 0 else -1
        
        # Robustness (inverse of standard error, capped)
        robustness = min(1.0 / (ate_result.se + 1e-6), 10.0) / 10.0  # Normalize to 0-1
        
        # Heterogeneity (from CATE if available)
        if cate_result is not None:
            # Coefficient of variation of CATE
            heterogeneity = cate_result.cate_std / (abs(cate_result.ate) + 1e-6)
            heterogeneity = min(heterogeneity, 2.0) / 2.0  # Normalize to 0-1
        else:
            heterogeneity = 0.5  # Default: moderate heterogeneity
        
        # Create score object
        score = TargetScore(
            name=name,
            effect_size=effect_size,
            effect_direction=effect_direction,
            robustness=robustness,
            heterogeneity=heterogeneity,
            specificity=specificity,
            ate=ate_result.ate,
            ate_ci_lower=ate_result.ci_lower,
            ate_ci_upper=ate_result.ci_upper,
            n_treated=ate_result.n_treated,
            n_control=ate_result.n_control,
        )
        
        # Compute composite score
        score.composite_score = self._compute_composite_score(score)
        
        return score
    
    def _compute_composite_score(self, score: TargetScore) -> float:
        """Compute weighted composite score."""
        cfg = self.config
        
        # Normalize effect size (log scale for large effects)
        effect_norm = np.log1p(score.effect_size) / np.log1p(5.0)  # Normalize assuming max ~5
        effect_norm = min(effect_norm, 1.0)
        
        # Direction bonus: add if effect is in beneficial direction
        direction_bonus = 0.1 if score.effect_direction == cfg.beneficial_direction else 0.0
        
        composite = (
            cfg.effect_weight * effect_norm
            + cfg.robustness_weight * score.robustness
            + cfg.heterogeneity_weight * score.heterogeneity  # Negative weight
            + cfg.specificity_weight * score.specificity
            + direction_bonus
        )
        
        return composite
    
    def rank(
        self,
        targets: dict[str, "ATEResult"],
        cate_results: dict[str, "CATEResult"] | None = None,
        specificities: dict[str, float] | None = None,
    ) -> list[TargetScore]:
        """Rank multiple targets by composite score.
        
        Args:
            targets: Dict mapping target name to ATEResult
            cate_results: Optional dict mapping target name to CATEResult
            specificities: Optional dict mapping target name to specificity score
            
        Returns:
            List of TargetScore objects, sorted by rank (best first)
        """
        cate_results = cate_results or {}
        specificities = specificities or {}
        
        scores = []
        for name, ate_result in targets.items():
            cate_result = cate_results.get(name)
            specificity = specificities.get(name, 1.0)
            
            score = self.score_target(name, ate_result, cate_result, specificity)
            scores.append(score)
        
        # Sort by composite score (descending)
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(scores):
            score.rank = i + 1
        
        # Apply go/no-go decisions
        for score in scores:
            score.decision = self._apply_decision_rules(score)
        
        return scores
    
    def _apply_decision_rules(self, score: TargetScore) -> Literal["go", "no-go", "maybe"]:
        """Apply go/no-go decision rules to a target."""
        cfg = self.config
        
        # Check minimum effect size
        if score.effect_size < cfg.min_effect_size:
            return "no-go"
        
        # Check CI width (robustness)
        ci_width = score.ate_ci_upper - score.ate_ci_lower
        if ci_width > cfg.max_ci_width:
            return "maybe"
        
        # Check sample size
        if score.n_treated < cfg.min_n_treated:
            return "maybe"
        
        # Check direction
        if score.effect_direction != cfg.beneficial_direction:
            return "no-go"
        
        # All criteria met
        return "go"


def rank_targets(
    ate_results: dict[str, "ATEResult"],
    cate_results: dict[str, "CATEResult"] | None = None,
    config: RankingConfig | None = None,
) -> list[TargetScore]:
    """Convenience function to rank targets.
    
    Args:
        ate_results: Dict mapping target name to ATEResult
        cate_results: Optional CATE results
        config: Ranking configuration
        
    Returns:
        Ranked list of TargetScore objects
    """
    ranker = TargetRanker(config)
    return ranker.rank(ate_results, cate_results)


def apply_go_nogo_rules(
    rankings: list[TargetScore],
    min_effect: float = 0.5,
    max_ci_width: float = 2.0,
    min_samples: int = 50,
) -> dict[str, list[TargetScore]]:
    """Apply go/no-go rules and categorize targets.
    
    Args:
        rankings: List of ranked TargetScore objects
        min_effect: Minimum effect size for "go"
        max_ci_width: Maximum CI width for "go"
        min_samples: Minimum treated samples for "go"
        
    Returns:
        Dict with keys "go", "no-go", "maybe" containing target lists
    """
    result = {"go": [], "no-go": [], "maybe": []}
    
    for score in rankings:
        ci_width = score.ate_ci_upper - score.ate_ci_lower
        
        if score.effect_size >= min_effect and ci_width <= max_ci_width and score.n_treated >= min_samples:
            result["go"].append(score)
        elif score.effect_size < min_effect * 0.5:
            result["no-go"].append(score)
        else:
            result["maybe"].append(score)
    
    return result


def print_rankings(rankings: list[TargetScore], top_n: int = 10) -> None:
    """Pretty print target rankings."""
    print("\n" + "=" * 80)
    print("Target Rankings")
    print("=" * 80)
    print(f"{'Rank':<6} {'Target':<15} {'Effect':>10} {'Robustness':>12} {'Score':>10} {'Decision':>10}")
    print("-" * 80)
    
    for score in rankings[:top_n]:
        print(
            f"{score.rank:<6} {score.name:<15} {score.effect_size:>10.3f} "
            f"{score.robustness:>12.3f} {score.composite_score:>10.3f} {score.decision:>10}"
        )
    
    if len(rankings) > top_n:
        print(f"... and {len(rankings) - top_n} more targets")
    
    print("=" * 80)
    
    # Summary
    decisions = [s.decision for s in rankings]
    print(f"Summary: {decisions.count('go')} go, {decisions.count('maybe')} maybe, {decisions.count('no-go')} no-go")
