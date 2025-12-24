"""Average Treatment Effect (ATE) estimation for perturbation biology.

This module provides ATE estimators designed for Perturb-seq and similar
perturbation experiments where we want to estimate the causal effect of
gene knockouts on biological outcomes.

Key estimators:
- NaiveATE: Simple difference in means (biased if confounders exist)
- PropensityATE: Inverse propensity weighting (IPW)
- DoublyRobustATE: Combines outcome modeling with propensity weighting
- DoWhyATE: Wrapper around DoWhy for full causal inference workflow

Usage:
    from causalbiolab.estimation.ate import DoublyRobustATE
    
    estimator = DoublyRobustATE()
    ate, ci = estimator.estimate(
        X=covariates,
        T=treatment,
        Y=outcome,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ATEResult:
    """Result of ATE estimation.
    
    Attributes:
        ate: Point estimate of average treatment effect
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        se: Standard error of the estimate
        method: Name of the estimation method
        n_treated: Number of treated units
        n_control: Number of control units
    """
    ate: float
    ci_lower: float
    ci_upper: float
    se: float
    method: str
    n_treated: int
    n_control: int
    
    def __repr__(self) -> str:
        return (
            f"ATEResult(ate={self.ate:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"method={self.method})"
        )


class ATEEstimator(ABC):
    """Abstract base class for ATE estimators."""
    
    @abstractmethod
    def estimate(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
        alpha: float = 0.05,
    ) -> ATEResult:
        """Estimate the average treatment effect.
        
        Args:
            X: Covariates matrix (n_samples, n_features)
            T: Treatment indicator (n_samples,), binary 0/1
            Y: Outcome variable (n_samples,)
            alpha: Significance level for confidence interval
            
        Returns:
            ATEResult with point estimate and confidence interval
        """
        pass
    
    def _validate_inputs(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Validate and convert inputs to numpy arrays."""
        X = np.asarray(X)
        T = np.asarray(T).ravel()
        Y = np.asarray(Y).ravel()
        
        if X.shape[0] != len(T) or X.shape[0] != len(Y):
            raise ValueError(
                f"Inconsistent sample sizes: X={X.shape[0]}, T={len(T)}, Y={len(Y)}"
            )
        
        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("Treatment T must be binary (0 or 1)")
        
        return X, T, Y
    
    def _compute_ci(
        self,
        ate: float,
        se: float,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute confidence interval using normal approximation."""
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        return ate - z * se, ate + z * se


class NaiveATE(ATEEstimator):
    """Naive difference-in-means estimator.
    
    This is the simplest estimator: E[Y|T=1] - E[Y|T=0].
    It is unbiased only under randomization (no confounding).
    
    For Perturb-seq data where cells are randomly assigned to perturbations,
    this can be a valid estimator if there are no batch effects or other
    confounders.
    """
    
    def estimate(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
        alpha: float = 0.05,
    ) -> ATEResult:
        """Estimate ATE using simple difference in means."""
        X, T, Y = self._validate_inputs(X, T, Y)
        
        # Split by treatment
        Y1 = Y[T == 1]
        Y0 = Y[T == 0]
        
        # Point estimate
        ate = Y1.mean() - Y0.mean()
        
        # Standard error (assuming independence)
        se = np.sqrt(Y1.var() / len(Y1) + Y0.var() / len(Y0))
        
        # Confidence interval
        ci_lower, ci_upper = self._compute_ci(ate, se, alpha)
        
        return ATEResult(
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se=se,
            method="naive_difference",
            n_treated=len(Y1),
            n_control=len(Y0),
        )


class PropensityATE(ATEEstimator):
    """Inverse Propensity Weighting (IPW) estimator.
    
    Estimates ATE by reweighting observations by the inverse of their
    propensity score (probability of receiving treatment given covariates).
    
    This corrects for confounding when the propensity model is correctly
    specified, but can have high variance when propensity scores are extreme.
    """
    
    def __init__(
        self,
        propensity_model: LogisticRegression | None = None,
        clip_propensity: tuple[float, float] = (0.01, 0.99),
    ):
        """Initialize IPW estimator.
        
        Args:
            propensity_model: Sklearn classifier for propensity estimation.
                If None, uses LogisticRegression with regularization.
            clip_propensity: Min/max bounds for propensity scores to avoid
                extreme weights.
        """
        self.propensity_model = propensity_model or LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
        )
        self.clip_propensity = clip_propensity
    
    def estimate(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
        alpha: float = 0.05,
    ) -> ATEResult:
        """Estimate ATE using inverse propensity weighting."""
        X, T, Y = self._validate_inputs(X, T, Y)
        
        # Fit propensity model
        self.propensity_model.fit(X, T)
        propensity = self.propensity_model.predict_proba(X)[:, 1]
        
        # Clip to avoid extreme weights
        propensity = np.clip(propensity, *self.clip_propensity)
        
        # IPW estimator
        # ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
        weights_treated = T / propensity
        weights_control = (1 - T) / (1 - propensity)
        
        ate = (weights_treated * Y).mean() - (weights_control * Y).mean()
        
        # Variance estimation (simplified)
        # Using influence function approach
        psi = weights_treated * Y - weights_control * Y - ate
        se = np.sqrt(psi.var() / len(Y))
        
        ci_lower, ci_upper = self._compute_ci(ate, se, alpha)
        
        return ATEResult(
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se=se,
            method="ipw",
            n_treated=int(T.sum()),
            n_control=int((1 - T).sum()),
        )


class DoublyRobustATE(ATEEstimator):
    """Doubly Robust (AIPW) estimator.
    
    Combines outcome regression with propensity weighting. This estimator
    is consistent if EITHER the outcome model OR the propensity model is
    correctly specified (hence "doubly robust").
    
    This is generally the recommended estimator for observational data
    as it provides protection against model misspecification.
    """
    
    def __init__(
        self,
        propensity_model: LogisticRegression | None = None,
        outcome_model: Ridge | None = None,
        clip_propensity: tuple[float, float] = (0.01, 0.99),
        cv_folds: int = 5,
    ):
        """Initialize doubly robust estimator.
        
        Args:
            propensity_model: Classifier for propensity estimation
            outcome_model: Regressor for outcome prediction
            clip_propensity: Bounds for propensity scores
            cv_folds: Number of cross-validation folds for out-of-sample predictions
        """
        self.propensity_model = propensity_model or LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
        )
        self.outcome_model = outcome_model or Ridge(alpha=1.0)
        self.clip_propensity = clip_propensity
        self.cv_folds = cv_folds
    
    def estimate(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
        alpha: float = 0.05,
    ) -> ATEResult:
        """Estimate ATE using doubly robust (AIPW) method."""
        X, T, Y = self._validate_inputs(X, T, Y)
        
        n = len(Y)
        
        # Get propensity scores (cross-validated to avoid overfitting)
        propensity = cross_val_predict(
            self.propensity_model, X, T, cv=self.cv_folds, method="predict_proba"
        )[:, 1]
        propensity = np.clip(propensity, *self.clip_propensity)
        
        # Get outcome predictions for both treatment arms
        # mu_1(x) = E[Y|X, T=1] and mu_0(x) = E[Y|X, T=0]
        X_with_T = np.column_stack([X, T])
        
        # Predict Y(1): outcome under treatment
        X_treated = np.column_stack([X, np.ones(n)])
        self.outcome_model.fit(X_with_T, Y)
        mu_1 = self.outcome_model.predict(X_treated)
        
        # Predict Y(0): outcome under control
        X_control = np.column_stack([X, np.zeros(n)])
        mu_0 = self.outcome_model.predict(X_control)
        
        # Doubly robust estimator (AIPW)
        # psi = mu_1(X) - mu_0(X) + T*(Y - mu_1(X))/e(X) - (1-T)*(Y - mu_0(X))/(1-e(X))
        psi = (
            mu_1 - mu_0
            + T * (Y - mu_1) / propensity
            - (1 - T) * (Y - mu_0) / (1 - propensity)
        )
        
        ate = psi.mean()
        se = np.sqrt(psi.var() / n)
        
        ci_lower, ci_upper = self._compute_ci(ate, se, alpha)
        
        return ATEResult(
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se=se,
            method="doubly_robust",
            n_treated=int(T.sum()),
            n_control=int((1 - T).sum()),
        )


def compare_estimators(
    X: NDArray[np.floating],
    T: NDArray[np.integer],
    Y: NDArray[np.floating],
    estimators: list[ATEEstimator] | None = None,
) -> dict[str, ATEResult]:
    """Compare multiple ATE estimators on the same data.
    
    Args:
        X: Covariates
        T: Treatment indicator
        Y: Outcome
        estimators: List of estimators to compare. If None, uses all three.
        
    Returns:
        Dictionary mapping method name to ATEResult
    """
    if estimators is None:
        estimators = [
            NaiveATE(),
            PropensityATE(),
            DoublyRobustATE(),
        ]
    
    results = {}
    for est in estimators:
        result = est.estimate(X, T, Y)
        results[result.method] = result
    
    return results


def print_comparison(results: dict[str, ATEResult]) -> None:
    """Pretty print comparison of ATE estimates."""
    print("\n" + "=" * 70)
    print("ATE Estimation Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'ATE':>10} {'SE':>10} {'95% CI':>25}")
    print("-" * 70)
    
    for method, result in results.items():
        ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        print(f"{method:<20} {result.ate:>10.4f} {result.se:>10.4f} {ci_str:>25}")
    
    print("=" * 70)
    print(f"N treated: {list(results.values())[0].n_treated}, "
          f"N control: {list(results.values())[0].n_control}")
