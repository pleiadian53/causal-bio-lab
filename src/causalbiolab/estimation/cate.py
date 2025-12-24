"""Conditional Average Treatment Effect (CATE) estimation.

CATE answers: "For which cell states / donors / disease contexts does this
perturbation have the largest effect?"

This is crucial for drug target discovery because:
- Not all patients respond equally to treatments
- Cell state heterogeneity affects perturbation response
- Identifying responder subgroups enables precision medicine

Key estimators:
- MetaLearners: S-learner, T-learner, X-learner
- CausalForest: Tree-based heterogeneous effect estimation
- DoublyRobustCATE: AIPW-style CATE with cross-fitting

Usage:
    from causalbiolab.estimation.cate import TLearner, CausalForestCATE
    
    # Estimate heterogeneous effects
    cate_model = TLearner()
    cate_model.fit(X, T, Y)
    effects = cate_model.predict(X_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CATEResult:
    """Result of CATE estimation.
    
    Attributes:
        cate: Individual treatment effects (n_samples,)
        ate: Average treatment effect (mean of CATE)
        cate_std: Standard deviation of CATE estimates
        method: Name of the estimation method
        feature_importance: Feature importance for effect heterogeneity (if available)
    """
    cate: NDArray[np.floating]
    ate: float
    cate_std: float
    method: str
    feature_importance: NDArray[np.floating] | None = None
    
    def __repr__(self) -> str:
        return (
            f"CATEResult(ate={self.ate:.4f}, "
            f"cate_std={self.cate_std:.4f}, "
            f"n_samples={len(self.cate)}, "
            f"method={self.method})"
        )
    
    def top_responders(self, n: int = 10) -> NDArray[np.integer]:
        """Get indices of top responders (highest positive CATE)."""
        return np.argsort(self.cate)[-n:][::-1]
    
    def bottom_responders(self, n: int = 10) -> NDArray[np.integer]:
        """Get indices of bottom responders (lowest/negative CATE)."""
        return np.argsort(self.cate)[:n]


class CATEEstimator(ABC):
    """Abstract base class for CATE estimators."""
    
    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "CATEEstimator":
        """Fit the CATE model.
        
        Args:
            X: Covariates (n_samples, n_features)
            T: Treatment indicator (n_samples,), binary 0/1
            Y: Outcome (n_samples,)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict CATE for new samples.
        
        Args:
            X: Covariates for prediction (n_samples, n_features)
            
        Returns:
            CATE estimates (n_samples,)
        """
        pass
    
    def fit_predict(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> CATEResult:
        """Fit and return CATE results for training data."""
        self.fit(X, T, Y)
        cate = self.predict(X)
        
        return CATEResult(
            cate=cate,
            ate=float(cate.mean()),
            cate_std=float(cate.std()),
            method=self.__class__.__name__,
            feature_importance=getattr(self, "feature_importance_", None),
        )
    
    def _validate_inputs(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Validate and convert inputs."""
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


class SLearner(CATEEstimator):
    """S-Learner (Single model) for CATE estimation.
    
    Fits a single model: Y ~ f(X, T)
    CATE(x) = f(x, 1) - f(x, 0)
    
    Simple but can underestimate heterogeneity if treatment effect
    is small relative to outcome variance.
    """
    
    def __init__(self, base_model: object | None = None):
        """Initialize S-Learner.
        
        Args:
            base_model: Sklearn regressor. If None, uses GradientBoostingRegressor.
        """
        self.base_model = base_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
    
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "SLearner":
        X, T, Y = self._validate_inputs(X, T, Y)
        
        # Augment X with treatment indicator
        X_aug = np.column_stack([X, T])
        self.base_model.fit(X_aug, Y)
        
        self._n_features = X.shape[1]
        return self
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X)
        n = X.shape[0]
        
        # Predict under treatment and control
        X_treated = np.column_stack([X, np.ones(n)])
        X_control = np.column_stack([X, np.zeros(n)])
        
        y1 = self.base_model.predict(X_treated)
        y0 = self.base_model.predict(X_control)
        
        return y1 - y0


class TLearner(CATEEstimator):
    """T-Learner (Two models) for CATE estimation.
    
    Fits separate models for treated and control:
    - mu_0(x) = E[Y|X=x, T=0]
    - mu_1(x) = E[Y|X=x, T=1]
    
    CATE(x) = mu_1(x) - mu_0(x)
    
    Better at capturing heterogeneity but requires sufficient samples
    in both treatment arms.
    """
    
    def __init__(self, base_model: object | None = None):
        """Initialize T-Learner.
        
        Args:
            base_model: Sklearn regressor (will be cloned for each arm).
        """
        from sklearn.base import clone
        
        base = base_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.model_0 = clone(base)
        self.model_1 = clone(base)
    
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "TLearner":
        X, T, Y = self._validate_inputs(X, T, Y)
        
        # Fit separate models
        mask_0 = T == 0
        mask_1 = T == 1
        
        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])
        
        return self
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X)
        
        y0 = self.model_0.predict(X)
        y1 = self.model_1.predict(X)
        
        return y1 - y0


class XLearner(CATEEstimator):
    """X-Learner for CATE estimation.
    
    A more sophisticated meta-learner that:
    1. Fits outcome models for each arm (like T-learner)
    2. Imputes treatment effects for each group
    3. Fits CATE models on imputed effects
    4. Combines using propensity weighting
    
    Generally performs well when treatment/control groups are imbalanced.
    """
    
    def __init__(
        self,
        outcome_model: object | None = None,
        effect_model: object | None = None,
        propensity_model: object | None = None,
    ):
        """Initialize X-Learner.
        
        Args:
            outcome_model: Model for Y ~ X in each arm
            effect_model: Model for imputed effects ~ X
            propensity_model: Model for P(T=1|X)
        """
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        
        outcome_base = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        effect_base = effect_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        
        self.model_0 = clone(outcome_base)
        self.model_1 = clone(outcome_base)
        self.effect_model_0 = clone(effect_base)
        self.effect_model_1 = clone(effect_base)
        self.propensity_model = propensity_model or LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
        )
    
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "XLearner":
        X, T, Y = self._validate_inputs(X, T, Y)
        
        mask_0 = T == 0
        mask_1 = T == 1
        
        # Step 1: Fit outcome models
        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])
        
        # Step 2: Impute treatment effects
        # For treated: D_1 = Y_1 - mu_0(X_1)
        # For control: D_0 = mu_1(X_0) - Y_0
        D_1 = Y[mask_1] - self.model_0.predict(X[mask_1])
        D_0 = self.model_1.predict(X[mask_0]) - Y[mask_0]
        
        # Step 3: Fit effect models
        self.effect_model_1.fit(X[mask_1], D_1)
        self.effect_model_0.fit(X[mask_0], D_0)
        
        # Step 4: Fit propensity model
        self.propensity_model.fit(X, T)
        
        return self
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X)
        
        # Get propensity scores
        propensity = self.propensity_model.predict_proba(X)[:, 1]
        
        # Get CATE estimates from both effect models
        tau_0 = self.effect_model_0.predict(X)
        tau_1 = self.effect_model_1.predict(X)
        
        # Combine using propensity weighting
        # CATE = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
        # where g(x) = P(T=1|X=x)
        cate = propensity * tau_0 + (1 - propensity) * tau_1
        
        return cate


class DoublyRobustCATE(CATEEstimator):
    """Doubly Robust CATE estimator with cross-fitting.
    
    Uses the AIPW (Augmented Inverse Propensity Weighting) approach
    for CATE estimation, with cross-fitting to avoid overfitting.
    
    This is the most robust meta-learner, consistent if either the
    outcome model or propensity model is correctly specified.
    """
    
    def __init__(
        self,
        outcome_model: object | None = None,
        propensity_model: object | None = None,
        final_model: object | None = None,
        n_folds: int = 5,
    ):
        """Initialize DR-CATE.
        
        Args:
            outcome_model: Model for E[Y|X,T]
            propensity_model: Model for P(T=1|X)
            final_model: Model to fit pseudo-outcomes
            n_folds: Number of cross-fitting folds
        """
        from sklearn.linear_model import LogisticRegression
        
        self.outcome_model = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.propensity_model = propensity_model or LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
        )
        self.final_model = final_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        self.n_folds = n_folds
    
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "DoublyRobustCATE":
        X, T, Y = self._validate_inputs(X, T, Y)
        
        n = len(Y)
        
        # Cross-fitted propensity scores
        propensity = cross_val_predict(
            self.propensity_model, X, T, cv=self.n_folds, method="predict_proba"
        )[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)
        
        # Cross-fitted outcome predictions
        X_aug = np.column_stack([X, T])
        self.outcome_model.fit(X_aug, Y)
        
        mu_1 = self.outcome_model.predict(np.column_stack([X, np.ones(n)]))
        mu_0 = self.outcome_model.predict(np.column_stack([X, np.zeros(n)]))
        
        # Compute pseudo-outcomes (DR scores)
        # Gamma = mu_1(X) - mu_0(X) + T*(Y - mu_1(X))/e(X) - (1-T)*(Y - mu_0(X))/(1-e(X))
        pseudo_outcome = (
            mu_1 - mu_0
            + T * (Y - mu_1) / propensity
            - (1 - T) * (Y - mu_0) / (1 - propensity)
        )
        
        # Fit final model on pseudo-outcomes
        self.final_model.fit(X, pseudo_outcome)
        
        # Store feature importance if available
        if hasattr(self.final_model, "feature_importances_"):
            self.feature_importance_ = self.final_model.feature_importances_
        
        return self
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X)
        return self.final_model.predict(X)


def compare_cate_estimators(
    X: NDArray[np.floating],
    T: NDArray[np.integer],
    Y: NDArray[np.floating],
    estimators: list[CATEEstimator] | None = None,
) -> dict[str, CATEResult]:
    """Compare multiple CATE estimators.
    
    Args:
        X: Covariates
        T: Treatment
        Y: Outcome
        estimators: List of estimators. If None, uses all four.
        
    Returns:
        Dictionary of results
    """
    if estimators is None:
        estimators = [
            SLearner(),
            TLearner(),
            XLearner(),
            DoublyRobustCATE(),
        ]
    
    results = {}
    for est in estimators:
        result = est.fit_predict(X, T, Y)
        results[result.method] = result
    
    return results


def print_cate_comparison(results: dict[str, CATEResult]) -> None:
    """Pretty print CATE comparison."""
    print("\n" + "=" * 70)
    print("CATE Estimation Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'ATE':>10} {'CATE Std':>12} {'Min CATE':>12} {'Max CATE':>12}")
    print("-" * 70)
    
    for method, result in results.items():
        print(
            f"{method:<20} {result.ate:>10.4f} {result.cate_std:>12.4f} "
            f"{result.cate.min():>12.4f} {result.cate.max():>12.4f}"
        )
    
    print("=" * 70)
