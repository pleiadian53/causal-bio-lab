# CATE Estimation with Meta-Learners: A Tutorial

**Source:** `notebooks/causal_inference/01_treatment_effects/01_treatment_effects.ipynb` (Section 7)

This tutorial provides a deep dive into **Meta-Learners** for estimating Conditional Average Treatment Effects (CATE). We'll cover all four main approaches, with special focus on **X-Learner** and **DR-Learner**.

---

## Table of Contents

1. [Why CATE? Beyond Average Effects](#1-why-cate-beyond-average-effects)
2. [The Meta-Learner Framework](#2-the-meta-learner-framework)
3. [S-Learner: The Simplest Approach](#3-s-learner-the-simplest-approach)
4. [T-Learner: Separate Models](#4-t-learner-separate-models)
5. [X-Learner: Impute and Combine](#5-x-learner-impute-and-combine)
6. [DR-Learner: Doubly Robust CATE](#6-dr-learner-doubly-robust-cate)
7. [Comparison and Guidelines](#7-comparison-and-guidelines)
8. [Implementation Guide](#8-implementation-guide)

---

## 1. Why CATE? Beyond Average Effects

### The Limitation of ATE

The Average Treatment Effect (ATE) tells us the **average** effect across the population:

$$\text{ATE} = \mathbb{E}[Y(1) - Y(0)]$$

But treatment effects often **vary** across individuals. For example:
- A drug might help some patients but harm others
- A gene knockout might have strong effects in some cell types but not others
- A marketing intervention might work for some customer segments but not others

### CATE: Personalized Treatment Effects

The **Conditional Average Treatment Effect (CATE)** captures this heterogeneity:

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

**Interpretation:** CATE is the expected treatment effect for individuals with covariates $X = x$.

### Example from the Notebook

The notebook simulates heterogeneous effects:

$$\tau(x) = 1 + 2x$$

- If $x = -2$: $\tau(x) = 1 + 2(-2) = -3$ (treatment is **harmful**)
- If $x = 0$: $\tau(x) = 1 + 2(0) = 1$ (treatment is **helpful**)
- If $x = 2$: $\tau(x) = 1 + 2(2) = 5$ (treatment is **very helpful**)

**Key insight:** The treatment helps units with high $X$ but harms units with low $X$. The ATE (about 1.0) masks this crucial heterogeneity.

---

## 2. The Meta-Learner Framework

### What Are Meta-Learners?

**Meta-learners** are algorithms that use standard machine learning models as building blocks to estimate CATE. They're called "meta" because they wrap around existing ML models.

### Why "Meta"?

Instead of designing a custom causal model, meta-learners:
1. Use off-the-shelf ML models (Random Forest, Gradient Boosting, Neural Networks, etc.)
2. Apply them in specific ways to estimate treatment effects
3. Combine predictions cleverly to get CATE estimates

### The Key Insight

All meta-learners solve the same fundamental problem:

> **We never observe both $Y(0)$ and $Y(1)$ for the same unit.**

Each meta-learner handles this differently:
- **S-Learner:** Models $E[Y \mid X, T]$ jointly, then computes difference
- **T-Learner:** Models $E[Y \mid X, T=0]$ and $E[Y \mid X, T=1]$ separately
- **X-Learner:** Imputes missing counterfactuals, then models effects directly
- **DR-Learner:** Uses doubly robust pseudo-outcomes, then models CATE

---

## 3. S-Learner: The Simplest Approach

### The Idea

**S-Learner (Single Model)** fits one model that includes treatment as a feature.

### Mathematical Formulation

**Step 1:** Fit a single model

$$\hat{\mu}(x, t) = \mathbb{E}[Y \mid X=x, T=t]$$

**Step 2:** Estimate CATE by predicting under both treatments

$$\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)$$

### Algorithm

```
1. Augment features: X_aug = [X, T]
2. Fit model: model.fit(X_aug, Y)
3. For prediction at x:
   - Predict under treatment: mu_1 = model.predict([x, 1])
   - Predict under control: mu_0 = model.predict([x, 0])
   - Return: tau(x) = mu_1 - mu_0
```

### Code Implementation

```python
class SLearner:
    def __init__(self, base_model=None):
        self.model = base_model or GradientBoostingRegressor(n_estimators=100, max_depth=3)
    
    def fit(self, X, T, Y):
        # Include treatment as a feature
        X_aug = np.column_stack([X, T])
        self.model.fit(X_aug, Y)
        return self
    
    def predict(self, X):
        n = len(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        # Predict under T=1 and T=0
        y1 = self.model.predict(np.column_stack([X, np.ones(n)]))
        y0 = self.model.predict(np.column_stack([X, np.zeros(n)]))
        return y1 - y0
```

### Data Shapes

- Input `X`: shape `(n, d)` — covariates (d features)
- Input `T`: shape `(n,)` — treatment assignments
- Input `Y`: shape `(n,)` — outcomes
- `X_aug`: shape `(n, d+1)` — covariates + treatment
- Output `tau(x)`: shape `(n,)` — CATE estimates

### Pros and Cons

| Pros | Cons |
|------|------|
| Simple, easy to implement | May underestimate heterogeneity |
| Uses all data efficiently | Treatment effect can get "regularized away" |
| Works with any ML model | Model may not learn treatment interactions |

### When S-Learner Fails

S-Learner can underestimate treatment effect heterogeneity when:
- Treatment effects are small relative to outcome variance
- Base model doesn't capture interactions well
- Treatment is highly regularized (e.g., in tree-based models with many splits)

---

## 4. T-Learner: Separate Models

### The Idea

**T-Learner (Two Models)** fits separate models for treated and control groups.

### Mathematical Formulation

**Step 1:** Fit separate models

$$\hat{\mu}_0(x) = \mathbb{E}[Y \mid X=x, T=0]$$
$$\hat{\mu}_1(x) = \mathbb{E}[Y \mid X=x, T=1]$$

**Step 2:** Estimate CATE as the difference

$$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

### Algorithm

```
1. Split data by treatment:
   - Control: (X_0, Y_0) where T=0
   - Treated: (X_1, Y_1) where T=1
2. Fit model_0 on (X_0, Y_0)
3. Fit model_1 on (X_1, Y_1)
4. For prediction at x:
   - Return: tau(x) = model_1.predict(x) - model_0.predict(x)
```

### Code Implementation

```python
class TLearner:
    def __init__(self, base_model=None):
        base = base_model or GradientBoostingRegressor(n_estimators=100, max_depth=3)
        self.model_0 = clone(base)  # Control model
        self.model_1 = clone(base)  # Treatment model
    
    def fit(self, X, T, Y):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        # Fit separate models
        self.model_0.fit(X[T == 0], Y[T == 0])
        self.model_1.fit(X[T == 1], Y[T == 1])
        return self
    
    def predict(self, X):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return self.model_1.predict(X) - self.model_0.predict(X)
```

### Data Shapes

- Input `X`: shape `(n, d)` — covariates
- Input `T`: shape `(n,)` — treatment assignments
- Input `Y`: shape `(n,)` — outcomes
- `X[T==0]`: shape `(n_0, d)` — control group covariates
- `X[T==1]`: shape `(n_1, d)` — treatment group covariates
- Output `tau(x)`: shape `(n,)` — CATE estimates

### Pros and Cons

| Pros | Cons |
|------|------|
| Better at capturing heterogeneity | Requires sufficient samples in both arms |
| Models can be tailored to each arm | Variance can be high if arms are imbalanced |
| No regularization of treatment effect | Doesn't share information between arms |

### When T-Learner Struggles

T-Learner can have high variance when:
- Treatment groups are imbalanced (e.g., 90/10 split)
- Sample size is small
- One group has few samples in certain regions of $X$

---

## 5. X-Learner: Impute and Combine

### The Idea

**X-Learner** is designed for situations with **imbalanced treatment groups**. It imputes counterfactuals and uses propensity weighting to combine estimates.

### The Key Insight

X-Learner asks:
> "What would the treatment effect be if we could observe both potential outcomes?"

It answers this by **imputing the missing counterfactual** for each unit, then modeling the imputed effects.

### Mathematical Formulation

**Step 1:** Fit base models (like T-Learner)

$$\hat{\mu}_0(x) = \mathbb{E}[Y \mid X=x, T=0]$$
$$\hat{\mu}_1(x) = \mathbb{E}[Y \mid X=x, T=1]$$

**Step 2:** Impute individual treatment effects

For treated units (we observe $Y_i(1)$, impute $Y_i(0)$):
$$\tilde{D}^1_i = Y_i - \hat{\mu}_0(X_i) \quad \text{for } T_i = 1$$

For control units (we observe $Y_i(0)$, impute $Y_i(1)$):
$$\tilde{D}^0_i = \hat{\mu}_1(X_i) - Y_i \quad \text{for } T_i = 0$$

**Interpretation:**
- $\tilde{D}^1_i$: Actual outcome minus predicted control outcome = imputed effect for treated
- $\tilde{D}^0_i$: Predicted treatment outcome minus actual outcome = imputed effect for control

**Step 3:** Fit effect models on imputed effects

$$\hat{\tau}_1(x) = \mathbb{E}[\tilde{D}^1 \mid X=x]$$
$$\hat{\tau}_0(x) = \mathbb{E}[\tilde{D}^0 \mid X=x]$$

**Step 4:** Combine using propensity weighting

$$\hat{\tau}(x) = e(x) \cdot \hat{\tau}_0(x) + (1 - e(x)) \cdot \hat{\tau}_1(x)$$

where $e(x) = P(T=1 \mid X=x)$ is the propensity score.

### Algorithm

```
# Step 1: Fit base models
model_0.fit(X[T==0], Y[T==0])
model_1.fit(X[T==1], Y[T==1])

# Step 2: Impute treatment effects
D1 = Y[T==1] - model_0.predict(X[T==1])  # For treated
D0 = model_1.predict(X[T==0]) - Y[T==0]  # For control

# Step 3: Fit effect models
tau_model_1.fit(X[T==1], D1)  # Effect model from treated
tau_model_0.fit(X[T==0], D0)  # Effect model from control

# Step 4: Fit propensity model
ps_model.fit(X, T)
e = ps_model.predict_proba(X)[:, 1]

# Step 5: Combine predictions
tau_1 = tau_model_1.predict(X)
tau_0 = tau_model_0.predict(X)
tau = e * tau_0 + (1 - e) * tau_1
```

### Code Implementation

```python
class XLearner:
    def __init__(self, base_model=None):
        base = base_model or GradientBoostingRegressor(n_estimators=100, max_depth=3)
        # Base outcome models
        self.model_0 = clone(base)  # E[Y|X, T=0]
        self.model_1 = clone(base)  # E[Y|X, T=1]
        # Effect models
        self.tau_model_0 = clone(base)  # Effect model from control
        self.tau_model_1 = clone(base)  # Effect model from treated
        # Propensity model
        self.ps_model = LogisticRegression(max_iter=1000)
    
    def fit(self, X, T, Y):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Step 1: Fit base outcome models
        self.model_0.fit(X[T == 0], Y[T == 0])
        self.model_1.fit(X[T == 1], Y[T == 1])
        
        # Step 2: Impute individual treatment effects
        # For treated: actual - predicted control
        D1 = Y[T == 1] - self.model_0.predict(X[T == 1])
        # For control: predicted treatment - actual
        D0 = self.model_1.predict(X[T == 0]) - Y[T == 0]
        
        # Step 3: Fit effect models on imputed effects
        self.tau_model_1.fit(X[T == 1], D1)
        self.tau_model_0.fit(X[T == 0], D0)
        
        # Step 4: Fit propensity model
        self.ps_model.fit(X, T)
        
        return self
    
    def predict(self, X):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Get propensity scores
        e = self.ps_model.predict_proba(X)[:, 1]
        
        # Get CATE predictions from both effect models
        tau_1 = self.tau_model_1.predict(X)  # From treated
        tau_0 = self.tau_model_0.predict(X)  # From control
        
        # Combine using propensity weighting
        # More weight to tau_0 when e(x) is high (more treated units to learn from)
        # More weight to tau_1 when e(x) is low (more control units to learn from)
        tau = e * tau_0 + (1 - e) * tau_1
        
        return tau
```

### Why the Weighting Works

The propensity weighting in Step 4 is subtle but crucial:

$$\hat{\tau}(x) = e(x) \cdot \hat{\tau}_0(x) + (1 - e(x)) \cdot \hat{\tau}_1(x)$$

**Intuition:**
- When $e(x)$ is high → many treated units at $x$ → $\hat{\tau}_1(x)$ is reliable
  - But we weight by $(1-e(x))$, so we use $\hat{\tau}_0(x)$ (from control)
  - Why? Because $\hat{\tau}_0$ was trained on control units, and there are **few** of those at high-propensity $x$
  - So we trust the model that was trained on the abundant group!
  
- When $e(x)$ is low → many control units at $x$ → $\hat{\tau}_0(x)$ is reliable
  - But we weight by $e(x)$, so we use $\hat{\tau}_1(x)$ (from treated)
  - Why? Because $\hat{\tau}_1$ was trained on treated units, and there are **few** of those at low-propensity $x$

**Key insight:** Each effect model is trained on one group but evaluated using information from the other group. This shares information between arms effectively.

### Data Shapes

- Input `X`: shape `(n, d)`
- Input `T`: shape `(n,)`
- Input `Y`: shape `(n,)`
- `D1`: shape `(n_1,)` — imputed effects for treated
- `D0`: shape `(n_0,)` — imputed effects for control
- `e`: shape `(n,)` — propensity scores
- Output `tau`: shape `(n,)`

### Pros and Cons

| Pros | Cons |
|------|------|
| Handles imbalanced groups well | More complex to implement |
| Shares info between arms efficiently | Requires propensity model |
| Generally lower variance than T-Learner | More models to fit and tune |

---

## 6. DR-Learner: Doubly Robust CATE

### The Idea

**DR-Learner** applies the doubly robust principle to CATE estimation. It creates **pseudo-outcomes** that are unbiased estimates of individual treatment effects, then fits a model to predict CATE.

### The Key Insight

DR-Learner asks:
> "Can we create an unbiased estimate of $Y(1) - Y(0)$ for each unit?"

The answer is **yes**, using the AIPW (Augmented Inverse Propensity Weighting) pseudo-outcome:

$$\tilde{Y}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)}$$

This pseudo-outcome has the property that $\mathbb{E}[\tilde{Y}_i \mid X_i] = \tau(X_i)$ under mild conditions.

### Mathematical Formulation

**Step 1:** Fit nuisance models

$$\hat{\mu}_0(x) = \mathbb{E}[Y \mid X=x, T=0]$$
$$\hat{\mu}_1(x) = \mathbb{E}[Y \mid X=x, T=1]$$
$$\hat{e}(x) = P(T=1 \mid X=x)$$

**Step 2:** Compute pseudo-outcomes

$$\tilde{Y}_i = \underbrace{\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)}_{\text{Regression estimate}} + \underbrace{\frac{T_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)}}_{\text{Correction for treated}} - \underbrace{\frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)}}_{\text{Correction for control}}$$

**Step 3:** Fit CATE model on pseudo-outcomes

$$\hat{\tau}(x) = \mathbb{E}[\tilde{Y} \mid X=x]$$

### Breaking Down the Pseudo-Outcome

Let's understand each term:

1. **$\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)$**: Regression-based CATE estimate
   - What the outcome model thinks the effect is
   - Biased if outcome model is wrong

2. **$\frac{T_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)}$**: IPW correction for treated
   - Only non-zero for treated units ($T_i = 1$)
   - Corrects the regression estimate using actual outcomes
   - Upweights treated units with low propensity

3. **$\frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)}$**: IPW correction for control
   - Only non-zero for control units ($T_i = 0$)
   - Corrects the regression estimate using actual outcomes
   - Upweights control units with high propensity

**Why "doubly robust"?**
- If outcome models ($\hat{\mu}_0$, $\hat{\mu}_1$) are correct → regression term is unbiased, correction terms are zero in expectation
- If propensity model ($\hat{e}$) is correct → IPW corrections fix any bias in regression estimates
- Need only **one** to be correct for consistent estimation!

### Algorithm

```
# Step 1: Fit nuisance models (outcome + propensity)
model_0.fit(X[T==0], Y[T==0])
model_1.fit(X[T==1], Y[T==1])
ps_model.fit(X, T)

# Step 2: Get predictions
mu_0 = model_0.predict(X)
mu_1 = model_1.predict(X)
e = ps_model.predict_proba(X)[:, 1]
e = np.clip(e, 0.01, 0.99)  # Clip for stability

# Step 3: Compute pseudo-outcomes
pseudo_Y = (mu_1 - mu_0 
            + T * (Y - mu_1) / e 
            - (1 - T) * (Y - mu_0) / (1 - e))

# Step 4: Fit CATE model on pseudo-outcomes
cate_model.fit(X, pseudo_Y)

# Step 5: Predict CATE
tau = cate_model.predict(X)
```

### Code Implementation

```python
class DRLearner:
    def __init__(self, base_model=None):
        base = base_model or GradientBoostingRegressor(n_estimators=100, max_depth=3)
        # Nuisance models
        self.model_0 = clone(base)  # E[Y|X, T=0]
        self.model_1 = clone(base)  # E[Y|X, T=1]
        self.ps_model = LogisticRegression(max_iter=1000)
        # Final CATE model
        self.cate_model = clone(base)
    
    def fit(self, X, T, Y, clip=(0.01, 0.99)):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Step 1: Fit nuisance models
        self.model_0.fit(X[T == 0], Y[T == 0])
        self.model_1.fit(X[T == 1], Y[T == 1])
        self.ps_model.fit(X, T)
        
        # Step 2: Get nuisance predictions
        mu_0 = self.model_0.predict(X)
        mu_1 = self.model_1.predict(X)
        e = self.ps_model.predict_proba(X)[:, 1]
        e = np.clip(e, *clip)  # Clip for stability
        
        # Step 3: Compute pseudo-outcomes
        pseudo_Y = (mu_1 - mu_0 
                    + T * (Y - mu_1) / e 
                    - (1 - T) * (Y - mu_0) / (1 - e))
        
        # Step 4: Fit CATE model on pseudo-outcomes
        self.cate_model.fit(X, pseudo_Y)
        
        return self
    
    def predict(self, X):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return self.cate_model.predict(X)
```

### Why DR-Learner is Powerful

1. **Doubly robust:** Consistent if either outcome or propensity model is correct
2. **Semiparametrically efficient:** Achieves the optimal variance under correct specification
3. **Flexible final stage:** Can use any ML model for CATE
4. **Handles confounding:** Propensity weighting addresses selection bias

### Data Shapes

- Input `X`: shape `(n, d)`
- Input `T`: shape `(n,)`
- Input `Y`: shape `(n,)`
- `mu_0`, `mu_1`: shape `(n,)` — predicted outcomes
- `e`: shape `(n,)` — propensity scores
- `pseudo_Y`: shape `(n,)` — pseudo-outcomes
- Output `tau`: shape `(n,)`

### Pros and Cons

| Pros | Cons |
|------|------|
| Most robust to model misspecification | Requires both outcome and propensity models |
| Optimal variance properties | More complex to implement |
| Strong theoretical guarantees | Can be unstable with extreme propensities |
| Handles confounding well | Requires propensity clipping |

---

## 7. Comparison and Guidelines

### Summary Table

| Learner | Models Needed | Key Idea | Best For |
|---------|---------------|----------|----------|
| **S-Learner** | 1 outcome model | Include T as feature | Simple baseline, large effects |
| **T-Learner** | 2 outcome models | Separate models per arm | Balanced groups, strong heterogeneity |
| **X-Learner** | 2+2 models + propensity | Impute + combine | Imbalanced groups |
| **DR-Learner** | 2 models + propensity + CATE | Pseudo-outcomes | Maximum robustness |

### When to Use Each

**Use S-Learner when:**
- You want a simple baseline
- Treatment effects are large and easy to detect
- Sample size is limited

**Use T-Learner when:**
- Treatment groups are roughly balanced
- You expect strong treatment effect heterogeneity
- Models can differ substantially between arms

**Use X-Learner when:**
- Treatment groups are **imbalanced** (e.g., 80/20 split)
- You want to leverage information from both arms
- Propensity scores are well-estimated

**Use DR-Learner when:**
- You want maximum robustness to model misspecification
- There is **confounding** in the data
- You can estimate propensity scores well
- This is your **default choice for observational data**

### Decision Flowchart

```
Start
  │
  ├─ Is this experimental (RCT) or observational data?
  │   │
  │   ├─ RCT: S-Learner or T-Learner (no confounding)
  │   │
  │   └─ Observational: DR-Learner (handles confounding)
  │
  ├─ Are treatment groups balanced?
  │   │
  │   ├─ Yes: T-Learner or DR-Learner
  │   │
  │   └─ No: X-Learner or DR-Learner
  │
  └─ How important is robustness?
      │
      ├─ Very important: DR-Learner
      │
      └─ Less critical: Simpler learner (S, T, or X)
```

### Performance Comparison (from notebook)

```
Method          ATE     RMSE    Correlation
----------------------------------------------
S-Learner      1.076   0.221   0.996
T-Learner      1.097   0.348   0.990
----------------------------------------------
True           1.023
```

**Observations:**
- Both capture the true ATE well
- S-Learner has lower RMSE in this case (simple true CATE function)
- Both have very high correlation with true CATE

---

## 8. Implementation Guide

### Complete Example

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

# Generate synthetic data with heterogeneous effects
def generate_heterogeneous_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    
    # Covariates
    X = rng.uniform(-2, 2, n)
    
    # Confounded treatment
    propensity = 1 / (1 + np.exp(-0.5 * X))
    T = rng.binomial(1, propensity)
    
    # True CATE: tau(x) = 1 + 2x
    true_cate = 1 + 2 * X
    
    # Potential outcomes
    Y0 = 5 + 1.5 * X + rng.normal(0, 1, n)
    Y1 = Y0 + true_cate
    
    # Observed outcome
    Y = np.where(T == 1, Y1, Y0)
    
    return X, T, Y, true_cate

# Generate data
X, T, Y, true_cate = generate_heterogeneous_data()

# Fit all learners
s_learner = SLearner().fit(X.reshape(-1, 1), T, Y)
t_learner = TLearner().fit(X.reshape(-1, 1), T, Y)
x_learner = XLearner().fit(X.reshape(-1, 1), T, Y)
dr_learner = DRLearner().fit(X.reshape(-1, 1), T, Y)

# Get CATE predictions
cate_s = s_learner.predict(X.reshape(-1, 1))
cate_t = t_learner.predict(X.reshape(-1, 1))
cate_x = x_learner.predict(X.reshape(-1, 1))
cate_dr = dr_learner.predict(X.reshape(-1, 1))

# Evaluate
def evaluate(name, cate_hat, true_cate):
    ate = cate_hat.mean()
    rmse = np.sqrt(((cate_hat - true_cate) ** 2).mean())
    corr = np.corrcoef(cate_hat, true_cate)[0, 1]
    print(f"{name:<15} {ate:>8.3f} {rmse:>8.3f} {corr:>8.3f}")

print(f"{'Method':<15} {'ATE':>8} {'RMSE':>8} {'Corr':>8}")
print("-" * 45)
evaluate("S-Learner", cate_s, true_cate)
evaluate("T-Learner", cate_t, true_cate)
evaluate("X-Learner", cate_x, true_cate)
evaluate("DR-Learner", cate_dr, true_cate)
print("-" * 45)
print(f"{'True':>15} {true_cate.mean():>8.3f}")
```

### Tips for Practice

1. **Always compare multiple learners**
   - If they disagree significantly, investigate why
   - Large disagreements may indicate model misspecification or confounding

2. **Use cross-validation**
   - Split data for nuisance model fitting vs. CATE estimation
   - Prevents overfitting in pseudo-outcome construction

3. **Clip propensity scores**
   - Always use `e = np.clip(e, 0.01, 0.99)`
   - Prevents numerical instability from extreme weights

4. **Check overlap**
   - Ensure all regions of $X$ have both treated and control units
   - Weak overlap leads to high variance estimates

5. **Start simple**
   - Begin with S-Learner or T-Learner
   - Move to X-Learner or DR-Learner if needed

---

## Further Reading

- **Notebook:** `notebooks/causal_inference/01_treatment_effects/01_treatment_effects.ipynb` — See these in action
- **Example:** `examples/01_treatment_effect_estimation.py` — Real-world application
- **Paper:** Künzel et al. (2019) "Meta-learners for Estimating Heterogeneous Treatment Effects"
- **Paper:** Kennedy (2020) "Towards optimal doubly robust estimation of heterogeneous causal effects"

