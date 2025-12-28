# Fundamental Assumptions of the Potential Outcomes Framework

**Companion document for:** `01_treatment_effects.ipynb`

This document explains the four key assumptions required for causal inference using the potential outcomes framework (Rubin Causal Model). These assumptions are essential for identifying and estimating treatment effects from observational data.

---

## Notation

For each unit $i$ (e.g., a cell, patient, or experimental unit):

| Symbol | Meaning | Example |
|--------|---------|---------|
| $T_i$ | Treatment indicator (0 or 1) | $T_i = 1$ if cell $i$ received gene knockout, $T_i = 0$ if control |
| $Y_i(0)$ | Potential outcome under control | Gene expression level if cell $i$ did NOT receive knockout |
| $Y_i(1)$ | Potential outcome under treatment | Gene expression level if cell $i$ DID receive knockout |
| $X_i$ | Covariates (pre-treatment variables) | Library size, cell cycle phase, baseline expression |
| $Y_i$ | Observed outcome | What we actually measure: $Y_i = T_i \cdot Y_i(1) + (1-T_i) \cdot Y_i(0)$ |
| $\tau_i$ | Individual Treatment Effect | $\tau_i = Y_i(1) - Y_i(0)$ (unobservable) |

**Key insight:** We only observe $Y_i$, not both $Y_i(0)$ and $Y_i(1)$. This is the **fundamental problem of causal inference**.

---

## 1. SUTVA (Stable Unit Treatment Value Assumption)

### Mathematical Statement

* **No interference:** $Y_i(T_1, T_2, ..., T_n) = Y_i(T_i)$
* **No hidden treatment versions:** All units receiving treatment $T_i = 1$ receive the same treatment

### In Plain English

1. **No interference:** One unit's treatment doesn't affect another unit's outcome
2. **No hidden versions:** There's only one way to receive treatment (or control)

### Example: Perturb-seq Experiment

**Context:** Cells receive gene knockouts

**SUTVA requires:**

✅ **No interference:**
* Cell $i$'s knockout doesn't affect cell $j$'s gene expression
* Cells are isolated (e.g., in separate wells, no cell-cell communication)
* Example: If we knock out gene X in cell 1, it doesn't change cell 2's expression

❌ **Violation (interference):**
* Cells are in the same culture, and knocked-out cells secrete factors that affect neighbors
* Example: Cell 1's knockout causes it to release cytokines that change cell 2's expression

✅ **No hidden versions:**
* All "knockout" treatments are identical (same guide RNA, same efficiency)
* All "control" treatments are identical (same mock treatment)

❌ **Violation (hidden versions):**
* Some knockouts are partial (50% efficiency), others are complete (100% efficiency)
* Some controls receive empty vector, others receive scrambled guide RNA

### Why It Matters

**Without SUTVA:**
* We can't define individual treatment effects: $\tau_i = Y_i(1) - Y_i(0)$ becomes ambiguous
* The potential outcome $Y_i(1)$ depends on what treatments other units received
* We can't estimate ATE from simple comparisons

**In the notebook code:**
```python
# This code assumes SUTVA holds
Y0 = rng.normal(5, 1, n)  # Y(0) depends only on unit i, not others
Y1 = Y0 + 2 + rng.normal(0, 0.5, n)  # Y(1) depends only on unit i
```

If SUTVA were violated, we'd need to model $Y_i(T_1, T_2, ..., T_n)$ instead of just $Y_i(T_i)$.

---

## 2. Consistency

### Mathematical Statement

$$Y_i = Y_i(T_i)$$

### In Plain English

The observed outcome equals the potential outcome for the treatment actually received.

### Example from the Notebook

**Context:** We observe $Y_i$ (the actual gene expression we measure).

**Consistency requires:**

✅ **If $T_i = 1$ (treated):**
* Then $Y_i = Y_i(1)$
* The observed expression equals what it would be under treatment

✅ **If $T_i = 0$ (control):**
* Then $Y_i = Y_i(0)$
* The observed expression equals what it would be under control

### In the Notebook Code

```python
# Observed outcome
Y = np.where(T == 1, Y1, Y0)
```

This code explicitly implements consistency:
* If $T_i = 1$: $Y_i = Y_i(1)$
* If $T_i = 0$: $Y_i = Y_i(0)$

**Mathematical formulation:**
$$Y_i = T_i \cdot Y_i(1) + (1-T_i) \cdot Y_i(0)$$

### Why It Matters

**Without consistency:**
* We can't link observed data to potential outcomes
* We can't estimate treatment effects from observed outcomes
* The fundamental problem becomes unsolvable

**Example violation:**
* A cell receives treatment ($T_i = 1$), but due to experimental error, we measure the wrong cell's outcome
* Or: Treatment is assigned but not actually delivered (non-compliance)

---

## 3. Ignorability (Unconfoundedness)

### Mathematical Statement

$$(Y(0), Y(1)) \perp\!\!\!\perp T \mid X$$

### In Plain English

Given covariates $X$, treatment assignment is independent of potential outcomes. Treatment is "as good as random" after conditioning on $X$.

### Example from the Notebook

**Context:** The notebook demonstrates both randomized and confounded scenarios.

**Ignorability requires:**

✅ **With randomization (RCT):**
* Treatment is randomly assigned: $T \perp\!\!\!\perp (Y(0), Y(1))$
* No need to condition on $X$ because treatment is independent of everything
* Example: Cells are randomly assigned to knockout vs. control

✅ **In observational data (with confounders):**
* After conditioning on $X$, treatment is independent of potential outcomes
* Example: After controlling for library size, treatment assignment is independent of potential outcomes

❌ **Violation (confounding):**
* Treatment assignment depends on unmeasured confounders
* Example from notebook: Library size ($X$) affects both treatment probability and outcome
  ```python
  # X affects treatment (confounding!)
  propensity = 1 / (1 + np.exp(-confounding_strength * X))
  T = rng.binomial(1, propensity)
  
  # X also affects outcome
  Y0 = 5 + 1.5 * X + rng.normal(0, 1, n)
  ```

### Why It Matters

**Without ignorability:**
* Naive comparison is biased:
  ```python
  naive_ate = Y[T == 1].mean() - Y[T == 0].mean()
  # This is BIASED when ignorability fails!
  ```
* The difference mixes true treatment effect with confounder effects
* We need methods like IPW, regression adjustment, or doubly robust estimation

**With ignorability (after conditioning on $X$):**
* We can estimate ATE using:
  * Regression: $E[Y \mid T, X]$
  * IPW: Reweight by propensity scores
  * Doubly robust: Combine both approaches

### In the Notebook

**Randomized case (ignorability holds):**
```python
# Random treatment assignment
T = rng.binomial(1, 0.5, n)  # 50% chance, independent of everything
# Here: (Y(0), Y(1)) ⟂ T (no conditioning needed)
```

**Confounded case (ignorability violated):**
```python
# Treatment depends on X
propensity = 1 / (1 + np.exp(-confounding_strength * X))
T = rng.binomial(1, propensity)
# Here: (Y(0), Y(1)) ⟂ T | X (need to condition on X)
```

---

## 4. Positivity (Overlap)

### Mathematical Statement

$$0 < P(T=1 \mid X) < 1 \quad \text{for all } X$$

### In Plain English

Every unit has a positive probability of receiving either treatment, regardless of their covariate values.

### Example from the Notebook

**Context:** Treatment assignment depends on confounder $X$ (library size).

**Positivity requires:**

✅ **For every value of $X$:**
* Some units with that $X$ value receive treatment: $P(T=1 \mid X) > 0$
* Some units with that $X$ value receive control: $P(T=0 \mid X) = 1 - P(T=1 \mid X) > 0$

**In the notebook code:**
```python
propensity = 1 / (1 + np.exp(-confounding_strength * X))
# This always gives: 0 < propensity < 1 (positivity holds!)
```

✅ **Example:**
* Cells with low library size ($X = -2$): $P(T=1 \mid X=-2) \approx 0.27$ (some get treated)
* Cells with high library size ($X = +2$): $P(T=1 \mid X=+2) \approx 0.73$ (some get control)

❌ **Violation (lack of overlap):**
* Cells with very high library size ($X > 3$) **always** get treatment: $P(T=1 \mid X > 3) = 1$
* Cells with very low library size ($X < -3$) **never** get treatment: $P(T=1 \mid X < -3) = 0$
* We can't estimate treatment effects for these extreme $X$ values

### Why It Matters

**Without positivity:**
* Some treatment effects are not identifiable
* Propensity scores can be 0 or 1, causing problems in IPW (division by zero)
* Regression extrapolates to regions with no data
* Example: If only young patients get treatment, we can't estimate treatment effect for old patients

**With positivity:**
* We can estimate treatment effects for all covariate values
* Propensity scores are well-behaved: $0 < e(X) < 1$
* We have data to compare treated vs. control within each $X$ stratum

### In the Notebook

The logistic function ensures positivity:
```python
propensity = 1 / (1 + np.exp(-confounding_strength * X))
# Always: 0 < propensity < 1 (positivity satisfied)
```

But if we had:
```python
# BAD: Violates positivity
propensity = np.where(X > 2, 1.0, 0.0)  # Some X values give P(T=1|X) = 1 or 0
```

This would violate positivity for extreme $X$ values.

---

## Summary: How These Assumptions Work Together

### In the Notebook's Randomized Example

```python
# SUTVA: Y0 and Y1 depend only on unit i
Y0 = rng.normal(5, 1, n)
Y1 = Y0 + 2 + rng.normal(0, 0.5, n)

# Consistency: Observed = potential for treatment received
Y = np.where(T == 1, Y1, Y0)

# Ignorability: Random assignment makes T independent of (Y(0), Y(1))
T = rng.binomial(1, 0.5, n)  # Random!

# Positivity: Every unit has 50% chance of either treatment
# P(T=1) = 0.5, P(T=0) = 0.5 for all units
```

### In the Notebook's Confounded Example

```python
# SUTVA: Still holds (Y0, Y1 depend only on unit i)
Y0 = 5 + 1.5 * X + rng.normal(0, 1, n)
Y1 = Y0 + true_ate

# Consistency: Still holds
Y = np.where(T == 1, Y1, Y0)

# Ignorability: VIOLATED without conditioning on X
# (Y(0), Y(1)) ⟂ T | X (need to condition on X)
propensity = 1 / (1 + np.exp(-confounding_strength * X))
T = rng.binomial(1, propensity)  # Depends on X!

# Positivity: Still holds (propensity always between 0 and 1)
```

### Key Takeaway

* **SUTVA** and **Consistency** are usually assumed to hold (design-level assumptions)
* **Ignorability** is the main challenge in observational data (solved by conditioning on $X$ or using instrumental variables)
* **Positivity** must be checked empirically (can sometimes be violated in practice)

---

## Connection to Other Causal Frameworks

### Structural Causal Models (Pearl's Framework)

* **SUTVA:** Implicit in the structural equations
* **Consistency:** Built into the do-operator
* **Ignorability:** Captured by the backdoor criterion
* **Positivity:** Required for identification

### Do-Calculus

* These assumptions are encoded in the rules for manipulating do-expressions
* The backdoor and frontdoor criteria incorporate versions of these assumptions

### Instrumental Variables

* Uses a different identification strategy (exclusion restriction, relevance, independence)
* Still relies on versions of SUTVA and consistency

---

## Bottom Line

These four assumptions are fundamental to causal inference across all frameworks, not just the Rubin Causal Model. The potential outcomes framework makes them explicit; other frameworks may embed them differently, but they are still required for valid causal inference from observational data.

**When working through the notebook:**
1. Identify which assumptions hold in each simulation
2. Understand how violations lead to biased estimates
3. Learn which estimation methods address which violations
4. Apply this understanding to real perturbation biology experiments
