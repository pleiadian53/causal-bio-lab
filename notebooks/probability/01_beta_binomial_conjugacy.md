# Beta-Binomial Conjugacy: The Math Behind Bayesian Updating

This document explains the mathematical foundations of the Bayesian updating example in [`01_beta_distribution.ipynb`](./01_beta_distribution.ipynb) (Cell 14).

---

## Table of Contents

1. [The Setup](#1-the-setup)
2. [Bayes' Theorem](#2-bayes-theorem)
3. [The Beta Prior](#3-the-beta-prior)
4. [The Binomial Likelihood](#4-the-binomial-likelihood)
5. [Deriving the Posterior](#5-deriving-the-posterior)
6. [Why "Conjugate"?](#6-why-conjugate)
7. [Interpreting the Parameters](#7-interpreting-the-parameters)
8. [Worked Example](#8-worked-example)

---

## 1. The Setup

**Goal:** Estimate an unknown probability $\theta$ (e.g., drug response rate) from observed data.

**Bayesian approach:**
1. Start with a **prior** belief about $\theta$
2. Observe **data** (successes and failures)
3. Update to get a **posterior** belief

---

## 2. Bayes' Theorem

The foundation of Bayesian inference:

$$
p(\theta \mid \text{data}) = \frac{p(\text{data} \mid \theta) \cdot p(\theta)}{p(\text{data})}
$$

Where:

- $p(\theta)$ = **Prior**: our belief about $\theta$ before seeing data
- $p(\text{data} \mid \theta)$ = **Likelihood**: probability of observing the data given $\theta$
- $p(\theta \mid \text{data})$ = **Posterior**: updated belief after seeing data
- $p(\text{data})$ = **Evidence** (normalizing constant, often ignored)

Since $p(\text{data})$ doesn't depend on $\theta$, we often write:

$$
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
$$

---

## 3. The Beta Prior

We model our prior belief about $\theta \in [0, 1]$ using a **Beta distribution**:

$$
p(\theta) = \text{Beta}(\theta \mid \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

Where:

- $\alpha, \beta > 0$ are shape parameters
- $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$ is the Beta function (normalizing constant)

**Key properties:**

- Mean: $\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}$
- Mode: $\frac{\alpha - 1}{\alpha + \beta - 2}$ (when $\alpha, \beta > 1$)
- Variance: $\text{Var}(\theta) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$

---

## 4. The Binomial Likelihood

We observe $k$ successes in $n$ trials. The likelihood is:

$$
p(\text{data} \mid \theta) = \text{Binomial}(k \mid n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

Where:

- $n$ = number of trials (e.g., patients)
- $k$ = number of successes (e.g., responders)
- $\theta$ = probability of success (what we're estimating)

---

## 5. Deriving the Posterior

Now we combine prior and likelihood:

$$
p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta) \cdot p(\theta)
$$

Substituting:

$$
p(\theta \mid k, n) \propto \left[ \theta^k (1-\theta)^{n-k} \right] \cdot \left[ \theta^{\alpha-1} (1-\theta)^{\beta-1} \right]
$$

Combining exponents:

$$
p(\theta \mid k, n) \propto \theta^{k + \alpha - 1} (1-\theta)^{(n-k) + \beta - 1}
$$

This is the kernel of a Beta distribution! Therefore:

$$
\boxed{p(\theta \mid k, n) = \text{Beta}(\theta \mid \alpha + k, \beta + n - k)}
$$

### The Update Rules

$$
\alpha_{\text{posterior}} = \alpha_{\text{prior}} + k \quad \text{(add successes)}
$$

$$
\beta_{\text{posterior}} = \beta_{\text{prior}} + (n - k) \quad \text{(add failures)}
$$

---

## 6. Why "Conjugate"?

A prior is **conjugate** to a likelihood if the posterior is in the same family as the prior.

| Prior | Likelihood | Posterior |
|-------|------------|-----------|
| Beta($\alpha$, $\beta$) | Binomial($n$, $\theta$) | Beta($\alpha + k$, $\beta + n - k$) |

**Benefits of conjugacy:**

1. **Closed-form posterior** — no numerical integration needed
2. **Interpretable updates** — just add counts to parameters
3. **Sequential updating** — can update as new data arrives

---

## 7. Interpreting the Parameters

The prior parameters $\alpha$ and $\beta$ can be interpreted as **pseudo-counts**:

- $\alpha$ = "prior successes" (pseudo-observations of success)
- $\beta$ = "prior failures" (pseudo-observations of failure)
- $\alpha + \beta$ = "prior sample size" (strength of prior belief)

### Prior Strength

| $\alpha + \beta$ | Interpretation |
|------------------|----------------|
| 2 | Very weak prior (easily overwhelmed by data) |
| 10 | Moderate prior |
| 100 | Strong prior (requires lots of data to shift) |

### Special Priors

| Prior | $\alpha$ | $\beta$ | Interpretation |
|-------|----------|---------|----------------|
| Uniform | 1 | 1 | No prior information |
| Jeffreys | 0.5 | 0.5 | Non-informative (invariant) |
| Haldane | 0 | 0 | Improper, but posterior is proper if $k > 0$ and $n - k > 0$ |

---

## 8. Worked Example

From the notebook (Cell 14):

### Setup

```python
# Prior: We expect ~30% response rate
prior_alpha = 3
prior_beta = 7

# Data: 8 responders out of 20 patients
n_responders = 8
n_patients = 20
```

### Prior

$$
\text{Prior: } \theta \sim \text{Beta}(3, 7)
$$

- Prior mean: $\frac{3}{3+7} = 0.30$ (30% expected response rate)
- Prior "sample size": $3 + 7 = 10$ pseudo-observations

### Likelihood

$$
\text{Data: } k = 8, n = 20
$$

- Observed rate: $\frac{8}{20} = 0.40$ (40%)

### Posterior Update

$$
\alpha_{\text{post}} = 3 + 8 = 11
$$

$$
\beta_{\text{post}} = 7 + (20 - 8) = 7 + 12 = 19
$$

$$
\text{Posterior: } \theta \sim \text{Beta}(11, 19)
$$

### Posterior Mean

$$
\mathbb{E}[\theta \mid \text{data}] = \frac{11}{11 + 19} = \frac{11}{30} \approx 0.367
$$

Notice the posterior mean (0.367) is between:
- Prior mean (0.30)
- Data mean (0.40)

This is a **weighted average**, where the weights depend on the "sample sizes":
- Prior contributes 10 pseudo-observations
- Data contributes 20 real observations
- Posterior is pulled more toward the data (larger sample)

### Posterior as Weighted Average

In general, the posterior mean can be written as:

$$
\mathbb{E}[\theta \mid \text{data}] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{k}{n}}_{\text{data mean}}
$$

For our example:
$$
= \frac{10}{30} \cdot 0.30 + \frac{20}{30} \cdot 0.40 = 0.10 + 0.267 = 0.367 \checkmark
$$

---

## Summary

The Beta-Binomial conjugacy gives us a simple, interpretable Bayesian update:

```text
Prior:     Beta(α, β)
Data:      k successes in n trials
Posterior: Beta(α + k, β + n - k)
```

**Key insights:**

1. Prior parameters are pseudo-counts
2. Posterior mean is a weighted average of prior and data
3. More data → posterior dominated by likelihood
4. Stronger prior → more data needed to shift beliefs

---

## Code Reference

See [`01_beta_distribution.ipynb`](./01_beta_distribution.ipynb), Cell 14 for the implementation.

```python
# Posterior (conjugate update)
posterior_alpha = prior_alpha + n_responders
posterior_beta = prior_beta + (n_patients - n_responders)
```

---

## Further Reading

- Gelman et al., *Bayesian Data Analysis* (Chapter 2)
- Murphy, *Machine Learning: A Probabilistic Perspective* (Chapter 3)
- Bishop, *Pattern Recognition and Machine Learning* (Chapter 2.1)
