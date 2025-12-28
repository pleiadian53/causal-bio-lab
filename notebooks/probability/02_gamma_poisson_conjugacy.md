# Gamma-Poisson Conjugacy and Poisson Processes

This document explains the mathematical foundations of Gamma-Poisson conjugacy in Bayesian inference and how the Gamma distribution arises naturally from Poisson processes.

---

## Part 1: Gamma-Poisson Conjugacy

### The Setup

In Bayesian inference, we want to estimate an unknown **rate parameter** $\lambda$ from count data. The conjugate prior framework provides closed-form posterior updates.

**Model:**
- **Likelihood:** $X_1, X_2, \ldots, X_n \mid \lambda \stackrel{\text{i.i.d.}}{\sim} \text{Poisson}(\lambda)$
  
  *(Read: "Given the rate $\lambda$, each observation $X_i$ independently follows a Poisson distribution with that rate.")*

- **Prior:** $\lambda \sim \text{Gamma}(\alpha, \beta)$ where $\alpha$ = shape, $\beta$ = rate

### Why Gamma is Conjugate to Poisson

A prior is **conjugate** to a likelihood if the posterior belongs to the same family as the prior. For Gamma-Poisson:

$$\text{Gamma prior} + \text{Poisson data} = \text{Gamma posterior}$$

### The Mathematical Derivation

**Step 1: Write the Poisson Likelihood**

For n independent observations $x_1, x_2, \ldots, x_n$:

$$P(X_1=x_1, \ldots, X_n=x_n \mid \lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum x_i} e^{-n\lambda}}{\prod x_i!}$$

> **Concrete Example:** Suppose we observe gene expression counts from 3 cells: $x_1 = 2$, $x_2 = 5$, $x_3 = 3$.
>
> The joint probability of seeing exactly these counts, given rate $\lambda = 4$, is:
>
> $$P(X_1=2, X_2=5, X_3=3 \mid \lambda=4) = \underbrace{\frac{4^2 e^{-4}}{2!}}_{\text{cell 1}} \times \underbrace{\frac{4^5 e^{-4}}{5!}}_{\text{cell 2}} \times \underbrace{\frac{4^3 e^{-4}}{3!}}_{\text{cell 3}}$$
>
> Since the cells are **independent**, the joint probability is just the **product** of individual probabilities. We can simplify:
>
> $$= \frac{4^{2+5+3} \cdot e^{-4 \times 3}}{2! \cdot 5! \cdot 3!} = \frac{4^{10} \cdot e^{-12}}{2! \cdot 5! \cdot 3!}$$
>
> Notice how:
> - The exponent of $\lambda$ becomes the **total count** $S = 2 + 5 + 3 = 10$
> - The exponent of $e$ becomes $-n\lambda = -3 \times 4 = -12$
> - The denominator (factorials) doesn't depend on $\lambda$, so we can ignore it when finding the MLE or posterior

Let $S = \sum_{i=1}^{n} x_i$ (total count). The likelihood is:

$$L(\lambda) \propto \lambda^S e^{-n\lambda}$$

*(The $\propto$ means "proportional to" — we drop constants that don't depend on $\lambda$.)*

**Step 2: Write the Gamma Prior**

$$p(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda} \propto \lambda^{\alpha-1} e^{-\beta\lambda}$$

**Step 3: Apply Bayes' Theorem**

$$p(\lambda \mid \text{data}) \propto L(\lambda) \cdot p(\lambda)$$

$$p(\lambda \mid \text{data}) \propto \lambda^S e^{-n\lambda} \cdot \lambda^{\alpha-1} e^{-\beta\lambda}$$

$$p(\lambda \mid \text{data}) \propto \lambda^{(\alpha + S) - 1} e^{-(\beta + n)\lambda}$$

**Step 4: Recognize the Gamma Form**

This is the kernel of a Gamma distribution! Therefore:

$$\lambda \mid \text{data} \sim \text{Gamma}(\alpha + S, \beta + n)$$

### The Conjugate Update Rule

| Parameter | Prior | Posterior |
|-----------|-------|-----------|
| Shape | $\alpha$ | $\alpha + \sum x_i$ |
| Rate | $\beta$ | $\beta + n$ |

**In words:**
- **Shape increases** by the total count (sum of all observations)
- **Rate increases** by the number of observations

### Posterior Mean: A Weighted Average

The posterior mean is:

$$E[\lambda \mid \text{data}] = \frac{\alpha + S}{\beta + n}$$

This can be rewritten as a **weighted average** of the prior mean and the MLE:

$$E[\lambda \mid \text{data}] = \underbrace{\frac{\beta}{\beta + n}}_{\text{prior weight}} \cdot \underbrace{\frac{\alpha}{\beta}}_{\text{prior mean}} + \underbrace{\frac{n}{\beta + n}}_{\text{data weight}} \cdot \underbrace{\frac{S}{n}}_{\text{MLE}}$$

**Interpretation:**
- With little data ($n$ small), the posterior is close to the prior
- With lots of data ($n$ large), the posterior converges to the MLE
- The prior acts as "pseudo-observations": $\alpha$ pseudo-counts from $\beta$ pseudo-samples

---

## Part 2: How Gamma Arises in Poisson Processes

The Gamma distribution appears naturally in Poisson processes in two fundamental ways.

### What is a Poisson Process?

A **Poisson process** with rate $\lambda$ models random events occurring over time:

- Events occur independently
- The rate $\lambda$ is constant (homogeneous process)
- The number of events in time interval $[0, t]$ follows $\text{Poisson}(\lambda t)$
- The waiting time between events follows $\text{Exponential}(1/\lambda)$

### Connection 1: Waiting Time for k Events

**Theorem:** The waiting time until the k-th event in a Poisson process with rate $\lambda$ follows:

$$T_k \sim \text{Gamma}(k, 1/\lambda)$$

**Proof Sketch:**

Let $W_1, W_2, \ldots, W_k$ be the inter-arrival times (time between consecutive events).

Each $W_i \sim \text{Exponential}(1/\lambda)$ independently.

The total waiting time is:

$$T_k = W_1 + W_2 + \cdots + W_k$$

By the sum-of-exponentials theorem (see `02_gamma_exponential_relationship.md`):

$$T_k \sim \text{Gamma}(k, 1/\lambda)$$

**Example:** If customers arrive at a rate of 2 per hour ($\lambda = 2$), the time until the 5th customer arrives follows Gamma(5, 0.5) with:
- Mean = $5 \times 0.5 = 2.5$ hours
- Variance = $5 \times 0.5^2 = 1.25$ hours²

### Connection 2: Bayesian Inference on the Rate

When we observe a Poisson process and want to infer its rate $\lambda$, the Gamma distribution is the natural prior because:

1. **Support matches:** Both $\lambda$ and Gamma are defined on $(0, \infty)$
2. **Conjugacy:** Gamma prior + Poisson likelihood = Gamma posterior
3. **Interpretability:** Prior parameters have intuitive meaning as pseudo-observations

**The Gamma Prior as "Prior Experience":**

If we set prior $\lambda \sim \text{Gamma}(\alpha, \beta)$:
- $\alpha$ represents "prior total count" (pseudo-observations)
- $\beta$ represents "prior exposure time" or "prior sample size"
- Prior mean $\alpha/\beta$ is our initial guess for the rate

### Connection 3: Gamma-Poisson Mixture = Negative Binomial

When the Poisson rate itself is uncertain and follows a Gamma distribution:

$$\lambda \sim \text{Gamma}(r, p/(1-p))$$
$$X \mid \lambda \sim \text{Poisson}(\lambda)$$

Then marginally:

$$X \sim \text{NegativeBinomial}(r, p)$$

This is crucial for modeling **overdispersion** (variance > mean) in count data, common in:
- scRNA-seq gene expression
- Clinical event counts
- Ecological abundance data

---

## Part 3: Code Example from the Notebook

The notebook cell demonstrates Gamma-Poisson conjugacy:

```python
# Prior: Gamma(5, rate=1) → mean = 5
prior_shape = 5
prior_rate = 1

# Data: 20 cells, total counts = 133
n_cells = 20
total_counts = 133

# Posterior: Gamma(5 + 133, 1 + 20) = Gamma(138, 21)
posterior_shape = prior_shape + total_counts  # 138
posterior_rate = prior_rate + n_cells          # 21

# Posterior mean
posterior_mean = posterior_shape / posterior_rate  # ≈ 6.57
```

**What the visualization shows:**
- **Blue curve (Prior):** Our initial belief, centered around 5
- **Red curve (Posterior):** Updated belief after seeing data, shifted toward the MLE
- **Green line (True rate):** The actual rate we're trying to estimate (7)
- **Orange line (MLE):** Sample mean = 133/20 = 6.65

The posterior mean (6.57) is between the prior mean (5) and MLE (6.65), demonstrating the weighted average property.

---

## Summary

### Gamma-Poisson Conjugacy

| Component | Formula |
|-----------|---------|
| Prior | $\lambda \sim \text{Gamma}(\alpha, \beta)$ |
| Likelihood | $X_i \mid \lambda \sim \text{Poisson}(\lambda)$ |
| Posterior | $\lambda \mid X \sim \text{Gamma}(\alpha + \sum X_i, \beta + n)$ |
| Posterior Mean | $\frac{\alpha + \sum X_i}{\beta + n}$ |

### Gamma in Poisson Processes

| Connection | Description |
|------------|-------------|
| Waiting times | Time until k-th event ~ Gamma(k, 1/λ) |
| Bayesian prior | Natural conjugate prior for Poisson rate |
| Overdispersion | Gamma-Poisson mixture = Negative Binomial |

---

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. Chapter 2.
- Ross, S. M. (2014). *Introduction to Probability Models*. Academic Press. Chapter 5.
- See `02_gamma_distribution.ipynb` for interactive visualizations.
- See `03_negative_binomial.ipynb` for the Gamma-Poisson mixture (overdispersion).
