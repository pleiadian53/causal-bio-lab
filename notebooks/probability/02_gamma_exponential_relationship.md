# Gamma Distribution and the Sum of Exponentials

This document provides a detailed explanation of the relationship between the Gamma distribution and the sum of independent Exponential random variables — a fundamental result in probability theory with important applications in causal inference and biological modeling.

## The Core Theorem

**If $X_1, X_2, \ldots, X_k$ are independent and identically distributed (i.i.d.) Exponential($\beta$) random variables, then:**

$$S = X_1 + X_2 + \cdots + X_k \sim \text{Gamma}(k, \beta)$$

Where:
- **k** = shape parameter (number of exponentials summed)
- **β** = scale parameter (same as the exponential scale)

---

## Why This Works: Three Perspectives

### 1. Moment Generating Function (MGF) Proof

The MGF of an Exponential($\beta$) random variable is:

$$M_X(t) = \frac{1}{1 - \beta t} \quad \text{for } t < 1/\beta$$

For independent random variables, the MGF of the sum is the product of individual MGFs:

$$M_S(t) = \prod_{i=1}^{k} M_{X_i}(t) = \left(\frac{1}{1 - \beta t}\right)^k$$

This is exactly the MGF of a Gamma(k, β) distribution, proving the result.

### 2. Convolution Approach

The PDF of the sum of independent random variables is the **convolution** of their individual PDFs. For two Exponential($\beta$) variables:

$$f_{X_1 + X_2}(s) = \int_0^s f_{X_1}(x) \cdot f_{X_2}(s-x) \, dx$$

Repeated convolution of k exponential PDFs yields the Gamma(k, β) PDF:

$$f_S(s) = \frac{s^{k-1} e^{-s/\beta}}{\beta^k \Gamma(k)}$$

### 3. Poisson Process Interpretation

This is the most intuitive perspective:

- Consider a **Poisson process** with rate $\lambda = 1/\beta$
- Events occur randomly over time
- The **waiting time** between consecutive events follows Exponential($\beta$)
- The **total time** until the k-th event is the sum of k exponential waiting times
- This total time follows Gamma(k, β)

---

## The Erlang Distribution

When **k is a positive integer**, the Gamma(k, β) distribution is called the **Erlang distribution**, named after Danish mathematician Agner Krarup Erlang who developed it for telephone network analysis.

### Key Properties of Erlang(k, β)

| Property | Formula |
|----------|---------|
| Mean | $k \cdot \beta$ |
| Variance | $k \cdot \beta^2$ |
| Mode | $(k-1) \cdot \beta$ for $k \geq 1$ |
| Coefficient of Variation | $1/\sqrt{k}$ |

**Note:** As k increases, the distribution becomes more symmetric and approaches a Normal distribution (Central Limit Theorem).

---

## Practical Example: Waiting Time for Multiple Events

**Scenario:** A cell receives signaling molecules at random times, following a Poisson process with an average of 1 molecule every 2 minutes (rate = 0.5/min, scale = 2 min).

**Question:** What is the distribution of time until the cell receives 5 molecules?

**Answer:** The waiting time follows Gamma(5, 2):
- Mean waiting time: $5 \times 2 = 10$ minutes
- Variance: $5 \times 2^2 = 20$ minutes²
- Standard deviation: $\sqrt{20} \approx 4.47$ minutes

```python
import numpy as np
from scipy import stats

# Parameters
k = 5      # number of events
scale = 2  # average time between events (minutes)

# Distribution
dist = stats.gamma(a=k, scale=scale)

print(f"Mean waiting time: {dist.mean():.2f} minutes")
print(f"95% CI: [{dist.ppf(0.025):.2f}, {dist.ppf(0.975):.2f}] minutes")
print(f"P(wait > 15 min): {1 - dist.cdf(15):.4f}")
```

---

## Validation: Simulation vs Theory

The notebook demonstrates this relationship empirically:

```python
# Sum of k exponentials
exp_samples = np.random.exponential(scale, (n_samples, k))
sum_exp = exp_samples.sum(axis=1)

# Direct Gamma samples
gamma_samples = np.random.gamma(k, scale, n_samples)
```

Both approaches produce samples from the same distribution, as shown by:
1. Overlapping histograms
2. Matching sample means and variances
3. Agreement with theoretical values

---

## Applications in Causal ML and Biology

### 1. Survival Analysis

The Gamma distribution models time-to-event data when events require multiple "hits" or stages:
- Time until tumor development (multi-stage carcinogenesis)
- Time until drug resistance (multiple mutations required)

### 2. Queueing Theory

In clinical trials and healthcare:
- Time until k patients are enrolled
- Service time distributions in hospitals

### 3. Bayesian Inference

The Gamma distribution is the conjugate prior for:
- **Poisson rate parameter** (count data)
- **Exponential rate parameter** (waiting times)

This enables closed-form posterior updates in Bayesian causal models.

### 4. Overdispersion Modeling

When Poisson rates vary across units (heterogeneity), modeling the rate with a Gamma distribution leads to the **Negative Binomial** distribution — crucial for scRNA-seq and other count data with overdispersion.

---

## Key Takeaways

1. **Sum of k Exponentials = Gamma(k, β)** — a fundamental result
2. **Erlang distribution** = Gamma with integer shape (waiting time for k events)
3. **Interpretation:** Total waiting time in a Poisson process
4. **Applications:** Survival analysis, queueing, Bayesian inference, overdispersion

---

## References

- Ross, S. M. (2014). *Introduction to Probability Models*. Academic Press.
- Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press.
- See `02_gamma_distribution.ipynb` for interactive visualizations.
