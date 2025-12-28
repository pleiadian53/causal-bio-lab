# Simulating a Poisson Process: Two Methods

This document explains the mathematical foundations behind the two standard methods for simulating a Poisson process. Both methods produce statistically equivalent results, but they approach the problem from different angles.

**See also:** `01_poisson_process.ipynb` for interactive demonstrations of these methods.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Method 1: Inter-Arrival Times (Exponential)](#2-method-1-inter-arrival-times-exponential)
3. [Method 2: Count Then Scatter (Uniform)](#3-method-2-count-then-scatter-uniform)
4. [Why Both Methods Work](#4-why-both-methods-work)
5. [Comparison](#5-comparison)
6. [Code Examples](#6-code-examples)

---

## 1. Overview

To simulate a Poisson process with rate $\lambda$ over the interval $[0, T]$, we need to generate random event times $0 < S_1 < S_2 < \cdots < S_N < T$.

There are two equivalent approaches:

| Method | Idea | Key Distribution |
|--------|------|------------------|
| **Inter-arrival** | Generate gaps between events, then cumsum | Exponential($\lambda$) |
| **Count-then-scatter** | Generate total count, then place uniformly | Poisson($\lambda T$) + Uniform$(0,T)$ |

Both produce valid Poisson process realizations.

---

## 2. Method 1: Inter-Arrival Times (Exponential)

### The Idea

Generate the **waiting times between consecutive events**, then accumulate them to get event times.

### Mathematical Foundation

A fundamental property of the Poisson process:

> **The inter-arrival times $T_1, T_2, T_3, \ldots$ are i.i.d. Exponential($\lambda$).**

Where:
- $T_i$ = time between event $i-1$ and event $i$
- $T_i \sim \text{Exponential}(\lambda)$ with PDF $f(t) = \lambda e^{-\lambda t}$
- Mean inter-arrival time: $E[T_i] = 1/\lambda$

### Why Exponential?

The exponential distribution is the **unique** continuous distribution with the **memoryless property**:

$$P(T > s + t \mid T > s) = P(T > t)$$

This memorylessness is exactly what we need for independent increments in the Poisson process. If the waiting time had memory, events would cluster or repel — violating the Poisson assumptions.

### The Algorithm

1. Initialize: $t = 0$, event list = empty
2. Generate $W \sim \text{Exponential}(\lambda)$ (time until next event)
3. Update: $t = t + W$
4. If $t > T$: stop
5. Else: record $t$ as an event time, go to step 2

### Event Times from Inter-Arrivals

The $n$-th event time is:

$$S_n = T_1 + T_2 + \cdots + T_n = \sum_{i=1}^{n} T_i$$

Since each $T_i \sim \text{Exponential}(\lambda)$:

$$S_n \sim \text{Gamma}(n, 1/\lambda)$$

This connects to the Gamma distribution (see `02_gamma_exponential_relationship.md`).

### Pseudocode

```python
def simulate_poisson_exponential(rate, T):
    events = []
    t = 0
    while True:
        wait = exponential(1/rate)  # or exponential(scale=1/rate)
        t = t + wait
        if t > T:
            break
        events.append(t)
    return events
```

---

## 3. Method 2: Count Then Scatter (Uniform)

### The Idea

First determine **how many** events occur, then **place them uniformly** in the interval.

### Mathematical Foundation

This method relies on a remarkable property:

> **Given that $N(T) = n$ events occurred in $[0, T]$, the event times are distributed as the order statistics of $n$ i.i.d. Uniform$(0, T)$ random variables.**

In other words: conditional on the count, the events are "scattered uniformly" over the interval.

### Why This Works: The Conditional Distribution Theorem

**Theorem:** Let $\{N(t)\}$ be a Poisson process with rate $\lambda$. Given $N(T) = n$, the joint distribution of the event times $(S_1, S_2, \ldots, S_n)$ is the same as the order statistics of $n$ i.i.d. Uniform$(0, T)$ random variables.

**Proof sketch:**

The joint density of $(S_1, \ldots, S_n)$ given $N(T) = n$ is:

$$f(s_1, \ldots, s_n \mid N(T) = n) = \frac{n!}{T^n}$$

for $0 < s_1 < s_2 < \cdots < s_n < T$.

This is exactly the density of the order statistics of $n$ Uniform$(0, T)$ samples!

**Intuition:** The Poisson process has stationary increments — no time is "special." So given that $n$ events happened, they're equally likely to be anywhere in the interval.

### The Algorithm

1. Generate $N \sim \text{Poisson}(\lambda T)$ (total count)
2. Generate $N$ independent Uniform$(0, T)$ random variables
3. Sort them to get ordered event times

### Pseudocode

```python
def simulate_poisson_uniform(rate, T):
    n = poisson(rate * T)
    events = sort(uniform(0, T, size=n))
    return events
```

---

## 4. Why Both Methods Work

### They're Mathematically Equivalent

Both methods produce event times with the same distribution:

| Property | Method 1 (Exponential) | Method 2 (Uniform) |
|----------|------------------------|-------------------|
| Marginal count $N(T)$ | Poisson($\lambda T$) | Poisson($\lambda T$) by construction |
| Event times given count | Order statistics of Uniform | Order statistics of Uniform by construction |
| Inter-arrival times | Exponential($\lambda$) by construction | Exponential($\lambda$) (can be verified) |

### Different Perspectives, Same Process

- **Method 1** builds the process **forward in time** — natural for simulation and queueing
- **Method 2** takes a **bird's eye view** — natural for spatial processes and theoretical analysis

### Verification: Inter-Arrivals from Method 2

If you simulate using Method 2 and compute the gaps between sorted uniform points, you get:

$$T_i = S_i - S_{i-1} \sim \text{Exponential}(\lambda)$$

This is a consequence of the **spacing theorem** for order statistics of uniforms.

---

## 5. Comparison

### When to Use Each Method

| Criterion | Method 1 (Exponential) | Method 2 (Uniform) |
|-----------|------------------------|-------------------|
| **Conceptual clarity** | More intuitive for temporal processes | More intuitive for spatial processes |
| **Computational efficiency** | Variable number of random draws | Fixed: 1 Poisson + N uniforms |
| **Memory** | Stream-friendly (generate on demand) | Requires storing all events |
| **Extensions** | Easy for non-homogeneous rates | Harder for non-homogeneous |
| **Queueing simulation** | Natural fit | Less natural |

### Computational Considerations

**Method 1:**
- Number of random draws: $N + 1$ (where $N$ is random)
- Can stop early if only need events up to some time
- Natural for streaming/online simulation

**Method 2:**
- Number of random draws: $1 + N$ (Poisson + uniforms)
- Must generate all events at once
- Requires sorting (but sorting $N$ items is fast)

For most practical purposes, both methods are equally efficient.

---

## 6. Code Examples

### Method 1: Inter-Arrival Times

```python
import numpy as np

def simulate_poisson_exponential(rate, T):
    """
    Simulate Poisson process via exponential inter-arrival times.
    
    Parameters
    ----------
    rate : float
        Events per unit time (λ)
    T : float
        Time horizon
        
    Returns
    -------
    events : ndarray
        Sorted event times in [0, T]
    """
    events = []
    t = 0
    while True:
        # Time until next event ~ Exponential(rate)
        # Note: np.random.exponential takes scale = 1/rate
        wait = np.random.exponential(1/rate)
        t += wait
        if t > T:
            break
        events.append(t)
    return np.array(events)
```

### Method 2: Count Then Scatter

```python
import numpy as np

def simulate_poisson_uniform(rate, T):
    """
    Simulate Poisson process via count + uniform scatter.
    
    Parameters
    ----------
    rate : float
        Events per unit time (λ)
    T : float
        Time horizon
        
    Returns
    -------
    events : ndarray
        Sorted event times in [0, T]
    """
    # Step 1: How many events?
    n = np.random.poisson(rate * T)
    
    # Step 2: Where are they? (uniform, then sort)
    if n == 0:
        return np.array([])
    events = np.sort(np.random.uniform(0, T, n))
    return events
```

### Verification

```python
# Both methods should give same statistical properties
rate = 5
T = 100
n_simulations = 10000

counts_exp = [len(simulate_poisson_exponential(rate, T)) for _ in range(n_simulations)]
counts_unif = [len(simulate_poisson_uniform(rate, T)) for _ in range(n_simulations)]

print(f"Method 1 (Exponential): mean = {np.mean(counts_exp):.2f}, var = {np.var(counts_exp):.2f}")
print(f"Method 2 (Uniform):     mean = {np.mean(counts_unif):.2f}, var = {np.var(counts_unif):.2f}")
print(f"Theory:                 mean = {rate * T}, var = {rate * T}")
```

---

## Summary

### Two Equivalent Simulation Methods

| Method | Generate | Then |
|--------|----------|------|
| **1. Inter-arrival** | Exponential($\lambda$) gaps | Cumulative sum |
| **2. Count-scatter** | Poisson($\lambda T$) count | Uniform$(0,T)$ + sort |

### Key Mathematical Facts

1. **Inter-arrival times are Exponential** — fundamental property of Poisson process
2. **Conditional on count, events are uniform** — order statistics theorem
3. **Both methods produce identical distributions** — just different perspectives

### The Deep Connection

These two methods reflect the **duality** between:
- **Temporal view**: Events unfold one by one (exponential gaps)
- **Global view**: Events are scattered uniformly (given the count)

Understanding both perspectives deepens your intuition for stochastic processes.

---

## References

- Ross, S. M. (2014). *Introduction to Probability Models*. Chapter 5.
- See `01_poisson_process.ipynb` for interactive demonstrations.
- See `01_distribution_vs_process.md` for conceptual overview.
