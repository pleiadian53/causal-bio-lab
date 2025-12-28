# Beyond the Poisson Process: When Reality Gets Complicated

We're now moving from the **static snapshot** world into the **dynamics of reality**. This tutorial introduces four generalizations of the Poisson process, each designed to handle a specific violation of the Poisson assumptions.

**Structure:** For each model, we'll cover:
1. What assumption it relaxes
2. Mathematical setup (notation first)
3. Why the Poisson distribution alone is insufficient
4. A concrete computational biology example

---

## Table of Contents

1. [The Baseline: Homogeneous Poisson Process](#1-the-baseline-homogeneous-poisson-process)
2. [Non-Homogeneous Poisson Process (NHPP)](#2-non-homogeneous-poisson-process-nhpp)
3. [Cox Process (Doubly Stochastic Poisson)](#3-cox-process-doubly-stochastic-poisson)
4. [Hawkes Process (Self-Exciting)](#4-hawkes-process-self-exciting)
5. [Renewal Processes](#5-renewal-processes)
6. [The Conceptual Ladder](#6-the-conceptual-ladder)
7. [FAQ: Process-Level Generalizations](#7-faq-process-level-generalizations)

---

## 1. The Baseline: Homogeneous Poisson Process

Let $N(t)$ be the number of events up to time $t$.

A **homogeneous Poisson process** assumes:

- A **constant rate** $\lambda$
- **Independent, memoryless** gaps between events
- **No influence** of past events on future ones

This gives:

$$N(t) \sim \text{Poisson}(\lambda t)$$

**Everything that follows exists because real systems violate at least one of these assumptions.**

| Assumption | What could go wrong? |
|------------|---------------------|
| Constant rate | Rate varies with time, space, or context |
| Known rate | Rate is uncertain or random |
| Independence | Events trigger other events |
| Memoryless gaps | System remembers past events |

---

## 2. Non-Homogeneous Poisson Process (NHPP)

### The Question

> *"What if the rate changes with time (or space)?"*

### Mathematical Setup

Instead of a constant rate $\lambda$, define a **rate function**:

$$\lambda(t) \geq 0$$

**Interpretation:** $\lambda(t) \, dt$ is approximately the expected number of events in the tiny interval $[t, t+dt]$.

Define the **cumulative intensity** (also called the integrated intensity):

$$\Lambda(t) = \int_0^t \lambda(s) \, ds$$

Then the counting variable satisfies:

$$N(t) \sim \text{Poisson}(\Lambda(t))$$

**Key observations:**

- Counts are *still Poisson*
- But the mean is **no longer linear** in $t$
- The rate function $\lambda(t)$ encodes *when* events are likely

### Why Poisson Distribution Alone Is Insufficient

If I only told you:

$$N(24\text{h}) \sim \text{Poisson}(240)$$

You would not know whether:

- Arrivals were **uniform** throughout the day
- **Concentrated in bursts** during peak hours
- **Absent overnight**

The **process** explains *how* those 240 events are distributed across time.

### Computational Biology Example: DNA Damage Under Circadian Regulation

DNA double-strand breaks occur at different rates depending on:

- **Replication phase** (S-phase has high replication stress)
- **Circadian rhythm** (repair enzymes fluctuate)
- **Drug exposure timing** (chemotherapy scheduling)

A simple model:

$$\lambda(t) = \begin{cases} 0.1 & \text{night (repair active)} \\ 0.6 & \text{S-phase (replication stress)} \end{cases}$$

Two 10-hour windows can have:

- The same *expected* number of breaks
- Wildly different **temporal clustering**

A Poisson distribution with mean $\lambda t$ cannot encode this. The NHPP can.

---

## 3. Cox Process (Doubly Stochastic Poisson)

### The Question

> *"What if the rate itself is random?"*

Now we relax a deeper assumption: that the rate is **known and fixed**.

### Mathematical Setup

Let $\Lambda(t)$ be a **random process** (not a fixed function).

Conditioned on a realization of $\Lambda(t)$:

$$N(t) \mid \Lambda \sim \text{Poisson}\left(\int_0^t \Lambda(s) \, ds\right)$$

**Unconditionally** (averaging over all possible rate realizations):

- Counts are **overdispersed** (variance > mean)
- Clustering appears *even without event-to-event interaction*
- The distribution is no longer Poisson

### Key Intuition: Two Levels of Randomness

| Level | Source of randomness |
|-------|---------------------|
| **Poisson** | Randomness in event timing (given the rate) |
| **Cox** | Randomness in the *environment* (the rate itself) |

The Cox process adds uncertainty about the world, not just about events.

### Computational Biology Example: Transcription Bursts in scRNA-seq

In single-cell RNA-seq:

- $\Lambda(t)$ = latent transcriptional activity of a gene
- $\Lambda(t)$ fluctuates due to chromatin accessibility, TF binding, cell state

**Conditioned on a fixed transcription rate:** counts are Poisson

**Across cells** (where rates vary):

- Counts become heavy-tailed
- Variance $\gg$ mean (overdispersion)
- Negative Binomial often fits well

This is why:

- **Poisson fails** for scRNA-seq data
- **Negative Binomial works** as an empirical fix
- **Cox process provides the generative explanation**

> The Poisson **distribution** tells you *what the counts look like*.
> The Cox **process** tells you *why*.

---

## 4. Hawkes Process (Self-Exciting)

### The Question

> *"What if events cause more events?"*

This breaks the **independence** assumption.

### Mathematical Setup

Define a **conditional intensity** that depends on past events:

$$\lambda(t) = \mu + \sum_{t_i < t} g(t - t_i)$$

Where:

- $\mu$ = **baseline rate** (spontaneous events)
- $t_i$ = times of previous events
- $g(\cdot)$ = **excitation kernel** (often exponential decay)

**Interpretation:** Each event temporarily increases the probability of future events.

A common choice for the kernel:

$$g(\tau) = \alpha \cdot e^{-\beta \tau}$$

- $\alpha$ = excitation strength (how much each event boosts the rate)
- $\beta$ = decay rate (how quickly the effect fades)

### Why Poisson Cannot Do This

| Model | Assumption about past |
|-------|----------------------|
| **Poisson** | Past events do not affect the future |
| **Hawkes** | Events leave echoes that trigger more events |

### Computational Biology Example: Neuronal Spike Trains

In neural recordings:

- One spike increases membrane potential
- Short-term facilitation raises firing probability
- **Bursts emerge naturally**

Poisson can match the **total spike count**, but it cannot reproduce:

- **Bursting patterns**
- **Temporal correlations** between spikes
- **Refractory vs. excitation** dynamics

Hawkes models the **causal dynamics**, not just the histogram.

### General Domain Example: Earthquakes and Aftershocks

- Main shock triggers aftershocks
- Aftershocks trigger more aftershocks
- Intensity decays over time

Poisson distribution might fit the *daily count*, but only Hawkes explains the **cascade structure**.

---

## 5. Renewal Processes

### The Question

> *"What if waiting times are not memoryless?"*

Now we break the **exponential gap** assumption.

### Mathematical Setup

Let the inter-arrival times be i.i.d. from some distribution $F$:

$$T_1, T_2, \ldots \stackrel{\text{i.i.d.}}{\sim} F$$

where $F$ is **not exponential**.

Define event times and the counting process:

$$S_n = \sum_{i=1}^n T_i, \quad N(t) = \max\{n : S_n \leq t\}$$

Now:

- Gaps have **memory**
- History matters for predicting the next event
- Counts are **no longer Poisson**

### Common Choices for $F$

| Distribution | Property | Use case |
|--------------|----------|----------|
| **Gamma** | Regularity (less variable than exponential) | Cell cycle timing |
| **Weibull** | Aging effects (increasing/decreasing hazard) | Equipment failure |
| **Log-normal** | Heavy tails | Reaction times |

### Computational Biology Example: Refractory Periods in Ion Channels

Ion channel dynamics:

- Channel opens (event)
- Must stay **closed for a minimum time** (refractory period)
- Re-opening probability depends on how long it has rested

Exponential gaps fail because they allow **immediate re-opening**.

A renewal process with **Gamma** or **Weibull** gaps captures the refractory behavior.

---

## 6. The Conceptual Ladder

Think of the Poisson process as the **zero-memory baseline**.

Each extension answers one question:

| Model | Question it answers |
|-------|---------------------|
| **NHPP** | What if the world changes over time? |
| **Cox** | What if the world is uncertain? |
| **Hawkes** | What if events influence each other? |
| **Renewal** | What if the system remembers? |

### Summary Comparison

| Model | Memory | Event Interaction | Rate Variability |
|-------|--------|-------------------|------------------|
| **Poisson** | None | None | None (constant) |
| **NHPP** | None | None | Deterministic $\lambda(t)$ |
| **Cox** | None | None | Random $\Lambda(t)$ |
| **Hawkes** | Yes | Yes (self-exciting) | Endogenous |
| **Renewal** | Yes | No | Implicit in gap distribution |

### The Key Insight

The Poisson **distribution** only answers:

> "How many?"

The Poisson **process and its descendants** answer:

> "How, when, and why?"

In computational biology, once you start caring about **mechanisms** — regulation, feedback, heterogeneity — you inevitably move beyond Poisson distributions into **process thinking**.

---

## 7. FAQ: Process-Level Generalizations

This section addresses deeper questions about the relationship between distributions and processes.

---

### Q1: Is there a Negative Binomial process?

#### The Short Answer

> There is **no unique canonical process** called "the negative binomial process" in the same way there is a Poisson process.

But...

> **Yes**, there *are* stochastic processes whose **marginal counts are negative binomial**, and they arise naturally by randomizing the Poisson process rate.

#### Why NB Is Derived, Not Primitive

First, recall the **Poisson-Gamma mixture**:

- $Y \mid \Lambda \sim \text{Poisson}(\Lambda)$
- $\Lambda \sim \text{Gamma}(\alpha, \beta)$

Then marginally:

$$Y \sim \text{Negative Binomial}(\alpha, \beta)$$

**Interpretation:**

- Poisson = randomness in event counts
- Gamma = randomness in the *rate*
- NB = uncertainty about the environment, collapsed into counts

This hints that NB is **not a primitive object**, but a *collapsed view* of something richer.

#### The Process-Level Generalization: Cox Process

The correct generalization is the **Cox process** (doubly stochastic Poisson):

**Step 1:** Let $N(t) \mid \Lambda(\cdot)$ be a Poisson process with **random intensity** $\Lambda(t)$.

$$N(t) \mid \Lambda \sim \text{Poisson}\left(\int_0^t \Lambda(s) \, ds\right)$$

**Step 2:** Specialize to a constant but random rate:

$$\Lambda(t) = \lambda \quad \text{(constant in time)}, \quad \lambda \sim \text{Gamma}(\alpha, \beta)$$

Then:

$$N(t) \mid \lambda \sim \text{Poisson}(\lambda t)$$

Marginalizing over $\lambda$:

$$N(t) \sim \text{Negative Binomial}$$

> A **negative binomial count process** is simply a **Poisson process with a Gamma-distributed rate**.

#### Why There's No Standalone "NB Process"

The Poisson process is fundamental because:

- It has **independent increments**
- It has a simple generative mechanism (exponential gaps)
- It is the **unique memoryless** counting process

The NB breaks independence at the process level:

- Increments are **no longer independent**
- Correlation appears because the *same random rate* affects all intervals

That correlation is why NB is **derived**, not primitive.

---

### Q2: Is there a Gamma process?

**Yes** — and this is where things get elegant.

#### Definition

A **Gamma process** $\{G(t)\}_{t \geq 0}$ is a stochastic process such that:

1. $G(0) = 0$
2. $G(t)$ has **independent increments**
3. For $s < t$:

$$G(t) - G(s) \sim \text{Gamma}(\alpha (t-s), \beta)$$

**Interpretation:**

- It is a **non-decreasing random measure**
- It accumulates *random mass* over time or space
- It is the continuous-time analogue of a Gamma random variable

**Crucially:**

- The Gamma process is **not a counting process** (it takes continuous values)
- It models *random intensity*, not discrete events

#### The Deep Connection

Here is the key hierarchy:

$$\text{Gamma process} \;\Rightarrow\; \text{Cox process} \;\Rightarrow\; \text{NB counts}$$

More explicitly:

1. Draw a Gamma process $G(t)$
2. Use its increments as intensity: $\Lambda(t) = \frac{dG(t)}{dt}$
3. Conditioned on $G$, generate a Poisson process
4. Marginally, counts become Negative Binomial-like

> **The Gamma process is the process-level object that generalizes the Gamma distribution**, just like the Poisson process generalizes the Poisson distribution.

> **NB is what you see when you collapse a Poisson-Gamma hierarchy into counts.**

---

### Q3: How does this apply to scRNA-seq?

#### The Generative Story

A common model for single-cell RNA-seq (UMI counts):

1. Each gene has a **latent expression intensity** (varies across cells)
2. That intensity fluctuates due to chromatin state, TF binding, cell cycle
3. **Conditional on intensity** → Poisson counts
4. **Across cells** → overdispersion

#### Process View

| Component | Role |
|-----------|------|
| **Gamma process** | Models latent transcriptional activity |
| **Poisson process** | Models molecule capture/counting |
| **Negative Binomial** | Emerges as the marginal count distribution |

This explains:

- Why **NB fits** scRNA-seq data
- Why **Poisson fails** (ignores rate heterogeneity)
- Why **rate uncertainty**, not molecule interaction, is the dominant source of variance

---

### Q4: Why does this matter conceptually?

#### The Common Mistake

> "NB is just a better Poisson."

#### The Correct View

> "NB is a *collapsed view* of a Poisson process living in a random environment."

Once you think in processes instead of distributions:

- **Overdispersion** stops being mysterious
- **Biological heterogeneity** becomes explicit
- **Modeling choices** become mechanistic, not cosmetic

---

## Summary: The Mental Map

| Object | What it represents |
|--------|-------------------|
| **Poisson distribution** | Snapshot of counts |
| **Poisson process** | Event dynamics over time |
| **Gamma distribution** | Uncertainty in rate |
| **Gamma process** | Evolving uncertainty over time |
| **Cox process** | Poisson process driven by random environment |
| **Negative Binomial** | What you see after marginalizing out the randomness |

> There is no standalone "negative binomial process" — because the *right* object is richer: a **Poisson process driven by a Gamma process**.

And once you see that, NB stops being a trick and starts being a story.

---

## References

- Ross, S. M. (2014). *Introduction to Probability Models*. Chapter 5.
- Kingman, J. F. C. (1993). *Poisson Processes*. Oxford University Press.
- See `01_distribution_vs_process.md` for the Poisson distribution vs. process distinction.
- See `01_poisson_process.ipynb` for interactive demonstrations.
