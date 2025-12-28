# Poisson Process: Frequently Asked Questions

This document addresses deeper conceptual questions about the Poisson process. These questions are where the Poisson process stops being a formula and becomes a way of thinking about randomness.

**Prerequisites:** Read `01_distribution_vs_process.md` first.

---

## Table of Contents

1. [Does a Poisson process have to live in time?](#1-does-a-poisson-process-have-to-live-in-time)
2. [Why is the exponential distribution memoryless?](#2-why-is-the-exponential-distribution-memoryless)
3. [Why do memoryless gaps imply Poisson counts?](#3-why-do-memoryless-gaps-imply-poisson-counts)
4. [If the distribution gives counts, why do we need the process?](#4-if-the-distribution-gives-counts-why-do-we-need-the-process)

---

## 1. Does a Poisson process have to live in time?

**Short answer: No.** Time is just the most familiar axis.

### The General Principle

A Poisson process is fundamentally about **events distributed randomly over a measure space**, not about clocks.

Time just happens to be a 1-dimensional measure we intuitively understand.

### Generalization

Replace "time" with any domain where "size" makes sense:

| Domain | Unit | Example application |
|--------|------|---------------------|
| Time | seconds, hours | Customer arrivals |
| Genomic position | base pairs | Mutation locations |
| Physical space | area, volume | Cell positions in tissue |
| Network | edges, nodes | Random graph models |

Then replace "rate per unit time" with **"intensity per unit measure"**.

### Example: Mutations Along the Genome

**Setup:**

- Events = mutations
- Domain = genomic coordinate (base pairs)
- Intensity $\lambda$ = mutations per base pair

**Model:**

Let $N(x)$ = number of mutations in a genomic region of length $x$ base pairs.

$$N(x) \sim \text{Poisson}(\lambda x)$$

This is a **spatial Poisson process** (or **Poisson point process**).

### Key Properties Still Hold

- **Independence:** Counts in disjoint regions are independent
- **Stationarity:** Equal-length regions have the same distribution
- **Scaling:** Expected count = intensity × region size

### Applications in Biology

This is why Poisson models appear in:

- **ChIP-seq:** Read counts in genomic windows
- **Mutation burden analysis:** Somatic mutations across the genome
- **scRNA-seq:** UMI counts per gene (with overdispersion caveats)
- **Spatial transcriptomics:** Cell type distributions in tissue

**Bottom line:** Time is optional. Measure is essential.

---

## 2. Why is the exponential distribution memoryless?

This is the conceptual heart of the theory.

### What "Memoryless" Means Mathematically

A non-negative random variable $T$ is **memoryless** if:

$$P(T > s + t \mid T > s) = P(T > t)$$

**Interpretation:** If you've already waited $s$ units without an event, your remaining waiting time distribution is unchanged. The system doesn't "remember" that you've been waiting.

- No aging
- No buildup
- No fatigue

### Proof That Exponential is Memoryless

For an exponential random variable with rate $\lambda$:

$$P(T > t) = e^{-\lambda t}$$

Check the memoryless property:

$$P(T > s + t \mid T > s) = \frac{P(T > s + t)}{P(T > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t)$$

The conditional probability equals the unconditional probability — no dependence on $s$.

### Uniqueness Theorem

**In continuous time, the exponential distribution is the *only* memoryless distribution.**

This is not an assumption — it's a mathematical theorem. If you want memoryless waiting times in continuous time, you *must* use the exponential distribution.

### Intuitive Interpretation

Think of a radioactive atom:

- It doesn't "age" — a 1000-year-old atom has the same decay probability as a new one
- Each instant, there's a constant probability of decay
- This constant hazard rate implies exponential waiting times

---

## 3. Why do memoryless gaps imply Poisson counts?

This is the key connection that makes everything work.

### Setup

Let:

- $T_1, T_2, T_3, \ldots$ = waiting times between consecutive events
- Each $T_i \sim \text{Exponential}(\lambda)$ (i.i.d.)

Define event times (arrival times):

$$S_n = T_1 + T_2 + \cdots + T_n$$

Define the counting process:

$$N(t) = \max\{n : S_n \leq t\}$$

(The number of events by time $t$)

### The Theorem

> **If waiting times are i.i.d. Exponential($\lambda$), then $N(t) \sim \text{Poisson}(\lambda t)$.**

### Why This Works (Intuition)

The exponential properties translate directly to Poisson process properties:

| Exponential property | Implies |
|---------------------|---------|
| Memoryless | Independent increments |
| Constant hazard rate $\lambda$ | Stationary increments |
| i.i.d. gaps | No clustering or inhibition |
| Continuous support | No simultaneous events |

These four properties **force** the Poisson distribution on the counts.

### The Deep Insight

> **Poisson is not chosen — it emerges.**

The structure of waiting times uniquely determines the count distribution. This is why Poisson processes appear "inevitable" in nature whenever:

1. Events occur independently
2. The rate is constant
3. There's no memory

---

## 4. If the distribution gives counts, why do we need the process?

This is subtle but important.

### What the Poisson Distribution Alone Gives You

If arrivals average 12 per hour, then:

$$N(t) \sim \text{Poisson}(12t)$$

You *can* calculate:

- Expected customers in 3 hours: $12 \times 3 = 36$
- $P(\text{exactly 40 customers in 3 hours})$
- Variance: 36 (equals mean for Poisson)

That's all correct. But notice what's missing.

### What the Process Gives You That the Distribution Cannot

The distribution alone **does not tell you**:

| Question | Requires process? |
|----------|-------------------|
| How many events in an interval? | No (distribution suffices) |
| *When* do events occur? | **Yes** |
| How do arrivals cluster in time? | **Yes** |
| How to simulate realistic arrival streams? | **Yes** |
| How arrivals interact with queues/services? | **Yes** |
| Conditional questions ("given no arrivals in 10 min...") | **Yes** |

### Concrete Example

> "Given that no customer has arrived in the last 20 minutes, how long until the next customer?"

**Answer using the process:**

By the memoryless property of exponential inter-arrival times:

$$P(\text{wait} > t \mid \text{already waited 20 min}) = P(\text{wait} > t) = e^{-\lambda t}$$

The answer is: **Exponential($\lambda$)** — the same as if you just started waiting!

This question is **meaningless** without a process model. The distribution only gives you counts, not conditional timing.

### The Analogy

| Model | Analogy |
|-------|---------|
| Poisson distribution | Photograph (snapshot of counts) |
| Poisson process | Video (full temporal dynamics) |

Same scene. Very different information content.

### When You Need the Process

Use the **process** when you care about:

- Arrival times, not just counts
- Simulating event streams
- Queueing and service systems
- Conditional probabilities given partial observations
- Joint behavior across multiple time intervals

Use the **distribution** when you only need:

- Expected count in a fixed interval
- Probability of specific count values
- Variance of counts

---

## Summary: A Unifying Mental Picture

| Concept | Role |
|---------|------|
| Poisson **distribution** | Marginal count (projection) |
| Poisson **process** | Generative mechanism (primary) |
| Exponential gaps | The engine |
| Memorylessness | The symmetry principle |
| Time/space/genome | Interchangeable axes |

Once you internalize that the **process is primary** and the **distribution is a projection**, everything lines up — including why these models keep reappearing in biology, physics, and queueing theory.

---

## Where This Leads

The homogeneous Poisson process assumes constant rate. Relaxing this leads to:

| Extension | Key change | Application |
|-----------|------------|-------------|
| **Non-homogeneous Poisson** | Rate $\lambda(t)$ varies | Time-varying mutation rates |
| **Cox process** | Rate is random | Heterogeneous populations |
| **Hawkes process** | Events cause events | Contagion, neural spikes |
| **Renewal process** | Non-exponential gaps | Aging, memory effects |

These are all answers to the question:

> *"What if the world remembers?"*

---

## References

- Ross, S. M. (2014). *Introduction to Probability Models*. Academic Press.
- Kingman, J. F. C. (1992). *Poisson Processes*. Oxford University Press.
- See `01_poisson_process.ipynb` for interactive demonstrations.
