# Why Causal Discovery Alone Doesn't Find Drug Targets

This document explores why causal discovery algorithms alone are likely insufficient for drug target identification—and what complementary approaches are needed.

> **Key insight:** Causal discovery tells you *what might regulate what*. It does not tell you *what to drug*.

---

## Table of Contents

1. [What Causal Discovery Actually Provides](#what-causal-discovery-actually-provides)
2. [What It Does NOT Provide](#what-it-does-not-provide)
3. [The Gap Between Structure and Targets](#the-gap-between-structure-and-targets)
4. [Failure Modes in Biological Data](#failure-modes-in-biological-data)
5. [What Industry Likely Does](#what-industry-likely-does)
6. [When Discovery IS Useful](#when-discovery-is-useful)
7. [The Right Mental Model](#the-right-mental-model)

---

## What Causal Discovery Actually Provides

Causal discovery algorithms (PC, GES, NOTEARS, etc.) learn **graph structure** from observational data:

- **Edges:** Gene A → Gene B (A regulates B)
- **Orientation:** Direction of causal influence
- **Conditional independencies:** What's independent given what

This is valuable for:

- Generating hypotheses about regulatory relationships
- Identifying candidate causal paths
- Understanding network topology

---

## What It Does NOT Provide

### 1. Drug Targets

A causal graph tells you that A → B → C → Disease.

It does **not** tell you:

- Which node is druggable
- Which intervention has the largest effect
- Whether the effect is robust across patients
- Whether the effect is specific (no off-target toxicity)

### 2. Effect Sizes

Knowing A → B exists says nothing about:

- How strong the effect is
- Whether it's clinically meaningful
- Whether it's large enough to matter for disease

### 3. Directionality You Can Trust

From observational data alone:

- Many edges remain **undirected** (Markov equivalence class)
- Latent confounders create spurious edges
- Feedback loops violate DAG assumptions

Without perturbation data, you cannot reliably orient edges.

### 4. Heterogeneity

A population-level graph doesn't capture:

- Patient-specific regulatory differences
- Cell-state-dependent effects
- Context-specific pathway activation

---

## The Gap Between Structure and Targets

Consider this scenario:

```text
Gene X → Pathway Y → Fibrosis Score
```

Causal discovery might correctly identify this path. But:

| Question | Discovery Answers? |
|----------|-------------------|
| Does knocking down X reduce fibrosis? | ❌ No |
| By how much? | ❌ No |
| In which patients? | ❌ No |
| Is X druggable? | ❌ No |
| Are there off-target effects? | ❌ No |

**The gap:** Structure ≠ Intervention effect ≠ Drug target

---

## Failure Modes in Biological Data

### 1. Latent Confounders

Gene expression data is riddled with unmeasured confounders:

- Batch effects
- Cell cycle state
- Donor effects
- Technical noise

These create spurious edges that look causal but aren't.

### 2. Feedback Loops

Biology is full of feedback:

- Transcription factor autoregulation
- Signaling pathway feedback
- Homeostatic mechanisms

Most discovery algorithms assume DAGs (no cycles). Feedback violates this.

### 3. High Dimensionality

With 20,000 genes and 10,000 cells:

- The search space is astronomical
- Statistical power is limited
- False discovery rates are high

### 4. Measurement Noise

scRNA-seq data has:

- Dropout (zero inflation)
- Technical variance
- Low counts for many genes

This degrades the conditional independence tests that discovery relies on.

### 5. Selection Bias

Your dataset is not a random sample of biology:

- Specific cell types
- Specific disease states
- Specific experimental conditions

Graphs learned here may not generalize.

---

## What Industry Likely Does

Based on public information from company websites and publications, here's what the workflow likely looks like:

### Ochre Bio's Stated Approach

From their public materials:

1. **Perturbation atlases:** Directly measure what happens when you knock down genes
2. **Patient disease atlases:** Observe disease phenotypes in human tissue
3. **Causal integration:** Link perturbation effects to disease outcomes
4. **Human validation:** Test in perfused livers, tissue slices

> "We make causal predictions about new drug targets" — their stated approach emphasizes combining perturbation data with disease data.

*Note: We can only speculate about internal methods based on public information.*

### A Plausible Industry Pattern

| Step | Method | Data Type |
|------|--------|-----------|
| 1. Hypothesis generation | Discovery + literature | Observational |
| 2. Effect estimation | Treatment effect methods | Perturbation |
| 3. Heterogeneity analysis | CATE estimation | Perturbation + patient |
| 4. Target ranking | Decision rules | Integrated |
| 5. Validation | Wet lab experiments | Interventional |

**Key insight:** Perturbation data is essential. You cannot skip it.

---

## When Discovery IS Useful

Causal discovery has legitimate uses:

### 1. Hypothesis Generation

Before expensive perturbation experiments, discovery can:

- Prioritize which genes to perturb
- Suggest pathway relationships to test
- Identify unexpected connections

### 2. Prior Knowledge Integration

Combine discovery with:

- Known pathway databases (KEGG, Reactome)
- Literature-derived relationships
- Expert knowledge

This constrains the search space and improves reliability.

### 3. Benchmarking

Discovery algorithms are useful for:

- Understanding method properties
- Comparing approaches on synthetic data
- Building intuition about causal inference

### 4. Exploratory Analysis

When you genuinely don't know the structure:

- New biological systems
- Understudied cell types
- Novel disease contexts

---

## The Right Mental Model

### Wrong Model

```text
Observational data → Causal discovery → Drug targets
```

### Right Model

```text
Observational data → Causal discovery → Hypotheses
                                            ↓
Perturbation data → Effect estimation → Candidate targets
                                            ↓
Patient data → Heterogeneity analysis → Prioritized targets
                                            ↓
Decision rules → Target ranking → Validation experiments
```

**The key difference:** Discovery is one input to a larger pipeline, not the answer.

---

## Implications for This Project

### What We Build

1. **Discovery module:** Learn structure, but document limitations
2. **Estimation module:** Quantify intervention effects (the core value)
3. **CATE module:** Identify responder subgroups
4. **Decision module:** Rank targets by effect size, robustness, specificity

### What We Emphasize

- Effect estimation > structure learning
- Perturbation data > observational data
- Uncertainty quantification > point estimates
- Decision rules > model outputs

### What We Document

- Failure modes and when methods break
- Assumptions and when they're violated
- Limitations and what we can't conclude

---

## Summary

| Causal Discovery | Treatment Effect Estimation |
|-----------------|---------------------------|
| Learns structure | Quantifies effects |
| Observational data | Perturbation data |
| Hypotheses | Actionable predictions |
| Many false positives | Validated effects |
| Population-level | Can be heterogeneous |

**Bottom line:** Causal discovery is a hypothesis generator, not a target finder. The real work happens in effect estimation and validation.

---

## References

- Ochre Bio: [Human Discovery Platform](https://www.ochre-bio.com/)
- Relation Therapeutics: [Lab-in-the-Loop](https://www.relationrx.com/)
- insitro: [POSH Platform](https://insitro.com/)
- [CausalBench](https://www.gsk.ai/causalbench-challenge/) — Benchmark showing discovery limitations
- Peters, Janzing, Schölkopf — *Elements of Causal Inference*
