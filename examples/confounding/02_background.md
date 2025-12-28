# Biological Background for Confounding Simulations

This document provides biological context for the confounding examples in `02_confounding_simulations.py`. Each section explains the biology behind a specific demo function.

---

## Table of Contents

1. [Cell Cycle Confounding](#1-cell-cycle-confounding-myc-vs-ribosomal-genes)
2. [Batch Effect Confounding](#2-batch-effect-confounding-technical-artifacts)
3. [Disease Severity Confounding](#3-disease-severity-confounding-the-protective-gene-trap)
4. [Treatment Effect Estimation](#4-treatment-effect-estimation-with-confounding)

---

## 1. Cell Cycle Confounding: MYC vs Ribosomal Genes

**Demo function:** `demo_cell_cycle_confounding()`

### The Cell Cycle Phases

The cell cycle is the process a cell goes through to divide into two daughter cells:

```text
G1 → S → G2 → M → (back to G1)
```

| Phase | Name | What Happens | Metabolic State |
|-------|------|--------------|-----------------|
| **G1** | Gap 1 | Cell grows, prepares for replication | Quieter |
| **S** | Synthesis | DNA replication | High demand |
| **G2** | Gap 2 | Checks DNA, prepares for division | High activity |
| **M** | Mitosis | Chromosomes separate, cell divides | Peak |

### Why MYC and Ribosomal Genes Correlate

**MYC (the "treatment" in our simulation):**

- Master regulator transcription factor
- Drives cell growth and proliferation
- Peaks at G1/S transition when cells commit to division
- Stays elevated through S phase, declines in G2/M

**Ribosomal genes (the "outcome"):**

- Encode ribosomal RNA and ribosomal proteins
- Ribosomes are the protein factories of the cell
- Dividing cells need more ribosomes to duplicate cellular machinery
- Expression spikes during S/G2 phases

### The Confounding Structure

```text
           Cell Cycle Phase (Z)
                 /  \
                /    \
               ↓      ↓
         MYC (C)   Ribosomal Genes (X)
```

**The observation:** High MYC ↔ High ribosomal genes (strong correlation!)

**The naive interpretation:** "MYC directly activates ribosomal genes!"

**The reality:**

- Cell cycle phase drives **both** MYC and ribosomal genes
- MYC does have a small direct effect (it binds rDNA promoters)
- But most of the correlation is driven by shared cell cycle regulation

**The consequence:** Knocking down MYC may reduce ribosomal genes less than expected, because cell cycle is the dominant driver.

> **Note:** This is a very real problem in scRNA-seq analysis. Cell cycle is one of the most common confounders when studying gene-gene relationships.

---

## 2. Batch Effect Confounding: Technical Artifacts

**Demo function:** `demo_batch_effect_confounding()`

### What Are Batch Effects?

In single-cell RNA-seq, samples processed on different days, lanes, or with different reagent lots can have systematic differences unrelated to biology:

| Batch Factor | Effect |
|--------------|--------|
| **Sequencing depth** | More reads → higher counts for all genes |
| **Capture efficiency** | Better capture → more transcripts detected |
| **Reagent lot** | Different amplification efficiency |
| **Operator** | Handling differences |
| **Time of day** | Cell stress from processing |

### How Batch Effects Create Spurious Correlations

```text
        Batch Effect (Z)
           /      \
          ↓        ↓
    Gene A (C)   Gene B (X)
```

**The scenario:**

- Gene A and Gene B are **causally independent**
- But batch effects inflate/deflate all genes together
- High-efficiency batch → both genes appear higher
- Low-efficiency batch → both genes appear lower

**The observation:** Gene A and Gene B are correlated!

**The reality:** They have no biological relationship. The correlation is purely technical.

### Why This Matters

- **Co-expression networks** can be dominated by batch effects
- **Differential expression** can reflect batch, not biology
- **Gene regulatory inference** can identify spurious edges

### Solutions

- **Batch correction methods:** ComBat, Harmony, scVI
- **Include batch as covariate** in statistical models
- **Randomize conditions across batches** when possible
- **Use perturbation data:** The perturbation is randomized within batches

---

## 3. Disease Severity Confounding: The Protective Gene Trap

**Demo function:** `demo_disease_severity_confounding()`

### The Biological Scenario

Consider a stress response gene like **HIF1A** (Hypoxia-Inducible Factor 1-alpha):

**What HIF1A does:**

- Activated when cells experience low oxygen (hypoxia)
- Triggers adaptive responses: glycolysis, angiogenesis, survival pathways
- Generally **protective**—helps cells survive stress

**The paradox:**

- In disease, HIF1A is often **elevated**
- Cell death is also **elevated**
- Observationally: HIF1A correlates with death

### The Confounding Structure

```text
      Disease Severity (Z)
           /        \
          ↓          ↓
    HIF1A (C)    Cell Death (X)
```

**What's happening:**

- Severe disease causes hypoxia → HIF1A activation
- Severe disease causes tissue damage → cell death
- HIF1A is trying to **protect** cells from dying

### The Dangerous Misinterpretation

**Naive conclusion:** "HIF1A causes cell death! Let's inhibit it!"

**Reality:** HIF1A is a **marker** of severity, not a **cause** of death. It's actually protective.

**Consequence of inhibition:** Cells lose their stress response → **more death**, not less.

### Real-World Examples

This pattern appears frequently in biology:

| Gene | Correlates With | But Actually Is |
|------|-----------------|-----------------|
| HIF1A | Cell death | Protective |
| Heat shock proteins | Protein aggregation | Protective |
| Autophagy genes | Cell stress | Protective |
| DNA repair genes | DNA damage | Protective |

> **Key insight:** Stress response genes correlate with bad outcomes because they're **markers of stress**, not causes of harm.

---

## 4. Treatment Effect Estimation with Confounding

**Demo function:** `demo_treatment_effect_estimation()`

### The Scenario: Job Training and Earnings

This example uses a classic causal inference scenario:

- **Treatment:** Taking a job training course
- **Outcome:** Earnings after the course
- **Confounder:** Age

### Why Age Confounds

```text
           Age (Z)
          /      \
         ↓        ↓
  Training (C)   Earnings (X)
```

**Age affects treatment:**

- Younger people are more likely to take training
- They have more career runway, more motivation to upskill
- Older workers may feel "too late" to retrain

**Age affects outcome:**

- Older workers have higher baseline earnings
- More experience → higher salary
- Seniority effects

### The Bias

**Naive comparison:**

- Compare earnings: treated vs. control
- Treated group is younger on average
- Younger people have lower baseline earnings
- This creates **negative bias** in the treatment effect estimate

**What we observe:**

- Naive ATE estimate is **lower** than the true effect
- Because we're comparing younger (treated) to older (control)

### The Fix

**Adjustment methods:**

- Include age as a covariate in regression
- Propensity score matching/weighting
- Doubly robust estimation

These methods "control for" age, giving us a less biased estimate of the training effect.

---

## Summary: Recognizing Confounding Patterns

| Example | Confounder | Treatment | Outcome | Trap |
|---------|------------|-----------|---------|------|
| Cell cycle | Cell cycle phase | MYC | Ribosomal genes | Correlation ≠ regulation |
| Batch effects | Technical batch | Gene A | Gene B | Artifact looks biological |
| Disease severity | Severity | HIF1A | Cell death | Marker ≠ cause |
| Job training | Age | Training | Earnings | Selection bias |

**The common thread:** Something upstream affects both the "treatment" and the "outcome," creating a correlation that doesn't reflect the true causal effect.

---

## References

- Regev et al., "The Human Cell Atlas" — Discusses batch effects in scRNA-seq
- Semenza, "HIF-1: mediator of physiological and pathophysiological responses to hypoxia"
- Hernán & Robins, *Causal Inference: What If* — Treatment effect estimation
- Scanpy documentation — Cell cycle scoring and batch correction