# causal-bio-lab

**Causal AI/ML for Computational Biology**: Research into causal inference, causal discovery, and causal representation learning for drug discovery, target identification, and treatment effect estimation.

## Overview

This project investigates causal machine learning approaches across computational biology, inspired by emerging platforms such as:

- **Causal Inference Platforms**: [biotx.ai](https://www.biotx.ai/) (causal genome mapping), [insitro](https://insitro.com/) (POSH platform)
- **Target Discovery**: [Ochre Bio](https://www.ochre-bio.com/) (liver disease), [Relation Therapeutics](https://www.relationrx.com/) (Lab-in-the-Loop)
- **Perturbation Biology**: [GEARS](https://github.com/snap-stanford/GEARS), [CPA](https://github.com/theislab/cpa) (perturbation response prediction)
- **Federated Causal Inference**: [Owkin](https://www.owkin.com/) (FedECA for clinical trials)

**Research Goals:**

1. **Learn** state-of-the-art causal discovery algorithms (PC, GES, NOTEARS) for gene regulatory network inference
2. **Implement** treatment effect estimation methods (ATE, ITE, CATE) for biological interventions
3. **Explore** counterfactual reasoning for perturbation response prediction and drug combination effects
4. **Investigate** causal representation learning and its connection to generative models

See [docs/INDUSTRY_LANDSCAPE.md](docs/INDUSTRY_LANDSCAPE.md) for a comprehensive survey of companies and technologies in this space.

## Project Structure

```text
causal-bio-lab/
├── src/causalbiolab/
│   ├── data/           # Data loading, preprocessing, path management
│   │   ├── paths.py        # Standardized data path management
│   │   ├── sc_preprocess.py    # scRNA-seq/Perturb-seq preprocessing
│   │   └── bulk_preprocess.py  # Bulk RNA-seq preprocessing
│   ├── discovery/      # Causal graph learning (PC, NOTEARS, etc.)
│   ├── estimation/     # Treatment effect estimation (ATE, ITE, CATE)
│   ├── counterfactual/ # Counterfactual prediction, perturbation response
│   ├── representation/ # Causal representation learning, identifiable VAEs
│   └── utils/          # Config, reproducibility
├── data/               # Local data storage (gitignored)
│   ├── perturbation/   # Perturb-seq, CRISPR screens
│   ├── observational/  # GTEx, TCGA, drug response
│   └── synthetic/      # SERGIO, CausalBench benchmarks
├── tests/
├── examples/
├── configs/
├── docs/
└── environment.yml     # Conda environment specification
```

## Installation

### Using mamba + poetry (recommended)

```bash
# Create conda environment
mamba create -n causalbiolab python=3.11 -y
mamba activate causalbiolab

# Install poetry if not available
pip install poetry

# Install package in editable mode
poetry install

# Optional: install causal inference dependencies
poetry install --with causal

# Optional: install dev dependencies
poetry install --with dev
```

### Quick start

```bash
# Verify installation
python -c "import causalbiolab; print(causalbiolab.__version__)"

# Run example (once implemented)
python examples/01_causal_discovery.py
```

## Milestones

### Milestone 0: Foundational Tutorials & Documentation ✅

- [x] **Causal Inference Tutorials**
  - [x] Treatment effects and potential outcomes framework
  - [x] Propensity score methods and IPW (inverse probability weighting)
  - [x] Do-calculus tutorial document (comprehensive guide with examples)
  - [ ] Do-calculus interactive notebook (hands-on exercises and applications)
  - [x] Identifying confounders and adjustment strategies
- [x] **Simulation Framework**
  - [x] Confounding simulation utilities
  - [x] Treatment effect estimation examples
  - [x] Cell cycle, batch effect, and disease severity confounders
- [x] **Notebooks**
  - [x] A/B testing fundamentals and multi-group comparisons
  - [x] Causal graphs and d-separation
  - [x] Sensitivity analysis methods

### Milestone 0.5: Structural Causal Models & Counterfactuals 🚧

- [x] **SCM Framework**
  - [x] Base SCM class with structural equations
  - [x] Intervention utilities (do-operator implementation)
  - [x] Counterfactual computation (abduction-action-prediction)
  - [x] Linear SCM for efficient counterfactuals
- [x] **Documentation**
  - [x] Comprehensive SCM tutorial covering three levels of causation
  - [x] Association vs intervention vs counterfactual reasoning
  - [x] Connection to potential outcomes and do-calculus
- [ ] **Examples & Notebooks**
  - [ ] Interactive SCM notebook with hands-on exercises
  - [ ] Biological SCM examples (gene regulation, drug response)
  - [ ] Counterfactual fairness and model explanation examples
- [ ] **Integration**
  - [ ] Connect SCMs to existing do-calculus tutorial
  - [ ] Show SCM implementation of IPW and propensity scores
  - [ ] Demonstrate mediation analysis with SCMs

### Milestone A: Causal Discovery on Gene Expression

- [ ] Implement constraint-based methods (PC algorithm)
- [ ] Implement score-based methods (GES)
- [ ] Implement continuous optimization (NOTEARS)
- [ ] Evaluate on synthetic + real gene expression data
- [ ] Benchmark against CausalBench

### Milestone B: Treatment Effect Estimation

- [ ] Integrate DoWhy for causal effect estimation
- [x] Implement propensity score methods (IPW, stabilized weights)
- [ ] Implement doubly robust estimators (AIPW, TMLE)
- [ ] Apply to drug response prediction
- [ ] Heterogeneous treatment effects (CATE)

### Milestone C: Counterfactual Perturbation Prediction

- [ ] Implement CPA-style perturbation autoencoder
- [ ] GEARS-style geometric deep learning for multigene perturbations
- [ ] Out-of-distribution prediction for unseen combinations
- [ ] Dose-response curve estimation

### Milestone D: Causal Representation Learning

- [ ] Identifiable VAE implementations
- [ ] Disentangled representations for biological factors
- [ ] Connection to generative models (link to genai-lab)
- [ ] Causal structure in latent space

## Key Concepts

### Causal Discovery vs Causal Inference

- **Causal Discovery**: Learning the causal graph structure from data
- **Causal Inference**: Estimating causal effects given a (known or assumed) causal graph

### Treatment Effects

- **ATE** (Average Treatment Effect): Population-level effect
- **ITE** (Individual Treatment Effect): Person-specific effect
- **CATE** (Conditional ATE): Effect for subgroups
- **ATT/ATC**: Effect on treated/control

### Counterfactual Reasoning

- "What would have happened if...?"
- Essential for drug repurposing, combination therapy
- Requires structural causal models (SCMs)

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| [DoWhy](https://github.com/py-why/dowhy) | End-to-end causal inference |
| [EconML](https://github.com/py-why/EconML) | Heterogeneous treatment effects |
| [CausalML](https://github.com/uber/causalml) | Uplift modeling |
| [gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) | Causal discovery |
| [NOTEARS](https://github.com/xunzheng/notears) | Continuous optimization for DAGs |
| [CausalNex](https://github.com/quantumblacklabs/causalnex) | Bayesian networks |

## References

### Academic

- [Elements of Causal Inference](https://mitpress.mit.edu/9780262037310/) — Peters, Janzing, Schölkopf
- [Causal Inference: What If](https://miguelhernan.org/whatifbook) — Hernán & Robins
- [GEARS](https://www.nature.com/articles/s41587-023-01905-6) — Multigene perturbation prediction
- [CPA](https://link.springer.com/article/10.15252/msb.202211517) — Compositional Perturbation Autoencoder
- [CausalBench](https://www.gsk.ai/causalbench-challenge/) — Gene network inference benchmark

### Industry

- [biotx.ai](https://www.biotx.ai/) — Causal modeling at scale
- [insitro](https://www.insitro.com/) — AI therapeutics on causal biology
- [Relation Therapeutics](https://www.relationrx.com/) — Lab-in-the-Loop causal discovery

## Related Projects

### [genai-lab](../genai-lab/) — Generative AI for Computational Biology

**Complementary Focus:** While `causal-bio-lab` focuses on **uncovering causal structures** and **estimating causal effects**, `genai-lab` focuses on **modeling data-generating processes** through generative models (VAE, diffusion, transformers).

**Synergy:**

- **Generative AI** learns rich representations of biological data and can simulate realistic perturbation responses
- **Causal ML** provides the framework to ensure these models capture true causal mechanisms, not just correlations (via causal graphs, structural equations, and causal discovery)
- **Together:** Causal generative models enable counterfactual reasoning, treatment effect prediction, and mechanistic understanding

**Key Integration Points:**

1. **Causal graphs from discovery algorithms** can constrain generative model architectures and latent space structure
2. **Causal inference methods** (do-calculus, structural equations, propensity scores) validate counterfactual predictions from generative models
3. **Causal representation learning** (Milestone D) bridges both projects—learning disentangled latent spaces that respect causal structure
4. **Perturbation prediction** benefits from both: generative models for realistic simulation + causal effect estimation for unbiased predictions

**Example Workflow:**

```text
1. Use genai-lab to train a VAE on gene expression data
2. Use causal-bio-lab to discover causal relationships between genes
3. Integrate causal structure into the VAE latent space (causal VAE)
4. Generate counterfactual perturbation responses with causal guarantees
```

See `genai-lab` Stage 5 (Counterfactual & Causal) for planned integration work.

## License

MIT
