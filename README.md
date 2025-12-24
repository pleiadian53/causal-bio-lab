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

### Milestone A: Causal Discovery on Gene Expression

- [ ] Implement constraint-based methods (PC algorithm)
- [ ] Implement score-based methods (GES)
- [ ] Implement continuous optimization (NOTEARS)
- [ ] Evaluate on synthetic + real gene expression data
- [ ] Benchmark against CausalBench

### Milestone B: Treatment Effect Estimation

- [ ] Integrate DoWhy for causal effect estimation
- [ ] Implement propensity score methods
- [ ] Implement doubly robust estimators
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
- [Causal Inference: What If](https://www.hsph.harvard.edu/research/causalab/) — Hernán & Robins (CausaLab)
- [GEARS](https://www.nature.com/articles/s41587-023-01905-6) — Multigene perturbation prediction
- [CPA](https://www.embopress.org/doi/full/10.15252/msb.202211517) — Compositional Perturbation Autoencoder
- [CausalBench](https://www.gsk.ai/causalbench-challenge/) — Gene network inference benchmark

### Industry

- [biotx.ai](https://www.biotx.ai/) — Causal modeling at scale
- [insitro](https://insitro.com/) — AI therapeutics on causal biology
- [Relation Therapeutics](https://www.relationrx.com/) — Lab-in-the-Loop causal discovery

## Related Projects

- [genai-lab](../genai-lab/) — Generative AI for computational biology (VAE, diffusion, counterfactual simulation)

## License

MIT
