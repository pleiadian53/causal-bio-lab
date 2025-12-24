# Industry Landscape: Causal AI/ML in Drug Discovery & Computational Biology

This document surveys companies, platforms, and research groups pioneering causal inference, causal discovery, and causal machine learning for drug discovery, target identification, and treatment effect estimation.

> **Last Updated:** December 2024  
> **Purpose:** Track industry developments, identify research directions, and gather ideas for this project.

---

## Table of Contents

1. [Causal Inference Platforms](#causal-inference-platforms)
2. [Perturbation Biology & Single-Cell Causal Methods](#perturbation-biology--single-cell-causal-methods)
3. [Target Discovery with Causal AI](#target-discovery-with-causal-ai)
4. [Federated Learning & Clinical Causal Inference](#federated-learning--clinical-causal-inference)
5. [Knowledge Graphs & Causal Reasoning](#knowledge-graphs--causal-reasoning)
6. [Causal Representation Learning](#causal-representation-learning)
7. [Open-Source Tools & Frameworks](#open-source-tools--frameworks)
8. [Academic Research Groups](#academic-research-groups)
9. [Key Observations & Research Directions](#key-observations--research-directions)

---

## Causal Inference Platforms

Companies building platforms specifically for causal inference in drug discovery.

### biotx.ai

| | |
|---|---|
| **Website** | [biotx.ai](https://www.biotx.ai/) |
| **Focus** | Causal modeling at scale for efficacy prediction |
| **Key Technology** | Proprietary causal genome dataset |
| **Location** | Potsdam, Germany |

**What They Do:**

- Map every locus on the genome to downstream effects on biomarker levels and disease risk
- Predict clinical trial success using causal models based on human genetic data
- Screen biomarkers, predict efficacy, identify top indications for drug assets
- Out-licensed six drug assets based on causal predictions

**Technical Approach:**

- Causal modeling rests on same theoretical framework as Randomized Controlled Trials (RCTs)
- Large-scale GWAS data integration with causal inference
- Enabled Series A/B funding for biopharma partners where preclinical data failed

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Direct application of causal inference to drug discovery
- Demonstrates commercial value of causal modeling
- Genetic instruments for causal effect estimation (Mendelian randomization style)

---

### insitro

| | |
|---|---|
| **Website** | [insitro.com](https://insitro.com/) |
| **Focus** | AI therapeutics built on causal biology |
| **Key Technology** | POSH (Pooled Optical Screening in Human cells) |
| **Founder** | Daphne Koller (Stanford, Coursera) |
| **Funding** | $700M+ raised |

**What They Do:**

- Self-supervised models trained on unbiased cellular morphology
- Reconstruct gene function and causal relationships without being told what to look for
- "Biology doesn't organize itself according to the features we've learned to measure"

**Technical Approach:**

- POSH platform validated in Nature Communications (Dec 2024)
- Interrogate the genome at scale while preserving phenotypic complexity
- Causal biology as foundation for AI therapeutics

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- State-of-the-art in causal biology for drug discovery
- Self-supervised learning for causal structure discovery
- Daphne Koller's vision of ML + causal inference

**Resources:**

- [Nature Communications Publication (2024)](https://www.nature.com/articles/s41467-024-55546-1)
- [McKinsey Interview with Daphne Koller](https://www.mckinsey.com/industries/life-sciences/our-insights/it-will-be-a-paradigm-shift-daphne-koller-on-machine-learning-in-drug-discovery)

---

### Cellarity

| | |
|---|---|
| **Website** | [cellarity.com](https://cellarity.com/) |
| **Focus** | Cell behavior-centric drug discovery |
| **Key Technology** | Cellarium™ digital twin |
| **Parent** | Flagship Pioneering |

**What They Do:**

- Shift focus from single molecular target to underlying cellular dysfunction
- Cellarium maps biological connections to AI-generated cell behavior screening
- Cellarity Maps™ — digital representations of disease biology

**Technical Approach:**

- High-dimensional transcriptomics at single-cell resolution
- Generalizable AI models linking chemistry directly to disease biology
- Revealed new biological pathways through multi-omics AI platform

**Relevance to This Project:** ⭐⭐⭐⭐

- Cell-centric view aligns with causal thinking about disease
- Multi-omics integration for causal pathway discovery

---

## Perturbation Biology & Single-Cell Causal Methods

Methods and companies focused on learning causal relationships from perturbation experiments.

### GEARS (Stanford)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/snap-stanford/GEARS) |
| **Focus** | Multigene perturbation prediction |
| **Key Technology** | Geometric deep learning |
| **Published** | Nature Biotechnology (2023) |

**What They Do:**

- Predict transcriptional outcomes of novel multigene perturbations
- Address combinatorial explosion in perturbation space
- Geometric deep learning on gene-gene interaction graphs

**Technical Approach:**

- Graph neural networks for perturbation response
- Out-of-distribution generalization to unseen combinations
- Trained on Perturb-seq data

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- State-of-the-art in perturbation prediction
- Directly relevant to Milestone C
- Open source implementation available

**Resources:**

- [Nature Biotechnology Paper](https://www.nature.com/articles/s41587-023-01905-6)
- [GitHub Repository](https://github.com/snap-stanford/GEARS)

---

### CPA (Meta AI / Theislab)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/theislab/cpa) |
| **Focus** | Compositional Perturbation Autoencoder |
| **Key Technology** | Deep generative framework |
| **Published** | Molecular Systems Biology (2023) |

**What They Do:**

- Learn effects of perturbations at single-cell level
- OOD predictions of unseen drug combinations
- Interpretable embeddings and dose-response curves
- Uncertainty estimates for predictions

**Technical Approach:**

- Variational autoencoder with compositional perturbation modeling
- Disentangles cell state from perturbation effects
- Additive model for combination effects

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Core method for Milestone C
- Connection to generative models (link to genai-lab)
- Well-documented open source

**Resources:**

- [MSB Paper](https://www.embopress.org/doi/full/10.15252/msb.202211517)
- [GitHub Repository](https://github.com/theislab/cpa)

---

### CausalBench (GSK)

| | |
|---|---|
| **Website** | [gsk.ai/causalbench-challenge](https://www.gsk.ai/causalbench-challenge/) |
| **Focus** | Benchmark for gene network inference |
| **Key Technology** | Large-scale evaluation framework |
| **Data** | Replogle et al. CRISPR perturbation datasets |

**What They Do:**

- Comprehensive benchmark for evaluating network inference methods
- Single-cell perturbation gene expression data
- Standardized evaluation metrics

**Technical Approach:**

- Two public CRISPR perturbation datasets
- Evaluation of causal discovery algorithms
- Leaderboard for method comparison

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Essential benchmark for Milestone A
- Real-world perturbation data
- Industry-backed evaluation standard

**Resources:**

- [arXiv Paper](https://arxiv.org/abs/2308.15395)
- [GitHub Starter](https://github.com/causalbench/causalbench-starter)

---

### Perturb-seq

| | |
|---|---|
| **Focus** | CRISPR screens + single-cell RNA-seq |
| **Key Technology** | Pooled genetic perturbation with transcriptomic readout |
| **Published** | Cell (2016) |

**What They Do:**

- Combine CRISPR perturbations with single-cell sequencing
- Scalable causal gene function discovery
- Direct measurement of perturbation → transcriptome effects

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Foundation data type for causal discovery in biology
- Enables interventional (not just observational) causal inference

---

## Target Discovery with Causal AI

Companies using causal approaches for drug target identification.

### Ochre Bio

| | |
|---|---|
| **Website** | [ochre-bio.com](https://www.ochre-bio.com/) |
| **Focus** | Liver disease therapeutics |
| **Key Technology** | Human Discovery Platform |
| **Partnerships** | GSK ($1B+ potential), Boehringer Ingelheim |

**What They Do:**

- $20M+ invested in human liver datasets
- Integrate gene perturbation atlases with patient disease atlases
- Make causal predictions about new drug targets

**Technical Approach:**

- Functional genomics + AI for causal target discovery
- Human validation models (perfused livers, tissue slices)
- In-house RNA therapeutics platform

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Original inspiration for this project
- Demonstrates causal inference for target discovery
- Strong industry validation (GSK, Boehringer partnerships)

---

### Relation Therapeutics

| | |
|---|---|
| **Website** | [relationrx.com](https://www.relationrx.com/) |
| **Focus** | Lab-in-the-Loop causal biology |
| **Key Technology** | Single-cell multi-omics + ML |
| **Funding** | $60M seed (DCVC, NVIDIA NVentures) |
| **Partnerships** | GSK ($15M equity), Novartis |

**What They Do:**

- Understand human biology "from cause to cure"
- Single-cell multi-omics directly from patient tissue
- Functional assays + machine learning for causal target discovery

**Technical Approach:**

- Lab-in-the-Loop: seamless integration of wet lab and ML
- Collaboration with Yoshua Bengio and Mila (Gates Foundation grant)
- NVIDIA Cambridge-1 supercomputer access

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- State-of-the-art integration of causal ML and biology
- Yoshua Bengio involvement signals deep ML expertise
- First indication: osteoporosis

**Resources:**

- [Novartis Collaboration Announcement](https://pharmaphorum.com/news/lab-loop-specialist-relation-attracts-novartis-alliance)

---

### Recursion Pharmaceuticals

| | |
|---|---|
| **Website** | [recursion.com](https://www.recursion.com/) |
| **Focus** | Phenomics + causal AI for drug discovery |
| **Key Technology** | Recursion OS, Maps of Biology |
| **Data** | 65 petabytes (phenomics, transcriptomics, proteomics, patient data) |

**What They Do:**

- Integrated patient data at every stage of drug discovery
- Causal AI models to initiate new drug programs
- Biomarker strategies from genetic driver mutations

**Technical Approach:**

- Automated wet lab with robotics and computer vision
- Self-supervised deep learning on cellular images
- Maps of Biology for perturbation embedding

**Relevance to This Project:** ⭐⭐⭐⭐

- Massive scale phenomics data
- Causal AI integrated with patient data
- Demonstrates industrial-scale causal biology

---

## Federated Learning & Clinical Causal Inference

Companies applying causal inference to clinical data with privacy preservation.

### Owkin

| | |
|---|---|
| **Website** | [owkin.com](https://www.owkin.com/) |
| **Focus** | Federated causal inference for clinical trials |
| **Key Technology** | FedECA (Federated External Control Arms) |
| **Published** | Nature Communications |

**What They Do:**

- Federated learning for causal inference with time-to-event data
- External control arms for clinical trials
- Uncover hidden causal relationships from biological data

**Technical Approach:**

- FedECA extends WebDISCO with propensity scoring and causal inference
- Federated learning behind hospital firewalls
- Applied to metastatic pancreatic cancer treatment comparison

**Relevance to This Project:** ⭐⭐⭐⭐

- Causal inference in clinical setting
- Privacy-preserving methods
- Treatment effect estimation from real-world data

**Resources:**

- [FedECA Nature Communications Paper](https://www.nature.com/articles/s41467-024-45847-2)

---

### Tempus

| | |
|---|---|
| **Website** | [tempus.com](https://www.tempus.com/) |
| **Focus** | Precision medicine |
| **Key Technology** | AI-driven clinical insights |

**What They Do:**

- Multi-modal patient data integration
- AI for treatment selection and clinical decision support
- Real-world evidence generation

**Relevance to This Project:** ⭐⭐⭐

- Clinical application of causal reasoning
- Treatment effect estimation in oncology

---

## Knowledge Graphs & Causal Reasoning

Companies using knowledge graphs for causal reasoning in drug discovery.

### Causaly

| | |
|---|---|
| **Website** | [causaly.com](https://www.causaly.com/) |
| **Focus** | Biomedical knowledge exploration |
| **Key Technology** | Bio Graph, Causaly Copilot |

**What They Do:**

- Knowledge graph with 100K+ pipeline drugs, 60K research programs
- Causal reasoning over biomedical literature
- Visual knowledge exploration for R&D teams

**Technical Approach:**

- AI-driven causal relationship extraction from literature
- Pipeline Graph for competitive intelligence
- Integration with drug discovery workflows

**Relevance to This Project:** ⭐⭐⭐

- Knowledge graph approach to causal reasoning
- Literature-based causal discovery

---

### BenevolentAI

| | |
|---|---|
| **Website** | [benevolent.com](https://www.benevolent.com/) |
| **Focus** | Knowledge graph drug discovery |
| **Key Technology** | Benevolent Platform |

**What They Do:**

- Knowledge graphs for drug target identification
- Causal reasoning over drug-perturbed expression
- AI-driven hypothesis generation

**Relevance to This Project:** ⭐⭐⭐

- Knowledge graph + causal reasoning
- Drug repurposing applications

---

## Causal Representation Learning

Companies and research groups working on learning causal structure in latent spaces.

### Valence Labs

| | |
|---|---|
| **Website** | [valencelabs.com](https://www.valencelabs.com/) |
| **Focus** | Virtual cells & causal representation learning |
| **Key Technology** | GFlowNets, causal representation learning |

**What They Do:**

- Research in causal representation learning for drug discovery
- Virtual cell models for patient response simulation
- Generative flow networks for molecular design

**Technical Approach:**

- Causal representation learning publications at top conferences
- Connection between generative models and causal inference
- Quantum mechanical simulations

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Directly relevant to Milestone D
- Bridge between generative AI and causal ML
- Strong research publications

**Resources:**

- [Valence Labs Publications](https://www.valencelabs.com/research/)
- [Virtual Cells Paper](https://www.valencelabs.com/publications/virtual-cells-predict-explain-discover/)

---

### causaLens

| | |
|---|---|
| **Website** | [causalens.com](https://causalens.com/) |
| **Focus** | Enterprise causal AI |
| **Key Technology** | decisionOS, CausalNet |
| **Partnerships** | Google Cloud |

**What They Do:**

- Enterprise platform for causal AI
- Structural causal models for prediction and intervention
- Causal decision trees and counterfactual estimation

**Technical Approach:**

- Automated causal discovery
- Integration with LLMs for causal reasoning
- Digital workers powered by causal AI

**Relevance to This Project:** ⭐⭐⭐

- Enterprise-grade causal AI platform
- Methodology applicable to biology

---

## Open-Source Tools & Frameworks

Key libraries for implementing causal methods.

### DoWhy (Microsoft/PyWhy)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/py-why/dowhy) |
| **Focus** | End-to-end causal inference |
| **Maintainer** | PyWhy community (Microsoft origin) |

**What It Does:**

- Unified language for causal inference
- Combines causal graphical models and potential outcomes
- Four-step workflow: Model → Identify → Estimate → Refute

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Core library for Milestone B
- Well-documented, actively maintained
- Industry adoption (Microsoft, Amazon, Uber)

---

### EconML (Microsoft)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/py-why/EconML) |
| **Focus** | Heterogeneous treatment effects |
| **Methods** | Double ML, Causal Forests, Meta-learners |

**What It Does:**

- CATE estimation with ML
- Double/debiased machine learning
- Causal forests and orthogonal random forests

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Essential for heterogeneous treatment effects
- Integrates with DoWhy

---

### CausalML (Uber)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/uber/causalml) |
| **Focus** | Uplift modeling and treatment effects |
| **Methods** | Meta-learners, tree-based methods |

**What It Does:**

- Uplift modeling for marketing/treatment
- S-learner, T-learner, X-learner
- Causal tree and forest methods

**Relevance to This Project:** ⭐⭐⭐⭐

- Complementary to EconML
- Good for uplift/treatment selection

---

### causal-learn

| | |
|---|---|
| **Website** | [GitHub](https://github.com/py-why/causal-learn) |
| **Focus** | Causal discovery algorithms |
| **Methods** | PC, FCI, GES, LiNGAM, NOTEARS |

**What It Does:**

- Constraint-based methods (PC, FCI)
- Score-based methods (GES)
- Functional causal models (LiNGAM)
- Continuous optimization (NOTEARS)

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Core library for Milestone A
- Comprehensive algorithm collection

---

### gCastle (Huawei)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) |
| **Focus** | Causal structure learning |
| **Methods** | 20+ algorithms |

**What It Does:**

- Gradient-based causal discovery
- Benchmark datasets and evaluation
- GPU acceleration

**Relevance to This Project:** ⭐⭐⭐⭐

- Alternative to causal-learn
- Good for benchmarking

---

## Academic Research Groups

Key academic groups advancing causal ML for biology.

| Group | Institution | Focus |
|-------|-------------|-------|
| **Yoshua Bengio** | Mila, Montreal | Causal representation learning, GFlowNets |
| **Bernhard Schölkopf** | MPI Tübingen | Causal inference, kernel methods |
| **Fabian Theis** | Helmholtz Munich | Single-cell ML, CPA |
| **Jure Leskovec** | Stanford | GEARS, graph ML for biology |
| **Caroline Uhler** | MIT | Causal inference, genomics |
| **David Sontag** | MIT | Clinical ML, causal inference |
| **Uri Shalit** | Technion | Treatment effect estimation |

---

## Key Observations & Research Directions

### Trends

1. **Perturbation data is key**: Perturb-seq and CRISPR screens provide interventional data for causal discovery
2. **Single-cell resolution**: Most advances happening at single-cell level
3. **Integration with generative models**: CPA, GEARS show connection between generation and causal reasoning
4. **Industry validation**: Major pharma (GSK, Novartis, Boehringer) investing in causal AI
5. **Federated approaches**: Privacy-preserving causal inference for clinical data

### Research Opportunities

1. **Causal discovery on multi-omics**: Beyond transcriptomics to proteomics, metabolomics
2. **Temporal causal models**: Time-series gene expression for dynamic causal inference
3. **Causal foundation models**: Pre-trained models with causal structure
4. **Identifiable representations**: Provably causal latent spaces
5. **Benchmarking**: More comprehensive evaluation on real biological data

### Connection to genai-lab

- **Counterfactual generation**: Both projects address "what if" questions
- **Latent space structure**: Causal structure in VAE latent spaces
- **Perturbation prediction**: CPA bridges generative and causal approaches
- **Treatment response**: Predicting response to interventions

---

## References

### Foundational Texts

- Peters, Janzing, Schölkopf — [Elements of Causal Inference](https://mitpress.mit.edu/9780262037310/)
- Hernán & Robins — [Causal Inference: What If](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- Pearl — [Causality](http://bayes.cs.ucla.edu/BOOK-2K/)

### Key Papers

- [GEARS (Nature Biotechnology 2023)](https://www.nature.com/articles/s41587-023-01905-6)
- [CPA (MSB 2023)](https://www.embopress.org/doi/full/10.15252/msb.202211517)
- [CausalBench](https://arxiv.org/abs/2308.15395)
- [Causal ML for single-cell (Nature Genetics 2025)](https://www.nature.com/articles/s41588-025-02124-2)

### Industry Resources

- [biotx.ai](https://www.biotx.ai/)
- [insitro](https://insitro.com/)
- [Relation Therapeutics](https://www.relationrx.com/)
- [Owkin FedECA](https://www.owkin.com/publications/fedeca-a-federated-external-control-arm-method-for-causal-inference-with-time-to-event-data-in-distributed-settings)
