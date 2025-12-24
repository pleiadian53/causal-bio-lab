# Documentation

Project-level documentation for causal-bio-lab.

## Contents

### Overview Documents

| Document | Description |
|----------|-------------|
| [INDUSTRY_LANDSCAPE.md](./INDUSTRY_LANDSCAPE.md) | Survey of companies and platforms working on causal AI/ML for drug discovery |

### Topic Directories

| Directory | Focus |
|-----------|-------|
| [discovery/](./discovery/) | Causal discovery algorithms and their limitations |
| [estimation/](./estimation/) | Treatment effect estimation (ATE, CATE) |
| [decision/](./decision/) | Target ranking and prioritization |
| [failure_modes/](./failure_modes/) | Common pitfalls and when methods break |

### Key Documents

- **[discovery/limits_of_discovery.md](./discovery/limits_of_discovery.md)** — Why causal discovery alone doesn't find drug targets

## Document Types

This project maintains two types of documentation:

1. **Project-level docs** (this directory)
   - Industry research and landscape analysis
   - Architecture decisions and design rationale
   - Research directions and ideas

2. **Package-level docs** (within `src/causalbiolab/`)
   - API documentation
   - Module-specific guides
   - Implementation notes

## Contributing

When adding new documents:

- Place topic-specific docs in the appropriate subdirectory
- Use descriptive filenames in lower_snake_case
- Include a "Last Updated" date
- Add entry to this README
