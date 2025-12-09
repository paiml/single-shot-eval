# Introduction

**Single-Shot Eval** is an OFFLINE-FIRST evaluation framework for Small Language Models (SLMs) using the Batuta sovereign AI stack.

## What is Single-Shot Evaluation?

Single-shot evaluation measures model quality based on whether generated code compiles and passes tests on the first attempt - without iterative refinement. This provides a rigorous measure of model capability.

## Key Features

- **Pareto Frontier Analysis**: Identify optimal models balancing quality vs latency
- **Bootstrap Confidence Intervals**: Statistical rigor with 95% CI
- **Batuta Stack Integration**: Uses `aprender` for ML and `alimentar` for data loading
- **Py2Rs Benchmark**: 10-level Python-to-Rust translation benchmark
- **Offline-First**: No HTTP dependencies - runs anywhere

## Research Foundation

Based on Princeton research on code generation evaluation:

> *"Code generation evaluation must move beyond simple pass rates to multi-dimensional analysis including latency, cost, and robustness."*
>
> — [Xu et al., 2024 (arXiv:2407.01502)](https://arxiv.org/abs/2407.01502)

## Quick Example

```bash
# Run evaluation with Pareto analysis
single-shot-eval run --task tasks/py2rs.yaml --model models/slm.apr

# Classify corpus by Py2Rs difficulty levels
single-shot-eval benchmark --corpus corpus/
```

## Architecture Overview

```
single-shot-eval
├── config.rs      # YAML task parsing
├── corpus.rs      # Py2Rs example corpus
├── pareto.rs      # Pareto frontier analysis
├── runner.rs      # Evaluation orchestration
├── report.rs      # Result reporting
└── inference.rs   # Model loading (aprender)
```

## Next Steps

- [Installation](./getting-started/installation.md) - Set up single-shot-eval
- [Quick Start](./getting-started/quick-start.md) - Run your first evaluation
- [CLI Usage](./getting-started/cli-usage.md) - Full command reference
