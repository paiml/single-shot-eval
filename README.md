# single-shot-eval

[![CI](https://github.com/paiml/single-shot-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/single-shot-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

**SLM Pareto Frontier Evaluation Framework** - OFFLINE-FIRST evaluation using Batuta sovereign stack.

> Prove that small models can beat frontier models on domain-specific tasks at 1/100th the cost.

## Core Thesis

A 100M-parameter SLM, when properly distilled using Depyler's single-shot compilation insights, can match or exceed frontier model performance on domain-specific tasks while reducing inference cost by 100-1000x.

**Key insight: 1% worse accuracy but 100x smaller and FREE.**

## Research Basis

Based on Princeton ["AI Agents That Matter" (2024)](https://arxiv.org/abs/2407.01502) methodology:

- 5 runs minimum with 95% CI using Student's t-distribution
- Convex Pareto frontiers (probability-weighted combinations)
- Dollar costs, not proxy metrics
- Ground truth via compilation + test execution (SWE-bench style)

## Architecture

```
Python Source (reprorusted-python-cli)
       |
Model Inference (SLM .apr | SaaS baselines)
       |
Rust Output
       |
Ground Truth Verification (cargo build && cargo test)
       |
Metrics (pass@1, cost, latency)
       |
Pareto Frontier Analysis
       |
Report (Princeton-compliant: 5 runs, 95% CI)
```

## Installation

```bash
# Build from source
cargo build --release

# Install globally
cargo install --path .
```

## Usage

### CLI

#### Run Evaluation Suite

```bash
# Evaluate .apr models against task configurations
single-shot-eval evaluate \
  --tasks "tasks/*.yaml" \
  --models "models/*.apr" \
  --corpus ./python-corpus \
  --runs 5 \
  --output results.json

# With baseline comparisons (requires claude/gemini CLI tools)
single-shot-eval evaluate \
  --tasks tasks/transpile.yaml \
  --models models/slm-100m.apr \
  --baselines claude,gemini
```

#### Verify Rust Code

```bash
# Check if generated Rust compiles
single-shot-eval verify --source output.rs

# With test verification
single-shot-eval verify --source output.rs --tests tests.rs
```

#### Corpus Statistics

```bash
# Show Python corpus statistics
single-shot-eval corpus-stats --path ./reprorusted-python-cli
```

#### Generate Reports

```bash
# Generate Pareto frontier report
single-shot-eval report --input ./results --output report.md
```

### Library

```rust
use single_shot_eval::{
    TaskLoader, TaskRunner, CompilerVerifier,
    analyze_pareto, ReportBuilder, EvalResult,
};
use std::time::Duration;

// Load task configurations
let loader = TaskLoader::load_glob("tasks/*.yaml")?;

// Configure runner
let runner = TaskRunner::new();

// Run evaluation
for task in loader.iter() {
    let result = runner.run_evaluation(task, "my-model")?;
    println!("Accuracy: {:.2}%", result.accuracy * 100.0);
}

// Pareto analysis
let results: Vec<EvalResult> = vec![/* ... */];
let analysis = analyze_pareto(&results);
println!("Frontier models: {:?}", analysis.frontier_models);

// Compiler verification
let verifier = CompilerVerifier::new();
let result = verifier.verify(rust_code, Some(test_code))?;
assert!(result.passes());

// Generate report
let mut builder = ReportBuilder::new("my-eval");
builder.add_result(result, accuracy_samples);
let report = builder.build();
println!("{}", report.to_markdown());
```

## Task Configuration Format

Task configurations are YAML files in `tasks/*.yaml`:

```yaml
task:
  id: python-to-rust-transpile
  domain: code-transpile
  description: Transpile Python functions to idiomatic Rust

evaluation:
  samples: 100
  timeout_ms: 30000
  metrics:
    - accuracy
    - cost
    - latency

prompts:
  system: |
    You are an expert code transpiler. Convert Python to idiomatic Rust.
  user_template: |
    Convert this Python code to Rust:
    ```python
    {python_code}
    ```

ground_truth:
  type: compiler
  run_tests: true
```

## Metrics & Statistical Methodology

### Princeton Compliance

- **Minimum 5 runs**: All evaluations run 5+ times for statistical validity
- **95% CI**: Confidence intervals using Student's t-distribution
- **Bootstrap CI**: 10,000 bootstrap samples for non-parametric estimation
- **Significance testing**: Welch's t-test and paired t-tests with Bonferroni correction

### Pareto Analysis

The framework computes multi-objective Pareto frontiers across:
- **Accuracy** (pass@1 on ground truth)
- **Cost** ($/1M tokens)
- **Latency** (p50, p99)

Trade-off analysis identifies models that are:
- On the Pareto frontier (not dominated)
- Dominated (better alternatives exist)
- Value leaders (best accuracy/cost ratio)

## Offline-First Architecture

This framework uses the **Batuta sovereign stack**:

- No HTTP dependencies
- `.apr` format (Aprender native) for models
- Shell exec to CLI tools for SaaS baselines (`claude`, `gemini`)
- Fully air-gapped operation possible

## Project Structure

```
src/
  lib.rs        # Library exports
  main.rs       # CLI entry point
  baselines.rs  # SaaS baseline wrappers
  compiler.rs   # Rust verification
  config.rs     # Task YAML parsing
  corpus.rs     # Python corpus handling
  metrics.rs    # Statistical analysis
  pareto.rs     # Pareto frontier computation
  report.rs     # Report generation
  runner.rs     # Evaluation orchestration
tasks/
  *.yaml        # Task configurations
tests/
  integration.rs # End-to-end tests
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch from `main`
3. Run `make lint` and `make test` before submitting
4. Ensure all tests pass and clippy reports no warnings
5. Submit a pull request

## License

MIT
