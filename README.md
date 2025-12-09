<div align="center">
  <img src="docs/hero.svg" alt="single-shot-eval" width="800">

  # single-shot-eval

  **SLM Pareto Frontier Evaluation Framework - OFFLINE-FIRST evaluation using Batuta sovereign stack.**

  [![CI](https://github.com/paiml/single-shot-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/single-shot-eval/actions/workflows/ci.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
</div>

---

Prove that small models can beat frontier models on domain-specific tasks at 1/100th the cost. Part of the [PAIML Stack](https://github.com/paiml).

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

## Requirements

### Hardware
- **CPU**: Any x86_64 or ARM64 processor
- **RAM**: Minimum 4GB (8GB recommended for large corpora)
- **Disk**: ~100MB for build artifacts

### Software
- **Rust**: 1.75+ (MSRV specified in Cargo.toml)
- **OS**: Linux (tested on Ubuntu 22.04), macOS, Windows
- **Optional**: `claude` CLI, `gemini` CLI for baseline comparisons

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

#### Benchmark Corpus

```bash
# Analyze Python corpus with Py2Rs 10-level classification
single-shot-eval benchmark --corpus ./experiments/python-corpus/

# With detailed per-example breakdown
single-shot-eval benchmark --corpus ./experiments/python-corpus/ --verbose
```

Output shows level distribution, difficulty breakdown, and visual coverage:
```
┌────────────────────────────────────────────────────────────────┐
│ Py2Rs 10-Level Benchmark Analysis                              │
├────────────────────────────────────────────────────────────────┤
│ Corpus:   50 examples                                          │
└────────────────────────────────────────────────────────────────┘

Level Distribution
──────────────────
L3  (Functions)      [████░░░░░░░░░░░░░░░░]  40% (20 examples)
L5  (ControlFlow)    [███░░░░░░░░░░░░░░░░░]  30% (15 examples)
...

Visual Summary (● = has examples, ○ = empty)
Levels 1-10: ○○●●●●○○○○
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

### Inference Parameters

Default inference parameters (configurable in task YAML):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.0 | Deterministic sampling for reproducibility |
| `max_tokens` | 4096 | Maximum output tokens |
| `timeout_ms` | 30000 | Per-example timeout |
| `runs` | 5 | Minimum runs for statistical validity |

### Known Limitations

- **Demo variance**: Bootstrap CI values may vary slightly between runs due to random sampling (expected behavior)
- **Baseline CLIs**: External `claude`/`gemini` CLI tools must be installed and authenticated separately
- **Corpus size**: Very large corpora (>10K examples) may require increased memory
- **GPU support**: Not required; all inference uses CPU-optimized Batuta stack

## Offline-First Architecture

This framework uses the **Batuta sovereign stack**:

- No HTTP dependencies
- `.apr` format (Aprender native) for models
- Shell exec to CLI tools for SaaS baselines (`claude`, `gemini`)
- Fully air-gapped operation possible

## Project Structure

```
src/
  lib.rs          # Library exports
  main.rs         # CLI entry point
  baselines.rs    # SaaS baseline wrappers
  bench_bridge.rs # aprender::bench integration (Py2Rs 10-level)
  compiler.rs     # Rust verification
  config.rs       # Task YAML parsing
  corpus.rs       # Python corpus handling
  inference.rs    # Model loading & inference
  metrics.rs      # Statistical analysis
  pareto.rs       # Pareto frontier computation
  report.rs       # Report generation
  runner.rs       # Evaluation orchestration
examples/
  demo.rs         # Pareto analysis demo
  benchmark.rs    # Py2Rs benchmark classification demo
tasks/
  *.yaml          # Task configurations
tests/
  integration.rs  # End-to-end tests
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
