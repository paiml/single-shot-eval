# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-shot-eval is a Rust-based evaluation framework for Small Language Models (SLMs) using Pareto frontier analysis. The core thesis: task-specific SLMs (100M-500M parameters) can match frontier model performance while reducing inference cost by 100-1000x.

The project integrates with the Batuta sovereign stack:
- **Depyler**: Static analysis of model computation graphs
- **Aprender**: Native Rust ML with deterministic training (depends on Trueno)
- **Entrenar**: Distillation framework with teacher-student protocols
- **Trueno**: SIMD-accelerated inference

## Build Commands

```bash
# Build
cargo build --release

# Test (requires 95% coverage)
cargo test

# Lint (strict mode, zero warnings)
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt --all

# Full CI pipeline
make ci
```

## PMAT Integration (EXTREME TDD)

This project uses PMAT for quality enforcement with Toyota Way principles.

### Quality Gates

```bash
# Run PMAT quality gates
make pmat-quality-gate

# Calculate Rust project score
make pmat-rust-score

# Full score with clippy and mutation testing
make pmat-rust-score-full

# Generate deep context
make pmat-context

# Validate documentation accuracy
make pmat-validate-docs

# Run TDG analysis
make pmat-tdg

# Show metrics and trends
make pmat-metrics
```

### Pre-commit Hooks

TDG enforcement hooks are installed:
- `.git/hooks/pre-commit` - TDG quality checks
- `.git/hooks/post-commit` - Baseline auto-update

Configuration: `.pmat/tdg-rules.toml`

### Coverage Requirements

```bash
# Generate coverage report
make coverage

# Generate HTML coverage report
make coverage-html

# Verify coverage meets 95% threshold
make coverage-check

# Run mutation testing (target: 80%)
make mutants
```

## Architecture

```
Evaluation Flow:
Task Definition (YAML) → Task Runner → [SLM Inference | Frontier APIs] → Metrics Engine → Pareto Analysis → Report

Core Components:
- src/config.rs: YAML task configuration parsing
- src/metrics.rs: Accuracy, F1, BLEU/ROUGE, latency, cost computation
- src/pareto.rs: Pareto frontier/dominance analysis
- src/runner.rs: Task execution engine
```

## Key Data Types

```rust
// Statistical evaluation configuration
pub struct EvalConfig {
    pub min_samples: usize,    // 1000 minimum per task
    pub bootstrap_n: usize,    // 10000 resamples for CI
    pub confidence: f64,       // 0.95
    pub seed: u64,             // Fixed for reproducibility
    pub alpha: f64,            // 0.05 significance level
}

// Pareto dominance: S1 dominates S2 if better/equal in all objectives
// and strictly better in at least one (accuracy ↑, cost ↓, latency ↓)
```

## Task Configuration Format

Tasks are defined in `tasks/*.yaml`:
```yaml
task:
  id: code-review
  domain: software
evaluation:
  metric: accuracy
  samples: 2000
  timeout_ms: 5000
prompts:
  system: "..."
  user_template: "..."
ground_truth:
  source: "datasets/code-review-gold.jsonl"
slm_optimization:
  attention_heads_required: 4
  context_length: 512
  quantization_viable: true
```

## Evaluation Commands

```bash
# Run full evaluation suite
make evaluate

# Quick smoke test (100 samples)
make evaluate-quick

# Generate Pareto frontier report
make report
```

## Quality Thresholds (per .pmat-metrics.toml)

| Metric | Threshold |
|--------|-----------|
| Test Coverage | ≥95% (EXTREME TDD) |
| Mutation Score | ≥80% |
| Cyclomatic Complexity | ≤15 per function |
| TDG Grade | A or better (≥90) |
| `unwrap()` calls | 0 (ZERO tolerance) |
| Clippy | Zero warnings |

## SLM Success Criteria

An SLM is production-viable when:
1. Accuracy within 5% of best frontier model on target task
2. At least 100x cheaper per inference
3. p99 latency < 100ms
4. >99% deterministic outputs
5. Non-dominated or within 2% of Pareto frontier

## Development Setup

```bash
# Install development dependencies
make setup

# Initialize PMAT integration (already done)
make pmat-init
```

## Five Whys Debugging (MANDATORY)

Per Toyota Way principles, use PMAT Five Whys for root cause analysis:

```bash
# Analyze an issue
pmat five-whys "Description of the issue"

# With custom depth
pmat why "Issue description" --depth 3

# Generate markdown report
pmat five-whys "Issue" --format markdown --output analysis.md
```

This is the ONLY acceptable debugging method per project policy.
