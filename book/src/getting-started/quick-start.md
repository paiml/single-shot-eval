# Quick Start

## Run the Demo

The fastest way to see single-shot-eval in action:

```bash
cargo run --example demo --release
```

This demonstrates:
- Pareto frontier analysis
- Bootstrap confidence intervals
- Statistical significance testing
- Result visualization

## Basic Evaluation

1. **Create a task definition** (`tasks/example.yaml`):

```yaml
task_id: py2rs-basic
description: Python to Rust translation
corpus_path: corpus/examples.jsonl
model_path: models/slm.apr
baselines:
  - name: claude-sonnet
    command: claude -m sonnet
  - name: gemini-pro
    command: gemini -m pro
```

2. **Run evaluation**:

```bash
single-shot-eval run --task tasks/example.yaml
```

3. **View results**:

The tool outputs:
- Individual model scores
- Pareto-optimal models (quality vs latency tradeoff)
- 95% confidence intervals via bootstrap
- Statistical significance comparisons

## Benchmark Classification

Classify your corpus by Py2Rs difficulty levels (1-10):

```bash
single-shot-eval benchmark --corpus corpus/
```

Output shows:
- Level distribution across 10 Py2Rs categories
- Difficulty breakdown (Trivial → Expert)
- Coverage analysis

## Using as a Library

```rust
use single_shot_eval::{ParetoFrontier, ModelResult, infer_difficulty};

// Analyze model results
let results = vec![
    ModelResult { name: "model-a".into(), accuracy: 0.85, latency_ms: 100.0 },
    ModelResult { name: "model-b".into(), accuracy: 0.92, latency_ms: 250.0 },
];

let frontier = ParetoFrontier::new(&results);
for optimal in frontier.optimal_models() {
    println!("{}: {:.1}% @ {}ms", optimal.name, optimal.accuracy * 100.0, optimal.latency_ms);
}
```
