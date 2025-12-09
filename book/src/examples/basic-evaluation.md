# Basic Evaluation

A simple end-to-end evaluation example.

## Setup

1. Create a task configuration:

```yaml
# tasks/basic.yaml
task_id: basic-py2rs
description: Basic Python to Rust evaluation
corpus_path: corpus/basic.jsonl
model_path: models/slm.apr
```

2. Prepare corpus (JSONL format):

```json
{"id": "ex1", "python": "def add(a, b): return a + b", "level": 1}
{"id": "ex2", "python": "for i in range(10): print(i)", "level": 2}
```

## Run Evaluation

```bash
single-shot-eval run --task tasks/basic.yaml
```

## Output

```
Evaluation Results
==================
Model: slm-local
Examples: 50
Accuracy: 84.0% [95% CI: 71.8%, 92.4%]
Latency: 42.3ms (mean)

No baselines configured.
```

## Programmatic Usage

```rust
use single_shot_eval::{TaskConfig, EvaluationRunner, Report};

fn main() -> anyhow::Result<()> {
    let config = TaskConfig::from_file("tasks/basic.yaml")?;
    let runner = EvaluationRunner::new(config)?;
    let results = runner.run()?;

    let report = Report::from_results(&results);
    println!("{}", report.to_table());

    Ok(())
}
```
