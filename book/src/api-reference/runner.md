# Runner Module

The `runner` module orchestrates evaluation execution.

## EvaluationRunner

```rust
use single_shot_eval::runner::{EvaluationRunner, EvaluationResult};
use single_shot_eval::config::TaskConfig;

let config = TaskConfig::from_file("tasks/example.yaml")?;
let runner = EvaluationRunner::new(config)?;

let results = runner.run()?;
```

## Run Options

```rust
let runner = EvaluationRunner::new(config)?
    .with_bootstrap_iterations(1000)  // CI iterations
    .with_timeout(Duration::from_secs(300))
    .with_parallel(true);
```

## EvaluationResult

```rust
pub struct EvaluationResult {
    pub model_name: String,
    pub accuracy: f64,
    pub latency_ms: f64,
    pub confidence_interval: (f64, f64),  // 95% CI
    pub examples_evaluated: usize,
}
```

## Baseline Execution

Baselines are executed via CLI commands:

```rust
// Internally runs: claude -m sonnet < input.txt
let baseline_result = runner.run_baseline("claude-sonnet")?;
```
