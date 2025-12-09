# Pareto Module

The `pareto` module implements Pareto frontier analysis for multi-objective optimization.

## ParetoFrontier

```rust
use single_shot_eval::{ParetoFrontier, ModelResult};

let results = vec![
    ModelResult { name: "fast".into(), accuracy: 0.85, latency_ms: 50.0 },
    ModelResult { name: "accurate".into(), accuracy: 0.95, latency_ms: 200.0 },
    ModelResult { name: "balanced".into(), accuracy: 0.90, latency_ms: 100.0 },
];

let frontier = ParetoFrontier::new(&results);
```

## Methods

### optimal_models()

Returns models on the Pareto frontier (not dominated by any other model).

```rust
for model in frontier.optimal_models() {
    println!("{} is Pareto-optimal", model.name);
}
```

### is_dominated()

Checks if a specific model is dominated by another.

```rust
let dominated = frontier.is_dominated(&results[2]);
```

### dominates()

Checks if model A dominates model B (better on all metrics).

```rust
let a_dominates_b = frontier.dominates(&model_a, &model_b);
```

## ModelResult

```rust
pub struct ModelResult {
    pub name: String,
    pub accuracy: f64,      // Higher is better
    pub latency_ms: f64,    // Lower is better
}
```
