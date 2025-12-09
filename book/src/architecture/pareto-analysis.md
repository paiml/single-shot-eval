# Pareto Analysis

Pareto frontier analysis identifies models that are not dominated by any other model across multiple metrics.

## What is Pareto Optimality?

A model is **Pareto optimal** if no other model is better on all metrics. For example, with accuracy and latency:

- Model A: 90% accuracy, 100ms latency
- Model B: 85% accuracy, 50ms latency
- Model C: 88% accuracy, 120ms latency

Models A and B are Pareto optimal (neither dominates the other). Model C is dominated by A (worse accuracy AND latency).

## API

```rust
use single_shot_eval::{ParetoFrontier, ModelResult};

let results = vec![
    ModelResult { name: "fast".into(), accuracy: 0.85, latency_ms: 50.0 },
    ModelResult { name: "accurate".into(), accuracy: 0.95, latency_ms: 200.0 },
    ModelResult { name: "balanced".into(), accuracy: 0.90, latency_ms: 100.0 },
];

let frontier = ParetoFrontier::new(&results);

// Get Pareto-optimal models
for model in frontier.optimal_models() {
    println!("{}: {:.1}% @ {}ms", model.name, model.accuracy * 100.0, model.latency_ms);
}

// Check if a specific model is dominated
let dominated = frontier.is_dominated(&results[2]);
```

## Metrics

The framework supports multi-dimensional Pareto analysis:

| Metric | Direction | Description |
|--------|-----------|-------------|
| Accuracy | Higher is better | Pass rate on test cases |
| Latency | Lower is better | Inference time (ms) |
| Cost | Lower is better | API cost per request |
| Tokens | Lower is better | Token count |

## Visualization

The CLI outputs a Pareto frontier visualization:

```
Accuracy
  ^
  |    * accurate (95%)
  |
  |         * balanced (90%)
  |
  |              * fast (85%)
  +----------------------------> Latency
       50ms   100ms   200ms
```

Models on the frontier are marked with `*`.
