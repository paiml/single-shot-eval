# Basic Evaluation

A simple end-to-end evaluation example using the demo.

## Run the Demo

The demo example shows Pareto frontier analysis with multiple models:

```bash
cargo run --example demo --release
```

## What the Demo Shows

### 1. Model Comparison

The demo compares 4 models on a sentiment task:

| Model | Accuracy | Cost | Latency |
|-------|----------|------|---------|
| slm-100m | 92% | $0.0001 | 15ms |
| gemini-flash | 93% | $0.075 | 400ms |
| gpt-4o-mini | 94% | $0.15 | 600ms |
| claude-haiku | 95% | $0.25 | 800ms |

### 2. Pareto Analysis

All 4 models are on the Pareto frontier (none is strictly dominated):

```
Pareto-optimal models:
  ◆ claude-haiku (acc: 95.0%, cost: $0.2500, lat: 800ms)
  ◆ gpt-4o-mini (acc: 94.0%, cost: $0.1500, lat: 600ms)
  ◆ gemini-flash (acc: 93.0%, cost: $0.0750, lat: 400ms)
  ◆ slm-100m (acc: 92.0%, cost: $0.0001, lat: 15ms)
```

### 3. Trade-off Analysis

The SLM shows exceptional value despite lower accuracy:

```
slm-100m trade-off:
  Accuracy gap: 3.0%
  Cost ratio: 2500x cheaper
  Latency ratio: 53x faster
  Value score: 129333.3x
```

**Key insight**: 3% accuracy loss buys you 2500x cost reduction and 53x speedup.

### 4. Statistical Analysis

Bootstrap confidence intervals and paired t-tests:

```
SLM accuracy: 92.11% [95% CI: 91.91% - 92.32%]
Paired t-test vs frontier:
  t-statistic: -16.942
  p-value: 0.0000
  Cohen's d: -1.694 (large)
  Significant: YES
```

### 5. Report Generation

Automated markdown report with tables:

```
Report Summary:
  Total Models: 4
  Frontier Models: 4
  Best Accuracy: 95.0%
  Best Cost: $0.0001/1K tokens
  Best Latency: 15ms
```

## Programmatic Usage

```rust
use single_shot_eval::{
    pareto::{analyze_pareto, compute_pareto_frontier, EvalResult},
    report::ReportBuilder,
    metrics::{bootstrap_ci, StatConfig},
};
use std::time::Duration;
use std::collections::HashMap;

fn main() {
    // Create evaluation results
    let results = vec![
        EvalResult {
            model_id: "slm-100m".to_string(),
            task_id: "sentiment".to_string(),
            accuracy: 0.92,
            cost: 0.0001,
            latency: Duration::from_millis(15),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "claude-haiku".to_string(),
            task_id: "sentiment".to_string(),
            accuracy: 0.95,
            cost: 0.25,
            latency: Duration::from_millis(800),
            metadata: HashMap::new(),
        },
    ];

    // Compute Pareto frontier
    let frontier = compute_pareto_frontier(&results);
    println!("Frontier models: {}", frontier.len());

    // Full analysis with trade-offs
    let analysis = analyze_pareto(&results);
    for tradeoff in &analysis.trade_offs {
        println!("{}: {:.0}x value", tradeoff.model_id, tradeoff.value_score);
    }

    // Bootstrap CI
    let accuracies: Vec<f64> = vec![0.91, 0.92, 0.93, 0.90, 0.94];
    let config = StatConfig::default();
    let (lower, upper) = bootstrap_ci(&accuracies, &config);
    println!("95% CI: [{:.1}%, {:.1}%]", lower * 100.0, upper * 100.0);

    // Generate report
    let mut builder = ReportBuilder::new("sentiment");
    for result in &results {
        let samples: Vec<f64> = vec![result.accuracy; 10];
        builder.add_result(result.clone(), samples);
    }
    let report = builder.build();
    println!("{}", report.to_markdown());
}
```

## Next Steps

- [Py2Rs Benchmark](./py2rs-benchmark.md) - Python-to-Rust difficulty classification
- [Custom Baselines](./custom-baselines.md) - Adding external model baselines
