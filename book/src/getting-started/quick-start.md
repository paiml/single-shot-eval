# Quick Start

## Run the Examples

The fastest way to see single-shot-eval in action:

### Demo Example (Pareto Analysis)

```bash
cargo run --example demo --release
```

**Output:**

```
=== Single-Shot Eval Demo ===

📊 Computing Pareto Frontier...

Pareto-optimal models:
  ◆ claude-haiku (acc: 95.0%, cost: $0.2500, lat: 800ms)
  ◆ gpt-4o-mini (acc: 94.0%, cost: $0.1500, lat: 600ms)
  ◆ gemini-flash (acc: 93.0%, cost: $0.0750, lat: 400ms)
  ◆ slm-100m (acc: 92.0%, cost: $0.0001, lat: 15ms)

📈 Trade-off Analysis...

Frontier size: 4 / 4 models
Dominated: 0 models
  slm-100m trade-off:
    Accuracy gap: 3.0%
    Cost ratio: 2500x cheaper
    Latency ratio: 53x faster
    Value score: 129333.3x

📉 Statistical Analysis...

SLM accuracy: 92.11% [95% CI: 91.91% - 92.32%]
Paired t-test vs frontier:
  t-statistic: -16.942
  p-value: 0.0000
  Cohen's d: -1.694 (large)
  Significant: YES

📝 Generating Report...

Report Summary:
  Total Models: 4
  Frontier Models: 4
  Best Accuracy: 95.0%
  Best Cost: $0.0001/1K tokens
  Best Latency: 15ms

✅ Demo complete!
```

### Benchmark Example (Py2Rs Classification)

```bash
cargo run --example benchmark --release
```

**Output:**

```
=== Py2Rs 10-Level Benchmark Demo ===

Py2Rs Benchmark Levels:
────────────────────────────────────────────────────────────────
  L 1 Hello                [Trivial] weight: 1.0
  L 2 Variables            [Trivial] weight: 1.5
  L 3 Functions            [Easy] weight: 2.0
  L 4 Collections          [Easy] weight: 3.0
  L 5 ControlFlow          [Medium] weight: 4.0
  L 6 ErrorHandling        [Medium] weight: 5.0
  L 7 OopTraits            [Hard] weight: 7.0
  L 8 Concurrency          [Hard] weight: 10.0
  L 9 FfiUnsafe            [Expert] weight: 15.0
  L10 Metaprogramming      [Expert] weight: 20.0

Example Classification:
────────────────────────────────────────────────────────────────
  hello           Difficulty: Trivial
  add             Difficulty: Easy
  fibonacci       Difficulty: Easy
  squares         Difficulty: Trivial
  binary_search   Difficulty: Easy
  safe_div        Difficulty: Medium
  shape           Difficulty: Hard
  fetch           Difficulty: Expert
  point           Difficulty: Expert

Difficulty Distribution:
────────────────────────────────────────────────────────────────
  Trivial   2 ██
  Easy      3 ███
  Medium    1 █
  Hard      1 █
  Expert    2 ██

Total Py2Rs benchmark weight: 68.5

✅ Benchmark demo complete!
```

## Using as a Library

### Pareto Analysis

```rust
use single_shot_eval::pareto::{analyze_pareto, EvalResult};
use std::time::Duration;
use std::collections::HashMap;

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

let analysis = analyze_pareto(&results);
println!("Frontier models: {}", analysis.frontier_models.len());
for tradeoff in &analysis.trade_offs {
    println!("{}: {:.0}x value score", tradeoff.model_id, tradeoff.value_score);
}
```

### Difficulty Classification

```rust
use single_shot_eval::{infer_difficulty, Difficulty, Py2RsLevel};

let python_code = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"#;

let difficulty = infer_difficulty(python_code);
println!("Difficulty: {:?}", difficulty); // Easy

// Get all Py2Rs levels
for level in Py2RsLevel::all() {
    println!("L{}: {:?} (weight: {:.1})",
        level.number(), level.difficulty(), level.weight());
}
```

### Statistical Analysis

```rust
use single_shot_eval::metrics::{bootstrap_ci, paired_t_test, StatConfig};

let accuracies: Vec<f64> = vec![0.91, 0.93, 0.92, 0.90, 0.94];
let config = StatConfig::default();

let (lower, upper) = bootstrap_ci(&accuracies, &config);
println!("95% CI: [{:.2}%, {:.2}%]", lower * 100.0, upper * 100.0);
```

## CLI Usage

See [CLI Usage](./cli-usage.md) for the full command reference.
