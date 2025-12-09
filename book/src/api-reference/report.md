# Report Module

The `report` module formats evaluation results for output.

## ReportFormat

```rust
use single_shot_eval::report::{Report, ReportFormat};

let report = Report::from_results(&results);

// Output formats
println!("{}", report.to_table());
println!("{}", report.to_json()?);
println!("{}", report.to_markdown());
```

## Table Output

```
┌─────────────┬──────────┬──────────┬─────────────────┐
│ Model       │ Accuracy │ Latency  │ 95% CI          │
├─────────────┼──────────┼──────────┼─────────────────┤
│ slm-local   │ 87.3%    │ 45ms     │ [85.1%, 89.5%]  │
│ claude-son  │ 92.1%    │ 850ms    │ [90.3%, 93.9%]  │
│ gemini-pro  │ 89.5%    │ 620ms    │ [87.2%, 91.8%]  │
└─────────────┴──────────┴──────────┴─────────────────┘
```

## JSON Output

```json
{
  "results": [
    {
      "model": "slm-local",
      "accuracy": 0.873,
      "latency_ms": 45.2,
      "ci_lower": 0.851,
      "ci_upper": 0.895
    }
  ],
  "pareto_optimal": ["slm-local", "claude-sonnet"]
}
```

## Statistical Annotations

Reports include:
- Bootstrap confidence intervals (default: 1000 iterations)
- Mann-Whitney U significance markers
- Pareto optimality indicators
