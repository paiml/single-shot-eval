# Custom Baselines

Configure external model baselines for comparison.

## Task Configuration

```yaml
# tasks/with-baselines.yaml
task_id: py2rs-comparison
description: Compare local SLM against cloud models
corpus_path: corpus/examples.jsonl
model_path: models/slm.apr
baselines:
  - name: claude-sonnet
    command: claude -m claude-sonnet-4-20250514
  - name: gemini-pro
    command: gemini -m gemini-pro
  - name: gpt-4o
    command: openai -m gpt-4o
```

## Running Comparison

```bash
single-shot-eval run --task tasks/with-baselines.yaml --output table
```

## Output with Pareto Analysis

```
Model Comparison Results
========================
┌─────────────┬──────────┬──────────┬─────────────────┬─────────┐
│ Model       │ Accuracy │ Latency  │ 95% CI          │ Pareto  │
├─────────────┼──────────┼──────────┼─────────────────┼─────────┤
│ slm-local   │ 87.3%    │ 45ms     │ [85.1%, 89.5%]  │ *       │
│ claude-son  │ 92.1%    │ 850ms    │ [90.3%, 93.9%]  │ *       │
│ gemini-pro  │ 89.5%    │ 620ms    │ [87.2%, 91.8%]  │         │
│ gpt-4o      │ 91.8%    │ 780ms    │ [89.9%, 93.7%]  │         │
└─────────────┴──────────┴──────────┴─────────────────┴─────────┘

* = Pareto optimal (not dominated by any other model)
```

## Custom Command Format

Baseline commands receive input via stdin and should output to stdout:

```bash
# Command format
echo "def add(a, b): return a + b" | claude -m claude-sonnet-4-20250514
```

## Timeout Configuration

```yaml
baselines:
  - name: slow-model
    command: custom-model --verbose
    timeout_seconds: 120  # Default: 60
```
