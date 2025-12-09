# CLI Usage

## Commands

### `run` - Execute Evaluation

```bash
single-shot-eval run [OPTIONS]
```

Options:
- `--task <PATH>` - Task YAML configuration file
- `--output <FORMAT>` - Output format (json, table, markdown)
- `--verbose` - Enable detailed logging

### `benchmark` - Corpus Classification

```bash
single-shot-eval benchmark [OPTIONS]
```

Options:
- `--corpus <PATH>` - Directory containing JSONL corpus files
- `--level <N>` - Filter by specific Py2Rs level (1-10)

### `pareto` - Analyze Results

```bash
single-shot-eval pareto [OPTIONS]
```

Options:
- `--input <PATH>` - Results JSON file
- `--metric <NAME>` - Primary metric (accuracy, latency, cost)

## Environment Variables

- `RUST_LOG` - Logging level (debug, info, warn, error)
- `SSE_MODEL_PATH` - Default model directory

## Examples

```bash
# Basic evaluation
single-shot-eval run --task tasks/py2rs.yaml

# Verbose output with JSON
RUST_LOG=debug single-shot-eval run --task tasks/py2rs.yaml --output json

# Classify corpus
single-shot-eval benchmark --corpus corpus/

# Filter by difficulty level
single-shot-eval benchmark --corpus corpus/ --level 7
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Task file not found |
| 4 | Model loading failed |
