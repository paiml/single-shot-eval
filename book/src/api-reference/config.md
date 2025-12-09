# Config Module

The `config` module handles task YAML parsing and validation.

## TaskConfig

```rust
use single_shot_eval::config::TaskConfig;

let config = TaskConfig::from_file("tasks/example.yaml")?;

println!("Task: {}", config.task_id);
println!("Corpus: {}", config.corpus_path);
```

## Structure

```yaml
task_id: string          # Unique task identifier
description: string      # Human-readable description
corpus_path: string      # Path to JSONL corpus
model_path: string       # Path to .apr model file
baselines:               # Optional baseline comparisons
  - name: string
    command: string
```

## Validation

The config parser validates:
- Required fields are present
- Paths exist (corpus, model)
- Baseline commands are executable
