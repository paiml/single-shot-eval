# Architecture Overview

Single-shot-eval is designed as a modular evaluation framework with clear separation of concerns.

## Module Structure

```
src/
├── lib.rs          # Public API re-exports
├── main.rs         # CLI entry point
├── config.rs       # Task YAML parsing
├── corpus.rs       # Py2Rs example corpus
├── bench_bridge.rs # Aprender benchmark integration
├── pareto.rs       # Pareto frontier analysis
├── runner.rs       # Evaluation orchestration
├── report.rs       # Result formatting
├── baselines.rs    # CLI baseline wrappers
└── inference.rs    # Model loading (aprender)
```

## Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Config    │───▶│   Runner    │───▶│   Report    │
│  (YAML)     │    │             │    │  (Output)   │
└─────────────┘    └──────┬──────┘    └─────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Corpus  │    │ Inference│    │ Baselines│
   │ (JSONL)  │    │ (aprender)│   │  (CLI)   │
   └──────────┘    └──────────┘    └──────────┘
                         │
                         ▼
                   ┌──────────┐
                   │  Pareto  │
                   │ Analysis │
                   └──────────┘
```

## Key Design Principles

### 1. Offline-First
No HTTP dependencies. All models load from local `.apr` files. Baselines execute via local CLI tools.

### 2. Statistical Rigor
- Bootstrap confidence intervals (1000 iterations)
- Mann-Whitney U significance testing
- Pareto dominance analysis

### 3. Batuta Stack Integration
Uses the sovereign AI stack:
- `aprender::bench` for Py2Rs benchmark levels
- `aprender::format` for model loading
- `alimentar` for JSONL corpus loading (tests)
- `realizar` for inference engine validation (tests)
