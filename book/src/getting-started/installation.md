# Installation

## Prerequisites

- Rust 1.75 or later
- Cargo (comes with Rust)

## From Source

```bash
git clone https://github.com/paiml/single-shot-eval
cd single-shot-eval
cargo build --release
```

The binary will be at `target/release/single-shot-eval`.

## As a Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
single_shot_eval = "0.1"
```

## Verify Installation

```bash
single-shot-eval --version
single-shot-eval --help
```

## Batuta Stack Dependencies

Single-shot-eval uses the Batuta sovereign AI stack:

| Crate | Purpose |
|-------|---------|
| `aprender` | ML library - model loading, inference, benchmarks |
| `entrenar` | Training/distillation/conversion |
| `realizar` | Inference engine (dev-dependency for testing) |
| `alimentar` | Data loading (dev-dependency for testing) |

All dependencies are from crates.io - no git dependencies required.
