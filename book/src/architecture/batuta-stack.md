# Batuta Stack Integration

Single-shot-eval is built on the Batuta sovereign AI stack - a collection of Rust crates for ML inference without external dependencies.

## Stack Components

| Crate | Version | Purpose |
|-------|---------|---------|
| `aprender` | 0.16 | Core ML library - model format, inference, metrics |
| `entrenar` | 0.2 | Training, distillation, model conversion |
| `realizar` | 0.2 | High-performance inference engine |
| `alimentar` | 0.2 | Data loading and preprocessing |
| `trueno` | 0.7 | SIMD/GPU acceleration (used by aprender) |

## Integration Points

### aprender::bench

The Py2Rs benchmark framework from aprender provides:

```rust
use aprender::bench::{Difficulty, Py2RsLevel};

// 10 canonical difficulty levels
let level = Py2RsLevel::Concurrency;
assert_eq!(level.number(), 8);
assert_eq!(level.difficulty(), Difficulty::Hard);
```

### aprender::format

Model loading via the `.apr` format:

```rust
use aprender::format::{load, ModelType};

let model = load::<NeuralModel>(path, ModelType::NeuralSequential)?;
```

### alimentar (Testing)

Used in integration tests to validate corpus loading:

```rust
use alimentar::loaders::JsonlLoader;

let loader = JsonlLoader::new(path)?;
for example in loader.iter() {
    // Process examples
}
```

## Why Batuta?

1. **No External Dependencies**: Pure Rust, no Python/PyTorch required
2. **Sovereign Stack**: Control the full inference pipeline
3. **OFFLINE-FIRST**: Works without network access
4. **High Performance**: SIMD acceleration via trueno
