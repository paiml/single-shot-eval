# Corpus Module

The `corpus` module provides the Py2Rs example corpus with 10 difficulty levels.

## Loading Examples

```rust
use single_shot_eval::corpus::{Corpus, Example};

let corpus = Corpus::from_jsonl("corpus/examples.jsonl")?;

for example in corpus.iter() {
    println!("Level {}: {}", example.level, example.description);
}
```

## Example Structure

```rust
pub struct Example {
    pub id: String,
    pub python_code: String,
    pub rust_code: String,
    pub level: u8,          // 1-10 (Py2Rs level)
    pub description: String,
}
```

## Py2Rs Levels

| Level | Name | Difficulty |
|-------|------|------------|
| 1 | Syntax | Trivial |
| 2 | Types | Easy |
| 3 | Control Flow | Easy |
| 4 | Functions | Medium |
| 5 | Data Structures | Medium |
| 6 | Error Handling | Medium |
| 7 | Modules | Hard |
| 8 | Concurrency | Hard |
| 9 | Memory | Expert |
| 10 | FFI | Expert |

## Filtering

```rust
// Filter by difficulty
let hard_examples: Vec<_> = corpus.iter()
    .filter(|e| e.level >= 7)
    .collect();
```
