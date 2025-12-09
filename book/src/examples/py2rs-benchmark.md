# Py2Rs Benchmark

Full benchmark using the 10-level Py2Rs classification system.

## Level Distribution

```bash
single-shot-eval benchmark --corpus corpus/
```

Output:
```
Py2Rs Level Distribution
========================
Level 1 (Syntax):       ████████████ 120 examples
Level 2 (Types):        ██████████   98 examples
Level 3 (Control):      █████████    87 examples
Level 4 (Functions):    ████████     76 examples
Level 5 (Data):         ███████      65 examples
Level 6 (Errors):       █████        54 examples
Level 7 (Modules):      ████         43 examples
Level 8 (Concurrency):  ███          32 examples
Level 9 (Memory):       ██           21 examples
Level 10 (FFI):         █            10 examples
```

## Per-Level Evaluation

```bash
# Evaluate only hard examples (levels 7-10)
single-shot-eval benchmark --corpus corpus/ --level 7
single-shot-eval benchmark --corpus corpus/ --level 8
single-shot-eval benchmark --corpus corpus/ --level 9
single-shot-eval benchmark --corpus corpus/ --level 10
```

## Difficulty Breakdown

```rust
use single_shot_eval::corpus::Corpus;

let corpus = Corpus::from_jsonl("corpus/examples.jsonl")?;

let trivial = corpus.iter().filter(|e| e.level == 1).count();
let easy = corpus.iter().filter(|e| e.level >= 2 && e.level <= 3).count();
let medium = corpus.iter().filter(|e| e.level >= 4 && e.level <= 6).count();
let hard = corpus.iter().filter(|e| e.level >= 7 && e.level <= 8).count();
let expert = corpus.iter().filter(|e| e.level >= 9).count();

println!("Trivial: {}, Easy: {}, Medium: {}, Hard: {}, Expert: {}",
         trivial, easy, medium, hard, expert);
```
