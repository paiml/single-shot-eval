# Py2Rs Benchmark

The Py2Rs benchmark classifies Python-to-Rust translation difficulty across 10 levels.

## Run the Benchmark Demo

```bash
cargo run --example benchmark --release
```

## The 10 Levels

Based on aprender's canonical Py2Rs framework:

| Level | Name | Difficulty | Weight |
|-------|------|------------|--------|
| L1 | Hello | Trivial | 1.0 |
| L2 | Variables | Trivial | 1.5 |
| L3 | Functions | Easy | 2.0 |
| L4 | Collections | Easy | 3.0 |
| L5 | ControlFlow | Medium | 4.0 |
| L6 | ErrorHandling | Medium | 5.0 |
| L7 | OopTraits | Hard | 7.0 |
| L8 | Concurrency | Hard | 10.0 |
| L9 | FfiUnsafe | Expert | 15.0 |
| L10 | Metaprogramming | Expert | 20.0 |

**Total weight: 68.5**

## Example Classification

The benchmark demo classifies Python snippets:

```
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
```

## Difficulty Distribution

```
Difficulty Distribution:
────────────────────────────────────────────────────────────────
  Trivial   2 ██
  Easy      3 ███
  Medium    1 █
  Hard      1 █
  Expert    2 ██
```

## Classification Rules

### Trivial (L1-L2)
- Simple print statements
- Variable assignments
- Basic type conversions

```python
# L1: Hello
print("Hello, World!")

# L2: Variables
x = 42
name = "Alice"
```

### Easy (L3-L4)
- Function definitions
- Return values
- List comprehensions

```python
# L3: Functions
def add(a, b):
    return a + b

# L4: Collections
squares = [x**2 for x in range(10)]
```

### Medium (L5-L6)
- Loops and conditionals
- Exception handling
- Error propagation

```python
# L5: ControlFlow
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid

# L6: ErrorHandling
def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
```

### Hard (L7-L8)
- Classes and inheritance
- Traits/interfaces
- Async/await

```python
# L7: OopTraits
class Shape:
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2

# L8: Concurrency
async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### Expert (L9-L10)
- FFI/unsafe code
- Decorators
- Metaclasses
- Dataclasses

```python
# L9: FFI/Unsafe - ctypes, raw pointers

# L10: Metaprogramming
@dataclass
class Point:
    x: float
    y: float
    def distance(self, other: 'Point') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
```

## Programmatic Usage

```rust
use single_shot_eval::{infer_difficulty, Difficulty, Py2RsLevel};

fn main() {
    // Classify a single example
    let python_code = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"#;

    let difficulty = infer_difficulty(python_code);
    println!("Difficulty: {:?}", difficulty);

    // Iterate all levels
    for level in Py2RsLevel::all() {
        println!(
            "L{:>2} {:20} [{:?}] weight: {:.1}",
            level.number(),
            format!("{:?}", level),
            level.difficulty(),
            level.weight()
        );
    }

    // Check difficulty distribution
    let examples = vec![
        r#"print("hello")"#,
        r#"def add(a, b): return a + b"#,
        r#"async def fetch(): pass"#,
    ];

    let mut counts = std::collections::HashMap::new();
    for ex in &examples {
        let diff = infer_difficulty(ex);
        *counts.entry(diff).or_insert(0) += 1;
    }

    for (diff, count) in counts {
        println!("{:?}: {}", diff, count);
    }
}
```

## CLI Usage

Classify a corpus from the command line:

```bash
single-shot-eval benchmark --corpus ./experiments/python-corpus/
```

Output:
```
┌────────────────────────────────────────────────────────────────┐
│ Py2Rs 10-Level Benchmark Analysis                              │
├────────────────────────────────────────────────────────────────┤
│ Corpus:   50 examples                                          │
└────────────────────────────────────────────────────────────────┘

Level Distribution
──────────────────
L3  (Functions)      [████░░░░░░░░░░░░░░░░]  40% (20 examples)
L5  (ControlFlow)    [███░░░░░░░░░░░░░░░░░]  30% (15 examples)
...

Visual Summary (● = has examples, ○ = empty)
Levels 1-10: ○○●●●●○○○○
```

## Next Steps

- [Custom Baselines](./custom-baselines.md) - Adding external model baselines
- [Pareto Analysis](../architecture/pareto-analysis.md) - Understanding trade-offs
