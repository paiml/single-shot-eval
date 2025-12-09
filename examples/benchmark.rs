//! Demo: Py2Rs 10-level benchmark classification
//!
//! Shows how to use the bench_bridge module to classify Python examples
//! against the canonical aprender Py2Rs benchmark framework.

use single_shot_eval::{infer_difficulty, Difficulty, Py2RsLevel};

fn main() {
    println!("=== Py2Rs 10-Level Benchmark Demo ===\n");

    // Example Python snippets at different levels
    let examples = [
        ("hello", r#"print("Hello, World!")"#),
        ("add", r#"def add(a, b):
    return a + b"#),
        ("fibonacci", r#"def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"#),
        ("squares", r#"squares = [x**2 for x in range(10)]"#),
        ("binary_search", r#"def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"#),
        ("safe_div", r#"def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None"#),
        ("shape", r#"class Shape:
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2"#),
        ("fetch", r#"async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()"#),
        ("point", r#"@dataclass
class Point:
    x: float
    y: float

    def distance(self, other: 'Point') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5"#),
    ];

    // Show Py2Rs levels and their characteristics
    println!("Py2Rs Benchmark Levels:");
    println!("────────────────────────────────────────────────────────────────");
    for level in Py2RsLevel::all() {
        println!(
            "  L{:>2} {:20} [{:?}] weight: {:.1}",
            level.number(),
            format!("{level:?}"),
            level.difficulty(),
            level.weight()
        );
    }
    println!();

    // Classify each example
    println!("Example Classification:");
    println!("────────────────────────────────────────────────────────────────");
    for (name, source) in &examples {
        let difficulty = infer_difficulty(source);
        println!(
            "  {:<15} Difficulty: {:?}",
            name, difficulty
        );
    }
    println!();

    // Summary statistics
    println!("Difficulty Distribution:");
    println!("────────────────────────────────────────────────────────────────");
    let mut counts = std::collections::HashMap::new();
    for (_, source) in &examples {
        let diff = infer_difficulty(source);
        *counts.entry(diff).or_insert(0) += 1;
    }

    for difficulty in [
        Difficulty::Trivial,
        Difficulty::Easy,
        Difficulty::Medium,
        Difficulty::Hard,
        Difficulty::Expert,
    ] {
        let count = counts.get(&difficulty).unwrap_or(&0);
        let bar = "█".repeat(*count);
        println!("  {:8} {:2} {}", format!("{difficulty:?}"), count, bar);
    }
    println!();

    // Show total weight coverage
    let total_weight: f32 = Py2RsLevel::all().iter().map(Py2RsLevel::weight).sum();
    println!("Total Py2Rs benchmark weight: {:.1}", total_weight);
    println!();

    println!("✅ Benchmark demo complete!");
}
