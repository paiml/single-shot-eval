//! Bridge module for `aprender::bench` integration.
//!
//! This module re-exports and extends `aprender::bench` types for use in
//! single-shot evaluation. It provides:
//!
//! - Re-exports of core types (`Difficulty`, `Example`, `ModelComparison`)
//! - `Py2RsLevel` for 10-level Python-to-Rust benchmarking
//! - Conversion from corpus `PythonExample` to bench `Example`
//!
//! # Example
//!
//! ```
//! use single_shot_eval::bench_bridge::{Difficulty, Py2RsLevel, to_bench_example};
//! use single_shot_eval::corpus::PythonExample;
//!
//! // Map difficulty levels
//! let level = Py2RsLevel::Functions;
//! assert_eq!(level.difficulty(), Difficulty::Easy);
//! ```

// Re-export aprender::bench types
pub use aprender::bench::{
    Difficulty, EvalResult as BenchEvalResult, EvalSuiteConfig, EvalTask, Example as BenchExample,
    ExampleResult as BenchExampleResult, ExampleStatus, ModelComparison, ParetoPoint,
    Recommendation,
};

// Re-export py2rs module
pub use aprender::bench::py2rs::{
    format_comparison_table, generate_canonical_examples, run_benchmark, LevelResult, Py2RsLevel,
    Py2RsScore,
};

use crate::corpus::PythonExample;

/// Convert a `PythonExample` to `aprender::bench::Example`
///
/// Maps the corpus example to the canonical bench format, inferring
/// difficulty from function complexity heuristics.
#[must_use]
pub fn to_bench_example(example: &PythonExample) -> BenchExample {
    let difficulty = infer_difficulty(&example.source);
    let expected = format!("Compile to valid Rust: {}", example.name);

    BenchExample::new(&example.name, &example.source, expected)
        .with_difficulty(difficulty)
        .with_tags(vec!["corpus".to_string(), "py2rs".to_string()])
}

/// Infer difficulty from Python source code
///
/// Uses heuristics based on code complexity:
/// - Trivial: < 3 lines, no control flow
/// - Easy: Simple functions, basic control flow
/// - Medium: Multiple functions, error handling
/// - Hard: Classes, complex algorithms
/// - Expert: Async, decorators, metaprogramming
#[must_use]
pub fn infer_difficulty(source: &str) -> Difficulty {
    let lines = source.lines().count();
    let has_class = source.contains("class ");
    let has_async = source.contains("async ") || source.contains("await ");
    let has_decorator = source.contains('@');
    let has_try = source.contains("try:") || source.contains("except ");
    let has_def = source.matches("def ").count();
    let has_lambda = source.contains("lambda ");
    let has_comprehension = source.contains(" for ") && (source.contains('[') || source.contains('{'));

    // Expert: async, decorators, or metaprogramming
    if has_async || (has_decorator && has_class) {
        return Difficulty::Expert;
    }

    // Hard: classes or complex patterns
    if has_class || (has_def > 2 && has_try) {
        return Difficulty::Hard;
    }

    // Medium: multiple functions, error handling, or comprehensions
    if has_try || has_def > 1 || (has_comprehension && lines > 5) {
        return Difficulty::Medium;
    }

    // Easy: single function with some logic
    if has_def > 0 || has_lambda || lines > 3 {
        return Difficulty::Easy;
    }

    // Trivial: simple statements
    Difficulty::Trivial
}

/// Map `Py2RsLevel` to corpus example ID pattern
///
/// Returns the expected example ID prefix for a given benchmark level.
#[must_use]
pub const fn level_to_example_pattern(level: Py2RsLevel) -> &'static str {
    match level {
        Py2RsLevel::Hello => "hello",
        Py2RsLevel::Variables => "abs|factorial|gcd",
        Py2RsLevel::Functions => "fibonacci|is_prime|lcm",
        Py2RsLevel::Collections => "sum|max|min|filter",
        Py2RsLevel::ControlFlow => "binary_search|sort",
        Py2RsLevel::ErrorHandling => "parse|read|config",
        Py2RsLevel::OopTraits => "class|shape|animal",
        Py2RsLevel::Concurrency => "async|concurrent",
        Py2RsLevel::FfiUnsafe => "ffi|unsafe|ctypes",
        Py2RsLevel::Metaprogramming => "decorator|metaclass|dataclass",
    }
}

/// Classify a corpus example into a `Py2RsLevel`
#[must_use]
pub fn classify_example_level(example: &PythonExample) -> Py2RsLevel {
    let source = &example.source;
    let name = &example.name;

    // Check for specific patterns
    if source.contains("async ") || source.contains("await ") {
        return Py2RsLevel::Concurrency;
    }
    if source.contains("@dataclass") || source.contains("@property") {
        return Py2RsLevel::Metaprogramming;
    }
    if source.contains("ctypes") || source.contains("cffi") {
        return Py2RsLevel::FfiUnsafe;
    }
    if source.contains("class ") && source.contains("def ") {
        return Py2RsLevel::OopTraits;
    }
    if source.contains("try:") || source.contains("except ") {
        return Py2RsLevel::ErrorHandling;
    }

    // Check by complexity
    let def_count = source.matches("def ").count();
    let has_loop = source.contains("for ") || source.contains("while ");
    let has_comprehension = (source.contains('[') && source.contains(" for "))
        || (source.contains('{') && source.contains(" for "));

    if has_comprehension {
        return Py2RsLevel::Collections;
    }
    if has_loop && def_count > 0 {
        return Py2RsLevel::ControlFlow;
    }
    if def_count > 0 {
        return Py2RsLevel::Functions;
    }

    // Check by name patterns
    let name_lower = name.to_lowercase();
    if name_lower.contains("abs") || name_lower.contains("factorial") {
        return Py2RsLevel::Variables;
    }

    Py2RsLevel::Hello
}

/// Batch convert corpus examples to bench examples
#[must_use]
pub fn batch_to_bench_examples(examples: &[PythonExample]) -> Vec<BenchExample> {
    examples.iter().map(to_bench_example).collect()
}

/// Create a `ModelComparison` from evaluation results
#[must_use]
pub fn create_model_comparison(task_id: &str) -> ModelComparison {
    ModelComparison::new(task_id)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_example(name: &str, source: &str) -> PythonExample {
        let mut files = HashMap::new();
        files.insert(format!("{name}.py"), source.to_string());
        PythonExample {
            name: name.to_string(),
            path: PathBuf::from("/test"),
            source: source.to_string(),
            tests: None,
            files,
            test_cases: Vec::new(),
        }
    }

    // TDD: Test re-exports work
    #[test]
    fn test_reexports_available() {
        // Verify re-exports compile and are accessible
        let _: Difficulty = Difficulty::Medium;
        let _: Py2RsLevel = Py2RsLevel::Hello;
        assert_eq!(Py2RsLevel::all().len(), 10);
    }

    // TDD: Test difficulty inference
    #[test]
    fn test_infer_difficulty_trivial() {
        let source = "x = 42";
        assert_eq!(infer_difficulty(source), Difficulty::Trivial);
    }

    #[test]
    fn test_infer_difficulty_easy() {
        let source = "def add(a, b):\n    return a + b";
        assert_eq!(infer_difficulty(source), Difficulty::Easy);
    }

    #[test]
    fn test_infer_difficulty_medium() {
        let source = "def foo():\n    pass\ndef bar():\n    pass";
        assert_eq!(infer_difficulty(source), Difficulty::Medium);
    }

    #[test]
    fn test_infer_difficulty_hard() {
        let source = "class Foo:\n    def __init__(self):\n        pass";
        assert_eq!(infer_difficulty(source), Difficulty::Hard);
    }

    #[test]
    fn test_infer_difficulty_expert() {
        let source = "async def fetch():\n    await response";
        assert_eq!(infer_difficulty(source), Difficulty::Expert);
    }

    // TDD: Test conversion
    #[test]
    fn test_to_bench_example() {
        let example = make_example("factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)");
        let bench_ex = to_bench_example(&example);

        assert_eq!(bench_ex.id, "factorial");
        assert!(bench_ex.input.contains("def factorial"));
        assert!(bench_ex.tags.contains(&"py2rs".to_string()));
    }

    // TDD: Test level classification
    #[test]
    fn test_classify_example_level_hello() {
        let example = make_example("hello", "print('hello world')");
        assert_eq!(classify_example_level(&example), Py2RsLevel::Hello);
    }

    #[test]
    fn test_classify_example_level_functions() {
        let example = make_example(
            "fibonacci",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        );
        assert_eq!(classify_example_level(&example), Py2RsLevel::Functions);
    }

    #[test]
    fn test_classify_example_level_collections() {
        let example = make_example("squares", "squares = [x**2 for x in range(10)]");
        assert_eq!(classify_example_level(&example), Py2RsLevel::Collections);
    }

    #[test]
    fn test_classify_example_level_error_handling() {
        let example = make_example(
            "safe_div",
            "def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return None",
        );
        assert_eq!(classify_example_level(&example), Py2RsLevel::ErrorHandling);
    }

    #[test]
    fn test_classify_example_level_oop() {
        let example = make_example(
            "shape",
            "class Shape:\n    def area(self):\n        pass",
        );
        assert_eq!(classify_example_level(&example), Py2RsLevel::OopTraits);
    }

    #[test]
    fn test_classify_example_level_async() {
        let example = make_example(
            "fetch",
            "async def fetch(url):\n    response = await client.get(url)\n    return response",
        );
        assert_eq!(classify_example_level(&example), Py2RsLevel::Concurrency);
    }

    #[test]
    fn test_classify_example_level_metaprogramming() {
        let example = make_example(
            "point",
            "@dataclass\nclass Point:\n    x: float\n    y: float",
        );
        assert_eq!(classify_example_level(&example), Py2RsLevel::Metaprogramming);
    }

    // TDD: Test batch conversion
    #[test]
    fn test_batch_to_bench_examples() {
        let examples = vec![
            make_example("hello", "print('hi')"),
            make_example("add", "def add(a, b): return a + b"),
        ];
        let bench_examples = batch_to_bench_examples(&examples);

        assert_eq!(bench_examples.len(), 2);
        assert_eq!(bench_examples[0].id, "hello");
        assert_eq!(bench_examples[1].id, "add");
    }

    // TDD: Test ModelComparison creation
    #[test]
    fn test_create_model_comparison() {
        let comparison = create_model_comparison("py2rs-test");
        assert_eq!(comparison.task_id, "py2rs-test");
        assert!(comparison.results.is_empty());
    }

    // TDD: Test Py2RsLevel weights sum
    #[test]
    fn test_py2rs_level_weights() {
        let total: f32 = Py2RsLevel::all().iter().map(Py2RsLevel::weight).sum();
        assert!((total - 68.5).abs() < 0.01);
    }

    // TDD: Test difficulty mapping consistency
    #[test]
    fn test_py2rs_level_difficulty_mapping() {
        // Verify aprender's level-to-difficulty mapping
        assert_eq!(Py2RsLevel::Hello.difficulty(), Difficulty::Trivial);
        assert_eq!(Py2RsLevel::Variables.difficulty(), Difficulty::Trivial);
        assert_eq!(Py2RsLevel::Functions.difficulty(), Difficulty::Easy);
        assert_eq!(Py2RsLevel::Collections.difficulty(), Difficulty::Easy);
        assert_eq!(Py2RsLevel::ControlFlow.difficulty(), Difficulty::Medium);
        assert_eq!(Py2RsLevel::ErrorHandling.difficulty(), Difficulty::Medium);
        assert_eq!(Py2RsLevel::OopTraits.difficulty(), Difficulty::Hard);
        assert_eq!(Py2RsLevel::Concurrency.difficulty(), Difficulty::Hard);
        assert_eq!(Py2RsLevel::FfiUnsafe.difficulty(), Difficulty::Expert);
        assert_eq!(Py2RsLevel::Metaprogramming.difficulty(), Difficulty::Expert);
    }
}
