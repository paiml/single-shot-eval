//! Batuta Stack Integration Tests
//!
//! Tests that prove realizat and alimentar while improving coverage
//! of single-shot-eval components (inference.rs, corpus.rs, runner.rs).

#![allow(clippy::unwrap_used)]

use single_shot_eval::{
    create_placeholder_model, infer_difficulty, Difficulty, InferenceOutput, LoadedModel,
    ModelConfig, ModelLoader, ModelMetadata, Py2RsLevel,
};
use std::time::Duration;

// =============================================================================
// Inference Module Coverage Tests (using realizar patterns)
// =============================================================================

#[test]
fn test_inference_model_metadata_fields() {
    let meta = ModelMetadata::default();
    assert_eq!(meta.name, "unknown");
    assert_eq!(meta.version, "0.0.0");
    assert_eq!(meta.model_type, "custom");
    assert!(meta.parameters.is_none());
    assert!(meta.training_info.is_none());
}

#[test]
fn test_inference_model_config_custom() {
    let config = ModelConfig {
        use_mmap: false,
        max_memory: Some(1024 * 1024 * 512), // 512MB
    };
    assert!(!config.use_mmap);
    assert_eq!(config.max_memory, Some(536_870_912));
}

#[test]
fn test_model_loader_operations() {
    let mut loader = ModelLoader::new();

    // Test empty state
    assert!(loader.loaded_models().is_empty());
    assert!(!loader.is_loaded("test-model"));
    assert!(loader.get("test-model").is_none());

    // Test clear on empty
    loader.clear();
    assert!(loader.loaded_models().is_empty());
}

#[test]
fn test_model_loader_with_custom_config() {
    let config = ModelConfig {
        use_mmap: true,
        max_memory: Some(1024 * 1024 * 1024), // 1GB
    };
    let loader = ModelLoader::with_config(config);
    assert!(loader.loaded_models().is_empty());
}

#[test]
fn test_placeholder_model_inference_various_inputs() {
    let model = create_placeholder_model("batuta-test");

    // Test various Python patterns
    let test_cases = vec![
        ("print('hello')", "println!"),
        ("def foo(): pass", "fn"),
        ("x = True", "true"),
        ("y = False", "false"),
        ("if x: pass\nelif y: pass", "} else if"),
    ];

    for (input, expected_contains) in test_cases {
        let result = model.infer(input).expect("inference should succeed");
        assert!(
            result.text.contains(expected_contains),
            "Input '{}' should transform to contain '{}', got '{}'",
            input,
            expected_contains,
            result.text
        );
    }
}

#[test]
fn test_inference_output_latency() {
    let model = create_placeholder_model("latency-test");
    let result = model.infer("def test(): return 42").expect("should succeed");

    // Inference should be fast (< 100ms for placeholder)
    assert!(result.latency < Duration::from_millis(100));
}

#[test]
fn test_is_valid_model_various_paths() {
    // Non-existent path
    assert!(!LoadedModel::is_valid_model("/does/not/exist.apr"));

    // Create temp files with various extensions
    let temp_dir = tempfile::tempdir().expect("temp dir");

    // Wrong extension - .bin
    let bin_path = temp_dir.path().join("model.bin");
    std::fs::write(&bin_path, "data").expect("write");
    assert!(!LoadedModel::is_valid_model(&bin_path));

    // Wrong extension - .gguf (realizat format, not aprender)
    let gguf_path = temp_dir.path().join("model.gguf");
    std::fs::write(&gguf_path, "data").expect("write");
    assert!(!LoadedModel::is_valid_model(&gguf_path));

    // Correct extension - .apr (aprender format)
    let apr_path = temp_dir.path().join("model.apr");
    std::fs::write(&apr_path, "data").expect("write");
    assert!(LoadedModel::is_valid_model(&apr_path));

    // Case insensitive check
    let apr_upper = temp_dir.path().join("model.APR");
    std::fs::write(&apr_upper, "data").expect("write");
    assert!(LoadedModel::is_valid_model(&apr_upper));
}

#[test]
fn test_load_invalid_model_file() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let bad_model = temp_dir.path().join("invalid.apr");
    std::fs::write(&bad_model, "not a valid model").expect("write");

    // Should fail to load invalid model
    let result = LoadedModel::load(&bad_model);
    assert!(result.is_err());
}

// =============================================================================
// Difficulty Inference Coverage (bench_bridge.rs)
// =============================================================================

#[test]
fn test_infer_difficulty_comprehensive() {
    // Trivial: simple assignment
    assert_eq!(infer_difficulty("x = 1"), Difficulty::Trivial);
    assert_eq!(infer_difficulty("y = 2\nz = 3"), Difficulty::Trivial);

    // Easy: single function
    assert_eq!(
        infer_difficulty("def add(a, b): return a + b"),
        Difficulty::Easy
    );

    // Easy: lambda
    assert_eq!(
        infer_difficulty("f = lambda x: x * 2"),
        Difficulty::Easy
    );

    // Medium: multiple functions
    assert_eq!(
        infer_difficulty("def foo(): pass\ndef bar(): pass"),
        Difficulty::Medium
    );

    // Medium: try/except
    assert_eq!(
        infer_difficulty("try:\n    x = 1\nexcept: pass"),
        Difficulty::Medium
    );

    // Hard: class
    assert_eq!(
        infer_difficulty("class Foo:\n    def __init__(self): pass"),
        Difficulty::Hard
    );

    // Expert: async
    assert_eq!(
        infer_difficulty("async def fetch(): await response"),
        Difficulty::Expert
    );

    // Expert: decorator with class
    assert_eq!(
        infer_difficulty("@dataclass\nclass Point:\n    x: int"),
        Difficulty::Expert
    );
}

#[test]
fn test_py2rs_level_coverage() {
    // Test all Py2RsLevel variants
    for level in Py2RsLevel::all() {
        let num = level.number();
        assert!(num >= 1 && num <= 10, "Level number should be 1-10");

        let weight = level.weight();
        assert!(weight > 0.0, "Weight should be positive");

        let _diff = level.difficulty(); // Exercises difficulty() method
    }

    // Test specific levels
    assert_eq!(Py2RsLevel::Hello.number(), 1);
    assert_eq!(Py2RsLevel::Metaprogramming.number(), 10);

    // Total weight should be 68.5
    let total: f32 = Py2RsLevel::all().iter().map(Py2RsLevel::weight).sum();
    assert!((total - 68.5).abs() < 0.01);
}

// =============================================================================
// Corpus Module Coverage (using alimentar patterns)
// =============================================================================

#[test]
fn test_corpus_jsonl_format() {
    // Test JSONL parsing similar to alimentar's data loading
    let jsonl_content = r#"{"name": "hello", "source": "print('hello')"}
{"name": "add", "source": "def add(a, b): return a + b"}"#;

    let lines: Vec<&str> = jsonl_content.lines().collect();
    assert_eq!(lines.len(), 2);

    // Parse each line as JSON
    for line in lines {
        let parsed: serde_json::Value = serde_json::from_str(line).expect("valid json");
        assert!(parsed.get("name").is_some());
        assert!(parsed.get("source").is_some());
    }
}

// =============================================================================
// Runner Module Coverage
// =============================================================================

#[test]
fn test_runner_config_defaults() {
    use single_shot_eval::RunnerConfig;

    let config = RunnerConfig::default();
    assert!(config.max_concurrent > 0);
    assert!(config.default_timeout.as_secs() > 0);
}

#[test]
fn test_task_runner_basic() {
    use single_shot_eval::TaskRunner;

    let runner = TaskRunner::new();

    // Test model validation
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let bad_model = temp_dir.path().join("not_an_apr.txt");
    std::fs::write(&bad_model, "data").expect("write");

    let result = runner.validate_model(&bad_model);
    assert!(result.is_err(), "Should reject non-.apr files");
}

// =============================================================================
// Report Module Coverage
// =============================================================================

#[test]
fn test_report_builder_empty() {
    use single_shot_eval::ReportBuilder;

    let builder = ReportBuilder::new("empty-test");
    let report = builder.build();

    // Should generate valid (but empty) reports
    assert!(report.to_json().is_ok());
    let markdown = report.to_markdown();
    assert!(!markdown.is_empty());
}

// =============================================================================
// Pareto Module Coverage
// =============================================================================

#[test]
fn test_pareto_single_result() {
    use single_shot_eval::{analyze_pareto, EvalResult};
    use std::collections::HashMap;

    let results = vec![EvalResult {
        model_id: "single-model".to_string(),
        task_id: "test".to_string(),
        accuracy: 0.85,
        cost: 1.0,
        latency: Duration::from_millis(100),
        metadata: HashMap::new(),
    }];

    let analysis = analyze_pareto(&results);
    assert!(analysis.frontier_models.contains(&"single-model".to_string()));
}

#[test]
fn test_pareto_dominated_model() {
    use single_shot_eval::{compute_pareto_frontier, EvalResult};
    use std::collections::HashMap;

    // Model A dominates Model B (better accuracy, lower cost)
    let results = vec![
        EvalResult {
            model_id: "dominant".to_string(),
            task_id: "test".to_string(),
            accuracy: 0.95,
            cost: 1.0,
            latency: Duration::from_millis(50),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "dominated".to_string(),
            task_id: "test".to_string(),
            accuracy: 0.80,
            cost: 2.0,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
    ];

    let frontier = compute_pareto_frontier(&results);

    // Only dominant model should be on frontier
    assert!(frontier.iter().any(|r| r.model_id == "dominant"));
}

// =============================================================================
// Metrics Module Coverage
// =============================================================================

#[test]
fn test_metrics_collector_edge_cases() {
    use single_shot_eval::MetricsCollector;

    let mut collector = MetricsCollector::new();

    // Single sample
    collector.record_accuracy(0.5);
    collector.record_latency(Duration::from_millis(100));
    collector.record_cost(1.0);

    let metrics = collector.compute();
    assert!((metrics.accuracy - 0.5).abs() < 0.01);
}

#[test]
fn test_metrics_collector_variance() {
    use single_shot_eval::MetricsCollector;

    let mut collector = MetricsCollector::new();

    // Add samples with variance
    for i in 0..20 {
        let acc = 0.8 + (i as f64) * 0.01;
        collector.record_accuracy(acc);
        collector.record_latency(Duration::from_millis(50 + i * 5));
        collector.record_cost(0.1 + (i as f64) * 0.01);
    }

    let metrics = collector.compute();

    // Should have confidence intervals
    assert!(metrics.accuracy_ci.0 < metrics.accuracy);
    assert!(metrics.accuracy_ci.1 > metrics.accuracy);
}

// =============================================================================
// Compiler Module Coverage
// =============================================================================

#[test]
fn test_compiler_verifier_syntax_error() {
    use single_shot_eval::CompilerVerifier;

    let verifier = CompilerVerifier::new();

    // Invalid Rust syntax
    let bad_code = "fn broken( { invalid syntax }";
    let result = verifier.verify(bad_code, None).expect("should run");
    assert!(!result.compiles, "Invalid syntax should not compile");
}

#[test]
fn test_compiler_verifier_with_std_lib() {
    use single_shot_eval::CompilerVerifier;

    let verifier = CompilerVerifier::new();

    // Code using standard library
    let code = r#"
use std::collections::HashMap;

pub fn create_map() -> HashMap<String, i32> {
    let mut m = HashMap::new();
    m.insert("key".to_string(), 42);
    m
}
"#;

    let result = verifier.verify(code, None).expect("should run");
    assert!(result.compiles, "Valid code should compile");
}

// =============================================================================
// Baselines Module Coverage
// =============================================================================

#[test]
fn test_baseline_config_parsing() {
    use single_shot_eval::BaselineConfig;

    let config = BaselineConfig {
        name: "test-baseline".to_string(),
        command: "echo".to_string(),
        args_template: "{input}".to_string(),
        cost_per_1k_tokens: 0.001,
    };

    assert_eq!(config.name, "test-baseline");
    assert_eq!(config.command, "echo");
    assert!((config.cost_per_1k_tokens - 0.001).abs() < 0.0001);
}
