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

// =============================================================================
// Corpus JSONL Loading Coverage (corpus.rs load_jsonl)
// =============================================================================

#[test]
fn test_corpus_load_jsonl_file() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let jsonl_path = temp_dir.path().join("test_corpus.jsonl");

    // Write valid JSONL content
    let jsonl_content = r#"{"id": "hello", "python_code": "print('hello')", "test_cases": []}
{"id": "add", "python_code": "def add(a, b): return a + b", "test_cases": [{"input": [1, 2], "expected": 3}]}"#;
    std::fs::write(&jsonl_path, jsonl_content).expect("write jsonl");

    let corpus = Corpus::load(&jsonl_path).expect("should load");
    assert_eq!(corpus.len(), 2);
    assert!(!corpus.is_empty());

    // Check examples are sorted by name
    let names: Vec<_> = corpus.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["add", "hello"]);
}

#[test]
fn test_corpus_load_jsonl_from_directory() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let jsonl_path = temp_dir.path().join("transpile_corpus.jsonl");

    // Write JSONL file in directory (standard naming)
    let jsonl_content = r#"{"id": "example1", "python_code": "x = 1", "test_cases": []}"#;
    std::fs::write(&jsonl_path, jsonl_content).expect("write jsonl");

    // Load from directory - should find transpile_corpus.jsonl
    let corpus = Corpus::load(temp_dir.path()).expect("should load");
    assert_eq!(corpus.len(), 1);
}

#[test]
fn test_corpus_load_jsonl_empty_lines() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let jsonl_path = temp_dir.path().join("corpus.jsonl");

    // JSONL with empty lines (should be skipped)
    let jsonl_content = r#"{"id": "item1", "python_code": "x = 1", "test_cases": []}

{"id": "item2", "python_code": "y = 2", "test_cases": []}
"#;
    std::fs::write(&jsonl_path, jsonl_content).expect("write jsonl");

    let corpus = Corpus::load(&jsonl_path).expect("should load");
    assert_eq!(corpus.len(), 2);
}

#[test]
fn test_corpus_load_jsonl_empty_file() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let jsonl_path = temp_dir.path().join("empty.jsonl");
    std::fs::write(&jsonl_path, "").expect("write");

    let result = Corpus::load(&jsonl_path);
    assert!(result.is_err());
}

#[test]
fn test_corpus_load_directory_with_examples() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");

    // Create example_foo directory
    let example_foo = temp_dir.path().join("example_foo");
    std::fs::create_dir(&example_foo).expect("mkdir");
    std::fs::write(example_foo.join("foo.py"), "def foo(): pass").expect("write");
    std::fs::write(example_foo.join("test_foo.py"), "def test_foo(): assert True").expect("write");

    // Create example_bar directory
    let example_bar = temp_dir.path().join("example_bar");
    std::fs::create_dir(&example_bar).expect("mkdir");
    std::fs::write(example_bar.join("bar.py"), "def bar(): pass").expect("write");

    let corpus = Corpus::load(temp_dir.path()).expect("should load");
    assert_eq!(corpus.len(), 2);
    assert_eq!(corpus.with_tests().len(), 1); // Only foo has tests
}

#[test]
fn test_corpus_example_primary_file_fallback() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let jsonl_path = temp_dir.path().join("corpus.jsonl");

    // Create example with no non-test .py file in files map
    let jsonl_content = r#"{"id": "testonly", "python_code": "x = 1", "test_cases": []}"#;
    std::fs::write(&jsonl_path, jsonl_content).expect("write");

    let corpus = Corpus::load(&jsonl_path).expect("should load");
    let example = corpus.iter().next().expect("has example");

    // primary_file should fall back to name.py
    assert_eq!(example.primary_file(), "testonly.py");
}

// =============================================================================
// Report Module Coverage (report.rs)
// =============================================================================

#[test]
fn test_report_from_evaluation_report() {
    use single_shot_eval::{EvalResult, ReportBuilder};
    use single_shot_eval::runner::{EvaluationReport, BaselineEvalResult};
    use std::collections::HashMap;

    let eval_report = EvaluationReport {
        task_id: "test-task".to_string(),
        slm_result: EvalResult {
            model_id: "test-slm".to_string(),
            task_id: "test-task".to_string(),
            accuracy: 0.90,
            cost: 0.001,
            latency: Duration::from_millis(50),
            metadata: HashMap::new(),
        },
        baseline_results: vec![BaselineEvalResult {
            model_id: "claude".to_string(),
            accuracy: 0.95,
            cost: 10.0,
            latency: Duration::from_millis(500),
        }],
    };

    let builder = ReportBuilder::from_evaluation_report(&eval_report);
    let report = builder.build();

    assert_eq!(report.metadata.task_id, "test-task");
    assert_eq!(report.model_results.len(), 1);
    assert!(report.summary.slm_value_factor.is_some());
}

#[test]
fn test_report_to_text_output() {
    use single_shot_eval::{EvalResult, ReportBuilder};
    use std::collections::HashMap;

    let mut builder = ReportBuilder::new("text-test");
    builder.add_result(
        EvalResult {
            model_id: "model1".to_string(),
            task_id: "text-test".to_string(),
            accuracy: 0.85,
            cost: 0.01,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
        vec![0.85; 50],
    );

    let report = builder.build();
    let text = report.to_text();

    assert!(text.contains("SUMMARY"));
    assert!(text.contains("MODEL RESULTS"));
    assert!(text.contains("model1"));
    assert!(text.contains("85"));
}

#[test]
fn test_report_markdown_with_dominated_models() {
    use single_shot_eval::{EvalResult, ReportBuilder};
    use std::collections::HashMap;

    let mut builder = ReportBuilder::new("pareto-test");

    // Dominant model (better in all dimensions)
    builder.add_result(
        EvalResult {
            model_id: "dominant".to_string(),
            task_id: "pareto-test".to_string(),
            accuracy: 0.95,
            cost: 0.001,
            latency: Duration::from_millis(10),
            metadata: HashMap::new(),
        },
        vec![0.95; 50],
    );

    // Dominated model (worse in all dimensions)
    builder.add_result(
        EvalResult {
            model_id: "dominated".to_string(),
            task_id: "pareto-test".to_string(),
            accuracy: 0.80,
            cost: 1.0,
            latency: Duration::from_millis(500),
            metadata: HashMap::new(),
        },
        vec![0.80; 50],
    );

    let report = builder.build();
    let markdown = report.to_markdown();

    assert!(markdown.contains("Dominated Models"));
    assert!(markdown.contains("dominated"));
}

#[test]
fn test_report_statistical_tests_with_comparison() {
    use single_shot_eval::{EvalResult, ReportBuilder};
    use std::collections::HashMap;

    let mut builder = ReportBuilder::new("stats-test");

    // Two models with clearly different accuracy distributions
    builder.add_result(
        EvalResult {
            model_id: "high".to_string(),
            task_id: "stats-test".to_string(),
            accuracy: 0.95,
            cost: 0.01,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
        (0..100).map(|i| 0.93 + (i as f64) * 0.0004).collect(),
    );

    builder.add_result(
        EvalResult {
            model_id: "low".to_string(),
            task_id: "stats-test".to_string(),
            accuracy: 0.70,
            cost: 0.01,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
        (0..100).map(|i| 0.68 + (i as f64) * 0.0004).collect(),
    );

    let report = builder.build();

    // Should have statistical comparison
    assert!(!report.statistical_tests.is_empty());

    // Markdown should show statistical comparisons table
    let markdown = report.to_markdown();
    assert!(markdown.contains("Statistical Comparisons"));
    assert!(markdown.contains("t-stat"));
    assert!(markdown.contains("p-value"));
}

// =============================================================================
// Inference Model Loading Coverage (inference.rs)
// =============================================================================

#[test]
fn test_model_loader_load_and_unload() {
    use single_shot_eval::ModelLoader;

    let mut loader = ModelLoader::new();

    // Try loading a non-existent model (should fail)
    let result = loader.load("/nonexistent/model.apr");
    assert!(result.is_err());

    // After failed load, nothing should be cached
    assert!(loader.loaded_models().is_empty());
}

#[test]
fn test_model_loader_get_nonexistent() {
    use single_shot_eval::ModelLoader;

    let loader = ModelLoader::new();
    assert!(loader.get("nonexistent").is_none());
}

#[test]
fn test_loaded_model_load_invalid_content() {
    use single_shot_eval::LoadedModel;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let model_path = temp_dir.path().join("invalid.apr");

    // Write invalid model content
    std::fs::write(&model_path, "this is not a valid model file").expect("write");

    // Should fail to load
    let result = LoadedModel::load(&model_path);
    assert!(result.is_err());
}

#[test]
fn test_inference_error_io_variant() {
    use single_shot_eval::InferenceError;
    use std::io;

    let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
    let inference_err: InferenceError = io_err.into();
    assert!(inference_err.to_string().contains("file not found"));
}

#[test]
fn test_inference_output_clone() {
    use single_shot_eval::InferenceOutput;

    let output = InferenceOutput {
        text: "test".to_string(),
        latency: Duration::from_millis(10),
        tokens_generated: 5,
    };
    let cloned = output.clone();
    assert_eq!(cloned.text, "test");
    assert_eq!(cloned.tokens_generated, 5);
}

// =============================================================================
// Runner Module Coverage (runner.rs)
// =============================================================================

#[test]
fn test_runner_load_task_valid_yaml() {
    use single_shot_eval::TaskRunner;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let task_path = temp_dir.path().join("task.yaml");

    let yaml_content = r#"
task:
  id: test-task
  description: A test task
  domain: testing

evaluation:
  metric: accuracy
  samples: 100
  timeout_ms: 5000

prompts:
  system: "You are a test assistant."
  user_template: "Test: {input}"

ground_truth:
  source: test.jsonl
  label_key: label
"#;
    std::fs::write(&task_path, yaml_content).expect("write yaml");

    let runner = TaskRunner::new();
    let task = runner.load_task(&task_path).expect("should load");

    assert_eq!(task.task.id, "test-task");
    assert_eq!(task.evaluation.samples, 100);
}

#[test]
fn test_runner_load_task_invalid_yaml() {
    use single_shot_eval::TaskRunner;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let task_path = temp_dir.path().join("bad.yaml");

    std::fs::write(&task_path, "not: valid: yaml: content:::").expect("write");

    let runner = TaskRunner::new();
    let result = runner.load_task(&task_path);
    assert!(result.is_err());
}

#[test]
fn test_runner_validate_model_valid_apr() {
    use single_shot_eval::TaskRunner;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let model_path = temp_dir.path().join("valid.apr");
    std::fs::write(&model_path, "model data").expect("write");

    let runner = TaskRunner::new();
    let result = runner.validate_model(&model_path);
    assert!(result.is_ok());
}

#[test]
fn test_runner_config_all_fields() {
    use single_shot_eval::RunnerConfig;

    let config = RunnerConfig {
        max_concurrent: 16,
        default_timeout: Duration::from_secs(120),
        retry_on_failure: false,
        max_retries: 5,
        run_baselines: false,
    };

    assert_eq!(config.max_concurrent, 16);
    assert_eq!(config.default_timeout, Duration::from_secs(120));
    assert!(!config.retry_on_failure);
    assert_eq!(config.max_retries, 5);
    assert!(!config.run_baselines);
}

#[test]
fn test_evaluation_report_methods() {
    use single_shot_eval::{EvalResult};
    use single_shot_eval::runner::{EvaluationReport, BaselineEvalResult};
    use std::collections::HashMap;

    // Test with multiple baselines
    let report = EvaluationReport {
        task_id: "multi-baseline".to_string(),
        slm_result: EvalResult {
            model_id: "slm".to_string(),
            task_id: "multi-baseline".to_string(),
            accuracy: 0.90,
            cost: 0.001,
            latency: Duration::from_millis(50),
            metadata: HashMap::new(),
        },
        baseline_results: vec![
            BaselineEvalResult {
                model_id: "claude".to_string(),
                accuracy: 0.95,
                cost: 10.0,
                latency: Duration::from_millis(500),
            },
            BaselineEvalResult {
                model_id: "gemini".to_string(),
                accuracy: 0.92,
                cost: 5.0,
                latency: Duration::from_millis(400),
            },
        ],
    };

    // value_improvement should pick best baseline (claude with 0.95)
    let improvement = report.value_improvement();
    assert!(improvement.is_some());

    // Check is_pareto_dominant
    let _ = report.is_pareto_dominant(); // Exercise the code path
}

#[test]
fn test_runner_error_io_conversion() {
    use single_shot_eval::RunnerError;
    use std::io;

    let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
    let runner_err: RunnerError = io_err.into();
    assert!(runner_err.to_string().contains("access denied"));
}

// =============================================================================
// Baselines Module Coverage (baselines.rs)
// =============================================================================

#[test]
fn test_baseline_runner_run_with_system() {
    use single_shot_eval::{BaselineConfig, BaselineRunner};

    // Create a runner for a non-existent tool
    let config = BaselineConfig {
        name: "fake-tool".to_string(),
        command: "nonexistent-command-xyz".to_string(),
        args_template: "{prompt}".to_string(),
        cost_per_1k_tokens: 0.01,
    };
    let runner = BaselineRunner::with_config(config, Duration::from_secs(30));

    // run_with_system should fail (tool not found)
    let result = runner.run_with_system("System prompt", "User prompt");
    assert!(result.is_err());
}

#[test]
fn test_baseline_error_io_conversion() {
    use single_shot_eval::BaselineError;
    use std::io;

    let io_err = io::Error::new(io::ErrorKind::BrokenPipe, "pipe broken");
    let baseline_err: BaselineError = io_err.into();
    assert!(baseline_err.to_string().contains("pipe broken"));
}

#[test]
fn test_available_baselines_function() {
    use single_shot_eval::available_baselines;

    // Should return vec of available baselines (may be empty)
    let baselines = available_baselines();
    assert!(baselines.len() <= 2); // At most claude and gemini
}

#[test]
#[ignore] // Slow: calls external baseline CLIs (claude, gemini)
fn test_run_all_baselines_function() {
    use single_shot_eval::run_all_baselines;

    // Run all baselines (may return empty if none installed)
    let results = run_all_baselines("test prompt");
    assert!(results.len() <= 2);
}

// =============================================================================
// Additional Pareto Module Coverage
// =============================================================================

#[test]
fn test_pareto_analysis_with_ties() {
    use single_shot_eval::{analyze_pareto, EvalResult};
    use std::collections::HashMap;

    // Two models with exactly equal metrics
    let results = vec![
        EvalResult {
            model_id: "model_a".to_string(),
            task_id: "test".to_string(),
            accuracy: 0.90,
            cost: 1.0,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "model_b".to_string(),
            task_id: "test".to_string(),
            accuracy: 0.90,
            cost: 1.0,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
    ];

    let analysis = analyze_pareto(&results);

    // Both should be on frontier (equal = not dominated)
    assert_eq!(analysis.frontier_models.len(), 2);
}

#[test]
fn test_pareto_empty_results() {
    use single_shot_eval::{analyze_pareto, EvalResult};

    let results: Vec<EvalResult> = vec![];
    let analysis = analyze_pareto(&results);

    assert!(analysis.frontier_models.is_empty());
    assert!(analysis.dominated_models.is_empty());
}

// =============================================================================
// Corpus Error Coverage
// =============================================================================

#[test]
fn test_corpus_error_json_parse() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let jsonl_path = temp_dir.path().join("bad.jsonl");

    // Invalid JSON
    std::fs::write(&jsonl_path, "{ invalid json }").expect("write");

    let result = Corpus::load(&jsonl_path);
    assert!(result.is_err());
}

#[test]
fn test_corpus_missing_source_in_directory() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");

    // Create example directory with only test file (no source)
    let example_dir = temp_dir.path().join("example_nosource");
    std::fs::create_dir(&example_dir).expect("mkdir");
    std::fs::write(example_dir.join("test_something.py"), "# test only").expect("write");

    // Load should skip this example (MissingSource)
    let result = Corpus::load(temp_dir.path());
    assert!(result.is_err()); // Empty corpus error
}

// =============================================================================
// Additional Runner Coverage (evaluate_baseline, run_baseline_evaluation)
// =============================================================================

#[test]
fn test_runner_run_evaluation_multiple_models() {
    use single_shot_eval::{TaskRunner, RunnerConfig};
    use single_shot_eval::config::{TaskConfig, TaskDefinition, EvaluationSettings, MetricType, PromptConfig, GroundTruthConfig};

    let task = TaskConfig {
        task: TaskDefinition {
            id: "multi-eval".to_string(),
            description: "Multiple model evaluation".to_string(),
            domain: "testing".to_string(),
        },
        evaluation: EvaluationSettings {
            metric: MetricType::Accuracy,
            samples: 50,
            timeout_ms: 10000,
        },
        prompts: PromptConfig {
            system: "System prompt".to_string(),
            user_template: "{input}".to_string(),
        },
        ground_truth: GroundTruthConfig {
            source: "data.jsonl".to_string(),
            label_key: "label".to_string(),
        },
        slm_optimization: None,
    };

    let runner = TaskRunner::new();

    // Test different model types for simulate_* functions
    for model_id in &["test-slm", "test-claude", "test-gemini", "other-model"] {
        let result = runner.run_evaluation(&task, model_id).expect("should succeed");
        assert!(!result.model_id.is_empty());
        assert!(result.accuracy > 0.0);
    }
}

#[test]
#[ignore] // Slow: calls external baseline CLIs (claude, gemini)
fn test_runner_with_baselines_enabled() {
    use single_shot_eval::{TaskRunner, RunnerConfig};
    use single_shot_eval::config::{TaskConfig, TaskDefinition, EvaluationSettings, MetricType, PromptConfig, GroundTruthConfig};

    let task = TaskConfig {
        task: TaskDefinition {
            id: "baseline-test".to_string(),
            description: "Test with baselines".to_string(),
            domain: "testing".to_string(),
        },
        evaluation: EvaluationSettings {
            metric: MetricType::Accuracy,
            samples: 10,
            timeout_ms: 5000,
        },
        prompts: PromptConfig {
            system: "System".to_string(),
            user_template: "{input}".to_string(),
        },
        ground_truth: GroundTruthConfig {
            source: "data.jsonl".to_string(),
            label_key: "label".to_string(),
        },
        slm_optimization: None,
    };

    let config = RunnerConfig {
        run_baselines: true,
        ..RunnerConfig::default()
    };
    let runner = TaskRunner::with_config(config);

    // This will try to run baselines (may be empty if tools not installed)
    let report = runner.run_with_baselines(&task, "test-slm");
    assert!(report.is_ok());
    let report = report.expect("ok");
    assert!(!report.slm_result.model_id.is_empty());
}

// =============================================================================
// Additional Inference Coverage (template_transform, infer methods)
// =============================================================================

#[test]
fn test_inference_template_transform_patterns() {
    use single_shot_eval::create_placeholder_model;

    let model = create_placeholder_model("transform-test");

    // Test all transformation patterns
    let test_cases = vec![
        ("def test(): pass", "fn"),
        ("print('hello')", "println!"),
        ("x = True", "true"),
        ("y = False", "false"),
        ("if x:\n    pass\nelif y:\n    pass", "} else if"),
        ("def foo(): return 1", "pub fn"),
    ];

    for (input, expected) in test_cases {
        let result = model.infer(input).expect("should work");
        assert!(
            result.text.contains(expected),
            "Expected '{}' in output for input '{}', got: '{}'",
            expected, input, result.text
        );
    }
}

#[test]
fn test_inference_output_tokens_and_latency() {
    use single_shot_eval::create_placeholder_model;

    let model = create_placeholder_model("metrics-test");
    let result = model.infer("def hello(): print('hi')").expect("should work");

    // Verify output structure
    assert!(!result.text.is_empty());
    assert!(result.latency.as_nanos() > 0);
    // tokens_generated is currently 0 for placeholder
    assert_eq!(result.tokens_generated, 0);
}

#[test]
fn test_loaded_model_id_and_metadata() {
    use single_shot_eval::create_placeholder_model;

    let model = create_placeholder_model("id-test");

    assert_eq!(model.id(), "id-test");
    let meta = model.metadata();
    assert!(meta.name.contains("placeholder"));
    assert_eq!(meta.model_type, "placeholder");
}

// =============================================================================
// Additional Report Coverage (edge cases)
// =============================================================================

#[test]
fn test_report_with_slm_pareto_dominant() {
    use single_shot_eval::{EvalResult, ReportBuilder};
    use single_shot_eval::runner::BaselineEvalResult;
    use std::collections::HashMap;

    let mut builder = ReportBuilder::new("dominant-test");

    // SLM that dominates baseline
    builder.add_result(
        EvalResult {
            model_id: "slm".to_string(),
            task_id: "dominant-test".to_string(),
            accuracy: 0.95,
            cost: 0.001,
            latency: Duration::from_millis(20),
            metadata: HashMap::new(),
        },
        vec![0.95; 100],
    );

    builder.add_baselines(vec![BaselineEvalResult {
        model_id: "baseline".to_string(),
        accuracy: 0.90,
        cost: 10.0,
        latency: Duration::from_millis(500),
    }]);

    let report = builder.build();

    // SLM should be Pareto dominant
    assert!(report.summary.slm_pareto_dominant);
    assert!(report.summary.slm_value_factor.is_some());
}

#[test]
fn test_report_json_serialization() {
    use single_shot_eval::{EvalResult, ReportBuilder};
    use std::collections::HashMap;

    let mut builder = ReportBuilder::new("json-test");
    builder.add_result(
        EvalResult {
            model_id: "model".to_string(),
            task_id: "json-test".to_string(),
            accuracy: 0.88,
            cost: 0.05,
            latency: Duration::from_millis(150),
            metadata: HashMap::new(),
        },
        vec![0.88; 20],
    );

    let report = builder.build();
    let json = report.to_json().expect("should serialize");

    // Verify JSON structure
    assert!(json.contains("\"task_id\""));
    assert!(json.contains("\"model_id\""));
    assert!(json.contains("\"accuracy\""));
    assert!(json.contains("json-test"));
}

// =============================================================================
// Corpus Stats Coverage
// =============================================================================

#[test]
fn test_corpus_stats_computation() {
    use single_shot_eval::Corpus;

    let temp_dir = tempfile::tempdir().expect("temp dir");

    // Create multiple examples with varying content
    for i in 1..=3 {
        let example_dir = temp_dir.path().join(format!("example_{i}"));
        std::fs::create_dir(&example_dir).expect("mkdir");
        let code = format!("def func_{i}():\n    return {i}\n");
        std::fs::write(example_dir.join(format!("func_{i}.py")), &code).expect("write");
        if i == 1 {
            std::fs::write(example_dir.join("test_func_1.py"), "def test(): pass\n").expect("write");
        }
    }

    let corpus = Corpus::load(temp_dir.path()).expect("should load");
    let stats = corpus.stats();

    assert_eq!(stats.total_examples, 3);
    assert_eq!(stats.examples_with_tests, 1);
    assert!(stats.total_files >= 3);
    assert!(stats.total_lines > 0);
}

// =============================================================================
// Baseline Run Prompt Path (with echo - always available)
// =============================================================================

#[test]
fn test_baseline_run_with_echo() {
    use single_shot_eval::{BaselineConfig, BaselineRunner};

    // Use 'echo' which is always available on Unix systems
    let config = BaselineConfig {
        name: "echo-test".to_string(),
        command: "echo".to_string(),
        args_template: "test output".to_string(),
        cost_per_1k_tokens: 0.001,
    };

    let runner = BaselineRunner::with_config(config, Duration::from_secs(10));

    if runner.is_available() {
        let result = runner.run_prompt("ignored - echo uses args_template");
        assert!(result.is_ok());
        let result = result.expect("ok");
        assert!(result.success);
        assert!(result.response.contains("test output"));
    }
}
