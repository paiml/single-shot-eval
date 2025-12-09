//! Integration tests for single-shot-eval CLI and library.
//!
//! These tests verify end-to-end functionality including:
//! - CLI commands work correctly
//! - Library APIs integrate properly
//! - Task configurations are valid
//! - Compiler verification pipeline works

// Allow less strict lints for test code
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::unwrap_used)]

use single_shot_eval::{
    analyze_pareto, compute_pareto_frontier, BatchVerifier, CompilerConfig, CompilerVerifier,
    EvalResult, MetricsCollector, ReportBuilder, RunnerConfig, TaskLoader, TaskRunner,
};
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;

// ============================================================================
// CLI Integration Tests
// ============================================================================

#[test]
fn test_cli_help_command() {
    let output = Command::new("cargo")
        .args(["run", "--quiet", "--", "--help"])
        .output()
        .expect("Failed to execute CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("single-shot-eval") || stdout.contains("SLM Pareto"),
        "Help should mention project name"
    );
    assert!(
        stdout.contains("evaluate") || stdout.contains("Evaluate"),
        "Help should list evaluate command"
    );
    assert!(
        stdout.contains("verify") || stdout.contains("Verify"),
        "Help should list verify command"
    );
}

#[test]
fn test_cli_verify_valid_rust() {
    // Create a temp file with valid Rust code
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let source_file = temp_dir.path().join("valid.rs");
    std::fs::write(&source_file, "pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
        .expect("Failed to write source file");

    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--",
            "verify",
            "--source",
            source_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("PASS") || output.status.success(),
        "Valid Rust should compile: {}",
        stdout
    );
}

#[test]
fn test_cli_verify_with_tests() {
    // Create temp files with valid Rust code and tests
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let source_file = temp_dir.path().join("source.rs");
    let test_file = temp_dir.path().join("tests.rs");

    std::fs::write(
        &source_file,
        "pub fn multiply(a: i32, b: i32) -> i32 { a * b }\n",
    )
    .expect("Failed to write source file");

    std::fs::write(
        &test_file,
        r#"
    #[test]
    fn test_multiply() {
        assert_eq!(super::multiply(3, 4), 12);
    }
"#,
    )
    .expect("Failed to write test file");

    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--",
            "verify",
            "--source",
            source_file.to_str().unwrap(),
            "--tests",
            test_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute CLI");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Either shows PASS or exits successfully
    assert!(
        stdout.contains("PASS") || output.status.success(),
        "Valid code with passing tests should succeed: {}",
        stdout
    );
}

// ============================================================================
// Task Configuration Integration Tests
// ============================================================================

#[test]
fn test_task_loader_loads_all_tasks() {
    // Should be able to load task configurations from tasks/*.yaml
    let loader = TaskLoader::load_glob("tasks/*.yaml").expect("Failed to load tasks");

    // We have at least one task config
    assert!(!loader.is_empty(), "Should have at least one task config");

    // Each loaded config should have required fields
    for config in loader.iter() {
        assert!(!config.task.id.is_empty(), "Task ID should not be empty");
        assert!(
            !config.task.domain.is_empty(),
            "Task domain should not be empty"
        );
        assert!(config.evaluation.samples > 0, "Should have positive samples");
    }
}

#[test]
fn test_task_loader_validates_yaml_structure() {
    let loader = TaskLoader::load_glob("tasks/*.yaml").expect("Failed to load tasks");

    for config in loader.iter() {
        // Verify evaluation config
        assert!(config.evaluation.timeout_ms > 0, "Timeout should be positive");

        // Verify prompts config has required fields
        let prompts = &config.prompts;
        assert!(
            !prompts.system.is_empty() || !prompts.user_template.is_empty(),
            "Should have at least system or user template"
        );
    }
}

// ============================================================================
// Compiler Verification Integration Tests
// ============================================================================

#[test]
fn test_compiler_verifier_simple_function() {
    let verifier = CompilerVerifier::new();

    let code = r#"
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;

    let result = verifier.verify(code, None).expect("Verification should succeed");
    assert!(result.compiles, "Simple function should compile");
}

#[test]
fn test_compiler_verifier_with_config() {
    // Test that custom config works without relying on external dependencies
    let config = CompilerConfig {
        edition: "2021".to_string(),
        release_mode: false,
        ..Default::default()
    };
    let verifier = CompilerVerifier::with_config(config);

    // Use only standard library features
    let code = r#"
use std::collections::HashMap;

pub struct Cache {
    data: HashMap<String, i32>,
}

impl Cache {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    pub fn insert(&mut self, key: String, value: i32) {
        self.data.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&i32> {
        self.data.get(key)
    }
}
"#;

    let result = verifier.verify(code, None).expect("Verification should succeed");
    assert!(result.compiles, "Standard library code should compile");
}

#[test]
fn test_compiler_verifier_failing_test() {
    let verifier = CompilerVerifier::new();

    let code = r#"
pub fn broken_add(a: i32, b: i32) -> i32 {
    a + b + 1  // Bug: adds extra 1
}
"#;

    let tests = r#"
    #[test]
    fn test_broken_add() {
        assert_eq!(super::broken_add(2, 3), 5);  // Will fail
    }
"#;

    let result = verifier.verify(code, Some(tests)).expect("Verification should run");
    assert!(result.compiles, "Code should compile even with bug");
    assert!(
        !result.tests_pass.unwrap_or(true),
        "Test should fail due to bug"
    );
    assert!(!result.passes(), "Overall should not pass");
}

#[test]
fn test_batch_verifier_multiple_samples() {
    let batch = BatchVerifier::new();

    let samples = vec![
        ("pub fn a() -> i32 { 1 }".to_string(), None),
        ("pub fn b() -> i32 { 2 }".to_string(), None),
        ("pub fn c() -> i32 { 3 }".to_string(), None),
        ("pub fn d( { invalid }".to_string(), None), // Invalid
    ];

    let result = batch.verify_batch(&samples);

    assert_eq!(result.total, 4, "Should have 4 total samples");
    assert_eq!(result.compiled, 3, "Should have 3 compiled samples");
    assert!(
        (result.compile_rate - 0.75).abs() < 0.01,
        "Compile rate should be ~75%"
    );
}

// ============================================================================
// Pareto Analysis Integration Tests
// ============================================================================

#[test]
fn test_pareto_analysis_with_realistic_data() {
    // Simulate realistic SLM vs frontier model comparison
    let results = vec![
        EvalResult {
            model_id: "claude-3.5-sonnet".to_string(),
            task_id: "code-transpile".to_string(),
            accuracy: 0.95,           // Frontier: high accuracy
            cost: 15.0,               // But expensive
            latency: Duration::from_millis(500),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "qwen2.5-coder-1.5b".to_string(),
            task_id: "code-transpile".to_string(),
            accuracy: 0.82,           // Good enough
            cost: 0.10,               // 150x cheaper!
            latency: Duration::from_millis(50),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "deepseek-coder-1.3b".to_string(),
            task_id: "code-transpile".to_string(),
            accuracy: 0.78,
            cost: 0.08,
            latency: Duration::from_millis(45),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "phi-2".to_string(),
            task_id: "code-transpile".to_string(),
            accuracy: 0.70,
            cost: 0.05,
            latency: Duration::from_millis(40),
            metadata: HashMap::new(),
        },
    ];

    let analysis = analyze_pareto(&results);

    // Both frontier and efficient models should be on Pareto frontier
    assert!(
        !analysis.frontier_models.is_empty(),
        "Should have frontier models"
    );
    assert!(
        analysis.frontier_models.contains(&"claude-3.5-sonnet".to_string())
            || analysis.frontier_models.contains(&"qwen2.5-coder-1.5b".to_string()),
        "Top models should be on frontier"
    );

    // Should have trade-off analysis
    assert!(
        !analysis.trade_offs.is_empty(),
        "Should have trade-off analysis"
    );
}

#[test]
fn test_compute_pareto_frontier_basic() {
    let results = vec![
        EvalResult {
            model_id: "a".to_string(),
            task_id: "t".to_string(),
            accuracy: 0.9,
            cost: 10.0,
            latency: Duration::from_millis(100),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "b".to_string(),
            task_id: "t".to_string(),
            accuracy: 0.7,
            cost: 1.0,
            latency: Duration::from_millis(50),
            metadata: HashMap::new(),
        },
    ];

    let frontier = compute_pareto_frontier(&results);

    // Both should be on frontier (trade-off between accuracy and cost)
    assert_eq!(frontier.len(), 2, "Both models should be on Pareto frontier");
}

// ============================================================================
// Report Generation Integration Tests
// ============================================================================

#[test]
fn test_report_builder_json_output() {
    let mut builder = ReportBuilder::new("integration-test");

    let result = EvalResult {
        model_id: "test-model".to_string(),
        task_id: "test-task".to_string(),
        accuracy: 0.85,
        cost: 0.50,
        latency: Duration::from_millis(100),
        metadata: HashMap::new(),
    };

    builder.add_result(result, vec![0.8, 0.85, 0.9]);
    let report = builder.build();

    let json = report.to_json().expect("Should serialize to JSON");
    assert!(json.contains("test-model"), "JSON should contain model ID");
    assert!(json.contains("0.85") || json.contains("85"), "JSON should contain accuracy");
}

#[test]
fn test_report_builder_markdown_output() {
    let mut builder = ReportBuilder::new("markdown-test");

    let result = EvalResult {
        model_id: "slm-100m".to_string(),
        task_id: "python-to-rust".to_string(),
        accuracy: 0.78,
        cost: 0.02,
        latency: Duration::from_millis(30),
        metadata: HashMap::new(),
    };

    builder.add_result(result, vec![0.75, 0.78, 0.80]);
    let report = builder.build();

    let markdown = report.to_markdown();
    assert!(
        markdown.contains("slm-100m") || markdown.contains("python-to-rust"),
        "Markdown should contain identifiers"
    );
}

// ============================================================================
// Metrics Collection Integration Tests
// ============================================================================

#[test]
fn test_metrics_collector_statistical_validity() {
    let mut collector = MetricsCollector::new();

    // Simulate Princeton methodology: 5+ runs
    for i in 0..10 {
        let accuracy = 0.80 + (i as f64 * 0.01); // 0.80 to 0.89
        let latency = Duration::from_millis(100 + i * 5);
        let cost = 0.01 + (i as f64 * 0.001);

        collector.record_accuracy(accuracy);
        collector.record_latency(latency);
        collector.record_cost(cost);
    }

    let metrics = collector.compute();

    // Verify statistical properties
    assert!(
        metrics.accuracy > 0.79 && metrics.accuracy < 0.90,
        "Mean accuracy should be in expected range"
    );
    assert!(
        metrics.accuracy_ci.0 < metrics.accuracy && metrics.accuracy < metrics.accuracy_ci.1,
        "Accuracy should be within CI"
    );
    assert!(
        metrics.latency_p50 < metrics.latency_p99,
        "P50 should be less than P99"
    );
}

// ============================================================================
// Runner Integration Tests
// ============================================================================

#[test]
fn test_task_runner_config() {
    let config = RunnerConfig {
        max_concurrent: 8,
        default_timeout: Duration::from_secs(60),
        retry_on_failure: true,
        max_retries: 5,
        run_baselines: false,
    };

    let runner = TaskRunner::with_config(config);

    // Load a task and verify runner can process it
    let loader = TaskLoader::load_glob("tasks/*.yaml").expect("Failed to load tasks");
    let tasks: Vec<_> = loader.iter().collect();
    if let Some(task_config) = tasks.first() {
        // Verify we can run evaluation (simulated, since we don't have real models)
        let result = runner.run_evaluation(task_config, "test-model");
        assert!(result.is_ok(), "Should be able to run simulated evaluation");
    }
}

#[test]
fn test_task_runner_validates_apr_format() {
    let runner = TaskRunner::new();

    // Create a non-.apr file
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let bad_model = temp_dir.path().join("model.bin");
    std::fs::write(&bad_model, "not an apr file").expect("Failed to write file");

    // Should reject non-.apr files
    let result = runner.validate_model(&bad_model);
    assert!(result.is_err(), "Should reject non-.apr files");
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_full_evaluation_workflow() {
    // This test verifies the complete workflow:
    // 1. Load task config
    // 2. Run evaluation
    // 3. Compute Pareto frontier
    // 4. Generate report

    // Step 1: Load task
    let loader = TaskLoader::load_glob("tasks/*.yaml").expect("Failed to load tasks");
    let task_config = loader.iter().next().expect("Need at least one task");

    // Step 2: Run evaluation
    let runner = TaskRunner::new();
    let result = runner
        .run_evaluation(task_config, "test-slm")
        .expect("Evaluation should succeed");

    // Verify result structure
    assert!(!result.model_id.is_empty());
    assert!(!result.task_id.is_empty());
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    assert!(result.cost >= 0.0);

    // Step 3: Pareto analysis
    let results = vec![result.clone()];
    let analysis = analyze_pareto(&results);
    assert!(
        !analysis.frontier_models.is_empty(),
        "Single model should be on frontier"
    );

    // Step 4: Generate report
    let mut builder = ReportBuilder::new("e2e-test");
    builder.add_result(result, vec![0.8]);
    let report = builder.build();

    assert!(
        report.to_json().is_ok(),
        "Should generate valid JSON report"
    );
    assert!(
        !report.to_markdown().is_empty(),
        "Should generate markdown report"
    );
}
