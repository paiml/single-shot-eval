//! Task execution engine for running evaluations.
//!
//! Orchestrates evaluation of .apr models against task definitions,
//! collecting metrics and optionally running baseline comparisons.

use crate::baselines::{BaselineError, BaselineRunner};
use crate::config::{validate_apr_format, TaskConfig};
use crate::metrics::MetricsCollector;
use crate::pareto::EvalResult;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during task execution
#[derive(Error, Debug)]
pub enum RunnerError {
    #[error("Model file not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model format (must be .apr): {0}")]
    InvalidFormat(String),

    #[error("Model loading failed: {0}")]
    LoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Ground truth file not found: {0}")]
    GroundTruthNotFound(String),

    #[error("Evaluation timeout after {0:?}")]
    Timeout(Duration),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Runner configuration
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Maximum concurrent evaluations
    pub max_concurrent: usize,
    /// Default timeout per inference
    pub default_timeout: Duration,
    /// Whether to retry on transient failures
    pub retry_on_failure: bool,
    /// Maximum retries
    pub max_retries: usize,
    /// Run baselines for comparison
    pub run_baselines: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            default_timeout: Duration::from_secs(30),
            retry_on_failure: true,
            max_retries: 3,
            run_baselines: true,
        }
    }
}

/// Single evaluation sample with ground truth
#[derive(Debug, Clone)]
pub struct EvalSample {
    /// Sample identifier
    pub id: String,
    /// Input prompt/data
    pub input: String,
    /// Expected output (ground truth)
    pub expected: String,
}

/// Result of a single inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Sample ID
    pub sample_id: String,
    /// Model output
    pub output: String,
    /// Latency
    pub latency: Duration,
    /// Whether the output matches expected (for accuracy)
    pub is_correct: bool,
}

/// Task runner for executing evaluations
pub struct TaskRunner {
    /// Configuration for the runner
    config: RunnerConfig,
}

impl TaskRunner {
    /// Create a new task runner with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RunnerConfig::default(),
        }
    }

    /// Create a new task runner with custom configuration
    #[must_use]
    pub const fn with_config(config: RunnerConfig) -> Self {
        Self { config }
    }

    /// Load task configuration from YAML file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load_task(&self, path: &Path) -> Result<TaskConfig> {
        let content = std::fs::read_to_string(path)?;
        let config: TaskConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Validate that a model path is a valid .apr file
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't have .apr extension or doesn't exist.
    pub fn validate_model(&self, model_path: &Path) -> Result<(), RunnerError> {
        if !model_path.exists() {
            return Err(RunnerError::ModelNotFound(model_path.display().to_string()));
        }

        validate_apr_format(model_path)
            .map_err(|_| RunnerError::InvalidFormat(model_path.display().to_string()))
    }

    /// Run evaluation for a single task and model
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails.
    pub fn run_evaluation(&self, task: &TaskConfig, model_id: &str) -> Result<EvalResult> {
        let mut collector = MetricsCollector::new();

        // In a real implementation, we would:
        // 1. Load the .apr model using aprender
        // 2. Load ground truth samples from the configured source
        // 3. Run inference on each sample
        // 4. Compute metrics based on the metric type
        //
        // For now, we simulate this with placeholder logic
        // that demonstrates the structure

        let num_samples = task.evaluation.samples;
        let timeout = Duration::from_millis(task.evaluation.timeout_ms);

        // Simulate evaluation
        for i in 0..num_samples {
            let start = Instant::now();

            // Simulated inference (in reality: aprender inference)
            let simulated_accuracy = simulate_accuracy(model_id, i);
            let simulated_latency = simulate_latency(model_id);

            if start.elapsed() > timeout {
                // Would break or handle timeout
                break;
            }

            collector.record_accuracy(simulated_accuracy);
            collector.record_latency(simulated_latency);
            collector.record_cost(estimate_cost(model_id, simulated_latency));
        }

        let metrics = collector.compute();

        Ok(EvalResult {
            model_id: model_id.to_string(),
            task_id: task.task.id.clone(),
            accuracy: metrics.accuracy,
            cost: metrics.cost_per_inference * 1_000_000.0, // per 1M tokens
            latency: metrics.latency_p99,
            metadata: HashMap::new(),
        })
    }

    /// Run evaluation with baseline comparison
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails.
    pub fn run_with_baselines(
        &self,
        task: &TaskConfig,
        model_id: &str,
    ) -> Result<EvaluationReport> {
        // Run SLM evaluation
        let slm_result = self.run_evaluation(task, model_id)?;

        // Run baseline evaluations if enabled
        let baseline_results = if self.config.run_baselines {
            self.run_baseline_evaluation(task)
        } else {
            Vec::new()
        };

        Ok(EvaluationReport {
            task_id: task.task.id.clone(),
            slm_result,
            baseline_results,
        })
    }

    /// Run baseline evaluation using CLI tools
    #[allow(clippy::unused_self)]
    fn run_baseline_evaluation(&self, task: &TaskConfig) -> Vec<BaselineEvalResult> {
        let mut results = Vec::new();

        // Try Claude baseline
        let claude = BaselineRunner::claude();
        if claude.is_available() {
            if let Ok(baseline_result) = Self::evaluate_baseline(&claude, task) {
                results.push(baseline_result);
            }
        }

        // Try Gemini baseline
        let gemini = BaselineRunner::gemini();
        if gemini.is_available() {
            if let Ok(baseline_result) = Self::evaluate_baseline(&gemini, task) {
                results.push(baseline_result);
            }
        }

        results
    }

    /// Evaluate a single baseline
    fn evaluate_baseline(
        runner: &BaselineRunner,
        task: &TaskConfig,
    ) -> Result<BaselineEvalResult, BaselineError> {
        let mut collector = MetricsCollector::new();

        // Run a subset of samples through the baseline
        let num_samples = task.evaluation.samples.min(10); // Limit baseline samples

        for _ in 0..num_samples {
            // Construct prompt from task template
            let prompt = format!(
                "{}\n\n{}",
                task.prompts.system,
                task.prompts.user_template.replace("{input}", "test input")
            );

            let result = runner.run_prompt(&prompt)?;

            collector.record_latency(result.latency);
            collector.record_cost(result.estimated_cost);

            // Accuracy would be computed by comparing to ground truth
            // For now, simulate
            collector.record_accuracy(if result.success { 0.95 } else { 0.0 });
        }

        let metrics = collector.compute();

        Ok(BaselineEvalResult {
            model_id: runner.name().to_string(),
            accuracy: metrics.accuracy,
            cost: metrics.cost_per_inference * 1_000_000.0,
            latency: metrics.latency_p99,
        })
    }

    /// Get current configuration
    #[must_use]
    pub const fn config(&self) -> &RunnerConfig {
        &self.config
    }
}

impl Default for TaskRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from baseline evaluation
#[derive(Debug, Clone)]
pub struct BaselineEvalResult {
    /// Model/tool identifier
    pub model_id: String,
    /// Accuracy score
    pub accuracy: f64,
    /// Cost per 1M tokens
    pub cost: f64,
    /// p99 latency
    pub latency: Duration,
}

/// Complete evaluation report with SLM and baselines
#[derive(Debug)]
pub struct EvaluationReport {
    /// Task identifier
    pub task_id: String,
    /// SLM evaluation result
    pub slm_result: EvalResult,
    /// Baseline results for comparison
    pub baseline_results: Vec<BaselineEvalResult>,
}

impl EvaluationReport {
    /// Calculate value improvement over best baseline
    #[must_use]
    pub fn value_improvement(&self) -> Option<f64> {
        if self.baseline_results.is_empty() {
            return None;
        }

        // Find best baseline by accuracy
        let best_baseline = self.baseline_results.iter().max_by(|a, b| {
            a.accuracy
                .partial_cmp(&b.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        // Calculate value score
        let accuracy_ratio = self.slm_result.accuracy / best_baseline.accuracy.max(0.001);
        let cost_ratio = best_baseline.cost / self.slm_result.cost.max(0.00001);
        let latency_ratio =
            best_baseline.latency.as_secs_f64() / self.slm_result.latency.as_secs_f64().max(0.001);

        Some(accuracy_ratio * cost_ratio * latency_ratio)
    }

    /// Check if SLM is Pareto-dominant over any baseline
    #[must_use]
    pub fn is_pareto_dominant(&self) -> bool {
        self.baseline_results.iter().any(|baseline| {
            self.slm_result.accuracy >= baseline.accuracy
                && self.slm_result.cost <= baseline.cost
                && self.slm_result.latency <= baseline.latency
                && (self.slm_result.accuracy > baseline.accuracy
                    || self.slm_result.cost < baseline.cost
                    || self.slm_result.latency < baseline.latency)
        })
    }
}

// ============================================================================
// Simulation helpers (replace with actual aprender integration)
// ============================================================================

/// Simulate accuracy based on model type
#[allow(clippy::suboptimal_flops)]
fn simulate_accuracy(model_id: &str, _sample_idx: usize) -> f64 {
    match model_id {
        id if id.contains("slm") => 0.90 + rand::random::<f64>() * 0.05,
        id if id.contains("claude") => 0.94 + rand::random::<f64>() * 0.03,
        id if id.contains("gemini") => 0.92 + rand::random::<f64>() * 0.04,
        _ => 0.85 + rand::random::<f64>() * 0.10,
    }
}

/// Simulate latency based on model type
fn simulate_latency(model_id: &str) -> Duration {
    let base_ms = match model_id {
        id if id.contains("slm") => 10,
        id if id.contains("claude") => 500,
        id if id.contains("gemini") => 400,
        _ => 100,
    };
    let jitter = (rand::random::<f64>() * 50.0) as u64;
    Duration::from_millis(base_ms + jitter)
}

/// Estimate cost based on model and latency
#[allow(clippy::unreadable_literal)]
fn estimate_cost(model_id: &str, _latency: Duration) -> f64 {
    match model_id {
        id if id.contains("slm") => 0.000001,
        id if id.contains("claude") => 0.01,
        id if id.contains("gemini") => 0.005,
        _ => 0.001,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        EvaluationSettings, GroundTruthConfig, MetricType, PromptConfig, TaskDefinition,
    };

    fn create_test_task() -> TaskConfig {
        TaskConfig {
            task: TaskDefinition {
                id: "test-task".to_string(),
                description: "Test task".to_string(),
                domain: "testing".to_string(),
            },
            evaluation: EvaluationSettings {
                metric: MetricType::Accuracy,
                samples: 100,
                timeout_ms: 5000,
            },
            prompts: PromptConfig {
                system: "You are a test assistant.".to_string(),
                user_template: "Test: {input}".to_string(),
            },
            ground_truth: GroundTruthConfig {
                source: "test.jsonl".to_string(),
                label_key: "label".to_string(),
            },
            slm_optimization: None,
        }
    }

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.max_concurrent, 4);
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert!(config.retry_on_failure);
        assert_eq!(config.max_retries, 3);
        assert!(config.run_baselines);
    }

    #[test]
    fn test_task_runner_new() {
        let runner = TaskRunner::new();
        assert_eq!(runner.config().max_concurrent, 4);
    }

    #[test]
    fn test_task_runner_with_config() {
        let config = RunnerConfig {
            max_concurrent: 8,
            default_timeout: Duration::from_secs(60),
            retry_on_failure: false,
            max_retries: 1,
            run_baselines: false,
        };
        let runner = TaskRunner::with_config(config);
        assert_eq!(runner.config().max_concurrent, 8);
        assert!(!runner.config().retry_on_failure);
        assert!(!runner.config().run_baselines);
    }

    #[test]
    fn test_run_evaluation() {
        let runner = TaskRunner::new();
        let task = create_test_task();

        let result = runner.run_evaluation(&task, "test-slm");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.model_id, "test-slm");
        assert_eq!(result.task_id, "test-task");
        assert!(result.accuracy > 0.0);
    }

    #[test]
    fn test_run_with_baselines_no_baselines() {
        let config = RunnerConfig {
            run_baselines: false,
            ..RunnerConfig::default()
        };
        let runner = TaskRunner::with_config(config);
        let task = create_test_task();

        let report = runner.run_with_baselines(&task, "test-slm");
        assert!(report.is_ok());

        let report = report.unwrap();
        assert!(report.baseline_results.is_empty());
    }

    #[test]
    fn test_evaluation_report_value_improvement_no_baselines() {
        let report = EvaluationReport {
            task_id: "test".to_string(),
            slm_result: EvalResult {
                model_id: "slm".to_string(),
                task_id: "test".to_string(),
                accuracy: 0.92,
                cost: 0.001,
                latency: Duration::from_millis(50),
                metadata: HashMap::new(),
            },
            baseline_results: Vec::new(),
        };

        assert!(report.value_improvement().is_none());
    }

    #[test]
    fn test_evaluation_report_value_improvement_with_baselines() {
        let report = EvaluationReport {
            task_id: "test".to_string(),
            slm_result: EvalResult {
                model_id: "slm".to_string(),
                task_id: "test".to_string(),
                accuracy: 0.92,
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

        let improvement = report.value_improvement();
        assert!(improvement.is_some());
        assert!(improvement.unwrap() > 1.0); // SLM should show value improvement
    }

    #[test]
    fn test_is_pareto_dominant() {
        // SLM dominates baseline (better in at least one, not worse in any)
        let report = EvaluationReport {
            task_id: "test".to_string(),
            slm_result: EvalResult {
                model_id: "slm".to_string(),
                task_id: "test".to_string(),
                accuracy: 0.95,
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

        assert!(report.is_pareto_dominant());
    }

    #[test]
    fn test_not_pareto_dominant() {
        // SLM doesn't dominate (worse accuracy)
        let report = EvaluationReport {
            task_id: "test".to_string(),
            slm_result: EvalResult {
                model_id: "slm".to_string(),
                task_id: "test".to_string(),
                accuracy: 0.80,
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

        assert!(!report.is_pareto_dominant());
    }

    #[test]
    fn test_runner_error_display() {
        let err = RunnerError::ModelNotFound("model.apr".to_string());
        assert!(err.to_string().contains("model.apr"));

        let err = RunnerError::InvalidFormat("model.gguf".to_string());
        assert!(err.to_string().contains(".apr"));
    }

    #[test]
    fn test_simulate_accuracy() {
        let slm_acc = simulate_accuracy("test-slm", 0);
        assert!((0.90..=0.95).contains(&slm_acc));

        let claude_acc = simulate_accuracy("test-claude", 0);
        assert!((0.94..=0.97).contains(&claude_acc));
    }

    #[test]
    fn test_simulate_latency() {
        let slm_lat = simulate_latency("test-slm");
        assert!(slm_lat <= Duration::from_millis(100));

        let claude_lat = simulate_latency("test-claude");
        assert!(claude_lat >= Duration::from_millis(500));
    }

    #[test]
    fn test_estimate_cost() {
        let slm_cost = estimate_cost("test-slm", Duration::from_millis(50));
        assert!(slm_cost < 0.00001);

        let claude_cost = estimate_cost("test-claude", Duration::from_millis(500));
        assert!(claude_cost > 0.001);
    }

    #[test]
    fn test_task_runner_default() {
        let runner = TaskRunner::default();
        assert_eq!(runner.config().max_concurrent, 4);
        assert!(runner.config().run_baselines);
    }

    #[test]
    fn test_validate_model_not_found() {
        let runner = TaskRunner::new();
        let result = runner.validate_model(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RunnerError::ModelNotFound(_)));
    }

    #[test]
    fn test_validate_model_invalid_format() {
        let runner = TaskRunner::new();
        // Create a temp file with wrong extension
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_model.gguf");
        std::fs::write(&temp_file, "dummy").unwrap();

        let result = runner.validate_model(&temp_file);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RunnerError::InvalidFormat(_)));

        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_load_task_not_found() {
        let runner = TaskRunner::new();
        let result = runner.load_task(Path::new("/nonexistent/task.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_simulate_accuracy_gemini() {
        let gemini_acc = simulate_accuracy("test-gemini", 0);
        assert!((0.92..=0.96).contains(&gemini_acc));
    }

    #[test]
    fn test_simulate_accuracy_unknown() {
        let unknown_acc = simulate_accuracy("unknown-model", 0);
        assert!((0.85..=0.95).contains(&unknown_acc));
    }

    #[test]
    fn test_simulate_latency_gemini() {
        let gemini_lat = simulate_latency("test-gemini");
        assert!(gemini_lat >= Duration::from_millis(400));
    }

    #[test]
    fn test_simulate_latency_unknown() {
        let unknown_lat = simulate_latency("unknown-model");
        assert!(unknown_lat >= Duration::from_millis(100));
        assert!(unknown_lat < Duration::from_millis(200));
    }

    #[test]
    fn test_estimate_cost_gemini() {
        let gemini_cost = estimate_cost("test-gemini", Duration::from_millis(400));
        assert!((0.004..=0.006).contains(&gemini_cost));
    }

    #[test]
    fn test_estimate_cost_unknown() {
        let unknown_cost = estimate_cost("unknown-model", Duration::from_millis(100));
        assert!((0.0005..=0.002).contains(&unknown_cost));
    }

    #[test]
    fn test_run_with_baselines_enabled() {
        let runner = TaskRunner::new();
        let task = create_test_task();

        // baselines are enabled by default, but CLI tools may not be available
        let report = runner.run_with_baselines(&task, "test-slm");
        assert!(report.is_ok());
        // baseline_results may be empty if tools aren't installed
    }

    #[test]
    fn test_eval_sample_fields() {
        let sample = EvalSample {
            id: "sample-1".to_string(),
            input: "test input".to_string(),
            expected: "expected output".to_string(),
        };
        assert_eq!(sample.id, "sample-1");
        assert_eq!(sample.input, "test input");
        assert_eq!(sample.expected, "expected output");
    }

    #[test]
    fn test_inference_result_fields() {
        let result = InferenceResult {
            sample_id: "sample-1".to_string(),
            output: "model output".to_string(),
            latency: Duration::from_millis(100),
            is_correct: true,
        };
        assert_eq!(result.sample_id, "sample-1");
        assert!(result.is_correct);
    }

    #[test]
    fn test_baseline_eval_result_fields() {
        let result = BaselineEvalResult {
            model_id: "claude".to_string(),
            accuracy: 0.95,
            cost: 10.0,
            latency: Duration::from_millis(500),
        };
        assert_eq!(result.model_id, "claude");
        assert!((0.94..=0.96).contains(&result.accuracy));
    }

    #[test]
    fn test_runner_error_variants() {
        let errors = vec![
            RunnerError::ModelNotFound("model.apr".into()),
            RunnerError::InvalidFormat("bad.gguf".into()),
            RunnerError::LoadError("failed to load".into()),
            RunnerError::InferenceError("inference failed".into()),
            RunnerError::GroundTruthNotFound("truth.jsonl".into()),
            RunnerError::Timeout(Duration::from_secs(30)),
        ];

        for err in errors {
            // Just verify Display works
            let _ = err.to_string();
        }
    }

    #[test]
    fn test_is_pareto_dominant_empty_baselines() {
        let report = EvaluationReport {
            task_id: "test".to_string(),
            slm_result: EvalResult {
                model_id: "slm".to_string(),
                task_id: "test".to_string(),
                accuracy: 0.95,
                cost: 0.001,
                latency: Duration::from_millis(50),
                metadata: HashMap::new(),
            },
            baseline_results: Vec::new(),
        };

        // No baselines to dominate
        assert!(!report.is_pareto_dominant());
    }

    #[test]
    fn test_is_pareto_dominant_equal() {
        // SLM equal to baseline - not dominant (need strictly better in at least one)
        let report = EvaluationReport {
            task_id: "test".to_string(),
            slm_result: EvalResult {
                model_id: "slm".to_string(),
                task_id: "test".to_string(),
                accuracy: 0.95,
                cost: 10.0,
                latency: Duration::from_millis(500),
                metadata: HashMap::new(),
            },
            baseline_results: vec![BaselineEvalResult {
                model_id: "claude".to_string(),
                accuracy: 0.95,
                cost: 10.0,
                latency: Duration::from_millis(500),
            }],
        };

        assert!(!report.is_pareto_dominant());
    }
}
