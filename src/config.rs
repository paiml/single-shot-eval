//! Configuration module for evaluation tasks and settings.
//!
//! Handles YAML task configuration loading with validation for .apr format requirement.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during configuration loading
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read configuration file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse YAML configuration: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("Model path must have .apr extension: {0}")]
    InvalidModelFormat(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid metric type: {0}")]
    InvalidMetric(String),
}

/// Statistical evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalConfig {
    /// Minimum samples per task
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,
    /// Bootstrap resamples for CI
    #[serde(default = "default_bootstrap_n")]
    pub bootstrap_n: usize,
    /// Confidence level
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    /// Random seed for reproducibility
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Maximum p-value for significance
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}

const fn default_min_samples() -> usize {
    1000
}
const fn default_bootstrap_n() -> usize {
    10000
}
const fn default_confidence() -> f64 {
    0.95
}
const fn default_seed() -> u64 {
    42
}
const fn default_alpha() -> f64 {
    0.05
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            min_samples: default_min_samples(),
            bootstrap_n: default_bootstrap_n(),
            confidence: default_confidence(),
            seed: default_seed(),
            alpha: default_alpha(),
        }
    }
}

/// Supported evaluation metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    /// Task accuracy (percentage correct)
    Accuracy,
    /// F1 score (harmonic mean of precision/recall)
    F1,
    /// Exact match (strict equality)
    ExactMatch,
    /// BLEU score (n-gram overlap for generation)
    Bleu,
    /// ROUGE-L score (longest common subsequence)
    RougeL,
}

impl std::str::FromStr for MetricType {
    type Err = ConfigError;

    /// Parse metric type from string
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::InvalidMetric` if the string doesn't match a known metric.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "accuracy" => Ok(Self::Accuracy),
            "f1" | "f1_score" => Ok(Self::F1),
            "exact_match" | "exactmatch" => Ok(Self::ExactMatch),
            "bleu" => Ok(Self::Bleu),
            "rouge_l" | "rougel" | "rouge-l" => Ok(Self::RougeL),
            _ => Err(ConfigError::InvalidMetric(s.to_string())),
        }
    }
}

/// Task configuration loaded from YAML
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskConfig {
    /// Task definition
    pub task: TaskDefinition,
    /// Evaluation settings
    pub evaluation: EvaluationSettings,
    /// Prompt templates
    pub prompts: PromptConfig,
    /// Ground truth configuration
    pub ground_truth: GroundTruthConfig,
    /// SLM optimization hints (optional)
    #[serde(default)]
    pub slm_optimization: Option<SlmOptimizationHints>,
}

/// Task definition section
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskDefinition {
    /// Task identifier
    pub id: String,
    /// Task description
    #[serde(default)]
    pub description: String,
    /// Task domain (software, nlp, data, etc.)
    #[serde(default)]
    pub domain: String,
}

/// Evaluation settings for a task
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvaluationSettings {
    /// Primary metric (accuracy, f1, bleu, rouge, `exact_match`)
    pub metric: MetricType,
    /// Number of samples
    #[serde(default = "default_min_samples")]
    pub samples: usize,
    /// Timeout per inference in milliseconds
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

const fn default_timeout_ms() -> u64 {
    5000
}

/// Prompt configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptConfig {
    /// System prompt
    pub system: String,
    /// User prompt template (with {placeholders})
    pub user_template: String,
}

/// Ground truth configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GroundTruthConfig {
    /// Source file path (JSONL format)
    pub source: String,
    /// Label key in the data
    #[serde(default = "default_label_key")]
    pub label_key: String,
}

fn default_label_key() -> String {
    "label".to_string()
}

/// SLM optimization hints from Depyler analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SlmOptimizationHints {
    /// Minimum attention heads required
    #[serde(default)]
    pub attention_heads_required: Option<usize>,
    /// Task-specific context length
    #[serde(default)]
    pub context_length: Option<usize>,
    /// Whether quantization is viable
    #[serde(default)]
    pub quantization_viable: Option<bool>,
}

impl TaskConfig {
    /// Load task configuration from YAML file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load task configuration from YAML string
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML cannot be parsed.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_yaml::from_str(yaml)?;
        Ok(config)
    }
}

/// Validate that a model path has .apr extension
///
/// # Errors
///
/// Returns `ConfigError::InvalidModelFormat` if the path doesn't end in `.apr`
pub fn validate_apr_format<P: AsRef<Path>>(path: P) -> Result<(), ConfigError> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()) {
        Some("apr") => Ok(()),
        _ => Err(ConfigError::InvalidModelFormat(path.display().to_string())),
    }
}

/// Configuration for available CLI baselines
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineConfig {
    /// Name of the baseline
    pub name: String,
    /// CLI command to invoke
    pub command: String,
    /// Arguments template
    pub args_template: String,
    /// Estimated cost per 1K tokens
    pub cost_per_1k_tokens: f64,
}

impl BaselineConfig {
    /// Create Claude CLI baseline configuration
    #[must_use]
    pub fn claude() -> Self {
        Self {
            name: "claude".to_string(),
            command: "claude".to_string(),
            args_template: "--print \"{prompt}\"".to_string(),
            cost_per_1k_tokens: 0.01,
        }
    }

    /// Create Gemini CLI baseline configuration
    #[must_use]
    pub fn gemini() -> Self {
        Self {
            name: "gemini".to_string(),
            command: "gemini".to_string(),
            args_template: "\"{prompt}\"".to_string(),
            cost_per_1k_tokens: 0.005,
        }
    }
}

/// Task loader for loading multiple task configurations from glob patterns
pub struct TaskLoader {
    tasks: Vec<TaskConfig>,
}

impl TaskLoader {
    /// Create a new empty task loader
    #[must_use]
    pub const fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    /// Load tasks from a glob pattern (e.g., "tasks/*.yaml")
    ///
    /// # Errors
    ///
    /// Returns an error if the glob pattern is invalid or files cannot be loaded.
    pub fn load_glob(pattern: &str) -> Result<Self, ConfigError> {
        let mut loader = Self::new();

        let paths = glob::glob(pattern)
            .map_err(|e| ConfigError::MissingField(format!("Invalid glob pattern: {e}")))?;

        for entry in paths {
            let path = entry
                .map_err(|e| ConfigError::IoError(std::io::Error::other(
                    format!("Glob error: {e}")
                )))?;

            let config = TaskConfig::load(&path)?;
            loader.tasks.push(config);
        }

        Ok(loader)
    }

    /// Load a single task from a file path
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be loaded.
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let config = TaskConfig::load(path)?;
        Ok(Self { tasks: vec![config] })
    }

    /// Get all loaded tasks
    #[must_use]
    pub fn tasks(&self) -> &[TaskConfig] {
        &self.tasks
    }

    /// Get the number of loaded tasks
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if no tasks are loaded
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Iterate over loaded tasks
    pub fn iter(&self) -> impl Iterator<Item = &TaskConfig> {
        self.tasks.iter()
    }
}

impl Default for TaskLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for TaskLoader {
    type Item = TaskConfig;
    type IntoIter = std::vec::IntoIter<TaskConfig>;

    fn into_iter(self) -> Self::IntoIter {
        self.tasks.into_iter()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // ==========================================================================
    // RED PHASE: Write failing tests first
    // ==========================================================================

    #[test]
    fn test_eval_config_default_values() {
        let config = EvalConfig::default();
        assert_eq!(config.min_samples, 1000);
        assert_eq!(config.bootstrap_n, 10000);
        assert!((config.confidence - 0.95).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
        assert!((config.alpha - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_config_serialization_roundtrip() {
        let config = EvalConfig::default();
        let yaml = serde_yaml::to_string(&config).expect("serialize");
        let parsed: EvalConfig = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_metric_type_parsing() {
        assert_eq!(
            MetricType::from_str("accuracy").unwrap(),
            MetricType::Accuracy
        );
        assert_eq!(MetricType::from_str("f1").unwrap(), MetricType::F1);
        assert_eq!(MetricType::from_str("f1_score").unwrap(), MetricType::F1);
        assert_eq!(
            MetricType::from_str("exact_match").unwrap(),
            MetricType::ExactMatch
        );
        assert_eq!(MetricType::from_str("bleu").unwrap(), MetricType::Bleu);
        assert_eq!(MetricType::from_str("rouge_l").unwrap(), MetricType::RougeL);
        assert_eq!(
            MetricType::from_str("ACCURACY").unwrap(),
            MetricType::Accuracy
        );
    }

    #[test]
    fn test_metric_type_invalid() {
        let result = MetricType::from_str("invalid_metric");
        assert!(result.is_err());
        assert!(matches!(result, Err(ConfigError::InvalidMetric(_))));
    }

    #[test]
    fn test_validate_apr_format_valid() {
        assert!(validate_apr_format("model.apr").is_ok());
        assert!(validate_apr_format("/path/to/model.apr").is_ok());
        assert!(validate_apr_format("./models/slm-v1.apr").is_ok());
    }

    #[test]
    fn test_validate_apr_format_invalid() {
        assert!(validate_apr_format("model.gguf").is_err());
        assert!(validate_apr_format("model.safetensors").is_err());
        assert!(validate_apr_format("model.pt").is_err());
        assert!(validate_apr_format("model").is_err());
    }

    #[test]
    fn test_task_config_from_yaml() {
        let yaml = r#"
task:
  id: test-task
  description: "A test task"
  domain: testing

evaluation:
  metric: accuracy
  samples: 100
  timeout_ms: 1000

prompts:
  system: "You are a test assistant."
  user_template: "Test: {input}"

ground_truth:
  source: "test-data.jsonl"
  label_key: "answer"
"#;
        let config = TaskConfig::from_yaml(yaml).expect("parse yaml");
        assert_eq!(config.task.id, "test-task");
        assert_eq!(config.evaluation.metric, MetricType::Accuracy);
        assert_eq!(config.evaluation.samples, 100);
        assert_eq!(config.prompts.system, "You are a test assistant.");
        assert_eq!(config.ground_truth.source, "test-data.jsonl");
    }

    #[test]
    fn test_task_config_with_slm_hints() {
        let yaml = r#"
task:
  id: optimized-task

evaluation:
  metric: f1
  samples: 500

prompts:
  system: "System"
  user_template: "User"

ground_truth:
  source: "data.jsonl"

slm_optimization:
  attention_heads_required: 4
  context_length: 512
  quantization_viable: true
"#;
        let config = TaskConfig::from_yaml(yaml).expect("parse yaml");
        let hints = config.slm_optimization.expect("should have hints");
        assert_eq!(hints.attention_heads_required, Some(4));
        assert_eq!(hints.context_length, Some(512));
        assert_eq!(hints.quantization_viable, Some(true));
    }

    #[test]
    fn test_task_config_minimal() {
        let yaml = r#"
task:
  id: minimal

evaluation:
  metric: exact_match

prompts:
  system: "sys"
  user_template: "user"

ground_truth:
  source: "data.jsonl"
"#;
        let config = TaskConfig::from_yaml(yaml).expect("parse yaml");
        assert_eq!(config.task.id, "minimal");
        assert_eq!(config.evaluation.samples, 1000); // default
        assert_eq!(config.evaluation.timeout_ms, 5000); // default
        assert_eq!(config.ground_truth.label_key, "label"); // default
        assert!(config.slm_optimization.is_none());
    }

    #[test]
    fn test_baseline_config_claude() {
        let baseline = BaselineConfig::claude();
        assert_eq!(baseline.name, "claude");
        assert_eq!(baseline.command, "claude");
        assert!(baseline.cost_per_1k_tokens > 0.0);
    }

    #[test]
    fn test_baseline_config_gemini() {
        let baseline = BaselineConfig::gemini();
        assert_eq!(baseline.name, "gemini");
        assert_eq!(baseline.command, "gemini");
        assert!(baseline.cost_per_1k_tokens > 0.0);
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::InvalidModelFormat("model.gguf".to_string());
        let msg = format!("{err}");
        assert!(msg.contains(".apr"));
        assert!(msg.contains("model.gguf"));
    }

    #[test]
    fn test_task_loader_new() {
        let loader = TaskLoader::new();
        assert!(loader.is_empty());
        assert_eq!(loader.len(), 0);
    }

    #[test]
    fn test_task_loader_default() {
        let loader = TaskLoader::default();
        assert!(loader.is_empty());
    }

    #[test]
    fn test_task_loader_load_glob() {
        // This should work with the existing tasks/*.yaml files
        let loader = TaskLoader::load_glob("tasks/*.yaml");
        assert!(loader.is_ok());
        let loader = loader.unwrap();
        assert!(!loader.is_empty());
    }

    #[test]
    fn test_task_loader_load_glob_no_matches() {
        let loader = TaskLoader::load_glob("nonexistent/*.yaml");
        assert!(loader.is_ok());
        let loader = loader.unwrap();
        assert!(loader.is_empty());
    }

    #[test]
    fn test_task_loader_load_file() {
        let loader = TaskLoader::load_file("tasks/code-completion.yaml");
        assert!(loader.is_ok());
        let loader = loader.unwrap();
        assert_eq!(loader.len(), 1);
        assert_eq!(loader.tasks()[0].task.id, "code-humaneval");
    }

    #[test]
    fn test_task_loader_load_file_not_found() {
        let result = TaskLoader::load_file("nonexistent.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_task_loader_iter() {
        let loader = TaskLoader::load_glob("tasks/*.yaml").unwrap();
        let count = loader.iter().count();
        assert!(count >= 1);
    }

    #[test]
    fn test_task_loader_into_iter() {
        let loader = TaskLoader::load_glob("tasks/*.yaml").unwrap();
        assert!(loader.into_iter().next().is_some());
    }

    #[test]
    fn test_task_loader_tasks_slice() {
        let loader = TaskLoader::load_glob("tasks/*.yaml").unwrap();
        let tasks = loader.tasks();
        assert!(!tasks.is_empty());
        // All tasks should have valid IDs
        for task in tasks {
            assert!(!task.task.id.is_empty());
        }
    }
}
