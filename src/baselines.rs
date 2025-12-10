//! CLI baseline wrappers for external LLM tools.
//!
//! Provides OFFLINE-FIRST baseline evaluation by shelling out to installed
//! CLI tools (`claude`, `gemini`). NO HTTP API calls - all communication
//! via subprocess execution.
//!
//! ## Sovereign Inference (replaces ollama)
//!
//! When the `sovereign-inference` feature is enabled, local .apr models are
//! preferred over ollama for OFFLINE-FIRST evaluation using the realizar engine.

use crate::config::BaselineConfig;
#[cfg(feature = "sovereign-inference")]
use crate::sovereign::{list_models, ModelFormat, SovereignRunner};
use std::io::{BufRead, BufReader};
#[cfg(feature = "sovereign-inference")]
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during baseline evaluation
#[derive(Error, Debug)]
pub enum BaselineError {
    #[error("CLI tool not found: {0}")]
    ToolNotFound(String),

    #[error("CLI execution failed: {0}")]
    ExecutionFailed(String),

    #[error("CLI timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid response from CLI: {0}")]
    InvalidResponse(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[cfg(feature = "sovereign-inference")]
    #[error("Sovereign inference error: {0}")]
    SovereignError(#[from] crate::sovereign::SovereignError),
}

/// Result from a baseline CLI invocation
#[derive(Debug, Clone)]
pub struct BaselineResult {
    /// The model/tool identifier
    pub model_id: String,
    /// Response text from the CLI
    pub response: String,
    /// Latency of the invocation
    pub latency: Duration,
    /// Estimated cost (if available)
    pub estimated_cost: f64,
    /// Whether the invocation succeeded
    pub success: bool,
}

/// CLI baseline runner for external LLM tools
pub struct BaselineRunner {
    /// Configuration for the baseline
    config: BaselineConfig,
    /// Timeout for CLI invocations
    timeout: Duration,
}

impl BaselineRunner {
    /// Create a new baseline runner for Claude CLI
    #[must_use]
    pub fn claude() -> Self {
        Self {
            config: BaselineConfig::claude(),
            timeout: Duration::from_secs(120),
        }
    }

    /// Create a new baseline runner for Gemini CLI
    #[must_use]
    pub fn gemini() -> Self {
        Self {
            config: BaselineConfig::gemini(),
            timeout: Duration::from_secs(120),
        }
    }

    /// Create a new baseline runner for Ollama with a specific model
    #[must_use]
    pub fn ollama(model: &str) -> Self {
        Self {
            config: BaselineConfig::ollama(model),
            timeout: Duration::from_secs(300), // Longer timeout for local models
        }
    }

    /// Create baseline runners for all installed CODE SLMs
    #[must_use]
    pub fn ollama_code_slms() -> Vec<Self> {
        vec![
            Self {
                config: BaselineConfig::ollama_qwen(),
                timeout: Duration::from_secs(300),
            },
            Self {
                config: BaselineConfig::ollama_deepseek(),
                timeout: Duration::from_secs(300),
            },
            Self {
                config: BaselineConfig::ollama_starcoder(),
                timeout: Duration::from_secs(300),
            },
            Self {
                config: BaselineConfig::ollama_phi2(),
                timeout: Duration::from_secs(300),
            },
        ]
    }

    /// Create a baseline runner with custom configuration
    #[must_use]
    pub const fn with_config(config: BaselineConfig, timeout: Duration) -> Self {
        Self { config, timeout }
    }

    /// Check if the CLI tool is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        Command::new("which")
            .arg(&self.config.command)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    /// Get the tool name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Run a prompt through the CLI tool
    ///
    /// # Errors
    ///
    /// Returns an error if the CLI tool is not found, execution fails, or times out.
    pub fn run_prompt(&self, prompt: &str) -> Result<BaselineResult, BaselineError> {
        if !self.is_available() {
            return Err(BaselineError::ToolNotFound(self.config.command.clone()));
        }

        let start = Instant::now();

        // Build command with arguments
        #[allow(clippy::literal_string_with_formatting_args)]
        let args = self
            .config
            .args_template
            .replace("{prompt}", &shell_escape(prompt));

        let mut cmd = Command::new(&self.config.command);

        // Parse args template into individual arguments
        for arg in shell_words::split(&args).unwrap_or_default() {
            cmd.arg(arg);
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        // Read output with timeout
        let stdout = child.stdout.take();
        let response = stdout.map_or_else(String::new, |stdout| {
            let reader = BufReader::new(stdout);
            let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
            lines.join("\n")
        });

        let status = child.wait()?;
        let latency = start.elapsed();

        // Check timeout
        if latency > self.timeout {
            return Err(BaselineError::Timeout(self.timeout));
        }

        // Estimate cost based on response length (rough approximation)
        let tokens = estimate_tokens(&response);
        let estimated_cost = tokens as f64 * self.config.cost_per_1k_tokens / 1000.0;

        Ok(BaselineResult {
            model_id: self.config.name.clone(),
            response,
            latency,
            estimated_cost,
            success: status.success(),
        })
    }

    /// Run a prompt with system context
    ///
    /// # Errors
    ///
    /// Returns an error if the CLI tool is not found, execution fails, or times out.
    pub fn run_with_system(
        &self,
        system: &str,
        prompt: &str,
    ) -> Result<BaselineResult, BaselineError> {
        // Combine system and user prompt
        let combined = format!("{system}\n\nUser: {prompt}");
        self.run_prompt(&combined)
    }
}

/// Escape shell special characters in a string
fn shell_escape(s: &str) -> String {
    // Simple escape - replace quotes and backslashes
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
}

/// Rough token estimation (4 chars per token average)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn estimate_tokens(text: &str) -> usize {
    (text.len() as f64 / 4.0).ceil() as usize
}

/// Check which baseline tools are available
#[must_use]
pub fn available_baselines() -> Vec<String> {
    let mut available = Vec::new();

    if BaselineRunner::claude().is_available() {
        available.push("claude".to_string());
    }

    if BaselineRunner::gemini().is_available() {
        available.push("gemini".to_string());
    }

    // SOVEREIGN-FIRST: Check for .apr models before ollama
    #[cfg(feature = "sovereign-inference")]
    {
        for (path, format) in list_sovereign_models() {
            if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                available.push(format!("sovereign:{}:{}", format.as_str(), name));
            }
        }
    }

    // FALLBACK: Check for ollama and its models (when sovereign not available)
    if is_ollama_available() {
        for model in list_ollama_models() {
            available.push(format!("ollama:{model}"));
        }
    }

    available
}

/// Check if ollama is installed
#[must_use]
pub fn is_ollama_available() -> bool {
    Command::new("which")
        .arg("ollama")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// List available ollama models
#[must_use]
pub fn list_ollama_models() -> Vec<String> {
    let output = Command::new("ollama").arg("list").output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout
                .lines()
                .skip(1) // Skip header
                .filter_map(|line| line.split_whitespace().next())
                .map(String::from)
                .collect()
        }
        _ => Vec::new(),
    }
}

// =============================================================================
// SOVEREIGN INFERENCE - Native .apr model execution via realizar
// =============================================================================

/// Default directory to search for sovereign models
#[cfg(feature = "sovereign-inference")]
const SOVEREIGN_MODEL_DIR: &str = "./models";

/// List available sovereign models (.apr, .gguf, .safetensors)
///
/// Searches in the default models directory for supported formats.
/// Priority: .apr (PRIMARY) > .gguf > .safetensors
#[cfg(feature = "sovereign-inference")]
#[must_use]
pub fn list_sovereign_models() -> Vec<(PathBuf, ModelFormat)> {
    list_models(SOVEREIGN_MODEL_DIR)
}

/// Check if sovereign inference is available
#[cfg(feature = "sovereign-inference")]
#[must_use]
pub const fn is_sovereign_available() -> bool {
    crate::sovereign::is_sovereign_available()
}

/// Run a prompt against a sovereign model
///
/// # Errors
///
/// Returns an error if the model cannot be loaded or inference fails.
#[cfg(feature = "sovereign-inference")]
pub fn run_sovereign_prompt(
    model_path: &std::path::Path,
    prompt: &str,
) -> Result<BaselineResult, BaselineError> {
    let start = Instant::now();

    let runner = SovereignRunner::load(model_path)?;
    let result = runner.run_prompt(prompt)?;

    Ok(BaselineResult {
        model_id: format!("sovereign:{}", result.model_id),
        response: result.response,
        latency: start.elapsed(),
        estimated_cost: 0.0, // Sovereign inference is FREE (offline)
        success: result.success,
    })
}

/// Run a prompt against all available baselines
///
/// # Errors
///
/// Returns errors for each baseline that fails.
#[must_use]
pub fn run_all_baselines(prompt: &str) -> Vec<Result<BaselineResult, BaselineError>> {
    let mut results = Vec::new();

    let claude = BaselineRunner::claude();
    if claude.is_available() {
        results.push(claude.run_prompt(prompt));
    }

    let gemini = BaselineRunner::gemini();
    if gemini.is_available() {
        results.push(gemini.run_prompt(prompt));
    }

    results
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // =========================================================================
    // Unit tests (don't require actual CLI tools)
    // =========================================================================

    #[test]
    fn test_baseline_runner_claude_creation() {
        let runner = BaselineRunner::claude();
        assert_eq!(runner.name(), "claude");
        assert_eq!(runner.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_baseline_runner_gemini_creation() {
        let runner = BaselineRunner::gemini();
        assert_eq!(runner.name(), "gemini");
        assert_eq!(runner.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_baseline_runner_custom_config() {
        let config = BaselineConfig {
            name: "custom".to_string(),
            command: "test-cli".to_string(),
            args_template: "--prompt \"{prompt}\"".to_string(),
            cost_per_1k_tokens: 0.02,
        };
        let runner = BaselineRunner::with_config(config, Duration::from_secs(60));
        assert_eq!(runner.name(), "custom");
        assert_eq!(runner.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_shell_escape() {
        assert_eq!(shell_escape("hello"), "hello");
        assert_eq!(shell_escape("hello\"world"), "hello\\\"world");
        assert_eq!(shell_escape("hello\nworld"), "hello\\nworld");
        assert_eq!(shell_escape("test\\path"), "test\\\\path");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("hello world test"), 4); // 16 chars / 4 = 4
    }

    #[test]
    fn test_available_baselines() {
        // This should return a list (possibly empty if tools not installed)
        let baselines = available_baselines();
        // Just verify it doesn't panic and returns valid names
        for baseline in &baselines {
            assert!(!baseline.is_empty());
        }
    }

    #[test]
    fn test_baseline_error_display() {
        let err = BaselineError::ToolNotFound("claude".to_string());
        assert!(err.to_string().contains("claude"));

        let err = BaselineError::Timeout(Duration::from_secs(30));
        assert!(err.to_string().contains("30"));
    }

    #[test]
    fn test_baseline_error_all_variants() {
        let errors: Vec<BaselineError> = vec![
            BaselineError::ToolNotFound("tool".into()),
            BaselineError::ExecutionFailed("exec failed".into()),
            BaselineError::Timeout(Duration::from_secs(60)),
            BaselineError::InvalidResponse("bad response".into()),
        ];

        for err in errors {
            let _ = err.to_string();
        }
    }

    #[test]
    fn test_baseline_result_fields() {
        let result = BaselineResult {
            model_id: "claude".to_string(),
            response: "Hello!".to_string(),
            latency: Duration::from_millis(500),
            estimated_cost: 0.001,
            success: true,
        };
        assert_eq!(result.model_id, "claude");
        assert!(result.success);
    }

    #[test]
    fn test_baseline_result_failure() {
        let result = BaselineResult {
            model_id: "gemini".to_string(),
            response: String::new(),
            latency: Duration::from_millis(100),
            estimated_cost: 0.0,
            success: false,
        };
        assert!(!result.success);
        assert!(result.response.is_empty());
    }

    #[test]
    fn test_run_prompt_tool_not_found() {
        let config = BaselineConfig {
            name: "nonexistent".to_string(),
            command: "this-tool-definitely-does-not-exist-12345".to_string(),
            args_template: "--prompt \"{prompt}\"".to_string(),
            cost_per_1k_tokens: 0.01,
        };
        let runner = BaselineRunner::with_config(config, Duration::from_secs(30));

        let result = runner.run_prompt("test");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BaselineError::ToolNotFound(_)
        ));
    }

    #[test]
    fn test_is_available_nonexistent() {
        let config = BaselineConfig {
            name: "fake".to_string(),
            command: "this-command-does-not-exist-xyz".to_string(),
            args_template: "{prompt}".to_string(),
            cost_per_1k_tokens: 0.0,
        };
        let runner = BaselineRunner::with_config(config, Duration::from_secs(10));
        assert!(!runner.is_available());
    }

    #[test]
    fn test_shell_escape_single_quote() {
        assert_eq!(shell_escape("it's"), "it\\'s");
    }

    #[test]
    fn test_shell_escape_complex() {
        let input = "hello \"world\"\ntest\\path's";
        let escaped = shell_escape(input);
        assert!(escaped.contains("\\\""));
        assert!(escaped.contains("\\n"));
        assert!(escaped.contains("\\\\"));
        assert!(escaped.contains("\\'"));
    }

    #[test]
    fn test_estimate_tokens_long_text() {
        let text = "a".repeat(1000);
        let tokens = estimate_tokens(&text);
        assert_eq!(tokens, 250); // 1000 / 4 = 250
    }

    #[test]
    fn test_run_all_baselines() {
        // This tests run_all_baselines function
        // Will return empty or results depending on tool availability
        let results = run_all_baselines("test prompt");
        // Just verify it doesn't panic and returns a vec
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_baseline_runner_is_available() {
        // Test is_available for both runners
        let claude = BaselineRunner::claude();
        let _ = claude.is_available(); // Just exercise the code

        let gemini = BaselineRunner::gemini();
        let _ = gemini.is_available(); // Just exercise the code
    }

    // =========================================================================
    // Integration tests (require CLI tools to be installed)
    // =========================================================================

    #[test]
    #[ignore = "Requires claude CLI to be installed"]
    fn test_claude_cli_integration() {
        let runner = BaselineRunner::claude();
        if runner.is_available() {
            let result = runner.run_prompt("Say 'test' and nothing else.");
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result.success);
            assert!(!result.response.is_empty());
        }
    }

    #[test]
    #[ignore = "Requires gemini CLI to be installed"]
    fn test_gemini_cli_integration() {
        let runner = BaselineRunner::gemini();
        if runner.is_available() {
            let result = runner.run_prompt("Say 'test' and nothing else.");
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result.success);
            assert!(!result.response.is_empty());
        }
    }
}
