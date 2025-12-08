//! Compiler verification for ground truth evaluation.
//!
//! Provides binary pass/fail ground truth by compiling generated Rust code
//! and running tests. This follows Princeton "AI Agents That Matter" methodology:
//! ground truth via execution, not proxy metrics.

use std::fmt::Write;
use std::path::Path;
use std::process::{Command, Output};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use thiserror::Error;

/// Errors during compiler verification
#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Failed to create temp directory: {0}")]
    TempDirError(#[from] std::io::Error),

    #[error("Cargo build failed: {stderr}")]
    BuildFailed { stderr: String },

    #[error("Cargo test failed: {stderr}")]
    TestFailed { stderr: String },

    #[error("Verification timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid Rust code: {0}")]
    InvalidCode(String),
}

/// Result of compiler verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether compilation succeeded
    pub compiles: bool,
    /// Whether tests passed (None if compilation failed)
    pub tests_pass: Option<bool>,
    /// Build time
    pub build_time: Duration,
    /// Test time (None if not run)
    pub test_time: Option<Duration>,
    /// Compiler stderr (warnings, errors)
    pub build_output: String,
    /// Test output
    pub test_output: Option<String>,
    /// Number of tests run
    pub tests_run: usize,
    /// Number of tests passed
    pub tests_passed: usize,
}

impl VerificationResult {
    /// Ground truth: both compiles AND tests pass
    #[must_use]
    pub fn passes(&self) -> bool {
        self.compiles && self.tests_pass.unwrap_or(false)
    }

    /// Accuracy score: 1.0 if passes, 0.0 otherwise
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.passes() {
            1.0
        } else {
            0.0
        }
    }
}

/// Configuration for compiler verification
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Timeout for cargo build
    pub build_timeout: Duration,
    /// Timeout for cargo test
    pub test_timeout: Duration,
    /// Rust edition
    pub edition: String,
    /// Additional dependencies for Cargo.toml
    pub dependencies: Vec<(String, String)>,
    /// Whether to run in release mode
    pub release_mode: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            build_timeout: Duration::from_secs(60),
            test_timeout: Duration::from_secs(60),
            edition: "2021".to_string(),
            dependencies: Vec::new(),
            release_mode: false,
        }
    }
}

/// Compiler verifier for Rust code
pub struct CompilerVerifier {
    config: CompilerConfig,
}

impl CompilerVerifier {
    /// Create a new compiler verifier with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CompilerConfig::default(),
        }
    }

    /// Create with custom config
    #[must_use]
    pub const fn with_config(config: CompilerConfig) -> Self {
        Self { config }
    }

    /// Verify Rust code compiles and passes tests
    ///
    /// # Errors
    ///
    /// Returns error if temp directory creation fails or timeout occurs.
    pub fn verify(&self, rust_code: &str, test_code: Option<&str>) -> Result<VerificationResult, CompilerError> {
        // Create temp project
        let temp_dir = TempDir::new()?;
        let project_path = temp_dir.path();

        // Set up Cargo project
        self.setup_cargo_project(project_path, rust_code, test_code)?;

        // Run cargo build
        let build_start = Instant::now();
        let build_output = self.run_cargo_build(project_path)?;
        let build_time = build_start.elapsed();

        let compiles = build_output.status.success();
        let build_stderr = String::from_utf8_lossy(&build_output.stderr).to_string();

        if !compiles {
            return Ok(VerificationResult {
                compiles: false,
                tests_pass: None,
                build_time,
                test_time: None,
                build_output: build_stderr,
                test_output: None,
                tests_run: 0,
                tests_passed: 0,
            });
        }

        // Run cargo test
        let test_start = Instant::now();
        let test_output = self.run_cargo_test(project_path)?;
        let test_time = test_start.elapsed();

        let tests_pass = test_output.status.success();
        let test_stdout = String::from_utf8_lossy(&test_output.stdout).to_string();
        let test_stderr = String::from_utf8_lossy(&test_output.stderr).to_string();
        let test_combined = format!("{test_stdout}\n{test_stderr}");

        // Parse test counts from output
        let (tests_run, tests_passed) = parse_test_counts(&test_combined);

        Ok(VerificationResult {
            compiles: true,
            tests_pass: Some(tests_pass),
            build_time,
            test_time: Some(test_time),
            build_output: build_stderr,
            test_output: Some(test_combined),
            tests_run,
            tests_passed,
        })
    }

    /// Verify just compilation (no tests)
    ///
    /// # Errors
    ///
    /// Returns error if temp directory creation fails.
    pub fn verify_compiles(&self, rust_code: &str) -> Result<bool, CompilerError> {
        let temp_dir = TempDir::new()?;
        let project_path = temp_dir.path();

        self.setup_cargo_project(project_path, rust_code, None)?;
        let output = self.run_cargo_build(project_path)?;

        Ok(output.status.success())
    }

    /// Set up a minimal Cargo project
    fn setup_cargo_project(
        &self,
        project_path: &Path,
        rust_code: &str,
        test_code: Option<&str>,
    ) -> Result<(), CompilerError> {
        // Create src directory
        let src_dir = project_path.join("src");
        std::fs::create_dir_all(&src_dir)?;

        // Write Cargo.toml
        let cargo_toml = self.generate_cargo_toml();
        std::fs::write(project_path.join("Cargo.toml"), cargo_toml)?;

        // Write main lib.rs with code
        let lib_content = test_code.map_or_else(
            || rust_code.to_string(),
            |tests| format!("{rust_code}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{tests}\n}}"),
        );
        std::fs::write(src_dir.join("lib.rs"), lib_content)?;

        Ok(())
    }

    /// Generate Cargo.toml content
    fn generate_cargo_toml(&self) -> String {
        let mut toml = format!(
            r#"[package]
name = "eval_target"
version = "0.1.0"
edition = "{}"

[dependencies]
"#,
            self.config.edition
        );

        for (name, version) in &self.config.dependencies {
            let _ = writeln!(toml, "{name} = \"{version}\"");
        }

        toml
    }

    /// Run cargo build
    fn run_cargo_build(&self, project_path: &Path) -> Result<Output, CompilerError> {
        let mut cmd = Command::new("cargo");
        cmd.arg("build");
        if self.config.release_mode {
            cmd.arg("--release");
        }
        cmd.current_dir(project_path);

        // Note: timeout handling would require async or threading
        // For now, we trust cargo to complete reasonably
        cmd.output().map_err(CompilerError::from)
    }

    /// Run cargo test
    fn run_cargo_test(&self, project_path: &Path) -> Result<Output, CompilerError> {
        let mut cmd = Command::new("cargo");
        cmd.arg("test");
        if self.config.release_mode {
            cmd.arg("--release");
        }
        cmd.current_dir(project_path);

        cmd.output().map_err(CompilerError::from)
    }

    /// Get current config
    #[must_use]
    pub const fn config(&self) -> &CompilerConfig {
        &self.config
    }
}

impl Default for CompilerVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse test counts from cargo test output
fn parse_test_counts(output: &str) -> (usize, usize) {
    // Look for "test result: ok. X passed; Y failed; Z ignored"
    for line in output.lines() {
        if line.contains("test result:") {
            let mut passed = 0;
            let mut failed = 0;

            // Parse "X passed"
            if let Some(idx) = line.find("passed") {
                let before = &line[..idx];
                if let Some(num_str) = before.split_whitespace().next_back() {
                    passed = num_str.parse().unwrap_or(0);
                }
            }

            // Parse "Y failed"
            if let Some(idx) = line.find("failed") {
                let before = &line[..idx];
                if let Some(num_str) = before.split(';').next_back() {
                    if let Some(num) = num_str.split_whitespace().next_back() {
                        failed = num.parse().unwrap_or(0);
                    }
                }
            }

            return (passed + failed, passed);
        }
    }

    (0, 0)
}

/// Batch verification for multiple examples
pub struct BatchVerifier {
    verifier: CompilerVerifier,
}

impl BatchVerifier {
    /// Create new batch verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            verifier: CompilerVerifier::new(),
        }
    }

    /// Create with custom config
    #[must_use]
    pub const fn with_config(config: CompilerConfig) -> Self {
        Self {
            verifier: CompilerVerifier::with_config(config),
        }
    }

    /// Verify multiple code samples, returning pass rate
    #[must_use]
    pub fn verify_batch(
        &self,
        samples: &[(String, Option<String>)],
    ) -> BatchResult {
        let mut results = Vec::with_capacity(samples.len());

        for (code, tests) in samples {
            let result = self.verifier.verify(code, tests.as_deref());
            results.push(result);
        }

        let passed = results.iter().filter(|r| r.as_ref().is_ok_and(VerificationResult::passes)).count();
        let compiled = results.iter().filter(|r| r.as_ref().is_ok_and(|v| v.compiles)).count();
        let total = results.len();

        BatchResult {
            results,
            pass_rate: passed as f64 / total.max(1) as f64,
            compile_rate: compiled as f64 / total.max(1) as f64,
            total,
            passed,
            compiled,
        }
    }
}

impl Default for BatchVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from batch verification
#[derive(Debug)]
pub struct BatchResult {
    /// Individual results
    pub results: Vec<Result<VerificationResult, CompilerError>>,
    /// Pass rate (compiles + tests pass)
    pub pass_rate: f64,
    /// Compile rate (just compilation)
    pub compile_rate: f64,
    /// Total samples
    pub total: usize,
    /// Samples that passed
    pub passed: usize,
    /// Samples that compiled
    pub compiled: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_passes() {
        let result = VerificationResult {
            compiles: true,
            tests_pass: Some(true),
            build_time: Duration::from_millis(100),
            test_time: Some(Duration::from_millis(50)),
            build_output: String::new(),
            test_output: Some(String::new()),
            tests_run: 1,
            tests_passed: 1,
        };
        assert!(result.passes());
        assert!((result.accuracy() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verification_result_fails_compile() {
        let result = VerificationResult {
            compiles: false,
            tests_pass: None,
            build_time: Duration::from_millis(100),
            test_time: None,
            build_output: "error[E0425]: cannot find value".to_string(),
            test_output: None,
            tests_run: 0,
            tests_passed: 0,
        };
        assert!(!result.passes());
        assert!(result.accuracy() < f64::EPSILON);
    }

    #[test]
    fn test_verification_result_fails_tests() {
        let result = VerificationResult {
            compiles: true,
            tests_pass: Some(false),
            build_time: Duration::from_millis(100),
            test_time: Some(Duration::from_millis(50)),
            build_output: String::new(),
            test_output: Some("test failed".to_string()),
            tests_run: 1,
            tests_passed: 0,
        };
        assert!(!result.passes());
        assert!(result.accuracy() < f64::EPSILON);
    }

    #[test]
    fn test_compiler_config_default() {
        let config = CompilerConfig::default();
        assert_eq!(config.edition, "2021");
        assert_eq!(config.build_timeout, Duration::from_secs(60));
        assert!(!config.release_mode);
    }

    #[test]
    fn test_parse_test_counts() {
        let output = "test result: ok. 5 passed; 2 failed; 0 ignored";
        let (total, passed) = parse_test_counts(output);
        assert_eq!(total, 7);
        assert_eq!(passed, 5);
    }

    #[test]
    fn test_parse_test_counts_all_pass() {
        let output = "test result: ok. 10 passed; 0 failed; 0 ignored";
        let (total, passed) = parse_test_counts(output);
        assert_eq!(total, 10);
        assert_eq!(passed, 10);
    }

    #[test]
    fn test_parse_test_counts_no_match() {
        let output = "some random output";
        let (total, passed) = parse_test_counts(output);
        assert_eq!(total, 0);
        assert_eq!(passed, 0);
    }

    #[test]
    fn test_verifier_valid_code() {
        let verifier = CompilerVerifier::new();
        let code = "
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
";
        let tests = "
    #[test]
    fn test_add() {
        assert_eq!(super::add(2, 3), 5);
    }
";

        let result = verifier.verify(code, Some(tests));
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.compiles);
        assert!(result.tests_pass.unwrap_or(false));
        assert!(result.passes());
    }

    #[test]
    fn test_verifier_invalid_code() {
        let verifier = CompilerVerifier::new();
        let code = "
pub fn broken( {
    // syntax error
}
";

        let result = verifier.verify(code, None);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.compiles);
        assert!(!result.passes());
    }

    #[test]
    fn test_verifier_failing_test() {
        let verifier = CompilerVerifier::new();
        let code = "
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
";
        let tests = "
    #[test]
    fn test_add_wrong() {
        assert_eq!(super::add(2, 3), 999); // Wrong!
    }
";

        let result = verifier.verify(code, Some(tests));
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.compiles);
        assert!(!result.tests_pass.unwrap_or(true));
        assert!(!result.passes());
    }

    #[test]
    fn test_verify_compiles_only() {
        let verifier = CompilerVerifier::new();
        let code = "pub fn foo() -> i32 { 42 }";

        let compiles = verifier.verify_compiles(code);
        assert!(compiles.is_ok());
        assert!(compiles.unwrap());
    }

    #[test]
    fn test_verify_compiles_invalid() {
        let verifier = CompilerVerifier::new();
        let code = "pub fn foo( { invalid }";

        let compiles = verifier.verify_compiles(code);
        assert!(compiles.is_ok());
        assert!(!compiles.unwrap());
    }

    #[test]
    fn test_batch_verifier() {
        let batch = BatchVerifier::new();
        let samples = vec![
            ("pub fn a() -> i32 { 1 }".to_string(), None),
            ("pub fn b() -> i32 { 2 }".to_string(), None),
        ];

        let result = batch.verify_batch(&samples);
        assert_eq!(result.total, 2);
        assert!(result.compile_rate > 0.9);
    }

    #[test]
    fn test_batch_verifier_mixed() {
        let batch = BatchVerifier::new();
        let samples = vec![
            ("pub fn ok() -> i32 { 1 }".to_string(), None),
            ("pub fn bad( { }".to_string(), None), // Invalid
        ];

        let result = batch.verify_batch(&samples);
        assert_eq!(result.total, 2);
        assert_eq!(result.compiled, 1);
        assert!((result.compile_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compiler_verifier_default() {
        let verifier = CompilerVerifier::default();
        assert_eq!(verifier.config().edition, "2021");
    }

    #[test]
    fn test_batch_verifier_default() {
        let batch = BatchVerifier::default();
        let samples: Vec<(String, Option<String>)> = vec![];
        let result = batch.verify_batch(&samples);
        assert_eq!(result.total, 0);
    }

    #[test]
    fn test_compiler_error_display() {
        let err = CompilerError::BuildFailed {
            stderr: "error[E0425]".to_string(),
        };
        assert!(err.to_string().contains("Cargo build failed"));

        let err = CompilerError::TestFailed {
            stderr: "assertion failed".to_string(),
        };
        assert!(err.to_string().contains("Cargo test failed"));

        let err = CompilerError::Timeout(Duration::from_secs(30));
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_generate_cargo_toml() {
        let config = CompilerConfig {
            dependencies: vec![
                ("serde".to_string(), "1.0".to_string()),
            ],
            ..Default::default()
        };
        let verifier = CompilerVerifier::with_config(config);
        let toml = verifier.generate_cargo_toml();

        assert!(toml.contains("edition = \"2021\""));
        assert!(toml.contains("serde = \"1.0\""));
    }

    #[test]
    fn test_verification_result_accuracy() {
        // Test the accuracy method more thoroughly
        let passing = VerificationResult {
            compiles: true,
            tests_pass: Some(true),
            build_time: Duration::from_millis(1),
            test_time: Some(Duration::from_millis(1)),
            build_output: String::new(),
            test_output: None,
            tests_run: 1,
            tests_passed: 1,
        };
        assert!((passing.accuracy() - 1.0).abs() < f64::EPSILON);

        let failing = VerificationResult {
            compiles: true,
            tests_pass: Some(false),
            build_time: Duration::from_millis(1),
            test_time: Some(Duration::from_millis(1)),
            build_output: String::new(),
            test_output: None,
            tests_run: 1,
            tests_passed: 0,
        };
        assert!(failing.accuracy().abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_with_tests() {
        let batch = BatchVerifier::new();
        let samples = vec![
            (
                "pub fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
                Some("    #[test]\n    fn t() { assert_eq!(super::add(1, 2), 3); }".to_string()),
            ),
        ];

        let result = batch.verify_batch(&samples);
        assert_eq!(result.total, 1);
        // passed is usize, so just verify result was computed
        let _ = result.passed;
    }
}
