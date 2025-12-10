//! Model validation with logit consistency checking
//!
//! Toyota Way principles:
//! - **Jidoka**: Logit consistency check halts on divergence
//! - No recursive dependency on external judge models
//!
//! ## Example
//!
//! ```rust,ignore
//! use single_shot_eval::validate::{LogitConsistencyChecker, validate_apr};
//!
//! let checker = LogitConsistencyChecker::default();
//! let result = checker.verify(&original_logits, &converted_logits)?;
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Validation errors
#[derive(Error, Debug)]
pub enum ValidationError {
    /// APR magic bytes invalid
    #[error("JIDOKA HALT: Invalid APR magic bytes in {path} - expected 0x41505221")]
    InvalidMagicBytes { path: PathBuf },

    /// Metadata schema invalid
    #[error("Invalid metadata schema: {0}")]
    InvalidMetadata(String),

    /// Logit consistency failed
    #[error("JIDOKA HALT: Logit consistency check failed - agreement {agreement:.2}% < required {required:.2}%")]
    LogitConsistencyFailed { agreement: f64, required: f64 },

    /// Model load failed
    #[error("Model load failed: {0}")]
    LoadFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// APR file magic bytes: "APR!" = 0x41505221
pub const APR_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x21];

/// Validate APR file magic bytes
///
/// # Errors
///
/// Returns `InvalidMagicBytes` if magic doesn't match - HALTS pipeline
pub fn validate_apr_magic(path: &Path) -> Result<(), ValidationError> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;

    if magic != APR_MAGIC {
        tracing::error!(
            path = %path.display(),
            expected = "0x41505221",
            actual = format!("0x{:02X}{:02X}{:02X}{:02X}", magic[0], magic[1], magic[2], magic[3]),
            "JIDOKA HALT: Invalid APR magic bytes"
        );
        return Err(ValidationError::InvalidMagicBytes {
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

/// Token with logit value
#[derive(Debug, Clone)]
pub struct TokenLogit {
    /// Token index in vocabulary
    pub token_id: usize,
    /// Logit value (pre-softmax)
    pub logit: f32,
}

/// Divergent sample details
#[derive(Debug, Clone)]
pub struct DivergentSample {
    /// Prompt that caused divergence
    pub prompt: String,
    /// Original model's top-k tokens
    pub original_top_k: Vec<TokenLogit>,
    /// Converted model's top-k tokens
    pub converted_top_k: Vec<TokenLogit>,
    /// Agreement score for this sample
    pub agreement: f64,
}

/// Logit consistency check result
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    /// Overall agreement rate (0.0 - 1.0)
    pub agreement_rate: f64,
    /// Total prompts checked
    pub total_prompts: usize,
    /// Number of divergent prompts
    pub divergent_count: usize,
    /// Details of divergent samples
    pub divergent_samples: Vec<DivergentSample>,
    /// Whether check passed
    pub passed: bool,
}

/// Logit consistency checker
///
/// Compares raw logit outputs between original and converted models.
/// This is deterministic and requires no external "judge" model.
#[derive(Debug, Clone)]
pub struct LogitConsistencyChecker {
    /// Top-k tokens to compare
    pub top_k: usize,
    /// Tolerance for logit value differences
    pub logit_tolerance: f32,
    /// Minimum agreement rate to pass
    pub min_agreement: f64,
}

impl Default for LogitConsistencyChecker {
    fn default() -> Self {
        Self {
            top_k: 10,
            logit_tolerance: 0.1,
            min_agreement: 0.9, // 90% agreement required
        }
    }
}

impl LogitConsistencyChecker {
    /// Create checker with custom parameters
    #[must_use]
    pub const fn new(top_k: usize, logit_tolerance: f32, min_agreement: f64) -> Self {
        Self {
            top_k,
            logit_tolerance,
            min_agreement,
        }
    }

    /// Verify logit consistency between original and converted outputs
    ///
    /// # Errors
    ///
    /// Returns `LogitConsistencyFailed` if agreement < `min_agreement` - HALTS pipeline
    pub fn verify(
        &self,
        original_outputs: &[Vec<f32>],
        converted_outputs: &[Vec<f32>],
        prompts: &[String],
    ) -> Result<ConsistencyResult, ValidationError> {
        if original_outputs.len() != converted_outputs.len() {
            return Err(ValidationError::InvalidMetadata(
                "Output count mismatch".to_string(),
            ));
        }

        let mut agreements = 0usize;
        let mut divergent_samples = Vec::new();

        for (i, (orig, conv)) in original_outputs.iter().zip(converted_outputs).enumerate() {
            let orig_top_k = self.get_top_k(orig);
            let conv_top_k = self.get_top_k(conv);

            let agreement = self.compute_agreement(&orig_top_k, &conv_top_k);

            if agreement >= self.min_agreement {
                agreements += 1;
            } else {
                let prompt = prompts.get(i).cloned().unwrap_or_default();
                divergent_samples.push(DivergentSample {
                    prompt,
                    original_top_k: orig_top_k,
                    converted_top_k: conv_top_k,
                    agreement,
                });
            }
        }

        let agreement_rate = if original_outputs.is_empty() {
            1.0
        } else {
            agreements as f64 / original_outputs.len() as f64
        };

        let passed = agreement_rate >= self.min_agreement;

        if !passed {
            tracing::error!(
                agreement_rate = %format!("{:.2}%", agreement_rate * 100.0),
                required = %format!("{:.2}%", self.min_agreement * 100.0),
                divergent_count = divergent_samples.len(),
                "JIDOKA HALT: Logit consistency check failed"
            );

            return Err(ValidationError::LogitConsistencyFailed {
                agreement: agreement_rate * 100.0,
                required: self.min_agreement * 100.0,
            });
        }

        Ok(ConsistencyResult {
            agreement_rate,
            total_prompts: original_outputs.len(),
            divergent_count: divergent_samples.len(),
            divergent_samples,
            passed,
        })
    }

    /// Extract top-k tokens with their logit values
    fn get_top_k(&self, logits: &[f32]) -> Vec<TokenLogit> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .into_iter()
            .take(self.top_k)
            .map(|(token_id, logit)| TokenLogit { token_id, logit })
            .collect()
    }

    /// Compute agreement between two top-k lists
    fn compute_agreement(&self, orig: &[TokenLogit], conv: &[TokenLogit]) -> f64 {
        let orig_tokens: HashSet<usize> = orig.iter().map(|t| t.token_id).collect();
        let conv_tokens: HashSet<usize> = conv.iter().map(|t| t.token_id).collect();

        // Token set overlap
        let token_overlap = orig_tokens.intersection(&conv_tokens).count();

        // Logit value similarity for matching tokens
        let mut logit_agreements = 0;
        for orig_t in orig {
            if let Some(conv_t) = conv.iter().find(|t| t.token_id == orig_t.token_id) {
                if (orig_t.logit - conv_t.logit).abs() <= self.logit_tolerance {
                    logit_agreements += 1;
                }
            }
        }

        // Combined score: 50% token overlap + 50% logit value agreement
        let effective_k = self.top_k.min(orig.len()).min(conv.len()).max(1);
        let token_score = token_overlap as f64 / effective_k as f64;
        let logit_score = f64::from(logit_agreements) / effective_k as f64;

        0.5f64.mul_add(token_score, 0.5 * logit_score)
    }
}

/// Validation config
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check magic bytes
    pub check_magic: bool,
    /// Check metadata schema
    pub check_metadata: bool,
    /// Run logit consistency check
    pub check_logit_consistency: bool,
    /// Logit checker settings
    pub logit_checker: LogitConsistencyChecker,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_magic: true,
            check_metadata: true,
            check_logit_consistency: true,
            logit_checker: LogitConsistencyChecker::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ==========================================================================
    // APR Magic Bytes Tests
    // ==========================================================================

    #[test]
    fn test_validate_apr_magic_valid() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let file_path = temp_dir.path().join("test.apr");

        // Write valid APR magic + some data
        let mut data = APR_MAGIC.to_vec();
        data.extend_from_slice(b"rest of file");
        std::fs::write(&file_path, &data).expect("write file");

        let result = validate_apr_magic(&file_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_apr_magic_invalid_halts() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let file_path = temp_dir.path().join("test.apr");

        // Write invalid magic
        std::fs::write(&file_path, b"BAAD").expect("write file");

        let result = validate_apr_magic(&file_path);
        assert!(result.is_err());

        if let Err(ValidationError::InvalidMagicBytes { path }) = result {
            assert_eq!(path, file_path);
        } else {
            panic!("Expected InvalidMagicBytes error");
        }
    }

    // ==========================================================================
    // LogitConsistencyChecker Tests
    // ==========================================================================

    #[test]
    fn test_logit_checker_default() {
        let checker = LogitConsistencyChecker::default();

        assert_eq!(checker.top_k, 10);
        assert!((checker.logit_tolerance - 0.1).abs() < 1e-6);
        assert!((checker.min_agreement - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_logit_checker_identical_outputs() {
        let checker = LogitConsistencyChecker::default();

        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let original = vec![logits.clone()];
        let converted = vec![logits];
        let prompts = vec!["test prompt".to_string()];

        let result = checker.verify(&original, &converted, &prompts);
        assert!(result.is_ok());

        let report = result.expect("should pass");
        assert!(report.passed);
        assert!((report.agreement_rate - 1.0).abs() < 1e-6);
        assert_eq!(report.divergent_count, 0);
    }

    #[test]
    fn test_logit_checker_small_difference_passes() {
        let checker = LogitConsistencyChecker {
            logit_tolerance: 0.1,
            min_agreement: 0.9,
            ..Default::default()
        };

        let original = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        // Small differences within tolerance
        let converted = vec![vec![
            1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05, 9.05, 10.05,
        ]];
        let prompts = vec!["test".to_string()];

        let result = checker.verify(&original, &converted, &prompts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_logit_checker_large_difference_halts() {
        let checker = LogitConsistencyChecker {
            top_k: 5,
            logit_tolerance: 0.1,
            min_agreement: 0.9,
        };

        let original = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        // Completely different values - different top-k order
        let converted = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0]];
        let prompts = vec!["test".to_string()];

        let result = checker.verify(&original, &converted, &prompts);
        assert!(result.is_err());

        if let Err(ValidationError::LogitConsistencyFailed {
            agreement,
            required,
        }) = result
        {
            assert!(agreement < required);
        } else {
            panic!("Expected LogitConsistencyFailed error");
        }
    }

    #[test]
    fn test_logit_checker_empty_inputs() {
        let checker = LogitConsistencyChecker::default();

        let original: Vec<Vec<f32>> = vec![];
        let converted: Vec<Vec<f32>> = vec![];
        let prompts: Vec<String> = vec![];

        let result = checker.verify(&original, &converted, &prompts);
        assert!(result.is_ok());

        let report = result.expect("should pass");
        assert!(report.passed);
        assert_eq!(report.total_prompts, 0);
    }

    #[test]
    fn test_logit_checker_mismatched_count() {
        let checker = LogitConsistencyChecker::default();

        let original = vec![vec![1.0, 2.0]];
        let converted = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let prompts = vec!["test".to_string()];

        let result = checker.verify(&original, &converted, &prompts);
        assert!(result.is_err());

        if let Err(ValidationError::InvalidMetadata(msg)) = result {
            assert!(msg.contains("mismatch"));
        } else {
            panic!("Expected InvalidMetadata error");
        }
    }

    // ==========================================================================
    // Top-K Extraction Tests
    // ==========================================================================

    #[test]
    fn test_get_top_k() {
        let checker = LogitConsistencyChecker {
            top_k: 3,
            ..Default::default()
        };

        let logits = vec![0.1, 0.5, 0.2, 0.9, 0.3];
        let top_k = checker.get_top_k(&logits);

        assert_eq!(top_k.len(), 3);
        // Should be sorted by logit descending
        assert_eq!(top_k[0].token_id, 3); // 0.9
        assert_eq!(top_k[1].token_id, 1); // 0.5
        assert_eq!(top_k[2].token_id, 4); // 0.3
    }

    // ==========================================================================
    // Agreement Computation Tests
    // ==========================================================================

    #[test]
    fn test_compute_agreement_identical() {
        let checker = LogitConsistencyChecker::default();

        let tokens = vec![
            TokenLogit {
                token_id: 1,
                logit: 1.0,
            },
            TokenLogit {
                token_id: 2,
                logit: 0.9,
            },
            TokenLogit {
                token_id: 3,
                logit: 0.8,
            },
        ];

        let agreement = checker.compute_agreement(&tokens, &tokens);
        assert!((agreement - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_agreement_no_overlap() {
        let checker = LogitConsistencyChecker {
            top_k: 3,
            ..Default::default()
        };

        let orig = vec![
            TokenLogit {
                token_id: 1,
                logit: 1.0,
            },
            TokenLogit {
                token_id: 2,
                logit: 0.9,
            },
        ];
        let conv = vec![
            TokenLogit {
                token_id: 3,
                logit: 1.0,
            },
            TokenLogit {
                token_id: 4,
                logit: 0.9,
            },
        ];

        let agreement = checker.compute_agreement(&orig, &conv);
        assert!(agreement < 0.5); // No token overlap
    }

    // ==========================================================================
    // ValidationConfig Tests
    // ==========================================================================

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();

        assert!(config.check_magic);
        assert!(config.check_metadata);
        assert!(config.check_logit_consistency);
    }

    // ==========================================================================
    // Error Display Tests
    // ==========================================================================

    #[test]
    fn test_error_display() {
        let err = ValidationError::InvalidMagicBytes {
            path: PathBuf::from("test.apr"),
        };
        assert!(err.to_string().contains("JIDOKA HALT"));
        assert!(err.to_string().contains("0x41505221"));

        let err = ValidationError::LogitConsistencyFailed {
            agreement: 75.0,
            required: 90.0,
        };
        assert!(err.to_string().contains("JIDOKA HALT"));
        assert!(err.to_string().contains("75.00%"));
        assert!(err.to_string().contains("90.00%"));
    }
}
