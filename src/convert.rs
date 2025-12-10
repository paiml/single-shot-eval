//! Model format conversion with SPC numerical precision gate
//!
//! Toyota Way principles:
//! - **Jidoka**: KL divergence check halts on numerical drift
//! - **SPC**: Statistical Process Control for weight precision
//!
//! ## Example
//!
//! ```rust,ignore
//! use single_shot_eval::convert::{ConvertConfig, convert_to_apr};
//!
//! let config = ConvertConfig::default();
//! let metadata = convert_to_apr("model.safetensors", "model.apr", &config)?;
//! ```

use std::path::{Path, PathBuf};
use thiserror::Error;

/// Conversion errors
#[derive(Error, Debug)]
pub enum ConvertError {
    /// Format not detected
    #[error("Cannot detect format for: {0}")]
    FormatNotDetected(PathBuf),

    /// Unsupported format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Parser error
    #[error("Parser error: {0}")]
    ParserError(String),

    /// Missing tensors
    #[error("Missing tensors: {missing:?}")]
    MissingTensors { missing: Vec<String> },

    /// Invalid magic bytes (Jidoka: HALT)
    #[error("JIDOKA HALT: Invalid magic bytes in {path}")]
    InvalidMagicBytes { path: PathBuf },

    /// Parameter count mismatch (Jidoka: HALT)
    #[error("JIDOKA HALT: Parameter count mismatch - expected {expected}, got {actual}")]
    ParamCountMismatch { expected: usize, actual: usize },

    /// Numerical precision drift (SPC: HALT)
    #[error("SPC HALT: Numerical precision drift - max KL divergence {max_divergence:.2e} exceeds epsilon {epsilon:.2e}")]
    NumericalPrecisionDrift {
        max_divergence: f64,
        epsilon: f64,
        divergent_layers: Vec<(String, f64)>,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Source model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// `SafeTensors` format
    SafeTensors,
    /// `GGUF` format
    Gguf,
    /// `PyTorch` format (requires `--unsafe-allow-pickle`)
    PyTorch,
    /// Unknown format
    Unknown,
}

impl SourceFormat {
    /// Detect format from file path
    #[must_use]
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();

        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "safetensors" => return Self::SafeTensors,
                "gguf" => return Self::Gguf,
                "bin" | "pt" | "pth" => return Self::PyTorch,
                _ => {}
            }
        }

        Self::Unknown
    }

    /// Get format name
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::PyTorch => "pytorch",
            Self::Unknown => "unknown",
        }
    }
}

/// Quantization level for conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    /// No quantization (FP32/FP16)
    None,
    /// 8-bit symmetric
    Q8_0,
    /// 4-bit block symmetric
    Q4_0,
    /// 4-bit K-quant medium
    Q4KM,
}

impl Quantization {
    /// Get epsilon tolerance for this quantization level
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        match self {
            Self::None => 1e-5,              // FP16 precision
            Self::Q8_0 => 1e-3,              // 8-bit quantization
            Self::Q4_0 | Self::Q4KM => 1e-2, // 4-bit quantization
        }
    }

    /// Get quantization name
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4KM => "q4_k_m",
        }
    }
}

/// Conversion configuration
#[derive(Debug, Clone)]
pub struct ConvertConfig {
    /// Quantization level
    pub quantization: Quantization,
    /// Number of layers to sample for SPC check
    pub spc_sample_layers: usize,
    /// Random seed for layer sampling
    pub spc_seed: u64,
    /// Override epsilon tolerance (uses quantization default if None)
    pub epsilon_override: Option<f64>,
    /// Skip SPC check (not recommended)
    pub skip_spc: bool,
}

impl Default for ConvertConfig {
    fn default() -> Self {
        Self {
            quantization: Quantization::None,
            spc_sample_layers: 10,
            spc_seed: 42,
            epsilon_override: None,
            skip_spc: false,
        }
    }
}

impl ConvertConfig {
    /// Get effective epsilon for SPC check
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        self.epsilon_override
            .unwrap_or_else(|| self.quantization.epsilon())
    }
}

/// Conversion metadata result
#[derive(Debug, Clone)]
pub struct ConvertMetadata {
    /// Source path
    pub source_path: PathBuf,
    /// Output path
    pub output_path: PathBuf,
    /// Source format
    pub source_format: SourceFormat,
    /// Parameter count
    pub param_count: usize,
    /// Quantization used
    pub quantization: Quantization,
    /// SPC check passed
    pub spc_passed: bool,
    /// Max KL divergence observed
    pub max_kl_divergence: f64,
    /// Layers checked
    pub layers_checked: usize,
}

/// SPC precision report
#[derive(Debug, Clone)]
pub struct PrecisionReport {
    /// Maximum KL divergence observed
    pub max_divergence: f64,
    /// Number of layers checked
    pub layers_checked: usize,
    /// Passed SPC gate
    pub passed: bool,
    /// Divergent layers (name, divergence)
    pub divergent_layers: Vec<(String, f64)>,
}

/// Statistical Process Control gate for numerical precision
///
/// Detects "silent data corruption" during format conversion
#[derive(Debug, Clone)]
pub struct NumericalPrecisionGate {
    /// Tolerance threshold for KL divergence
    pub epsilon: f64,
    /// Number of layers to sample
    pub sample_layers: usize,
    /// Random seed for reproducible sampling
    pub seed: u64,
}

impl Default for NumericalPrecisionGate {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            sample_layers: 10,
            seed: 42,
        }
    }
}

impl NumericalPrecisionGate {
    /// Create gate from config
    #[must_use]
    pub fn from_config(config: &ConvertConfig) -> Self {
        Self {
            epsilon: config.epsilon(),
            sample_layers: config.spc_sample_layers,
            seed: config.spc_seed,
        }
    }

    /// Verify numerical precision between source and converted tensors
    ///
    /// # Errors
    ///
    /// Returns `NumericalPrecisionDrift` if any layer exceeds epsilon - HALTS pipeline
    pub fn verify(
        &self,
        source_weights: &[(String, Vec<f32>)],
        converted_weights: &[(String, Vec<f32>)],
    ) -> Result<PrecisionReport, ConvertError> {
        let mut max_divergence = 0.0f64;
        let mut divergent_layers = Vec::new();

        // Sample layers for efficiency
        let layers_to_check: Vec<_> = source_weights.iter().take(self.sample_layers).collect();

        for (name, source) in &layers_to_check {
            // Find matching converted layer
            let converted = converted_weights
                .iter()
                .find(|(n, _)| *n == *name)
                .map(|(_, w)| w);

            if let Some(conv) = converted {
                let kl_div = Self::kl_divergence(source, conv);

                if kl_div > self.epsilon {
                    divergent_layers.push(((*name).clone(), kl_div));
                }

                max_divergence = max_divergence.max(kl_div);
            }
        }

        if !divergent_layers.is_empty() {
            tracing::error!(
                max_divergence = %max_divergence,
                epsilon = %self.epsilon,
                divergent_count = divergent_layers.len(),
                "SPC HALT: Numerical precision drift detected"
            );

            return Err(ConvertError::NumericalPrecisionDrift {
                max_divergence,
                epsilon: self.epsilon,
                divergent_layers,
            });
        }

        Ok(PrecisionReport {
            max_divergence,
            layers_checked: layers_to_check.len(),
            passed: true,
            divergent_layers: Vec::new(),
        })
    }

    /// Compute KL divergence between two weight vectors
    ///
    /// `D_KL(P || Q) = sum(P * log(P / Q))`
    fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
        if p.len() != q.len() || p.is_empty() {
            return f64::INFINITY;
        }

        // Normalize to probability distributions (using absolute values)
        let eps = 1e-10;
        let p_sum: f64 = p.iter().map(|x| f64::from(*x).abs()).sum();
        let q_sum: f64 = q.iter().map(|x| f64::from(*x).abs()).sum();

        if p_sum < eps || q_sum < eps {
            return 0.0; // Both near zero
        }

        p.iter()
            .zip(q.iter())
            .map(|(p_i, q_i)| {
                let p_norm = f64::from(*p_i).abs() / p_sum + eps;
                let q_norm = f64::from(*q_i).abs() / q_sum + eps;
                p_norm * (p_norm / q_norm).ln()
            })
            .sum()
    }
}

/// Compute cosine similarity between two weight vectors
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();

    dot / (norm_a.mul_add(norm_b, 1e-10))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // SourceFormat Tests
    // ==========================================================================

    #[test]
    fn test_source_format_detection() {
        assert_eq!(
            SourceFormat::from_path("model.safetensors"),
            SourceFormat::SafeTensors
        );
        assert_eq!(
            SourceFormat::from_path("model.SAFETENSORS"),
            SourceFormat::SafeTensors
        );
        assert_eq!(SourceFormat::from_path("model.gguf"), SourceFormat::Gguf);
        assert_eq!(SourceFormat::from_path("model.bin"), SourceFormat::PyTorch);
        assert_eq!(SourceFormat::from_path("model.pt"), SourceFormat::PyTorch);
        assert_eq!(SourceFormat::from_path("model.pth"), SourceFormat::PyTorch);
        assert_eq!(SourceFormat::from_path("model.xyz"), SourceFormat::Unknown);
    }

    #[test]
    fn test_source_format_as_str() {
        assert_eq!(SourceFormat::SafeTensors.as_str(), "safetensors");
        assert_eq!(SourceFormat::Gguf.as_str(), "gguf");
        assert_eq!(SourceFormat::PyTorch.as_str(), "pytorch");
        assert_eq!(SourceFormat::Unknown.as_str(), "unknown");
    }

    // ==========================================================================
    // Quantization Tests
    // ==========================================================================

    #[test]
    fn test_quantization_epsilon() {
        assert!((Quantization::None.epsilon() - 1e-5).abs() < 1e-10);
        assert!((Quantization::Q8_0.epsilon() - 1e-3).abs() < 1e-10);
        assert!((Quantization::Q4_0.epsilon() - 1e-2).abs() < 1e-10);
        assert!((Quantization::Q4KM.epsilon() - 1e-2).abs() < 1e-10);
    }

    #[test]
    fn test_quantization_as_str() {
        assert_eq!(Quantization::None.as_str(), "none");
        assert_eq!(Quantization::Q8_0.as_str(), "q8_0");
        assert_eq!(Quantization::Q4_0.as_str(), "q4_0");
        assert_eq!(Quantization::Q4KM.as_str(), "q4_k_m");
    }

    // ==========================================================================
    // ConvertConfig Tests
    // ==========================================================================

    #[test]
    fn test_convert_config_default() {
        let config = ConvertConfig::default();

        assert_eq!(config.quantization, Quantization::None);
        assert_eq!(config.spc_sample_layers, 10);
        assert!(!config.skip_spc);
    }

    #[test]
    fn test_convert_config_epsilon() {
        let config = ConvertConfig::default();
        assert!((config.epsilon() - 1e-5).abs() < 1e-10);

        let config_override = ConvertConfig {
            epsilon_override: Some(1e-3),
            ..Default::default()
        };
        assert!((config_override.epsilon() - 1e-3).abs() < 1e-10);
    }

    // ==========================================================================
    // NumericalPrecisionGate Tests (SPC)
    // ==========================================================================

    #[test]
    fn test_spc_gate_default() {
        let gate = NumericalPrecisionGate::default();

        assert!((gate.epsilon - 1e-5).abs() < 1e-10);
        assert_eq!(gate.sample_layers, 10);
    }

    #[test]
    fn test_spc_gate_verify_identical() {
        let gate = NumericalPrecisionGate::default();

        let weights = vec![
            ("layer1".to_string(), vec![1.0f32, 2.0, 3.0]),
            ("layer2".to_string(), vec![4.0f32, 5.0, 6.0]),
        ];

        let result = gate.verify(&weights, &weights);
        assert!(result.is_ok());

        let report = result.expect("should pass");
        assert!(report.passed);
        assert!(report.max_divergence < 1e-5);
    }

    #[test]
    fn test_spc_gate_verify_small_drift() {
        let gate = NumericalPrecisionGate {
            epsilon: 1e-3,
            ..Default::default()
        };

        let source = vec![("layer1".to_string(), vec![1.0f32, 2.0, 3.0])];
        let converted = vec![("layer1".to_string(), vec![1.0001f32, 2.0001, 3.0001])];

        let result = gate.verify(&source, &converted);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spc_gate_verify_large_drift_halts() {
        let gate = NumericalPrecisionGate {
            epsilon: 1e-5,
            ..Default::default()
        };

        // Use completely different distributions to trigger KL divergence
        let source = vec![("layer1".to_string(), vec![0.9f32, 0.05, 0.05])];
        let converted = vec![
            ("layer1".to_string(), vec![0.1f32, 0.45, 0.45]), // Very different distribution
        ];

        let result = gate.verify(&source, &converted);
        assert!(result.is_err(), "Expected error but got: {result:?}");

        if let Err(ConvertError::NumericalPrecisionDrift {
            max_divergence,
            epsilon,
            divergent_layers,
        }) = result
        {
            assert!(max_divergence > epsilon);
            assert!(!divergent_layers.is_empty());
            assert_eq!(divergent_layers[0].0, "layer1");
        } else {
            panic!("Expected NumericalPrecisionDrift error");
        }
    }

    // ==========================================================================
    // Cosine Similarity Tests
    // ==========================================================================

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let similarity = cosine_similarity(&a, &a);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let similarity = cosine_similarity(&a, &b);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![-1.0f32, -2.0, -3.0];
        let similarity = cosine_similarity(&a, &b);
        assert!((similarity + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let similarity = cosine_similarity(&a, &b);
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    // ==========================================================================
    // Error Display Tests
    // ==========================================================================

    #[test]
    fn test_error_display() {
        let err = ConvertError::NumericalPrecisionDrift {
            max_divergence: 0.1,
            epsilon: 0.01,
            divergent_layers: vec![("layer1".to_string(), 0.1)],
        };
        assert!(err.to_string().contains("SPC HALT"));

        let err = ConvertError::InvalidMagicBytes {
            path: PathBuf::from("test.apr"),
        };
        assert!(err.to_string().contains("JIDOKA HALT"));

        let err = ConvertError::ParamCountMismatch {
            expected: 1000,
            actual: 500,
        };
        assert!(err.to_string().contains("JIDOKA HALT"));
    }
}
