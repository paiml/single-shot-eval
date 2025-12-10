//! Integration tests for download-convert-test pipeline
//!
//! Based on docs/specifications/download-convert-test-spec.md
//! Toyota Way principles:
//! - Jidoka: Tests halt on quality gate failure
//! - Poka-yoke: Unsafe formats rejected
//! - SPC: Numerical precision validated

use single_shot_eval::{
    validate_apr_magic, validate_format_safety, CacheManager, ConvertConfig, DownloadConfig,
    LogitConsistencyChecker, NumericalPrecisionGate, Quantization, SourceFormat, ValidationConfig,
    APR_MAGIC,
};
use std::path::PathBuf;
use tempfile::TempDir;

// =============================================================================
// SECTION 1: Download Module Tests (Jidoka + Poka-yoke)
// =============================================================================

#[test]
fn test_download_config_defaults() {
    let config = DownloadConfig::default();

    // Verify Toyota Way defaults
    assert!(config.verify_checksum, "Jidoka: Checksum verification must be ON by default");
    assert!(
        !config.unsafe_allow_pickle,
        "Poka-yoke: Pickle must be OFF by default"
    );
    assert_eq!(
        config.max_cache_bytes,
        10 * 1024 * 1024 * 1024, // 10 GiB
        "JIT: 10GB cache limit"
    );
}

#[test]
fn test_poka_yoke_pickle_rejection() {
    let config = DownloadConfig::default();

    // Pickle formats must be rejected
    let pickle_extensions = ["model.bin", "model.pt", "model.pth", "model.pkl"];

    for filename in &pickle_extensions {
        let path = PathBuf::from(filename);
        let result = validate_format_safety(&path, &config);
        assert!(
            result.is_err(),
            "Poka-yoke FAIL: {filename} should be rejected"
        );
    }
}

#[test]
fn test_safe_formats_accepted() {
    let config = DownloadConfig::default();

    // Safe formats must be accepted
    let safe_formats = ["model.safetensors", "model.gguf", "model.apr"];

    for filename in &safe_formats {
        let path = PathBuf::from(filename);
        let result = validate_format_safety(&path, &config);
        assert!(
            result.is_ok(),
            "Safe format {filename} should be accepted: {result:?}"
        );
    }
}

#[test]
fn test_unsafe_pickle_flag_overrides() {
    let config = DownloadConfig {
        unsafe_allow_pickle: true,
        ..Default::default()
    };

    // With --unsafe-allow-pickle, pickle formats allowed
    let path = PathBuf::from("model.bin");
    let result = validate_format_safety(&path, &config);
    assert!(
        result.is_ok(),
        "With unsafe flag, pickle should be allowed"
    );
}

#[test]
fn test_cache_manager_lru_policy() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let cache_path = temp_dir.path().to_path_buf();

    // Create small cache for testing
    let config = DownloadConfig {
        cache_dir: cache_path,
        max_cache_bytes: 1024, // 1KB limit for test
        ..Default::default()
    };

    let manager = CacheManager::new(config);

    // Verify LRU policy properties
    assert!(manager.has_space(512), "Should allow file under limit");
    assert!(!manager.has_space(2048), "Should reject file over limit");
}

// =============================================================================
// SECTION 2: Convert Module Tests (SPC Gate)
// =============================================================================

#[test]
fn test_convert_config_defaults() {
    let config = ConvertConfig::default();

    assert_eq!(
        config.quantization,
        Quantization::None,
        "Default quantization should be None"
    );
    assert!(!config.skip_spc, "SPC check should be enabled by default");
    assert_eq!(
        config.spc_sample_layers, 10,
        "Default SPC sample layers"
    );
}

#[test]
fn test_quantization_epsilon_values() {
    // Verify epsilon thresholds per spec Table 6.1
    assert!(
        (Quantization::None.epsilon() - 1e-5).abs() < 1e-10,
        "FP16/FP32 epsilon should be 1e-5"
    );
    assert!(
        (Quantization::Q8_0.epsilon() - 1e-3).abs() < 1e-10,
        "Q8_0 epsilon should be 1e-3"
    );
    assert!(
        (Quantization::Q4_0.epsilon() - 1e-2).abs() < 1e-10,
        "Q4_0 epsilon should be 1e-2"
    );
    assert!(
        (Quantization::Q4KM.epsilon() - 1e-2).abs() < 1e-10,
        "Q4_K_M epsilon should be 1e-2"
    );
}

#[test]
fn test_source_format_detection() {
    assert_eq!(
        SourceFormat::from_path("model.safetensors"),
        SourceFormat::SafeTensors
    );
    assert_eq!(SourceFormat::from_path("model.gguf"), SourceFormat::Gguf);
    assert_eq!(SourceFormat::from_path("model.bin"), SourceFormat::PyTorch);
    assert_eq!(SourceFormat::from_path("model.pt"), SourceFormat::PyTorch);
    assert_eq!(
        SourceFormat::from_path("model.unknown"),
        SourceFormat::Unknown
    );
}

#[test]
fn test_spc_gate_identical_weights_pass() {
    let gate = NumericalPrecisionGate::default();

    let weights = vec![
        ("layer1".to_string(), vec![1.0f32, 2.0, 3.0, 4.0]),
        ("layer2".to_string(), vec![5.0f32, 6.0, 7.0, 8.0]),
    ];

    // Identical weights should pass SPC gate
    let result = gate.verify(&weights, &weights);
    assert!(result.is_ok(), "Identical weights should pass SPC");

    let report = result.expect("report");
    assert!(report.passed, "SPC report should show passed");
    assert!(
        report.max_divergence < gate.epsilon,
        "Max divergence should be below epsilon"
    );
}

#[test]
fn test_spc_gate_drift_detection() {
    let gate = NumericalPrecisionGate {
        epsilon: 1e-5,
        sample_layers: 10,
        seed: 42,
    };

    // Create weights with significant drift
    let source = vec![("layer1".to_string(), vec![0.9f32, 0.05, 0.05])];
    let drifted = vec![("layer1".to_string(), vec![0.1f32, 0.45, 0.45])];

    let result = gate.verify(&source, &drifted);
    assert!(result.is_err(), "Drifted weights should fail SPC");

    if let Err(e) = result {
        assert!(
            e.to_string().contains("SPC HALT"),
            "Error should indicate SPC halt"
        );
    }
}

// =============================================================================
// SECTION 3: Validate Module Tests (Logit Consistency)
// =============================================================================

#[test]
fn test_apr_magic_bytes_constant() {
    // APR! in ASCII
    assert_eq!(APR_MAGIC, [0x41, 0x50, 0x52, 0x21]);
}

#[test]
fn test_validate_apr_magic_valid_file() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("test.apr");

    // Write valid APR magic
    let mut data = APR_MAGIC.to_vec();
    data.extend_from_slice(b"model data here");
    std::fs::write(&file_path, &data).expect("write file");

    let result = validate_apr_magic(&file_path);
    assert!(result.is_ok(), "Valid APR magic should pass");
}

#[test]
fn test_validate_apr_magic_invalid_halts() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("bad.apr");

    // Write invalid magic
    std::fs::write(&file_path, b"BAAD").expect("write file");

    let result = validate_apr_magic(&file_path);
    assert!(result.is_err(), "Invalid magic should fail");

    if let Err(e) = result {
        assert!(
            e.to_string().contains("JIDOKA HALT"),
            "Error should indicate Jidoka halt"
        );
    }
}

#[test]
fn test_logit_consistency_checker_defaults() {
    let checker = LogitConsistencyChecker::default();

    assert_eq!(checker.top_k, 10, "Default top-k should be 10");
    assert!(
        (checker.logit_tolerance - 0.1).abs() < 1e-6,
        "Default logit tolerance should be 0.1"
    );
    assert!(
        (checker.min_agreement - 0.9).abs() < 1e-6,
        "Default min agreement should be 90%"
    );
}

#[test]
fn test_logit_consistency_identical_outputs() {
    let checker = LogitConsistencyChecker::default();

    let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let original = vec![logits.clone()];
    let converted = vec![logits];
    let prompts = vec!["test prompt".to_string()];

    let result = checker.verify(&original, &converted, &prompts);
    assert!(result.is_ok(), "Identical outputs should pass");

    let report = result.expect("report");
    assert!(report.passed, "Report should show passed");
    assert!(
        (report.agreement_rate - 1.0).abs() < 1e-6,
        "Agreement rate should be 100%"
    );
}

#[test]
fn test_logit_consistency_divergent_halts() {
    let checker = LogitConsistencyChecker {
        top_k: 5,
        logit_tolerance: 0.1,
        min_agreement: 0.9,
    };

    // Completely different logit distributions
    let original = vec![vec![1.0, 0.9, 0.8, 0.7, 0.6]];
    let converted = vec![vec![0.6, 0.7, 0.8, 0.9, 1.0]]; // Reversed order
    let prompts = vec!["test".to_string()];

    let result = checker.verify(&original, &converted, &prompts);
    assert!(result.is_err(), "Divergent logits should fail");

    if let Err(e) = result {
        assert!(
            e.to_string().contains("JIDOKA HALT"),
            "Error should indicate Jidoka halt"
        );
    }
}

// =============================================================================
// SECTION 4: Validation Config Tests
// =============================================================================

#[test]
fn test_validation_config_defaults() {
    let config = ValidationConfig::default();

    assert!(config.check_magic, "Magic check should be ON by default");
    assert!(
        config.check_metadata,
        "Metadata check should be ON by default"
    );
    assert!(
        config.check_logit_consistency,
        "Logit consistency should be ON by default"
    );
}

// =============================================================================
// SECTION 5: End-to-End Pipeline Tests
// =============================================================================

#[test]
fn test_validation_prompts_file_exists() {
    let prompts_path = PathBuf::from("prompts/validation-prompts.yaml");
    assert!(
        prompts_path.exists(),
        "Validation prompts file must exist at prompts/validation-prompts.yaml"
    );
}

#[test]
fn test_model_config_file_exists() {
    let config_path = PathBuf::from("models/test-models.yaml");
    assert!(
        config_path.exists(),
        "Model config file must exist at models/test-models.yaml"
    );
}

#[test]
fn test_validation_prompts_yaml_valid() {
    let prompts_path = PathBuf::from("prompts/validation-prompts.yaml");
    let content = std::fs::read_to_string(&prompts_path).expect("read prompts file");

    // Check for required sections
    assert!(
        content.contains("metadata:"),
        "Prompts file must have metadata section"
    );
    assert!(
        content.contains("prompts:"),
        "Prompts file must have prompts section"
    );
    assert!(
        content.contains("category: sanity"),
        "Prompts file must have sanity checks"
    );
    assert!(
        content.contains("category: mmlu"),
        "Prompts file must have MMLU prompts"
    );
    assert!(
        content.contains("category: code"),
        "Prompts file must have code prompts"
    );
}

#[test]
fn test_model_config_yaml_valid() {
    let config_path = PathBuf::from("models/test-models.yaml");
    let content = std::fs::read_to_string(&config_path).expect("read config file");

    // Check for required sections
    assert!(
        content.contains("metadata:"),
        "Config file must have metadata section"
    );
    assert!(
        content.contains("models:"),
        "Config file must have models section"
    );
    assert!(
        content.contains("validation:"),
        "Config file must have validation section"
    );
    assert!(
        content.contains("tier: nano"),
        "Config file must have nano tier models"
    );
    assert!(
        content.contains("tier: small"),
        "Config file must have small tier models"
    );
}

// =============================================================================
// SECTION 6: Toyota Way Principle Verification
// =============================================================================

#[test]
fn test_jidoka_halt_on_error() {
    // Jidoka: System should halt and signal on quality issues

    // Test 1: Invalid magic bytes should halt
    let temp_dir = TempDir::new().expect("create temp dir");
    let file_path = temp_dir.path().join("invalid.apr");
    std::fs::write(&file_path, b"XXXX").expect("write file");
    let result = validate_apr_magic(&file_path);
    assert!(
        result.is_err(),
        "Jidoka: Must halt on invalid magic bytes"
    );

    // Test 2: Numerical drift should halt
    let gate = NumericalPrecisionGate {
        epsilon: 1e-5,
        ..Default::default()
    };
    let source = vec![("layer".to_string(), vec![0.9f32, 0.05, 0.05])];
    let drifted = vec![("layer".to_string(), vec![0.1f32, 0.45, 0.45])];
    let result = gate.verify(&source, &drifted);
    assert!(result.is_err(), "Jidoka: Must halt on numerical drift");
}

#[test]
fn test_poka_yoke_error_prevention() {
    // Poka-yoke: System should prevent errors through design

    let config = DownloadConfig::default();

    // Pickle files are rejected by default - cannot accidentally allow
    assert!(
        !config.unsafe_allow_pickle,
        "Poka-yoke: Pickle rejection ON by default"
    );

    // Checksum verification is on by default - cannot accidentally skip
    assert!(
        config.verify_checksum,
        "Poka-yoke: Checksum verification ON by default"
    );

    // SPC check is on by default - cannot accidentally skip
    let convert_config = ConvertConfig::default();
    assert!(
        !convert_config.skip_spc,
        "Poka-yoke: SPC check ON by default"
    );
}

#[test]
fn test_spc_statistical_process_control() {
    // SPC: Verify numerical precision is maintained

    let gate = NumericalPrecisionGate::default();

    // Test with weights that should pass
    let weights = vec![
        ("layer1".to_string(), vec![1.0f32, 2.0, 3.0]),
        ("layer2".to_string(), vec![4.0f32, 5.0, 6.0]),
    ];

    let result = gate.verify(&weights, &weights);
    assert!(result.is_ok(), "SPC: Identical weights must pass");

    let report = result.expect("report");
    assert!(report.passed, "SPC: Report must show passed");
    assert!(
        report.divergent_layers.is_empty(),
        "SPC: No divergent layers"
    );
}
