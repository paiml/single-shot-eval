//! # Single-Shot Eval
//!
//! SLM Pareto Frontier Evaluation Framework for evaluating Small Language Models
//! via single-shot compile Python to Rust methodology.
//!
//! ## Core Thesis
//!
//! A 100M-parameter SLM, when properly distilled using Depyler's single-shot
//! compilation insights, can match or exceed frontier model performance on
//! domain-specific tasks while reducing inference cost by 100-1000x.
//!
//! ## Research Basis
//!
//! Based on Princeton "AI Agents That Matter" (2024) methodology:
//! - 5 runs minimum with 95% CI using Student's t-distribution
//! - Convex Pareto frontiers (probability-weighted combinations)
//! - Dollar costs, not proxy metrics
//! - Ground truth via compilation + test execution (SWE-bench style)
//!
//! ## Architecture
//!
//! ```text
//! Python Source (reprorusted-python-cli)
//!        ↓
//! Model Inference (SLM .apr | SaaS baselines)
//!        ↓
//! Rust Output
//!        ↓
//! Ground Truth Verification (cargo build && cargo test)
//!        ↓
//! Metrics (pass@1, cost, latency)
//!        ↓
//! Pareto Frontier Analysis
//!        ↓
//! Report (Princeton-compliant: 5 runs, 95% CI)
//! ```

pub mod baselines;
pub mod bench_bridge;
pub mod compiler;
pub mod config;
pub mod convert;
pub mod corpus;
pub mod download;
pub mod inference;
pub mod metrics;
pub mod pareto;
pub mod report;
pub mod runner;
pub mod sovereign;
pub mod validate;

pub use baselines::{
    available_baselines, run_all_baselines, BaselineError, BaselineResult, BaselineRunner,
};
pub use compiler::{
    BatchResult, BatchVerifier, CompilerConfig, CompilerError, CompilerVerifier, VerificationResult,
};
pub use config::{
    validate_apr_format, BaselineConfig, EvalConfig, EvaluationSettings, GroundTruthConfig,
    MetricType, PromptConfig, TaskConfig, TaskDefinition, TaskLoader,
};
pub use corpus::{Corpus, CorpusError, CorpusStats, PythonExample};
pub use inference::{
    create_placeholder_model, InferenceError, InferenceOutput, LoadedModel, ModelConfig,
    ModelLoader, ModelMetadata,
};
pub use metrics::{
    bonferroni_correction, bootstrap_ci, paired_t_test, welch_t_test, AggregatedMetrics,
    MetricsCollector, SignificanceResult, StatConfig,
};
pub use pareto::{
    analyze_pareto, compute_pareto_frontier, EvalResult, ParetoAnalysis, TradeOffAnalysis,
};
pub use report::{FullReport, ModelReport, ReportBuilder, ReportMetadata, ReportSummary};
pub use runner::{
    BaselineEvalResult, EvalSample, EvaluationReport, InferenceResult, RunnerConfig, RunnerError,
    TaskRunner,
};

// Re-export aprender::bench integration (SSE-012)
pub use bench_bridge::{
    batch_to_bench_examples, classify_example_level, create_model_comparison, infer_difficulty,
    level_to_example_pattern, to_bench_example, BenchEvalResult, BenchExample, BenchExampleResult,
    Difficulty, EvalSuiteConfig, EvalTask, ExampleStatus, LevelResult, ModelComparison,
    ParetoPoint, Py2RsLevel, Py2RsScore, Recommendation,
};

// Re-export sovereign inference - native .apr model execution
pub use sovereign::{
    is_sovereign_available, list_models, ModelFormat, SovereignError, SovereignResult,
    SovereignRunner,
};

// Re-export download module (HuggingFace integration with JIT caching)
pub use download::{
    compute_sha256, validate_format_safety, verify_checksum, CacheEntry, CacheManager,
    DownloadConfig, DownloadError, ModelArtifact, SafetyClass,
};

// Re-export convert module (format conversion with SPC gate)
pub use convert::{
    cosine_similarity, ConvertConfig, ConvertError, ConvertMetadata, NumericalPrecisionGate,
    PrecisionReport, Quantization, SourceFormat,
};

// Re-export validate module (logit consistency checking)
pub use validate::{
    validate_apr_magic, ConsistencyResult, DivergentSample, LogitConsistencyChecker, TokenLogit,
    ValidationConfig, ValidationError, APR_MAGIC,
};
