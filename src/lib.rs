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
pub mod compiler;
pub mod config;
pub mod corpus;
pub mod metrics;
pub mod pareto;
pub mod report;
pub mod runner;

pub use baselines::{
    available_baselines, run_all_baselines, BaselineError, BaselineResult, BaselineRunner,
};
pub use config::{
    validate_apr_format, BaselineConfig, EvalConfig, EvaluationSettings, GroundTruthConfig,
    MetricType, PromptConfig, TaskConfig, TaskDefinition, TaskLoader,
};
pub use compiler::{
    BatchResult, BatchVerifier, CompilerConfig, CompilerError, CompilerVerifier, VerificationResult,
};
pub use corpus::{Corpus, CorpusError, CorpusStats, PythonExample};
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
