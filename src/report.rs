//! Report generation module for evaluation results.
//!
//! Generates comprehensive reports showing:
//! - Pareto frontier analysis
//! - Trade-off comparisons
//! - Statistical significance
//! - Value improvement metrics

use crate::metrics::{bootstrap_ci, welch_t_test, SignificanceResult, StatConfig};
use crate::pareto::{analyze_pareto, EvalResult, ParetoAnalysis};
use crate::runner::{BaselineEvalResult, EvaluationReport};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use tabled::{Table, Tabled};

/// Full evaluation report with all analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Pareto analysis results
    pub pareto_analysis: ParetoAnalysis,
    /// Statistical comparisons
    pub statistical_tests: Vec<StatisticalComparison>,
    /// Per-model detailed results
    pub model_results: Vec<ModelReport>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,
    /// Task identifier
    pub task_id: String,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Framework version
    pub framework_version: String,
    /// Statistical configuration used
    pub stat_config: StatConfigSummary,
}

/// Statistical configuration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatConfigSummary {
    /// Bootstrap resamples
    pub bootstrap_n: usize,
    /// Confidence level
    pub confidence: f64,
    /// Significance threshold
    pub alpha: f64,
}

impl From<&StatConfig> for StatConfigSummary {
    fn from(config: &StatConfig) -> Self {
        Self {
            bootstrap_n: config.bootstrap_n,
            confidence: config.confidence,
            alpha: config.alpha,
        }
    }
}

/// High-level summary of evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total models evaluated
    pub total_models: usize,
    /// Models on Pareto frontier
    pub frontier_models: usize,
    /// Best accuracy achieved
    pub best_accuracy: f64,
    /// Best cost efficiency (lowest cost)
    pub best_cost: f64,
    /// Best latency (lowest)
    pub best_latency_ms: u64,
    /// SLM value improvement factor (if applicable)
    pub slm_value_factor: Option<f64>,
    /// Whether SLM is Pareto-dominant
    pub slm_pareto_dominant: bool,
}

/// Statistical comparison between two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison {
    /// Model A identifier
    pub model_a: String,
    /// Model B identifier
    pub model_b: String,
    /// Metric being compared
    pub metric: String,
    /// Significance test result
    pub significance: SignificanceResult,
}

/// Detailed per-model report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReport {
    /// Model identifier
    pub model_id: String,
    /// Is on Pareto frontier
    pub on_frontier: bool,
    /// Accuracy with confidence interval
    pub accuracy: MetricWithCI,
    /// Cost per 1M tokens
    pub cost: f64,
    /// p99 latency in milliseconds
    pub latency_ms: u64,
    /// Value score relative to best baseline
    pub value_score: Option<f64>,
}

/// Metric value with confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWithCI {
    /// Point estimate
    pub value: f64,
    /// Lower bound of CI
    pub ci_lower: f64,
    /// Upper bound of CI
    pub ci_upper: f64,
}

/// Report builder for constructing full reports
pub struct ReportBuilder {
    task_id: String,
    stat_config: StatConfig,
    results: Vec<EvalResult>,
    baseline_results: Vec<BaselineEvalResult>,
    accuracy_samples: Vec<Vec<f64>>,
}

impl ReportBuilder {
    /// Create a new report builder
    #[must_use]
    pub fn new(task_id: &str) -> Self {
        Self {
            task_id: task_id.to_string(),
            stat_config: StatConfig::default(),
            results: Vec::new(),
            baseline_results: Vec::new(),
            accuracy_samples: Vec::new(),
        }
    }

    /// Set statistical configuration
    #[must_use]
    pub const fn with_stat_config(mut self, config: StatConfig) -> Self {
        self.stat_config = config;
        self
    }

    /// Add an evaluation result with accuracy samples
    pub fn add_result(&mut self, result: EvalResult, accuracy_samples: Vec<f64>) {
        self.results.push(result);
        self.accuracy_samples.push(accuracy_samples);
    }

    /// Add baseline results
    pub fn add_baselines(&mut self, baselines: Vec<BaselineEvalResult>) {
        self.baseline_results = baselines;
    }

    /// Build from an evaluation report
    #[must_use]
    pub fn from_evaluation_report(report: &EvaluationReport) -> Self {
        let mut builder = Self::new(&report.task_id);

        // Add SLM result
        builder.results.push(report.slm_result.clone());
        // Add placeholder samples (in real implementation, these would come from actual evaluation)
        builder
            .accuracy_samples
            .push(vec![report.slm_result.accuracy; 100]);

        // Add baseline results
        builder
            .baseline_results
            .clone_from(&report.baseline_results);

        builder
    }

    /// Build the full report
    #[must_use]
    pub fn build(self) -> FullReport {
        let pareto_analysis = analyze_pareto(&self.results);

        let summary = self.build_summary(&pareto_analysis);
        let model_results = self.build_model_reports(&pareto_analysis);
        let statistical_tests = self.build_statistical_tests();

        FullReport {
            metadata: ReportMetadata {
                title: format!("Evaluation Report: {}", self.task_id),
                task_id: self.task_id,
                generated_at: Utc::now(),
                framework_version: env!("CARGO_PKG_VERSION").to_string(),
                stat_config: StatConfigSummary::from(&self.stat_config),
            },
            summary,
            pareto_analysis,
            statistical_tests,
            model_results,
        }
    }

    fn build_summary(&self, pareto: &ParetoAnalysis) -> ReportSummary {
        let best_accuracy = self.results.iter().map(|r| r.accuracy).fold(0.0, f64::max);

        let best_cost = self.results.iter().map(|r| r.cost).fold(f64::MAX, f64::min);

        let best_latency_ms = self
            .results
            .iter()
            .map(|r| r.latency.as_millis() as u64)
            .min()
            .unwrap_or(0);

        // Calculate SLM value factor if we have baselines
        let slm_value_factor = if !self.baseline_results.is_empty() && !self.results.is_empty() {
            let slm = &self.results[0];
            let best_baseline = self.baseline_results.iter().max_by(|a, b| {
                a.accuracy
                    .partial_cmp(&b.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            best_baseline.map(|baseline| {
                let acc_ratio = slm.accuracy / baseline.accuracy.max(0.001);
                let cost_ratio = baseline.cost / slm.cost.max(0.00001);
                let lat_ratio =
                    baseline.latency.as_secs_f64() / slm.latency.as_secs_f64().max(0.001);
                acc_ratio * cost_ratio * lat_ratio
            })
        } else {
            None
        };

        let slm_pareto_dominant = !self.baseline_results.is_empty()
            && !self.results.is_empty()
            && self.baseline_results.iter().any(|baseline| {
                let slm = &self.results[0];
                slm.accuracy >= baseline.accuracy
                    && slm.cost <= baseline.cost
                    && slm.latency.as_millis() as u64 <= baseline.latency.as_millis() as u64
                    && (slm.accuracy > baseline.accuracy
                        || slm.cost < baseline.cost
                        || slm.latency < baseline.latency)
            });

        ReportSummary {
            total_models: self.results.len() + self.baseline_results.len(),
            frontier_models: pareto.frontier_models.len(),
            best_accuracy,
            best_cost,
            best_latency_ms,
            slm_value_factor,
            slm_pareto_dominant,
        }
    }

    fn build_model_reports(&self, pareto: &ParetoAnalysis) -> Vec<ModelReport> {
        self.results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                let on_frontier = pareto.frontier_models.contains(&result.model_id);

                let (ci_lower, ci_upper) = if i < self.accuracy_samples.len() {
                    bootstrap_ci(&self.accuracy_samples[i], &self.stat_config)
                } else {
                    (result.accuracy, result.accuracy)
                };

                let value_score = pareto
                    .trade_offs
                    .iter()
                    .find(|t| t.model_id == result.model_id)
                    .map(|t| t.value_score);

                ModelReport {
                    model_id: result.model_id.clone(),
                    on_frontier,
                    accuracy: MetricWithCI {
                        value: result.accuracy,
                        ci_lower,
                        ci_upper,
                    },
                    cost: result.cost,
                    latency_ms: result.latency.as_millis() as u64,
                    value_score,
                }
            })
            .collect()
    }

    fn build_statistical_tests(&self) -> Vec<StatisticalComparison> {
        let mut comparisons = Vec::new();

        // Compare each pair of models
        for i in 0..self.accuracy_samples.len() {
            for j in (i + 1)..self.accuracy_samples.len() {
                if let Some(significance) = welch_t_test(
                    &self.accuracy_samples[i],
                    &self.accuracy_samples[j],
                    self.stat_config.alpha,
                ) {
                    comparisons.push(StatisticalComparison {
                        model_a: self.results[i].model_id.clone(),
                        model_b: self.results[j].model_id.clone(),
                        metric: "accuracy".to_string(),
                        significance,
                    });
                }
            }
        }

        comparisons
    }
}

/// Table row for text/markdown output
#[derive(Tabled)]
struct ResultTableRow {
    #[tabled(rename = "Model")]
    model: String,
    #[tabled(rename = "Accuracy")]
    accuracy: String,
    #[tabled(rename = "Cost/1M")]
    cost: String,
    #[tabled(rename = "Latency (p99)")]
    latency: String,
    #[tabled(rename = "Frontier")]
    frontier: String,
    #[tabled(rename = "Value")]
    value: String,
}

impl FullReport {
    /// Render report as JSON
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render report as markdown
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Title
        writeln!(output, "# {}", self.metadata.title).ok();
        writeln!(output).ok();
        writeln!(
            output,
            "**Generated:** {}",
            self.metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        )
        .ok();
        writeln!(
            output,
            "**Framework Version:** {}",
            self.metadata.framework_version
        )
        .ok();
        writeln!(output).ok();

        // Summary
        writeln!(output, "## Summary").ok();
        writeln!(output).ok();
        writeln!(output, "| Metric | Value |").ok();
        writeln!(output, "|--------|-------|").ok();
        writeln!(output, "| Total Models | {} |", self.summary.total_models).ok();
        writeln!(
            output,
            "| Frontier Models | {} |",
            self.summary.frontier_models
        )
        .ok();
        writeln!(
            output,
            "| Best Accuracy | {:.2}% |",
            self.summary.best_accuracy * 100.0
        )
        .ok();
        writeln!(output, "| Best Cost | ${:.6}/1M |", self.summary.best_cost).ok();
        writeln!(
            output,
            "| Best Latency | {}ms |",
            self.summary.best_latency_ms
        )
        .ok();
        if let Some(factor) = self.summary.slm_value_factor {
            writeln!(output, "| SLM Value Factor | {factor:.1}x |").ok();
        }
        writeln!(
            output,
            "| SLM Pareto Dominant | {} |",
            if self.summary.slm_pareto_dominant {
                "Yes"
            } else {
                "No"
            }
        )
        .ok();
        writeln!(output).ok();

        // Results table
        writeln!(output, "## Model Results").ok();
        writeln!(output).ok();

        let rows: Vec<ResultTableRow> = self
            .model_results
            .iter()
            .map(|r| ResultTableRow {
                model: r.model_id.clone(),
                accuracy: format!(
                    "{:.2}% [{:.2}-{:.2}]",
                    r.accuracy.value * 100.0,
                    r.accuracy.ci_lower * 100.0,
                    r.accuracy.ci_upper * 100.0
                ),
                cost: format!("${:.6}", r.cost),
                latency: format!("{}ms", r.latency_ms),
                frontier: if r.on_frontier { "✓" } else { "" }.to_string(),
                value: r
                    .value_score
                    .map_or_else(|| "-".to_string(), |v| format!("{v:.1}x")),
            })
            .collect();

        let table = Table::new(rows).to_string();
        writeln!(output, "{table}").ok();
        writeln!(output).ok();

        // Pareto frontier
        writeln!(output, "## Pareto Frontier").ok();
        writeln!(output).ok();
        writeln!(
            output,
            "**Frontier Models:** {}",
            self.pareto_analysis.frontier_models.join(", ")
        )
        .ok();
        writeln!(output).ok();
        if !self.pareto_analysis.dominated_models.is_empty() {
            writeln!(
                output,
                "**Dominated Models:** {}",
                self.pareto_analysis.dominated_models.join(", ")
            )
            .ok();
        }
        writeln!(output).ok();

        // Statistical tests
        if !self.statistical_tests.is_empty() {
            writeln!(output, "## Statistical Comparisons").ok();
            writeln!(output).ok();
            writeln!(
                output,
                "| Comparison | t-stat | p-value | Effect Size | Significant |"
            )
            .ok();
            writeln!(
                output,
                "|------------|--------|---------|-------------|-------------|"
            )
            .ok();
            for test in &self.statistical_tests {
                writeln!(
                    output,
                    "| {} vs {} | {:.3} | {:.4} | {} ({:.2}) | {} |",
                    test.model_a,
                    test.model_b,
                    test.significance.t_statistic,
                    test.significance.p_value,
                    test.significance.effect_interpretation,
                    test.significance.cohens_d,
                    if test.significance.is_significant {
                        "Yes"
                    } else {
                        "No"
                    }
                )
                .ok();
            }
            writeln!(output).ok();
        }

        // Configuration
        writeln!(output, "## Configuration").ok();
        writeln!(output).ok();
        writeln!(
            output,
            "- Bootstrap resamples: {}",
            self.metadata.stat_config.bootstrap_n
        )
        .ok();
        writeln!(
            output,
            "- Confidence level: {}%",
            self.metadata.stat_config.confidence * 100.0
        )
        .ok();
        writeln!(
            output,
            "- Significance threshold (α): {}",
            self.metadata.stat_config.alpha
        )
        .ok();

        output
    }

    /// Render report as plain text table
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut output = String::new();

        writeln!(
            output,
            "═══════════════════════════════════════════════════════════════"
        )
        .ok();
        writeln!(output, "  {}", self.metadata.title).ok();
        writeln!(
            output,
            "═══════════════════════════════════════════════════════════════"
        )
        .ok();
        writeln!(output).ok();

        writeln!(output, "SUMMARY").ok();
        writeln!(
            output,
            "───────────────────────────────────────────────────────────────"
        )
        .ok();
        writeln!(output, "  Total Models:     {}", self.summary.total_models).ok();
        writeln!(
            output,
            "  Frontier Models:  {}",
            self.summary.frontier_models
        )
        .ok();
        writeln!(
            output,
            "  Best Accuracy:    {:.2}%",
            self.summary.best_accuracy * 100.0
        )
        .ok();
        writeln!(
            output,
            "  Best Cost:        ${:.6}/1M tokens",
            self.summary.best_cost
        )
        .ok();
        writeln!(
            output,
            "  Best Latency:     {}ms",
            self.summary.best_latency_ms
        )
        .ok();
        if let Some(factor) = self.summary.slm_value_factor {
            writeln!(output, "  SLM Value Factor: {factor:.1}x").ok();
        }
        writeln!(
            output,
            "  SLM Dominant:     {}",
            if self.summary.slm_pareto_dominant {
                "YES"
            } else {
                "NO"
            }
        )
        .ok();
        writeln!(output).ok();

        // Results table
        writeln!(output, "MODEL RESULTS").ok();
        writeln!(
            output,
            "───────────────────────────────────────────────────────────────"
        )
        .ok();

        let rows: Vec<ResultTableRow> = self
            .model_results
            .iter()
            .map(|r| ResultTableRow {
                model: r.model_id.clone(),
                accuracy: format!("{:.2}%", r.accuracy.value * 100.0),
                cost: format!("${:.6}", r.cost),
                latency: format!("{}ms", r.latency_ms),
                frontier: if r.on_frontier { "✓" } else { "" }.to_string(),
                value: r
                    .value_score
                    .map_or_else(|| "-".to_string(), |v| format!("{v:.1}x")),
            })
            .collect();

        let table = Table::new(rows).to_string();
        writeln!(output, "{table}").ok();

        output
    }
}

#[cfg(test)]
#[allow(clippy::suboptimal_flops, clippy::cast_lossless, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_result(model: &str, accuracy: f64, cost: f64, latency_ms: u64) -> EvalResult {
        EvalResult {
            model_id: model.to_string(),
            task_id: "test-task".to_string(),
            accuracy,
            cost,
            latency: Duration::from_millis(latency_ms),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_report_builder_new() {
        let builder = ReportBuilder::new("test-task");
        assert_eq!(builder.task_id, "test-task");
    }

    #[test]
    fn test_report_builder_with_stat_config() {
        let config = StatConfig {
            bootstrap_n: 5000,
            confidence: 0.99,
            alpha: 0.01,
            seed: 123,
        };
        let builder = ReportBuilder::new("test").with_stat_config(config);
        assert_eq!(builder.stat_config.bootstrap_n, 5000);
    }

    #[test]
    fn test_report_builder_add_result() {
        let mut builder = ReportBuilder::new("test");
        let result = create_test_result("slm", 0.92, 0.001, 50);
        let samples = vec![0.91, 0.92, 0.93];

        builder.add_result(result, samples);

        assert_eq!(builder.results.len(), 1);
        assert_eq!(builder.accuracy_samples.len(), 1);
    }

    #[test]
    fn test_report_builder_build() {
        let mut builder = ReportBuilder::new("test-task");

        builder.add_result(
            create_test_result("slm", 0.92, 0.001, 50),
            (0..100).map(|i| 0.90 + (i as f64 * 0.0004)).collect(),
        );
        builder.add_result(
            create_test_result("baseline", 0.95, 10.0, 500),
            (0..100).map(|i| 0.93 + (i as f64 * 0.0004)).collect(),
        );

        let report = builder.build();

        assert_eq!(report.metadata.task_id, "test-task");
        assert_eq!(report.summary.total_models, 2);
        assert_eq!(report.model_results.len(), 2);
    }

    #[test]
    fn test_full_report_to_json() {
        let mut builder = ReportBuilder::new("test");
        builder.add_result(create_test_result("slm", 0.92, 0.001, 50), vec![0.92; 10]);

        let report = builder.build();
        let json = report.to_json();

        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("test"));
        assert!(json_str.contains("slm"));
    }

    #[test]
    fn test_full_report_to_markdown() {
        let mut builder = ReportBuilder::new("test");
        builder.add_result(create_test_result("slm", 0.92, 0.001, 50), vec![0.92; 10]);

        let report = builder.build();
        let markdown = report.to_markdown();

        assert!(markdown.contains("# Evaluation Report"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("slm"));
    }

    #[test]
    fn test_full_report_to_text() {
        let mut builder = ReportBuilder::new("test");
        builder.add_result(create_test_result("slm", 0.92, 0.001, 50), vec![0.92; 10]);

        let report = builder.build();
        let text = report.to_text();

        assert!(text.contains("SUMMARY"));
        assert!(text.contains("slm"));
    }

    #[test]
    fn test_report_summary_with_baselines() {
        let mut builder = ReportBuilder::new("test");
        builder.add_result(create_test_result("slm", 0.92, 0.001, 50), vec![0.92; 10]);
        builder.add_baselines(vec![BaselineEvalResult {
            model_id: "claude".to_string(),
            accuracy: 0.95,
            cost: 10.0,
            latency: Duration::from_millis(500),
        }]);

        let report = builder.build();

        assert!(report.summary.slm_value_factor.is_some());
        // SLM should show high value due to much lower cost/latency
        assert!(report.summary.slm_value_factor.unwrap() > 1.0);
    }

    #[test]
    fn test_report_pareto_analysis() {
        let mut builder = ReportBuilder::new("test");

        // SLM: lower accuracy but much better cost/latency
        builder.add_result(create_test_result("slm", 0.90, 0.001, 50), vec![0.90; 10]);
        // Baseline: higher accuracy but expensive
        builder.add_result(
            create_test_result("baseline", 0.95, 10.0, 500),
            vec![0.95; 10],
        );

        let report = builder.build();

        // Both should be on frontier (Pareto incomparable)
        assert_eq!(report.pareto_analysis.frontier_models.len(), 2);
    }

    #[test]
    fn test_stat_config_summary_from() {
        let config = StatConfig {
            bootstrap_n: 5000,
            confidence: 0.99,
            alpha: 0.01,
            seed: 42,
        };
        let summary = StatConfigSummary::from(&config);

        assert_eq!(summary.bootstrap_n, 5000);
        assert!((summary.confidence - 0.99).abs() < f64::EPSILON);
        assert!((summary.alpha - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_with_ci() {
        let metric = MetricWithCI {
            value: 0.92,
            ci_lower: 0.90,
            ci_upper: 0.94,
        };
        assert!(metric.ci_lower < metric.value);
        assert!(metric.value < metric.ci_upper);
    }

    #[test]
    fn test_statistical_comparison() {
        let mut builder = ReportBuilder::new("test");

        // Two models with different distributions
        builder.add_result(
            create_test_result("model_a", 0.92, 0.001, 50),
            (0..100).map(|i| 0.90 + (i as f64 * 0.0004)).collect(),
        );
        builder.add_result(
            create_test_result("model_b", 0.80, 0.01, 100),
            (0..100).map(|i| 0.78 + (i as f64 * 0.0004)).collect(),
        );

        let report = builder.build();

        // Should have at least one statistical comparison
        assert!(!report.statistical_tests.is_empty());

        // The comparison should show significant difference
        let comparison = &report.statistical_tests[0];
        assert!(comparison.significance.is_significant);
    }
}
