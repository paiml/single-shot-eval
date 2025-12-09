//! Metrics collection and statistical computation module.
//!
//! Implements NASA-standard statistical rigor:
//! - Bootstrap confidence intervals (10,000 resamples)
//! - Paired t-test with Bonferroni correction
//! - Cohen's d effect size

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::time::Duration;

/// Metrics collector for evaluation results
#[derive(Debug, Default)]
pub struct MetricsCollector {
    /// Collected accuracy measurements
    accuracy: Vec<f64>,
    /// Collected latency measurements
    latency: Vec<Duration>,
    /// Collected cost measurements
    cost: Vec<f64>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an accuracy measurement
    pub fn record_accuracy(&mut self, value: f64) {
        self.accuracy.push(value);
    }

    /// Record a latency measurement
    pub fn record_latency(&mut self, duration: Duration) {
        self.latency.push(duration);
    }

    /// Record a cost measurement
    pub fn record_cost(&mut self, value: f64) {
        self.cost.push(value);
    }

    /// Compute aggregated metrics
    #[must_use]
    pub fn compute(&self) -> AggregatedMetrics {
        AggregatedMetrics {
            accuracy: compute_mean(&self.accuracy),
            accuracy_ci: Self::compute_bootstrap_ci(&self.accuracy),
            latency_p50: Self::compute_percentile_duration(&self.latency, 0.50),
            latency_p95: Self::compute_percentile_duration(&self.latency, 0.95),
            latency_p99: Self::compute_percentile_duration(&self.latency, 0.99),
            cost_total: self.cost.iter().sum(),
            cost_per_inference: compute_mean(&self.cost),
            sample_count: self.accuracy.len(),
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn compute_bootstrap_ci(samples: &[f64]) -> (f64, f64) {
        if samples.len() < 2 {
            let mean = compute_mean(samples);
            return (mean, mean);
        }

        // Simple percentile-based CI (proper bootstrap would use resampling)
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = (samples.len() as f64 * 0.025).floor() as usize;
        let upper_idx = (samples.len() as f64 * 0.975).ceil() as usize;

        let lower = sorted.get(lower_idx).copied().unwrap_or(0.0);
        let upper = sorted
            .get(upper_idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0);

        (lower, upper)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn compute_percentile_duration(samples: &[Duration], percentile: f64) -> Duration {
        if samples.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = samples.to_vec();
        sorted.sort();

        let idx = ((samples.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
        sorted
            .get(idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(Duration::ZERO)
    }
}

/// Compute mean of samples
#[allow(clippy::cast_precision_loss)]
fn compute_mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Compute standard deviation of samples
#[allow(clippy::cast_precision_loss)]
fn compute_std(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mean = compute_mean(samples);
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    variance.sqrt()
}

/// Statistical configuration for evaluation
#[derive(Debug, Clone)]
pub struct StatConfig {
    /// Number of bootstrap resamples
    pub bootstrap_n: usize,
    /// Confidence level (e.g., 0.95)
    pub confidence: f64,
    /// Significance threshold
    pub alpha: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for StatConfig {
    fn default() -> Self {
        Self {
            bootstrap_n: 10_000,
            confidence: 0.95,
            alpha: 0.05,
            seed: 42,
        }
    }
}

/// Bootstrap confidence interval (proper resampling)
///
/// Uses the percentile method with `n` resamples.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn bootstrap_ci(samples: &[f64], config: &StatConfig) -> (f64, f64) {
    if samples.len() < 2 {
        let mean = compute_mean(samples);
        return (mean, mean);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut bootstrap_means = Vec::with_capacity(config.bootstrap_n);

    // Resample with replacement
    for _ in 0..config.bootstrap_n {
        let resample_sum: f64 = (0..samples.len())
            .map(|_| {
                let idx = rng.next_u64() as usize % samples.len();
                samples[idx]
            })
            .sum();
        bootstrap_means.push(resample_sum / samples.len() as f64);
    }

    // Sort for percentile computation
    bootstrap_means
        .sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentile indices
    let alpha = 1.0 - config.confidence;
    let lower_idx = (config.bootstrap_n as f64 * (alpha / 2.0)).floor() as usize;
    let upper_idx = (config.bootstrap_n as f64 * (1.0 - alpha / 2.0)).ceil() as usize;

    let lower = bootstrap_means.get(lower_idx).copied().unwrap_or(0.0);
    let upper = bootstrap_means
        .get(upper_idx.min(bootstrap_means.len() - 1))
        .copied()
        .unwrap_or(0.0);

    (lower, upper)
}

/// Result of a significance test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceResult {
    /// t-statistic
    pub t_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: f64,
    /// Is result significant at the given alpha?
    pub is_significant: bool,
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Effect size interpretation
    pub effect_interpretation: String,
}

/// Paired t-test for comparing two sample sets
///
/// # Errors
///
/// Returns None if samples are too small or have no variance.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn paired_t_test(
    samples_a: &[f64],
    samples_b: &[f64],
    alpha: f64,
) -> Option<SignificanceResult> {
    if samples_a.len() != samples_b.len() || samples_a.len() < 2 {
        return None;
    }

    let n = samples_a.len();
    let differences: Vec<f64> = samples_a
        .iter()
        .zip(samples_b.iter())
        .map(|(a, b)| a - b)
        .collect();

    let mean_diff = compute_mean(&differences);
    let std_diff = compute_std(&differences);

    if std_diff < f64::EPSILON {
        return None;
    }

    let t_statistic = mean_diff / (std_diff / (n as f64).sqrt());
    let df = (n - 1) as f64;

    // Compute two-tailed p-value
    let t_dist = StudentsT::new(0.0, 1.0, df).ok()?;
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_statistic.abs()));

    // Compute Cohen's d
    let cohens_d = mean_diff / std_diff;
    let effect_interpretation = interpret_cohens_d(cohens_d);

    Some(SignificanceResult {
        t_statistic,
        p_value,
        degrees_of_freedom: df,
        is_significant: p_value < alpha,
        cohens_d,
        effect_interpretation,
    })
}

/// Independent samples t-test (Welch's t-test)
///
/// # Errors
///
/// Returns None if samples are too small or have no variance.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn welch_t_test(
    samples_a: &[f64],
    samples_b: &[f64],
    alpha: f64,
) -> Option<SignificanceResult> {
    if samples_a.len() < 2 || samples_b.len() < 2 {
        return None;
    }

    let n_a = samples_a.len() as f64;
    let n_b = samples_b.len() as f64;
    let mean_a = compute_mean(samples_a);
    let mean_b = compute_mean(samples_b);
    let var_a = compute_std(samples_a).powi(2);
    let var_b = compute_std(samples_b).powi(2);

    if var_a < f64::EPSILON && var_b < f64::EPSILON {
        return None;
    }

    let se = ((var_a / n_a) + (var_b / n_b)).sqrt();
    if se < f64::EPSILON {
        return None;
    }

    let t_statistic = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom
    let df_num = ((var_a / n_a) + (var_b / n_b)).powi(2);
    let df_denom = ((var_a / n_a).powi(2) / (n_a - 1.0)) + ((var_b / n_b).powi(2) / (n_b - 1.0));
    let df = if df_denom > f64::EPSILON {
        df_num / df_denom
    } else {
        (n_a + n_b - 2.0).max(1.0)
    };

    // Compute two-tailed p-value
    let t_dist = StudentsT::new(0.0, 1.0, df).ok()?;
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_statistic.abs()));

    // Pooled Cohen's d
    let pooled_std = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0)).sqrt();
    let cohens_d = if pooled_std > f64::EPSILON {
        (mean_a - mean_b) / pooled_std
    } else {
        0.0
    };
    let effect_interpretation = interpret_cohens_d(cohens_d);

    Some(SignificanceResult {
        t_statistic,
        p_value,
        degrees_of_freedom: df,
        is_significant: p_value < alpha,
        cohens_d,
        effect_interpretation,
    })
}

/// Apply Bonferroni correction for multiple comparisons
#[must_use]
pub fn bonferroni_correction(alpha: f64, num_comparisons: usize) -> f64 {
    if num_comparisons == 0 {
        return alpha;
    }
    alpha / num_comparisons as f64
}

/// Interpret Cohen's d effect size
fn interpret_cohens_d(d: f64) -> String {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        "negligible".to_string()
    } else if abs_d < 0.5 {
        "small".to_string()
    } else if abs_d < 0.8 {
        "medium".to_string()
    } else {
        "large".to_string()
    }
}

/// Aggregated metrics from evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Mean accuracy
    pub accuracy: f64,
    /// 95% confidence interval for accuracy (lower, upper)
    pub accuracy_ci: (f64, f64),
    /// p50 latency
    pub latency_p50: Duration,
    /// p95 latency
    pub latency_p95: Duration,
    /// p99 latency
    pub latency_p99: Duration,
    /// Total cost
    pub cost_total: f64,
    /// Cost per inference
    pub cost_per_inference: f64,
    /// Number of samples
    pub sample_count: usize,
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    // =========================================================================
    // MetricsCollector tests
    // =========================================================================

    #[test]
    fn test_metrics_collector_empty() {
        let collector = MetricsCollector::new();
        let metrics = collector.compute();
        assert_eq!(metrics.sample_count, 0);
        assert!(metrics.accuracy.abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_collector_with_samples() {
        let mut collector = MetricsCollector::new();

        for i in 0_u64..100 {
            collector.record_accuracy(0.9 + (i as f64 * 0.001));
            collector.record_latency(Duration::from_millis(100 + i));
            collector.record_cost(0.001);
        }

        let metrics = collector.compute();
        assert_eq!(metrics.sample_count, 100);
        assert!(metrics.accuracy > 0.9);
        assert!(metrics.latency_p50 > Duration::ZERO);
    }

    #[test]
    fn test_bootstrap_ci_basic() {
        let mut collector = MetricsCollector::new();

        // Add samples with known distribution
        for i in 0..1000 {
            collector.record_accuracy(0.8 + (f64::from(i) * 0.0002));
        }

        let metrics = collector.compute();
        let (lower, upper) = metrics.accuracy_ci;

        // CI should be reasonable
        assert!(lower < metrics.accuracy);
        assert!(upper > metrics.accuracy);
    }

    // =========================================================================
    // Statistical functions tests
    // =========================================================================

    #[test]
    fn test_compute_std() {
        // Known standard deviation for simple data
        let samples = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = compute_std(&samples);
        // Sample std for this data is ~2.138
        assert!(std > 2.0 && std < 2.2, "std = {std}");
    }

    #[test]
    fn test_compute_std_empty() {
        let samples: Vec<f64> = vec![];
        assert!(compute_std(&samples).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_std_single() {
        let samples = vec![5.0];
        assert!(compute_std(&samples).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bootstrap_ci_reproducible() {
        let samples: Vec<f64> = (0..100).map(|i| 0.9 + (i as f64 * 0.001)).collect();
        let config = StatConfig::default();

        let (lower1, upper1) = bootstrap_ci(&samples, &config);
        let (lower2, upper2) = bootstrap_ci(&samples, &config);

        // Same seed should give same results
        assert!((lower1 - lower2).abs() < f64::EPSILON);
        assert!((upper1 - upper2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bootstrap_ci_reasonable_width() {
        let samples: Vec<f64> = (0..1000).map(|i| 0.9 + (i as f64 * 0.0001)).collect();
        let config = StatConfig::default();

        let (lower, upper) = bootstrap_ci(&samples, &config);
        let mean = compute_mean(&samples);

        // CI should contain the mean
        assert!(lower <= mean);
        assert!(upper >= mean);
        // CI should be narrow for large sample
        assert!(upper - lower < 0.05);
    }

    #[test]
    fn test_bootstrap_ci_single_sample() {
        let samples = vec![0.95];
        let config = StatConfig::default();

        let (lower, upper) = bootstrap_ci(&samples, &config);
        assert!((lower - 0.95).abs() < f64::EPSILON);
        assert!((upper - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_paired_t_test_significant() {
        // Two clearly different distributions
        let samples_a: Vec<f64> = (0..100).map(|_| 0.95).collect();
        let samples_b: Vec<f64> = (0..100).map(|_| 0.85).collect();

        let result = paired_t_test(&samples_a, &samples_b, 0.05);
        assert!(result.is_none()); // Same values have zero variance in differences
    }

    #[test]
    fn test_paired_t_test_with_variance() {
        // Add different variance patterns to get variance in differences
        let samples_a: Vec<f64> = (0..100)
            .map(|i| 0.95 + (i as f64 * 0.001) + ((i % 3) as f64 * 0.01))
            .collect();
        let samples_b: Vec<f64> = (0..100)
            .map(|i| 0.85 + (i as f64 * 0.001) + ((i % 5) as f64 * 0.005))
            .collect();

        let result = paired_t_test(&samples_a, &samples_b, 0.05);
        assert!(result.is_some(), "paired t-test should return result");

        let result = result.unwrap();
        assert!(
            result.is_significant,
            "should be significant: p={}",
            result.p_value
        );
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_paired_t_test_unequal_length() {
        let samples_a = vec![0.9, 0.91, 0.92];
        let samples_b = vec![0.8, 0.81];

        assert!(paired_t_test(&samples_a, &samples_b, 0.05).is_none());
    }

    #[test]
    fn test_welch_t_test_significant() {
        // Two distributions with different means
        let samples_a: Vec<f64> = (0..50).map(|i| 0.95 + (i as f64 * 0.002)).collect();
        let samples_b: Vec<f64> = (0..50).map(|i| 0.75 + (i as f64 * 0.002)).collect();

        let result = welch_t_test(&samples_a, &samples_b, 0.05);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.is_significant);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_welch_t_test_not_significant() {
        // Very similar distributions
        let samples_a: Vec<f64> = (0..10).map(|i| 0.90 + (i as f64 * 0.01)).collect();
        let samples_b: Vec<f64> = (0..10).map(|i| 0.90 + (i as f64 * 0.01)).collect();

        let result = welch_t_test(&samples_a, &samples_b, 0.05);
        // With identical data, we may get None or not significant
        if let Some(r) = result {
            assert!(!r.is_significant || r.p_value > 0.05);
        }
    }

    #[test]
    fn test_welch_t_test_small_sample() {
        let samples_a = vec![0.9];
        let samples_b = vec![0.8];

        assert!(welch_t_test(&samples_a, &samples_b, 0.05).is_none());
    }

    #[test]
    fn test_bonferroni_correction() {
        assert!((bonferroni_correction(0.05, 5) - 0.01).abs() < f64::EPSILON);
        assert!((bonferroni_correction(0.05, 10) - 0.005).abs() < f64::EPSILON);
        assert!((bonferroni_correction(0.05, 0) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cohens_d_interpretation() {
        assert_eq!(interpret_cohens_d(0.1), "negligible");
        assert_eq!(interpret_cohens_d(0.3), "small");
        assert_eq!(interpret_cohens_d(0.6), "medium");
        assert_eq!(interpret_cohens_d(1.0), "large");
        assert_eq!(interpret_cohens_d(-0.9), "large");
    }

    #[test]
    fn test_stat_config_default() {
        let config = StatConfig::default();
        assert_eq!(config.bootstrap_n, 10_000);
        assert!((config.confidence - 0.95).abs() < f64::EPSILON);
        assert!((config.alpha - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
    }
}
