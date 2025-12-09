//! Pareto frontier analysis for multi-objective optimization.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::time::Duration;

/// Evaluation result for a single model on a single task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Model identifier
    pub model_id: String,
    /// Task identifier
    pub task_id: String,
    /// Accuracy score (higher is better)
    pub accuracy: f64,
    /// Cost per 1M tokens (lower is better)
    pub cost: f64,
    /// p99 latency (lower is better)
    pub latency: Duration,
    /// Additional metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl EvalResult {
    /// Check if self dominates other (better or equal in all, strictly better in one)
    ///
    /// A solution S₁ DOMINATES S₂ if and only if:
    /// - S₁ is no worse than S₂ in all objectives
    /// - S₁ is strictly better than S₂ in at least one objective
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        let dominated_accuracy = self.accuracy >= other.accuracy;
        let dominated_cost = self.cost <= other.cost;
        let dominated_latency = self.latency <= other.latency;

        let all_dominated = dominated_accuracy && dominated_cost && dominated_latency;
        let strictly_better = self.accuracy > other.accuracy
            || self.cost < other.cost
            || self.latency < other.latency;

        all_dominated && strictly_better
    }

    /// Calculate value score: `V = (1 - accuracy_gap) * cost_ratio * latency_ratio`
    /// Relative to a baseline result
    #[must_use]
    pub fn value_score(&self, baseline: &Self) -> f64 {
        let accuracy_gap = baseline.accuracy - self.accuracy;
        let cost_ratio = if self.cost > 0.0 {
            baseline.cost / self.cost
        } else {
            f64::MAX
        };
        let latency_ratio = if self.latency.is_zero() {
            f64::MAX
        } else {
            baseline.latency.as_secs_f64() / self.latency.as_secs_f64()
        };

        (1.0 - accuracy_gap) * cost_ratio * latency_ratio
    }
}

/// Compute Pareto frontier from evaluation results
///
/// The Pareto frontier is the set of all non-dominated solutions.
#[must_use]
pub fn compute_pareto_frontier(results: &[EvalResult]) -> Vec<&EvalResult> {
    let mut frontier = Vec::new();

    for candidate in results {
        let is_dominated = results.iter().any(|other| other.dominates(candidate));

        if !is_dominated {
            frontier.push(candidate);
        }
    }

    // Sort by primary objective (accuracy) descending
    frontier.sort_by(|a, b| {
        b.accuracy
            .partial_cmp(&a.accuracy)
            .unwrap_or(Ordering::Equal)
    });

    frontier
}

/// Pareto frontier analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoAnalysis {
    /// Models on the Pareto frontier
    pub frontier_models: Vec<String>,
    /// Models dominated by others
    pub dominated_models: Vec<String>,
    /// Trade-off analysis for each model
    pub trade_offs: Vec<TradeOffAnalysis>,
}

/// Trade-off analysis for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOffAnalysis {
    /// Model identifier
    pub model_id: String,
    /// Is on Pareto frontier
    pub on_frontier: bool,
    /// Accuracy gap from best frontier model
    pub accuracy_gap: f64,
    /// Cost ratio compared to frontier
    pub cost_ratio: f64,
    /// Latency ratio compared to frontier
    pub latency_ratio: f64,
    /// Computed value score
    pub value_score: f64,
}

/// Analyze Pareto frontier and compute trade-offs
#[must_use]
pub fn analyze_pareto(results: &[EvalResult]) -> ParetoAnalysis {
    let frontier = compute_pareto_frontier(results);
    let frontier_ids: std::collections::HashSet<_> =
        frontier.iter().map(|r| r.model_id.clone()).collect();

    // Find best frontier model by accuracy
    let best_frontier = frontier.iter().max_by(|a, b| {
        a.accuracy
            .partial_cmp(&b.accuracy)
            .unwrap_or(Ordering::Equal)
    });

    let trade_offs: Vec<_> = results
        .iter()
        .map(|result| {
            let on_frontier = frontier_ids.contains(&result.model_id);
            let (accuracy_gap, cost_ratio, latency_ratio, value_score) =
                best_frontier.map_or((0.0, 1.0, 1.0, 1.0), |best| {
                    (
                        best.accuracy - result.accuracy,
                        if result.cost > 0.0 {
                            best.cost / result.cost
                        } else {
                            0.0
                        },
                        if result.latency.is_zero() {
                            0.0
                        } else {
                            best.latency.as_secs_f64() / result.latency.as_secs_f64()
                        },
                        result.value_score(best),
                    )
                });

            TradeOffAnalysis {
                model_id: result.model_id.clone(),
                on_frontier,
                accuracy_gap,
                cost_ratio,
                latency_ratio,
                value_score,
            }
        })
        .collect();

    let dominated_models: Vec<_> = results
        .iter()
        .filter(|r| !frontier_ids.contains(&r.model_id))
        .map(|r| r.model_id.clone())
        .collect();

    ParetoAnalysis {
        frontier_models: frontier.iter().map(|r| r.model_id.clone()).collect(),
        dominated_models,
        trade_offs,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn create_result(model: &str, accuracy: f64, cost: f64, latency_ms: u64) -> EvalResult {
        EvalResult {
            model_id: model.to_string(),
            task_id: "test".to_string(),
            accuracy,
            cost,
            latency: Duration::from_millis(latency_ms),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_dominance_strict_better() {
        let a = create_result("a", 0.95, 1.0, 100);
        let b = create_result("b", 0.90, 2.0, 200);

        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_dominance_equal() {
        let a = create_result("a", 0.95, 1.0, 100);
        let b = create_result("b", 0.95, 1.0, 100);

        // Neither dominates when equal
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_dominance_pareto_incomparable() {
        // a: better accuracy, worse cost
        let a = create_result("a", 0.95, 2.0, 100);
        // b: worse accuracy, better cost
        let b = create_result("b", 0.90, 1.0, 100);

        // Neither dominates (Pareto incomparable)
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_pareto_frontier() {
        let results = vec![
            create_result("frontier_high_acc", 0.95, 10.0, 1000),
            create_result("frontier_low_cost", 0.85, 0.1, 50),
            create_result("dominated", 0.80, 5.0, 500),
        ];

        let frontier = compute_pareto_frontier(&results);

        assert_eq!(frontier.len(), 2);
        assert!(frontier.iter().any(|r| r.model_id == "frontier_high_acc"));
        assert!(frontier.iter().any(|r| r.model_id == "frontier_low_cost"));
        assert!(!frontier.iter().any(|r| r.model_id == "dominated"));
    }

    #[test]
    fn test_value_score() {
        let baseline = create_result("baseline", 0.95, 10.0, 1000);
        let slm = create_result("slm", 0.92, 0.1, 50);

        let value = slm.value_score(&baseline);

        // SLM should have high value (close accuracy, much better cost/latency)
        assert!(value > 100.0, "Value score should be >100x: {value}");
    }

    #[test]
    fn test_pareto_analysis() {
        let results = vec![
            create_result("claude", 0.95, 15.0, 1000),
            create_result("gemini", 0.93, 5.0, 800),
            create_result("slm", 0.90, 0.01, 50),
            create_result("bad_model", 0.70, 10.0, 2000),
        ];

        let analysis = analyze_pareto(&results);

        // All except bad_model should be on frontier
        assert_eq!(analysis.frontier_models.len(), 3);
        assert_eq!(analysis.dominated_models.len(), 1);
        assert!(analysis.dominated_models.contains(&"bad_model".to_string()));

        // SLM should have best value score
        let slm_trade_off = analysis.trade_offs.iter().find(|t| t.model_id == "slm");
        assert!(slm_trade_off.is_some(), "SLM should be in trade-offs");
        assert!(slm_trade_off.is_some_and(|t| t.value_score > 10.0));
    }
}
