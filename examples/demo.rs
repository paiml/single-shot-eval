//! Demo: single-shot-eval library in action
use single_shot_eval::{
    baselines::available_baselines,
    metrics::{bootstrap_ci, paired_t_test, StatConfig},
    pareto::{analyze_pareto, compute_pareto_frontier, EvalResult},
    report::ReportBuilder,
};
use std::collections::HashMap;
use std::time::Duration;

#[allow(clippy::too_many_lines)]
fn main() {
    println!("=== Single-Shot Eval Demo ===\n");

    // 1. Create evaluation results (simulating SLM vs frontier models)
    let results = vec![
        EvalResult {
            model_id: "slm-100m".to_string(),
            task_id: "sentiment".to_string(),
            accuracy: 0.92,
            cost: 0.0001, // $0.0001 per 1K tokens
            latency: Duration::from_millis(15),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "claude-haiku".to_string(),
            task_id: "sentiment".to_string(),
            accuracy: 0.95,
            cost: 0.25, // $0.25 per 1K tokens
            latency: Duration::from_millis(800),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "gemini-flash".to_string(),
            task_id: "sentiment".to_string(),
            accuracy: 0.93,
            cost: 0.075,
            latency: Duration::from_millis(400),
            metadata: HashMap::new(),
        },
        EvalResult {
            model_id: "gpt-4o-mini".to_string(),
            task_id: "sentiment".to_string(),
            accuracy: 0.94,
            cost: 0.15,
            latency: Duration::from_millis(600),
            metadata: HashMap::new(),
        },
    ];

    // 2. Compute Pareto frontier
    println!("📊 Computing Pareto Frontier...\n");
    let frontier = compute_pareto_frontier(&results);

    println!("Pareto-optimal models:");
    for model in &frontier {
        println!(
            "  ◆ {} (acc: {:.1}%, cost: ${:.4}, lat: {:?})",
            model.model_id,
            model.accuracy * 100.0,
            model.cost,
            model.latency
        );
    }

    // 3. Full Pareto analysis with trade-offs
    println!("\n📈 Trade-off Analysis...\n");
    let analysis = analyze_pareto(&results);

    println!(
        "Frontier size: {} / {} models",
        analysis.frontier_models.len(),
        results.len()
    );
    println!("Dominated: {} models", analysis.dominated_models.len());
    for tradeoff in &analysis.trade_offs {
        println!("  {} trade-off:", tradeoff.model_id);
        println!("    Accuracy gap: {:.1}%", tradeoff.accuracy_gap * 100.0);
        println!("    Cost ratio: {:.0}x cheaper", tradeoff.cost_ratio);
        println!("    Latency ratio: {:.0}x faster", tradeoff.latency_ratio);
        println!("    Value score: {:.1}x", tradeoff.value_score);
    }

    // 4. Statistical analysis with bootstrap CI
    println!("\n📉 Statistical Analysis...\n");
    let slm_accuracies: Vec<f64> = (0..100)
        .map(|_| rand::random::<f64>().mul_add(0.04, 0.90))
        .collect();
    let frontier_accuracies: Vec<f64> = (0..100)
        .map(|_| rand::random::<f64>().mul_add(0.04, 0.93))
        .collect();

    let config = StatConfig::default();
    let (lower, upper) = bootstrap_ci(&slm_accuracies, &config);
    let mean: f64 = slm_accuracies.iter().sum::<f64>() / slm_accuracies.len() as f64;
    println!(
        "SLM accuracy: {:.2}% [95% CI: {:.2}% - {:.2}%]",
        mean * 100.0,
        lower * 100.0,
        upper * 100.0
    );

    // Paired t-test
    if let Some(sig) = paired_t_test(&slm_accuracies, &frontier_accuracies, config.alpha) {
        println!("Paired t-test vs frontier:");
        println!("  t-statistic: {:.3}", sig.t_statistic);
        println!("  p-value: {:.4}", sig.p_value);
        println!(
            "  Cohen's d: {:.3} ({})",
            sig.cohens_d,
            if sig.cohens_d.abs() < 0.2 {
                "negligible"
            } else if sig.cohens_d.abs() < 0.5 {
                "small"
            } else if sig.cohens_d.abs() < 0.8 {
                "medium"
            } else {
                "large"
            }
        );
        println!(
            "  Significant: {}",
            if sig.is_significant { "YES" } else { "NO" }
        );
    }

    // 5. Check available baselines
    println!("\n🔌 Available CLI Baselines...\n");
    let baselines = available_baselines();
    if baselines.is_empty() {
        println!("  (none installed - evaluation runs offline)");
    } else {
        for b in baselines {
            println!("  ✓ {b}");
        }
    }

    // 6. Generate report
    println!("\n📝 Generating Report...\n");
    let mut builder = ReportBuilder::new("sentiment");
    for result in &results {
        // Generate simulated accuracy samples for the result
        let samples: Vec<f64> = (0..50)
            .map(|_| rand::random::<f64>().mul_add(0.04, result.accuracy - 0.02))
            .collect();
        builder.add_result(result.clone(), samples);
    }
    let report = builder.build();

    println!("Report Summary:");
    println!("  Total Models: {}", report.summary.total_models);
    println!("  Frontier Models: {}", report.summary.frontier_models);
    println!(
        "  Best Accuracy: {:.1}%",
        report.summary.best_accuracy * 100.0
    );
    println!("  Best Cost: ${:.4}/1K tokens", report.summary.best_cost);
    println!("  Best Latency: {}ms", report.summary.best_latency_ms);

    // Show markdown output (first 800 chars)
    let md = report.to_markdown();
    println!("\n--- Markdown Report Preview ---\n");
    println!("{}", &md[..md.len().min(800)]);
    println!("...\n");

    println!("✅ Demo complete!");
}
