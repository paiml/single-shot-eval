//! Benchmarks for Pareto frontier computation

#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use single_shot_eval::pareto::{compute_pareto_frontier, EvalResult};
use std::time::Duration;

fn create_results(n: usize) -> Vec<EvalResult> {
    (0..n)
        .map(|i| {
            let i_f = i as f64;
            let n_f = n as f64;
            let accuracy = (i_f / n_f).mul_add(0.25, 0.7);
            let cost = (i_f / n_f).mul_add(-9.9, 10.0);
            let latency = 1000 - (i * 900 / n);

            EvalResult {
                model_id: format!("model_{i}"),
                task_id: "benchmark".to_string(),
                accuracy,
                cost,
                latency: Duration::from_millis(latency as u64),
                metadata: std::collections::HashMap::new(),
            }
        })
        .collect()
}

fn benchmark_pareto_frontier(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_frontier");

    for size in &[10, 50, 100, 500] {
        let results = create_results(*size);

        group.bench_function(format!("compute_{size}_models"), |b| {
            b.iter(|| compute_pareto_frontier(black_box(&results)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_pareto_frontier);
criterion_main!(benches);
