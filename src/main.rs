//! Single-Shot Eval CLI
//!
//! SLM Pareto Frontier Evaluation Framework

use clap::{Parser, Subcommand};
use single_shot_eval::{
    available_baselines, analyze_pareto, classify_example_level, CompilerVerifier, Corpus,
    Difficulty, EvalResult, Py2RsLevel, ReportBuilder, RunnerConfig, TaskLoader, TaskRunner,
};
use std::collections::HashMap;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "single-shot-eval")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run evaluation suite
    Evaluate {
        /// Task configuration files (glob pattern)
        #[arg(long, default_value = "tasks/*.yaml")]
        tasks: String,

        /// Model files to evaluate (glob pattern)
        #[arg(long, default_value = "models/*.apr")]
        models: String,

        /// Path to Python corpus (reprorusted-python-cli)
        #[arg(long)]
        corpus: Option<String>,

        /// Baseline models to compare against
        #[arg(long, value_delimiter = ',')]
        baselines: Vec<String>,

        /// Output directory for results
        #[arg(long)]
        output: Option<String>,

        /// Number of samples per task (overrides task config)
        #[arg(long)]
        samples: Option<usize>,

        /// Number of evaluation runs (Princeton methodology requires 5+)
        #[arg(long, default_value = "5")]
        runs: usize,
    },

    /// Generate Pareto frontier report
    Report {
        /// Input results directory
        #[arg(long)]
        input: String,

        /// Output report file
        #[arg(long)]
        output: String,
    },

    /// Analyze models with Depyler
    Analyze {
        /// Model files to analyze
        #[arg(long)]
        models: String,

        /// Output report file
        #[arg(long)]
        output: Option<String>,
    },

    /// Verify Rust code compiles and passes tests
    Verify {
        /// Path to Rust source file
        #[arg(long)]
        source: String,

        /// Path to test file (optional)
        #[arg(long)]
        tests: Option<String>,
    },

    /// Show corpus statistics
    CorpusStats {
        /// Path to corpus directory
        #[arg(long)]
        path: String,
    },

    /// Analyze corpus with `Py2Rs` 10-level benchmark classification
    Benchmark {
        /// Path to corpus directory or JSONL file
        #[arg(long)]
        corpus: String,

        /// Show detailed per-example classification
        #[arg(long)]
        verbose: bool,
    },
}

#[allow(clippy::too_many_lines)]
fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    if cli.verbose {
        tracing::info!("Verbose mode enabled");
    }

    match cli.command {
        Commands::Evaluate {
            tasks,
            models,
            corpus,
            baselines,
            output,
            samples,
            runs,
        } => {
            tracing::info!(
                tasks = %tasks,
                models = %models,
                corpus = ?corpus,
                baselines = ?baselines,
                output = ?output,
                samples = ?samples,
                runs = runs,
                "Starting evaluation (Princeton methodology: {runs} runs)"
            );

            // Load corpus if provided
            #[allow(clippy::option_if_let_else)]
            let corpus_data = if let Some(corpus_path) = &corpus {
                match Corpus::load(corpus_path) {
                    Ok(c) => {
                        let stats = c.stats();
                        println!("Loaded corpus: {} examples, {} with tests",
                            stats.total_examples, stats.examples_with_tests);
                        Some(c)
                    }
                    Err(e) => {
                        eprintln!("Failed to load corpus: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                None
            };

            // Load task configurations
            let task_loader = match TaskLoader::load_glob(&tasks) {
                Ok(loader) => {
                    if loader.is_empty() {
                        eprintln!("No task configurations found matching: {tasks}");
                        std::process::exit(1);
                    }
                    println!("Loaded {} task configuration(s)", loader.len());
                    loader
                }
                Err(e) => {
                    eprintln!("Failed to load tasks: {e}");
                    std::process::exit(1);
                }
            };

            // Find model files
            let model_paths: Vec<_> = match glob::glob(&models) {
                Ok(paths) => paths.filter_map(Result::ok).collect(),
                Err(e) => {
                    eprintln!("Invalid model glob pattern: {e}");
                    std::process::exit(1);
                }
            };

            if model_paths.is_empty() {
                eprintln!("No model files found matching: {models}");
                eprintln!("Note: Models must be .apr format (Aprender native format)");
                std::process::exit(1);
            }
            println!("Found {} model file(s)", model_paths.len());

            // Configure runner
            let runner_config = RunnerConfig {
                run_baselines: !baselines.is_empty() || !available_baselines().is_empty(),
                ..RunnerConfig::default()
            };
            let runner = TaskRunner::with_config(runner_config);

            // Run evaluations (Princeton methodology: multiple runs)
            println!("\nStarting evaluation ({runs} runs per Princeton methodology)...");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let mut all_results: Vec<EvalResult> = Vec::new();

            for task_config in task_loader.iter() {
                println!("\nTask: {} ({})", task_config.task.id, task_config.task.domain);

                for model_path in &model_paths {
                    let model_id = model_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");

                    // Validate model format
                    if let Err(e) = runner.validate_model(model_path) {
                        eprintln!("  Skipping {model_id}: {e}");
                        continue;
                    }

                    print!("  Evaluating {model_id}... ");

                    // Run multiple iterations for statistical validity
                    let mut run_results = Vec::new();
                    for _run in 0..runs {
                        match runner.run_evaluation(task_config, model_id) {
                            Ok(result) => run_results.push(result),
                            Err(e) => eprintln!("run failed: {e}"),
                        }
                    }

                    if run_results.is_empty() {
                        println!("FAILED (no successful runs)");
                        continue;
                    }

                    // Aggregate results (use last for now, could average)
                    if let Some(result) = run_results.pop() {
                        println!(
                            "accuracy={:.2}%, cost=${:.6}/1M, latency={:?}",
                            result.accuracy * 100.0,
                            result.cost,
                            result.latency
                        );
                        all_results.push(result);
                    }
                }
            }

            if all_results.is_empty() {
                eprintln!("\nNo successful evaluations completed");
                std::process::exit(1);
            }

            // Pareto analysis
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("PARETO FRONTIER ANALYSIS");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let pareto = analyze_pareto(&all_results);
            println!("\nFrontier models: {}", pareto.frontier_models.join(", "));
            if !pareto.dominated_models.is_empty() {
                println!("Dominated models: {}", pareto.dominated_models.join(", "));
            }

            println!("\nTrade-off analysis:");
            for trade_off in &pareto.trade_offs {
                let frontier_marker = if trade_off.on_frontier { "✓" } else { " " };
                println!(
                    "  {} {} - accuracy_gap: {:.2}%, cost_ratio: {:.1}x, value: {:.1}x",
                    frontier_marker,
                    trade_off.model_id,
                    trade_off.accuracy_gap * 100.0,
                    trade_off.cost_ratio,
                    trade_off.value_score
                );
            }

            // Generate report if output specified
            if let Some(output_path) = &output {
                let mut builder = ReportBuilder::new("evaluation");
                for result in &all_results {
                    // Use repeated accuracy as placeholder samples (real impl would collect actual samples)
                    builder.add_result(result.clone(), vec![result.accuracy; 100]);
                }

                let report = builder.build();
                let is_json = std::path::Path::new(output_path)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
                let output_str = if is_json {
                    report.to_json().unwrap_or_else(|e| format!("JSON error: {e}"))
                } else {
                    report.to_markdown()
                };

                match std::fs::write(output_path, &output_str) {
                    Ok(()) => println!("\nReport written to: {output_path}"),
                    Err(e) => eprintln!("\nFailed to write report: {e}"),
                }
            }

            // Print corpus info if loaded
            if let Some(c) = corpus_data {
                println!("\nCorpus used: {} examples", c.len());
            }

            println!("\nEvaluation complete ({runs} runs per model)");
        }
        Commands::Report { input, output } => {
            tracing::info!(
                input = %input,
                output = %output,
                "Generating report"
            );
            println!("Report generation not yet implemented");
        }
        Commands::Analyze { models, output } => {
            tracing::info!(
                models = %models,
                output = ?output,
                "Analyzing models"
            );
            println!("Analysis not yet implemented");
        }
        Commands::Verify { source, tests } => {
            tracing::info!(
                source = %source,
                tests = ?tests,
                "Verifying Rust code"
            );

            let rust_code = match std::fs::read_to_string(&source) {
                Ok(code) => code,
                Err(e) => {
                    eprintln!("Failed to read source file: {e}");
                    std::process::exit(1);
                }
            };

            let test_code = tests.as_ref().and_then(|p| std::fs::read_to_string(p).ok());

            let verifier = CompilerVerifier::new();
            match verifier.verify(&rust_code, test_code.as_deref()) {
                Ok(result) => {
                    println!("Compilation: {}", if result.compiles { "PASS" } else { "FAIL" });
                    if let Some(tests_pass) = result.tests_pass {
                        println!("Tests: {}", if tests_pass { "PASS" } else { "FAIL" });
                        println!("Tests run: {}, passed: {}", result.tests_run, result.tests_passed);
                    }
                    println!("Build time: {:?}", result.build_time);
                    if let Some(test_time) = result.test_time {
                        println!("Test time: {test_time:?}");
                    }
                    println!("\nGround truth: {}", if result.passes() { "PASS" } else { "FAIL" });

                    if !result.passes() {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Verification error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::CorpusStats { path } => {
            tracing::info!(
                path = %path,
                "Loading corpus statistics"
            );

            match Corpus::load(&path) {
                Ok(corpus) => {
                    let stats = corpus.stats();
                    println!("Corpus Statistics");
                    println!("=================");
                    println!("Path: {path}");
                    println!("Total examples: {}", stats.total_examples);
                    println!("Examples with tests: {}", stats.examples_with_tests);
                    println!("Total Python files: {}", stats.total_files);
                    println!("Total lines of code: {}", stats.total_lines);
                    println!();
                    println!("Examples:");
                    for example in corpus.iter().take(10) {
                        let has_tests = if example.tests.is_some() { "[tests]" } else { "" };
                        println!("  - {} {}", example.name, has_tests);
                    }
                    if corpus.len() > 10 {
                        println!("  ... and {} more", corpus.len() - 10);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load corpus: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Benchmark { corpus, verbose } => {
            tracing::info!(
                corpus = %corpus,
                verbose = verbose,
                "Analyzing corpus with Py2Rs 10-level benchmark"
            );

            match Corpus::load(&corpus) {
                Ok(corpus_data) => {
                    println!("┌────────────────────────────────────────────────────────────────┐");
                    println!("│ Py2Rs 10-Level Benchmark Analysis                              │");
                    println!("├────────────────────────────────────────────────────────────────┤");
                    println!("│ Corpus: {:>4} examples                                      │",
                        corpus_data.len());
                    println!("└────────────────────────────────────────────────────────────────┘");
                    println!();

                    // Classify all examples by level
                    let mut level_counts: HashMap<Py2RsLevel, Vec<String>> = HashMap::new();
                    let mut difficulty_counts: HashMap<Difficulty, usize> = HashMap::new();

                    for example in corpus_data.iter() {
                        let level = classify_example_level(example);
                        let difficulty = level.difficulty();

                        level_counts
                            .entry(level)
                            .or_default()
                            .push(example.name.clone());
                        *difficulty_counts.entry(difficulty).or_insert(0) += 1;
                    }

                    // Display level distribution
                    println!("Level Distribution (Py2Rs 10-Level Framework)");
                    println!("─────────────────────────────────────────────");

                    for level in Py2RsLevel::all() {
                        let count = level_counts.get(&level).map_or(0, Vec::len);
                        let bar_len = (count * 30) / corpus_data.len().max(1);
                        let bar: String = "█".repeat(bar_len);
                        let difficulty = level.difficulty();
                        let weight = level.weight();

                        println!(
                            "L{:2} {:15} [{:5}] {:30} ({} examples, weight {:.1})",
                            level.number(),
                            level.name(),
                            format!("{difficulty:?}"),
                            bar,
                            count,
                            weight
                        );

                        if verbose {
                            if let Some(examples) = level_counts.get(&level) {
                                for name in examples.iter().take(5) {
                                    println!("     └─ {name}");
                                }
                                if examples.len() > 5 {
                                    println!("     └─ ... and {} more", examples.len() - 5);
                                }
                            }
                        }
                    }

                    println!();
                    println!("Difficulty Breakdown");
                    println!("────────────────────");

                    let total = corpus_data.len();
                    for difficulty in Difficulty::all() {
                        let count = difficulty_counts.get(&difficulty).copied().unwrap_or(0);
                        let pct = if total > 0 {
                            (count as f64 / total as f64) * 100.0
                        } else {
                            0.0
                        };
                        println!(
                            "{:8} : {:3} examples ({:5.1}%)",
                            difficulty.name(),
                            count,
                            pct
                        );
                    }

                    // Compute composite score potential
                    println!();
                    println!("Benchmark Coverage");
                    println!("──────────────────");

                    let total_weight: f32 = Py2RsLevel::all()
                        .iter()
                        .map(Py2RsLevel::weight)
                        .sum();

                    let covered_weight: f32 = Py2RsLevel::all()
                        .iter()
                        .filter(|l| level_counts.get(l).is_some_and(|v| !v.is_empty()))
                        .map(Py2RsLevel::weight)
                        .sum();

                    let coverage_pct = (covered_weight / total_weight) * 100.0;

                    let covered_levels: Vec<_> = Py2RsLevel::all()
                        .iter()
                        .filter(|l| level_counts.get(l).is_some_and(|v| !v.is_empty()))
                        .map(|l| format!("L{}", l.number()))
                        .collect();

                    println!("Levels covered: {} / 10", covered_levels.len());
                    println!("Weight covered: {covered_weight:.1} / {total_weight:.1} ({coverage_pct:.1}%)");
                    println!("Levels: {}", covered_levels.join(", "));

                    // Visual summary
                    println!();
                    println!("Visual Summary (● = has examples, ○ = empty)");
                    let visual: String = Py2RsLevel::all()
                        .iter()
                        .map(|l| {
                            if level_counts.get(l).is_some_and(|v| !v.is_empty()) {
                                '●'
                            } else {
                                '○'
                            }
                        })
                        .collect();
                    println!("Levels 1-10: {visual}");
                }
                Err(e) => {
                    eprintln!("Failed to load corpus: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
