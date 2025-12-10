//! Single-Shot Eval CLI
//!
//! SLM Pareto Frontier Evaluation Framework

use clap::{Parser, Subcommand};
use single_shot_eval::{
    analyze_pareto, available_baselines, classify_example_level, CompilerVerifier, Corpus,
    Difficulty, EvalResult, Py2RsLevel, ReportBuilder, RunnerConfig, TaskLoader, TaskRunner,
};
use std::collections::HashMap;
use std::path::PathBuf;
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

/// Arguments for the evaluate command
struct EvaluateArgs {
    tasks: String,
    models: String,
    corpus: Option<String>,
    baselines: Vec<String>,
    output: Option<String>,
    runs: usize,
}

fn main() {
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
            samples: _,
            runs,
        } => handle_evaluate(&EvaluateArgs {
            tasks,
            models,
            corpus,
            baselines,
            output,
            runs,
        }),
        Commands::Report { input, output } => handle_report(&input, &output),
        Commands::Analyze { models, output } => handle_analyze(&models, output.as_deref()),
        Commands::Verify { source, tests } => handle_verify(&source, tests.as_deref()),
        Commands::CorpusStats { path } => handle_corpus_stats(&path),
        Commands::Benchmark { corpus, verbose } => handle_benchmark(&corpus, verbose),
    }
}

fn handle_evaluate(args: &EvaluateArgs) {
    tracing::info!(
        tasks = %args.tasks,
        models = %args.models,
        corpus = ?args.corpus,
        baselines = ?args.baselines,
        output = ?args.output,
        runs = args.runs,
        "Starting evaluation (Princeton methodology: {} runs)", args.runs
    );

    let corpus_data = load_corpus_if_provided(args.corpus.as_ref());
    let task_loader = load_task_configs(&args.tasks);
    let model_paths = find_model_files(&args.models);
    let runner = create_task_runner(&args.baselines);

    println!(
        "\nStarting evaluation ({} runs per Princeton methodology)...",
        args.runs
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let all_results = run_evaluations(&task_loader, &model_paths, &runner, args.runs);

    if all_results.is_empty() {
        eprintln!("\nNo successful evaluations completed");
        std::process::exit(1);
    }

    print_pareto_analysis(&all_results);
    generate_report_if_requested(args.output.as_ref(), &all_results);

    if let Some(c) = corpus_data {
        println!("\nCorpus used: {} examples", c.len());
    }

    println!("\nEvaluation complete ({} runs per model)", args.runs);
}

fn load_corpus_if_provided(corpus_path: Option<&String>) -> Option<Corpus> {
    let path = corpus_path?;
    match Corpus::load(path) {
        Ok(c) => {
            let stats = c.stats();
            println!(
                "Loaded corpus: {} examples, {} with tests",
                stats.total_examples, stats.examples_with_tests
            );
            Some(c)
        }
        Err(e) => {
            eprintln!("Failed to load corpus: {e}");
            std::process::exit(1);
        }
    }
}

fn load_task_configs(tasks_glob: &str) -> TaskLoader {
    match TaskLoader::load_glob(tasks_glob) {
        Ok(loader) => {
            if loader.is_empty() {
                eprintln!("No task configurations found matching: {tasks_glob}");
                std::process::exit(1);
            }
            println!("Loaded {} task configuration(s)", loader.len());
            loader
        }
        Err(e) => {
            eprintln!("Failed to load tasks: {e}");
            std::process::exit(1);
        }
    }
}

fn find_model_files(models_glob: &str) -> Vec<PathBuf> {
    let paths: Vec<_> = match glob::glob(models_glob) {
        Ok(paths) => paths.filter_map(Result::ok).collect(),
        Err(e) => {
            eprintln!("Invalid model glob pattern: {e}");
            std::process::exit(1);
        }
    };

    if paths.is_empty() {
        eprintln!("No model files found matching: {models_glob}");
        eprintln!("Note: Models must be .apr format (Aprender native format)");
        std::process::exit(1);
    }
    println!("Found {} model file(s)", paths.len());
    paths
}

fn create_task_runner(baselines: &[String]) -> TaskRunner {
    let runner_config = RunnerConfig {
        run_baselines: !baselines.is_empty() || !available_baselines().is_empty(),
        ..RunnerConfig::default()
    };
    TaskRunner::with_config(runner_config)
}

fn run_evaluations(
    task_loader: &TaskLoader,
    model_paths: &[PathBuf],
    runner: &TaskRunner,
    runs: usize,
) -> Vec<EvalResult> {
    let mut all_results = Vec::new();

    for task_config in task_loader.iter() {
        println!(
            "\nTask: {} ({})",
            task_config.task.id, task_config.task.domain
        );

        for model_path in model_paths {
            let model_id = model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            if let Err(e) = runner.validate_model(model_path) {
                eprintln!("  Skipping {model_id}: {e}");
                continue;
            }

            print!("  Evaluating {model_id}... ");

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

    all_results
}

fn print_pareto_analysis(results: &[EvalResult]) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PARETO FRONTIER ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let pareto = analyze_pareto(results);
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
}

fn generate_report_if_requested(output_path: Option<&String>, results: &[EvalResult]) {
    let Some(path) = output_path else { return };

    let mut builder = ReportBuilder::new("evaluation");
    for result in results {
        builder.add_result(result.clone(), vec![result.accuracy; 100]);
    }

    let report = builder.build();
    let is_json = std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
    let output_str = if is_json {
        report
            .to_json()
            .unwrap_or_else(|e| format!("JSON error: {e}"))
    } else {
        report.to_markdown()
    };

    match std::fs::write(path, &output_str) {
        Ok(()) => println!("\nReport written to: {path}"),
        Err(e) => eprintln!("\nFailed to write report: {e}"),
    }
}

fn handle_report(input: &str, output: &str) {
    tracing::info!(input = %input, output = %output, "Generating report");
    println!("Report generation not yet implemented");
}

fn handle_analyze(models: &str, output: Option<&str>) {
    tracing::info!(models = %models, output = ?output, "Analyzing models");
    println!("Analysis not yet implemented");
}

fn handle_verify(source: &str, tests: Option<&str>) {
    tracing::info!(source = %source, tests = ?tests, "Verifying Rust code");

    let rust_code = match std::fs::read_to_string(source) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Failed to read source file: {e}");
            std::process::exit(1);
        }
    };

    let test_code = tests.and_then(|p| std::fs::read_to_string(p).ok());

    let verifier = CompilerVerifier::new();
    match verifier.verify(&rust_code, test_code.as_deref()) {
        Ok(result) => {
            println!(
                "Compilation: {}",
                if result.compiles { "PASS" } else { "FAIL" }
            );
            if let Some(tests_pass) = result.tests_pass {
                println!("Tests: {}", if tests_pass { "PASS" } else { "FAIL" });
                println!(
                    "Tests run: {}, passed: {}",
                    result.tests_run, result.tests_passed
                );
            }
            println!("Build time: {:?}", result.build_time);
            if let Some(test_time) = result.test_time {
                println!("Test time: {test_time:?}");
            }
            println!(
                "\nGround truth: {}",
                if result.passes() { "PASS" } else { "FAIL" }
            );

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

fn handle_corpus_stats(path: &str) {
    tracing::info!(path = %path, "Loading corpus statistics");

    match Corpus::load(path) {
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
                let has_tests = if example.tests.is_some() {
                    "[tests]"
                } else {
                    ""
                };
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

fn handle_benchmark(corpus_path: &str, verbose: bool) {
    tracing::info!(corpus = %corpus_path, verbose = verbose, "Analyzing corpus with Py2Rs 10-level benchmark");

    match Corpus::load(corpus_path) {
        Ok(corpus_data) => {
            print_benchmark_header(corpus_data.len());
            let (level_counts, difficulty_counts) = classify_corpus(&corpus_data);
            print_level_distribution(&level_counts, corpus_data.len(), verbose);
            print_difficulty_breakdown(&difficulty_counts, corpus_data.len());
            print_benchmark_coverage(&level_counts);
        }
        Err(e) => {
            eprintln!("Failed to load corpus: {e}");
            std::process::exit(1);
        }
    }
}

fn print_benchmark_header(count: usize) {
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Py2Rs 10-Level Benchmark Analysis                              │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ Corpus: {count:>4} examples                                      │");
    println!("└────────────────────────────────────────────────────────────────┘");
    println!();
}

fn classify_corpus(
    corpus: &Corpus,
) -> (HashMap<Py2RsLevel, Vec<String>>, HashMap<Difficulty, usize>) {
    let mut level_counts: HashMap<Py2RsLevel, Vec<String>> = HashMap::new();
    let mut difficulty_counts: HashMap<Difficulty, usize> = HashMap::new();

    for example in corpus.iter() {
        let level = classify_example_level(example);
        let difficulty = level.difficulty();

        level_counts
            .entry(level)
            .or_default()
            .push(example.name.clone());
        *difficulty_counts.entry(difficulty).or_insert(0) += 1;
    }

    (level_counts, difficulty_counts)
}

fn print_level_distribution(
    level_counts: &HashMap<Py2RsLevel, Vec<String>>,
    total: usize,
    verbose: bool,
) {
    println!("Level Distribution (Py2Rs 10-Level Framework)");
    println!("─────────────────────────────────────────────");

    for level in Py2RsLevel::all() {
        let count = level_counts.get(&level).map_or(0, Vec::len);
        let bar_len = (count * 30) / total.max(1);
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
}

fn print_difficulty_breakdown(difficulty_counts: &HashMap<Difficulty, usize>, total: usize) {
    println!();
    println!("Difficulty Breakdown");
    println!("────────────────────");

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
}

fn print_benchmark_coverage(level_counts: &HashMap<Py2RsLevel, Vec<String>>) {
    println!();
    println!("Benchmark Coverage");
    println!("──────────────────");

    let total_weight: f32 = Py2RsLevel::all().iter().map(Py2RsLevel::weight).sum();

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
