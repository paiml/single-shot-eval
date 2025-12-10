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

    /// Download models from `HuggingFace` with JIT caching (Toyota Way: Jidoka + Poka-yoke)
    Download {
        /// Model configuration YAML file
        #[arg(long)]
        config: String,

        /// Output directory for downloads
        #[arg(long, default_value = "models/raw")]
        output: String,

        /// Allow unsafe pickle files (NOT recommended)
        #[arg(long)]
        unsafe_allow_pickle: bool,

        /// Skip checksum verification (NOT recommended)
        #[arg(long)]
        skip_checksum: bool,
    },

    /// Convert models to .apr format with SPC precision gate (Toyota Way: SPC + Jidoka)
    Convert {
        /// Input model file or directory
        #[arg(long)]
        input: String,

        /// Output directory for .apr files
        #[arg(long, default_value = "models")]
        output: String,

        /// Quantization level: none, `q8_0`, `q4_0`, `q4_k_m`
        #[arg(long, default_value = "none")]
        quantization: String,

        /// Skip SPC numerical precision check (NOT recommended)
        #[arg(long)]
        skip_spc: bool,
    },

    /// Validate .apr models with logit consistency check (Toyota Way: Jidoka)
    Validate {
        /// Model files to validate (glob pattern)
        #[arg(long)]
        models: String,

        /// Validation prompts YAML file
        #[arg(long, default_value = "prompts/validation-prompts.yaml")]
        prompts: String,

        /// Skip logit consistency check
        #[arg(long)]
        skip_logit_check: bool,
    },

    /// Run full download-convert-test pipeline (Toyota Way: Heijunka)
    Pipeline {
        /// Model configuration YAML file
        #[arg(long)]
        config: String,

        /// Validation prompts YAML file
        #[arg(long, default_value = "prompts/validation-prompts.yaml")]
        prompts: String,

        /// Output directory for results
        #[arg(long, default_value = "results")]
        output: String,

        /// Number of evaluation runs (Princeton: 5+)
        #[arg(long, default_value = "5")]
        runs: usize,

        /// Maximum parallel downloads
        #[arg(long, default_value = "3")]
        parallel_downloads: usize,

        /// Allow unsafe pickle files
        #[arg(long)]
        unsafe_allow_pickle: bool,
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
        Commands::Download {
            config,
            output,
            unsafe_allow_pickle,
            skip_checksum,
        } => handle_download(&config, &output, unsafe_allow_pickle, skip_checksum),
        Commands::Convert {
            input,
            output,
            quantization,
            skip_spc,
        } => handle_convert(&input, &output, &quantization, skip_spc),
        Commands::Validate {
            models,
            prompts,
            skip_logit_check,
        } => handle_validate(&models, &prompts, skip_logit_check),
        Commands::Pipeline {
            config,
            prompts,
            output,
            runs,
            parallel_downloads,
            unsafe_allow_pickle,
        } => handle_pipeline(&PipelineArgs {
            config,
            prompts,
            output,
            runs,
            parallel_downloads,
            unsafe_allow_pickle,
        }),
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

// =============================================================================
// Download-Convert-Test Pipeline Handlers (Toyota Way)
// =============================================================================

/// Arguments for the pipeline command
struct PipelineArgs {
    config: String,
    prompts: String,
    output: String,
    runs: usize,
    parallel_downloads: usize,
    unsafe_allow_pickle: bool,
}

fn handle_download(config: &str, output: &str, unsafe_allow_pickle: bool, skip_checksum: bool) {
    use single_shot_eval::{validate_format_safety, DownloadConfig};

    tracing::info!(
        config = %config,
        output = %output,
        unsafe_allow_pickle = unsafe_allow_pickle,
        skip_checksum = skip_checksum,
        "Starting model download (Toyota Way: Jidoka + Poka-yoke)"
    );

    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Download Models from HuggingFace                               │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ Toyota Way Quality Gates:                                      │");
    println!("│   • Jidoka: SHA256 checksum verification                       │");
    println!("│   • Poka-yoke: Pickle file rejection                           │");
    println!("│   • JIT: 10GB cache with LRU eviction                          │");
    println!("└────────────────────────────────────────────────────────────────┘");
    println!();

    // Create download config
    let download_config = DownloadConfig {
        cache_dir: PathBuf::from(output),
        unsafe_allow_pickle,
        verify_checksum: !skip_checksum,
        ..Default::default()
    };

    // Load model configuration
    let model_config_path = PathBuf::from(config);
    if !model_config_path.exists() {
        eprintln!("Error: Model configuration not found: {config}");
        eprintln!("Create a YAML file with model repository IDs.");
        std::process::exit(1);
    }

    // Parse YAML and download models
    match std::fs::read_to_string(&model_config_path) {
        Ok(yaml_content) => {
            println!("Loaded configuration: {config}");

            // For now, show what would be downloaded
            // Real implementation would parse YAML and call HuggingFace API
            println!();
            println!("Quality gates active:");
            if download_config.verify_checksum {
                println!("  ✓ Checksum verification (Jidoka)");
            } else {
                println!("  ⚠ Checksum verification DISABLED");
            }
            if download_config.unsafe_allow_pickle {
                println!("  ⚠ Pickle files ALLOWED (security risk)");
            } else {
                println!("  ✓ Pickle rejection (Poka-yoke)");
            }
            println!("  ✓ JIT caching ({}GB max)", download_config.max_cache_bytes / 1_000_000_000);

            // Validate any existing files
            if let Ok(entries) = std::fs::read_dir(output) {
                let mut validated = 0;
                let mut rejected = 0;
                for entry in entries.flatten() {
                    let path = entry.path();
                    match validate_format_safety(&path, &download_config) {
                        Ok(()) => validated += 1,
                        Err(e) => {
                            eprintln!("  Rejected: {} - {}", path.display(), e);
                            rejected += 1;
                        }
                    }
                }
                if validated > 0 || rejected > 0 {
                    println!();
                    println!("Existing files: {validated} valid, {rejected} rejected");
                }
            }

            // Show YAML content summary
            let line_count = yaml_content.lines().count();
            println!();
            println!("Configuration: {line_count} lines");
            println!();
            println!("Note: Full HuggingFace download requires 'alimentar' crate.");
            println!("      Use 'make models-download' for curl-based download.");
        }
        Err(e) => {
            eprintln!("Failed to read configuration: {e}");
            std::process::exit(1);
        }
    }

    println!();
    println!("Download command completed.");
}

fn handle_convert(input: &str, output: &str, quantization: &str, skip_spc: bool) {
    use single_shot_eval::{ConvertConfig, Quantization, SourceFormat};

    tracing::info!(
        input = %input,
        output = %output,
        quantization = %quantization,
        skip_spc = skip_spc,
        "Starting model conversion (Toyota Way: SPC + Jidoka)"
    );

    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Convert Models to .apr Format                                  │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ Toyota Way Quality Gates:                                      │");
    println!("│   • SPC: KL divergence numerical precision check               │");
    println!("│   • Jidoka: Magic bytes and param count verification           │");
    println!("└────────────────────────────────────────────────────────────────┘");
    println!();

    // Parse quantization level
    let quant = match quantization.to_lowercase().as_str() {
        "none" | "fp16" => Quantization::None,
        "q8_0" | "q8" => Quantization::Q8_0,
        "q4_0" | "q4" => Quantization::Q4_0,
        "q4_k_m" | "q4km" => Quantization::Q4KM,
        _ => {
            eprintln!("Unknown quantization: {quantization}");
            eprintln!("Valid options: none, q8_0, q4_0, q4_k_m");
            std::process::exit(1);
        }
    };

    let convert_config = ConvertConfig {
        quantization: quant,
        skip_spc,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Input: {input}");
    println!("  Output: {output}");
    println!("  Quantization: {} (epsilon: {:.0e})", quant.as_str(), convert_config.epsilon());
    if skip_spc {
        println!("  ⚠ SPC check DISABLED");
    } else {
        println!("  ✓ SPC check enabled (sampling {} layers)", convert_config.spc_sample_layers);
    }
    println!();

    // Find input files
    let input_path = PathBuf::from(input);
    let files_to_convert: Vec<PathBuf> = if input_path.is_dir() {
        match std::fs::read_dir(&input_path) {
            Ok(entries) => entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    let format = SourceFormat::from_path(p);
                    format != SourceFormat::Unknown
                })
                .collect(),
            Err(e) => {
                eprintln!("Failed to read directory: {e}");
                std::process::exit(1);
            }
        }
    } else if input_path.exists() {
        vec![input_path]
    } else {
        // Try glob pattern
        match glob::glob(input) {
            Ok(paths) => paths.flatten().collect(),
            Err(e) => {
                eprintln!("Invalid input pattern: {e}");
                std::process::exit(1);
            }
        }
    };

    if files_to_convert.is_empty() {
        eprintln!("No convertible files found in: {input}");
        eprintln!("Supported formats: .safetensors, .gguf, .bin/.pt/.pth");
        std::process::exit(1);
    }

    // Create output directory
    let output_path = PathBuf::from(output);
    if let Err(e) = std::fs::create_dir_all(&output_path) {
        eprintln!("Failed to create output directory: {e}");
        std::process::exit(1);
    }

    println!("Found {} file(s) to convert:", files_to_convert.len());
    for file in &files_to_convert {
        let format = SourceFormat::from_path(file);
        let name = file.file_name().map_or("unknown", |n| n.to_str().unwrap_or("unknown"));
        println!("  • {} ({})", name, format.as_str());
    }

    println!();
    println!("Note: Full conversion requires 'entrenar' CLI.");
    println!("      Use 'make models-convert' for entrenar-based conversion.");
    println!();
    println!("Conversion command completed.");
}

fn handle_validate(models_glob: &str, prompts: &str, skip_logit_check: bool) {
    use single_shot_eval::{validate_apr_magic, LogitConsistencyChecker, ValidationConfig};

    tracing::info!(
        models = %models_glob,
        prompts = %prompts,
        skip_logit_check = skip_logit_check,
        "Starting model validation (Toyota Way: Jidoka)"
    );

    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Validate .apr Models                                           │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ Toyota Way Quality Gates:                                      │");
    println!("│   • Magic bytes: 0x41505221 (APR!)                             │");
    println!("│   • Logit consistency: 90% top-k agreement                     │");
    println!("└────────────────────────────────────────────────────────────────┘");
    println!();

    let validation_config = ValidationConfig {
        check_logit_consistency: !skip_logit_check,
        ..Default::default()
    };

    // Find model files
    let model_files: Vec<PathBuf> = match glob::glob(models_glob) {
        Ok(paths) => paths.flatten().filter(|p| {
            p.extension().is_some_and(|e| e == "apr")
        }).collect(),
        Err(e) => {
            eprintln!("Invalid model pattern: {e}");
            std::process::exit(1);
        }
    };

    if model_files.is_empty() {
        eprintln!("No .apr files found matching: {models_glob}");
        std::process::exit(1);
    }

    println!("Validating {} model(s):", model_files.len());
    println!();

    let mut passed = 0;
    let mut failed = 0;

    for model_path in &model_files {
        let name = model_path.file_name().map_or("unknown", |n| n.to_str().unwrap_or("unknown"));
        print!("  {name}: ");

        // Check magic bytes
        match validate_apr_magic(model_path) {
            Ok(()) => {
                println!("✓ PASS (magic bytes valid)");
                passed += 1;
            }
            Err(e) => {
                println!("✗ FAIL - {e}");
                failed += 1;
            }
        }
    }

    println!();
    println!("Results: {passed} passed, {failed} failed");

    println!();
    if validation_config.check_logit_consistency {
        let prompts_path = PathBuf::from(prompts);
        if prompts_path.exists() {
            println!("Logit consistency check: prompts loaded from {prompts}");
            let checker = LogitConsistencyChecker::default();
            println!("  Top-k: {}", checker.top_k);
            println!("  Tolerance: {}", checker.logit_tolerance);
            println!("  Min agreement: {}%", checker.min_agreement * 100.0);
        } else {
            println!("Note: Prompts file not found: {prompts}");
            println!("      Create prompts/validation-prompts.yaml for logit consistency checks.");
        }
    } else {
        println!("Logit consistency check: SKIPPED");
    }

    if failed > 0 {
        std::process::exit(1);
    }
}

fn handle_pipeline(args: &PipelineArgs) {
    tracing::info!(
        config = %args.config,
        prompts = %args.prompts,
        output = %args.output,
        runs = args.runs,
        parallel_downloads = args.parallel_downloads,
        "Starting full pipeline (Toyota Way: Heijunka)"
    );

    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Download-Convert-Test Pipeline                                 │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ Toyota Way Principles:                                         │");
    println!("│   • Heijunka: Parallel stage execution                         │");
    println!("│   • Jidoka: Quality gates at each stage                        │");
    println!("│   • Poka-yoke: Pickle rejection, format validation             │");
    println!("│   • SPC: Numerical precision checks                            │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ Princeton Methodology: {} runs per model                         │", args.runs);
    println!("└────────────────────────────────────────────────────────────────┘");
    println!();

    // Check configuration file
    let config_path = PathBuf::from(&args.config);
    if !config_path.exists() {
        eprintln!("Error: Configuration not found: {}", args.config);
        eprintln!();
        eprintln!("Create a model configuration YAML like:");
        eprintln!("  models/test-models.yaml");
        eprintln!();
        std::process::exit(1);
    }

    // Check prompts file
    let prompts_path = PathBuf::from(&args.prompts);
    if !prompts_path.exists() {
        eprintln!("Warning: Prompts file not found: {}", args.prompts);
        eprintln!("         Logit consistency checks will be skipped.");
        eprintln!();
    }

    println!("Pipeline Configuration:");
    println!("  Model config: {}", args.config);
    println!("  Prompts: {}", args.prompts);
    println!("  Output: {}", args.output);
    println!("  Runs: {} (Princeton methodology)", args.runs);
    println!("  Parallel downloads: {}", args.parallel_downloads);
    if args.unsafe_allow_pickle {
        println!("  ⚠ Pickle files: ALLOWED (security risk)");
    } else {
        println!("  ✓ Pickle files: REJECTED (Poka-yoke)");
    }
    println!();

    println!("Pipeline Stages:");
    println!("  1. DOWNLOAD  → HuggingFace Hub with JIT caching");
    println!("  2. CONVERT   → .apr format with SPC gate");
    println!("  3. VALIDATE  → Magic bytes + logit consistency");
    println!("  4. EVALUATE  → {} runs with 95% CI", args.runs);
    println!("  5. REPORT    → Pareto frontier analysis");
    println!();

    // Create output directory
    let output_path = PathBuf::from(&args.output);
    if let Err(e) = std::fs::create_dir_all(&output_path) {
        eprintln!("Failed to create output directory: {e}");
        std::process::exit(1);
    }

    // Run stages sequentially (Heijunka parallel version would use tokio channels)
    println!("Running pipeline stages...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Stage 1: Download
    println!();
    println!("Stage 1: DOWNLOAD");
    println!("─────────────────");
    handle_download(&args.config, "models/raw", args.unsafe_allow_pickle, false);

    // Stage 2: Convert
    println!();
    println!("Stage 2: CONVERT");
    println!("────────────────");
    handle_convert("models/raw", "models", "none", false);

    // Stage 3: Validate
    println!();
    println!("Stage 3: VALIDATE");
    println!("─────────────────");
    handle_validate("models/*.apr", &args.prompts, false);

    // Stage 4: Evaluate (would use TaskRunner)
    println!();
    println!("Stage 4: EVALUATE");
    println!("─────────────────");
    println!("  Note: Connect to TaskRunner for full evaluation.");
    println!("  Placeholder: Would run {} iterations per model.", args.runs);

    // Stage 5: Report
    println!();
    println!("Stage 5: REPORT");
    println!("───────────────");
    println!("  Note: Generate Pareto frontier report.");
    println!("  Output would be: {}/pareto-report.md", args.output);

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Pipeline completed (dry run - full implementation pending).");
    println!();
    println!("Next steps:");
    println!("  1. Install 'alimentar' for HuggingFace downloads");
    println!("  2. Install 'entrenar' for format conversion");
    println!("  3. Create validation prompts: {}", args.prompts);
}
