//! Single-Shot Eval CLI
//!
//! SLM Pareto Frontier Evaluation Framework

use clap::{Parser, Subcommand};
use single_shot_eval::{CompilerVerifier, Corpus};
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
            if let Some(corpus_path) = &corpus {
                match Corpus::load(corpus_path) {
                    Ok(c) => {
                        let stats = c.stats();
                        println!("Loaded corpus: {} examples, {} with tests",
                            stats.total_examples, stats.examples_with_tests);
                    }
                    Err(e) => {
                        eprintln!("Failed to load corpus: {e}");
                        std::process::exit(1);
                    }
                }
            }

            println!("Evaluation not yet fully implemented");
            println!("Will run {runs} iterations per Princeton methodology");
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
    }
}
