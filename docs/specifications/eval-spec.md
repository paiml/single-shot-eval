# Single-Shot Eval: SLM Pareto Frontier Analysis Specification

**Version**: 1.2.1
**Status**: APPROVED
**Authors**: Claude Code
**Date**: 2025-12-08
**Research Basis**: Princeton "AI Agents That Matter" (2024), SWE-bench (ICLR 2024)
**Repository**: `paiml/single-shot-eval`

---

## Executive Summary

This specification defines a **proof-of-concept evaluation framework** for quantifying Pareto trade-offs between model accuracy, size, and cost. The framework validates methodology that will later be generalized in `aprender` for evaluating ANY agent task.

**Core Question**: *"If a model is 1% worse but 100x smaller and FREE, is that acceptable?"*

We answer this scientifically using Princeton's "AI Agents That Matter" methodology.

**Why Python→Rust?** The single-shot compilation domain (via `reprorusted-python-cli`) is **the test case, not the goal**. Depyler solves this problem better. But we *understand* this domain deeply, making it the perfect first benchmark to validate our evaluation methodology before scaling.

**Core Thesis**: Properly distilled SLMs can achieve production-quality results on targeted tasks. The framework quantifies exactly how much accuracy you trade for cost/size savings—with statistical rigor (5 runs, 95% CI, convex Pareto frontiers).

**Future**: Once validated here, this methodology moves to `aprender` for MANY evaluation types across different agent domains.

**CRITICAL CONSTRAINTS**:

1. **OFFLINE-FIRST**: NO HTTP API calls. All model inference is LOCAL via aprender.
   - Primary: Evaluate `.apr` models using `aprender` (deterministic, reproducible)
   - Optional baseline: Shell out to installed CLI agents (`claude`, `gemini`, `llm`) if available
   - NO API keys required - fully air-gapped capable

2. **MANDATORY .apr Format**: ALL models MUST be in Aprender's `.apr` format for evaluation. External formats (GGUF, SafeTensors, PyTorch, ONNX) MUST be converted using `entrenar convert` before evaluation. NO exceptions.

3. **No Duplication Policy**: This crate MUST NOT duplicate functionality from:
   - `aprender` (v0.15.0+) - ML algorithms, metrics, .apr format, inference
   - `entrenar` (v0.2.5+) - Training, distillation, quantization, conversion
   - `trueno` (v0.8.0+) - Tensor operations, SIMD acceleration

   If functionality exists in these crates, USE IT. Do not reimplement.

4. **Crates.io Only**: Always use published versions from crates.io. Never use git dependencies.

**Methodology Alignment**:
- **Toyota Way**: Genchi Genbutsu (go and see the data), Jidoka (quality built-in)
- **Google AI Principles**: Rigorous empirical validation, reproducibility, ablation studies
- **NASA AI Standards**: Mission-critical reliability, failure mode analysis, uncertainty quantification

---

## Table of Contents

1. [Motivation & Background](#motivation--background)
2. [Princeton Research Integration](#princeton-research-integration)
3. [Architecture Overview](#architecture-overview)
4. [Evaluation Framework](#evaluation-framework)
5. [Pareto Frontier Analysis](#pareto-frontier-analysis)
6. [Ground Truth: Single-Shot Compilation](#ground-truth-single-shot-compilation)
7. [Benchmark Tasks](#benchmark-tasks)
8. [Implementation Plan](#implementation-plan)
9. [Quality Gates](#quality-gates)
10. [Peer-Reviewed Citations](#peer-reviewed-citations)
11. [Appendix: Toyota Way for AI Evaluation](#appendix-toyota-way-for-ai-evaluation)

---

## 1. Motivation & Background

### 1.1 The SLM Opportunity

Recent research demonstrates that task-specific small models can match or exceed general-purpose large models, especially when trained on high-quality synthetic data or "textbook quality" corpora [1, 26, 27]. Additionally, compute-optimal scaling laws suggest that smaller models trained on more tokens can outperform larger, undertrained models [28].

| Model Size | Parameters | Inference Cost | Task-Specific Accuracy |
|------------|------------|----------------|------------------------|
| Frontier (Claude/GPT-4) | 175B+ | $0.01-0.06/1K tokens | 95% (general) |
| Mid-tier (Llama-7B) | 7B | $0.001/1K tokens | 85% (general) |
| **SLM (Distilled)** | **100M-500M** | **$0.00001/1K tokens** | **92%+ (targeted)** |

**Key Insight**: The Pareto frontier for cost vs. accuracy is not monotonic—properly distilled SLMs occupy a superior position for domain-specific workloads [4, 5].

### 1.2 Why Batuta Stack?

The Batuta sovereign stack provides unique advantages and MUST be used exclusively:

| Crate | Version | Role | single-shot-eval Usage |
|-------|---------|------|------------------------|
| **trueno** | 0.8.0 | SIMD-accelerated tensor ops | Indirect (via aprender/entrenar) |
| **aprender** | 0.15.0 | ML algorithms, metrics, .apr format | Model loading, inference, metrics |
| **entrenar** | 0.2.5 | Training, distillation, conversion | Format conversion, distillation |
| **alimentar** | 0.2.2 | Dataset loading | Dataset access (via entrenar) |

**Stack Responsibilities** (NO DUPLICATION ALLOWED):

1. **Depyler**: Static analysis of model computation graphs reveals optimization opportunities invisible to runtime-only tools [6]
2. **Aprender**: Native Rust ML with deterministic inference, `.apr` model format, accuracy/F1/MSE metrics
3. **Entrenar**: Distillation framework with teacher-student protocols, format conversion (GGUF/SafeTensors → .apr)
4. **Trueno**: SIMD-accelerated inference with predictable latency (used internally by aprender)

**What single-shot-eval Provides** (ONLY these):
- Pareto frontier analysis (multi-objective optimization)
- CLI baseline wrappers (shell exec to `claude`, `gemini` if installed)
- Task configuration and orchestration
- Report generation

### 1.3 Evaluation Philosophy

Following Google's AI evaluation principles [7] and NASA's V&V standards [8]:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION TRIANGLE                                       │
│                                                                             │
│                         Accuracy                                            │
│                            ▲                                                │
│                           /│\                                               │
│                          / │ \                                              │
│                         /  │  \                                             │
│                        /   │   \                                            │
│                       / PARETO  \                                           │
│                      / FRONTIER  \                                          │
│                     /      │      \                                         │
│                    /       │       \                                        │
│                   ▼────────┴────────▼                                       │
│               Cost                Latency                                   │
│                                                                             │
│  Goal: Find SLMs that dominate the frontier for specific task domains       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 MANDATORY .apr Format Requirement

**ALL models evaluated by single-shot-eval MUST be in `.apr` format.**

The `.apr` (Aprender) format is the native model serialization format of the Batuta stack, providing:

- **Deterministic Loading**: Bit-exact model reconstruction
- **Quantization Support**: Q4_0, Q8_0 block formats
- **Metadata**: Training provenance, hyperparameters, checksums
- **Security**: Optional Ed25519 signatures, AES-GCM encryption
- **Compression**: Optional zstd compression

**Format Conversion Pipeline**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MODEL FORMAT CONVERSION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  External Formats                    Batuta Native                          │
│  ┌──────────────┐                   ┌──────────────┐                       │
│  │    GGUF      │──┐                │              │                       │
│  │ (llama.cpp)  │  │                │              │                       │
│  └──────────────┘  │                │              │                       │
│  ┌──────────────┐  │  entrenar      │     .apr     │    aprender           │
│  │ SafeTensors  │──┼──convert──────▶│    format    │────inference─────▶ ✓  │
│  │ (HuggingFace)│  │                │              │                       │
│  └──────────────┘  │                │              │                       │
│  ┌──────────────┐  │                │              │                       │
│  │   PyTorch    │──┘                │              │                       │
│  │   (.pt)      │                   └──────────────┘                       │
│  └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Conversion Commands**:

```bash
# Convert GGUF to .apr
entrenar convert --input model.gguf --output model.apr

# Convert SafeTensors to .apr
entrenar convert --input model.safetensors --output model.apr

# Convert with quantization
entrenar convert --input model.safetensors --output model-q4.apr --quantize q4_0

# Convert from HuggingFace Hub directly
entrenar convert --hf-repo meta-llama/Llama-3.2-3B --output llama3.apr
```

**Validation**: The evaluation engine MUST reject any model not in `.apr` format with a clear error message directing users to `entrenar convert`.

---

## 2. Princeton Research Integration

This framework incorporates key methodological advances from Princeton NLP research, particularly the "AI Agents That Matter" paper (2024) and SWE-bench evaluation framework (ICLR 2024).

### 2.1 "AI Agents That Matter" - Key Findings

**Source**: https://agents.cs.princeton.edu/ (Kapoor et al., 2024)

Princeton researchers discovered that **simple baselines offer Pareto improvements over state-of-the-art agents** when cost is properly accounted for. Their key recommendations, which we adopt:

| Princeton Finding | Our Implementation |
|-------------------|-------------------|
| **5 runs minimum** with 95% CI | All evaluations run 5× with Student's t-distribution CI |
| **Convex Pareto frontier** | Probability-weighted combinations of agents |
| **Dollar costs, not proxies** | Actual API costs + compute amortization |
| **Report token counts** | Input/output tokens logged for future recalculation |
| **Holdout sets required** | Strict train/dev/test splits for reprorusted corpus |

### 2.2 Statistical Methodology (Princeton Protocol)

```rust
/// Princeton-compliant evaluation configuration
pub struct PrincetonEvalConfig {
    /// Minimum runs per configuration (Princeton: 5)
    pub min_runs: usize,              // 5
    /// Confidence level for CI
    pub confidence: f64,              // 0.95
    /// Use Student's t for small samples
    pub use_t_distribution: bool,     // true
    /// Report both mean and confidence bounds
    pub report_ci: bool,              // true
}
```

**Confidence Interval Computation**:

```
CI = mean ± t_{α/2, n-1} × (std_dev / √n)

Where:
  - n = 5 (minimum runs)
  - α = 0.05 (95% confidence)
  - t_{0.025, 4} = 2.776 (Student's t critical value)
```

### 2.3 Convex Pareto Frontier

Princeton's key insight: The Pareto frontier must be **convex** because we can always combine agents probabilistically:

```
"Given any two agents on the frontier, we can always choose a strategy
that picks agent 1 with probability p and agent 2 with probability 1-p."
```

This means dominated regions include not just worse agents, but also any agent below the line connecting two frontier points.

```
Accuracy (%)
    │
100 ┤              A ────────── Convex Hull
    │             /│\
 95 ┤            / │ \ ────── Frontier points
    │           /  │  \
 90 ┤          /   │   B
    │         /    │    \
 85 ┤        / DOMINATED  \
    │       /      │       \
 80 ┤      C───────┼────────D
    │              │
    └──────┴───────┴────────┴────▶ Cost ($)
           $0.01   $0.10    $1.00

Points in the shaded region are dominated by a probabilistic
combination of frontier agents, NOT just by individual agents.
```

### 2.4 SWE-bench Ground Truth Methodology

**Source**: Princeton SWE-bench (ICLR 2024)

SWE-bench introduced rigorous ground truth verification for code generation:

| Concept | Definition | Our Adaptation |
|---------|------------|----------------|
| **FAIL_TO_PASS** | Tests that fail before the fix, pass after | Our generated Rust must compile AND pass tests |
| **PASS_TO_PASS** | Existing tests that must remain passing | Not applicable (we generate complete programs) |
| **Resolved** | All FAIL_TO_PASS tests pass | `cargo build && cargo test` succeeds |

**Critical SWE-bench Finding** (2024 follow-up research):
> "7.8% of 'correct' patches fail the developer-written test suite. 29.6% of plausible patches induce different behavior than ground truth patches."

This reinforces our approach: **compilation + test passage is necessary but may not be sufficient**. We additionally:
1. Compare behavioral equivalence via property-based testing
2. Run deterministic inputs through both Python and Rust versions
3. Flag divergent behavior for manual review

### 2.5 Criticism of Current Evaluation Practices

Princeton identified pervasive issues in agent evaluation that we explicitly address:

| Issue | Problem | Our Solution |
|-------|---------|--------------|
| **Cost blindness** | "SOTA agents are needlessly complex and costly" | Every result includes dollar cost |
| **Single-run reporting** | No variance estimation | 5 runs × 95% CI mandatory |
| **Benchmark overfitting** | No holdout sets | Strict corpus splits |
| **Reproducibility failures** | Non-deterministic environments | Containerized evaluation, fixed seeds |
| **Missing baselines** | No simple baseline comparison | Always compare to pass@1 baseline |

### 2.6 HumanEval Lessons

Princeton's HumanEval analysis showed:

> "Simple repetition baselines (calling models multiple times) outperform complex
> agent architectures while costing less."

For single-shot compilation, this means:
- **pass@1**: Single attempt accuracy (our primary metric)
- **pass@k**: Best of k attempts (useful for understanding variance)
- **cost-adjusted pass@k**: What matters for production deployment

```rust
/// Cost-adjusted pass rate
pub fn cost_adjusted_pass_rate(
    pass_rate: f64,
    attempts: usize,
    cost_per_attempt: f64,
) -> f64 {
    // Princeton formula: accuracy / total_cost
    pass_rate / (attempts as f64 * cost_per_attempt)
}
```

---

## 3. Architecture Overview

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SINGLE-SHOT-EVAL ARCHITECTURE                           │
│                     (Thin Orchestration Layer)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    BATUTA STACK (crates.io)                           │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │  │
│  │  │   entrenar   │    │   aprender   │    │   trueno     │            │  │
│  │  │   v0.2.5     │    │   v0.15.0    │    │   v0.8.0     │            │  │
│  │  │ ─────────────│    │ ─────────────│    │ ─────────────│            │  │
│  │  │ • convert    │    │ • .apr load  │    │ • SIMD ops   │            │  │
│  │  │ • distill    │───▶│ • inference  │───▶│ • tensors    │            │  │
│  │  │ • quantize   │    │ • metrics    │    │ • matmul     │            │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              SINGLE-SHOT-EVAL (THIS CRATE - ORCHESTRATION ONLY)       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐   │  │
│  │  │ Task Config │  │   Pareto    │  │  Baseline   │  │   Report   │   │  │
│  │  │   (YAML)    │  │  Frontier   │  │    APIs     │  │ Generator  │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              OPTIONAL BASELINES (shell exec to installed CLIs)        │  │
│  │  ┌─────────────────┐    ┌─────────────────┐                          │  │
│  │  │  claude         │    │  gemini         │                          │  │
│  │  │  (Claude Code)  │    │  (Gemini CLI)   │                          │  │
│  │  └─────────────────┘    └─────────────────┘                          │  │
│  │  NO HTTP APIs - shells out to `claude` and `gemini` if installed      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

LEGEND:
  ───▶  = Uses as dependency
  .apr  = ONLY supported model format
  SLM inference = aprender ONLY (no custom inference code)
```

### 2.2 Data Flow

```
                              ┌──────────────────┐
                              │  External Model  │
                              │ (GGUF/SafeTensors│
                              │  /PyTorch)       │
                              └────────┬─────────┘
                                       │
                                       ▼ entrenar convert (REQUIRED)
                              ┌──────────────────┐
Input Dataset                 │   .apr Model     │
     │                        │ (Batuta Native)  │
     ▼                        └────────┬─────────┘
┌────────────────┐                     │
│ Task Definition │                    │
│ (YAML config)  │                     │
└────────────────┘                     │
     │                                 │
     ▼                                 ▼
┌────────────────────────────────────────────────────────────┐
│                    EVALUATION ORCHESTRATION                 │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────┐  │
│  │ SLM Inference  │    │ CLI Baseline   │    │ Ground   │  │
│  │ (aprender)     │    │ (optional)     │    │ Truth    │  │
│  │ .apr ONLY      │    │ shell exec     │    │ (Labels) │  │
│  └────────┬───────┘    └───────┬────────┘    └────┬─────┘  │
│           │                    │                  │        │
│           └─────────┬──────────┘                  │        │
│                     ▼                             │        │
│           ┌────────────────┐                      │        │
│           │ Metrics Engine │◀─────────────────────┘        │
│           │ (aprender)     │                               │
│           │ - accuracy()   │                               │
│           │ - f1_score()   │                               │
│           │ - mse()        │                               │
│           └───────┬────────┘                               │
│                   │                                        │
│                   ▼                                        │
│           ┌────────────────┐                               │
│           │ Pareto Analysis│ ◀── THIS CRATE PROVIDES       │
│           │ (single-shot)  │                               │
│           │ - Dominance    │                               │
│           │ - Frontier     │                               │
│           │ - Value Score  │                               │
│           └───────┬────────┘                               │
│                   │                                        │
│                   ▼                                        │
│           ┌────────────────┐                               │
│           │ Report Gen     │ ◀── THIS CRATE PROVIDES       │
│           │ - Markdown     │                               │
│           │ - Tables       │                               │
│           └────────────────┘                               │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Evaluation Framework

### 3.1 Metrics Taxonomy

Following the ML evaluation best practices from Google Research [9] and Microsoft Research [10].

**CRITICAL: Use `aprender::metrics` for all ML metrics. DO NOT reimplement.**

| Metric Category | Metric | Unit | Source |
|-----------------|--------|------|--------|
| **Accuracy** | Task Accuracy | % | `aprender::metrics::accuracy()` |
| | F1 Score | [0,1] | `aprender::metrics::f1_score()` |
| | BLEU/ROUGE | [0,1] | `aprender::metrics` (if available) or `aprender::text` |
| | Exact Match | % | `aprender::metrics::accuracy()` (strict mode) |
| **Efficiency** | Inference Latency | ms | `std::time::Instant` (this crate) |
| | Throughput | tokens/sec | Computed (this crate) |
| | Memory Peak | MB | RSS monitoring (this crate) |
| **Cost** | API Cost | $/1K tokens | Provider pricing (this crate) |
| | Compute Cost | $/hour | Hardware amortization (this crate) |
| | TCO | $/1M inferences | Computed (this crate) |
| **Reliability** | Consistency | % | Computed (this crate) |
| | Failure Rate | % | Computed (this crate) |

**Metrics Responsibility Matrix:**

| Metric Type | Provided By | Rationale |
|-------------|-------------|-----------|
| ML accuracy metrics | `aprender` | Core ML library, already tested (96.94% coverage) |
| Latency/timing | `single-shot-eval` | Evaluation-specific timing |
| Cost computation | `single-shot-eval` | Provider-specific pricing logic |
| Pareto analysis | `single-shot-eval` | Unique to this evaluation framework |

### 3.2 Statistical Rigor

All evaluations follow NASA's statistical standards for AI systems [8]:

1. **Sample Size**: Minimum n=1000 per task, power analysis for effect size detection
2. **Confidence Intervals**: 95% CI via bootstrap (10,000 resamples)
3. **Significance Testing**: Paired t-test with Bonferroni correction for multiple comparisons
4. **Effect Size**: Cohen's d reported for all comparisons
5. **Reproducibility**: Fixed random seeds, deterministic execution

```rust
/// Statistical evaluation configuration
pub struct EvalConfig {
    /// Minimum samples per task
    pub min_samples: usize,           // 1000
    /// Bootstrap resamples for CI
    pub bootstrap_n: usize,           // 10000
    /// Confidence level
    pub confidence: f64,              // 0.95
    /// Random seed for reproducibility
    pub seed: u64,                    // Deterministic
    /// Maximum p-value for significance
    pub alpha: f64,                   // 0.05
}
```

### 3.3 Baseline Models (OFFLINE ONLY)

**Primary evaluation**: `.apr` models via `aprender` (LOCAL, deterministic)

**Optional baselines**: Shell exec to installed CLI tools (NO HTTP APIs)

| Baseline | CLI Command | Detection | Usage |
|----------|-------------|-----------|-------|
| Claude Code | `claude` | `which claude` | `claude --print "prompt"` |
| Gemini CLI | `gemini` | `which gemini` | `gemini "prompt"` |

**Cost Model** (for Pareto analysis):

| Model Type | Estimated Cost | Latency Profile |
|------------|----------------|-----------------|
| Frontier (claude/gemini CLI) | ~$0.01/1K tokens | 500ms-2s |
| **SLM (.apr via aprender)** | **~$0.0001/1K tokens** | **<100ms** |

**NOTE**: Baseline CLI tools are OPTIONAL. Evaluation works fully offline with just `.apr` models. When baselines are unavailable, Pareto analysis uses cached/historical baseline data or skips baseline comparison.

---

## 4. Pareto Frontier Analysis

### 4.1 Multi-Objective Optimization

The Pareto frontier represents the set of non-dominated solutions in the accuracy-cost-latency space [12, 13]:

```
Definition: A solution S₁ DOMINATES S₂ if and only if:
  - S₁ is no worse than S₂ in all objectives
  - S₁ is strictly better than S₂ in at least one objective

The PARETO FRONTIER is the set of all non-dominated solutions.
```

### 4.2 Visualization

```
Accuracy (%)
    │
100 ┤                                    ★ Claude 3.5 Sonnet
    │                              ★ GPT-4o
 95 ┤                    ★ Gemini Pro
    │              ◆ SLM-Optimized ←── TARGET REGION
 90 ┤        ★ Claude Haiku
    │  ★ Gemini Flash
 85 ┤
    │      ★ GPT-4o-mini
 80 ┤            ★ Llama 3B
    │
 75 ┤
    │
 70 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────▶
       $0.01 $0.1  $1   $10  $100                    Cost ($/1M tokens)
                                                     (log scale)

Legend:
  ★ = Baseline models
  ◆ = SLM candidates
  ─ = Pareto frontier
```

### 4.3 Frontier Computation Algorithm

```rust
/// Compute Pareto frontier from evaluation results
pub fn compute_pareto_frontier(results: &[EvalResult]) -> Vec<&EvalResult> {
    let mut frontier = Vec::new();

    for candidate in results {
        let is_dominated = results.iter().any(|other| {
            other.dominates(candidate)
        });

        if !is_dominated {
            frontier.push(candidate);
        }
    }

    // Sort by primary objective (accuracy) descending
    frontier.sort_by(|a, b| {
        b.accuracy.partial_cmp(&a.accuracy).unwrap_or(Ordering::Equal)
    });

    frontier
}

impl EvalResult {
    /// Check if self dominates other (better or equal in all, strictly better in one)
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
}
```

### 4.4 Trade-off Analysis

For each SLM candidate, compute:

1. **Accuracy Gap**: `Δ_acc = acc_frontier - acc_slm`
2. **Cost Ratio**: `R_cost = cost_frontier / cost_slm`
3. **Latency Ratio**: `R_lat = lat_frontier / lat_slm`
4. **Value Score**: `V = (1 - Δ_acc) × R_cost × R_lat`

**Target**: Find SLMs with V > 100 (100x better value than frontier) [14].

---

## 6. Ground Truth: Single-Shot Compilation

### 6.1 Corpus: reprorusted-python-cli

Our ground truth corpus is derived from `reprorusted-python-cli`, a collection of 299 Python CLI programs with 6,745+ tests:

```
../reprorusted-python-cli/examples/
├── example_abs/           # Absolute value CLI
│   ├── abs_tool.py       # Python source (INPUT)
│   └── test_abs_tool.py  # Python tests (REFERENCE)
├── example_age_calculator/
├── example_binary_codec/
├── ... (299 examples total)
└── example_zip_util/
```

### 6.2 Evaluation Protocol

For each Python example, we evaluate models on their ability to produce **functionally correct Rust**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE-SHOT COMPILATION GROUND TRUTH                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Python Source (abs_tool.py)                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         MODEL INFERENCE                               │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │  │
│  │  │ SLM (.apr)  │    │   Claude    │    │   Gemini    │              │  │
│  │  │ via aprender│    │   (SaaS)    │    │   (SaaS)    │              │  │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘              │  │
│  │         │                  │                  │                      │  │
│  │         └─────────────────┬┴─────────────────┘                      │  │
│  │                           │                                          │  │
│  │                     Rust Output                                      │  │
│  └───────────────────────────┼──────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     GROUND TRUTH VERIFICATION                         │  │
│  │                                                                       │  │
│  │  Step 1: cargo build --release                                       │  │
│  │          └── Must compile without errors                             │  │
│  │                                                                       │  │
│  │  Step 2: cargo test                                                  │  │
│  │          └── Must pass all generated tests                           │  │
│  │                                                                       │  │
│  │  Step 3: Behavioral equivalence (optional)                           │  │
│  │          └── Compare outputs with Python reference                   │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│                     ┌────────────────┐                                     │
│                     │   PASS / FAIL  │                                     │
│                     │   + Latency    │                                     │
│                     │   + Cost       │                                     │
│                     └────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Ground Truth Metrics

Following SWE-bench and Princeton methodology:

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **compile_pass** | `cargo build` succeeds | Binary (0/1) |
| **test_pass** | `cargo test` succeeds | Binary (0/1) |
| **pass@1** | First-attempt success rate | compile_pass ∧ test_pass |
| **behavioral_equiv** | Output matches Python reference | Property-based testing |

### 6.4 Corpus Statistics

```
Corpus: reprorusted-python-cli
─────────────────────────────
Examples:              299
Total Python files:    607
Lines of Python:       ~45,000
Test assertions:       6,745+

Complexity Distribution:
  Simple (< 50 LOC):   127 (42%)
  Medium (50-200 LOC): 134 (45%)
  Complex (> 200 LOC):  38 (13%)

Domain Coverage:
  CLI utilities:       89
  Data processing:     67
  Text manipulation:   54
  Math/algorithms:     43
  File operations:     46
```

### 6.5 Implementation

```rust
/// Ground truth verification for single-shot compilation
pub struct GroundTruthVerifier {
    /// Temporary directory for Rust projects
    temp_dir: PathBuf,
    /// Timeout for compilation
    compile_timeout: Duration,
    /// Timeout for tests
    test_timeout: Duration,
}

impl GroundTruthVerifier {
    /// Verify generated Rust code against ground truth
    pub fn verify(&self, rust_code: &str, example: &PythonExample) -> VerificationResult {
        // Create Cargo project
        let project_dir = self.create_cargo_project(rust_code)?;

        // Step 1: Compile
        let compile_start = Instant::now();
        let compile_result = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(&project_dir)
            .output()?;
        let compile_latency = compile_start.elapsed();

        if !compile_result.status.success() {
            return VerificationResult {
                compile_pass: false,
                test_pass: false,
                compile_latency,
                compile_error: Some(String::from_utf8_lossy(&compile_result.stderr).to_string()),
                ..Default::default()
            };
        }

        // Step 2: Run tests
        let test_start = Instant::now();
        let test_result = Command::new("cargo")
            .args(["test"])
            .current_dir(&project_dir)
            .output()?;
        let test_latency = test_start.elapsed();

        VerificationResult {
            compile_pass: true,
            test_pass: test_result.status.success(),
            compile_latency,
            test_latency,
            tests_run: parse_test_count(&test_result.stdout),
            tests_passed: parse_tests_passed(&test_result.stdout),
            ..Default::default()
        }
    }
}

/// Result of ground truth verification
#[derive(Debug, Default)]
pub struct VerificationResult {
    pub compile_pass: bool,
    pub test_pass: bool,
    pub compile_latency: Duration,
    pub test_latency: Duration,
    pub compile_error: Option<String>,
    pub tests_run: usize,
    pub tests_passed: usize,
}
```

### 6.6 Princeton Protocol Compliance

For each model, we run the evaluation **5 times** per example and report:

```rust
/// Princeton-compliant result aggregation
pub struct PrincetonResult {
    /// Model identifier
    pub model_id: String,
    /// Example identifier
    pub example_id: String,
    /// pass@1 rate across 5 runs
    pub pass_at_1: f64,
    /// 95% confidence interval (Student's t)
    pub ci_lower: f64,
    pub ci_upper: f64,
    /// Mean inference cost ($)
    pub mean_cost: f64,
    /// Mean total latency (inference + compile + test)
    pub mean_latency: Duration,
    /// Input/output token counts for cost recalculation
    pub input_tokens: usize,
    pub output_tokens: usize,
}
```

---

## 7. Benchmark Tasks

### 7.1 Task Selection Criteria

Following the HELM benchmark methodology [15] and BIG-Bench principles [16]:

1. **Real-world relevance**: Tasks reflect production use cases
2. **Discriminative power**: Tasks differentiate model capabilities (aligned with HumanEval [30] and MBPP [31] for code)
3. **Reproducibility**: Deterministic evaluation possible
4. **Domain coverage**: Multiple domains represented
5. **Automated Evaluation Validity**: Utilizing "LLM-as-a-Judge" paradigms with proven correlation to human preference [32, 33]

### 6.2 Task Suite

| Task ID | Domain | Description | Metric | Samples |
|---------|--------|-------------|--------|---------|
| `code-review` | Software | Identify bugs in code snippets | Accuracy | 2,000 |
| `sql-gen` | Data | Generate SQL from natural language | Exact Match | 1,500 |
| `sentiment` | NLP | Classify sentiment (positive/negative/neutral) | F1 | 3,000 |
| `summarize` | NLP | Summarize technical documents | ROUGE-L | 1,000 |
| `classify-intent` | NLP | Classify user intent (10 classes) | Accuracy | 2,500 |
| `extract-entities` | NLP | Named entity recognition | F1 | 2,000 |
| `translate-code` | Software | Transpile Python → Rust | BLEU | 1,000 |
| `qa-retrieval` | Knowledge | Answer questions given context | Exact Match | 2,000 |

### 6.3 Task Configuration

```yaml
# tasks/code-review.yaml
task:
  id: code-review
  description: "Identify bugs in code snippets"
  domain: software

evaluation:
  metric: accuracy
  samples: 2000
  timeout_ms: 5000

prompts:
  system: |
    You are a code reviewer. Identify if the following code contains a bug.
    Respond with ONLY "BUG" or "NO_BUG".
  user_template: |
    ```{language}
    {code}
    ```

ground_truth:
  source: "datasets/code-review-gold.jsonl"
  label_key: "has_bug"

slm_optimization:
  # Depyler hints for this task
  attention_heads_required: 4  # Minimum needed
  context_length: 512          # Task-specific
  quantization_viable: true    # Low precision OK
```

---

## 7. Implementation Plan

### 7.1 Directory Structure

```
single-shot-eval/
├── Cargo.toml                  # Dependencies: aprender, entrenar (crates.io ONLY)
├── Makefile                    # make evaluate, make report
├── CLAUDE.md                   # Claude Code guidance
├── README.md
├── .pmat-metrics.toml          # PMAT quality thresholds
├── docs/
│   ├── specifications/
│   │   └── eval-spec.md        # This document
│   └── roadmaps/
│       └── roadmap.yaml        # PMAT work tickets
├── src/
│   ├── lib.rs                  # Library exports
│   ├── config.rs               # YAML task configuration (THIS CRATE)
│   ├── pareto.rs               # Pareto frontier analysis (THIS CRATE - UNIQUE)
│   ├── runner.rs               # Task orchestration (THIS CRATE)
│   ├── report.rs               # Report generation (THIS CRATE - UNIQUE)
│   └── baseline.rs             # CLI baseline wrappers (shell exec to claude/gemini)
│   # NOTE: NO src/slm/ directory - use aprender directly for .apr loading/inference
│   # NOTE: NO src/metrics.rs with ML metrics - use aprender::metrics
│   # NOTE: NO HTTP API clients - baselines via shell exec ONLY
├── tasks/
│   ├── code-review.yaml
│   ├── sql-gen.yaml
│   └── ...
├── datasets/
│   ├── code-review-gold.jsonl
│   └── ...
├── models/
│   └── *.apr                   # .apr format ONLY (use entrenar convert for others)
├── results/
│   └── .gitkeep
├── benches/
│   └── pareto.rs               # Criterion benchmarks
└── tests/
    ├── integration/
    └── unit/

# FORBIDDEN: Do NOT create these (functionality exists in dependencies)
# ❌ src/slm/              - Use aprender directly
# ❌ src/metrics.rs        - Use aprender::metrics
# ❌ src/inference.rs      - Use aprender
# ❌ src/distillation.rs   - Use entrenar-distill
# ❌ src/quantization.rs   - Use entrenar
```

### 7.2 Makefile Targets

```makefile
# Primary evaluation target (.apr models ONLY)
evaluate: check-deps validate-apr-format
	@echo "Running full evaluation suite..."
	cargo run --release -- evaluate \
		--tasks tasks/*.yaml \
		--models models/*.apr \
		--baselines claude-haiku,gemini-flash \
		--output results/$(shell date +%Y%m%d_%H%M%S)

# MANDATORY: Validate all models are .apr format
validate-apr-format:
	@echo "Validating model format (.apr only)..."
	@for f in models/*; do \
		if [ "$${f##*.}" != "apr" ]; then \
			echo "ERROR: Non-.apr model found: $$f"; \
			echo "Convert using: entrenar convert --input $$f --output $${f%.*}.apr"; \
			exit 1; \
		fi; \
	done
	@echo "✓ All models are .apr format"

# Convert external model formats to .apr (uses entrenar)
convert:
	@echo "Converting models to .apr format..."
	entrenar convert --input $(INPUT) --output $(OUTPUT)

# Quick smoke test
evaluate-quick:
	cargo run --release -- evaluate \
		--tasks tasks/sentiment.yaml \
		--samples 100 \
		--baselines claude-haiku

# Generate Pareto frontier report
report:
	cargo run --release -- report \
		--input results/latest \
		--output reports/pareto-analysis.md

# PMAT integration (required for all commits)
pmat-quality-gate:
	pmat quality-gate --strict

# CI validation (EXTREME TDD: 95% coverage, zero clippy warnings)
ci: fmt-check clippy test coverage-check pmat-quality-gate
	@echo "CI passed"

.PHONY: evaluate evaluate-quick report convert validate-apr-format ci
```

### 7.3 Implementation Phases

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **P0: Foundation** | Week 1-2 | Config parsing, metric framework, task runner |
| **P1: Baselines** | Week 3 | Claude/Gemini/OpenAI API clients |
| **P2: SLM Integration** | Week 4 | Aprender inference, APR loading |
| **P3: Pareto Analysis** | Week 5 | Frontier computation, visualization |
| **P4: Depyler Integration** | Week 6 | Static analysis, optimization recommendations |
| **P5: Reporting** | Week 7 | Markdown reports, charts, CI integration |
| **P6: Documentation** | Week 8 | User guide, API docs, examples |

---

## 8. Quality Gates

### 8.1 25-Point Evaluation Checklist

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | `cargo build --release` succeeds | Build | Exit code 0 |
| 2 | `cargo test` passes (>95% coverage) | Test | All green, coverage verified |
| 3 | `cargo clippy -- -D warnings` clean | Lint | Zero warnings |
| 4 | `cargo fmt --check` passes | Format | No changes needed |
| 5 | API keys validated at startup | Config | Clear error if missing |
| 6 | Tasks load without error | Config | Schema validation passes |
| 7 | Metrics computed correctly (golden tests) | Correctness | ±0.1% of known values |
| 8 | Pareto dominance correct (unit tests) | Correctness | All edge cases covered |
| 9 | Statistical tests valid (p-values) | Statistics | Bootstrap CI working |
| 10 | Baseline API calls succeed | Integration | Rate limiting handled |
| 11 | SLM inference matches Aprender | Integration | Bit-exact results |
| 12 | Depyler analysis runs | Integration | Report generated |
| 13 | Results reproducible (fixed seed) | Reproducibility | 3 runs identical |
| 14 | Timeout handling works | Resilience | No hangs on slow models |
| 15 | Memory bounded (<8GB) | Resources | Peak RSS monitored |
| 16 | Pareto frontier plotted | Visualization | PNG/SVG generated |
| 17 | Markdown report generated | Reporting | Valid markdown |
| 18 | CI pipeline green | DevOps | GitHub Actions pass |
| 19 | Documentation complete | Docs | README, API docs |
| 20 | Examples run successfully | Usability | All examples work |
| 21 | Error messages actionable | UX | Clear guidance |
| 22 | Logging comprehensive | Observability | DEBUG level available |
| 23 | IIUR compliance | Methodology | Isolated, idempotent |
| 24 | Toyota Way documented | Process | Principles applied |
| 25 | Security audit clean | Security | No credential leaks |

### 8.2 Acceptance Criteria for SLM Success

An SLM is considered **production-viable** if:

1. **Accuracy**: Within 5% of best frontier model on target task
2. **Cost**: At least 100x cheaper per inference
3. **Latency**: p99 < 100ms (frontier p99 often > 1s)
4. **Consistency**: >99% deterministic outputs
5. **Pareto Status**: Non-dominated or within 2% of frontier

---

## 9. Peer-Reviewed Citations

### Foundational Works on Model Efficiency

[1] Hinton, G., Vinyals, O., & Dean, J. (2015). **Distilling the Knowledge in a Neural Network**. *NeurIPS Workshop*. https://arxiv.org/abs/1503.02531

[2] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. *NeurIPS EMC² Workshop*. https://arxiv.org/abs/1910.01108

[3] Sun, S., Cheng, Y., Gan, Z., & Liu, J. (2019). **Patient Knowledge Distillation for BERT Model Compression**. *EMNLP*. https://arxiv.org/abs/1908.09355

[4] Jiao, X., et al. (2020). **TinyBERT: Distilling BERT for Natural Language Understanding**. *EMNLP Findings*. https://arxiv.org/abs/1909.10351

[5] Sun, Z., et al. (2020). **MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices**. *ACL*. https://arxiv.org/abs/2004.02984

### Pareto Frontier and Multi-Objective Optimization

[6] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). **A fast and elitist multiobjective genetic algorithm: NSGA-II**. *IEEE Transactions on Evolutionary Computation*. https://doi.org/10.1109/4235.996017

[7] Miettinen, K. (1999). **Nonlinear Multiobjective Optimization**. *Springer*. ISBN: 978-0792382782

[8] Emmerich, M., & Deutz, A. (2018). **A tutorial on multiobjective optimization: fundamentals and evolutionary methods**. *Natural Computing*. https://doi.org/10.1007/s11047-018-9685-y

### Evaluation Methodology

[9] Liang, P., et al. (2022). **Holistic Evaluation of Language Models (HELM)**. *Stanford CRFM*. https://arxiv.org/abs/2211.09110

[10] Srivastava, A., et al. (2022). **Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models (BIG-Bench)**. *arXiv*. https://arxiv.org/abs/2206.04615

[11] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). **On Calibration of Modern Neural Networks**. *ICML*. https://arxiv.org/abs/1706.04599

[12] Papineni, K., et al. (2002). **BLEU: a Method for Automatic Evaluation of Machine Translation**. *ACL*. https://aclanthology.org/P02-1040/

[13] Lin, C.-Y. (2004). **ROUGE: A Package for Automatic Evaluation of Summaries**. *ACL Workshop*. https://aclanthology.org/W04-1013/

### Quantization and Compression

[14] Dettmers, T., et al. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. *NeurIPS*. https://arxiv.org/abs/2208.07339

[15] Frantar, E., et al. (2023). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**. *ICLR*. https://arxiv.org/abs/2210.17323

[16] Lin, J., et al. (2023). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**. *arXiv*. https://arxiv.org/abs/2306.00978

### Production ML Systems

[17] Sculley, D., et al. (2015). **Hidden Technical Debt in Machine Learning Systems**. *NeurIPS*. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems

[18] Polyzotis, N., et al. (2018). **Data Lifecycle Challenges in Production Machine Learning**. *SIGMOD Record*. https://doi.org/10.1145/3299887.3299891

[19] Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2022). **Challenges in Deploying Machine Learning: a Survey of Case Studies**. *ACM Computing Surveys*. https://arxiv.org/abs/2011.09926

### NASA and Safety-Critical AI

[20] NASA. (2020). **NASA's Principles for Ethical Use of Artificial Intelligence**. *NASA Technical Reports*. https://www.nasa.gov/ai-principles

[21] Amodei, D., et al. (2016). **Concrete Problems in AI Safety**. *arXiv*. https://arxiv.org/abs/1606.06565

[22] Hendrycks, D., et al. (2021). **Unsolved Problems in ML Safety**. *arXiv*. https://arxiv.org/abs/2109.13916

### Google AI Practices

[23] Google. (2023). **Google AI Principles**. https://ai.google/responsibility/principles/

[24] Zinkevich, M. (2017). **Rules of Machine Learning: Best Practices for ML Engineering**. *Google Research*. https://developers.google.com/machine-learning/guides/rules-of-ml

[25] Breck, E., et al. (2017). **The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction**. *IEEE BigData*. https://research.google/pubs/pub46555/

### Modern SLM Architectures & Reasoning

[26] Gunasekar, S., et al. (2023). **Textbooks Are All You Need**. *arXiv*. https://arxiv.org/abs/2306.11644

[27] Mukherjee, S., et al. (2023). **Orca: Progressive Learning from Complex Explanation Traces of GPT-4**. *arXiv*. https://arxiv.org/abs/2306.02707

[28] Hoffmann, J., et al. (2022). **Training Compute-Optimal Large Language Models**. *NeurIPS*. https://arxiv.org/abs/2203.15556

[29] Wei, J., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. *NeurIPS*. https://arxiv.org/abs/2201.11903

### Code & Automated Evaluation

[30] Chen, M., et al. (2021). **Evaluating Large Language Models Trained on Code**. *arXiv*. https://arxiv.org/abs/2107.03374

[31] Austin, J., et al. (2021). **Program Synthesis with Large Language Models**. *arXiv*. https://arxiv.org/abs/2108.07732

[32] Zheng, L., et al. (2023). **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**. *NeurIPS*. https://arxiv.org/abs/2306.05685

[33] Dubois, Y., et al. (2024). **AlpacaEval 2.0: A Better Leaderboard for Instruction-Tuned LLMs**. *arXiv*. https://arxiv.org/abs/2305.14387

### Efficient Inference & Architecture

[34] Dao, T., et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. *NeurIPS*. https://arxiv.org/abs/2205.14135

[35] Kwon, W., et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention**. *SOSP*. https://arxiv.org/abs/2309.06180

### Princeton Research (Critical Methodology Sources)

[36] Kapoor, S., et al. (2024). **AI Agents That Matter**. *Princeton University*. https://arxiv.org/abs/2407.01502 / https://agents.cs.princeton.edu/
- Key findings: Simple baselines offer Pareto improvements over SOTA agents; 5 runs minimum with 95% CI; convex Pareto frontiers; dollar costs not proxies

[37] Jimenez, C. E., et al. (2024). **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**. *ICLR 2024 (Oral)*. https://arxiv.org/abs/2310.06770
- Key methodology: FAIL_TO_PASS/PASS_TO_PASS tests; ground truth via test execution; Docker-based reproducible evaluation

[38] OpenAI. (2024). **Introducing SWE-bench Verified**. https://openai.com/index/introducing-swe-bench-verified/
- Key insight: 500 human-verified solvable problems; engineer-confirmed ground truth

[39] Gu, A., et al. (2024). **Are "Solved Issues" in SWE-bench Really Solved Correctly?**. *arXiv*. https://arxiv.org/abs/2503.15223
- Critical finding: 7.8% of "correct" patches fail developer tests; 29.6% induce behavioral divergence

[40] Park, C., et al. (2024). **Towards Pareto Optimal Throughput in Small Language Model Serving**. *arXiv*. https://arxiv.org/abs/2404.03353
- SLM-specific: Pareto-optimal throughput within single accelerator capacity

---

## 10. Appendix: Toyota Way for AI Evaluation

### A.1 Principles Applied

| Toyota Principle | AI Evaluation Application |
|------------------|--------------------------|
| **Genchi Genbutsu** (Go and see) | Manually inspect model outputs, don't just trust metrics |
| **Jidoka** (Quality built-in) | Automated quality gates catch regressions immediately |
| **Kaizen** (Continuous improvement) | Each evaluation cycle improves the SLM |
| **Heijunka** (Leveling) | Consistent evaluation cadence, no batch-only testing |
| **Muda** (Eliminate waste) | Remove evaluation steps that don't provide signal |
| **Poka-yoke** (Error-proofing) | Schema validation, type safety, CI gates |
| **Nemawashi** (Consensus building) | Document decisions, share evaluation reports |

### A.2 Google AI Adaptation

| Google Principle | Implementation |
|------------------|----------------|
| **Empirical rigor** | Statistical significance testing, bootstrap CIs |
| **Reproducibility** | Fixed seeds, deterministic execution, version pinning |
| **Ablation studies** | Isolate contribution of each optimization |
| **Negative results** | Document what doesn't work (prevents repeating failures) |

### A.3 NASA AI Adaptation

| NASA Standard | Implementation |
|---------------|----------------|
| **V&V (Verification & Validation)** | Unit tests verify correctness; integration tests validate behavior |
| **Failure Mode Analysis** | Document edge cases where SLMs fail vs. frontier |
| **Uncertainty Quantification** | Report confidence intervals, not just point estimates |
| **Mission Assurance** | Quality gates prevent deployment of degraded models |
| **Independent Review** | Evaluation code reviewed separately from model code |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0-draft | 2025-12-08 | Claude Code | Initial specification |
| 1.1.0 | 2025-12-08 | Claude Code | OFFLINE-FIRST: No HTTP APIs, .apr format mandatory, use aprender/entrenar exclusively |
| 1.2.0 | 2025-12-08 | Claude Code | Princeton research integration: "AI Agents That Matter" methodology (5 runs, 95% CI, convex Pareto), SWE-bench ground truth (FAIL_TO_PASS), reprorusted-python-cli corpus, single-shot compilation evaluation protocol |
| 1.2.1 | 2025-12-08 | Claude Code | Clarified strategic context: POC framework for ANY agent evaluation, Python→Rust is test case (depyler solves it better), methodology migrates to aprender for scale |

---

**Next Steps**:
1. Review specification with stakeholders
2. Create repository structure
3. Implement P0 foundation
4. Begin baseline API integrations
5. Distill first SLM candidate using Entrenar
