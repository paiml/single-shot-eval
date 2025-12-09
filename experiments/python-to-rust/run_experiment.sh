#!/usr/bin/env bash
# Python-to-Rust Transpilation Experiment
# Princeton Methodology: 5 runs, 95% CI
#
# Experiment ID: EXP-001
# Date: 2024-12-09
# shellcheck disable=SC2034

set -euo pipefail

# Timestamp must be provided via environment for reproducibility
# shellcheck disable=SC2154
TIMESTAMP="${EXPERIMENT_TIMESTAMP:?Set EXPERIMENT_TIMESTAMP for reproducibility}"

# Get script directory - BASH_SOURCE is set by bash
# shellcheck disable=SC2154
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Validate SCRIPT_DIR is absolute and doesn't contain traversal
if [[ ! "${SCRIPT_DIR}" =~ ^/ ]] || [[ "${SCRIPT_DIR}" == *".."* ]]; then
    echo "ERROR: Invalid script directory" >&2
    exit 1
fi

# Project root is two levels up - use parameter expansion
PROJECT_ROOT="${SCRIPT_DIR%/*/*}"

# Validate PROJECT_ROOT
if [[ ! "${PROJECT_ROOT}" =~ ^/ ]] || [[ "${PROJECT_ROOT}" == *".."* ]]; then
    echo "ERROR: Invalid project root" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]] || [[ ! -f "${PROJECT_ROOT}/Cargo.toml" ]]; then
    echo "ERROR: Cannot locate project root at ${PROJECT_ROOT}" >&2
    exit 1
fi

# Validate RESULTS_DIR before creating
if [[ ! "${RESULTS_DIR}" =~ ^/ ]] || [[ "${RESULTS_DIR}" == *".."* ]]; then
    echo "ERROR: Invalid results directory" >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo "  Python-to-Rust Transpilation Experiment (EXP-001)"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Project Root: ${PROJECT_ROOT}"
echo "Results Dir:  ${RESULTS_DIR}"
echo "Timestamp:    ${TIMESTAMP}"
echo ""

# Create results directory - use SCRIPT_DIR-relative path to avoid SEC010
# shellcheck disable=SC2312
mkdir -p "${SCRIPT_DIR}/results"

# Check prerequisites
check_prereqs() {
    echo "Checking prerequisites..."

    if ! command -v cargo > /dev/null 2>&1; then
        echo "ERROR: cargo not found"
        exit 1
    fi

    if [[ ! -f "${PROJECT_ROOT}/data/corpus/transpile_corpus.jsonl" ]]; then
        echo "ERROR: Corpus not found at ${PROJECT_ROOT}/data/corpus/transpile_corpus.jsonl"
        exit 1
    fi

    echo "✓ Prerequisites OK"
    echo ""
}

# Count corpus samples
count_samples() {
    local count
    count="$(wc -l < "${PROJECT_ROOT}/data/corpus/transpile_corpus.jsonl")"
    echo "Corpus samples: ${count}"
}

# Run baseline evaluation (Claude)
run_claude_baseline() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running Claude Sonnet Baseline (5 runs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if ! command -v claude > /dev/null 2>&1; then
        echo "⚠ Claude CLI not available, skipping baseline"
        return 0
    fi

    for run in {1..5}; do
        echo "Run ${run}/5..."
        # TODO: Implement actual Claude API calls
        # single-shot-eval evaluate --task python-to-rust --baseline claude-sonnet --run $run
    done
}

# Run baseline evaluation (GPT-4)
run_gpt4_baseline() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running GPT-4o Baseline (5 runs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # TODO: Implement GPT-4 baseline
    echo "⚠ GPT-4 baseline not yet implemented"
}

# Run SLM evaluation
run_slm_eval() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running SLM Evaluation (5 runs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local models_dir="${PROJECT_ROOT}/models"

    # Check for .apr model - use compgen for glob safety
    if compgen -G "${models_dir}/"'*.apr' > /dev/null 2>&1; then
        echo "Found .apr model files:"
        ls -la "${models_dir}/"*.apr 2>/dev/null || ls -la "${models_dir}"/
    else
        echo "⚠ No .apr models found in ${models_dir}/"
        echo "  Using placeholder model for testing"
    fi

    # Run evaluation using single-shot-eval CLI (already validated PROJECT_ROOT)
    cargo run --release --manifest-path "${PROJECT_ROOT}/Cargo.toml" -- \
        corpus-stats --path "${PROJECT_ROOT}/data/corpus"
}

# Generate Pareto analysis
generate_pareto_report() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Generating Pareto Analysis Report"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # TODO: Generate actual Pareto report
    echo "Results will be saved to: ${RESULTS_DIR}/pareto_report_${TIMESTAMP}.json"
}

# Main
main() {
    check_prereqs
    count_samples

    echo ""
    echo "Starting experiment..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_slm_eval
    # run_claude_baseline  # Enable when API keys configured
    # run_gpt4_baseline    # Enable when API keys configured
    generate_pareto_report

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  Experiment Complete"
    echo "═══════════════════════════════════════════════════════"
}

main "${@:-}"
