#!/usr/bin/env bash
# Benchmark runner that updates README.md with results
# Usage: ./scripts/bench-update-readme.sh
#
# Parses Criterion benchmark output and updates README.md between markers
# shellcheck disable=SC2154

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${SCRIPT_DIR%/*}"
README="${PROJECT_DIR}/README.md"
BENCH_OUTPUT="${PROJECT_DIR}/target/bench_output.txt"

# Validate paths
if [[ ! "${PROJECT_DIR}" =~ ^/ ]] || [[ "${PROJECT_DIR}" == *".."* ]]; then
    echo "ERROR: Invalid project directory" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/Cargo.toml" ]]; then
    echo "ERROR: Cannot locate project root at ${PROJECT_DIR}" >&2
    exit 1
fi

echo "=== Pareto Frontier Benchmark Runner ==="
echo "Project: ${PROJECT_DIR}"
echo ""

# Run benchmarks and capture output
run_benchmarks() {
    echo "Running benchmarks..."
    cargo bench --manifest-path "${PROJECT_DIR}/Cargo.toml" 2>&1 | tee "${BENCH_OUTPUT}"
    echo ""
    echo "Benchmarks complete."
}

# Parse Criterion output into markdown table
generate_table() {
    local current_bench=""
    local hostname_val
    local date_val

    hostname_val="$(hostname)"
    # DET002: timestamp is intentional for benchmark tracking
    # shellcheck disable=SC2312
    date_val="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    echo ""
    echo "### Pareto Frontier Benchmarks"
    echo ""
    echo "Performance benchmarks for Pareto frontier computation on varying model counts."
    echo ""
    echo "| Benchmark | Models | Time (median) | Lower bound | Upper bound |"
    echo "|-----------|--------|---------------|-------------|-------------|"

    # Parse lines like:
    # pareto_frontier/compute_10_models
    #                         time:   [59.892 ns 60.147 ns 60.414 ns]
    while IFS= read -r line; do
        if [[ "${line}" =~ ^pareto_frontier/compute_([0-9]+)_models ]]; then
            current_bench="${BASH_REMATCH[1]}"
        elif [[ "${line}" =~ time:\ +\[([0-9.]+)\ +([a-zµ]+)\ +([0-9.]+)\ +([a-zµ]+)\ +([0-9.]+)\ +([a-zµ]+)\] ]] && [[ -n "${current_bench}" ]]; then
            local lower="${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
            local median="${BASH_REMATCH[3]} ${BASH_REMATCH[4]}"
            local upper="${BASH_REMATCH[5]} ${BASH_REMATCH[6]}"
            echo "| \`compute_pareto_frontier\` | ${current_bench} | ${median} | ${lower} | ${upper} |"
            current_bench=""
        fi
    done < "${BENCH_OUTPUT}"

    echo ""
    echo "_Benchmarks run on ${hostname_val} at ${date_val}_"
    echo ""
}

# Update README with benchmark table
update_readme() {
    local table_content="$1"
    local start_marker="<!-- BENCHMARK_TABLE_START -->"
    local end_marker="<!-- BENCHMARK_TABLE_END -->"

    if [[ ! -f "${README}" ]]; then
        echo "ERROR: README.md not found at ${README}" >&2
        exit 1
    fi

    # Check if markers exist
    if grep -q "${start_marker}" "${README}"; then
        # Replace existing table using awk
        awk -v start="${start_marker}" -v end="${end_marker}" -v table="${table_content}" '
            $0 ~ start { print; print table; skip=1; next }
            $0 ~ end { skip=0 }
            !skip { print }
        ' "${README}" > "${README}.tmp" && mv "${README}.tmp" "${README}"
        echo "Updated existing benchmark table in README.md"
    else
        echo "WARNING: Benchmark table markers not found in README.md"
        echo "Add these markers to README.md where you want the table:"
        echo "  ${start_marker}"
        echo "  ${end_marker}"
        echo ""
        echo "Generated table:"
        echo "${table_content}"
    fi
}

# Main
main() {
    local table_result
    run_benchmarks
    table_result="$(generate_table)"
    update_readme "${table_result}"
    echo ""
    echo "=== Benchmark Update Complete ==="
}

main "${@:-}"
