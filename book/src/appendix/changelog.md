# Changelog

## [Unreleased]

### Added
- mdBook documentation
- PMAT quality gates configuration
- CI coverage reporting with Codecov
- Security audit job with cargo-audit
- Multi-platform matrix builds

## [0.1.0] - Initial Release

### Added
- Core evaluation framework
- Pareto frontier analysis with multi-dimensional support
- Bootstrap confidence intervals (1000 iterations)
- Mann-Whitney U significance testing
- Py2Rs 10-level benchmark classification
- CLI with `run`, `benchmark`, and `pareto` commands
- YAML task configuration
- JSONL corpus support
- Baseline comparison via CLI wrappers
- Integration with Batuta stack (aprender, alimentar)
- JSON, table, and markdown output formats

### Architecture
- Offline-first design (no HTTP dependencies)
- Pure Rust implementation
- Modular structure (config, corpus, runner, report, pareto)
