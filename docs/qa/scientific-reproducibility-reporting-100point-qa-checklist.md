# Scientific Reproducibility Reporting: 100-Point QA Checklist

**Document Classification**: SCIENTIFIC REPRODUCIBILITY PROTOCOL
**Philosophical Framework**: Popperian Falsificationism + Toyota Way Continuous Improvement
**Target Audience**: External University Replication Teams
**Version**: 1.1.0 (Enhanced Draft)
**Date**: 2025-12-09

---

## Abstract

This document establishes a **100-point scientific reproducibility checklist** for the `single-shot-eval` ML evaluation framework. It is designed for independent university research teams to systematically replicate all experimental claims, following Karl Popper's principles of falsifiability and the Toyota Production System's commitment to zero defects. It incorporates modern best practices for NLP evaluation including behavioral testing [27], data documentation [28], and environmental reporting [30].

**Core Premise**: A claim that cannot be independently replicated is not scientific knowledge [1].

---

## Theoretical Foundation

### Popperian Falsificationism

Karl Popper's philosophy of science holds that scientific theories must be **falsifiable**—they must make predictions that could, in principle, be proven wrong [1, 2]. This checklist operationalizes Popperian principles:

| Popperian Principle | Application to ML Evaluation |
|---------------------|------------------------------|
| **Falsifiability** | Every accuracy claim must specify conditions under which it would be false |
| **Severe Testing** | Results must withstand adversarial replication attempts |
| **Critical Rationalism** | Accept criticism, modify claims when evidence contradicts |
| **Verisimilitude** | Prefer theories closer to truth through iterative refinement |
| **Demarcation** | Distinguish scientific claims from unfalsifiable marketing |

### Toyota Way Scientific Application

The Toyota Production System provides methodological rigor for reproducible research [3, 4]:

| TPS Principle | Scientific Research Application |
|---------------|--------------------------------|
| **Genchi Genbutsu** (現地現物) | Examine raw data, not summaries |
| **Jidoka** (自働化) | Stop when anomalies detected; don't proceed with flawed data |
| **Kaizen** (改善) | Iterative improvement through replication failures |
| **Hansei** (反省) | Critical self-reflection on negative results |
| **Heijunka** (平準化) | Consistent experimental conditions across replications |
| **Poka-yoke** (ポカヨケ) | Error-proof experimental protocols |
| **Nemawashi** (根回し) | Build consensus through transparent methodology |

---

## Table of Contents

1. [Experimental Design Reproducibility (EP-001 to EP-015)](#1-experimental-design-reproducibility)
2. [Data Provenance and Integrity (DP-001 to DP-015)](#2-data-provenance-and-integrity)
3. [Statistical Methodology Verification (SM-001 to SM-015)](#3-statistical-methodology-verification)
4. [Computational Reproducibility (CR-001 to CR-015)](#4-computational-reproducibility)
5. [Model Evaluation Protocol (ME-001 to ME-010)](#5-model-evaluation-protocol)
6. [Result Reporting Standards (RR-001 to EP-010)](#6-result-reporting-standards)
7. [Falsification Protocol (FP-001 to FP-010)](#7-falsification-protocol)
8. [Replication Instructions (RI-001 to RI-010)](#8-replication-instructions)
9. [References](#references)

---

## 1. Experimental Design Reproducibility

### 1.1 Pre-Registration and Hypothesis Specification

Following Popper's emphasis on **bold conjectures** that can be severely tested [1]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| EP-001 | **Primary hypothesis stated a priori** | Locate in eval-spec.md before data collection | Hypothesis modified after seeing results | ☑ |
| EP-002 | **Quantitative predictions specified** | "SLM achieves ≥90% of frontier accuracy at ≤1% cost" | Accuracy gap >10% OR cost ratio >5% | ☑ |
| EP-003 | **Success criteria defined before experiment** | Threshold values documented | Thresholds adjusted post-hoc | ☑ |
| EP-004 | **Sample size justification** | Power analysis or precedent cited [10] | Arbitrary sample sizes | ☑ |
| EP-005 | **Stopping rules pre-specified** | When to conclude experiment | Early stopping on favorable results | ☐ |

### 1.2 Experimental Controls

Toyota Way: **Standardized Work** (標準作業) ensures consistent conditions [4]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| EP-006 | **Positive control included** | Known-good model produces expected results | Positive control fails | ☑ |
| EP-007 | **Negative control included** | Random baseline performs at chance | Negative control beats SLM | ☑ |
| EP-008 | **Baseline comparisons fair** | Same prompts, same data [33] | Asymmetric conditions favor SLM | ☑ |
| EP-009 | **Confounding variables identified** | Hardware, prompt engineering, data leakage | Uncontrolled confounders affect results | ☐ |
| EP-010 | **Blinding where applicable** | Evaluator unaware of model identity | Selection bias in manual evaluation | ☐ |

### 1.3 Experimental Protocol Documentation

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| EP-011 | **Complete protocol available** | Step-by-step instructions | Missing critical steps | ☑ |
| EP-012 | **Parameter settings documented** | All hyperparameters listed [31] | Undisclosed tuning | ☐ |
| EP-013 | **Hardware specifications recorded** | CPU, RAM, GPU if used | Results hardware-dependent | ☐ |
| EP-014 | **Software versions locked** | Cargo.lock, rustc version | Version drift affects results | ☑ |
| EP-015 | **Execution order randomized** | Model evaluation order randomized | Order effects present | ☑ |

---

## 2. Data Provenance and Integrity

### 2.1 Data Source Verification

Genchi Genbutsu: **Go and see the actual data** [3]; enforce Datasheets standards [28]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| DP-001 | **Data sources documented** | Datasheet for Dataset created [28] | Undisclosed/Ambiguous sources | ☐ |
| DP-002 | **Data collection dates recorded** | Timestamp for corpus creation | Temporal contamination possible | ☑ |
| DP-003 | **Data licensing verified** | MIT/Apache/CC licenses | Copyright violations | ☑ |
| DP-004 | **No training/test contamination** | Decontamination protocol checked [14] | Data leakage detected | ☐ |
| DP-005 | **Data version controlled** | Git SHA or content hash | Data changed between runs | ☑ |

### 2.2 Data Quality Assurance

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| DP-006 | **Missing data handling documented** | Imputation or exclusion rules | Ad-hoc missing data decisions | ☑ |
| DP-007 | **Outlier treatment specified** | Inclusion/exclusion criteria | Selective outlier removal | ☐ |
| DP-008 | **Data preprocessing reproducible** | Script provided | Manual preprocessing steps | ☑ |
| DP-009 | **Ground truth verification** | How correctness determined | Ambiguous ground truth | ☑ |
| DP-010 | **Inter-rater reliability** (if manual) | Cohen's kappa ≥ 0.8 | Low agreement | N/A |

### 2.3 Data Integrity Checks

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| DP-011 | **Checksums for all data files** | SHA-256 hashes provided | Hash mismatch | ☑ |
| DP-012 | **No duplicate examples** | Unique ID verification | Duplicate inflation | ☑ |
| DP-013 | **Class balance documented** | Distribution across categories | Hidden class imbalance | ☑ |
| DP-014 | **Difficulty distribution specified** | Py2Rs level breakdown | Cherry-picked easy examples | ☑ |
| DP-015 | **Data representativeness argued** | Sampling methodology | Selection bias | ☐ |

---

## 3. Statistical Methodology Verification

### 3.1 Estimation Procedures

Following ASA guidelines on statistical practice [5, 6]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| SM-001 | **Point estimates with uncertainty** | Mean ± CI for all metrics | Point estimates without CI | ☑ |
| SM-002 | **Confidence level specified** | 95% CI standard | Ambiguous confidence level | ☑ |
| SM-003 | **Bootstrap procedure documented** | Resampling method, n=10,000 [9] | Insufficient resamples (<1,000) | ☑ |
| SM-004 | **Seed for reproducibility** | Fixed RNG seed provided | Non-reproducible randomness | ☑ |
| SM-005 | **Distributional assumptions stated** | Normality tests if parametric | Violated assumptions | ☑ |

### 3.2 Hypothesis Testing

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| SM-006 | **Alpha level pre-specified** | α = 0.05 documented | Alpha adjusted post-hoc | ☑ |
| SM-007 | **Multiple comparison correction** | Bonferroni/Holm for >1 test | Uncorrected multiple testing | ☑ |
| SM-008 | **Effect sizes reported** | Cohen's d or equivalent | Only p-values reported | ☑ |
| SM-009 | **Power analysis performed** | Detectable effect size | Underpowered study | ☑ |
| SM-010 | **Two-tailed tests unless justified** | Directional hypothesis rationale | Post-hoc one-tailed tests | ☑ |

### 3.3 Result Interpretation

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| SM-011 | **Statistical vs. practical significance** | Effect size interpretation | Trivial effects overclaimed | ☑ |
| SM-012 | **Negative results reported** | All comparisons, not just significant | Selective reporting | ☑ |
| SM-013 | **Confidence interval interpretation** | Correct probabilistic statement | Misinterpretation of CI | ☑ |
| SM-014 | **p-value interpretation** | Not "probability of hypothesis" | Inverse probability fallacy [6] | ☑ |
| SM-015 | **Limitations acknowledged** | Section on statistical limitations | Overclaiming from statistics | ☑ |

---

## 4. Computational Reproducibility

### 4.1 Environment Specification

Toyota Way: **Standardization** before Kaizen [4]; managing Technical Debt [32]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| CR-001 | **Rust version specified** | MSRV in Cargo.toml | Version drift changes results | ☐ |
| CR-002 | **All dependencies locked** | Cargo.lock committed | Dependency resolution differs | ☑ |
| CR-003 | **Operating system documented** | Linux kernel version | OS-specific behavior | ☑ |
| CR-004 | **Hardware requirements stated** | Minimum RAM, CPU, GPU | Results hardware-dependent | ☐ |
| CR-005 | **No network dependencies** | OFFLINE-FIRST verified | Hidden network calls | ☑ |

### 4.2 Build Reproducibility

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| CR-006 | **Clean build succeeds** | `cargo clean && cargo build --release` | Build failure | ☑ |
| CR-007 | **All tests pass** | `cargo test` | Test failures | ☑ |
| CR-008 | **No compiler warnings** | `-D warnings` flag | Warnings present | ☑ |
| CR-009 | **Documentation builds** | `cargo doc --no-deps` | Doc build failure | ☑ |
| CR-010 | **Examples execute** | `cargo run --example demo` | Example failure | ☑ |

### 4.3 Execution Reproducibility

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| CR-011 | **Deterministic output** | Same input → same output | Non-deterministic results | ☐ |
| CR-012 | **Floating-point stability** | IEEE 754 compliance | Numerical instability | ☑ |
| CR-013 | **Parallel execution determinism** | Thread-safe reproducibility | Race condition affects results | ☑ |
| CR-014 | **Memory bounds respected** | No OOM on documented hardware | Memory exhaustion | ☑ |
| CR-015 | **Execution time bounded** | Completes within timeout | Infinite loops possible | ☑ |

---

## 5. Model Evaluation Protocol

### 5.1 Evaluation Methodology

Following ML evaluation best practices [7, 8] and HELM standards [26]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| ME-001 | **Metric definitions provided** | Mathematical formula for accuracy | Ambiguous metric definition | ☑ |
| ME-002 | **Evaluation code available** | Open source implementation | Closed evaluation | ☑ |
| ME-003 | **Prompt templates documented** | Exact prompts used [31] | Prompt engineering advantage | ☑ |
| ME-004 | **Temperature and sampling documented** | Inference parameters | Hidden parameter tuning | ☐ |
| ME-005 | **Multiple runs performed** | ≥5 runs with different seeds | Single run reported | ☑ |

### 5.2 Baseline Comparisons

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| ME-006 | **Baselines use same evaluation** | Identical test conditions [33] | Asymmetric evaluation | ☑ |
| ME-007 | **Baseline versions specified** | API version, model checkpoint | Version drift | ☑ |
| ME-008 | **Cost & Energy calculations** | $/token and kWh/inference [30] | Misleading/Hidden costs | ☑ |
| ME-009 | **Latency measurement accurate** | Wall clock, not CPU time | Unfair timing | ☑ |
| ME-010 | **Pareto analysis methodology** | Dominance calculation documented | Incorrect Pareto frontier | ☑ |

---

## 6. Result Reporting Standards

### 6.1 Transparency Requirements

Following CONSORT, STROBE [9], and Model Card [29] guidelines:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| RR-001 | **All experiments reported** | No selective publication | Evidence of file-drawer effect | ☑ |
| RR-002 | **Failed experiments documented** | What didn't work | Only successes reported | ☐ |
| RR-003 | **Raw data available** | Underlying measurements | Only aggregates provided | ☑ |
| RR-004 | **Analysis scripts provided** | Code to reproduce figures | Manual calculations | ☑ |
| RR-005 | **Model Limitations (Model Card)** | Limitations section / Model Card [29] | Overclaiming / Hidden bias | ☑ |

### 6.2 Numerical Reporting

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| RR-006 | **Significant figures appropriate** | Precision matches measurement | False precision | ☑ |
| RR-007 | **Units specified** | ms, $/1K tokens, etc. | Ambiguous units | ☑ |
| RR-008 | **Ranges for all metrics** | Min, max, mean, std | Single summary statistic | ☑ |
| RR-009 | **Sample sizes stated** | n for each comparison | Missing sample sizes | ☑ |
| RR-010 | **Rounding rules consistent** | Same decimal places throughout | Inconsistent precision | ☑ |

---

## 7. Falsification Protocol

### 7.1 Adversarial Replication

Implementing Popper's **severe testing** [1, 2] and Behavioral Testing [27]:

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| FP-001 | **Behavioral/Adversarial testing** | CheckList methodology [27] used | Cherry-picked easy cases | ☑ |
| FP-002 | **Out-of-distribution testing** | Performance on novel domains | Overfitting to test distribution | ☐ |
| FP-003 | **Stress testing** | Maximum input sizes, complexity | Failure under stress | ☐ |
| FP-004 | **Negative result acceptance** | What would disprove the thesis | Unfalsifiable claims | ☑ |
| FP-005 | **Independent replication encouraged** | All materials for replication | Barriers to replication | ☑ |

### 7.2 Bias Detection

| Check ID | Criterion | Verification Method | Falsification Condition | Status |
|----------|-----------|---------------------|------------------------|--------|
| FP-006 | **Confirmation bias check** | Pre-registration comparison | Hypothesis drift | ☑ |
| FP-007 | **p-hacking detection** | p-curve analysis [18] | Suspicious p-value distribution | ☐ |
| FP-008 | **HARKing detection** | Hypothesizing After Results Known | Post-hoc hypotheses as a priori | ☐ |
| FP-009 | **Benford's law analysis** | First-digit distribution [20] | Fabricated numbers | ☐ |
| FP-010 | **GRIM test** (if applicable) | Granularity-Related Inconsistency [19] | Impossible means | N/A |

---

## 8. Replication Instructions

### 8.1 Step-by-Step Replication Protocol

For external university teams to execute:

```bash
# Step 1: Clone repository
git clone https://github.com/paiml/single-shot-eval.git
cd single-shot-eval

# Step 2: Verify environment
rustc --version  # Should match MSRV
cargo --version

# Step 3: Build from source
cargo build --release

# Step 4: Run test suite
cargo test

# Step 5: Verify coverage
make coverage

# Step 6: Execute demo
cargo run --example demo --release

# Step 7: Record outputs
# Compare to documented values in README.md
```

### 8.2 Replication Checklist

| Check ID | Replication Step | Expected Outcome | Actual Outcome | Discrepancy |
|----------|------------------|------------------|----------------|-------------|
| RI-001 | Repository clones successfully | Exit 0 | ☑ | |
| RI-002 | Dependencies resolve | Cargo.lock matches | ☑ | |
| RI-003 | Build completes without error | Exit 0, no warnings | ☑ | |
| RI-004 | All tests pass | ≥237 tests passed, 6 ignored | ☑ | |
| RI-005 | Coverage meets threshold | ≥94% | ☐ | Run `make coverage` first |
| RI-006 | Demo produces expected output | Pareto analysis matches | ☑ | |
| RI-007 | Statistical results reproducible | CI bounds match | ☑ | |
| RI-008 | Cost calculations verify | Within 5% of claimed | ☑ | |
| RI-009 | Latency measurements verify | Within 10% of claimed | ☑ | |
| RI-010 | Overall conclusion supported | Thesis falsified or confirmed | ☑ | Thesis Confirmed |

---

## References

### Philosophy of Science

[1] Popper, K. R. (1959). **The Logic of Scientific Discovery**. Routledge. ISBN: 978-0415278447

[2] Popper, K. R. (1963). **Conjectures and Refutations: The Growth of Scientific Knowledge**. Routledge. ISBN: 978-0415285940

[3] Lakatos, I. (1978). **The Methodology of Scientific Research Programmes**. Cambridge University Press. https://doi.org/10.1017/CBO9780511621123

### Toyota Production System

[4] Liker, J. K. (2004). **The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer**. McGraw-Hill. ISBN: 978-0071392310

[5] Ohno, T. (1988). **Toyota Production System: Beyond Large-Scale Production**. Productivity Press. ISBN: 978-0915299140

[6] Deming, W. E. (1986). **Out of the Crisis**. MIT Press. ISBN: 978-0262541152

### Statistical Methods

[7] Wasserstein, R. L., & Lazar, N. A. (2016). **The ASA Statement on p-Values: Context, Process, and Purpose**. *The American Statistician*, 70(2), 129-133. https://doi.org/10.1080/00031305.2016.1154108

[8] Wasserstein, R. L., Schirm, A. L., & Lazar, N. A. (2019). **Moving to a World Beyond "p < 0.05"**. *The American Statistician*, 73(sup1), 1-19. https://doi.org/10.1080/00031305.2019.1583913

[9] Efron, B., & Tibshirani, R. J. (1993). **An Introduction to the Bootstrap**. Chapman & Hall/CRC. ISBN: 978-0412042317

[10] Cohen, J. (1988). **Statistical Power Analysis for the Behavioral Sciences** (2nd ed.). Lawrence Erlbaum Associates. ISBN: 978-0805802832

### Reproducibility Crisis

[11] Open Science Collaboration. (2015). **Estimating the reproducibility of psychological science**. *Science*, 349(6251), aac4716. https://doi.org/10.1126/science.aac4716

[12] Baker, M. (2016). **1,500 scientists lift the lid on reproducibility**. *Nature*, 533(7604), 452-454. https://doi.org/10.1038/533452a

[13] Ioannidis, J. P. A. (2005). **Why Most Published Research Findings Are False**. *PLOS Medicine*, 2(8), e124. https://doi.org/10.1371/journal.pmed.0020124

### ML Evaluation Standards

[14] Kapoor, S., & Narayanan, A. (2023). **Leakage and the Reproducibility Crisis in ML-based Science**. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

[15] Lipton, Z. C., & Steinhardt, J. (2019). **Troubling Trends in Machine Learning Scholarship**. *Queue*, 17(1), 45-77. https://doi.org/10.1145/3317287.3328534

[16] Gundersen, O. E., & Kjensmo, S. (2018). **State of the Art: Reproducibility in Artificial Intelligence**. *AAAI*. https://ojs.aaai.org/index.php/AAAI/article/view/11503

[17] Pineau, J., et al. (2021). **Improving Reproducibility in Machine Learning Research**. *JMLR*, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html

### Fraud Detection

[18] Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014). **P-curve: A key to the file-drawer**. *Journal of Experimental Psychology: General*, 143(2), 534-547. https://doi.org/10.1037/a0033242

[19] Brown, N. J. L., & Heathers, J. A. J. (2017). **The GRIM Test: A Simple Technique Detects Numerous Anomalies in the Reporting of Results in Psychology**. *Social Psychological and Personality Science*, 8(4), 363-369. https://doi.org/10.1177/1948550616673876

[20] Diekmann, A. (2007). **Not the First Digit! Using Benford's Law to Detect Fraudulent Scientific Data**. *Journal of Applied Statistics*, 34(3), 321-329. https://doi.org/10.1080/02664760601004940

### Software Engineering for Science

[21] Wilson, G., et al. (2017). **Good enough practices in scientific computing**. *PLOS Computational Biology*, 13(6), e1005510. https://doi.org/10.1371/journal.pcbi.1005510

[22] Peng, R. D. (2011). **Reproducible Research in Computational Science**. *Science*, 334(6060), 1226-1227. https://doi.org/10.1126/science.1213847

[23] Sandve, G. K., et al. (2013). **Ten Simple Rules for Reproducible Computational Research**. *PLOS Computational Biology*, 9(10), e1003285. https://doi.org/10.1371/journal.pcbi.1003285

### ACM Artifact Evaluation

[24] ACM. (2023). **Artifact Review and Badging Guidelines**. ACM Digital Library. https://www.acm.org/publications/policies/artifact-review-and-badging-current

[25] Hermann, B., et al. (2020). **The Hitchhiker's Guide to Artifact Evaluation**. *ECOOP*. https://doi.org/10.4230/LIPIcs.ECOOP.2020.25

### NLP & LLM Evaluation Specifics

[26] Liang, P., et al. (2022). **Holistic Evaluation of Language Models (HELM)**. *arXiv preprint arXiv:2211.09110*.

[27] Ribeiro, M. T., et al. (2020). **Beyond Accuracy: Behavioral Testing of NLP Models with CheckList**. *ACL*. https://doi.org/10.18653/v1/2020.acl-main.442

[28] Gebru, T., et al. (2021). **Datasheets for Datasets**. *Communications of the ACM*, 64(12), 86-92. https://doi.org/10.1145/3458723

[29] Mitchell, M., et al. (2019). **Model Cards for Model Reporting**. *FAT* *. https://doi.org/10.1145/3287560.3287596

[30] Strubell, E., Ganesh, A., & McCallum, A. (2019). **Energy and Policy Considerations for Deep Learning in NLP**. *ACL*. https://doi.org/10.18653/v1/P19-1355

[31] Dodge, J., et al. (2019). **Show Your Work: Improved Reporting of Experimental Results**. *EMNLP*. https://doi.org/10.18653/v1/D19-1224

[32] Sculley, D., et al. (2015). **Hidden Technical Debt in Machine Learning Systems**. *NeurIPS*.

[33] Bowman, S. R., & Dahl, G. E. (2021). **What Will it Take to Fix Benchmarking in Natural Language Understanding?**. *NAACL*. https://doi.org/10.18653/v1/2021.naacl-main.385

[34] Wei, J., et al. (2022). **Emergent Abilities of Large Language Models**. *TMLR*.

[35] Agarwal, A., et al. (2018). **A Reductions Approach to Fair Classification**. *ICML*.

---

## Appendix A: Popperian Checklist Summary

| Popperian Criterion | Checklist Sections | Evidence of Compliance |
|---------------------|-------------------|----------------------|
| Falsifiability | FP-001 to FP-005 | Conditions for rejection stated |
| Severe Testing | ME-001 to ME-010 | Adversarial evaluation |
| Reproducibility | CR-001 to CR-015 | Independent replication possible |
| Transparency | RR-001 to RR-010 | All data and methods available |
| Self-Correction | SM-011 to SM-015 | Negative results reported |

---

## Appendix B: Toyota Way Checklist Mapping

| TPS Principle | Checklist Sections | Research Application |
|---------------|-------------------|---------------------|
| **Genchi Genbutsu** | DP-001 to DP-015 | Examine raw data |
| **Jidoka** | FP-006 to FP-010 | Stop on anomalies |
| **Kaizen** | RI-001 to RI-010 | Iterative improvement |
| **Standardization** | CR-001 to CR-015 | Reproducible environment |
| **Poka-yoke** | SM-001 to SM-015 | Error-proof statistics |

---

## Appendix C: University Replication Team Sign-Off

### Pre-Replication Certification

| Role | Name | Institution | Date | Signature |
|------|------|-------------|------|-----------|
| Principal Investigator | | | | |
| Statistical Consultant | | | | |
| Software Engineer | | | | |
| Domain Expert | | | | |

### Post-Replication Certification

| Outcome | Status | Evidence | Comments |
|---------|--------|----------|----------|
| All 100 checks completed | ☐ | | |
| Primary hypothesis tested | ☑ | | |
| Results replicated | ☑ / ☐ | | |
| Discrepancies documented | ☐ | | |
| Final determination | CONFIRMED / FALSIFIED / INCONCLUSIVE | | |

---

## Document Status

**DRAFT FOR REVIEW**: This document is awaiting team review before implementation. Please provide feedback on:

1. **Coverage**: Are there missing reproducibility criteria?
2. **Feasibility**: Can all checks be practically executed?
3. **Rigor**: Are the falsification conditions appropriate?
4. **Citations**: Are additional peer-reviewed sources needed?
5. **Clarity**: Is the protocol clear for external teams?

---

*"The criterion of the scientific status of a theory is its falsifiability, or refutability, or testability."* — Karl Popper [1]

*"True quality comes from recognizing the need to fix problems at their root cause."* — Inspired by Taiichi Ohno [5]
