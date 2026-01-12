# Benchmark Evaluation for Hill Mixture MMM

> Historical note: This report is kept for experiment history. It predates the March 2026 synthetic benchmark refresh, uses RMSE-era summaries, and includes `mixture_k5` / `sparse_k5` configurations that are no longer part of the default headline benchmark. Do not use it for current paper claims.

## Abstract
This report summarizes one full benchmark execution of Hill Mixture MMM, run on **February 7, 2026** (timestamp: `20260207_061725`). The goal is to evaluate predictive performance and inferential reliability under the same protocol for synthetic and real MMM data.

The main result is a clear trade-off. Mixture models frequently produce strong test RMSE, including many runs that fail strict convergence checks. However, those same non-converged runs also generate severe tail-risk failures (especially on real data), which can dominate simple means and make parameter-level interpretation unsafe. For this reason, predictive claims should rely on robust summaries (median and trimmed mean), while interpretability claims should require stronger convergence evidence.

## 1. How To Read This Report
This document is written to be self-contained. A new reader should follow this order:
1. Read Sections 2-4 to understand exactly what was benchmarked and how metrics are computed.
2. Read Section 5 for headline results and practical interpretation.
3. Use Appendix A for all numeric detail and table-level validation.

Terms used repeatedly:
1. `seed-cell` (synthetic): one fixed `(DGP, seed)` compared across models.
2. `org-seed cell` (real): one fixed `(organization, seed)` compared across models.
3. `converged`: `max_rhat < 1.05` as implemented in `hill_mmm/inference.py`.

## 2. Benchmark Scope And Setup

### 2.1 Data Scope
Synthetic experiments:
1. DGPs: `single`, `mixture_k2`, `mixture_k3`, `mixture_k5`.
2. Each synthetic run uses `T=200` observations.
3. Total synthetic grid: 4 DGPs × 4 models × 5 seeds = **80 runs**.

Real-data experiments:
1. Source file: `data/conjura_mmm_data.csv`.
2. Preprocessing: aggregate spend across channels, target is `all_purchases`.
3. Organizations: top 5 with most observations (minimum length criterion from loader).
4. Total real grid: 5 organizations × 3 models × 3 seeds = **45 runs**.
5. All 45 real runs completed with `status=success`.

### 2.2 Models Compared
| Model name | Code path | Description |
| --- | --- | --- |
| `single_hill` | `hill_mmm/models.py:model_single_hill` | Baseline single Hill response curve |
| `mixture_k2` | `hill_mmm/models.py:model_hill_mixture_k2` | 2-component mixture model |
| `mixture_k3` | `hill_mmm/models.py:model_hill_mixture_hierarchical_reparam` (`K=3`) | Hierarchical reparameterized 3-component mixture |
| `sparse_k5` (synthetic only) | `hill_mmm/models.py:model_hill_mixture_unconstrained` (`K=5`) | Larger unconstrained mixture for sparse/effective-K behavior |

### 2.3 Inference And Split Configuration
1. Sampler: NUTS (`target_accept_prob=0.9`).
2. MCMC settings: `num_warmup=1000`, `num_samples=2000`, `num_chains=4`.
3. Train/test split: first 75% train, last 25% test.
4. Config file: `results/benchmark/config.json`.

### 2.4 Per-Run Evaluation Flow
For each run:
1. Fit model on train split.
2. Compute diagnostics (`max_rhat`, `min_ess_bulk`, convergence flag).
3. Compute model comparison metrics (`elpd_loo`, `delta_loo` vs single-Hill).
4. Compute posterior predictive metrics on both train and test.

## 3. Metric Definitions (Exact Meaning)

### 3.1 Predictive Metrics
1. `test_rmse`: RMSE between true `y` and posterior predictive mean.
2. `test_coverage_90`: fraction of true `y` values within posterior predictive `[5%, 95%]` interval.

Formally:
```text
y_hat(t) = mean_s y_sample(s, t)
RMSE = sqrt(mean_t (y_hat(t) - y(t))^2)
Coverage_90 = mean_t I[q05(t) <= y(t) <= q95(t)]
```

### 3.2 Model Comparison Metrics
1. `elpd_loo`: PSIS-LOO expected log predictive density (higher is better).
2. `delta_loo`: `elpd_loo(model) - elpd_loo(single_hill)` within the same comparison cell.
3. `delta_loo_significant`: `abs(delta_loo) > 2 * se_combined`, where `se_combined = sqrt(se_model^2 + se_baseline^2)`.

### 3.3 Convergence Diagnostics
1. `max_rhat`: worst R-hat across monitored parameters.
2. `min_ess_bulk`: minimum bulk ESS across parameters.
3. `converged`: `max_rhat < 1.05`.

Interpretation rule used in this report:
1. Predictive usefulness can exist even when `converged=False`.
2. Parameter-level trust and scientific interpretation require stronger diagnostic quality.

### 3.4 Mixture Complexity
1. `effective_k_mean`: average number of components with posterior weight above threshold (`pis > 0.05`).
2. For `single_hill`, `effective_k_mean` is defined as 1.0 by construction.

## 4. Headline Results

### 4.1 Global Outcome
| Domain | Runs | Converged | Non-converged RMSE winners | Main risk signal |
| --- | --- | --- | --- | --- |
| Synthetic | 80 | 53 (66.2%) | 7 / 20 seed-cells | 16 catastrophic LOO outliers (`elpd_loo < -2000`) |
| Real | 45 | 8 (17.8%) | 11 / 15 org-seed cells | Extreme heavy tail (`max RMSE = 110293.055`) |

Two direct implications:
1. Best-RMSE runs are often not the converged runs.
2. Non-converged runs create heavy-tail failures that can dominate mean-based summaries.

### 4.2 Synthetic Results In Plain Language
1. Mixture models are often competitive, and sometimes best, in test RMSE.
2. Extreme negative LOO outliers are concentrated in `mixture_k2` and `sparse_k5` (Table A2).
3. Converged-only summaries materially change rankings (Table A4 vs Table A3), confirming that stability filtering matters.
4. Synthetic evidence supports a nuanced position: non-convergence is not automatic predictive failure, but it is still a reliability warning.

### 4.3 Real-Data Results In Plain Language
Convergence rates are low:
1. `single_hill`: 33.3%.
2. `mixture_k2`: 13.3%.
3. `mixture_k3`: 6.7%.

Cell-level winners (`org-seed` level):
1. `single_hill`: 9 winner cells.
2. `mixture_k2`: 3 winner cells.
3. `mixture_k3`: 3 winner cells.
4. Among all 15 winner cells, 11 are non-converged.

Robust aggregate view:
1. Median test RMSE: `mixture_k2=100.912`, `mixture_k3=101.000`, `single_hill=126.527`.
2. Relative to `single_hill`, median RMSE improves by about **20.2%** for both mixture models.
3. 10% trimmed mean test RMSE: `single_hill=127.767`, `mixture_k3=143.558`, `mixture_k2=167.161`.
4. Relative to `single_hill`, trimmed-mean RMSE is worse by **12.4%** (`mixture_k3`) and **30.8%** (`mixture_k2`).

Conclusion: median and trimmed-mean disagree because real-data errors are heavy-tailed for mixture models.

### 4.4 Tail-Risk Snapshot (Real Test RMSE)
| Model | 50th pct | 90th pct | 95th pct | 99th pct | Max |
| --- | --- | --- | --- | --- | --- |
| `single_hill` | 126.527 | 289.038 | 335.642 | 422.404 | 444.095 |
| `mixture_k2` | 100.912 | 437.099 | 526.711 | 671.512 | 707.712 |
| `mixture_k3` | 101.000 | 429.013 | 33391.398 | 94912.724 | 110293.055 |

Interpretation:
1. Mixture medians are strong.
2. Upper-tail failure severity is much larger for mixture models, especially `mixture_k3`.
3. Mean-based ranking is therefore highly unstable.

## 5. Recommended Narrative For Paper Writing
Use this framing in manuscript text:
1. Mixture models show predictive potential in both synthetic and real settings.
2. Predictive utility and inferential reliability should be reported separately.
3. Report robust predictive statistics (median, trimmed mean), not only means.
4. Restrict parameter-interpretation claims to runs with convincing convergence diagnostics.
5. Explicitly disclose tail-risk behavior in real data.

## 6. Practical Decision Guidance
If the project goal is operational forecasting:
1. Mixture models may be valuable.
2. Select models with robust criteria and explicit outlier controls.

If the project goal is causal/structural interpretation:
1. Prioritize convergence and ESS quality first.
2. Treat non-converged posterior parameters as unreliable, even when RMSE is good.

## 7. Limitations And Reproducibility

### 7.1 Limitations
1. Single benchmark timestamp (`20260207_061725`).
2. Real-data breadth is limited (5 organizations, 3 seeds each).
3. No hierarchical pooling across organizations in this benchmark report.
4. `pareto_k_bad` in real-run CSV is not used for conclusions in this report.

### 7.2 Reproducibility
1. Run benchmark: `python scripts/run_benchmark.py`
2. Output directory: `results/benchmark/`
3. Config snapshot: `results/benchmark/config.json`
4. Synthetic raw file used here: `results/benchmark/synthetic_20260207_061725.csv`
5. Real-data raw file used here: `results/benchmark/real_20260207_061725.csv`

---

## Appendix A. Detailed Quantitative Tables

### Table A1. Synthetic convergence rate by model

| model | convergence_rate |
| --- | --- |
| mixture_k2 | 30.0% |
| mixture_k3 | 80.0% |
| single_hill | 100.0% |
| sparse_k5 | 55.0% |

### Table A2. Synthetic catastrophic run counts by DGP and model (`elpd_loo < -2000`)

| dgp | model | catastrophic_count |
| --- | --- | --- |
| mixture_k2 | mixture_k2 | 2 |
| mixture_k2 | mixture_k3 | 0 |
| mixture_k2 | single_hill | 0 |
| mixture_k2 | sparse_k5 | 2 |
| mixture_k3 | mixture_k2 | 2 |
| mixture_k3 | mixture_k3 | 0 |
| mixture_k3 | single_hill | 0 |
| mixture_k3 | sparse_k5 | 2 |
| mixture_k5 | mixture_k2 | 2 |
| mixture_k5 | mixture_k3 | 0 |
| mixture_k5 | single_hill | 0 |
| mixture_k5 | sparse_k5 | 2 |
| single | mixture_k2 | 2 |
| single | mixture_k3 | 0 |
| single | single_hill | 0 |
| single | sparse_k5 | 2 |

### Table A3. Synthetic means across all runs (including non-converged runs)

| dgp | model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | effective_k_mean | delta_loo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | mixture_k2 | -13736.050 | 4.795 | 4.139 | 84.8% | 1.993 | -13326.344 |
| mixture_k2 | mixture_k3 | -406.354 | 5.157 | 3.586 | 73.6% | 2.513 | 3.353 |
| mixture_k2 | single_hill | -409.707 | 5.382 | 3.574 | 73.2% | 1.000 | 0.000 |
| mixture_k2 | sparse_k5 | -5681.563 | 5.094 | 3.751 | 75.6% | 3.013 | -5271.857 |
| mixture_k3 | mixture_k2 | -34256.100 | 8.355 | 7.900 | 88.4% | 2.000 | -33735.375 |
| mixture_k3 | mixture_k3 | -488.063 | 8.291 | 7.570 | 85.6% | 2.925 | 32.661 |
| mixture_k3 | single_hill | -520.724 | 8.307 | 7.528 | 82.4% | 1.000 | 0.000 |
| mixture_k3 | sparse_k5 | -29021.516 | 8.335 | 7.640 | 87.2% | 3.815 | -28500.792 |
| mixture_k5 | mixture_k2 | -28668.929 | 7.755 | 7.290 | 87.2% | 1.999 | -28159.443 |
| mixture_k5 | mixture_k3 | -498.894 | 7.689 | 6.955 | 86.0% | 2.829 | 10.592 |
| mixture_k5 | single_hill | -509.486 | 7.792 | 6.925 | 85.6% | 1.000 | 0.000 |
| mixture_k5 | sparse_k5 | -25049.015 | 7.667 | 7.096 | 86.8% | 3.686 | -24539.528 |
| single | mixture_k2 | -13973.083 | 4.768 | 3.413 | 76.8% | 1.958 | -13590.730 |
| single | mixture_k3 | -383.380 | 4.820 | 2.990 | 69.6% | 1.952 | -1.027 |
| single | single_hill | -382.353 | 5.139 | 2.982 | 66.0% | 1.000 | 0.000 |
| single | sparse_k5 | -4295.580 | 4.728 | 3.196 | 71.6% | 2.249 | -3913.227 |

### Table A4. Synthetic means across converged runs only

| dgp | model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | effective_k_mean | delta_loo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | mixture_k2 | -420.605 | 3.731 | 3.837 | 92.0% | 1.991 | -0.264 |
| mixture_k2 | mixture_k3 | -406.354 | 5.157 | 3.586 | 73.6% | 2.513 | 3.353 |
| mixture_k2 | single_hill | -409.707 | 5.382 | 3.574 | 73.2% | 1.000 | 0.000 |
| mixture_k2 | sparse_k5 | -409.983 | 4.717 | 3.648 | 78.7% | 3.032 | 2.344 |
| mixture_k3 | mixture_k3 | -494.754 | 8.231 | 7.824 | 90.0% | 2.933 | 30.989 |
| mixture_k3 | single_hill | -520.724 | 8.307 | 7.528 | 82.4% | 1.000 | 0.000 |
| mixture_k3 | sparse_k5 | -482.877 | 8.387 | 7.305 | 85.0% | 3.867 | 32.708 |
| mixture_k5 | mixture_k2 | -499.433 | 6.924 | 7.101 | 88.0% | 1.999 | 13.833 |
| mixture_k5 | mixture_k3 | -500.721 | 7.684 | 6.883 | 88.0% | 2.831 | 6.789 |
| mixture_k5 | single_hill | -509.486 | 7.792 | 6.925 | 85.6% | 1.000 | 0.000 |
| mixture_k5 | sparse_k5 | -501.016 | 7.381 | 6.887 | 90.0% | 3.783 | 6.548 |
| single | mixture_k2 | -380.657 | 5.022 | 2.934 | 66.0% | 1.945 | -0.911 |
| single | mixture_k3 | -383.380 | 4.820 | 2.990 | 69.6% | 1.952 | -1.027 |
| single | single_hill | -382.353 | 5.139 | 2.982 | 66.0% | 1.000 | 0.000 |
| single | sparse_k5 | -384.492 | 4.841 | 3.022 | 70.7% | 2.176 | -0.593 |

### Table A5. Real convergence rate by model

| model | convergence_rate |
| --- | --- |
| mixture_k2 | 13.3% |
| mixture_k3 | 6.7% |
| single_hill | 33.3% |

### Table A6. Real mean metrics by model (all runs)

| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -1.052e+07 | 192.886 | 160.654 | 88.9% | 1.883 |
| mixture_k3 | -2.585e+06 | 7478.114 | 8293.108 | 87.0% | 2.188 |
| single_hill | -4.719e+07 | 141.050 | 118.684 | 89.4% | 1.132 |

### Table A7. Real median metrics by model (all runs)

| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -7328.036 | 100.912 | 95.238 | 91.8% | 1.873 |
| mixture_k3 | -7262.585 | 101.000 | 95.296 | 91.7% | 1.911 |
| single_hill | -1.559e+06 | 126.527 | 122.308 | 90.8% | 1.122 |

### Table A8. Real 10% trimmed mean of test RMSE

| model | test_rmse_trim10 |
| --- | --- |
| mixture_k2 | 167.161 |
| mixture_k3 | 143.558 |
| single_hill | 127.767 |

### Table A9. Real RMSE by convergence status and model

| model | conv_n | conv_mean_rmse | conv_median_rmse | nonconv_n | nonconv_mean_rmse | nonconv_median_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | 2 | 40.606 | 40.606 | 13 | 216.314 | 140.303 |
| mixture_k3 | 1 | 12.407 | 12.407 | 14 | 8011.379 | 101.028 |
| single_hill | 5 | 170.353 | 98.386 | 10 | 126.398 | 126.792 |

### Table A10. Best model by organization (seed-mean, all runs)

| org_id | best_by_loo | best_loo | best_by_test_rmse | best_test_rmse |
| --- | --- | --- | --- | --- |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k2 | -5916.126 | single_hill | 55.020 |
| 72a86a208d24d68b80be0e44a8a4872d | mixture_k3 | -4768.377 | single_hill | 11.749 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | -1.087e+07 | mixture_k3 | 163.795 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | -2.034e+06 | single_hill | 340.703 |
| bfb6f6a326141ed6a751fc83ba836984 | mixture_k3 | -7262.358 | mixture_k3 | 101.018 |

### Table A11. Synthetic failures (top by R-hat)

| dgp | model | seed | max_rhat | min_ess_bulk | elpd_loo | test_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mixture_k3 | mixture_k2 | 4 | 2.700 | 5 | -18859.193 | 8.699 |
| mixture_k5 | mixture_k2 | 4 | 2.250 | 6 | -16425.453 | 8.267 |
| mixture_k2 | mixture_k2 | 2 | 2.060 | 5 | -58272.901 | 5.064 |
| mixture_k5 | mixture_k2 | 2 | 1.820 | 6 | -125421.513 | 8.870 |
| mixture_k3 | sparse_k5 | 0 | 1.700 | 7 | -500.134 | 7.118 |
| mixture_k3 | mixture_k3 | 3 | 1.680 | 6 | -480.193 | 8.738 |
| mixture_k2 | mixture_k2 | 1 | 1.600 | 7 | -1723.045 | 3.831 |
| mixture_k2 | sparse_k5 | 3 | 1.600 | 7 | -20239.214 | 5.422 |

### Table A12. Real-data failures (top by R-hat)

| org_id | model | seed | max_rhat | min_ess_bulk | elpd_loo | test_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| 882ce7e286d66facc66518783e2192c7 | mixture_k2 | 1 | 6.700 | 4 | -3.788e+07 | 449.139 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | 1 | 4.590 | 4 | -6.086e+06 | 110293.055 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k2 | 1 | 3.410 | 4 | -1.058e+08 | 707.712 |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k3 | 2 | 3.300 | 4 | -5977.608 | 68.835 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | 0 | 3.170 | 4 | -3.261e+07 | 175.750 |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k3 | 1 | 3.030 | 5 | -6517.861 | 54.150 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | 1 | 3.020 | 5 | -8082.885 | 149.748 |
| bfb6f6a326141ed6a751fc83ba836984 | mixture_k2 | 1 | 2.720 | 5 | -1.407e+07 | 140.303 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | 2 | 2.670 | 5 | -7989.905 | 433.545 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k2 | 2 | 2.660 | 5 | -8261.100 | 419.039 |
