# Hill Mixture MMM Validation Summary Report

> Historical note: This report predates the March 2026 synthetic benchmark refresh. It reflects earlier synthetic DGP parameters and RMSE-centered validation summaries, and should be treated as implementation history rather than the current benchmark specification.

**Date**: 2026-02-06  
**Status**: Comprehensive validation completed  
**Key Finding**: Mixture K=2 converges on real data; K=3 does not

---

## Executive Summary

This report documents all validation experiments conducted for the Bayesian Hill Mixture Marketing Mix Model. After extensive experimentation with different configurations, we found that:

1. **Single Hill model** consistently converges on both synthetic and real data
2. **Mixture K=3** does NOT converge on real data despite various interventions
3. **Mixture K=2** successfully converges on real data with proper diagnostics
4. Predictive performance (LOO) favors mixture models despite convergence issues

---

## 1. Datasets

### 1.1 Synthetic Data

Generated using `hill_mmm/data.py` with the following Data Generating Processes (DGPs):

| DGP | K_true | Description | Parameters |
|-----|--------|-------------|------------|
| `single` | 1 | Single Hill curve (null hypothesis) | A=30, k=s_median, n=1.5 |
| `mixture_k2` | 2 | 2-component mixture | π=[0.6,0.4], A=[20,40], n=[2.0,1.2] |
| `mixture_k3` | 3 | 3-component mixture (standard test) | π=[0.4,0.3,0.3], A=[15,30,60], n=[2.0,1.5,1.0] |
| `mixture_k5` | 5 | 5-component mixture (sparse discovery test) | π=[0.3,0.25,0.2,0.15,0.1] |

**Common settings**:
- T = 200 time points (default)
- σ = 3.0 (observation noise)
- α = 0.5 (adstock decay)
- Spend: log-normal distribution (mean=1.5, sigma=0.6)

**Reproducibility**: `seed=42` for all experiments unless otherwise specified.

### 1.2 Real Data

| Field | Value |
|-------|-------|
| Organisation ID | `ab0b8d655e7140272cab371a515e009a` |
| Vertical | Food & Drink |
| Territory | All Territories |
| Currency | USD |
| Date Range | 2020-08-29 to 2023-11-21 |
| Observations (T) | 1180 |
| Target Variable | `all_purchases` |
| Total Target | 72,187 |
| Total Spend | $1,424,599 |

**Marketing Channels (8)**:
1. `google_paid_search_spend`
2. `google_shopping_spend`
3. `google_pmax_spend`
4. `google_display_spend`
5. `google_video_spend`
6. `meta_facebook_spend`
7. `meta_instagram_spend`
8. `meta_other_spend`

---

## 2. Experiments and Methods

### 2.1 Experiment Overview

| Experiment | Branch/Worktree | Method | Key Question |
|------------|-----------------|--------|---------------|
| Baseline validation | `main` | MCMC (standard) | Does mixture improve over single Hill? |
| Longer chains | `experiment-longer-chains` | MCMC (extended) | Does more sampling help convergence? |
| Tempering | `experiment-tempering` | MCMC + K=2 | Does simpler mixture converge? |
| VI validation | `experiment-vi` | SVI (AutoNormal) | Is variational inference viable? |
| Label switching | `main` | Diagnostic methods | How to diagnose mixture convergence? |
| Hyperprior design | `main` | Prior comparison | Which hyperpriors avoid funnels? |

### 2.2 MCMC Configurations

| Configuration | Warmup | Samples | Chains | Use Case |
|---------------|--------|---------|--------|----------|
| Quick test | 200-500 | 400-500 | 2 | Rapid iteration |
| Standard | 1000 | 2000 | 4 | Primary validation |
| Extended | 2000 | 4000 | 6 | Convergence rescue attempt |

### 2.3 Convergence Diagnostics

We implemented multiple diagnostic approaches for mixture models:

| Diagnostic | Implementation | Purpose |
|------------|----------------|----------|
| Standard R-hat | ArviZ `az.rhat()` | Quick screening (often unreliable for mixtures) |
| Rank-normalized R-hat | `method="rank"` | More robust R-hat variant |
| Log-likelihood R-hat | `compute_label_invariant_diagnostics()` | Label-invariant convergence |
| Relabeled R-hat | `compute_diagnostics_on_relabeled()` | Post-hoc sorted components |
| Label switching rate | `check_label_switching()` | Detect permutation jumping |

**Key insight**: Standard R-hat is fundamentally inappropriate for mixture models due to label switching. We developed label-invariant diagnostics (see `LABEL_SWITCHING_DIAGNOSTICS.md`).

---

## 3. Results

### 3.1 Real Data Validation Results

#### Main Worktree (Standard MCMC)

| Model | ELPD-LOO | SE | p_loo | Max R-hat | Converged | Time (sec) |
|-------|----------|-----|-------|-----------|-----------|------------|
| Single Hill | -5126.1 | 63.7 | 15.7 | 1.00 | **Yes** | 130 |
| Mixture K=3 | -4953.5 | 43.9 | 73.1 | 2.97 | No | 908 |

**Delta LOO**: +172.6 (Mixture better, but unreliable due to non-convergence)

#### Longer Chains Worktree (Extended MCMC)

| Model | ELPD-LOO | SE | p_loo | Max R-hat | Converged | Time (sec) |
|-------|----------|-----|-------|-----------|-----------|------------|
| Single Hill | -5126.1 | 63.7 | 15.6 | 1.00 | **Yes** | 401 |
| Mixture K=3 | -4963.7 | 44.6 | 93.8 | 3.44→2.09* | No | 2674 |

*After relabeling. Standard R-hat was 3.44.

**Finding**: Extended MCMC did NOT resolve K=3 convergence.

#### Tempering Worktree (K=2 Mixture)

| Model | ELPD-LOO | SE | p_loo | Max R-hat (relabeled) | Converged | Time (sec) |
|-------|----------|-----|-------|-----------------------|-----------|------------|
| Single Hill | -5126.1 | 63.7 | 15.7 | 1.00 | **Yes** | 140 |
| **Mixture K=2** | -4980.6 | 42.2 | 28.6 | **1.0006** | **Yes** | 811 |

**Key Finding**: Mixture K=2 CONVERGES on real data!

**Convergence diagnostics for K=2**:
- Label-invariant R-hat: 1.0003
- Relabeled component R-hats: all ≤ 1.001
- Pareto-k issues: 1 bad, 0 very bad (acceptable)

**Estimated parameters (K=2)**:
```
Component 1: A=217.0, k=3285.5, n=0.76, π=3.0%
Component 2: A=355.4, k=25403.5, n=1.43, π=97.0%
```

### 3.2 Synthetic Data Results

#### Label Switching Diagnostics Experiment

Data: `mixture_k3`, T=200, seed=42

| Approach | Standard R-hat | Relabeled R-hat | RMSE | Coverage 90% |
|----------|----------------|-----------------|------|---------------|
| Constrained (in-MCMC) | 1.82 | 1.64 | 7.43 | 92.5% |
| Unconstrained + Relabel | 1.70 | **1.01** | 7.43 | 93.0% |

**Finding**: Post-hoc relabeling achieves excellent convergence (R-hat=1.01) on synthetic data.

#### Ordering Comparison (Full MCMC)

Data: `mixture_k3`, T=200, warmup=1000, samples=2000, chains=4

| Approach | Max R-hat | Min ESS | RMSE | Coverage 90% | Time (sec) |
|----------|-----------|---------|------|--------------|------------|
| Constrained | 1.82 | 6 | 7.45 | 92.5% | 108 |
| Unconstrained | 2.12→1.01* | 5 | 7.44 | 92.0% | 94 |

*After relabeling.

### 3.3 Variational Inference Results

Data: Real organisation, T=1180

| Model | Method | ELBO | Converged | Time (sec) |
|-------|--------|------|-----------|------------|
| Single Hill | SVI (AutoNormal) | -44217 | No | 2.4 |
| Mixture K=2 | SVI (AutoNormal) | -5101 | **Yes** | 3.3 |

**VI Parameters (K=2)**:
```
Component 1: A=62.3, k=2619.1, n=1.40, π=95.6%
Component 2: A=154.1, k=923.4, n=1.04, π=4.4%
```

**Finding**: VI converges for mixture but parameter estimates differ significantly from MCMC.

---

## 4. Hyperprior Design Experiments

### 4.1 The Funnel Problem

Hierarchical models with scale parameters create "funnel" posterior geometries that challenge MCMC:

```
    σ large → Wide posterior (needs large step size)
         |
         ↓
    σ → 0  → Funnel vertex (needs tiny step size)
```

### 4.2 Hyperprior Comparison

| Prior | Mode | P(σ < 0.1) | Convergence | Flexibility |
|-------|------|------------|-------------|-------------|
| HalfNormal(0.1) | 0.00 | 68% | Excellent | Poor |
| HalfNormal(1.0) | 0.00 | 7.6% | Poor | Good |
| InverseGamma(3,1) | 0.25 | 3% | Partial | Good |
| **LogNormal(-1,0.5)** | **0.28** | **1.4%** | **Good** | **Good** |

### 4.3 Recommended Hyperpriors

```python
sigma_log_A = numpyro.sample("sigma_log_A", dist.LogNormal(-1.0, 0.5))
sigma_log_n = numpyro.sample("sigma_log_n", dist.LogNormal(-1.5, 0.5))
```

**Rationale**: LogNormal has mode away from zero, preventing funnel vertex while allowing flexibility.

---

## 5. Key Insights

### 5.1 Model Complexity vs Data

| Model | Real Data (T=1180) | Synthetic (T=200) |
|-------|-------------------|-------------------|
| Single Hill | Converges | Converges |
| Mixture K=2 | **Converges** | Converges |
| Mixture K=3 | Does NOT converge | Converges (with relabeling) |

**Interpretation**: Real data complexity or limited variation may not support K=3 mixture. K=2 appears to be the practical limit for this dataset.

### 5.2 Label Switching Best Practices

Based on extensive experimentation:

1. **Do NOT constrain during MCMC** - Let labels switch freely
2. **Apply post-hoc relabeling** - Sort by k values after sampling
3. **Use multiple diagnostics**:
   - Primary: Relabeled component R-hat
   - Supplementary: Log-likelihood R-hat
   - Validation: RMSE, Coverage

### 5.3 Convergence Thresholds for Mixtures

| Metric | Standard Models | Mixture Models |
|--------|-----------------|----------------|
| R-hat | ≤ 1.05 | ≤ 1.01 (relabeled) |
| ESS (bulk) | ≥ 100 | ≥ 400 |
| Pareto-k bad | 0 | ≤ 5% of observations |

### 5.4 Predictive Performance Summary

| Model | ELPD-LOO | Interpretation |
|-------|----------|----------------|
| Single Hill | -5126 | Baseline |
| Mixture K=2 | -4981 | +145 improvement (reliable) |
| Mixture K=3 | -4954 | +172 improvement (unreliable - not converged) |

**Conclusion**: K=2 mixture provides reliable 145-point improvement over single Hill.

---

## 6. Reproduction Instructions

### 6.1 Environment Setup

```bash
# Clone and install
git clone https://github.com/your-repo/Bayesian-Marketing-Mix-Modeling
cd Bayesian-Marketing-Mix-Modeling
uv sync  # or pip install -e .
```

### 6.2 Run Synthetic Benchmark

```bash
python scripts/run_benchmarks.py --quick
# Or full benchmark:
python scripts/run_benchmarks.py \
    --dgp single mixture_k2 mixture_k3 \
    --seeds 0 1 2 3 4
```

### 6.3 Run Real Data Validation

```bash
python scripts/run_real_data_validation.py \
    --org-id ab0b8d655e7140272cab371a515e009a \
    --warmup 1000 \
    --samples 2000 \
    --chains 4
```

### 6.4 Run Label Switching Diagnostics

```bash
python scripts/legacy/test_ordering_comparison.py
# Results in: results/label_switching_diagnostics/
```

### 6.5 Access Experiment Worktrees

```bash
# List all worktrees
git worktree list

# Access tempering experiment (K=2 converged)
cd /Users/shohei/Documents/repo/mmm-experiment-tempering

# Access longer chains experiment
cd /Users/shohei/Documents/repo/mmm-experiment-longer-chains
```

---

## 7. File Reference

### Results Files

| File | Description |
|------|-------------|
| `results/real_data_validation/*.json` | Real data MCMC results |
| `results/vi_validation/*.json` | Variational inference results |
| `results/label_switching_diagnostics/results.json` | Diagnostic comparison |
| `results/ordering_comparison/results.json` | Constrained vs unconstrained |
| `benchmark_quick_results.csv` | Synthetic data quick benchmark |

### Documentation Files

| File | Description |
|------|-------------|
| `../research-summary.md` | Literature review and project motivation |
| `LABEL_SWITCHING_DIAGNOSTICS.md` | Label switching investigation |
| `HALFNORMAL_CONVERGENCE_EXPERIMENTS.md` | HalfNormal hyperprior experiments |
| `HYPERPRIOR_DESIGN.md` | Hyperprior design rationale |
| `CONVERGENCE_EXPERIMENTS_HANDOVER.md` | Convergence experiment handover |

### Source Code

| File | Description |
|------|-------------|
| `hill_mmm/models.py` | NumPyro model definitions |
| `hill_mmm/data.py` | Synthetic data generation |
| `hill_mmm/inference.py` | MCMC and diagnostics |
| `hill_mmm/metrics.py` | Evaluation metrics |
| `hill_mmm/benchmark.py` | Benchmark runner |

---

## 8. Conclusions and Recommendations

### 8.1 Summary of Findings

1. **Single Hill**: Reliable baseline, always converges
2. **Mixture K=2**: Recommended for real data - converges and improves predictive performance
3. **Mixture K=3**: Does not converge on real data despite extensive efforts
4. **Post-hoc relabeling**: Essential for mixture model diagnostics
5. **LogNormal hyperpriors**: Recommended for hierarchical scale parameters

### 8.2 Recommended Workflow

```
1. Start with Single Hill as baseline
2. Fit Mixture K=2
3. Use post-hoc relabeling for diagnostics
4. Check relabeled R-hat ≤ 1.01
5. Compare ELPD-LOO with Single Hill
6. Only consider K=3 if K=2 shows clear multimodality
```

### 8.3 Open Questions

1. Can alternative inference methods (e.g., SMC, tempering) enable K=3 convergence?
2. Would model reparameterization help?
3. Is the real data genuinely bimodal or is K=2 capturing noise?

### 8.4 Next Steps

1. Validate K=2 mixture on additional real datasets
2. Investigate parameter interpretation (what do the two components represent?)
3. Consider model selection criteria beyond LOO
4. Explore causal identification under mixture assumptions

---

## Document History

| Date | Author | Change |
|------|--------|--------|
| 2026-02-06 | Session 60a301b3 | Initial comprehensive report |
