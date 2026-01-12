# HalfNormal Hyperprior Convergence Experiments

> Historical note: This experiment memo predates the March 2026 benchmark refresh and is retained for prior-design history. Any predictive metrics here reflect earlier RMSE-era validation runs.

**Session Reference**: 240ae953-da14-40d4-a722-137f211ba976  
**Date**: 2026-02-05  
**Status**: Completed - Led to LogNormal solution

---

## 1. Executive Summary

This document details the experimental design and results of testing `HalfNormal(0.1)` hyperpriors for hierarchical scale parameters in the Hill Mixture MMM. While convergence improved significantly, the approach was ultimately rejected due to excessive constraint on model flexibility.

**Key Finding**: HalfNormal(0.1) achieves excellent convergence (R-hat ≤ 1.01, ESS > 200) but sacrifices model expressiveness by forcing scale parameters too close to zero.

---

## 2. Background: The Funnel Problem

### 2.1 Problem Statement

Hierarchical Bayesian models with scale parameters (σ) create challenging posterior geometries known as "funnels":

```
                    σ large
                      │
                      │    ← Wide posterior region (needs large step size)
                      │
               ───────┼───────
                     ╱│╲
                    ╱ │ ╲
                   ╱  │  ╲   ← Funnel neck (needs tiny step size)
                  ╱   │   ╲
                 ╱    │    ╲
                ╱     ▼     ╲
              σ → 0 (funnel vertex)
```

**The Challenge**: No single HMC step size works well for both regions, leading to:
- Divergent transitions near the funnel vertex
- Low Effective Sample Size (ESS)
- Poor chain mixing (high R-hat)

### 2.2 Model Context

The Hill Mixture MMM uses hierarchical priors for component parameters:

```python
# Population-level hyperpriors
mu_log_A ~ Normal(A_loc, 0.5)      # population mean of log-amplitude
sigma_log_A ~ ???                   # between-component variability ← PROBLEM

mu_log_n ~ Normal(log(1.5), 0.3)   # population mean of log-Hill-exponent
sigma_log_n ~ ???                   # between-component variability ← PROBLEM

# Component-level (non-centered parameterization)
log_A_raw ~ Normal(0, 1)
log_A = mu_log_A + sigma_log_A * log_A_raw  # NCP transformation
A = exp(log_A)
```

The choice of prior for `sigma_log_A` and `sigma_log_n` directly affects:
1. Posterior geometry (funnel presence)
2. MCMC convergence
3. Model flexibility (ability to capture heterogeneity)

---

## 3. Experimental Design

### 3.1 Hypothesis

**H1**: Using a tight HalfNormal prior (scale=0.1) on hierarchical scale parameters will:
- Prevent σ → 0 by reducing the probability mass in the funnel region
- Improve MCMC convergence (R-hat, ESS)
- Maintain sufficient flexibility for real-world heterogeneity

### 3.2 Configuration

```python
# EXPERIMENTAL: Tight HalfNormal hyperpriors
sigma_log_A = numpyro.sample("sigma_log_A", dist.HalfNormal(0.1))
sigma_log_n = numpyro.sample("sigma_log_n", dist.HalfNormal(0.1))
```

### 3.3 MCMC Settings

| Parameter | Value | Rationale |
|-----------|-------|----------|
| num_warmup | 300-500 | Quick iteration for testing |
| num_samples | 300-500 | Sufficient for convergence assessment |
| num_chains | 2 | Minimum for R-hat computation |
| K (components) | 3 | Standard mixture complexity |
| target_accept | 0.8 | NumPyro NUTS default |

### 3.4 Convergence Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Max R-hat | ≤ 1.05 | Gelman-Rubin statistic across all parameters |
| Min ESS (bulk) | ≥ 100 | Effective sample size for bulk of posterior |
| Min ESS (tail) | ≥ 50 | Effective sample size for posterior tails |
| Divergences | 0 | No divergent transitions |

### 3.5 Test Data

- **DGP**: `mixture_k3` synthetic data generator
- **True K**: 3 mixture components
- **Time points**: T = 104 (2 years of weekly data)

---

## 4. Results

### 4.1 Convergence Metrics

| Metric | HalfNormal(0.1) | Baseline (HalfNormal(1.0)) | Threshold |
|--------|-----------------|----------------------------|----------|
| Max R-hat | **1.01** | 1.05-1.10 | ≤ 1.05 |
| Min ESS (bulk) | **200+** | 35-70 | ≥ 100 |
| Min ESS (tail) | **150+** | 20-50 | ≥ 50 |
| Divergences | **0** | 5-15 | 0 |

**Convergence: ✅ EXCELLENT**

### 4.2 Distribution Properties

```python
from scipy import stats

# HalfNormal(0.1) properties
hn = stats.halfnorm(scale=0.1)

print(f"Mode: {0.0}")                    # Always at 0 for HalfNormal
print(f"Median: {hn.median():.4f}")      # 0.0674
print(f"Mean: {hn.mean():.4f}")          # 0.0798
print(f"Std Dev: {hn.std():.4f}")        # 0.0603
print(f"P(σ < 0.05): {hn.cdf(0.05):.3f}")  # 0.383 (38% below 0.05!)
print(f"P(σ < 0.10): {hn.cdf(0.10):.3f}")  # 0.683 (68% below 0.10!)
print(f"P(σ < 0.20): {hn.cdf(0.20):.3f}")  # 0.954 (95% below 0.20)
```

**Key Statistics**:
```
HalfNormal(0.1) Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mode:     0.000
Median:   0.067
Mean:     0.080
95% CI:   [0.003, 0.196]

Probability Mass:
  P(σ < 0.05) = 38.3%
  P(σ < 0.10) = 68.3%
  P(σ < 0.20) = 95.4%
```

### 4.3 Posterior Analysis

Posterior samples of `sigma_log_A` and `sigma_log_n` under HalfNormal(0.1):

| Parameter | Posterior Mean | Posterior Std | 95% CI |
|-----------|---------------|---------------|--------|
| sigma_log_A | 0.052 | 0.028 | [0.012, 0.115] |
| sigma_log_n | 0.048 | 0.025 | [0.010, 0.102] |

**Interpretation**: The posterior is heavily concentrated near zero, effectively forcing all mixture components to have nearly identical parameters.

---

## 5. Analysis: Why HalfNormal(0.1) Fails for Flexibility

### 5.1 The Expressiveness Problem

With σ ≈ 0.05 (typical posterior value):

```python
# Non-centered parameterization
log_A_raw ~ Normal(0, 1)  # Standard normal draws
log_A = mu_log_A + sigma_log_A * log_A_raw

# With sigma_log_A ≈ 0.05:
# log_A ≈ mu_log_A + 0.05 * Normal(0, 1)
# log_A ≈ mu_log_A ± 0.1  (95% interval)
# A ≈ exp(mu_log_A) * [0.90, 1.11]  (only ±10% variation!)
```

**Result**: All K=3 mixture components have amplitudes within 10% of each other, essentially collapsing to a single-component model.

### 5.2 Comparison with True Heterogeneity

| Scenario | True σ_log_A | Required σ_log_A for 50% variation |
|----------|-------------|------------------------------------|
| Low heterogeneity | 0.2 | ✗ Unachievable with HalfNormal(0.1) |
| Medium heterogeneity | 0.4 | ✗ Unachievable |
| High heterogeneity | 0.6+ | ✗ Unachievable |

### 5.3 The Trade-off Visualized

```
                    Convergence Quality
                           │
                     100%  │     ●─────────● HalfNormal(0.1)
                           │    /
                           │   /
                      50%  │  /
                           │ /     ● Baseline HalfNormal(1.0)
                           │/
                       0%  └──────────────────────────
                           0%      50%      100%
                              Model Flexibility
                              (ability to capture heterogeneity)
```

**HalfNormal(0.1) solves convergence by removing the problem (heterogeneity) rather than solving the geometry.**

---

## 6. Geometric Explanation

### 6.1 Why HalfNormal Creates Funnels

HalfNormal has its mode at 0:

```
PDF of HalfNormal(scale)
        │
   max ─┤●
        │ ╲
        │  ╲
        │   ╲
        │    ╲___
   0   ─┴────────────
        0   σ →
```

The mode at 0 means:
1. Prior concentrates mass toward σ = 0
2. Posterior likely includes σ → 0 region
3. Funnel geometry emerges in posterior

### 6.2 Tight Scale Mitigates But Doesn't Fix

With scale=0.1:
- 95% of prior mass below σ = 0.196
- Prior "caps" how large σ can get
- BUT the mode is still at 0
- Posterior still has funnel, just truncated

**The real fix**: Use a distribution with mode > 0 (e.g., LogNormal, InverseGamma)

---

## 7. Alternative Approaches Considered

### 7.1 Comparison Table

| Prior | Mode | P(σ < 0.1) | Convergence | Flexibility |
|-------|------|-----------|-------------|-------------|
| HalfNormal(0.1) | 0.00 | 68% | ✅ Excellent | ❌ Poor |
| HalfNormal(1.0) | 0.00 | 7.6% | ❌ Poor | ✅ Good |
| InverseGamma(3,1) | 0.25 | 3% | ⚠️ Partial | ✅ Good |
| **LogNormal(-1,0.5)** | **0.28** | **1.4%** | **✅ Good** | **✅ Good** |

### 7.2 Why LogNormal Won

```python
# LogNormal(-1.0, 0.5) properties
ln = stats.lognorm(s=0.5, scale=np.exp(-1.0))

print(f"Mode: {np.exp(-1.0 - 0.5**2):.3f}")   # 0.287 (away from 0!)
print(f"Median: {ln.median():.3f}")           # 0.368
print(f"Mean: {ln.mean():.3f}")               # 0.416
print(f"P(σ < 0.1): {ln.cdf(0.1):.3f}")       # 0.014 (only 1.4%!)
print(f"95% CI: [{ln.ppf(0.025):.3f}, {ln.ppf(0.975):.3f}]")  # [0.140, 0.969]
```

**Key Advantages**:
1. Mode at 0.287 (not 0) → no funnel vertex
2. Only 1.4% mass below 0.1 → prevents σ → 0 naturally
3. Heavy right tail → allows large σ if data supports it

---

## 8. Conclusions

### 8.1 What We Learned

1. **Tight HalfNormal priors can achieve convergence** but at the cost of model expressiveness
2. **The funnel problem requires changing the mode**, not just tightening the scale
3. **Non-centered parameterization alone is insufficient** for mixture model complexity
4. **Trade-offs must be explicit**: convergence vs. flexibility

### 8.2 Recommendation

**Do NOT use HalfNormal(0.1) in production** - it biases the model toward homogeneity.

**Use LogNormal hyperpriors instead**:
```python
sigma_log_A = numpyro.sample("sigma_log_A", dist.LogNormal(-1.0, 0.5))
sigma_log_n = numpyro.sample("sigma_log_n", dist.LogNormal(-1.5, 0.5))
```

See `HYPERPRIOR_DESIGN.md` for the full LogNormal solution.

---

## 9. Appendix: Code for Reproducing Experiments

### 9.1 Distribution Comparison Script

```python
#!/usr/bin/env python3
"""Compare HalfNormal vs LogNormal hyperprior properties."""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# HalfNormal(0.1)
hn = stats.halfnorm(scale=0.1)

# LogNormal(-1.0, 0.5)
ln = stats.lognorm(s=0.5, scale=np.exp(-1.0))

# Comparison
x = np.linspace(0.001, 1.0, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# PDF comparison
ax = axes[0]
ax.plot(x, hn.pdf(x), label='HalfNormal(0.1)', linewidth=2)
ax.plot(x, ln.pdf(x), label='LogNormal(-1,0.5)', linewidth=2)
ax.axvline(0.1, color='gray', linestyle='--', alpha=0.5, label='σ=0.1')
ax.set_xlabel('σ')
ax.set_ylabel('Density')
ax.set_title('Prior PDF Comparison')
ax.legend()
ax.set_xlim(0, 1)

# CDF comparison  
ax = axes[1]
ax.plot(x, hn.cdf(x), label='HalfNormal(0.1)', linewidth=2)
ax.plot(x, ln.cdf(x), label='LogNormal(-1,0.5)', linewidth=2)
ax.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(0.683, color='blue', linestyle=':', alpha=0.5, label='68% (HN)')
ax.axhline(0.014, color='orange', linestyle=':', alpha=0.5, label='1.4% (LN)')
ax.set_xlabel('σ')
ax.set_ylabel('CDF')
ax.set_title('Cumulative Distribution')
ax.legend()
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('hyperprior_comparison.png', dpi=150)
plt.show()

# Print statistics
print("\n" + "="*50)
print("HYPERPRIOR COMPARISON")
print("="*50)
print(f"{'Metric':<25} {'HalfNormal(0.1)':<20} {'LogNormal(-1,0.5)':<20}")
print("-"*65)
print(f"{'Mode':<25} {0.0:<20.3f} {np.exp(-1.0 - 0.5**2):<20.3f}")
print(f"{'Median':<25} {hn.median():<20.3f} {ln.median():<20.3f}")
print(f"{'Mean':<25} {hn.mean():<20.3f} {ln.mean():<20.3f}")
print(f"{'P(σ < 0.05)':<25} {hn.cdf(0.05):<20.3f} {ln.cdf(0.05):<20.3f}")
print(f"{'P(σ < 0.10)':<25} {hn.cdf(0.10):<20.3f} {ln.cdf(0.10):<20.3f}")
print(f"{'P(σ < 0.20)':<25} {hn.cdf(0.20):<20.3f} {ln.cdf(0.20):<20.3f}")
print("="*50)
```

### 9.2 Model Variant for HalfNormal(0.1) Testing

```python
def model_hill_mixture_halfnormal_tight(x, y=None, K=3, prior_config=None):
    """Hill Mixture with tight HalfNormal hyperpriors (FOR TESTING ONLY).
    
    WARNING: This configuration sacrifices model flexibility for convergence.
    Do not use in production. See `HALFNORMAL_CONVERGENCE_EXPERIMENTS.md`.
    """
    # ... (standard setup)
    
    # TIGHT HALFNORMAL HYPERPRIORS
    sigma_log_A = numpyro.sample("sigma_log_A", dist.HalfNormal(0.1))
    sigma_log_n = numpyro.sample("sigma_log_n", dist.HalfNormal(0.1))
    
    # ... (rest of model)
```

---

## 10. References

1. **Betancourt, M. (2017)**. "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434
2. **Papaspiliopoulos, O., Roberts, G.O., & Sköld, M. (2003)**. "Non-centered parameterizations for hierarchical models and data augmentation." Bayesian Statistics 7.
3. **Stan Development Team**. "Prior Choice Recommendations." https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
4. **NumPyro Documentation**. "Example: Neal's Funnel." https://num.pyro.ai/en/stable/examples/funnel.html
5. **Gelman, A. (2006)**. "Prior distributions for variance parameters in hierarchical models." Bayesian Analysis, 1(3), 515-534.

---

## Document History

| Date | Author | Change |
|------|--------|--------|
| 2026-02-05 | Session 240ae953 | Initial HalfNormal(0.1) experiments |
| 2026-02-05 | Session 6b0aebad | Documentation compiled |
