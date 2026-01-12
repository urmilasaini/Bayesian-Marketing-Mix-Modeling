# Hill Mixture MMM Convergence Experiments - Handover Document

> Historical note: This handover memo summarizes earlier convergence experiments. It references pre-refresh benchmark settings, including `mixture_k5`, and should be treated as historical context rather than current benchmark guidance.

**Date**: 2026-02-05  
**Status**: Preliminary experiments completed, robust validation needed

---

## 1. Background: The Convergence Problem

### Problem Statement

The hierarchical Hill Mixture MMM (`model_hill_mixture_hierarchical_reparam`) experiences inconsistent MCMC convergence:
- R-hat values exceed 1.05 threshold
- ESS (Effective Sample Size) often below 100
- Convergence varies across different datasets

### Root Cause Analysis

The "funnel" geometry in hierarchical models where:
- Hierarchical scale parameters (`sigma_log_A`, `sigma_log_n`) approach zero
- Creates difficult posterior geometry for HMC sampler
- Standard remedy: Non-centered parameterization (NCP) - already implemented
- NCP alone insufficient for this mixture model complexity

---

## 2. Literature-Based Approaches Tested

Four approaches from prior research were tested:

| Approach | Reference | Hypothesis |
|----------|-----------|------------|
| A. Sparse Dirichlet | Malsiner-Walli et al. (2016) | Sparse prior auto-selects K, improves mixing |
| B. InverseGamma | Betancourt (2017), Stan best practices | Prevents σ→0 funnel |
| C. Marginalization | NumPyro/Pyro enumeration | Eliminates discrete latent label switching |
| D. Combined | A + B together | Synergistic improvement |

---

## 3. Experiment Design

### Setup

- **Git worktrees**: Isolated branches for parallel experiments
- **Model variants**: Modified `hill_mmm/models.py` in each worktree
- **Test script**: `scripts/test_convergence.py`

### MCMC Settings (Quick Test)

```python
num_warmup = 300-500
num_samples = 300-500
num_chains = 2
K = 3  # mixture components
```

### Convergence Criteria

| Metric | Threshold | Meaning |
|--------|-----------|--------|
| Max R-hat | ≤ 1.05 | Chain convergence |
| Min ESS (bulk) | ≥ 100 | Effective samples |

---

## 4. Experiment Results

### Summary Table

| Experiment | R-hat | ESS | Converged | Notes |
|------------|-------|-----|-----------|-------|
| Baseline (current) | ~1.05-1.10 | ~35-70 | ❌ | Inconsistent |
| A: Sparse Dirichlet | 1.50 | 4 | ❌ | Made worse |
| B: InverseGamma | **1.04** | 70 | ⚠️ Partial | Best R-hat, low ESS |
| C: Marginalization | 1.06 | 35 | ❌ | Not applicable* |
| D: Combined (A+B) | 2.23 | 3 | ❌ | Worst performance |

\* Model already uses `MixtureSameFamily` which marginalizes discrete assignments.

### Detailed Findings

#### A. Sparse Dirichlet Prior

**Change**:
```python
# Before: stick-breaking with Beta(1,1)
# After:
pis = numpyro.sample("pis", dist.Dirichlet(0.1 * jnp.ones(K)))
```

**Result**: Failed. The sparse Dirichlet (α=0.1) creates sharp boundaries where weights approach 0, adding difficult geometry rather than helping.

#### B. InverseGamma Hyperpriors

**Change**:
```python
# Before
sigma_log_A = numpyro.sample("sigma_log_A", dist.LogNormal(-1.0, 0.5))
sigma_log_n = numpyro.sample("sigma_log_n", dist.LogNormal(-1.5, 0.5))

# After
sigma_log_A = numpyro.sample("sigma_log_A", dist.InverseGamma(3.0, 1.0))
sigma_log_n = numpyro.sample("sigma_log_n", dist.InverseGamma(4.0, 1.0))
```

**Result**: Partial success. Best R-hat (1.04) but ESS still below threshold (70 vs 100).

#### C. Marginalization (config_enumerate)

**Change**: Added `@config_enumerate` decorator for parallel enumeration.

**Result**: Not applicable. The model already uses `MixtureSameFamily` which inherently marginalizes discrete assignments. Adding enumeration on top provides no benefit.

#### D. Combined Approach

**Change**: InverseGamma + Sparse Dirichlet together.

**Result**: Adverse interaction. Combined approach performed worst (R-hat=2.23), suggesting the modifications interfere with each other.

---

## 5. Critical Limitations of Current Experiments

### ⚠️ Validation Weaknesses

1. **Single dataset**: Only `mixture_k3` DGP scenario tested
2. **Single seed**: No variance estimation across random seeds
3. **Quick MCMC settings**: warmup=300-500, may be insufficient
4. **No real data**: Only synthetic data used
5. **Single organization**: Real data experiments need multiple orgs

### Required for Robust Conclusions

- Multiple random seeds (e.g., 5 seeds)
- Multiple DGP scenarios (single, mixture_k2, mixture_k3, mixture_k5)
- Multiple real-data organizations
- Full MCMC settings (warmup=1000+, samples=2000+, chains=4)

---

## 6. Next Steps: Recommended Experiments

### Phase 1: Robust Validation of InverseGamma (Priority: High)

The InverseGamma approach showed promise. Validate properly:

```bash
# Run with proper settings
python scripts/test_convergence.py \
    --n-orgs 5 \
    --warmup 1000 \
    --samples 1000 \
    --chains 4 \
    --seed 42  # Then repeat with seeds 0,1,2,3,4
```

**Metrics to collect**:
- Convergence rate (% of orgs that converge)
- Mean/std of R-hat and ESS across seeds
- Time per organization

### Phase 2: Alternative Approaches (Priority: Medium)

| Approach | Rationale | Implementation |
|----------|-----------|----------------|
| **Reduce K** | Simpler model may converge better | Test K=2 vs K=3 |
| **Two-stage fitting** | Fit baseline first, then mixture | Separate MCMC runs |
| **Variational inference** | SVI may handle geometry better | NumPyro SVI instead of MCMC |
| **Stronger NCP** | More aggressive reparameterization | Apply NCP to all parameters |

### Phase 3: Synthetic Data Validation (Priority: Medium)

Use `hill_mmm/data.py` to test parameter recovery:

```python
from hill_mixture_mmm.inference import run_inference

results = run_benchmark_suite(
    dgp_names=["single", "mixture_k2", "mixture_k3"],
    model_names=["hierarchical_reparam_k3"],
    seeds=[0, 1, 2, 3, 4],
    num_warmup=1000,
    num_samples=2000,
)
```

---

## 7. Git Worktrees (Cleanup Needed)

Experiment worktrees created:

```
/Users/shohei/Documents/repo/mmm-sparse-dirichlet     [experiment-sparse-dirichlet]
/Users/shohei/Documents/repo/mmm-inversegamma         [experiment-inversegamma]
/Users/shohei/Documents/repo/mmm-marginalization      [experiment-marginalization]
```

To clean up:
```bash
git worktree remove /Users/shohei/Documents/repo/mmm-sparse-dirichlet
git worktree remove /Users/shohei/Documents/repo/mmm-inversegamma
git worktree remove /Users/shohei/Documents/repo/mmm-marginalization
git branch -d experiment-sparse-dirichlet
git branch -d experiment-inversegamma
git branch -d experiment-marginalization
```

Or to preserve InverseGamma experiment:
```bash
# Keep only the promising branch
git worktree remove /Users/shohei/Documents/repo/mmm-sparse-dirichlet
git worktree remove /Users/shohei/Documents/repo/mmm-marginalization
git branch -d experiment-sparse-dirichlet
git branch -d experiment-marginalization
# Merge InverseGamma if it proves effective
```

---

## 8. Key References

1. **Papaspiliopoulos, Roberts & Sköld (2003)** - Non-centered parameterization theory
2. **Gorinova, Moore, Hoffman (2019)** - "Automatic Reparameterisation of Probabilistic Programs" [arXiv:1906.03028]
3. **Malsiner-Walli, Frühwirth-Schnatter & Grün (2016)** - "Model-based clustering based on sparse finite Gaussian mixtures" Statistics and Computing
4. **Betancourt (2017)** - "A Conceptual Introduction to Hamiltonian Monte Carlo" [arXiv:1701.02434]
5. **NumPyro Documentation** - Neal's Funnel example, GMM tutorial

---

## 9. Contact / Questions

For questions about this handover:
- Review `hill_mmm/models.py` for current model implementation
- Review `scripts/test_convergence.py` for convergence testing methodology
- Check git branches for experiment implementations
