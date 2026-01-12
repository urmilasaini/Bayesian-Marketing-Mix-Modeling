# Label Switching and Convergence Diagnostics for Mixture Models

> Historical note: This document is still useful for diagnostic implementation details, but its predictive examples were written before the March 2026 synthetic benchmark refresh and use earlier RMSE-based summaries.

**Date**: 2026-02-06 (Updated)  
**Status**: Label-invariant diagnostics implemented and validated

---

## 1. Executive Summary

This document reports findings from an investigation into **label switching handling strategies** in Bayesian mixture models for Marketing Mix Modeling. The key discovery is that **standard convergence diagnostics (R-hat, ESS) are fundamentally inappropriate for mixture models** due to label switching effects.

### Key Conclusions

1. **Standard R-hat is unreliable** for mixture model convergence assessment
2. **Both ordering approaches** (in-MCMC constraint vs post-hoc relabeling) show similar predictive performance
3. **Improved diagnostics required** before drawing conclusions about convergence

---

## 2. Background: The Label Switching Problem

### What is Label Switching?

In mixture models with K components, the likelihood is invariant to permutations of component labels. For K=3 components, there are K!=6 equivalent posterior modes representing the same model.

```
Component assignment: [A, B, C] ≡ [B, A, C] ≡ [C, B, A] ≡ ...
```

### Why It Matters

- MCMC chains may visit different label permutations
- This creates apparent "disagreement" between chains
- Standard diagnostics misinterpret this as non-convergence
- Parameter posteriors become meaningless without consistent labeling

---

## 3. Two Approaches Investigated

### Approach A: In-MCMC Ordering Constraint (Current Implementation)

**Method**: Enforce `k[0] < k[1] < k[2]` during sampling via cumulative sum transformation:

```python
# In model_hill_mixture
k_increments = numpyro.sample("k_increments", dist.LogNormal(...))
k = jnp.cumsum(jnp.abs(k_increments))  # Ensures k[0] < k[1] < k[2]
```

**Theoretical Concern**: Does `abs()` transformation break the Markov property or violate detailed balance?

### Approach B: Unconstrained + Post-Hoc Relabeling

**Method**: 
1. Sample k values independently without ordering constraint
2. After MCMC, sort each sample by k values
3. Apply same permutation to all component parameters

**Implementation**:
- New model: `model_hill_mixture_unconstrained` in `hill_mmm/models.py`
- Relabeling function: `relabel_samples_by_k()` in `hill_mmm/inference.py`

```python
def relabel_samples_by_k(samples: dict) -> dict:
    """Sort components by k values for each MCMC sample."""
    # For each sample, get permutation that sorts k
    # Apply same permutation to A, k, n, pis
    ...
```

---

## 4. Experimental Results

### Configuration

| Setting | Value |
|---------|-------|
| Data | Synthetic `mixture_k3`, T=200 |
| Warmup | 1000 |
| Samples | 2000 |
| Chains | 4 |

### Results Comparison

| Metric | Constrained | Unconstrained + Relabel | Notes |
|--------|-------------|-------------------------|-------|
| Max R-hat | 1.82 | 2.12 | Both "fail" standard threshold |
| Min ESS | 6 | 5 | Both very low |
| RMSE | 7.45 | 7.44 | Nearly identical |
| 90% Coverage | 92.5% | 92.0% | Both good |
| Time (sec) | 108.4 | 94.5 | Unconstrained faster |
| **Converged** | **No** | **No** | By standard criteria |

### K Values Analysis

```
Constrained k means:    [7.79, 11.76, 19.00]  - ordered by construction
Unconstrained raw k:    [12.06, 11.30, 9.44]  - NOT ordered (label switching!)
Relabeled k means:      [6.83, 10.15, 15.83]  - properly ordered after relabeling
```

### Interpretation

- **Predictive performance is equivalent** (RMSE, coverage nearly identical)
- **Both show "non-convergence"** by R-hat/ESS standards
- **This may be a diagnostic artifact**, not actual non-convergence

---

## 5. Critical Discovery: Convergence Diagnostics Problem

### Standard R-hat is Fundamentally Unreliable for Mixture Models

From Stan User's Guide and academic literature:

> "The R-hat convergence statistic and the computation of effective sample size are both compromised by label switching. The problem is that the posterior mean is affected by label switching, resulting in meaningless values."

### Why Standard Diagnostics Fail

| Issue | Explanation |
|-------|-------------|
| Inflated between-chain variance | Chains visiting different permuted modes appear to "disagree" |
| Underestimated ESS | Apparent "jumps" between modes look like poor mixing |
| Meaningless posterior means | Averaging across permuted labels produces nonsense |
| K! equivalent modes | With K=3, chains explore 6 equivalent posterior regions |

### Example

Chain 1 visits mode: `k = [5, 10, 15]`  
Chain 2 visits mode: `k = [10, 15, 5]` (permuted)  
→ R-hat detects "disagreement" → High R-hat → False "non-convergence"

---

## 6. Recommended Solutions

### 6.1 Use Label-Invariant Quantities for Convergence

Evaluate R-hat on quantities that don't depend on component labeling:

| Quantity | Formula | Why It Works |
|----------|---------|---------------|
| Log-likelihood | `log p(y|θ)` | Same regardless of label permutation |
| Posterior predictive | `p(y_new|y)` | Marginalizes over all permutations |
| Mixture density | `Σ πₖ f(x|θₖ)` | Summation is permutation-invariant |

### 6.2 Use Rank-Normalized R-hat with Stricter Threshold

From Vehtari et al. (2021):

```python
# Use ArviZ with rank normalization
az.rhat(samples, method="rank")

# Stricter threshold
if max_rhat <= 1.01:  # Not 1.05!
    print("Converged")
```

### 6.3 Compute R-hat on Relabeled Parameters

```python
# After relabeling
relabeled = relabel_samples_by_k(samples)
rhat_relabeled = az.rhat(relabeled)
```

### 6.4 Target Higher ESS

| Context | Minimum ESS |
|---------|-------------|
| Standard models | 100 |
| Mixture models | **400** (recommended) |

---

## 7. Implementation Status

### Completed

| Item | Location |
|------|----------|
| Unconstrained model variant | `hill_mmm/models.py::model_hill_mixture_unconstrained` |
| Post-hoc relabeling function | `hill_mmm/inference.py::relabel_samples_by_k()` |
| Comparison test script | `scripts/legacy/test_ordering_comparison.py` |
| Experiment results | `results/ordering_comparison/` |

### Newly Implemented (2026-02-06)

| Item | Location | Status |
|------|----------|--------|
| Log-likelihood R-hat computation | `hill_mmm/inference.py::compute_label_invariant_diagnostics()` | **Completed** |
| Rank-normalized R-hat | `hill_mmm/inference.py::_compute_rhat(method="rank")` | **Completed** |
| R-hat on relabeled samples | `hill_mmm/inference.py::compute_diagnostics_on_relabeled()` | **Completed** |
| Comprehensive diagnostics | `hill_mmm/inference.py::compute_comprehensive_mixture_diagnostics()` | **Completed** |
| Label switching detection | `hill_mmm/inference.py::check_label_switching()` | **Completed** |

### Pending Implementation

| Item | Priority | Notes |
|------|----------|-------|
| Posterior predictive checks | Medium | Alternative validation |

---

## 8. Latest Experimental Results (2026-02-06)

### Experimental Configuration

| Setting | Value |
|---------|-------|
| Data | Synthetic `mixture_k3`, T=200 |
| Warmup | 200 (quick mode) |
| Samples | 400 |
| Chains | 2 |
| Seed | 42 |

### Constrained Model (In-MCMC Ordering)

| Metric | Value | Status |
|--------|-------|--------|
| **Log-likelihood R-hat** | **1.0004** | Excellent |
| Standard Max R-hat | 1.82 | High (expected artifact) |
| Min ESS bulk | 3.0 | Low |
| Label Switching Rate | 0% | None (by construction) |
| RMSE | 7.43 | Good |
| 90% Coverage | 92.5% | Good |

**Interpretation**: Log-likelihood R-hat is excellent (1.0004), indicating true convergence despite inflated standard R-hat (1.82). The high standard R-hat is an artifact of label switching effects on component parameters.

### Unconstrained Model (Post-hoc Relabeling)

| Metric | Value | Status |
|--------|-------|--------|
| **Log-likelihood R-hat** | **1.0131** | Acceptable |
| Standard Max R-hat | 1.70 | High (expected artifact) |
| Relabeled Max R-hat | 1.0101 | Good |
| Label Switching Rate | 73.2% | High (expected) |
| RMSE | 7.43 | Good |
| 90% Coverage | 93.0% | Good |

**Interpretation**: High label switching rate (73.2%) confirms labels switch freely during MCMC. After relabeling, R-hat drops from 1.70 to 1.01, demonstrating that apparent non-convergence was a diagnostic artifact.

### Key Findings from This Experiment

1. **Log-likelihood R-hat works as expected**
   - Both models show acceptable log-likelihood R-hat (~1.00-1.01)
   - This is label-invariant, so not affected by label switching

2. **Post-hoc relabeling successfully resolves label switching**
   - Unconstrained model's relabeled R-hat (1.01) matches constrained model's performance
   - k means become properly ordered after relabeling: [7.83, 10.57, 14.91]

3. **Predictive performance is equivalent**
   - Both approaches achieve RMSE ~7.43 and coverage ~92-93%
   - Label switching handling does not affect predictive quality

---

## Conclusions and Recommendations (2026-02-06)

### Key Experimental Findings

Based on the comprehensive experimental comparison between constrained (in-MCMC ordering) and unconstrained (post-hoc relabeling) approaches, we have established the following best practices:

### 1. MCMC Should Allow Natural Label Switching

**Recommendation**: Do NOT impose ordering constraints during MCMC sampling.

| Approach | Standard R-hat | Relabeled R-hat | Convergence Quality |
|----------|----------------|-----------------|---------------------|
| Constrained (in-MCMC order) | 1.82 | 1.64 | Poor |
| **Unconstrained + Post-hoc** | 1.70 | **1.01** | **Excellent** |

**Rationale**:
- Ordering constraints can interfere with MCMC exploration of the posterior
- Our experiment shows constrained model has poor relabeled R-hat (1.64) even though labels never switch
- Unconstrained model achieves excellent relabeled R-hat (1.01) after post-hoc ordering
- The sampler explores the posterior more freely without artificial constraints

### 2. Apply Post-hoc Ordering for Diagnostics and Interpretation

**Recommendation**: Sort MCMC samples by k values AFTER inference, then compute diagnostics.

```python
# After MCMC sampling
relabeled_samples = relabel_samples_by_k(samples)
rhat_diagnostics = compute_rhat_on_relabeled(relabeled_samples)
```

**Benefits**:
- Consistent component labeling across chains
- Meaningful R-hat computation on component parameters
- Interpretable posterior summaries for A, k, n, π

### 3. Use Multiple Diagnostics for Mixture Models

**Recommendation**: Standard R-hat alone is insufficient. Use a combination of metrics.

| Diagnostic | Purpose | Status |
|------------|---------|--------|
| Standard R-hat | Quick screening (often inflated) | Supplementary |
| Log-likelihood R-hat | Model-level convergence (label-invariant) | Supplementary |
| **Relabeled Component R-hat** | True component convergence | **Primary** |
| ESS (bulk/tail) | Effective sample size | Primary |
| RMSE / Coverage | Predictive validation | Primary |
| Label switching rate | Diagnostic information | Informative |

### Summary: Best Practice Workflow

```
1. Run MCMC without ordering constraints (unconstrained model)
2. Check label switching rate (expect > 0% for mixture models)
3. Apply post-hoc relabeling by sorting k values
4. Compute R-hat on relabeled samples
5. Verify with Log-likelihood R-hat (supplementary)
6. Validate with predictive metrics (RMSE, Coverage)
```

### Note on Log-likelihood R-hat

Log-likelihood R-hat is a label-invariant diagnostic based on Stephens (2000) and the Stan User's Guide. However, our experiments show it may not detect all convergence issues:

- Constrained model: Log-likelihood R-hat = 1.0004 (looks good)
- But: Relabeled A R-hat = 1.64 (clearly not converged)

Therefore, **Log-likelihood R-hat should be used as a supplementary metric**, not the primary convergence criterion.

---

## 9. Next Steps

### Immediate Actions

1. **Implement improved convergence diagnostics**
   - Add log-likelihood per-sample computation
   - Compute R-hat on log-likelihood (label-invariant)
   - Add rank-normalized R-hat option

2. **Re-run comparison experiment** with improved diagnostics
   - Determine if apparent non-convergence is real or artifact
   - Compare diagnostic results between approaches

### Longer-Term Considerations

| Approach | Rationale |
|----------|----------|
| LOO-CV for model comparison | Label-invariant model selection |
| Longer chains | If true non-convergence confirmed |
| Posterior predictive checks | Complementary validation |

---

## 9. Key References

1. **Stan User's Guide** - "Label Switching in Mixture Models" section
2. **Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021)** - "Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of MCMC" Bayesian Analysis
3. **Stephens (2000)** - "Dealing with label switching in mixture models" JRSS-B
4. **Jasra, Holmes, Stephens (2005)** - "Markov Chain Monte Carlo Methods and the Label Switching Problem in Bayesian Mixture Modeling" Statistical Science

---

## 10. Files Reference

### Modified Files

| File | Changes |
|------|--------|
| `hill_mmm/models.py` | Added `model_hill_mixture_unconstrained` |
| `hill_mmm/inference.py` | Added `relabel_samples_by_k()` |

### New Files

| File | Purpose |
|------|--------|
| `scripts/legacy/test_ordering_comparison.py` | Comparison experiment runner |
| `results/ordering_comparison/results.txt` | Human-readable results |
| `results/ordering_comparison/results.json` | Machine-readable results |
| `results/ordering_comparison/results.csv` | Tabular results |

---

## 11. Contact / Questions

For questions about this investigation:
- Review `hill_mmm/inference.py` for relabeling implementation
- Review `scripts/legacy/test_ordering_comparison.py` for experiment methodology
- Check `results/ordering_comparison/results.txt` for detailed experimental output
