# Real Data Validation Progress Report

> Historical note: This progress log predates the current paper-facing synthetic benchmark. It retains earlier RMSE-based checkpoints and exploratory real-data notes for implementation history only.

## Summary

This document tracks progress on validating Hill mixture MMM models using real Conjura eCommerce dataset.

**Branch**: `feat/real-data-loader`  
**Date**: 2026-02-04

---

## 1. Completed Work

### 1.1 Data Loader Module (`hill_mmm/data_loader.py`)

✅ **Created comprehensive data loader** for Conjura MMM dataset with:
- `TimeSeriesConfig`: Configuration dataclass for loading options
- `LoadedData`: Output dataclass with x, y arrays and metadata
- `list_timeseries()`: List available organisation/territory time series
- `load_timeseries()`: Load a single organisation's data with organisation-level isolation
- `get_active_channels()`: Identify channels with sufficient non-zero data
- `select_representative_timeseries()`: Sample representative time series for benchmarking

**Key Features**:
- Organisation-level data isolation (no data sharing between organisations)
- Auto-detection of active ad spend channels
- Aggregated spend mode (Phase 1: sum all channels into single x)
- Train/test splitting capability

### 1.2 Real Benchmark Runner (`hill_mmm/real_benchmark.py`)

✅ **Created benchmark infrastructure** with:
- `RealBenchmarkConfig`: Configuration for benchmark runs
- `run_real_experiment()`: Single experiment runner
- `run_real_benchmark()`: Multi-organisation benchmark coordinator

### 1.3 Unit Tests (`tests/test_data_loader.py`)

✅ **17 unit tests** covering:
- `list_timeseries()` functionality
- `get_active_channels()` detection
- `load_timeseries()` data loading
- `select_representative_timeseries()` sampling
- Edge cases (empty data, missing organisations)

### 1.4 Quick Validation Script (`scripts/run_quick_validation.py`)

✅ **Validated end-to-end pipeline**:
- Successfully ran single_hill model on real data
- MCMC converged (R-hat: 1.020 < 1.05)
- Train RMSE: 9.31, Test RMSE: 10.81
- Train Coverage 90%: 89.6%, Test Coverage 90%: 82.2%
- ELPD-LOO: -4798.0

---

## 2. Current Issues

### 2.1 Model Comparison Script Errors

The `scripts/run_model_comparison.py` encountered issues:

1. **`compute_loo()` API mismatch**: 
   - Current code calls `compute_loo(mcmc, model_fn, x, y, ...)`
   - Actual signature is `compute_loo(mcmc)` (takes only MCMC object)
   - **Fix needed**: Change to `compute_loo(mcmc)` only

2. **Mixture model convergence issues**:
   - `mixture_k3`: R-hat: 2.260, ESS: 3 (NOT converged)
   - `sparse_k5`: R-hat: 2.290, ESS: 3 (NOT converged)
   - Step sizes extremely small (~1e-9) indicating difficult geometry
   - **Likely cause**: Identifiability issues with mixture components on real data

### 2.2 Single Hill Model Results

**Worked successfully**:
- R-hat: 1.020 (good convergence)
- ESS: 137 (acceptable)
- Step size: ~3.95e-02 (healthy)

---

## 3. Next Steps

### 3.1 Immediate Fixes (Priority 1)

1. **Fix `compute_loo()` call** in `run_model_comparison.py`:
   ```python
   # Change from:
   loo_result = compute_loo(mcmc, spec.fn, x_train, y_train, prior_config, **spec.kwargs)
   # To:
   loo_result = compute_loo(mcmc)
   ```

2. **Fix predictions/metrics computation** to not pass model kwargs to LOO

### 3.2 Model Improvements (Priority 2)

1. **Investigate mixture model convergence**:
   - Try more warmup iterations (1000+)
   - Consider reparameterization
   - Check if data supports multiple mixture components

2. **Compare single vs mixture on multiple organisations**:
   - Run on 5-10 representative organisations
   - Aggregate delta-LOO scores

### 3.3 Multi-Channel Support (Priority 3)

1. **Phase 2**: Extend to multi-channel x (T, C) instead of aggregated
2. **Channel-specific Hill curves** for each ad channel

---

## 4. Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `hill_mmm/data_loader.py` | ✅ NEW | Data loading for Conjura dataset |
| `hill_mmm/real_benchmark.py` | ✅ NEW | Benchmark runner for real data |
| `tests/test_data_loader.py` | ✅ NEW | 17 unit tests |
| `scripts/run_quick_validation.py` | ✅ NEW | Quick validation script |
| `scripts/run_model_comparison.py` | ⚠️ NEW | Has API errors, needs fix |

---

## 5. Git Commits on `feat/real-data-loader`

1. `5dcaa63` - feat: Add data loader for Conjura MMM dataset
2. `4052406` - feat: Add real data benchmark runner for Conjura dataset
3. `8d5d211` - fix: Handle empty results in list_timeseries
4. `e269fc6` - test: Add unit tests for data_loader module
5. `b7a31b9` - chore: Update uv.lock from test execution

---

## 6. Key Findings

### 6.1 Single Hill Model Performance

- **Works well** on real Conjura data
- Convergence is reliable
- Reasonable predictive accuracy

### 6.2 Mixture Models

- **Not converging** with current settings
- May require:
  - More data per organisation
  - Better initialization
  - Different priors
  - Reparameterization for identifiability

### 6.3 Data Characteristics

- 143 time series available (organisation × territory combinations)
- Longest series: 1751 days
- Most organisations have 1-3 active ad channels
- Verticals: Autos & Vehicles, and others

---

## 7. How to Continue

```bash
# Switch to the branch
git checkout feat/real-data-loader

# Run quick validation (works)
uv run python scripts/run_quick_validation.py

# Fix and run model comparison
# Edit scripts/run_model_comparison.py first to fix compute_loo() call
uv run python scripts/run_model_comparison.py
```

---

**Last Updated**: 2026-02-04 15:14 JST
