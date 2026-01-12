# Synthetic Benchmark Experimental Design

## 1. Overview

The default synthetic benchmark evaluates two model families against three Data Generating Processes (DGPs) in a fully crossed design. Each cell of the DGP x Model matrix is run across multiple random seeds to assess robustness. The benchmark is implemented as a pytest test suite (`tests/test_benchmark_synthetic.py`) with two tiers: **smoke** (single seed) and **full** (multi-seed).

```
Default test matrix:  3 DGPs  x  3 Models  x  {1, 5} seeds  =  {9, 45} cells
```

---

## 2. Data Generating Processes (DGPs)

All DGPs share a common structure:

```
x_t ~ LogNormal(mu=1.5, sigma=0.6),   t = 1, ..., T
s_t = x_t + alpha * s_{t-1}            (geometric adstock, alpha = 0.5)
baseline_t = beta_0 + beta_1 * t_std    (beta_0 = 50, beta_1 = 2)
```

where `t_std` is a standardized time index over `[0, 1]`.

### 2.1 Single Hill (`single`)

One response curve. Tests whether mixture models overfit when the true DGP has K=1.

```
y_t = baseline_t + A * s_t^n / (k^n + s_t^n) + eps_t,   eps_t ~ N(0, sigma^2)
```

| Parameter | Value |
|-----------|-------|
| A         | 30.0 |
| k         | median(s) |
| n         | 1.5 |
| sigma     | 3.0 |

### 2.2 Mixture K=2 (`mixture_k2`)

Two-component mixture. Tests basic mixture recovery.

```
z_t ~ Categorical(pi),   y_t = baseline_t + A_{z_t} * s_t^{n_{z_t}} / (k_{z_t}^{n_{z_t}} + s_t^{n_{z_t}}) + eps_t
```

| Component | pi   | A    | k definition            | n   |
|-----------|------|------|-------------------------|-----|
| 1         | 0.55 | 14.0 | q0.30 of adstocked `s`  | 1.5 |
| 2         | 0.45 | 30.0 | q0.80 of adstocked `s`  | 1.1 |

The K=2 curves are designed to stay ordered on the realized support and to remain separated by more than the observation noise scale on average.

### 2.3 Mixture K=3 (`mixture_k3`)

Three-component mixture. Tests recovery with more components.

| Component | pi   | A    | k / median(s) | n   |
|-----------|------|------|---------------|-----|
| 1         | 0.40 | 15.0 | 0.4           | 2.0 |
| 2         | 0.30 | 30.0 | 1.0           | 1.5 |
| 3         | 0.30 | 60.0 | 1.8           | 1.0 |

The K=3 case remains the harder benchmark with partial overlap between neighboring components.

All DGPs use T = 200 observations and sigma = 3.0.

For `mixture_k2`, the half-saturation points are tied to quantiles of the realized adstocked spend support instead of fixed multiples of `median(s)`. This makes the headline K=2 benchmark easier to justify in the paper: the DGP is intentionally identifiable on the support that is actually observed, rather than only on a hypothetical wider spend range.

---

## 3. Models Under Evaluation

### 3.1 Single Hill (`single_hill`)

Standard single-curve model (baseline):

```
alpha ~ Beta(2, 2)
A     ~ LogNormal(mu_A, 0.8)
k     ~ LogNormal(log(median(s)), 0.7)
n     ~ LogNormal(log(1.5), 0.4)
sigma ~ HalfNormal(sigma_scale)

y_t   = baseline_t + A * s_t^n / (k^n + s_t^n) + eps_t
```

### 3.2 Hierarchical Mixture (`mixture_k2`, `mixture_k3`)

All mixture variants use the same model function with K in {2, 3}.

**Mixture weights** (stick-breaking):

```
v_j ~ Beta(1, 1),   j = 1, ..., K-1
pi_1 = v_1
pi_j = v_j * prod_{i=1}^{j-1} (1 - v_i)
pi_K = prod_{i=1}^{K-1} (1 - v_i)
```

**Hierarchical priors** (non-centered parameterization):

```
mu_log_A    ~ N(mu_A^0, 0.5)
sigma_log_A ~ LogNormal(-1.2, 0.4)       # median ~ 0.30
mu_log_n    ~ N(log(1.5), 0.3)
sigma_log_n ~ LogNormal(-1.7, 0.4)       # median ~ 0.18

eta_j^A ~ N(0, 1)
log_A_j = mu_log_A + anchor_A_j + sigma_log_A * eta_j^A
A_j     = exp(log_A_j)

eta_j^n ~ N(0, 1)
log_n_j = mu_log_n + anchor_n_j + sigma_log_n * eta_j^n
n_j     = exp(log_n_j)
```

where `anchor_A_j = linspace(-1, 1, K) * 0.65` and `anchor_n_j = linspace(-0.8, 0.8, K) * 0.39` stabilize component separation.

**Ordered k** (identifiability constraint via cumulative increments):

```
log_k_1    ~ N(log(median(s)), 0.7)
Delta_j    ~ |N(0, 1)| * 0.7,   j = 1, ..., K-1
log_k_j    = log_k_1 + sum_{i=1}^{j-1} Delta_i,   j = 2, ..., K
k_j        = exp(log_k_j)
```

**Likelihood** (Gaussian mixture):

```
y_t ~ sum_{j=1}^{K} pi_j * N(baseline_t + A_j * s_t^{n_j} / (k_j^{n_j} + s_t^{n_j}), sigma^2)
```

Key reparameterizations applied: `LocScaleReparam(centered=0)` on intercept, slope, log_k_base, mu_log_A, mu_log_n.

---

## 4. Inference Configuration

| Setting          | Single Hill | Mixture K=2     | Mixture K=3     | Exception                          |
|------------------|-------------|-----------------|-----------------|------------------------------------|
| Warmup           | 600         | 900             | 1200            | Single on K3 DGP: 800              |
| Samples          | = warmup    | = warmup        | = warmup        |                                    |
| Chains           | 2           | 2               | 2               | Env var `HILL_MMM_SYNTHETIC_CHAINS` |
| Target accept    | 0.90        | 0.90            | 0.90            |                                    |
| Max tree depth   | 10          | 10              | 10              |                                    |
| Train/test split | 75% / 25%   | 75% / 25%       | 75% / 25%       |                                    |

**Seeds**: Smoke = `[0]`, Full = `[0, 1, 2, 3, 4]`.

---

## 5. Evaluation Metrics

### 5.1 MCMC Convergence Diagnostics (Three-Tier System)

| Diagnostic     | Pass      | Warn        | Fail    |
|----------------|-----------|-------------|---------|
| R-hat          | <= 1.01   | (1.01, 1.05]| > 1.05  |
| ESS per chain  | >= 100    | [50, 100)   | < 50    |
| Divergences    | = 0       | (0, 5]      | > 5     |
| BFMI           | >= 0.3    | [0.2, 0.3)  | < 0.2   |
| Tree depth hits| = 0       | (0, 10]     | > 10    |

For mixture models, convergence is assessed at three levels:

- **Standard**: Raw R-hat and ESS on all parameters
- **Label-invariant**: Diagnostics on permutation-invariant quantities (e.g., log-likelihood)
- **Relabeled**: Diagnostics after deterministic relabeling by k-ordering

### 5.2 Predictive Performance

- **MAPE** (Mean Absolute Percentage Error) on test set:

```
MAPE = (100 / T_test) * sum_t |y_t - y_hat_t| / (|y_t| + eps)
```

- **90% Interval Coverage**: Fraction of y_t falling within the posterior 5th--95th percentile
- **CRPS** (Continuous Ranked Probability Score): Proper scoring rule for the full posterior predictive distribution
- **nRMSE**: RMSE normalized by the test-set outcome range
- **LOO-CV** (PSIS-LOO): Expected log pointwise predictive density
- **WAIC**: Widely Applicable Information Criterion

### 5.3 Latent Mean Recovery (Synthetic Only)

Compares the posterior mean function to the true noise-free mean:

```
MAPE_mu = (100 / T) * sum_t |mu_hat_t - mu_true_t| / max(|mu_true_t|, eps)
MAE     = (1 / T) * sum_t |mu_hat_t - mu_true_t|
RMSE_mu = sqrt((1 / T) * sum_t (mu_hat_t - mu_true_t)^2)
nRMSE_mu = RMSE_mu / max(max_t mu_true_t - min_t mu_true_t, eps)
CRPS_mu = mean_t CRPS(F_mu_t, mu_true_t)
Coverage_90 = fraction of mu_true_t within [q_0.05, q_0.95]
Coverage_95 = fraction of mu_true_t within [q_0.025, q_0.975]
```

### 5.3.1 Interpretation for Mixture DGPs

For mixture models, the benchmark intentionally separates two predictive targets:

- **Observed-outcome prediction** asks whether the posterior predictive distribution matches the realized noisy observations `y_t`.
- **Latent-mean recovery** asks whether the fitted response surface matches the noise-free ground truth `mu_true_t`.

These are not interchangeable in synthetic mixture settings. The DGP samples one latent component per time step, while the fitted mixture model also exposes a marginal soft expectation across components. As a result:

- Posterior predictive `MAPE` on `y_t` is best interpreted as **marginal predictive fit**.
- Posterior predictive `CRPS` on `y_t` is the preferred **probabilistic predictive score**.
- Latent-mean `nRMSE_mu` is the preferred **headline recovery score** for the underlying response function.
- Latent-mean `MAPE_mu` remains a secondary descriptive metric.

The headline K=2 synthetic quality gate uses latent-mean `nRMSE_mu` rather than observed-outcome `MAPE`. This keeps the benchmark aligned with the scientific question for the paper: whether the model recovers the underlying heterogeneous Hill response, not whether a single point forecast reproduces the realized per-time latent draw.

### 5.4 Parameter Recovery

For scalar parameters (alpha, sigma, intercept, slope): checks whether the true value falls within the 95% credible interval.

### 5.5 Component Recovery (Mixture DGPs)

**Permutation-invariant alignment**: Exhaustive search over all permutations of posterior components matched to true components, minimizing a combined cost:

```
cost = weighted_nRMSE + w_unmatched_true + w_unmatched_posterior
```

Per matched pair (j_true, j_post):

- **Curve RMSE / nRMSE**: Evaluated on a normalized grid u in [0, 4] with 128 points
- **pi absolute error**: |pi_true - pi_post|
- **A relative error**: |A_true - A_post| / |A_true|
- **k-ratio relative error**: |r_k_true - r_k_post| / |r_k_true| where r_k = k / median(s)
- **n absolute error**: |n_true - n_post|

### 5.6 Effective K

```
K_eff = sum_{j=1}^{K} 1[pi_j > 0.05]
```

Counted per posterior sample, then averaged.

### 5.7 Across-Seed Component Stability (Full Benchmark Only)

For all C(5,2) = 10 seed pairs, aligns recovered components and aggregates:

- Active K consistency (mode fraction)
- Weighted curve nRMSE (mean, std, max)
- Weight and parameter error distributions

---

## 6. Quality Gates (Thresholds)

### 6.1 Smoke Test Gates

| Gate                       | Condition                                              |
|----------------------------|--------------------------------------------------------|
| Max test MAPE              | <= 5.0% (single DGP only)                              |
| Max latent test nRMSE      | <= 0.18 (`mixture_k2` DGP only)                        |
| Truth metrics              | Required (latent recovery + parameter recovery finite) |
| Reportable diagnostics     | Not required                                           |
| Finite LOO/WAIC            | Required                                               |
| Finite predictive metrics  | Required (`MAPE`, `nRMSE`, `CRPS`, coverage)           |

### 6.2 Full Benchmark Gates

Same as smoke, plus:

| Gate                       | Condition                                              |
|----------------------------|--------------------------------------------------------|
| Reportable diagnostics     | Required (publication status must not be "Fail")       |
| Max latent test nRMSE      | <= 0.15 (`mixture_k2` DGP only)                        |

### 6.3 Across-Seed Stability Assertions

- Number of seeds matches expected count (5)
- Pair count = C(5,2) = 10
- Stability JSON artifact is written

---

## 7. Test Execution

| Tier  | Command | Seeds | Cells | Opt-in |
|-------|---------|-------|-------|--------|
| Smoke | `pytest tests/test_benchmark_synthetic.py -m benchmark_smoke` | `[0]` | 4 | Default |
| Full  | `HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK=1 pytest tests/test_benchmark_synthetic.py -m benchmark_full` | `[0..4]` | 45 + 9 stability | Env var |

### 7.1 Artifacts Produced

Per cell: `paper/figures/synthetic/{model_name}/`

- `*_summary.json` -- Full diagnostic and metric summary
- `*_predictive.png` -- Observed vs posterior predictive plot
- `*_response.png` -- Component response curve recovery plot

Per DGP x Model (full only):

- `*_across_seed_stability.json` -- Cross-seed component alignment summary

After full benchmark completion:

- Publication figures (`fig0`--`fig4`) regenerated from fresh summaries

---

## 8. Design Rationale

| Choice | Rationale |
|--------|-----------|
| 3 DGPs x 3 models | Focuses the headline benchmark on the single, K=2, and K=3 cases used in the paper |
| T = 200 | Balances statistical power with realistic sample size |
| Observed-support `k` quantiles for K=2 | Ensures the headline K=2 mixture DGP is identifiable on the spend range that is actually sampled |
| Latent-mean nRMSE gate for K=2 | Aligns the synthetic quality gate with the mixture DGP's noise-free ground truth while avoiding the low-denominator instability of MAPE |
| k-ordering constraint | Resolves label switching in mixture components |
| Hierarchical priors | Partial pooling prevents component collapse |
| Non-centered parameterization | Avoids Neal's funnel geometry in hierarchical models |
| Component anchor offsets | Stabilizes component separation during sampling |
| 5 seeds | Assesses robustness; 10 pairwise comparisons for stability |
| Three-tier diagnostics | Separates publication viability (Pass/Warn) from hard failures |

---

## 9. Source Code References

| Component | File |
|-----------|------|
| Test suite | `tests/test_benchmark_synthetic.py` |
| DGP definitions | `src/hill_mixture_mmm/data.py` |
| Model definitions | `src/hill_mixture_mmm/models.py` |
| Core transforms | `src/hill_mixture_mmm/transforms.py` |
| Benchmark harness | `src/hill_mixture_mmm/benchmark.py` |
| Evaluation metrics | `src/hill_mixture_mmm/metrics.py` |
| Inference utilities | `src/hill_mixture_mmm/inference.py` |
| Figure generation | `src/hill_mixture_mmm/paper_figures.py` |
