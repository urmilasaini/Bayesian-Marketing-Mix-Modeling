# Component Resolvability Benchmark: Experimental Design

## 1. Overview

The component resolvability benchmark evaluates whether mixture Hill models can recover the effective number of distinct response components as the true component separation varies. It uses a controlled synthetic DGP where the spacing between Hill saturation points ($k$ values) is systematically varied.

```
Test matrix:  9 profiles  x  2 models  x  {1, 5} seeds  =  {18, 90} cells
```

The benchmark is implemented in `tests/test_benchmark_component_resolvability.py` with two tiers:
- **Smoke:** 3 representative profiles, 1 seed
- **Full:** all 9 profiles, 5 seeds (opt-in via `HILL_MMM_RUN_FULL_COMPONENT_RESOLVABILITY_BENCHMARK=1`)

---

## 2. Data Generating Process

### 2.1 Observation model

The DGP generates a time series of length $T$ from a finite mixture of Hill response curves:

$$
y_t \mid z_t \sim \mathcal{N}\!\bigl(\mu_t^{(z_t)},\; \sigma^2\bigr), \qquad z_t \sim \mathrm{Categorical}(\boldsymbol{\pi})
$$

where the component-specific mean is:

$$
\mu_t^{(k)} = \underbrace{\beta_0 + \beta_1 \, \tilde{t}}_{\text{baseline}} + \underbrace{A_k \,\frac{s_t^{\,n_k}}{k_k^{\,n_k} + s_t^{\,n_k}}}_{\text{Hill response of component } k}
$$

and the mixture-expected latent mean is:

$$
\bar\mu_t = \beta_0 + \beta_1 \, \tilde{t} + \sum_{k=1}^{K} \pi_k \, A_k \,\frac{s_t^{\,n_k}}{k_k^{\,n_k} + s_t^{\,n_k}}
$$

### 2.2 Spend and adstock

Raw media spend is drawn from a log-normal distribution:

$$
x_t \sim \mathrm{LogNormal}(\mu_x,\; \sigma_x), \qquad t = 1, \dots, T
$$

Geometric adstock transforms raw spend into effective exposure:

$$
s_t = x_t + \alpha \, s_{t-1}, \qquad s_0 = 0
$$

The standardized time index used for the baseline trend is:

$$
\tilde{t} = \frac{2(t - 1)}{T - 1} - 1 \in [-1, 1]
$$

### 2.3 Controlled $k$-spacing

The saturation half-points $k_k$ are parameterized via a center ratio $c$ and a spacing parameter $\delta$:

| $K_\mathrm{true}$ | $k$-ratio vector |
|---|---|
| 1 | $[c]$ |
| 2 | $[c - \delta,\; c + \delta]$ |
| 3 | $[c - \delta,\; c,\; c + \delta]$ |

The absolute saturation points are:

$$
k_k = r_k \cdot \mathrm{median}(s)
$$

where $r_k$ is the $k$-th entry of the ratio vector. The constraint $c - \delta > 0$ ensures all ratios are positive.

### 2.4 Default DGP parameters

| Parameter | Symbol | Default |
|---|---|---|
| Time series length | $T$ | 200 |
| Observation noise | $\sigma$ | 3.0 |
| Adstock decay | $\alpha$ | 0.5 |
| Baseline intercept | $\beta_0$ | 50.0 |
| Baseline slope | $\beta_1$ | 2.0 |
| Spend log-mean | $\mu_x$ | 1.5 |
| Spend log-std | $\sigma_x$ | 0.6 |
| Center $k$-ratio | $c$ | 0.9 |

---

## 3. Profile Library

Each profile fixes $\delta$, $\boldsymbol{\pi}$, $\mathbf{A}$, and $\mathbf{n}$ for a given $K_\mathrm{true}$.

### 3.1 $K=1$ profiles

| Profile ID | $\delta$ | $\boldsymbol{\pi}$ | $\mathbf{A}$ | $\mathbf{n}$ |
|---|---|---|---|---|
| `tv00_anchor` | 0.00 | $(1.0)$ | $(50)$ | $(2.5)$ |

### 3.2 $K=2$ profiles

| Profile ID | $\delta$ | $\boldsymbol{\pi}$ | $\mathbf{A}$ | $\mathbf{n}$ |
|---|---|---|---|---|
| `tv07_low` | 0.05 | $(0.60, 0.40)$ | $(50, 50)$ | $(2.5, 2.5)$ |
| `tv27_mid` | 0.20 | $(0.60, 0.40)$ | $(50, 50)$ | $(2.5, 2.5)$ |
| `tv59_high` | 0.45 | $(0.60, 0.40)$ | $(50, 50)$ | $(2.5, 2.5)$ |
| `tv94_extreme` | 0.80 | $(0.60, 0.40)$ | $(50, 50)$ | $(2.5, 2.5)$ |

### 3.3 $K=3$ profiles

| Profile ID | $\delta$ | $\boldsymbol{\pi}$ | $\mathbf{A}$ | $\mathbf{n}$ |
|---|---|---|---|---|
| `tv05_low` | 0.05 | $(0.50, 0.30, 0.20)$ | $(50, 50, 50)$ | $(2.5, 2.5, 2.5)$ |
| `tv18_mid` | 0.20 | $(0.50, 0.30, 0.20)$ | $(50, 50, 50)$ | $(2.5, 2.5, 2.5)$ |
| `tv41_high` | 0.45 | $(0.50, 0.30, 0.20)$ | $(50, 50, 50)$ | $(2.5, 2.5, 2.5)$ |
| `tv73_extreme` | 0.80 | $(0.50, 0.30, 0.20)$ | $(50, 50, 50)$ | $(2.5, 2.5, 2.5)$ |

The profile ID encodes the approximate mean pairwise TV distance (e.g., `tv27` $\approx 0.27$).

---

## 4. Inference Model

### 4.1 Adstock and baseline

$$
\alpha \sim \mathrm{Beta}(2, 2)
$$
$$
\mathrm{intercept} \sim \mathcal{N}(\hat\beta_0,\; \sigma_{\beta_0})
$$
$$
\mathrm{slope} \sim \mathcal{N}(0,\; \sigma_{\beta_1})
$$

where $\hat\beta_0$ and scales are computed from data via empirical Bayes.

### 4.2 Stick-breaking mixture weights

For a $K$-component mixture, the weights are constructed via stick-breaking:

$$
v_i \sim \mathrm{Beta}(\alpha_s, \beta_s), \qquad i = 1, \dots, K{-}1
$$

$$
\pi_1 = v_1, \qquad \pi_i = v_i \prod_{j=1}^{i-1}(1 - v_j), \qquad \pi_K = \prod_{j=1}^{K-1}(1 - v_j)
$$

Benchmark prior overrides:

| | $K=2$ | $K=3$ |
|---|---|---|
| $\alpha_s$ | 0.7 | 5.0 |
| $\beta_s$ | 0.7 | 2.5 |

### 4.3 Hierarchical amplitude and exponent

The component amplitudes and exponents share a hierarchical structure with non-centered parameterization (NCP):

$$
\mu_{\log A} \sim \mathcal{N}(\hat{A}_{\mathrm{loc}},\; 0.5), \qquad
\sigma_{\log A} \sim \mathrm{LogNormal}(\lambda_A,\; \tau_A)
$$
$$
\epsilon^A_k \sim \mathcal{N}(0, 1), \qquad
\log A_k = \mu_{\log A} + a^A_k + \sigma_{\log A} \, \epsilon^A_k
$$

$$
\mu_{\log n} \sim \mathcal{N}(\log 1.5,\; 0.3), \qquad
\sigma_{\log n} \sim \mathrm{LogNormal}(\lambda_n,\; \tau_n)
$$
$$
\epsilon^n_k \sim \mathcal{N}(0, 1), \qquad
\log n_k = \mu_{\log n} + a^n_k + \sigma_{\log n} \, \epsilon^n_k
$$

where $a^A_k$ and $a^n_k$ are deterministic component anchors that spread components apart for identifiability:

$$
a^A_k = \mathrm{linspace}(-1, 1, K)_k \cdot \rho, \qquad
a^n_k = \mathrm{linspace}(-0.8, 0.8, K)_k \cdot \rho \cdot \gamma_n
$$

with anchor strength $\rho$ (default: 0.8 for $K{=}2$, 0.7 for $K{=}3$) and $\gamma_n = 0.4$ for $K{=}2$, $0.6$ for $K{=}3$.

Benchmark prior overrides for the hierarchical scales:

| Parameter | $K=2$ | $K=3$ |
|---|---|---|
| $\lambda_A$ | $-1.3$ | $-1.5$ |
| $\tau_A$ | $0.20$ | $0.18$ |
| $\lambda_n$ | $-1.8$ | $-1.9$ |
| $\tau_n$ | $0.20$ | $0.18$ |

### 4.4 Ordered $k$-values via quantile anchoring

To enforce component ordering ($k_1 < k_2 < \cdots < k_K$), the model anchors $k$-values to quantiles of the adstocked spend $s$:

$$
\hat{k}^{\mathrm{anch}}_i = \mathrm{quantile}(s,\; q_i), \qquad i = 1, \dots, K
$$

| $K$ | Anchor quantiles $(q_1, \dots, q_K)$ |
|---|---|
| 2 | $(0.25, 0.85)$ |
| 3 | $(0.20, 0.55, 0.90)$ |

The base log-$k$ and increments are:

$$
\log k_{\mathrm{base}} \sim \mathcal{N}\!\bigl(\log \hat{k}^{\mathrm{anch}}_1,\; \sigma_{\mathrm{anchor}}\bigr)
$$
$$
\eta_i \sim \mathcal{N}(0, 1), \qquad i = 1, \dots, K{-}1
$$
$$
g_i = \max\!\bigl(\log \hat{k}^{\mathrm{anch}}_{i+1} - \log \hat{k}^{\mathrm{anch}}_i,\; 10^{-3}\bigr)
$$

For $K=2$, increments are forced positive via folding:

$$
\Delta_i = g_i + |\eta_i| \cdot \sigma_{\mathrm{inc}}
$$

For $K \geq 3$, increments use a soft floor:

$$
\Delta_i = \max\!\bigl(g_i + \eta_i \cdot \sigma_{\mathrm{inc}},\; 10^{-3}\bigr)
$$

The ordered log-$k$ values are:

$$
\log k_1 = \log k_{\mathrm{base}}, \qquad \log k_j = \log k_{\mathrm{base}} + \sum_{i=1}^{j-1} \Delta_i
$$

Benchmark prior overrides:

| Parameter | $K=2$ | $K=3$ |
|---|---|---|
| $\sigma_{\mathrm{anchor}}$ | 0.06 | 0.06 |
| $\sigma_{\mathrm{inc}}$ | 0.04 | 0.04 |

### 4.5 Likelihood

$$
\sigma \sim \mathrm{HalfNormal}(\sigma_{\mathrm{scale}})
$$

$$
y_t \sim \sum_{k=1}^{K} \pi_k \, \mathcal{N}\!\bigl(\mu_t^{(k)},\; \sigma^2\bigr)
$$

where $\mu_t^{(k)} = \mathrm{baseline}_t + A_k \, s_t^{n_k} / (k_k^{n_k} + s_t^{n_k})$.

### 4.6 Non-centered reparameterization

`LocScaleReparam(centered=0)` is applied to: `intercept`, `slope`, `log_k_base`, `mu_log_A`, `mu_log_n`.

---

## 5. MCMC Configuration

All runs use NUTS with the following settings:

| Setting | $K=2$ (quick) | $K=3$ (quick) | $K=2$ (full) | $K=3$ (full) |
|---|---|---|---|---|
| `num_warmup` | 1400 | 2200 | 2200 | 3200 |
| `num_samples` | 1000 | 1800 | 1800 | 2600 |
| `num_chains` | 2 | 2 | 2 | 2 |
| `target_accept_prob` | 0.997 | 0.995 | 0.99 | 0.997 |
| `max_tree_depth` | 16 | 16 | 14 | 17 |
| `init_strategy` | median | median | median | uniform |
| `dense_mass` | false | false | false | false |

The inference seed is offset from the data seed: $s_{\mathrm{inf}} = s + 7$ (quick $K{=}2$), $s + 17$ (quick $K{=}3$, full $K{=}2$), $s + 97$ (full $K{=}3$).

---

## 6. Evaluation Metrics

### 6.1 True component separation (horizontal axis)

Let $f_k(u) = A_k \, u^{n_k} / (k_k^{n_k} + u^{n_k})$ be the Hill response curve of component $k$, evaluated on a grid $u \in [0, u_{\max}]$ of size $G$ (defaults: $u_{\max} = 4.0$, $G = 128$). The incremental response mass is:

$$
m_k = \frac{\Delta f_k}{\|\Delta f_k\|_1}, \qquad (\Delta f_k)_j = f_k(u_{j+1}) - f_k(u_j)
$$

The primary horizontal axis is the **mean pairwise cosine distance**:

$$
d_{\cos}(i, j) = 1 - \frac{m_i \cdot m_j}{\|m_i\| \, \|m_j\|}
$$

$$
\bar{d}_{\cos} = \binom{K}{2}^{-1} \sum_{i < j} d_{\cos}(i, j)
$$

### 6.2 Effective component count (vertical axis)

The vertical axis is the **Shannon effective count** (Hill number $q=1$):

$$
{}^1\!D = \exp\!\Bigl(-\sum_{k=1}^{K} w_k \log w_k\Bigr)
$$

where $w_k = \hat\pi_k$ are the posterior mean mixture weights (all $K$ components, including near-zero ones).

---

## 7. Convergence Diagnostics

### 7.1 HMC sampler diagnostics

| Diagnostic | Pass | Warn | Fail |
|---|---|---|---|
| Divergent transitions | 0 | 1--5 | $>5$ |
| Min BFMI | $\geq 0.3$ | $[0.2, 0.3)$ | $< 0.2$ |
| Tree depth hits | 0 | 1--10 | $>10$ |

### 7.2 Label-invariant diagnostics

For mixture models, standard $\hat{R}$ on component parameters is unreliable due to label switching. Instead, $\hat{R}$ and ESS are computed on:

1. **Log-likelihood** (label-invariant by definition):

$$
\ell_d = \sum_{t=1}^{T} \log \sum_{k=1}^{K} \pi_k^{(d)} \, \phi\!\bigl(y_t \mid \mu_t^{(k,d)}, \sigma^{(d)}\bigr)
$$

2. **Scalar parameters** (intercept, slope, $\sigma$, $\alpha$) that are not affected by label permutations.

| Diagnostic | Pass | Warn | Fail |
|---|---|---|---|
| $\hat{R}_{\max}$ (label-invariant) | $\leq 1.01$ | $(1.01, 1.05]$ | $> 1.05$ |
| ESS$_{\mathrm{bulk}}$ / chain | $\geq 100$ | $[50, 100)$ | $< 50$ |
| ESS$_{\mathrm{tail}}$ / chain | $\geq 100$ | $[50, 100)$ | $< 50$ |

### 7.3 Relabeled diagnostics

Posterior samples are relabeled by sorting components by $k$ at each draw. Then $\hat{R}$ and ESS are computed per component for $A$, $k$, $n$, $\pi$. The same thresholds as label-invariant apply.

### 7.4 Publication status

$$
\mathrm{publication\_status} = \max(\mathrm{sampler\_status},\; \mathrm{mixing\_status})
$$

where $\max$ follows the ordering Pass $<$ Warn $<$ Fail, and:
- **sampler_status** is determined by HMC diagnostics (Section 7.1)
- **mixing_status** is determined by label-invariant diagnostics (Section 7.2)
- **interpretation_status** is determined by relabeled diagnostics (Section 7.3)

### 7.5 Effective convergence

A case is "effectively converged" when all of the following hold:
- HMC diagnostics pass (0 divergences, BFMI $\geq 0.3$, 0 tree depth hits)
- Label-invariant $\hat{R}_{\max} \leq 1.01$ and ESS $\geq 100 \times C$ (where $C$ = number of chains)
- Relabeled $\hat{R}_{\max} \leq 1.01$ and ESS $\geq 100 \times C$

---

## 8. Test Design

### 8.1 Models under test

| Model name | Model function | $K$ |
|---|---|---|
| `mixture_k2` | `model_hill_mixture_hierarchical_reparam` | 2 |
| `mixture_k3` | `model_hill_mixture_hierarchical_reparam` | 3 |

### 8.2 Tiers

| Tier | Profiles | Seeds | Guard |
|---|---|---|---|
| Smoke | `tv00_anchor`, `tv59_high`, `tv41_high` | $\{0\}$ | None |
| Full | All 9 profiles | $\{0, 1, 2, 3, 4\}$ | Env var required |

### 8.3 Pass criteria

The test uses `_controlled_thresholds()` which requires:
- `require_reportable_diagnostics = True` (publication_status $\neq$ Fail)
- `require_finite_loo_waic = True`
- `require_finite_predictive_metrics = True`
- `require_truth_metrics = True`

All numeric thresholds (max_rhat, min_ess, max_divergences, etc.) are set to `None` in the test --- the only hard gate is that publication_status must not be Fail and that LOO/WAIC/predictive/truth metrics are finite.

### 8.4 Artifact generation

The `test_component_resolvability_selected_metric_artifacts` test aggregates results across all seeds and produces:
- `selected_metric_results.csv`: per-case raw metrics
- `selected_metric_summary.csv`: grouped means by ($K_{\mathrm{true}}$, profile, model)
- `selected_metric_comparison.png`: Shannon count vs true cosine separation scatter plot
- `metadata.json`: run configuration

---

## 9. References

- Leinster, T. & Cobbold, C. A. (2012). Measuring diversity: the importance of species similarity. *Ecology*, 93(3), 477--489.
- Hill, M. O. (1973). Diversity and evenness: a unifying notation and its consequences. *Ecology*, 54(2), 427--432.
- Vehtari, A. et al. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667--718.
