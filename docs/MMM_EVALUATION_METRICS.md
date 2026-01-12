# MMM Evaluation Metrics for Synthetic and Applied Benchmarks

**Date:** 2026-03-29

## Summary

The MMM literature does not treat model evaluation as a single-number problem. The common pattern across Google Meridian, Robyn, PyMC-Marketing, and the older Google MMM papers is to separate at least five layers:

1. **Sampler/model health**: whether posterior inference is trustworthy.
2. **Predictive fit**: whether the model predicts observed outcomes well.
3. **Probabilistic calibration**: whether predictive intervals and posterior predictive distributions are well calibrated.
4. **Attribution or causal recovery**: whether the model recovers media contributions, ROI, or other causal quantities of interest.
5. **Decision robustness**: whether budget allocation recommendations are stable under posterior uncertainty.

This matters for our synthetic benchmark because `MAPE` and even `ELPD-LOO` mainly assess predictive fit, while the scientific question of this repository is closer to **response-function recovery** and **heterogeneity recovery**. Those are not the same target.

## What The Current Synthetic Benchmark Measures

The current synthetic benchmark mainly reports:

- `ELPD-LOO` / `WAIC`
- observed-outcome `MAPE`
- latent-mean `MAPE`
- 90% interval coverage
- `RMSE` in logs and some component-level `nRMSE`

This is already better than a pure RMSE benchmark, but it still leaves gaps:

- `MAPE` is unstable when the denominator is small and can overweight low-volume periods.
- point-error metrics on `y_t` do not tell us whether media effects or mixture structure were recovered correctly.
- 90% coverage alone does not distinguish between sharp calibrated posteriors and very wide but nominally calibrated ones.
- `ELPD-LOO` rewards predictive density, but a model can improve fit by absorbing media signal into trend or baseline rather than recovering the true causal decomposition.

## Literature Review

### 1. Foundational Google MMM Papers

**Jin et al. (2017), "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects"** uses:

- posterior ROAS and mROAS summaries
- model selection over shape/carryover specifications with `BIC`
- simulation-based recovery checks

The paper explicitly reports that the model can work well on large data, but that small samples can produce biased estimates and that posterior uncertainty propagates into budget optimization. That is an argument for evaluating:

- parameter/effect recovery, not only prediction
- uncertainty width and calibration
- stability of optimal allocation, not only point estimates

Source:
- [Google Research publication page](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/)

**Chan and Perry (2017), "Challenges and Opportunities in Media Mix Modeling"** is less about a metric checklist and more about why predictive fit is not enough. The paper frames MMM as a causal measurement problem with persistent identification challenges such as confounding, limited variation, and multicollinearity.

Source:
- [Google Research publication page](https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/)

**Sun et al. (2017), "Geo-level Bayesian Hierarchical Media Mix Modeling"** emphasizes that more granular data improves estimation by tightening credible intervals. That implies interval quality and uncertainty reduction are first-class outcomes, not secondary reporting artifacts.

Source:
- [Google Research publication page](https://research.google/pubs/geo-level-bayesian-hierarchical-media-mix-modeling/)

### 2. Meridian

Google Meridian is explicit that **causal reliability must be checked before business use**. Its official health checks cover:

- `max_r_hat` for convergence
- negative baseline probability
- Bayesian posterior predictive p-value (`PPP`)
- `R-squared`, `MAPE`, and `wMAPE`
- prior-posterior ROI shift
- ROI consistency against business priors

Meridian also states that goodness-of-fit metrics are only guidance and that the primary goal of MMM is causal inference rather than prediction. This is a direct warning against using `MAPE` as the primary benchmark target.

Important Meridian ideas for this repository:

- use `wMAPE` rather than plain `MAPE` when scale heterogeneity matters
- add a **global posterior predictive adequacy** check such as Bayesian `PPP`
- inspect **baseline plausibility**, because bad baselines can masquerade as good fit
- distinguish whether the posterior learned from data or simply reproduced priors

Sources:
- [Meridian model health checks](https://developers.google.com/meridian/docs/post-modeling/health-checks)
- [Zhang et al. (2023/2024), "Media Mix Model Calibration With Bayesian Priors"](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/a09f404fdc3107fafb7a52cc5af6a80e4d0fda2b.pdf)

### 3. Robyn

Robyn operationalizes evaluation as a **multi-objective model-selection problem**. Its main objectives are:

- `NRMSE` as prediction error
- `DECOMP.RSSD` as business decomposition error
- `MAPE.LIFT` as calibration error when lift tests are available

Robyn also uses time-series validation and a Pareto frontier rather than a single objective.

Two implications are useful here:

- predictive fit should usually be normalized (`NRMSE`, not only raw `RMSE`)
- a practical MMM benchmark should include a metric for **decomposition quality**, not just outcome fit

At the same time, `DECOMP.RSSD` is a heuristic based on spend share versus effect share. It is useful operationally, but in synthetic benchmarks with known ground truth it is weaker than direct contribution-recovery metrics.

Sources:
- [Runge et al. (2025 revision), "Packaging Up Media Mix Modeling: An Introduction to Robyn's Open-Source Approach"](https://arxiv.org/abs/2403.14674)
- [Robyn key features documentation](https://facebookexperimental.github.io/Robyn/docs/features/)

### 4. PyMC-Marketing

PyMC-Marketing documentation and benchmark material make two points that are especially relevant for our benchmark design.

First, PyMC-Marketing treats predictive metrics as **posterior distributions**, not only point estimates. Its evaluation utilities summarize distributions of:

- Bayesian `R-squared`
- `RMSE`
- `NRMSE`
- `MAE`
- `NMAE`
- `MAPE`

Second, the project explicitly recommends:

- **time-slice cross-validation**
- **parameter stability** checks across folds
- **CRPS** for evaluating predictive distributions
- calibration with **lift tests**

Their PyMC-Marketing vs Meridian benchmark then separates:

- convergence and sampling
- goodness-of-fit
- contribution recovery

and reports contribution-recovery metrics such as:

- bias
- `SRMSE`
- `CRPS`

This separation is close to what we need: a model can improve in-sample fit while worsening contribution recovery.

Sources:
- [PyMC-Marketing evaluation module](https://www.pymc-marketing.io/en/0.14.1/_modules/pymc_marketing/mmm/evaluation.html)
- [PyMC-Marketing time-slice CV and parameter stability notebook](https://www.pymc-marketing.io/en/0.9.0/notebooks/mmm/mmm_time_slice_cross_validation.html)
- [PyMC-Marketing lift-test calibration case study](https://www.pymc-marketing.io/en/0.9.0/notebooks/mmm/mmm_roas.html)
- [PyMC Labs benchmark: PyMC-Marketing vs Meridian v1.2.1](https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-meridian-baseline-modeling-mmm)

## What Seems Appropriate For This Repository

### Keep

These metrics are still useful and should remain:

- `ELPD-LOO`: proper Bayesian predictive comparison across model classes
- interval coverage: a calibration check
- component-level `nRMSE`: directly relevant for mixture recovery
- HMC diagnostics: required before interpreting any posterior

### Demote

These should be treated as secondary rather than headline metrics:

- plain observed-outcome `MAPE`
- plain latent-mean `MAPE`
- raw `RMSE` logs without scale normalization

Reason:

- they are not proper scoring rules
- they can mis-rank models when the target is near zero
- they do not evaluate whether attribution or mixture structure was recovered

### Add

The literature suggests adding the following metrics, in roughly this priority order.

#### A. Proper probabilistic prediction metric on holdout data

Add at least one proper scoring rule:

- **CRPS** on `y_t`
- optionally holdout **log score** or retained `ELPD` if pointwise log likelihood is available for held-out observations

Why:

- unlike `MAPE`, `CRPS` rewards both sharpness and calibration
- PyMC-Marketing explicitly uses `CRPS` for time-slice evaluation

#### B. Normalized latent-function recovery

For synthetic data, the benchmark target is the latent response surface. Prefer:

- latent `nRMSE` or standardized `MAE`
- 90% and 95% latent coverage
- optionally latent `CRPS` if a full posterior over `mu_t` is stored

Why:

- `MAPE_mu` can be dominated by low-signal periods
- `nRMSE` and standardized `MAE` are more stable across DGPs and scales

#### C. Contribution-recovery metrics

When the DGP exposes true channel contributions, add:

- channel contribution bias
- contribution `SRMSE`
- contribution `CRPS`

This aligns with the PyMC-Marketing vs Meridian benchmark and is conceptually stronger than spend-share heuristics for synthetic data.

#### D. Mixture/component recovery as a headline metric

For mixture DGPs, promote the existing permutation-invariant component recovery from an auxiliary metric to a primary one:

- weighted curve `nRMSE`
- component count recovery (`effective K` error)
- component parameter recovery after permutation matching

This is the most direct way to test the paper's central claim.

#### E. Residual structure diagnostics

For observed-outcome prediction, add at least one residual-structure check:

- residual autocorrelation plot or summary statistic
- optionally Durbin-Watson on posterior mean residuals

Why:

- good fit metrics can still hide systematic temporal misspecification
- PyMC benchmark reporting uses residual structure to distinguish models with similar fit

## Recommended Headline Metric Bundle

For this repository's synthetic benchmark, a literature-aligned headline bundle would be:

1. **Inference health**
   - `R-hat`, ESS, divergences, BFMI, tree-depth hits
2. **Predictive density**
   - `ELPD-LOO`
   - holdout `CRPS`
3. **Latent response recovery**
   - latent `nRMSE` or standardized `MAE`
   - latent 90% and 95% coverage
4. **Mixture recovery**
   - weighted component-curve `nRMSE`
   - `effective K` recovery
5. **Attribution recovery**
   - contribution bias / `SRMSE` / `CRPS` if per-channel truth is available

Under this bundle, observed-outcome `MAPE` becomes a supporting metric rather than a decision metric.

## Practical Recommendation For The Current Test Suite

### Minimal change

If we want the smallest possible change to the current benchmark design:

- keep `ELPD-LOO`
- keep coverage
- keep current component recovery
- keep `MAPE` only as a secondary logged metric
- add latent `nRMSE`
- add holdout `CRPS`

### Better research-facing change

If we want the benchmark to support paper claims more directly:

- make latent recovery and component recovery primary for synthetic cases
- add per-channel contribution recovery for every synthetic DGP
- report observed `MAPE` only in a supplementary table

## Bottom Line

The literature does not support using `MAPE` alone, or even as the main score, for MMM quality. Meridian explicitly warns that fit metrics are only guidance. Robyn supplements prediction error with decomposition and calibration objectives. PyMC-Marketing separates predictive fit from contribution recovery and uses `CRPS`, time-slice validation, and parameter stability checks.

For this repository, the most appropriate interpretation is:

- `ELPD-LOO` is useful for predictive density
- coverage is useful for calibration
- `MAPE` is acceptable as a secondary descriptive metric
- the primary synthetic benchmark should emphasize **latent response recovery, contribution recovery, and permutation-invariant mixture recovery**

## References

- [Jin et al. (2017), Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/)
- [Chan and Perry (2017), Challenges and Opportunities in Media Mix Modeling](https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/)
- [Sun et al. (2017), Geo-level Bayesian Hierarchical Media Mix Modeling](https://research.google/pubs/geo-level-bayesian-hierarchical-media-mix-modeling/)
- [Zhang et al. (2023/2024), Media Mix Model Calibration With Bayesian Priors](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/a09f404fdc3107fafb7a52cc5af6a80e4d0fda2b.pdf)
- [Meridian model health checks](https://developers.google.com/meridian/docs/post-modeling/health-checks)
- [Runge et al. (2025 revision), Packaging Up Media Mix Modeling: An Introduction to Robyn's Open-Source Approach](https://arxiv.org/abs/2403.14674)
- [Robyn key features documentation](https://facebookexperimental.github.io/Robyn/docs/features/)
- [PyMC-Marketing evaluation module](https://www.pymc-marketing.io/en/0.14.1/_modules/pymc_marketing/mmm/evaluation.html)
- [PyMC-Marketing time-slice CV and parameter stability notebook](https://www.pymc-marketing.io/en/0.9.0/notebooks/mmm/mmm_time_slice_cross_validation.html)
- [PyMC-Marketing lift-test calibration case study](https://www.pymc-marketing.io/en/0.9.0/notebooks/mmm/mmm_roas.html)
- [PyMC Labs benchmark: PyMC-Marketing vs Meridian v1.2.1](https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-meridian-baseline-modeling-mmm)
