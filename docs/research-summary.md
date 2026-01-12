# MMM Response Curve Research Summary

**Date:** 2026-01-15  
**Focus:** Improving spend-response functional forms for descriptive accuracy (not causal inference)  
**Next Step:** Implement mixture models for heterogeneous consumer response

---

## 1. Critical Assessment of Current MMM Tools

### 1.1 Meta's Robyn: Fundamental Flaws

Lennox (2025) published a detailed technical critique identifying eight major issues:

**Overparameterization:** Typical applications have ~45 parameters for ~100-150 observations (ratio approaching 1:2). This is a textbook overfitting scenario. Ridge regularization relies on asymptotic properties that do not hold in small samples.

**Cross-validation invalid:** Time-series data violates the independence assumption required for traditional cross-validation, leading to over-optimistic performance metrics.

**No convergence guarantees:** Evolutionary algorithms (Nevergrad) use fixed iterations rather than proper stopping criteria. From the critique:

> *"Fixed iteration limits increase the likelihood of settling on suboptimal solutions that are neither globally optimal nor robust."*

**Model selection by RMSE:** Measures predictive accuracy, not explanatory power. Does not penalize complexity.

**Coefficient instability:** ROAS estimates "fluctuate dramatically" depending on hyperparameters and data partitions.

**Verdict from Lennox:**
> *"Robyn is not a reliable tool for causal inference or robust decision-making for anything but the most simple and low-dimensional problems. Practitioners should approach Robyn with extreme caution."*

### 1.2 Google's Meridian

Google's own documentation acknowledges limitations:

> *"The primary goal of marketing mix modeling (MMM) is the accurate estimation of causal marketing effects. However, directly validating the quality of causal inference is difficult and requires well-designed experiments."*

Recast's analysis notes:
> *"Running an MMM is trivially easy, and the hard part is knowing if your answers are right or wrong."*

### 1.3 PyMC-Marketing vs Meridian Benchmark

PyMC Labs (September 2025) published a rigorous head-to-head benchmark across four dataset scales (startup to enterprise). Key findings:

| Metric | PyMC-Marketing | Meridian |
|--------|----------------|----------|
| Sampling speed | 2-20x faster | Baseline |
| Channel contribution SRMSE | 0.41 ± 0.22 | 0.70 ± 0.39 |
| R² (startup data) | 0.87 ± 0.02 | 0.73 ± 0.02 |
| MAPE (startup data) | 7.2% ± 0.6% | 10.4% ± 0.5% |
| Durbin-Watson | 1.96 ± 0.17 | 1.13 ± 0.10 |
| Enterprise scale | Successful | Failed to converge |

**Critical finding on Durbin-Watson:** Meridian's low scores (ideal ~2.0) indicate systematic autocorrelation in residuals:

> *"This suggests the model misses important patterns in the data. This problem persists across dataset sizes."*

**On channel contribution recovery:**
> *"PyMC-Marketing offers more precise estimates with narrower credible intervals for decomposing sales into channel contributions... Meridian consistently underestimates the contribution for both channels."*

---

## 2. The Identification Problem

### 2.1 Nonlinear Effects May Be Artifacts

Dew, Padilla & Shchetkina (2024). "Your MMM is Broken: Identification of Nonlinear and Time-varying Effects in Marketing Mix Models." arXiv:2408.07678

> *"We show that nonlinear and time-varying effects are often not identifiable from standard marketing mix data: while certain data patterns may be suggestive of nonlinear effects, such patterns may also emerge under simpler models that incorporate dynamics in marketing effectiveness."*

**Key insight:** The saturation curves (Hill functions) that all modern MMM tools use may be fitting artifacts rather than true phenomena. Autocorrelated marketing variables (common in practice) exacerbate this conflation.

> *"This lack of identification is problematic because nonlinearities and dynamics suggest fundamentally different optimal marketing allocations."*

### 2.2 Multicollinearity Is Inescapable

From Chan & Perry (2017), Google Research:

> *"Another consequence is that the estimated relationship can change radically due to small changes in the data or the addition or subtraction of seemingly unrelated variables in the model."*

Sources of correlation that cannot be eliminated:
- Seasonal coordination (all spending increases during Q4)
- Campaign orchestration (TV drives search; social accompanies display)
- Competitive response (spend increases when competitor does)
- Budget cycles (annual planning creates synchronized changes)

### 2.3 Limited Data Problem

From Jin et al. (2017), Google Research:

> *"Large scale simulation studies show that the model can be estimated well on a large data set, but it may produce biased estimates for the typical sample size of a couple of years of weekly national-level data."*

Typical MMM: 156 observations (3 years weekly), 20+ parameters needed. Rule of thumb requires 10-20 observations per parameter = 200-400 minimum.

---

## 3. Alternative Functional Forms for Response Curves

### 3.1 Parametric Saturation Functions

**Hill function (current standard):**
```
y = x^a / (x^a + g^a)
```
- `a` (slope): controls S-curve steepness
- `g` (half-saturation): determines 50% effect point
- Limitation: 2 parameters, symmetric, global shape only

**PyMC-Marketing implementations:**
- `HillSaturation`
- `MichaelisMentenSaturation`: `αx / (λ + x)`
- `LogisticSaturation`
- `TanhSaturation`

**DeepCausalMMM (Tirumala, 2025):** Learns Hill parameters from data with constraint `a ≥ 2.0` for proper saturation shape.

### 3.2 Adstock Alternatives

**Geometric decay (standard):** Single parameter θ, exponential decay, fixed rate.

**Weibull (Robyn default):** Two parameters (shape, scale), allows variable decay rate over time.

From Robyn documentation:
> *"Geometric assumes exponential decay, while Weibull methods are more flexible and can model both growth and decay of advertising effect."*

**Weibull PDF variant:** Can model delayed peak effects (effect builds before decaying).

**Negative binomial:** Recast's approach, more flexible than Weibull.

**Learned via neural networks:** NNN (Google, 2025) uses Transformer attention to learn temporal patterns without fixed functional forms.

### 3.3 Shape-Constrained Splines

Flexible nonparametric approach while maintaining interpretable constraints (monotonicity, concavity).

**Spline basis types:**
- **I-splines:** Monotone basis for non-decreasing/increasing fits
- **M-splines:** Non-negative basis (building block for I-splines)
- **C-splines:** Convex/concave basis for diminishing returns
- **B-splines + constraints:** General with linear constraints

**Software:**
- R `scam` package (Shape Constrained Additive Models, Pya & Wood)
- R `cgam` package
- Python: implementable in PyMC with constrained coefficients

From Wikipedia on I-splines:
> *"I-splines can be used as basis splines for regression analysis and data transformation when monotonicity is desired (constraining the regression coefficients to be non-negative for a non-decreasing fit)."*

### 3.4 Time-Varying Coefficient Models

**Uber BTVC (Ng, Wang & Dai, 2021).** arXiv:2106.03322

Addresses that response coefficients change over time (creative fatigue, strategy shifts).

**Approach:** Coefficients as weighted sum of local latent variables:
```
β_{t,p} = Σ_j w_j(t) · b_{j,p}
```

Where `w_j(t)` is kernel-based weighting by time distance.

**On limitations of alternatives:**
> *"Although the Kalman filter provides analytical solutions, it has limited room for further customization such as applying restrictions on coefficient signs (e.g., positive coefficient sign for marketing spend) or a t-distributed noise process that is more outlier robust."*

Uses Stochastic Variational Inference (SVI) for posterior estimation.

### 3.5 Neural Network Approaches

**NNN: Next-Generation Neural Networks for Marketing Mix Modeling**  
Mulc et al. (April 2025). arXiv:2504.06212

Key innovations:
- **Embeddings** capture both spend quantity AND creative quality
- **Transformer attention** replaces fixed adstock functions
- **L1 regularization** enables training with limited data
- **Intermediate signals** (Search) model indirect effects

> *"Instead of relying solely on scalar inputs (e.g., spend), NNN utilizes high-dimensional embeddings to represent both marketing activities and crucial organic signals. These embeddings capture not only quantitative volume but also qualitative aspects like ad creative content."*

**DeepCausalMMM**  
Tirumala (October 2025). arXiv:2510.13087

- GRU network learns adstock/lag automatically
- DAG learning for channel interdependencies
- Hill saturation with learned parameters
- Multi-region support with shared/region-specific parameters

Performance on real data: 91.8% holdout R², 3.0% train-test gap.

---

## 4. Heterogeneous Consumer Response

### 4.1 The Problem

A single aggregate response curve assumes homogeneous consumer behavior. Reality:
- Different segments respond differently
- Heavy vs light buyers
- Brand loyalists vs switchers
- Geographic variation
- Demographic variation

Aggregate curves are weighted averages of segment-specific responses, potentially masking true patterns.

### 4.2 Latent Class / Mixture Model Approach

**Formulation:**
```
P(response | spend) = Σ_k π_k · f_k(spend | θ_k)
```

- K latent segments with different response functions f_k
- Segment membership probabilities π_k estimated from data
- Each segment has own parameters θ_k

From Wedel & Kamakura, and related literature on latent segmentation:
> *"Response segments are determined probabilistically using a latent mixture model. The approach simultaneously calibrates sales response on two dimensions: across segments and the three purchase behaviors."*

**Advantages:**
1. Maintains interpretability (each segment has a response curve)
2. Captures heterogeneity without observed segment labels
3. More realistic than single aggregate curve
4. Natural extension of Hill-based models

**Implementation approaches:**
- EM algorithm (frequentist)
- Bayesian inference with PyMC (full uncertainty quantification)
- Model selection via BIC/WAIC for number of segments K
- Identifiability constraints (ordering segments by parameter values)

### 4.3 Hierarchical Models (Alternative)

- Geographic hierarchy (national → regional → local)
- Product hierarchy (brand → SKU)
- Partial pooling for regularization
- Random coefficient models where each unit has own coefficients drawn from common distribution

---

## 5. Key References

### Foundational Critiques

- **Chan, D. & Perry, M. (2017).** "Challenges and Opportunities in Media Mix Modeling." Google Research White Paper.

- **Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017).** "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Research.

- **Lennox, P. (2025).** "Eight Major Issues and Technical Limitations of Meta's Robyn." LinkedIn.

- **Data-Dive (2023).** "A Critical Review of Marketing Mix Modeling — From Hype to Reality."

### Identification Problem

- **Dew, R., Padilla, N., & Shchetkina, A. (2024).** "Your MMM is Broken: Identification of Nonlinear and Time-varying Effects in Marketing Mix Models." arXiv:2408.07678

### Tool Benchmarks

- **PyMC Labs (2025).** "PyMC-Marketing vs. Meridian: A Quantitative Comparison of Open Source MMM Libraries."

### Alternative Approaches

- **Ng, E., Wang, Z., & Dai, A. (2021).** "Bayesian Time Varying Coefficient Model with Applications to Marketing Mix Modeling." Uber. arXiv:2106.03322

- **Mulc, T. et al. (2025).** "NNN: Next-Generation Neural Networks for Marketing Mix Modeling." Google. arXiv:2504.06212

- **Tirumala, A.P. (2025).** "DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling." arXiv:2510.13087

- **Pya, N. & Wood, S.** R `scam` package for Shape Constrained Additive Models.

### Heterogeneity / Mixture Models

- **Wedel, M. & Kamakura, W.** "Market Segmentation: Conceptual and Methodological Foundations."

- **Allenby, G.M. & Rossi, P.E. (1998).** "Marketing Models of Consumer Heterogeneity." Journal of Econometrics.

---

## 6. Implementation Direction

**Chosen approach:** Bayesian mixture of Hill saturation functions in PyMC.

**Model structure:**
```python
# K mixture components
π ~ Dirichlet(α)  # segment probabilities

for k in range(K):
    a_k ~ Prior()  # Hill slope for segment k
    g_k ~ Prior()  # Half-saturation for segment k
    
    # Hill response for segment k
    f_k(x) = x^a_k / (x^a_k + g_k^a_k)

# Mixture likelihood
y ~ Σ_k π_k · Normal(f_k(x), σ)
```

**Implementation considerations:**
- Label switching: order segments by g_k values
- Model selection: WAIC/LOO-CV for choosing K
- Priors: informed by single-component fits
- Comparison: benchmark against single-curve baseline

**Existing code:** `hill_mixture_mmm.py` contains initial exploration.
