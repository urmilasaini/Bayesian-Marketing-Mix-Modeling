# Bayesian Hill Mixture Models for Marketing Mix Modeling

Code, data, and per-fit artifacts for the preprint:

> **When Can a Mixture-of-Hill MMM Recover Its Components? A Resolvability Diagnostic and a Synthetic-to-Real Transfer Study**
> Shohei Yoshida, Urmila Saini, Mizuki Oka
> (arXiv ID pending — manuscript source lives in the [`paper/`](https://github.com/urmilasaini/Bayesian-Marketing-Mix-Modeling-paper) submodule)

Standard Marketing Mix Models (MMMs) fit a single Hill saturation curve per channel. This project studies a Bayesian **predictive mixture** of K Hill curves — a mixture over the observation likelihood, not over latent consumer segments — implemented in NumPyro with an ordering reparameterisation, post-hoc relabeling, and label-invariant convergence diagnostics. The paper reports three experiments:

1. **Synthetic benchmark** (3 DGPs × 3 models × 5 seeds): when the data-generating process is itself a Hill mixture, the model recovers it.
2. **Component resolvability sweep** (9 profiles × 2 models × 5 seeds): an empirical transition near cosine distance ≈ 0.10, below which posteriors collapse to fewer effective components.
3. **Real-data benchmark** (3 Conjura organisations × 3 models × 3 seeds): Mixture K=3 improves estimated ELPD-LOO everywhere, but publication-ready convergence is organisation-dependent in a way the resolvability axis can diagnose.

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/urmilasaini/Bayesian-Marketing-Mix-Modeling.git
cd Bayesian-Marketing-Mix-Modeling
uv sync
```

## Reproducing the paper

Every fit records its random seed and sampler settings in a per-fit JSON summary under `paper/figures/{synthetic,real}/`.

```bash
# Fast unit tests (no MCMC)
uv run pytest tests/ -m "not slow"

# Synthetic benchmark (full: 5 seeds; also regenerates Figures 0-3 and 5)
HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK=1 uv run pytest tests/test_benchmark_synthetic.py -m benchmark_full

# Component resolvability sweep (Section 6)
uv run python scripts/run_component_resolvability_sweep.py

# Real-data benchmark (Section 7; opt-in, slow)
HILL_MMM_RUN_FULL_REAL_BENCHMARK=1 uv run pytest tests/test_benchmark_real.py -m benchmark_full

# Real-data paper figures and posterior separation overlay (Sections 7.2-7.4)
uv run python scripts/build_real_paper_figures.py
uv run python scripts/compute_posterior_separation.py
```

## Repository layout

| Path | Contents |
|------|----------|
| `src/hill_mixture_mmm/` | Models (NumPyro), transforms, synthetic DGPs, inference and label-invariant diagnostics, metrics, benchmark harness |
| `scripts/` | Benchmark drivers, resolvability sweep, figure builders |
| `tests/` | Pytest suite; the benchmark tests are the canonical experiment entry points |
| `paper/` | Git submodule with the LaTeX manuscript, figures, and per-fit JSON artifacts |
| `data/` | Conjura MMM dataset and data dictionary (see attribution below) |
| `results/` | Benchmark summary CSVs |
| `docs/` | Design notes; `docs/archive/` keeps historical experiment logs |

See [`AGENTS.md`](AGENTS.md) for a more detailed code map and the paper-section-to-module mapping.

## Dataset

The real-data experiments use the **Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce Brands**, redistributed here under CC BY 4.0:

> Anderson, A. (2024). Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce Brands. figshare. https://doi.org/10.6084/m9.figshare.25314841.v3 — Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

See [`data/README.md`](data/README.md) for the field dictionary and provenance notes.

## Citation

Citation entry will be added once the preprint is assigned an arXiv ID.

## License

Code is released under the [MIT License](LICENSE). The dataset in `data/` is CC BY 4.0 (see above).
