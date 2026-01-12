#!/usr/bin/env python
"""Run synthetic and real MMM benchmarks."""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import numpyro
import pandas as pd

from hill_mixture_mmm.baseline import standardized_time_index
from hill_mixture_mmm.benchmark import evaluate_diagnostic_summary
from hill_mixture_mmm.data import DGP_CONFIGS, DGPConfig, compute_prior_config, generate_data
from hill_mixture_mmm.inference import (
    compute_comprehensive_mixture_diagnostics,
    compute_convergence_diagnostics,
    compute_hmc_diagnostics,
    compute_loo,
    compute_predictions,
    compute_waic,
    run_inference,
)
from hill_mixture_mmm.metrics import (
    compute_delta_loo,
    compute_effective_k,
    compute_latent_recovery,
    compute_parameter_recovery,
    compute_predictive_metrics,
)
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam, model_single_hill


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    name: str
    fn: Callable
    kwargs: dict


MODEL_SPECS = [
    ModelSpec("single_hill", model_single_hill, {}),
    ModelSpec("mixture_k2", model_hill_mixture_hierarchical_reparam, {"K": 2}),
    ModelSpec("mixture_k3", model_hill_mixture_hierarchical_reparam, {"K": 3}),
]


def _reference_latent_mean(meta: dict) -> np.ndarray | None:
    """Return the synthetic latent target aligned with posterior mean predictions."""
    if "mu_expected_true" in meta:
        return np.asarray(meta["mu_expected_true"])
    if "mu_true" in meta:
        return np.asarray(meta["mu_true"])
    return None


def _prepare_experiment_data(dgp_config: DGPConfig, train_ratio: float) -> dict:
    """Generate data, split train/test, and compute prior config."""
    x, y, meta = generate_data(dgp_config)
    T = len(y)
    T_train = int(T * train_ratio)
    t_std_full = standardized_time_index(T)
    x_train, y_train = x[:T_train], y[:T_train]
    x_test, y_test = x[T_train:], y[T_train:]
    prior_config = compute_prior_config(x_train, y_train)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "prior_config": prior_config,
        "meta": meta,
        "T": T,
        "T_train": T_train,
        "t_std_train": t_std_full[:T_train],
    }


def _fit_once(
    dgp_config: DGPConfig,
    model_spec: ModelSpec,
    x_train,
    y_train,
    t_std_train,
    prior_config: dict,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
) -> dict:
    """Run one inference pass and compute convergence diagnostics."""
    is_mixture_model = "K" in model_spec.kwargs
    inference_seed = dgp_config.seed
    used_target_accept = 0.95 if model_spec.name == "mixture_k2" else 0.90
    used_max_tree_depth = 10

    mcmc = run_inference(
        model_spec.fn,
        x_train,
        y_train,
        seed=inference_seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        prior_config=prior_config,
        t_std=t_std_train,
        target_accept_prob=used_target_accept,
        max_tree_depth=used_max_tree_depth,
        **model_spec.kwargs,
    )
    hmc_diagnostics = compute_hmc_diagnostics(mcmc, max_tree_depth=used_max_tree_depth)

    if is_mixture_model:
        mixture_diagnostics = compute_comprehensive_mixture_diagnostics(mcmc, x_train, y_train)
        convergence = mixture_diagnostics["standard"]
        label_invariant = mixture_diagnostics["label_invariant"]
        relabeled = mixture_diagnostics["relabeled"]
    else:
        convergence = compute_convergence_diagnostics(mcmc)
        label_invariant = None
        relabeled = None

    diagnostic_status = evaluate_diagnostic_summary(
        convergence=convergence,
        hmc_diagnostics=hmc_diagnostics,
        label_invariant=label_invariant,
        relabeled=relabeled,
        num_chains_used=num_chains,
    )

    return {
        "mcmc": mcmc,
        "convergence": convergence,
        "hmc_diagnostics": hmc_diagnostics,
        "is_mixture_model": is_mixture_model,
        "label_invariant": label_invariant,
        "relabeled": relabeled,
        "diagnostic_status": diagnostic_status,
        "inference_seed": inference_seed,
        "used_warmup": num_warmup,
        "used_samples": num_samples,
        "used_target_accept": used_target_accept,
        "used_max_tree_depth": used_max_tree_depth,
    }


def _compute_train_test_metrics(
    mcmc,
    model_spec: ModelSpec,
    prior_config: dict,
    x_train,
    y_train,
    x_test,
    y_test,
) -> tuple[dict, dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute posterior predictive metrics on train and test splits."""
    total_time = len(x_train) + len(x_test)
    pred_train = compute_predictions(
        mcmc,
        model_spec.fn,
        x_train,
        prior_config=prior_config,
        total_time=total_time,
        **model_spec.kwargs,
    )
    pred_test = compute_predictions(
        mcmc,
        model_spec.fn,
        x_test,
        prior_config=prior_config,
        history_x=x_train,
        total_time=total_time,
        **model_spec.kwargs,
    )

    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

    return train_metrics, test_metrics, pred_train, pred_test


def run_single_experiment(
    dgp_config: DGPConfig,
    model_spec: ModelSpec,
    train_ratio: float = 0.75,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
) -> dict:
    """Run one DGP-model experiment."""
    prepared = _prepare_experiment_data(dgp_config, train_ratio)
    fit = _fit_once(
        dgp_config=dgp_config,
        model_spec=model_spec,
        x_train=prepared["x_train"],
        y_train=prepared["y_train"],
        t_std_train=prepared["t_std_train"],
        prior_config=prepared["prior_config"],
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc = fit["mcmc"]
    convergence = fit["convergence"]
    diagnostic_status = fit["diagnostic_status"]
    loo = compute_loo(mcmc)
    waic = compute_waic(mcmc)
    effective_k = compute_effective_k(mcmc)
    param_recovery = compute_parameter_recovery(mcmc, prepared["meta"])
    train_metrics, test_metrics, pred_train, pred_test = _compute_train_test_metrics(
        mcmc=mcmc,
        model_spec=model_spec,
        prior_config=prepared["prior_config"],
        x_train=prepared["x_train"],
        y_train=prepared["y_train"],
        x_test=prepared["x_test"],
        y_test=prepared["y_test"],
    )
    mu_train_samples = pred_train.get("mu", pred_train.get("mu_expected"))
    mu_test_samples = pred_test.get("mu", pred_test.get("mu_expected"))
    latent_truth = _reference_latent_mean(prepared["meta"])
    train_latent = compute_latent_recovery(
        latent_truth[: prepared["T_train"]],
        mu_train_samples,
    )
    test_latent = compute_latent_recovery(
        latent_truth[prepared["T_train"] :],
        mu_test_samples,
    )

    return {
        "dgp": dgp_config.dgp_type,
        "K_true": prepared["meta"]["K_true"],
        "model": model_spec.name,
        "seed": dgp_config.seed,
        "T": prepared["T"],
        "T_train": prepared["T_train"],
        "max_rhat": convergence["max_rhat"],
        "min_ess_bulk": convergence["min_ess_bulk"],
        "min_ess_tail": convergence["min_ess_tail"],
        "converged_standard": convergence["converged"],
        "strict_converged": diagnostic_status["strict_converged"],
        "converged": diagnostic_status["strict_converged"],
        "publication_status": diagnostic_status["publication_status"],
        "sampler_status": diagnostic_status["sampler_status"],
        "mixing_status": diagnostic_status["mixing_status"],
        "interpretation_status": diagnostic_status["interpretation_status"],
        "benchmark_pass": diagnostic_status["benchmark_pass"],
        "num_divergences": fit["hmc_diagnostics"]["num_divergences"],
        "min_bfmi": fit["hmc_diagnostics"]["min_bfmi"],
        "tree_depth_hits": fit["hmc_diagnostics"]["tree_depth_hits"],
        "rhat_log_lik": (
            fit["label_invariant"]["rhat_log_lik"] if fit["label_invariant"] is not None else None
        ),
        "label_invariant_max_rhat": (
            fit["label_invariant"]["max_rhat"] if fit["label_invariant"] is not None else None
        ),
        "relabeled_max_rhat": (
            fit["relabeled"]["max_rhat"] if fit["relabeled"] is not None else None
        ),
        "inference_seed": fit["inference_seed"],
        "num_warmup_used": fit["used_warmup"],
        "num_samples_used": fit["used_samples"],
        "target_accept_prob_used": fit["used_target_accept"],
        "max_tree_depth_used": fit["used_max_tree_depth"],
        "elpd_loo": loo.get("elpd_loo"),
        "loo_se": loo.get("se"),
        "p_loo": loo.get("p_loo"),
        "elpd_waic": waic.get("elpd_waic"),
        "waic_se": waic.get("se"),
        "train_mape": train_metrics["mape"],
        "test_mape": test_metrics["mape"],
        "train_coverage_90": train_metrics["coverage_90"],
        "test_coverage_90": test_metrics["coverage_90"],
        "train_mu_mape": train_latent["mape"],
        "test_mu_mape": test_latent["mape"],
        "train_mu_coverage_90": train_latent["coverage_90"],
        "test_mu_coverage_90": test_latent["coverage_90"],
        "effective_k_mean": effective_k["effective_k_mean"],
        "effective_k_std": effective_k["effective_k_std"],
        "alpha_in_ci": param_recovery.get("alpha", {}).get("in_ci"),
        "sigma_in_ci": param_recovery.get("sigma", {}).get("in_ci"),
        "alpha_true": prepared["meta"]["alpha_true"],
        "alpha_est": param_recovery.get("alpha", {}).get("mean"),
    }


def run_benchmark_suite(
    dgp_names: list[str] | None = None,
    model_names: list[str] | None = None,
    seeds: list[int] | None = None,
    train_ratio: float = 0.75,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the benchmark suite."""
    numpyro.set_host_device_count(num_chains)

    if dgp_names is None:
        dgp_names = list(DGP_CONFIGS.keys())
    if model_names is None:
        model_names = [m.name for m in MODEL_SPECS]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    models = [m for m in MODEL_SPECS if m.name in model_names]

    results = []
    total = len(dgp_names) * len(models) * len(seeds)
    current = 0

    for dgp_name in dgp_names:
        base_config = DGP_CONFIGS[dgp_name]

        for seed in seeds:
            config = DGPConfig(
                dgp_type=base_config.dgp_type,
                T=base_config.T,
                sigma=base_config.sigma,
                alpha=base_config.alpha,
                intercept=base_config.intercept,
                slope=base_config.slope,
                seed=seed,
            )

            for model_spec in models:
                current += 1
                if verbose:
                    print(
                        f"[{current}/{total}] DGP={dgp_name}, Model={model_spec.name}, Seed={seed}"
                    )

                result = run_single_experiment(
                    config,
                    model_spec,
                    train_ratio=train_ratio,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                results.append(result)

    df = pd.DataFrame(results)

    df = _add_delta_loo(df)

    return df


def _add_delta_loo(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta LOO columns relative to single_hill baseline."""
    df = df.copy()
    df["delta_loo"] = None
    df["delta_loo_significant"] = None

    for (dgp, seed), group in df.groupby(["dgp", "seed"]):  # type: ignore[misc]
        baseline_row = group[group["model"] == "single_hill"]
        if len(baseline_row) == 0:
            continue

        baseline_loo = {
            "elpd_loo": baseline_row["elpd_loo"].iloc[0],  # type: ignore[union-attr]
            "se": baseline_row["loo_se"].iloc[0],  # type: ignore[union-attr]
        }

        for idx in group.index:
            if df.loc[idx, "model"] == "single_hill":
                df.loc[idx, "delta_loo"] = 0.0
                df.loc[idx, "delta_loo_significant"] = False
            else:
                model_loo = {
                    "elpd_loo": df.loc[idx, "elpd_loo"],
                    "se": df.loc[idx, "loo_se"],
                }
                delta = compute_delta_loo(model_loo, baseline_loo)
                df.loc[idx, "delta_loo"] = delta["delta_loo"]
                df.loc[idx, "delta_loo_significant"] = delta["significant"]

    return df


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark results across seeds."""
    metrics = [
        "benchmark_pass",
        "strict_converged",
        "converged",
        "elpd_loo",
        "test_mape",
        "train_mape",
        "test_coverage_90",
        "test_mu_mape",
        "train_mu_mape",
        "test_mu_coverage_90",
        "effective_k_mean",
        "alpha_in_ci",
        "sigma_in_ci",
        "delta_loo_significant",
        "delta_loo",
    ]

    summary = df.groupby(["dgp", "K_true", "model"])[metrics].agg(["mean", "std"]).round(2)

    return summary  # type: ignore[return-value]


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for dgp in df["dgp"].unique():
        K_true = df[df["dgp"] == dgp]["K_true"].iloc[0]  # type: ignore[union-attr]
        print(f"\nDGP: {dgp} (K_true={K_true})")
        print("-" * 60)

        dgp_data = df[df["dgp"] == dgp]

        print(
            f"{'Model':<15} {'LOO':>10} {'Test MAPE':>12} {'Cov 90%':>10} {'Eff K':>8} {'\u0394LOO':>10}"
        )
        print("-" * 60)

        for model in dgp_data["model"].unique():  # type: ignore[union-attr]
            model_data = dgp_data[dgp_data["model"] == model]
            loo = model_data["elpd_loo"].mean()
            test_mape = model_data["test_mape"].mean()
            cov = model_data["test_coverage_90"].mean()
            eff_k = model_data["effective_k_mean"].mean()
            delta = model_data["delta_loo"].mean()

            delta_str = f"{delta:+.1f}" if not pd.isna(delta) else "N/A"  # type: ignore[arg-type]
            print(
                f"{model:<15} {loo:>10.1f} {test_mape:>12.3f} "
                f"{cov:>10.1%} {eff_k:>8.2f} {delta_str:>10}"
            )


def export_results_csv(
    df: pd.DataFrame, path: str | Path, include_summary: bool = True
) -> None:
    """Export benchmark results to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_path.with_suffix(".csv")
    df.to_csv(raw_path, index=False)
    print(f"Raw results exported to {raw_path}")

    if include_summary:
        summary = summarize_benchmark(df)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
        summary.to_csv(summary_path)
        print(f"Summary exported to {summary_path}")


def export_results_json(
    df: pd.DataFrame, path: str | Path, include_summary: bool = True
) -> None:
    """Export benchmark results to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_path.with_suffix(".json")
    df.to_json(raw_path, orient="records", indent=2)
    print(f"Raw results exported to {raw_path}")

    if include_summary:
        summary = summarize_benchmark(df)
        summary_reset = summary.reset_index()
        summary_reset.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in summary_reset.columns
        ]
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        summary_reset.to_json(summary_path, orient="records", indent=2)
        print(f"Summary exported to {summary_path}")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    synthetic_dgps: list[str]
    synthetic_models: list[str]
    synthetic_seeds: list[int]

    real_n_orgs: int
    real_models: list[str]
    real_seeds: list[int]

    num_warmup: int
    num_samples: int
    num_chains: int

    train_ratio: float

    output_dir: str


def get_default_config() -> BenchmarkConfig:
    """Get default full experiment configuration."""
    return BenchmarkConfig(
        synthetic_dgps=["single", "mixture_k2", "mixture_k3"],
        synthetic_models=["single_hill", "mixture_k2", "mixture_k3"],
        synthetic_seeds=[0, 1, 2, 3, 4],
        real_n_orgs=10,
        real_models=["single_hill", "mixture_k2", "mixture_k3"],
        real_seeds=[0, 1, 2],
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        train_ratio=0.75,
        output_dir="results/benchmark",
    )


def get_quick_config() -> BenchmarkConfig:
    """Get quick test configuration."""
    return BenchmarkConfig(
        synthetic_dgps=["single", "mixture_k2"],
        synthetic_models=["single_hill", "mixture_k2"],
        synthetic_seeds=[0],
        real_n_orgs=1,
        real_models=["single_hill", "mixture_k2"],
        real_seeds=[0],
        num_warmup=200,
        num_samples=200,
        num_chains=2,
        train_ratio=0.75,
        output_dir="results/benchmark_quick",
    )


def run_synthetic_experiments(config: BenchmarkConfig) -> pd.DataFrame:
    """Run synthetic data experiments."""
    print("=" * 60)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("=" * 60)
    n_exp = len(config.synthetic_dgps) * len(config.synthetic_models) * len(config.synthetic_seeds)
    print(f"DGPs: {config.synthetic_dgps}")
    print(f"Models: {config.synthetic_models}")
    print(f"Seeds: {config.synthetic_seeds}")
    print(f"Total: {n_exp} experiments")
    print()

    start_time = time.time()

    results = run_benchmark_suite(
        dgp_names=config.synthetic_dgps,
        model_names=config.synthetic_models,
        seeds=config.synthetic_seeds,
        train_ratio=config.train_ratio,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\nSynthetic experiments completed in {elapsed / 60:.1f} minutes")

    return results


def run_real_data_experiments(config: BenchmarkConfig) -> pd.DataFrame:
    """Run real data experiments."""
    from hill_mixture_mmm.data_loader import load_real_data, select_representative_timeseries

    print("=" * 60)
    print("REAL DATA EXPERIMENTS")
    print("=" * 60)
    n_exp = config.real_n_orgs * len(config.real_models) * len(config.real_seeds)
    print(f"Organizations: {config.real_n_orgs}")
    print(f"Models: {config.real_models}")
    print(f"Seeds: {config.real_seeds}")
    print(f"Total: {n_exp} experiments")
    print()

    csv_path = Path("data/conjura_mmm_data.csv")
    if not csv_path.exists():
        print(f"WARNING: Real data not found at {csv_path}")
        print("Skipping real data experiments.")
        return pd.DataFrame()

    df_real = load_real_data(str(csv_path))

    selected_org_ids = select_representative_timeseries(
        str(csv_path),
        n=config.real_n_orgs,
        selection_criteria="most_data",
        min_length=200,
        min_channels=1,
        seed=42,
    )

    model_lookup = {m.name: m for m in MODEL_SPECS}

    results = []
    total = n_exp
    current = 0
    start_time = time.time()

    for org_id in selected_org_ids:
        org_df = df_real[df_real["organization_id"] == org_id]
        if len(org_df) == 0:
            print(f"WARNING: No rows found for organization_id={org_id}, skipping")
            continue

        x = np.asarray(org_df["spend"].values)
        y = np.asarray(org_df["revenue"].values)
        T = len(y)
        T_train = int(T * config.train_ratio)
        t_std_full = standardized_time_index(T)

        x_train, y_train = x[:T_train], y[:T_train]
        x_test, y_test = x[T_train:], y[T_train:]

        prior_config = compute_prior_config(x_train, y_train)

        for model_name in config.real_models:
            model_spec = model_lookup.get(model_name)
            if model_spec is None:
                print(f"WARNING: Model {model_name} not found, skipping")
                continue

            for seed in config.real_seeds:
                current += 1
                print(f"[{current}/{total}] Org={org_id}, Model={model_name}, Seed={seed}")

                exp_start = time.time()

                try:
                    real_warmup = 2000
                    mcmc = run_inference(
                        model_spec.fn,
                        x_train,
                        y_train,
                        seed=seed,
                        num_warmup=real_warmup,
                        num_samples=config.num_samples,
                        num_chains=config.num_chains,
                        prior_config=prior_config,
                        t_std=t_std_full[:T_train],
                        **model_spec.kwargs,
                    )

                    exp_time = time.time() - exp_start

                    samples = mcmc.get_samples()
                    is_mixture = "k" in samples and len(samples["k"].shape) > 1
                    if is_mixture:
                        mixture_diag = compute_comprehensive_mixture_diagnostics(mcmc, x_train, y_train)
                        convergence = mixture_diag["standard"]
                        label_invariant = mixture_diag["label_invariant"]
                        relabeled = mixture_diag["relabeled"]
                        label_switching = mixture_diag["label_switching"]
                    else:
                        mixture_diag = None
                        convergence = compute_convergence_diagnostics(mcmc)
                        label_invariant = None
                        relabeled = None
                        label_switching = None

                    hmc_diagnostics = compute_hmc_diagnostics(mcmc)
                    diagnostic_status = evaluate_diagnostic_summary(
                        convergence=convergence,
                        hmc_diagnostics=hmc_diagnostics,
                        label_invariant=label_invariant,
                        relabeled=relabeled,
                        num_chains_used=config.num_chains,
                    )
                    loo = compute_loo(mcmc)
                    waic = compute_waic(mcmc)

                    pred_train = compute_predictions(
                        mcmc,
                        model_spec.fn,
                        x_train,
                        prior_config=prior_config,
                        total_time=T,
                        **model_spec.kwargs,
                    )
                    pred_test = compute_predictions(
                        mcmc,
                        model_spec.fn,
                        x_test,
                        prior_config=prior_config,
                        history_x=x_train,
                        total_time=T,
                        **model_spec.kwargs,
                    )

                    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
                    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

                    result = {
                        "org_id": str(org_id),
                        "model": model_name,
                        "seed": seed,
                        "T": T,
                        "T_train": T_train,
                        "T_test": T - T_train,
                        "max_rhat": convergence["max_rhat"],
                        "min_ess_bulk": convergence["min_ess_bulk"],
                        "min_ess_tail": convergence["min_ess_tail"],
                        "converged_standard": convergence["converged"],
                        "strict_converged": diagnostic_status["strict_converged"],
                        "converged": diagnostic_status["strict_converged"],
                        "publication_status": diagnostic_status["publication_status"],
                        "sampler_status": diagnostic_status["sampler_status"],
                        "mixing_status": diagnostic_status["mixing_status"],
                        "interpretation_status": diagnostic_status["interpretation_status"],
                        "benchmark_pass": diagnostic_status["benchmark_pass"],
                        "num_divergences": hmc_diagnostics["num_divergences"],
                        "min_bfmi": hmc_diagnostics["min_bfmi"],
                        "tree_depth_hits": hmc_diagnostics["tree_depth_hits"],
                        "elpd_loo": loo.get("elpd_loo"),
                        "loo_se": loo.get("se"),
                        "p_loo": loo.get("p_loo"),
                        "pareto_k_bad": loo.get("pareto_k_bad", 0),
                        "pareto_k_very_bad": loo.get("pareto_k_very_bad", 0),
                        "elpd_waic": waic.get("elpd_waic"),
                        "waic_se": waic.get("se"),
                        "p_waic": waic.get("p_waic"),
                        "rhat_log_lik": (
                            label_invariant["rhat_log_lik"] if label_invariant is not None else None
                        ),
                        "label_invariant_max_rhat": (
                            label_invariant["max_rhat"] if label_invariant is not None else None
                        ),
                        "relabeled_max_rhat": (
                            relabeled["max_rhat"] if relabeled is not None else None
                        ),
                        "switching_rate": (
                            label_switching["switching_rate"] if label_switching is not None else None
                        ),
                        "train_mape": train_metrics["mape"],
                        "test_mape": test_metrics["mape"],
                        "train_coverage_90": train_metrics["coverage_90"],
                        "test_coverage_90": test_metrics["coverage_90"],
                        "time_seconds": exp_time,
                        "status": "success",
                    }

                except Exception as e:
                    print(f"  ERROR: {e}")
                    result = {
                        "org_id": str(org_id),
                        "model": model_name,
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                    }

                results.append(result)

    elapsed = time.time() - start_time
    print(f"\nReal data experiments completed in {elapsed / 60:.1f} minutes")

    return pd.DataFrame(results)


def save_results(
    synthetic_results: pd.DataFrame | None,
    real_results: pd.DataFrame | None,
    config: BenchmarkConfig,
) -> None:
    """Save experiment results to CSV and JSON."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")

    if synthetic_results is not None and len(synthetic_results) > 0:
        csv_path = output_dir / f"synthetic_{timestamp}.csv"
        synthetic_results.to_csv(csv_path, index=False)
        print(f"Synthetic results saved to {csv_path}")

        json_path = output_dir / f"synthetic_{timestamp}.json"
        synthetic_results.to_json(json_path, orient="records", indent=2)

        summary = summarize_benchmark(synthetic_results)
        summary_path = output_dir / f"synthetic_{timestamp}_summary.csv"
        summary.to_csv(summary_path)

    if real_results is not None and len(real_results) > 0:
        csv_path = output_dir / f"real_{timestamp}.csv"
        real_results.to_csv(csv_path, index=False)
        print(f"Real results saved to {csv_path}")

        json_path = output_dir / f"real_{timestamp}.json"
        real_results.to_json(json_path, orient="records", indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    if synthetic_results is not None and len(synthetic_results) > 0:
        acceptable = synthetic_results["benchmark_pass"] if "benchmark_pass" in synthetic_results else synthetic_results["converged"]
        print(f"\nSynthetic experiments: {len(synthetic_results)}")
        print(f"  Acceptable diagnostics: {acceptable.sum()} / {len(synthetic_results)}")
        if "strict_converged" in synthetic_results:
            print(
                f"  Strict Pass: {synthetic_results['strict_converged'].sum()} / {len(synthetic_results)}"
            )
        print(f"  Mean ELPD-LOO: {synthetic_results['elpd_loo'].mean():.1f}")
        print(f"  Mean test MAPE: {synthetic_results['test_mape'].mean():.3f}%")

    if real_results is not None and len(real_results) > 0:
        success = real_results[real_results["status"] == "success"]
        print(f"\nReal data experiments: {len(real_results)}")
        print(f"  Successful: {len(success)} / {len(real_results)}")
        if len(success) > 0:
            acceptable = success["benchmark_pass"] if "benchmark_pass" in success else success["converged"]
            print(f"  Acceptable diagnostics: {acceptable.sum()} / {len(success)}")
            if "strict_converged" in success:
                print(f"  Strict Pass: {success['strict_converged'].sum()} / {len(success)}")
            elpd_arr = np.array(success["elpd_loo"])
            mape_arr = np.array(success["test_mape"])
            print(
                f"  ELPD-LOO - Mean: {np.nanmean(elpd_arr):.1f}, Median: {np.nanmedian(elpd_arr):.1f}"
            )
            print(
                f"  Test MAPE - Mean: {np.nanmean(mape_arr):.3f}%, Median: {np.nanmedian(mape_arr):.3f}%"
            )
            if "elpd_waic" in success.columns:
                waic_arr = np.array(success["elpd_waic"])
                valid_waic = waic_arr[~np.isnan(waic_arr)]
                if len(valid_waic) > 0:
                    print(
                        f"  ELPD-WAIC - Mean: {np.mean(valid_waic):.1f}, Median: {np.median(valid_waic):.1f}"
                    )
            if "pareto_k_bad" in success.columns:
                total_bad = success["pareto_k_bad"].sum()
                print(f"  Total Pareto-k > 0.7: {total_bad}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Hill Mixture MMM benchmarks (synthetic and/or real data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --synthetic-only
  python scripts/run_benchmark.py --real-only
  python scripts/run_benchmark.py --quick
  python scripts/run_benchmark.py --dgp single mixture_k2 --synthetic-only
""",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with reduced experiments",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only synthetic experiments",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Run only real data experiments",
    )
    parser.add_argument(
        "--dgp",
        nargs="+",
        default=None,
        help="DGP scenarios to run (default: all)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (default: [0,1,2,3,4] for synthetic, [0,1,2] for real)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Number of MCMC chains (default: 4, quick: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides default)",
    )

    args = parser.parse_args()

    config = get_quick_config() if args.quick else get_default_config()

    if args.dgp:
        config.synthetic_dgps = args.dgp
    if args.model:
        config.synthetic_models = args.model
        config.real_models = [
            m for m in args.model if m in ["single_hill", "mixture_k2", "mixture_k3"]
        ]
    if args.seeds:
        config.synthetic_seeds = args.seeds
        config.real_seeds = args.seeds
    if args.chains:
        config.num_chains = args.chains
    if args.output:
        config.output_dir = args.output

    numpyro.set_host_device_count(config.num_chains)

    print("\n" + "#" * 60)
    print("# HILL MIXTURE MMM BENCHMARK")
    print("#" * 60)
    print(f"\nConfiguration: {'QUICK' if args.quick else 'FULL'}")
    print(f"Output directory: {config.output_dir}")
    print()

    synthetic_results = None
    real_results = None

    if not args.real_only:
        synthetic_results = run_synthetic_experiments(config)

    if not args.synthetic_only:
        real_results = run_real_data_experiments(config)

    save_results(synthetic_results, real_results, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
