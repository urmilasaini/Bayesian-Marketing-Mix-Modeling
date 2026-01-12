"""Benchmark harness for test-driven model evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baseline import standardized_time_index
from .data import DGPConfig, compute_prior_config, generate_data
from .data_loader import TimeSeriesConfig, load_timeseries
from .inference import (
    compute_comprehensive_mixture_diagnostics,
    compute_convergence_diagnostics,
    compute_hmc_diagnostics,
    compute_loo,
    compute_predictions,
    compute_waic,
    relabel_samples_by_k,
    run_inference,
)
from .metrics import (
    compute_component_curve_tv_separation,
    compute_delta_loo,
    compute_effective_k,
    compute_latent_recovery,
    compute_parameter_recovery,
    compute_permutation_invariant_component_recovery,
    compute_predictive_metrics,
    compute_similarity_adjusted_effective_count,
    summarize_component_posterior,
    summarize_true_components,
)
from .models import model_hill_mixture_hierarchical_reparam, model_single_hill
from .transforms import hill


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a benchmarked model."""

    name: str
    fn: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_SPECS: dict[str, ModelSpec] = {
    "single_hill": ModelSpec("single_hill", model_single_hill, {}),
    "mixture_k2": ModelSpec("mixture_k2", model_hill_mixture_hierarchical_reparam, {"K": 2}),
    "mixture_k3": ModelSpec("mixture_k3", model_hill_mixture_hierarchical_reparam, {"K": 3}),
}

PASS_WARN_FAIL_ORDER = {"NotApplicable": -1, "Pass": 0, "Warn": 1, "Fail": 2}
PASS_WARN_FAIL_THRESHOLDS = {
    "rhat_pass_max": 1.01,
    "rhat_fail_max": 1.05,
    "ess_pass_min_per_chain": 100.0,
    "ess_fail_min_per_chain": 50.0,
    "divergences_pass_max": 0,
    "divergences_fail_max": 5,
    "bfmi_pass_min": 0.3,
    "bfmi_fail_min": 0.2,
    "tree_depth_pass_max": 0,
    "tree_depth_fail_max": 10,
}


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Inference and split settings for a benchmark case."""

    seed: int = 42
    train_ratio: float = 0.75
    num_warmup: int = 300
    num_samples: int = 300
    num_chains: int = 2
    target_accept_prob: float = 0.90
    max_tree_depth: int = 10
    dense_mass: bool = False
    init_strategy: str = "uniform"
    progress_bar: bool = False
    prior_config_override: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkThresholds:
    """Pass/fail thresholds for one benchmark case."""

    max_rhat: float | None = 1.05
    min_ess_bulk: float | None = None
    min_ess_tail: float | None = None
    min_ess_bulk_per_chain: float | None = 50.0
    min_ess_tail_per_chain: float | None = 50.0
    max_label_invariant_rhat: float | None = None
    min_label_invariant_ess_bulk_per_chain: float | None = None
    min_label_invariant_ess_tail_per_chain: float | None = None
    max_relabeled_rhat: float | None = None
    min_relabeled_ess_bulk_per_chain: float | None = None
    min_relabeled_ess_tail_per_chain: float | None = None
    max_divergences: int | None = 5
    min_bfmi: float | None = 0.2
    max_tree_depth_hits: int | None = 10
    min_test_coverage_90: float | None = None
    max_test_mape: float | None = None
    max_test_crps: float | None = None
    max_test_mu_mape: float | None = None
    max_test_mu_nrmse: float | None = None
    min_test_mu_coverage_90: float | None = None
    max_component_weighted_curve_nrmse: float | None = None
    max_component_curve_nrmse: float | None = None
    max_component_effective_k_error: float | None = None
    require_alpha_in_ci: bool = False
    require_sigma_in_ci: bool = False
    effective_k_bounds: tuple[float, float] | None = None
    max_pareto_k_bad: int | None = None
    max_pareto_k_very_bad: int | None = None
    require_reportable_diagnostics: bool = False
    require_finite_loo_waic: bool = True
    require_finite_predictive_metrics: bool = True
    require_truth_metrics: bool = False


@dataclass(frozen=True)
class ComparisonThresholds:
    """Pass/fail thresholds for comparing two benchmark cases."""

    min_delta_loo: float | None = None
    max_delta_mape: float | None = None
    max_candidate_mape_ratio: float | None = None


@dataclass
class BenchmarkCaseResult:
    """Structured benchmark output for one model on one dataset."""

    label: str
    domain: str
    dataset_name: str
    model_name: str
    seed: int
    train_ratio: float
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    dates_train: np.ndarray | None
    dates_test: np.ndarray | None
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    loo: dict[str, Any]
    waic: dict[str, Any]
    convergence: dict[str, Any]
    hmc_diagnostics: dict[str, Any]
    label_invariant: dict[str, Any] | None
    relabeled: dict[str, Any] | None
    label_switching: dict[str, Any] | None
    component_summary: dict[str, Any] | None
    component_recovery: dict[str, Any] | None
    converged: bool
    effective_k: dict[str, Any]
    parameter_recovery: dict[str, Any] | None
    latent_train: dict[str, float] | None
    latent_test: dict[str, float] | None
    meta: dict[str, Any] | None
    samples: dict[str, np.ndarray]
    fit_summary: dict[str, Any]


def get_model_spec(model_name: str) -> ModelSpec:
    """Return benchmark model specification."""
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


def _required_ess(num_chains: int, minimum_per_chain: float) -> float:
    """Convert an ESS-per-chain rule into a total ESS threshold."""
    return float(num_chains) * float(minimum_per_chain)


def _append_nonfinite_scalar_errors(
    errors: list[str],
    metrics: dict[str, Any] | None,
    *,
    keys: Sequence[str],
    label: str,
) -> None:
    """Append errors for missing or non-finite scalar metrics."""
    if metrics is None:
        errors.append(f"{label} metrics are unavailable")
        return

    for key in keys:
        value = metrics.get(key, np.nan)
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            scalar = np.nan
        if not np.isfinite(scalar):
            errors.append(f"{label}.{key} is not finite")


def _append_truth_metric_errors(errors: list[str], result: BenchmarkCaseResult) -> None:
    """Append errors when synthetic truth-based metrics are unavailable."""
    if result.latent_test is None:
        errors.append("latent test metrics are unavailable")
    else:
        _append_nonfinite_scalar_errors(
            errors,
            result.latent_test,
            keys=("mape", "mae", "nrmse", "crps", "coverage_90", "coverage_95"),
            label="latent_test",
        )

    if not result.parameter_recovery:
        errors.append("parameter recovery metrics are unavailable")


def _merge_statuses(statuses: Sequence[str]) -> str:
    """Return the most severe Pass/Warn/Fail status."""
    normalized = [
        status
        for status in statuses
        if status in PASS_WARN_FAIL_ORDER and status != "NotApplicable"
    ]
    if not normalized:
        return "NotApplicable"
    return max(normalized, key=lambda status: PASS_WARN_FAIL_ORDER[status])


def _evaluate_upper_bound(
    *,
    name: str,
    value: float | None,
    pass_max: float,
    fail_max: float,
) -> dict[str, Any]:
    """Evaluate an upper-bound diagnostic into Pass/Warn/Fail."""
    if value is None or not np.isfinite(value):
        return {
            "name": name,
            "status": "Fail",
            "value": None,
            "direction": "upper",
            "pass_threshold": float(pass_max),
            "fail_threshold": float(fail_max),
            "message": f"{name} is unavailable",
        }
    if value <= pass_max:
        status = "Pass"
        message = None
    elif value <= fail_max:
        status = "Warn"
        message = f"{name}={value:.3f} exceeds pass threshold {pass_max:.3f}"
    else:
        status = "Fail"
        message = f"{name}={value:.3f} exceeds fail threshold {fail_max:.3f}"
    return {
        "name": name,
        "status": status,
        "value": float(value),
        "direction": "upper",
        "pass_threshold": float(pass_max),
        "fail_threshold": float(fail_max),
        "message": message,
    }


def _evaluate_lower_bound(
    *,
    name: str,
    value: float | None,
    pass_min: float,
    fail_min: float,
) -> dict[str, Any]:
    """Evaluate a lower-bound diagnostic into Pass/Warn/Fail."""
    if value is None or not np.isfinite(value):
        return {
            "name": name,
            "status": "Fail",
            "value": None,
            "direction": "lower",
            "pass_threshold": float(pass_min),
            "fail_threshold": float(fail_min),
            "message": f"{name} is unavailable",
        }
    if value >= pass_min:
        status = "Pass"
        message = None
    elif value >= fail_min:
        status = "Warn"
        message = f"{name}={value:.1f} is below pass threshold {pass_min:.1f}"
    else:
        status = "Fail"
        message = f"{name}={value:.1f} is below fail threshold {fail_min:.1f}"
    return {
        "name": name,
        "status": status,
        "value": float(value),
        "direction": "lower",
        "pass_threshold": float(pass_min),
        "fail_threshold": float(fail_min),
        "message": message,
    }


def evaluate_diagnostic_summary(
    *,
    convergence: dict[str, Any],
    hmc_diagnostics: dict[str, Any],
    label_invariant: dict[str, Any] | None,
    relabeled: dict[str, Any] | None,
    num_chains_used: int,
    strict_converged: bool | None = None,
) -> dict[str, Any]:
    """Classify diagnostics into publication and interpretation statuses."""
    thresholds = PASS_WARN_FAIL_THRESHOLDS
    checks: list[dict[str, Any]] = []

    sampler_checks = [
        _evaluate_upper_bound(
            name="num_divergences",
            value=float(hmc_diagnostics.get("num_divergences", np.nan)),
            pass_max=float(thresholds["divergences_pass_max"]),
            fail_max=float(thresholds["divergences_fail_max"]),
        ),
        _evaluate_lower_bound(
            name="min_bfmi",
            value=float(hmc_diagnostics.get("min_bfmi", np.nan)),
            pass_min=float(thresholds["bfmi_pass_min"]),
            fail_min=float(thresholds["bfmi_fail_min"]),
        ),
        _evaluate_upper_bound(
            name="tree_depth_hits",
            value=float(hmc_diagnostics.get("tree_depth_hits", np.nan)),
            pass_max=float(thresholds["tree_depth_pass_max"]),
            fail_max=float(thresholds["tree_depth_fail_max"]),
        ),
    ]
    for check in sampler_checks:
        check["group"] = "sampler"
    checks.extend(sampler_checks)

    if label_invariant is None:
        mixing_checks = [
            _evaluate_upper_bound(
                name="max_rhat",
                value=float(convergence.get("max_rhat", np.nan)),
                pass_max=float(thresholds["rhat_pass_max"]),
                fail_max=float(thresholds["rhat_fail_max"]),
            ),
            _evaluate_lower_bound(
                name="min_ess_bulk_per_chain",
                value=float(convergence.get("min_ess_bulk", np.nan)) / max(num_chains_used, 1),
                pass_min=float(thresholds["ess_pass_min_per_chain"]),
                fail_min=float(thresholds["ess_fail_min_per_chain"]),
            ),
            _evaluate_lower_bound(
                name="min_ess_tail_per_chain",
                value=float(convergence.get("min_ess_tail", np.nan)) / max(num_chains_used, 1),
                pass_min=float(thresholds["ess_pass_min_per_chain"]),
                fail_min=float(thresholds["ess_fail_min_per_chain"]),
            ),
        ]
        interpretation_checks: list[dict[str, Any]] = []
    else:
        mixing_checks = [
            _evaluate_upper_bound(
                name="label_invariant_max_rhat",
                value=float(label_invariant.get("max_rhat", np.nan)),
                pass_max=float(thresholds["rhat_pass_max"]),
                fail_max=float(thresholds["rhat_fail_max"]),
            ),
            _evaluate_lower_bound(
                name="label_invariant_min_ess_bulk_per_chain",
                value=float(label_invariant.get("min_ess_bulk", np.nan)) / max(num_chains_used, 1),
                pass_min=float(thresholds["ess_pass_min_per_chain"]),
                fail_min=float(thresholds["ess_fail_min_per_chain"]),
            ),
            _evaluate_lower_bound(
                name="label_invariant_min_ess_tail_per_chain",
                value=float(label_invariant.get("min_ess_tail", np.nan)) / max(num_chains_used, 1),
                pass_min=float(thresholds["ess_pass_min_per_chain"]),
                fail_min=float(thresholds["ess_fail_min_per_chain"]),
            ),
        ]
        if relabeled is None:
            interpretation_checks = [
                {
                    "name": "relabeled_diagnostics",
                    "status": "Fail",
                    "value": None,
                    "direction": "none",
                    "pass_threshold": None,
                    "fail_threshold": None,
                    "group": "interpretation",
                    "message": "relabeled diagnostics are unavailable",
                }
            ]
        else:
            interpretation_checks = [
                _evaluate_upper_bound(
                    name="relabeled_max_rhat",
                    value=float(relabeled.get("max_rhat", np.nan)),
                    pass_max=float(thresholds["rhat_pass_max"]),
                    fail_max=float(thresholds["rhat_fail_max"]),
                ),
                _evaluate_lower_bound(
                    name="relabeled_min_ess_bulk_per_chain",
                    value=float(relabeled.get("min_ess_bulk", np.nan)) / max(num_chains_used, 1),
                    pass_min=float(thresholds["ess_pass_min_per_chain"]),
                    fail_min=float(thresholds["ess_fail_min_per_chain"]),
                ),
                _evaluate_lower_bound(
                    name="relabeled_min_ess_tail_per_chain",
                    value=float(relabeled.get("min_ess_tail", np.nan)) / max(num_chains_used, 1),
                    pass_min=float(thresholds["ess_pass_min_per_chain"]),
                    fail_min=float(thresholds["ess_fail_min_per_chain"]),
                ),
            ]

    for check in mixing_checks:
        check["group"] = "mixing"
    for check in interpretation_checks:
        check["group"] = check.get("group", "interpretation")

    checks.extend(mixing_checks)
    checks.extend(interpretation_checks)

    sampler_status = _merge_statuses([check["status"] for check in sampler_checks])
    mixing_status = _merge_statuses([check["status"] for check in mixing_checks])
    interpretation_status = _merge_statuses([check["status"] for check in interpretation_checks])
    publication_status = _merge_statuses([sampler_status, mixing_status])
    if strict_converged is None:
        strict_converged = publication_status == "Pass" and interpretation_status in {
            "Pass",
            "NotApplicable",
        }

    return {
        "publication_status": publication_status,
        "sampler_status": sampler_status,
        "mixing_status": mixing_status,
        "interpretation_status": interpretation_status,
        "benchmark_pass": publication_status != "Fail",
        "strict_converged": bool(strict_converged),
        "warnings": [
            check["message"]
            for check in checks
            if check["status"] == "Warn" and check.get("message") is not None
        ],
        "failures": [
            check["message"]
            for check in checks
            if check["status"] == "Fail" and check.get("message") is not None
        ],
        "checks": checks,
    }


def evaluate_case_diagnostic_status(result: BenchmarkCaseResult) -> dict[str, Any]:
    """Classify benchmark diagnostics into publication and interpretation statuses."""
    return evaluate_diagnostic_summary(
        convergence=result.convergence,
        hmc_diagnostics=result.hmc_diagnostics,
        label_invariant=result.label_invariant,
        relabeled=result.relabeled,
        num_chains_used=int(result.fit_summary.get("num_chains_used", 1)),
        strict_converged=bool(result.converged),
    )


def _passes_hmc_requirements(
    hmc_diagnostics: dict[str, Any],
    *,
    max_divergences: int = 0,
    min_bfmi: float = 0.3,
    max_tree_depth_hits: int = 0,
) -> bool:
    """Return True when HMC diagnostics satisfy the publication-quality rules."""
    if int(hmc_diagnostics.get("num_divergences", 0)) > max_divergences:
        return False
    if float(hmc_diagnostics.get("min_bfmi", np.nan)) < min_bfmi:
        return False
    if int(hmc_diagnostics.get("tree_depth_hits", 0)) > max_tree_depth_hits:
        return False
    return True


def _is_effectively_converged(
    *,
    model_name: str,
    convergence: dict[str, Any],
    hmc_diagnostics: dict[str, Any],
    label_invariant: dict[str, Any] | None,
    relabeled: dict[str, Any] | None,
    num_chains: int,
    rhat_threshold: float = 1.01,
    min_ess_per_chain: float = 100.0,
) -> bool:
    """Evaluate publication-quality convergence for benchmark outputs."""
    required_ess = _required_ess(num_chains, min_ess_per_chain)
    if not _passes_hmc_requirements(hmc_diagnostics):
        return False

    if model_name == "single_hill":
        return bool(
            float(convergence.get("max_rhat", np.inf)) <= rhat_threshold
            and float(convergence.get("min_ess_bulk", 0.0)) >= required_ess
            and float(convergence.get("min_ess_tail", 0.0)) >= required_ess
        )

    if label_invariant is None or relabeled is None:
        return False

    return bool(
        float(label_invariant.get("max_rhat", np.inf)) <= rhat_threshold
        and float(label_invariant.get("min_ess_bulk", 0.0)) >= required_ess
        and float(label_invariant.get("min_ess_tail", 0.0)) >= required_ess
        and float(relabeled.get("max_rhat", np.inf)) <= rhat_threshold
        and float(relabeled.get("min_ess_bulk", 0.0)) >= required_ess
        and float(relabeled.get("min_ess_tail", 0.0)) >= required_ess
    )


def _extract_latent_samples(predictions: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return latent mean samples from posterior predictive output."""
    if "mu_expected" in predictions:
        return np.asarray(predictions["mu_expected"], dtype=np.float32)
    if "mu" in predictions:
        return np.asarray(predictions["mu"], dtype=np.float32)
    return None


def _reference_latent_mean(meta: dict[str, Any]) -> np.ndarray | None:
    """Return the synthetic latent target aligned with posterior mean predictions."""
    if "mu_expected_true" in meta:
        return np.asarray(meta["mu_expected_true"], dtype=np.float32)
    if "mu_true" in meta:
        return np.asarray(meta["mu_true"], dtype=np.float32)
    return None


def _reference_latent_label(meta: dict[str, Any]) -> str:
    """Return a descriptive label for the plotted latent target."""
    return "True Expected Latent Mean" if "mu_expected_true" in meta else "True Latent Mean"


def _fit_case(
    model_spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    t_std_train: np.ndarray,
    prior_config: dict[str, Any],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    """Run one inference pass and return diagnostics."""
    numpyro.set_host_device_count(max(1, config.num_chains))

    is_mixture_model = "K" in model_spec.kwargs
    fit_summary = {
        "inference_seed": config.seed,
        "num_warmup_used": config.num_warmup,
        "num_samples_used": config.num_samples,
        "num_chains_used": config.num_chains,
        "target_accept_prob_used": config.target_accept_prob,
        "max_tree_depth_used": config.max_tree_depth,
        "dense_mass_used": bool(config.dense_mass),
        "init_strategy_used": config.init_strategy,
    }

    mcmc = run_inference(
        model_spec.fn,
        x_train,
        y_train,
        seed=fit_summary["inference_seed"],
        num_warmup=fit_summary["num_warmup_used"],
        num_samples=fit_summary["num_samples_used"],
        num_chains=config.num_chains,
        prior_config=prior_config,
        t_std=t_std_train,
        target_accept_prob=fit_summary["target_accept_prob_used"],
        max_tree_depth=fit_summary["max_tree_depth_used"],
        dense_mass=fit_summary["dense_mass_used"],
        init_strategy=fit_summary["init_strategy_used"],
        progress_bar=config.progress_bar,
        **model_spec.kwargs,
    )
    hmc_diagnostics = compute_hmc_diagnostics(
        mcmc,
        max_tree_depth=fit_summary["max_tree_depth_used"],
    )
    if is_mixture_model:
        diagnostics = compute_comprehensive_mixture_diagnostics(mcmc, x_train, y_train)
        convergence = diagnostics["standard"]
        label_invariant = diagnostics["label_invariant"]
        relabeled = diagnostics["relabeled"]
        label_switching = diagnostics["label_switching"]
    else:
        convergence = compute_convergence_diagnostics(mcmc)
        label_invariant = None
        relabeled = None
        label_switching = None

    effective_convergence = _is_effectively_converged(
        model_name=model_spec.name,
        convergence=convergence,
        hmc_diagnostics=hmc_diagnostics,
        label_invariant=label_invariant,
        relabeled=relabeled,
        num_chains=fit_summary["num_chains_used"],
    )

    return {
        "mcmc": mcmc,
        "convergence": convergence,
        "hmc_diagnostics": hmc_diagnostics,
        "label_invariant": label_invariant,
        "relabeled": relabeled,
        "label_switching": label_switching,
        "converged": effective_convergence,
        "fit_summary": fit_summary,
    }


def _run_case_from_series(
    *,
    label: str,
    domain: str,
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | None,
    model_spec: ModelSpec,
    config: BenchmarkRunConfig,
    meta: dict[str, Any] | None = None,
) -> BenchmarkCaseResult:
    """Run one benchmark case from a prepared single-series dataset."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    T = len(y)
    T_train = int(T * config.train_ratio)
    if T_train <= 0 or T_train >= T:
        raise ValueError("train_ratio must leave at least one train and one test observation")

    x_train, y_train = x[:T_train], y[:T_train]
    x_test, y_test = x[T_train:], y[T_train:]
    dates_train = None if dates is None else np.asarray(dates[:T_train])
    dates_test = None if dates is None else np.asarray(dates[T_train:])

    prior_config = compute_prior_config(x_train, y_train)
    if config.prior_config_override:
        prior_config.update(config.prior_config_override)
    t_std_train = np.asarray(standardized_time_index(T)[:T_train], dtype=np.float32)

    fit = _fit_case(model_spec, x_train, y_train, t_std_train, prior_config, config)
    mcmc = fit["mcmc"]

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

    latent_train = None
    latent_test = None
    if meta is not None:
        latent_truth = _reference_latent_mean(meta)
        train_mu = _extract_latent_samples(pred_train)
        test_mu = _extract_latent_samples(pred_test)
        if latent_truth is not None and train_mu is not None and test_mu is not None:
            latent_train = compute_latent_recovery(latent_truth[:T_train], train_mu)
            latent_test = compute_latent_recovery(latent_truth[T_train:], test_mu)

    parameter_recovery = None
    if meta is not None:
        parameter_recovery = compute_parameter_recovery(mcmc, meta)

    samples = mcmc.get_samples()
    if "pis" in samples and np.asarray(samples["pis"]).ndim == 2:
        samples = relabel_samples_by_k(samples)

    component_scale = 1.0
    if meta is not None and "s_median" in meta:
        component_scale = float(meta["s_median"])
    component_summary = summarize_component_posterior(samples, scale_reference=component_scale)

    component_recovery = None
    if meta is not None:
        component_recovery = compute_permutation_invariant_component_recovery(samples, meta)

    return BenchmarkCaseResult(
        label=label,
        domain=domain,
        dataset_name=dataset_name,
        model_name=model_spec.name,
        seed=config.seed,
        train_ratio=config.train_ratio,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        dates_train=dates_train,
        dates_test=dates_test,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        loo=compute_loo(mcmc),
        waic=compute_waic(mcmc),
        convergence=fit["convergence"],
        hmc_diagnostics=fit["hmc_diagnostics"],
        label_invariant=fit["label_invariant"],
        relabeled=fit["relabeled"],
        label_switching=fit["label_switching"],
        component_summary=component_summary,
        component_recovery=component_recovery,
        converged=fit["converged"],
        effective_k=compute_effective_k(mcmc),
        parameter_recovery=parameter_recovery,
        latent_train=latent_train,
        latent_test=latent_test,
        meta=meta,
        samples={k: np.asarray(v) for k, v in samples.items()},
        fit_summary=fit["fit_summary"],
    )


def run_synthetic_benchmark_case(
    *,
    dgp_config: DGPConfig,
    model_name: str,
    config: BenchmarkRunConfig,
    label: str | None = None,
) -> BenchmarkCaseResult:
    """Run one synthetic benchmark case."""
    x, y, meta = generate_data(dgp_config)
    dataset_label = dgp_config.dataset_label
    case_label = label or f"synthetic_{dataset_label}_{model_name}"
    return _run_case_from_series(
        label=case_label,
        domain="synthetic",
        dataset_name=dataset_label,
        x=x,
        y=y,
        dates=None,
        model_spec=get_model_spec(model_name),
        config=config,
        meta=meta,
    )


def run_prepared_synthetic_benchmark_case(
    *,
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    model_name: str,
    config: BenchmarkRunConfig,
    label: str | None = None,
) -> BenchmarkCaseResult:
    """Run one synthetic benchmark case from pre-generated arrays and metadata."""
    case_label = label or f"synthetic_{dataset_name}_{model_name}"
    return _run_case_from_series(
        label=case_label,
        domain="synthetic",
        dataset_name=dataset_name,
        x=x,
        y=y,
        dates=None,
        model_spec=get_model_spec(model_name),
        config=config,
        meta=meta,
    )


def run_real_benchmark_case(
    *,
    csv_path: str | Path,
    timeseries_config: TimeSeriesConfig,
    model_name: str,
    config: BenchmarkRunConfig,
    label: str | None = None,
) -> BenchmarkCaseResult:
    """Run one real-data benchmark case."""
    loaded = load_timeseries(csv_path, timeseries_config)
    case_label = label or f"real_{timeseries_config.organisation_id}_{model_name}"
    return _run_case_from_series(
        label=case_label,
        domain="real",
        dataset_name=str(timeseries_config.organisation_id),
        x=loaded.x,
        y=loaded.y,
        dates=loaded.dates,
        model_spec=get_model_spec(model_name),
        config=config,
        meta=loaded.meta,
    )


def case_summary(result: BenchmarkCaseResult) -> dict[str, Any]:
    """Return a compact JSON-serializable summary."""
    diagnostic_status = evaluate_case_diagnostic_status(result)
    component_effective_count = None
    if result.component_summary is not None:
        component_effective_count = compute_similarity_adjusted_effective_count(
            result.component_summary
        )
    true_component_summary = None
    true_component_separation = None
    if result.meta is not None and {"A_true", "k_true", "n_true", "pi_true"}.issubset(result.meta):
        true_component_summary = summarize_true_components(result.meta)
        true_component_separation = compute_component_curve_tv_separation(true_component_summary)
    summary = {
        "label": result.label,
        "domain": result.domain,
        "dataset_name": result.dataset_name,
        "model_name": result.model_name,
        "seed": result.seed,
        "train_size": int(len(result.y_train)),
        "test_size": int(len(result.y_test)),
        "converged": bool(result.converged),
        "publication_status": diagnostic_status["publication_status"],
        "interpretation_status": diagnostic_status["interpretation_status"],
        "benchmark_pass": bool(diagnostic_status["benchmark_pass"]),
        "convergence": {
            "max_rhat": float(result.convergence["max_rhat"]),
            "min_ess_bulk": float(result.convergence["min_ess_bulk"]),
            "min_ess_tail": float(result.convergence["min_ess_tail"]),
        },
        "hmc_diagnostics": {
            "num_divergences": int(result.hmc_diagnostics["num_divergences"]),
            "min_bfmi": float(result.hmc_diagnostics["min_bfmi"]),
            "tree_depth_hits": int(result.hmc_diagnostics["tree_depth_hits"]),
            "max_num_steps": int(result.hmc_diagnostics["max_num_steps"]),
            "mean_accept_prob": float(result.hmc_diagnostics["mean_accept_prob"]),
        },
        "loo": {
            "elpd_loo": float(result.loo.get("elpd_loo", np.nan)),
            "se": float(result.loo.get("se", np.nan)),
            "pareto_k_bad": int(result.loo.get("pareto_k_bad", 0)),
            "pareto_k_very_bad": int(result.loo.get("pareto_k_very_bad", 0)),
        },
        "waic": {
            "elpd_waic": float(result.waic.get("elpd_waic", np.nan)),
            "se": float(result.waic.get("se", np.nan)),
        },
        "train_metrics": {
            "mape": float(result.train_metrics["mape"]),
            "nrmse": float(result.train_metrics["nrmse"]),
            "crps": float(result.train_metrics["crps"]),
            "coverage_90": float(result.train_metrics["coverage_90"]),
        },
        "test_metrics": {
            "mape": float(result.test_metrics["mape"]),
            "nrmse": float(result.test_metrics["nrmse"]),
            "crps": float(result.test_metrics["crps"]),
            "coverage_90": float(result.test_metrics["coverage_90"]),
        },
        "effective_k": {
            "mean": float(result.effective_k["effective_k_mean"]),
            "std": float(result.effective_k["effective_k_std"]),
        },
        "diagnostic_status": diagnostic_status,
        "fit_summary": dict(result.fit_summary),
    }
    if component_effective_count is not None or true_component_separation is not None:
        summary["component_count_plot"] = {
            "curve_basis": "l1_normalized_hill_curve_increments",
            "curve_grid_max": 4.0,
            "grid_size": 128,
            "gamma": float(component_effective_count["gamma"])
            if component_effective_count
            else 0.5,
            "true_separation": (
                float(true_component_separation["mean_pairwise_tv"])
                if true_component_separation is not None
                else None
            ),
            "estimated_effective_count": (
                float(component_effective_count["effective_count"])
                if component_effective_count is not None
                else None
            ),
        }
    if result.label_invariant is not None:
        summary["label_invariant"] = {
            "rhat_log_lik": float(result.label_invariant["rhat_log_lik"]),
            "max_rhat": float(result.label_invariant["max_rhat"]),
            "min_ess_bulk": float(result.label_invariant["min_ess_bulk"]),
            "min_ess_tail": float(result.label_invariant["min_ess_tail"]),
            "threshold": float(result.label_invariant["threshold"]),
        }
    if result.relabeled is not None:
        summary["relabeled"] = {
            "max_rhat": float(result.relabeled["max_rhat"]),
            "min_ess_bulk": float(result.relabeled["min_ess_bulk"]),
            "min_ess_tail": float(result.relabeled["min_ess_tail"]),
            "threshold": float(result.relabeled["threshold"]),
            "component_rhats": {
                name: {
                    "max": float(metrics["max"]),
                    "per_component": [float(value) for value in metrics["per_component"]],
                }
                for name, metrics in result.relabeled["component_rhats"].items()
            },
        }
    if result.label_switching is not None:
        summary["label_switching"] = {
            "switching_rate": float(result.label_switching["switching_rate"]),
            "n_unique_orderings": int(result.label_switching["n_unique_orderings"]),
            "mode_ordering": [int(value) for value in result.label_switching["mode_ordering"]],
            "mode_count": int(result.label_switching["mode_count"]),
            "top_orderings": [
                {
                    "ordering": [int(value) for value in ordering],
                    "count": int(count),
                }
                for ordering, count in result.label_switching["top_orderings"]
            ],
        }
    if result.component_summary is not None:
        summary["component_summary"] = {
            "K_total": int(result.component_summary["K_total"]),
            "K_active": int(result.component_summary["K_active"]),
            "weight_threshold": float(result.component_summary["weight_threshold"]),
            "scale_reference": float(result.component_summary["scale_reference"]),
            "components": [
                {
                    "index": int(component["index"]),
                    "A_mean": float(component["A_mean"]),
                    "A_std": float(component["A_std"]),
                    "k_mean": float(component["k_mean"]),
                    "k_std": float(component["k_std"]),
                    "k_ratio_mean": float(component["k_ratio_mean"]),
                    "k_ratio_std": float(component["k_ratio_std"]),
                    "n_mean": float(component["n_mean"]),
                    "n_std": float(component["n_std"]),
                    "pi_mean": float(component["pi_mean"]),
                    "pi_std": float(component["pi_std"]),
                    "active": bool(component["active"]),
                }
                for component in result.component_summary["components"]
            ],
        }
    if true_component_summary is not None:
        summary["true_component_summary"] = {
            "K_total": int(true_component_summary["K_total"]),
            "K_active": int(true_component_summary["K_active"]),
            "weight_threshold": float(true_component_summary["weight_threshold"]),
            "scale_reference": float(true_component_summary["scale_reference"]),
            "components": [
                {
                    "index": int(component["index"]),
                    "A_mean": float(component["A_mean"]),
                    "A_std": float(component["A_std"]),
                    "k_mean": float(component["k_mean"]),
                    "k_std": float(component["k_std"]),
                    "k_ratio_mean": float(component["k_ratio_mean"]),
                    "k_ratio_std": float(component["k_ratio_std"]),
                    "n_mean": float(component["n_mean"]),
                    "n_std": float(component["n_std"]),
                    "pi_mean": float(component["pi_mean"]),
                    "pi_std": float(component["pi_std"]),
                    "active": bool(component["active"]),
                }
                for component in true_component_summary["components"]
            ],
        }
    if result.component_recovery is not None:
        summary["component_recovery"] = {
            "K_true": int(result.component_recovery["K_true"]),
            "K_true_active": int(result.component_recovery["K_true_active"]),
            "K_posterior": int(result.component_recovery["K_posterior"]),
            "K_posterior_active": int(result.component_recovery["K_posterior_active"]),
            "effective_k_error": float(result.component_recovery["effective_k_error"]),
            "weighted_curve_rmse": float(result.component_recovery["weighted_curve_rmse"]),
            "weighted_curve_nrmse": float(result.component_recovery["weighted_curve_nrmse"]),
            "weighted_pi_abs_error": float(result.component_recovery["weighted_pi_abs_error"]),
            "mean_A_rel_error": float(result.component_recovery["mean_A_rel_error"]),
            "mean_k_ratio_rel_error": float(result.component_recovery["mean_k_ratio_rel_error"]),
            "mean_n_abs_error": float(result.component_recovery["mean_n_abs_error"]),
            "unmatched_true_weight": float(result.component_recovery["unmatched_reference_weight"]),
            "unmatched_posterior_weight": float(
                result.component_recovery["unmatched_candidate_weight"]
            ),
            "matched_components": [
                {
                    "true_component_index": int(component["reference_component_index"]),
                    "posterior_component_index": int(component["candidate_component_index"]),
                    "curve_rmse": float(component["curve_rmse"]),
                    "curve_nrmse": float(component["curve_nrmse"]),
                    "pi_abs_error": float(component["pi_abs_error"]),
                    "A_rel_error": float(component["A_rel_error"]),
                    "k_ratio_rel_error": float(component["k_ratio_rel_error"]),
                    "n_abs_error": float(component["n_abs_error"]),
                }
                for component in result.component_recovery["matched_components"]
            ],
        }
    if result.parameter_recovery is not None:
        summary["parameter_recovery"] = {
            name: {
                key: value
                for key, value in metrics.items()
                if key in {"true", "mean", "ci_low", "ci_high", "in_ci"}
            }
            for name, metrics in result.parameter_recovery.items()
            if isinstance(metrics, dict)
        }
    if result.latent_train is not None:
        summary["latent_train"] = dict(result.latent_train)
    if result.latent_test is not None:
        summary["latent_test"] = dict(result.latent_test)
    return summary


def compare_case_results(
    baseline: BenchmarkCaseResult,
    candidate: BenchmarkCaseResult,
) -> dict[str, float]:
    """Compute comparison metrics between two benchmark cases."""
    delta = compute_delta_loo(candidate.loo, baseline.loo)
    return {
        "delta_loo": float(delta["delta_loo"]),
        "delta_loo_se": float(delta["se"]),
        "delta_loo_significant": float(bool(delta["significant"])),
        "delta_test_mape": float(candidate.test_metrics["mape"] - baseline.test_metrics["mape"]),
        "candidate_mape_ratio": float(
            candidate.test_metrics["mape"] / baseline.test_metrics["mape"]
        ),
        "delta_test_coverage_90": float(
            candidate.test_metrics["coverage_90"] - baseline.test_metrics["coverage_90"]
        ),
    }


def assert_case_passes(result: BenchmarkCaseResult, thresholds: BenchmarkThresholds) -> None:
    """Raise an AssertionError if a benchmark case fails its thresholds."""
    errors: list[str] = []
    num_chains_used = int(result.fit_summary.get("num_chains_used", 1))
    diagnostic_status: dict[str, Any] | None = None

    if thresholds.require_reportable_diagnostics:
        diagnostic_status = evaluate_case_diagnostic_status(result)
        if diagnostic_status["publication_status"] == "Fail":
            errors.append("publication_status=Fail")
            errors.extend(diagnostic_status["failures"])

    if (
        thresholds.max_rhat is not None
        and float(result.convergence["max_rhat"]) > thresholds.max_rhat
    ):
        errors.append(
            f"max_rhat={result.convergence['max_rhat']:.3f} exceeds {thresholds.max_rhat:.3f}"
        )
    if (
        thresholds.min_ess_bulk is not None
        and float(result.convergence["min_ess_bulk"]) < thresholds.min_ess_bulk
    ):
        errors.append(
            f"min_ess_bulk={result.convergence['min_ess_bulk']:.1f} is below {thresholds.min_ess_bulk:.1f}"
        )
    if (
        thresholds.min_ess_tail is not None
        and float(result.convergence["min_ess_tail"]) < thresholds.min_ess_tail
    ):
        errors.append(
            f"min_ess_tail={result.convergence['min_ess_tail']:.1f} is below {thresholds.min_ess_tail:.1f}"
        )
    if thresholds.min_ess_bulk_per_chain is not None:
        required_ess_bulk = _required_ess(num_chains_used, thresholds.min_ess_bulk_per_chain)
        if float(result.convergence["min_ess_bulk"]) < required_ess_bulk:
            errors.append(
                f"min_ess_bulk={result.convergence['min_ess_bulk']:.1f} is below "
                f"{required_ess_bulk:.1f} ({thresholds.min_ess_bulk_per_chain:.1f} per chain)"
            )
    if thresholds.min_ess_tail_per_chain is not None:
        required_ess_tail = _required_ess(num_chains_used, thresholds.min_ess_tail_per_chain)
        if float(result.convergence["min_ess_tail"]) < required_ess_tail:
            errors.append(
                f"min_ess_tail={result.convergence['min_ess_tail']:.1f} is below "
                f"{required_ess_tail:.1f} ({thresholds.min_ess_tail_per_chain:.1f} per chain)"
            )
    if thresholds.max_label_invariant_rhat is not None and result.label_invariant is not None:
        label_max_rhat = float(result.label_invariant["max_rhat"])
        if label_max_rhat > thresholds.max_label_invariant_rhat:
            errors.append(
                f"label_invariant_max_rhat={label_max_rhat:.3f} exceeds "
                f"{thresholds.max_label_invariant_rhat:.3f}"
            )
    elif thresholds.max_label_invariant_rhat is not None:
        errors.append("label-invariant diagnostics are unavailable")
    if thresholds.min_label_invariant_ess_bulk_per_chain is not None:
        if result.label_invariant is None:
            errors.append("label-invariant diagnostics are unavailable")
        else:
            required_label_ess_bulk = _required_ess(
                num_chains_used, thresholds.min_label_invariant_ess_bulk_per_chain
            )
            if float(result.label_invariant["min_ess_bulk"]) < required_label_ess_bulk:
                errors.append(
                    f"label_invariant_min_ess_bulk={result.label_invariant['min_ess_bulk']:.1f} "
                    f"is below {required_label_ess_bulk:.1f} "
                    f"({thresholds.min_label_invariant_ess_bulk_per_chain:.1f} per chain)"
                )
    if thresholds.min_label_invariant_ess_tail_per_chain is not None:
        if result.label_invariant is None:
            errors.append("label-invariant diagnostics are unavailable")
        else:
            required_label_ess_tail = _required_ess(
                num_chains_used, thresholds.min_label_invariant_ess_tail_per_chain
            )
            if float(result.label_invariant["min_ess_tail"]) < required_label_ess_tail:
                errors.append(
                    f"label_invariant_min_ess_tail={result.label_invariant['min_ess_tail']:.1f} "
                    f"is below {required_label_ess_tail:.1f} "
                    f"({thresholds.min_label_invariant_ess_tail_per_chain:.1f} per chain)"
                )
    if thresholds.max_relabeled_rhat is not None:
        if result.relabeled is None:
            errors.append("relabeled diagnostics are unavailable")
        elif float(result.relabeled["max_rhat"]) > thresholds.max_relabeled_rhat:
            errors.append(
                f"relabeled_max_rhat={result.relabeled['max_rhat']:.3f} exceeds "
                f"{thresholds.max_relabeled_rhat:.3f}"
            )
    if thresholds.min_relabeled_ess_bulk_per_chain is not None:
        if result.relabeled is None:
            errors.append("relabeled diagnostics are unavailable")
        else:
            required_relabeled_ess_bulk = _required_ess(
                num_chains_used, thresholds.min_relabeled_ess_bulk_per_chain
            )
            if float(result.relabeled["min_ess_bulk"]) < required_relabeled_ess_bulk:
                errors.append(
                    f"relabeled_min_ess_bulk={result.relabeled['min_ess_bulk']:.1f} is below "
                    f"{required_relabeled_ess_bulk:.1f} "
                    f"({thresholds.min_relabeled_ess_bulk_per_chain:.1f} per chain)"
                )
    if thresholds.min_relabeled_ess_tail_per_chain is not None:
        if result.relabeled is None:
            errors.append("relabeled diagnostics are unavailable")
        else:
            required_relabeled_ess_tail = _required_ess(
                num_chains_used, thresholds.min_relabeled_ess_tail_per_chain
            )
            if float(result.relabeled["min_ess_tail"]) < required_relabeled_ess_tail:
                errors.append(
                    f"relabeled_min_ess_tail={result.relabeled['min_ess_tail']:.1f} is below "
                    f"{required_relabeled_ess_tail:.1f} "
                    f"({thresholds.min_relabeled_ess_tail_per_chain:.1f} per chain)"
                )
    if thresholds.max_divergences is not None:
        num_divergences = int(result.hmc_diagnostics["num_divergences"])
        if num_divergences > thresholds.max_divergences:
            errors.append(f"num_divergences={num_divergences} exceeds {thresholds.max_divergences}")
    if thresholds.min_bfmi is not None:
        min_bfmi = float(result.hmc_diagnostics["min_bfmi"])
        if min_bfmi < thresholds.min_bfmi:
            errors.append(f"min_bfmi={min_bfmi:.3f} is below {thresholds.min_bfmi:.3f}")
    if thresholds.max_tree_depth_hits is not None:
        tree_depth_hits = int(result.hmc_diagnostics["tree_depth_hits"])
        if tree_depth_hits > thresholds.max_tree_depth_hits:
            errors.append(
                f"tree_depth_hits={tree_depth_hits} exceeds {thresholds.max_tree_depth_hits}"
            )

    if thresholds.require_finite_loo_waic:
        _append_nonfinite_scalar_errors(
            errors,
            result.loo,
            keys=("elpd_loo", "se"),
            label="loo",
        )
        _append_nonfinite_scalar_errors(
            errors,
            result.waic,
            keys=("elpd_waic", "se"),
            label="waic",
        )

    if thresholds.require_finite_predictive_metrics:
        _append_nonfinite_scalar_errors(
            errors,
            result.train_metrics,
            keys=("mape", "nrmse", "crps", "coverage_90"),
            label="train_metrics",
        )
        _append_nonfinite_scalar_errors(
            errors,
            result.test_metrics,
            keys=("mape", "nrmse", "crps", "coverage_90"),
            label="test_metrics",
        )

    if thresholds.require_truth_metrics:
        _append_truth_metric_errors(errors, result)

    if thresholds.max_component_weighted_curve_nrmse is not None:
        if result.component_recovery is None:
            errors.append("component recovery metrics are unavailable")
        else:
            weighted_curve_nrmse = float(result.component_recovery["weighted_curve_nrmse"])
            if weighted_curve_nrmse > thresholds.max_component_weighted_curve_nrmse:
                errors.append(
                    f"component_weighted_curve_nrmse={weighted_curve_nrmse:.3f} exceeds "
                    f"{thresholds.max_component_weighted_curve_nrmse:.3f}"
                )
    if thresholds.max_component_curve_nrmse is not None:
        if result.component_recovery is None:
            errors.append("component recovery metrics are unavailable")
        else:
            matched_components = list(result.component_recovery.get("matched_components", []))
            if not matched_components:
                errors.append("matched component recovery metrics are unavailable")
            else:
                max_component_curve_nrmse = max(
                    float(component["curve_nrmse"]) for component in matched_components
                )
                if max_component_curve_nrmse > thresholds.max_component_curve_nrmse:
                    errors.append(
                        f"component_max_curve_nrmse={max_component_curve_nrmse:.3f} exceeds "
                        f"{thresholds.max_component_curve_nrmse:.3f}"
                    )
    if thresholds.max_component_effective_k_error is not None:
        if result.component_recovery is None:
            errors.append("component recovery metrics are unavailable")
        else:
            effective_k_error = float(result.component_recovery["effective_k_error"])
            if effective_k_error > thresholds.max_component_effective_k_error:
                errors.append(
                    f"component_effective_k_error={effective_k_error:.3f} exceeds "
                    f"{thresholds.max_component_effective_k_error:.3f}"
                )

    if thresholds.min_test_coverage_90 is not None:
        coverage = float(result.test_metrics["coverage_90"])
        if coverage < thresholds.min_test_coverage_90:
            errors.append(
                f"test_coverage_90={coverage:.3f} is below {thresholds.min_test_coverage_90:.3f}"
            )
    if thresholds.max_test_mape is not None:
        mape = float(result.test_metrics["mape"])
        if mape > thresholds.max_test_mape:
            errors.append(f"test_mape={mape:.3f} exceeds {thresholds.max_test_mape:.3f}")
    if thresholds.max_test_crps is not None:
        crps = float(result.test_metrics["crps"])
        if crps > thresholds.max_test_crps:
            errors.append(f"test_crps={crps:.3f} exceeds {thresholds.max_test_crps:.3f}")

    if thresholds.max_test_mu_mape is not None:
        if result.latent_test is None:
            errors.append("latent test metrics are unavailable")
        elif float(result.latent_test["mape"]) > thresholds.max_test_mu_mape:
            errors.append(
                f"test_mu_mape={result.latent_test['mape']:.3f} exceeds "
                f"{thresholds.max_test_mu_mape:.3f}"
            )
    if thresholds.max_test_mu_nrmse is not None:
        if result.latent_test is None:
            errors.append("latent test metrics are unavailable")
        elif float(result.latent_test["nrmse"]) > thresholds.max_test_mu_nrmse:
            errors.append(
                f"test_mu_nrmse={result.latent_test['nrmse']:.3f} exceeds "
                f"{thresholds.max_test_mu_nrmse:.3f}"
            )
    if thresholds.min_test_mu_coverage_90 is not None:
        if result.latent_test is None:
            errors.append("latent test metrics are unavailable")
        elif float(result.latent_test["coverage_90"]) < thresholds.min_test_mu_coverage_90:
            errors.append(
                f"test_mu_coverage_90={result.latent_test['coverage_90']:.3f} is below "
                f"{thresholds.min_test_mu_coverage_90:.3f}"
            )

    if thresholds.require_alpha_in_ci:
        alpha_in_ci = bool(
            result.parameter_recovery and result.parameter_recovery["alpha"]["in_ci"]
        )
        if not alpha_in_ci:
            errors.append("alpha_true is not within the posterior interval")
    if thresholds.require_sigma_in_ci:
        sigma_in_ci = bool(
            result.parameter_recovery and result.parameter_recovery["sigma"]["in_ci"]
        )
        if not sigma_in_ci:
            errors.append("sigma_true is not within the posterior interval")

    if thresholds.effective_k_bounds is not None:
        lower, upper = thresholds.effective_k_bounds
        effective_k = float(result.effective_k["effective_k_mean"])
        if not lower <= effective_k <= upper:
            errors.append(
                f"effective_k_mean={effective_k:.3f} is outside [{lower:.3f}, {upper:.3f}]"
            )

    if thresholds.max_pareto_k_bad is not None:
        pareto_bad = int(result.loo.get("pareto_k_bad", 0))
        if pareto_bad > thresholds.max_pareto_k_bad:
            errors.append(f"pareto_k_bad={pareto_bad} exceeds {thresholds.max_pareto_k_bad}")
    if thresholds.max_pareto_k_very_bad is not None:
        pareto_very_bad = int(result.loo.get("pareto_k_very_bad", 0))
        if pareto_very_bad > thresholds.max_pareto_k_very_bad:
            errors.append(
                f"pareto_k_very_bad={pareto_very_bad} exceeds {thresholds.max_pareto_k_very_bad}"
            )

    if errors:
        if diagnostic_status is None:
            diagnostic_status = evaluate_case_diagnostic_status(result)
        diagnostics_lines: list[str] = []
        diagnostics_lines.append(f"- publication_status={diagnostic_status['publication_status']}")
        diagnostics_lines.append(f"- sampler_status={diagnostic_status['sampler_status']}")
        diagnostics_lines.append(f"- mixing_status={diagnostic_status['mixing_status']}")
        diagnostics_lines.append(
            f"- interpretation_status={diagnostic_status['interpretation_status']}"
        )
        diagnostics_lines.append(f"- max_rhat={float(result.convergence['max_rhat']):.3f}")
        diagnostics_lines.append(f"- min_ess_bulk={float(result.convergence['min_ess_bulk']):.1f}")
        diagnostics_lines.append(f"- min_ess_tail={float(result.convergence['min_ess_tail']):.1f}")
        diagnostics_lines.append(
            f"- num_divergences={int(result.hmc_diagnostics['num_divergences'])}"
        )
        diagnostics_lines.append(f"- min_bfmi={float(result.hmc_diagnostics['min_bfmi']):.3f}")
        diagnostics_lines.append(
            f"- tree_depth_hits={int(result.hmc_diagnostics['tree_depth_hits'])}"
        )
        if result.label_invariant is not None:
            diagnostics_lines.append(
                f"- rhat_log_lik={float(result.label_invariant['rhat_log_lik']):.3f}"
            )
            diagnostics_lines.append(
                f"- label_invariant_max_rhat={float(result.label_invariant['max_rhat']):.3f}"
            )
            diagnostics_lines.append(
                f"- label_invariant_min_ess_bulk={float(result.label_invariant['min_ess_bulk']):.1f}"
            )
            diagnostics_lines.append(
                f"- label_invariant_min_ess_tail={float(result.label_invariant['min_ess_tail']):.1f}"
            )
        if result.relabeled is not None:
            diagnostics_lines.append(
                f"- relabeled_max_rhat={float(result.relabeled['max_rhat']):.3f}"
            )
            diagnostics_lines.append(
                f"- relabeled_min_ess_bulk={float(result.relabeled['min_ess_bulk']):.1f}"
            )
            diagnostics_lines.append(
                f"- relabeled_min_ess_tail={float(result.relabeled['min_ess_tail']):.1f}"
            )
        if result.label_switching is not None:
            diagnostics_lines.append(
                f"- switching_rate={float(result.label_switching['switching_rate']):.3f}"
            )
            diagnostics_lines.append(
                f"- n_unique_orderings={int(result.label_switching['n_unique_orderings'])}"
            )
        message = "\n".join(
            [f"{result.label} failed benchmark thresholds:"]
            + [f"- {e}" for e in errors]
            + ["Diagnostics:"]
            + diagnostics_lines
        )
        raise AssertionError(message)


def assert_comparison_passes(
    baseline: BenchmarkCaseResult,
    candidate: BenchmarkCaseResult,
    thresholds: ComparisonThresholds,
) -> dict[str, float]:
    """Raise if a model comparison fails its thresholds."""
    comparison = compare_case_results(baseline, candidate)
    errors: list[str] = []

    if thresholds.min_delta_loo is not None and comparison["delta_loo"] < thresholds.min_delta_loo:
        errors.append(
            f"delta_loo={comparison['delta_loo']:.3f} is below {thresholds.min_delta_loo:.3f}"
        )
    if (
        thresholds.max_delta_mape is not None
        and comparison["delta_test_mape"] > thresholds.max_delta_mape
    ):
        errors.append(
            f"delta_test_mape={comparison['delta_test_mape']:.3f} exceeds "
            f"{thresholds.max_delta_mape:.3f}"
        )
    if (
        thresholds.max_candidate_mape_ratio is not None
        and comparison["candidate_mape_ratio"] > thresholds.max_candidate_mape_ratio
    ):
        errors.append(
            f"candidate_mape_ratio={comparison['candidate_mape_ratio']:.3f} exceeds "
            f"{thresholds.max_candidate_mape_ratio:.3f}"
        )

    if errors:
        message = "\n".join(
            [f"{candidate.label} failed comparison thresholds against {baseline.label}:"]
            + [f"- {e}" for e in errors]
        )
        raise AssertionError(message)

    return comparison


def plot_observed_vs_predictive(result: BenchmarkCaseResult, output_path: str | Path) -> Path:
    """Save observed vs posterior predictive plot for one benchmark case."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_idx = np.arange(len(result.y_train))
    test_idx = np.arange(len(result.y_train), len(result.y_train) + len(result.y_test))
    full_idx = np.concatenate([train_idx, test_idx])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvline(train_idx[-1], color="0.5", linestyle="--", linewidth=1, label="Train/Test Split")
    ax.plot(
        full_idx[: len(result.y_train)],
        result.y_train,
        color="black",
        linewidth=1.5,
        label="Observed",
    )
    ax.plot(test_idx, result.y_test, color="black", linewidth=1.5)

    train_mean = np.asarray(result.train_metrics["y_pred_mean"])
    test_mean = np.asarray(result.test_metrics["y_pred_mean"])
    train_q05 = np.asarray(result.train_metrics["q05"])
    train_q95 = np.asarray(result.train_metrics["q95"])
    test_q05 = np.asarray(result.test_metrics["q05"])
    test_q95 = np.asarray(result.test_metrics["q95"])

    ax.plot(train_idx, train_mean, color="#1f77b4", linewidth=2, label="Posterior Mean")
    ax.fill_between(train_idx, train_q05, train_q95, color="#1f77b4", alpha=0.15)
    ax.plot(test_idx, test_mean, color="#1f77b4", linewidth=2)
    ax.fill_between(test_idx, test_q05, test_q95, color="#1f77b4", alpha=0.15, label="90% Interval")

    if result.meta is not None:
        latent_truth = _reference_latent_mean(result.meta)
        latent_label = _reference_latent_label(result.meta)
    else:
        latent_truth = None
        latent_label = "True Latent Mean"
    if latent_truth is not None:
        ax.plot(
            full_idx,
            latent_truth,
            color="#2ca02c",
            linestyle=":",
            linewidth=2,
            label=latent_label,
        )

    ax.set_title(f"{result.label}: Observed vs Posterior Predictive")
    ax.set_xlabel("Time")
    ax.set_ylabel("Response")
    ax.legend(loc="upper left", ncols=3)
    ax.text(
        0.99,
        0.02,
        (
            f"test CRPS={result.test_metrics['crps']:.3f}, "
            f"nRMSE={result.test_metrics['nrmse']:.3f}, "
            f"coverage={result.test_metrics['coverage_90']:.1%}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_response_curves(result: BenchmarkCaseResult, output_path: str | Path) -> Path:
    """Save an overlaid component-response plot for one benchmark case."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_full = np.concatenate([result.x_train, result.x_test])
    samples = result.samples
    if result.meta is not None and "s_max" in result.meta:
        grid_max = max(float(result.meta["s_max"]) * 1.1, 1.0)
        x_label = "Adstocked Spend"
    else:
        grid_max = max(float(np.max(x_full)) * 1.1, 1.0)
        x_label = "Spend (Proxy)"
    grid = np.linspace(0.0, grid_max, 200, dtype=np.float32)

    def _posterior_component_payload() -> tuple[np.ndarray, np.ndarray]:
        A = np.asarray(samples["A"], dtype=np.float32)
        k = np.asarray(samples["k"], dtype=np.float32)
        n = np.asarray(samples["n"], dtype=np.float32)
        if A.ndim == 1:
            A = A[:, None]
            k = k[:, None]
            n = n[:, None]
        if "pis" in samples:
            pis = np.asarray(samples["pis"], dtype=np.float32)
            if pis.ndim == 1:
                pis = pis[:, None]
        else:
            pis = np.ones((A.shape[0], 1), dtype=np.float32)

        grid_terms = grid[None, :, None] ** n[:, None, :]
        denom = k[:, None, :] ** n[:, None, :] + grid_terms + 1e-12
        component_curves = A[:, None, :] * grid_terms / denom
        return component_curves, pis

    posterior_component_curves, posterior_pis = _posterior_component_payload()
    posterior_weights = posterior_pis.mean(axis=0)
    posterior_means = posterior_component_curves.mean(axis=0)
    posterior_lowers = np.quantile(posterior_component_curves, 0.05, axis=0)
    posterior_uppers = np.quantile(posterior_component_curves, 0.95, axis=0)
    posterior_component_indices = list(range(posterior_component_curves.shape[-1]))
    posterior_component_active: dict[int, bool] = {idx: True for idx in posterior_component_indices}
    if result.component_summary is not None:
        posterior_component_active = {
            int(component["index"]): bool(component["active"])
            for component in result.component_summary["components"]
        }

    true_components = None
    true_weights = None
    true_component_indices: list[int] = []
    realized_effect_x = None
    realized_effect_y = None
    if result.meta is not None and {"A_true", "k_true", "n_true", "pi_true"}.issubset(result.meta):
        A_true = np.asarray(result.meta["A_true"], dtype=np.float32)
        k_true = np.asarray(result.meta["k_true"], dtype=np.float32)
        n_true = np.asarray(result.meta["n_true"], dtype=np.float32)
        true_weights = np.asarray(result.meta["pi_true"], dtype=np.float32)
        true_components = np.stack(
            [
                np.asarray(hill(grid, A_i, k_i, n_i), dtype=np.float32)
                for A_i, k_i, n_i in zip(A_true, k_true, n_true, strict=True)
            ],
            axis=0,
        )
        true_component_indices = list(range(true_components.shape[0]))
        if {"s", "baseline"}.issubset(result.meta):
            realized_effect_x = np.asarray(result.meta["s"], dtype=np.float32)
            baseline = np.asarray(result.meta["baseline"], dtype=np.float32)
            y_full = np.concatenate([result.y_train, result.y_test])
            if len(realized_effect_x) == len(y_full) == len(baseline):
                realized_effect_y = y_full - baseline
            else:
                realized_effect_x = None

    panel_specs: list[dict[str, Any]] = []
    used_true: set[int] = set()
    used_posterior: set[int] = set()
    matched_components = []
    if result.component_recovery is not None:
        matched_components = list(result.component_recovery.get("matched_components", []))

    for match in sorted(
        matched_components,
        key=lambda item: int(
            item.get("true_component_index", item.get("reference_component_index", -1))
        ),
    ):
        true_idx = int(match.get("true_component_index", match.get("reference_component_index")))
        posterior_idx = int(
            match.get("posterior_component_index", match.get("candidate_component_index"))
        )
        panel_specs.append(
            {
                "true_idx": true_idx,
                "posterior_idx": posterior_idx,
                "match_metrics": match,
            }
        )
        used_true.add(true_idx)
        used_posterior.add(posterior_idx)

    if true_components is not None and not panel_specs:
        paired = min(len(true_component_indices), len(posterior_component_indices))
        for idx in range(paired):
            true_idx = true_component_indices[idx]
            posterior_idx = posterior_component_indices[idx]
            panel_specs.append(
                {
                    "true_idx": true_idx,
                    "posterior_idx": posterior_idx,
                    "match_metrics": None,
                }
            )
            used_true.add(true_idx)
            used_posterior.add(posterior_idx)

    for true_idx in true_component_indices:
        if true_idx not in used_true:
            panel_specs.append({"true_idx": true_idx, "posterior_idx": None, "match_metrics": None})
    for posterior_idx in posterior_component_indices:
        if posterior_idx not in used_posterior:
            panel_specs.append(
                {"true_idx": None, "posterior_idx": posterior_idx, "match_metrics": None}
            )
    if not panel_specs:
        panel_specs.append({"true_idx": None, "posterior_idx": 0, "match_metrics": None})

    all_y_max = [float(np.max(posterior_uppers[:, idx])) for idx in posterior_component_indices]
    if true_components is not None:
        all_y_max.extend(float(np.max(true_components[idx])) for idx in true_component_indices)
    if realized_effect_y is not None:
        all_y_max.append(float(np.max(realized_effect_y)))
    y_max = max(all_y_max, default=1.0) * 1.08

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    color_by_true: dict[int, str] = {}
    color_by_posterior: dict[int, str] = {}

    if realized_effect_x is not None and realized_effect_y is not None:
        ax.scatter(
            realized_effect_x,
            realized_effect_y,
            s=22,
            color="0.55",
            alpha=0.45,
            linewidths=0.0,
            label="Observations",
            zorder=1,
        )

    color_index = 0
    for spec in panel_specs:
        color = color_cycle[color_index % len(color_cycle)]
        if spec["true_idx"] is not None and spec["true_idx"] not in color_by_true:
            color_by_true[spec["true_idx"]] = color
        if spec["posterior_idx"] is not None and spec["posterior_idx"] not in color_by_posterior:
            color_by_posterior[spec["posterior_idx"]] = color
        if spec["true_idx"] is not None or spec["posterior_idx"] is not None:
            color_index += 1

    for posterior_idx in posterior_component_indices:
        color = color_by_posterior.get(
            posterior_idx,
            color_cycle[posterior_idx % len(color_cycle)],
        )
        is_active = posterior_component_active.get(posterior_idx, True)
        label_suffix = f" (π={posterior_weights[posterior_idx]:.2f})"
        if not is_active:
            label_suffix += ", inactive"
        ax.fill_between(
            grid,
            posterior_lowers[:, posterior_idx],
            posterior_uppers[:, posterior_idx],
            color=color,
            alpha=0.14 if is_active else 0.07,
        )
        ax.plot(
            grid,
            posterior_means[:, posterior_idx],
            color=color,
            linewidth=2.2,
            alpha=1.0 if is_active else 0.55,
            label=f"Estimated {posterior_idx + 1}{label_suffix}",
        )

    if true_components is not None:
        for true_idx in true_component_indices:
            color = color_by_true.get(true_idx, color_cycle[true_idx % len(color_cycle)])
            label_suffix = ""
            if true_weights is not None:
                label_suffix = f" (π={true_weights[true_idx]:.2f})"
            ax.plot(
                grid,
                true_components[true_idx],
                color=color,
                linewidth=2.2,
                linestyle="--",
                label=f"True {true_idx + 1}{label_suffix}",
            )

    annotation_lines = []
    if result.component_recovery is not None:
        annotation_lines.extend(
            [
                f"weighted curve nRMSE={float(result.component_recovery['weighted_curve_nrmse']):.2f}",
                f"effective K error={float(result.component_recovery['effective_k_error']):.2f}",
                f"unmatched true weight={float(result.component_recovery['unmatched_reference_weight']):.2f}",
                f"unmatched posterior weight={float(result.component_recovery['unmatched_candidate_weight']):.2f}",
            ]
        )
    if annotation_lines:
        ax.text(
            0.99,
            0.02,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    ax.set_title(f"{result.label}: Response Curve Recovery")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Response (y − baseline)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, grid_max)
    ax.set_ylim(0.0, y_max)
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_case_comparison(
    results: Sequence[BenchmarkCaseResult],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save comparison plot across benchmark cases."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [result.model_name for result in results]
    loo_values = [float(result.loo.get("elpd_loo", np.nan)) for result in results]
    crps_values = [float(result.test_metrics["crps"]) for result in results]
    latent_nrmse_values = [
        float((result.latent_test or {}).get("nrmse", result.test_metrics["nrmse"]))
        for result in results
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    metrics = [
        ("ELPD-LOO", loo_values, "#1f77b4"),
        ("Test CRPS", crps_values, "#ff7f0e"),
        ("Latent/Test nRMSE", latent_nrmse_values, "#2ca02c"),
    ]

    for ax, (metric_name, values, color) in zip(axes, metrics, strict=True):
        ax.bar(labels, values, color=color, alpha=0.85)
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=20)
        if "Coverage" in metric_name:
            ax.axhline(0.90, color="0.4", linestyle="--", linewidth=1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def resolve_case_artifact_dir(output_root: str | Path, result: BenchmarkCaseResult) -> Path:
    """Return the artifact directory for one benchmark case."""
    return Path(output_root) / result.domain / result.model_name


def resolve_comparison_artifact_dir(output_root: str | Path, domain: str) -> Path:
    """Return the artifact directory for comparison plots within one domain."""
    return Path(output_root) / domain / "_comparisons"


def save_case_artifacts(result: BenchmarkCaseResult, output_root: str | Path) -> dict[str, Path]:
    """Write a summary JSON and default plots for one benchmark case."""
    output_dir = resolve_case_artifact_dir(output_root, result)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{result.label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(case_summary(result), fh, indent=2)

    predictive_path = plot_observed_vs_predictive(
        result, output_dir / f"{result.label}_predictive.png"
    )
    response_path = plot_response_curves(result, output_dir / f"{result.label}_response.png")

    return {
        "summary": summary_path,
        "predictive": predictive_path,
        "response": response_path,
    }
