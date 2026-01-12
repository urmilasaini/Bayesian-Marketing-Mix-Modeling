from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from time import time

import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    BenchmarkThresholds,
    assert_case_passes,
    run_synthetic_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data import DGPConfig
from hill_mixture_mmm.metrics import compute_across_seed_component_stability
from hill_mixture_mmm.paper_figures import (
    DEFAULT_FIGURE_IDS,
    generate_publication_figures,
    load_synthetic_results_from_artifacts,
)

SMOKE_DGP_NAMES = ["single", "mixture_k2"]
SMOKE_MODEL_NAMES = ["single_hill", "mixture_k2"]
FULL_DGP_NAMES = ["single", "mixture_k2", "mixture_k3"]
FULL_MODEL_NAMES = ["single_hill", "mixture_k2", "mixture_k3"]

SMOKE_SYNTHETIC_SEEDS = [0]
FULL_SYNTHETIC_SEEDS = [0, 1, 2, 3, 4]
FULL_SYNTHETIC_EXPECTED_CASES = (
    len(FULL_DGP_NAMES) * len(FULL_MODEL_NAMES) * len(FULL_SYNTHETIC_SEEDS)
)
PAPER_FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"


def _require_full_synthetic_benchmark() -> None:
    enabled = os.getenv("HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full synthetic benchmark is opt-in; set "
            "HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK=1 to run it"
        )


@pytest.fixture(scope="module", autouse=True)
def _generate_selected_publication_figures_after_full_benchmark() -> None:
    started_at = time()
    yield

    enabled = os.getenv("HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return

    summary_paths = sorted(
        path
        for path in PAPER_FIGURES_DIR.glob("synthetic/*/*_seed*_summary.json")
        if path.stat().st_mtime >= started_at - 1.0
    )
    if not summary_paths:
        return

    results = load_synthetic_results_from_artifacts(
        PAPER_FIGURES_DIR,
        summary_paths=summary_paths,
    )
    if len(results) != FULL_SYNTHETIC_EXPECTED_CASES:
        return

    generated = generate_publication_figures(
        output_dir=PAPER_FIGURES_DIR,
        artifact_root=PAPER_FIGURES_DIR,
        summary_paths=summary_paths,
        figure_ids=DEFAULT_FIGURE_IDS,
    )
    for path in generated.values():
        assert path.exists(), f"Expected publication figure at {path}"


def _synthetic_run_config(dgp_name: str, model_name: str, seed: int) -> BenchmarkRunConfig:
    num_chains = int(os.getenv("HILL_MMM_SYNTHETIC_CHAINS", "2"))
    if model_name == "mixture_k2":
        target_accept_prob = 0.95
    elif model_name == "mixture_k3":
        target_accept_prob = 0.97
    else:
        target_accept_prob = 0.90
    dense_mass = False
    init_strategy = "median" if model_name in {"mixture_k2", "mixture_k3"} else "uniform"
    if model_name == "single_hill":
        warmup = 800 if dgp_name == "mixture_k3" else 600
        samples = warmup
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=warmup,
            num_samples=samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            dense_mass=dense_mass,
            init_strategy=init_strategy,
            progress_bar=False,
        )

    if model_name == "mixture_k3":
        warmup = 1600
        samples = 1200
    else:
        warmup = 900
        samples = warmup

    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        init_strategy=init_strategy,
        progress_bar=False,
    )


_RELAXED_BASE = BenchmarkThresholds(
    max_rhat=None,
    min_ess_bulk_per_chain=None,
    min_ess_tail_per_chain=None,
    max_divergences=None,
    min_bfmi=None,
    max_tree_depth_hits=None,
    require_finite_loo_waic=True,
    require_finite_predictive_metrics=True,
)


def _synthetic_thresholds(dgp_name: str, model_name: str) -> BenchmarkThresholds:
    overrides: dict = {"require_reportable_diagnostics": True, "require_truth_metrics": True}
    if dgp_name == "single":
        overrides["max_test_mape"] = 5.0
    elif dgp_name == "mixture_k2":
        overrides["max_test_mu_nrmse"] = 0.15
    if dgp_name == "mixture_k3" and model_name == "mixture_k3":
        overrides["max_component_weighted_curve_nrmse"] = 0.15
        overrides["max_component_curve_nrmse"] = 0.25
        overrides["max_component_effective_k_error"] = 0.0
    return replace(_RELAXED_BASE, **overrides)


def _synthetic_smoke_thresholds(dgp_name: str, model_name: str) -> BenchmarkThresholds:
    del model_name
    overrides: dict = {"require_truth_metrics": True}
    if dgp_name == "single":
        overrides["max_test_mape"] = 5.0
    elif dgp_name == "mixture_k2":
        overrides["max_test_mu_nrmse"] = 0.18
    return replace(_RELAXED_BASE, **overrides)


def _run_and_assert_case(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
    thresholds: BenchmarkThresholds,
) -> None:
    result = run_synthetic_benchmark_case(
        dgp_config=DGPConfig(dgp_type=dgp_name, T=200, seed=seed),
        model_name=model_name,
        config=_synthetic_run_config(dgp_name, model_name, seed),
        label=f"synthetic_{dgp_name}_{model_name}_seed{seed}",
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    assert_case_passes(result, thresholds)

    for path in artifacts.values():
        assert path.exists(), f"Expected synthetic artifact at {path}"


def _case_summary_path(
    benchmark_output_root: Path,
    dgp_name: str,
    model_name: str,
    seed: int,
) -> Path:
    return (
        benchmark_output_root
        / "synthetic"
        / model_name
        / f"synthetic_{dgp_name}_{model_name}_seed{seed}_summary.json"
    )


@pytest.mark.slow
@pytest.mark.benchmark_smoke
@pytest.mark.parametrize("seed", SMOKE_SYNTHETIC_SEEDS)
@pytest.mark.parametrize("dgp_name", SMOKE_DGP_NAMES)
@pytest.mark.parametrize("model_name", SMOKE_MODEL_NAMES)
def test_synthetic_benchmark_smoke_matrix(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _run_and_assert_case(
        dgp_name,
        model_name,
        seed,
        benchmark_output_root,
        _synthetic_smoke_thresholds(dgp_name, model_name),
    )


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("seed", FULL_SYNTHETIC_SEEDS)
@pytest.mark.parametrize("dgp_name", FULL_DGP_NAMES)
@pytest.mark.parametrize("model_name", FULL_MODEL_NAMES)
def test_synthetic_benchmark_full_matrix(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _require_full_synthetic_benchmark()
    _run_and_assert_case(
        dgp_name,
        model_name,
        seed,
        benchmark_output_root,
        _synthetic_thresholds(dgp_name, model_name),
    )


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("dgp_name", FULL_DGP_NAMES)
@pytest.mark.parametrize("model_name", FULL_MODEL_NAMES)
def test_synthetic_benchmark_full_across_seed_stability(
    dgp_name: str,
    model_name: str,
    benchmark_output_root: Path,
) -> None:
    _require_full_synthetic_benchmark()

    summaries = []
    for seed in FULL_SYNTHETIC_SEEDS:
        summary_path = _case_summary_path(benchmark_output_root, dgp_name, model_name, seed)
        assert summary_path.exists(), f"Expected benchmark summary at {summary_path}"
        with summary_path.open("r", encoding="utf-8") as fh:
            summaries.append(json.load(fh))

    stability = compute_across_seed_component_stability(summaries)
    stability_path = (
        benchmark_output_root
        / "synthetic"
        / model_name
        / f"synthetic_{dgp_name}_{model_name}_across_seed_stability.json"
    )
    with stability_path.open("w", encoding="utf-8") as fh:
        json.dump(stability, fh, indent=2)

    assert stability["num_seeds"] == len(FULL_SYNTHETIC_SEEDS)
    assert (
        stability["pair_count"]
        == (len(FULL_SYNTHETIC_SEEDS) * (len(FULL_SYNTHETIC_SEEDS) - 1)) // 2
    )
    assert stability_path.exists(), f"Expected across-seed stability summary at {stability_path}"
