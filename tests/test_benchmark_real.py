"""Opt-in real-data benchmark tests with pass/fail quality gates.

Structure mirrors test_benchmark_synthetic.py:
  - parametrised matrix over (dataset_label × model_name × seed)
  - smoke / full tiers gated by environment variables
  - across-seed component stability test (full only)

Dataset selection is kept separate from test logic.  To activate the
benchmarks, populate ``REAL_DATA_CONFIGS`` with one or more entries
mapping a short label to a ``TimeSeriesConfig``.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    BenchmarkThresholds,
    assert_case_passes,
    run_real_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data_loader import TimeSeriesConfig
from hill_mixture_mmm.metrics import compute_across_seed_component_stability

# ---------------------------------------------------------------------------
# Data configuration — fill in when dataset selection is finalised.
#
# Each entry maps a short label (used in test IDs and artifact paths) to a
# ``TimeSeriesConfig``.  The test matrix parametrises over these labels, so
# adding / removing entries automatically adjusts the test surface.
#
# Example:
#   REAL_DATA_CONFIGS = {
#       "retail_001": TimeSeriesConfig(
#           organisation_id="72a86a208d24d68b80be0e44a8a4872d",
#           target_col="all_purchases",
#           aggregate_spend=True,
#           min_series_length=200,
#       ),
#       "fashion_002": TimeSeriesConfig(
#           organisation_id="...",
#           target_col="all_purchases",
#           aggregate_spend=True,
#           min_series_length=200,
#       ),
#   }
# ---------------------------------------------------------------------------
REAL_DATA_CONFIGS: dict[str, TimeSeriesConfig] = {
    "toys_hobbies": TimeSeriesConfig(
        organisation_id="200b4005ea8477754ef37438e66b2f4c",
        target_col="all_purchases",
        aggregate_spend=True,
        min_series_length=200,
    ),
    "beauty_fitness": TimeSeriesConfig(
        organisation_id="8cfe3c2df337e1e53d19b8dc4b878cac",
        target_col="all_purchases",
        aggregate_spend=True,
        min_series_length=200,
    ),
    "home_garden": TimeSeriesConfig(
        organisation_id="7059e30b528ed5f14ee9921de13248e5",
        target_col="all_purchases",
        aggregate_spend=True,
        min_series_length=200,
    ),
}

CSV_PATH = Path(__file__).parent.parent / "data" / "conjura_mmm_data.csv"

# ---------------------------------------------------------------------------
# Model / seed axes
# ---------------------------------------------------------------------------
SMOKE_MODEL_NAMES = ["single_hill", "mixture_k2"]
FULL_MODEL_NAMES = ["single_hill", "mixture_k2", "mixture_k3"]

SMOKE_SEEDS = [0]
FULL_SEEDS = [0, 1, 2]

# Derived label list — falls back to a dummy so parametrize doesn't error on
# an empty sequence; the test body skips immediately via _skip_if_no_configs.
_DATASET_LABELS: list[str] = sorted(REAL_DATA_CONFIGS) or ["__no_data_configured__"]


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------


def _skip_if_no_configs() -> None:
    if not REAL_DATA_CONFIGS:
        pytest.skip(
            "no REAL_DATA_CONFIGS defined yet — populate the dict to run real-data benchmarks"
        )


def _require_full_real_benchmark() -> None:
    enabled = os.getenv("HILL_MMM_RUN_FULL_REAL_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full real-data benchmark is opt-in; set HILL_MMM_RUN_FULL_REAL_BENCHMARK=1 to run it"
        )


# ---------------------------------------------------------------------------
# Run configuration (per-model tuning, same pattern as synthetic)
# ---------------------------------------------------------------------------


def _real_run_config(model_name: str, seed: int) -> BenchmarkRunConfig:
    num_chains = int(os.getenv("HILL_MMM_REAL_CHAINS", "2"))
    if model_name == "mixture_k2":
        target_accept_prob = 0.95
    elif model_name == "mixture_k3":
        target_accept_prob = 0.97
    else:
        target_accept_prob = 0.90

    init_strategy = "median" if model_name in {"mixture_k2", "mixture_k3"} else "uniform"

    if model_name == "single_hill":
        warmup = 600
        samples = warmup
    elif model_name == "mixture_k3":
        warmup = 1600
        samples = 1200
    else:  # mixture_k2
        warmup = 900
        samples = warmup

    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
        dense_mass=False,
        init_strategy=init_strategy,
        progress_bar=False,
    )


# ---------------------------------------------------------------------------
# Thresholds — real data has no ground truth; only predictive & diagnostic
# gates apply.
# ---------------------------------------------------------------------------

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


def _real_thresholds(model_name: str) -> BenchmarkThresholds:
    """Full-run thresholds.  Adjust once baseline results are available."""
    overrides: dict = {"require_reportable_diagnostics": True}
    overrides["max_test_mape"] = 30.0
    if model_name in {"mixture_k2", "mixture_k3"}:
        overrides["max_test_mu_nrmse"] = 0.30
    return replace(_RELAXED_BASE, **overrides)


def _real_smoke_thresholds(model_name: str) -> BenchmarkThresholds:
    """Smoke thresholds — loose; just ensure the pipeline runs end-to-end."""
    del model_name
    return replace(_RELAXED_BASE, max_test_mape=50.0)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_and_assert_real_case(
    dataset_label: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
    thresholds: BenchmarkThresholds,
) -> None:
    ts_config = REAL_DATA_CONFIGS[dataset_label]
    result = run_real_benchmark_case(
        csv_path=CSV_PATH,
        timeseries_config=ts_config,
        model_name=model_name,
        config=_real_run_config(model_name, seed),
        label=f"real_{dataset_label}_{model_name}_seed{seed}",
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    assert_case_passes(result, thresholds)

    for path in artifacts.values():
        assert path.exists(), f"Expected real-data artifact at {path}"


def _case_summary_path(
    benchmark_output_root: Path,
    dataset_label: str,
    model_name: str,
    seed: int,
) -> Path:
    org_id = REAL_DATA_CONFIGS[dataset_label].organisation_id
    return (
        benchmark_output_root
        / "real"
        / str(org_id)
        / f"real_{dataset_label}_{model_name}_seed{seed}_summary.json"
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark_smoke
@pytest.mark.parametrize("seed", SMOKE_SEEDS)
@pytest.mark.parametrize("dataset_label", _DATASET_LABELS)
@pytest.mark.parametrize("model_name", SMOKE_MODEL_NAMES)
def test_real_benchmark_smoke_matrix(
    dataset_label: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _skip_if_no_configs()
    _run_and_assert_real_case(
        dataset_label,
        model_name,
        seed,
        benchmark_output_root,
        _real_smoke_thresholds(model_name),
    )


# ---------------------------------------------------------------------------
# Full tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("seed", FULL_SEEDS)
@pytest.mark.parametrize("dataset_label", _DATASET_LABELS)
@pytest.mark.parametrize("model_name", FULL_MODEL_NAMES)
def test_real_benchmark_full_matrix(
    dataset_label: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _skip_if_no_configs()
    _require_full_real_benchmark()
    _run_and_assert_real_case(
        dataset_label,
        model_name,
        seed,
        benchmark_output_root,
        _real_thresholds(model_name),
    )


# ---------------------------------------------------------------------------
# Across-seed stability (full only)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("dataset_label", _DATASET_LABELS)
@pytest.mark.parametrize("model_name", FULL_MODEL_NAMES)
def test_real_benchmark_full_across_seed_stability(
    dataset_label: str,
    model_name: str,
    benchmark_output_root: Path,
) -> None:
    _skip_if_no_configs()
    _require_full_real_benchmark()

    summaries = []
    for seed in FULL_SEEDS:
        summary_path = _case_summary_path(
            benchmark_output_root,
            dataset_label,
            model_name,
            seed,
        )
        assert summary_path.exists(), f"Expected benchmark summary at {summary_path}"
        with summary_path.open("r", encoding="utf-8") as fh:
            summaries.append(json.load(fh))

    stability = compute_across_seed_component_stability(summaries)

    org_id = REAL_DATA_CONFIGS[dataset_label].organisation_id
    stability_path = (
        benchmark_output_root
        / "real"
        / str(org_id)
        / f"real_{dataset_label}_{model_name}_across_seed_stability.json"
    )
    with stability_path.open("w", encoding="utf-8") as fh:
        json.dump(stability, fh, indent=2)

    assert stability["num_seeds"] == len(FULL_SEEDS)
    expected_pairs = (len(FULL_SEEDS) * (len(FULL_SEEDS) - 1)) // 2
    assert stability["pair_count"] == expected_pairs
    assert stability_path.exists()
