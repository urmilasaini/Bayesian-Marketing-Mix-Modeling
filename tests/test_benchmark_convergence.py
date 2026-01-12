from __future__ import annotations

import numpy as np
import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkCaseResult,
    BenchmarkThresholds,
    ComparisonThresholds,
    assert_case_passes,
    assert_comparison_passes,
    compare_case_results,
    evaluate_case_diagnostic_status,
)
from hill_mixture_mmm.inference import compute_hmc_diagnostics


class _DummyMCMC:
    def __init__(self, extra_fields: dict[str, np.ndarray]):
        self._extra_fields = extra_fields

    def get_extra_fields(self, group_by_chain: bool = False):
        assert group_by_chain
        return self._extra_fields


def _make_mixture_result(
    *,
    converged: bool = True,
    label_invariant_max_rhat: float = 1.0,
    label_invariant_min_ess: float = 400.0,
    relabeled_max_rhat: float = 1.0,
    relabeled_min_ess: float = 400.0,
    num_divergences: int = 0,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        label="synthetic_case",
        domain="synthetic",
        dataset_name="mixture_k3",
        model_name="mixture_k3",
        seed=0,
        train_ratio=0.75,
        x_train=np.array([1.0], dtype=np.float32),
        x_test=np.array([1.0], dtype=np.float32),
        y_train=np.array([1.0], dtype=np.float32),
        y_test=np.array([1.0], dtype=np.float32),
        dates_train=None,
        dates_test=None,
        train_metrics={
            "mape": 1.0,
            "rmse": 0.1,
            "nrmse": 0.1,
            "crps": 0.1,
            "coverage_90": 0.9,
            "y_pred_mean": np.array([1.0]),
        },
        test_metrics={
            "mape": 1.0,
            "rmse": 0.1,
            "nrmse": 0.1,
            "crps": 0.1,
            "coverage_90": 0.9,
            "y_pred_mean": np.array([1.0]),
        },
        loo={"elpd_loo": -1.0, "se": 0.1, "pareto_k_bad": 0, "pareto_k_very_bad": 0},
        waic={"elpd_waic": -1.0, "se": 0.1},
        convergence={
            "max_rhat": 1.0,
            "min_ess_bulk": 400.0,
            "min_ess_tail": 400.0,
            "converged": True,
            "ess_sufficient": True,
        },
        hmc_diagnostics={
            "num_divergences": num_divergences,
            "has_divergence": num_divergences > 0,
            "bfmi_by_chain": [0.8, 0.9],
            "min_bfmi": 0.8,
            "bfmi_ok": True,
            "max_tree_depth": 10,
            "tree_depth_hits": 0,
            "max_tree_depth_hit": False,
            "max_num_steps": 16,
            "mean_accept_prob": 0.95,
        },
        label_invariant={
            "rhat_log_lik": 1.0,
            "rhat_scalars": {"sigma": 1.0},
            "ess_bulk_log_lik": label_invariant_min_ess,
            "ess_tail_log_lik": label_invariant_min_ess,
            "ess_bulk_scalars": {"sigma": label_invariant_min_ess},
            "ess_tail_scalars": {"sigma": label_invariant_min_ess},
            "min_ess_bulk": label_invariant_min_ess,
            "min_ess_tail": label_invariant_min_ess,
            "max_rhat": label_invariant_max_rhat,
            "converged": label_invariant_max_rhat < 1.01,
            "method": "rank",
            "threshold": 1.01,
        },
        relabeled={
            "component_rhats": {
                "A": {"per_component": [relabeled_max_rhat], "max": relabeled_max_rhat}
            },
            "component_ess_bulk": {
                "A": {"per_component": [relabeled_min_ess], "min": relabeled_min_ess}
            },
            "component_ess_tail": {
                "A": {"per_component": [relabeled_min_ess], "min": relabeled_min_ess}
            },
            "max_rhat": relabeled_max_rhat,
            "min_ess_bulk": relabeled_min_ess,
            "min_ess_tail": relabeled_min_ess,
            "converged": relabeled_max_rhat < 1.01,
            "method": "rank",
            "threshold": 1.01,
        },
        label_switching={
            "switching_rate": 0.0,
            "n_unique_orderings": 1,
            "mode_ordering": [0, 1, 2],
            "mode_count": 10,
            "top_orderings": [([0, 1, 2], 10)],
        },
        component_summary=None,
        component_recovery=None,
        converged=converged,
        effective_k={"effective_k_mean": 2.5, "effective_k_std": 0.1},
        parameter_recovery=None,
        latent_train=None,
        latent_test=None,
        meta=None,
        samples={},
        fit_summary={
            "inference_seed": 0,
            "num_warmup_used": 200,
            "num_samples_used": 200,
            "num_chains_used": 2,
            "target_accept_prob_used": 0.9,
            "max_tree_depth_used": 10,
        },
    )


def _reportability_thresholds(*, require_truth_metrics: bool = False) -> BenchmarkThresholds:
    return BenchmarkThresholds(
        max_rhat=None,
        min_ess_bulk=None,
        min_ess_tail=None,
        min_ess_bulk_per_chain=None,
        min_ess_tail_per_chain=None,
        max_label_invariant_rhat=None,
        min_label_invariant_ess_bulk_per_chain=None,
        min_label_invariant_ess_tail_per_chain=None,
        max_relabeled_rhat=None,
        min_relabeled_ess_bulk_per_chain=None,
        min_relabeled_ess_tail_per_chain=None,
        max_divergences=None,
        min_bfmi=None,
        max_tree_depth_hits=None,
        min_test_coverage_90=None,
        max_test_mape=None,
        max_test_mu_mape=None,
        min_test_mu_coverage_90=None,
        require_alpha_in_ci=False,
        require_sigma_in_ci=False,
        effective_k_bounds=None,
        max_pareto_k_bad=None,
        max_pareto_k_very_bad=None,
        require_reportable_diagnostics=True,
        require_truth_metrics=require_truth_metrics,
    )


def test_compute_hmc_diagnostics_reports_bfmi_and_tree_depth_hits():
    mcmc = _DummyMCMC(
        {
            "diverging": np.array(
                [
                    [False, False, False, False, False, True, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False, False, False],
                ]
            ),
            "energy": np.array(
                [
                    np.linspace(0.0, 10.0, 11),
                    np.linspace(0.0, 5.0, 11),
                ]
            ),
            "num_steps": np.array(
                [
                    [16, 16, 16, 16, 16, 512, 16, 16, 16, 16, 16],
                    [4, 8, 16, 8, 4, 8, 16, 8, 4, 8, 16],
                ]
            ),
            "accept_prob": np.array(
                [
                    [0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9],
                    [0.95, 0.95, 0.9, 0.9, 0.95, 0.95, 0.9, 0.9, 0.95, 0.95, 0.9],
                ]
            ),
        }
    )

    diagnostics = compute_hmc_diagnostics(mcmc, max_tree_depth=10)

    assert diagnostics["num_divergences"] == 1
    assert diagnostics["tree_depth_hits"] == 1
    assert diagnostics["max_num_steps"] == 512
    assert diagnostics["min_bfmi"] < 0.3


def test_evaluate_case_diagnostic_status_warns_on_marginal_label_invariant_rhat():
    result = _make_mixture_result(converged=False, label_invariant_max_rhat=1.02)
    status = evaluate_case_diagnostic_status(result)

    assert status["publication_status"] == "Warn"
    assert "label_invariant_max_rhat=1.020 exceeds pass threshold 1.010" in status["warnings"]


def test_evaluate_case_diagnostic_status_warns_on_borderline_divergences():
    result = _make_mixture_result(num_divergences=3)
    status = evaluate_case_diagnostic_status(result)

    assert status["publication_status"] == "Warn"
    assert status["sampler_status"] == "Warn"
    assert "num_divergences=3.000 exceeds pass threshold 0.000" in status["warnings"]


def test_assert_case_passes_uses_fail_thresholds_for_label_invariant_rhat():
    warning_result = _make_mixture_result(converged=False, label_invariant_max_rhat=1.02)
    assert_case_passes(
        warning_result,
        BenchmarkThresholds(
            max_rhat=None,
            min_ess_bulk=None,
            min_ess_tail=None,
            min_ess_bulk_per_chain=None,
            min_ess_tail_per_chain=None,
            max_label_invariant_rhat=1.05,
            min_label_invariant_ess_bulk_per_chain=50.0,
            min_label_invariant_ess_tail_per_chain=50.0,
            max_relabeled_rhat=None,
            min_relabeled_ess_bulk_per_chain=None,
            min_relabeled_ess_tail_per_chain=None,
            max_divergences=5,
            min_bfmi=0.2,
            max_tree_depth_hits=10,
        ),
    )

    failing_result = _make_mixture_result(converged=False, label_invariant_max_rhat=1.06)
    with pytest.raises(AssertionError, match="label_invariant_max_rhat=1.060 exceeds 1.050"):
        assert_case_passes(
            failing_result,
            BenchmarkThresholds(
                max_rhat=None,
                min_ess_bulk=None,
                min_ess_tail=None,
                min_ess_bulk_per_chain=None,
                min_ess_tail_per_chain=None,
                max_label_invariant_rhat=1.05,
                min_label_invariant_ess_bulk_per_chain=50.0,
                min_label_invariant_ess_tail_per_chain=50.0,
                max_relabeled_rhat=None,
                min_relabeled_ess_bulk_per_chain=None,
                min_relabeled_ess_tail_per_chain=None,
                max_divergences=5,
                min_bfmi=0.2,
                max_tree_depth_hits=10,
            ),
        )


def test_assert_case_passes_allows_small_divergence_counts_but_fails_above_five():
    warning_result = _make_mixture_result(num_divergences=3)
    assert_case_passes(
        warning_result,
        BenchmarkThresholds(
            max_rhat=None,
            min_ess_bulk=None,
            min_ess_tail=None,
            min_ess_bulk_per_chain=None,
            min_ess_tail_per_chain=None,
            max_label_invariant_rhat=1.05,
            min_label_invariant_ess_bulk_per_chain=50.0,
            min_label_invariant_ess_tail_per_chain=50.0,
            max_relabeled_rhat=None,
            min_relabeled_ess_bulk_per_chain=None,
            min_relabeled_ess_tail_per_chain=None,
            max_divergences=5,
            min_bfmi=0.2,
            max_tree_depth_hits=10,
        ),
    )

    failing_result = _make_mixture_result(num_divergences=6)
    with pytest.raises(AssertionError, match="num_divergences=6 exceeds 5"):
        assert_case_passes(
            failing_result,
            BenchmarkThresholds(
                max_rhat=None,
                min_ess_bulk=None,
                min_ess_tail=None,
                min_ess_bulk_per_chain=None,
                min_ess_tail_per_chain=None,
                max_label_invariant_rhat=1.05,
                min_label_invariant_ess_bulk_per_chain=50.0,
                min_label_invariant_ess_tail_per_chain=50.0,
                max_relabeled_rhat=None,
                min_relabeled_ess_bulk_per_chain=None,
                min_relabeled_ess_tail_per_chain=None,
                max_divergences=5,
                min_bfmi=0.2,
                max_tree_depth_hits=10,
            ),
        )


def test_assert_case_passes_can_gate_on_publication_status() -> None:
    warning_result = _make_mixture_result(num_divergences=3)
    assert_case_passes(warning_result, _reportability_thresholds())

    failing_result = _make_mixture_result(label_invariant_max_rhat=1.06)
    with pytest.raises(AssertionError, match="publication_status=Fail"):
        assert_case_passes(failing_result, _reportability_thresholds())


def test_assert_case_passes_rejects_nonfinite_predictive_metrics() -> None:
    result = _make_mixture_result()
    result.test_metrics["mape"] = np.nan

    with pytest.raises(AssertionError, match="test_metrics.mape is not finite"):
        assert_case_passes(result, _reportability_thresholds())


def test_assert_case_passes_requires_synthetic_truth_metrics_when_requested() -> None:
    result = _make_mixture_result()

    with pytest.raises(AssertionError, match="latent test metrics are unavailable"):
        assert_case_passes(result, _reportability_thresholds(require_truth_metrics=True))


def test_assert_case_passes_enforces_max_test_mape() -> None:
    passing_result = _make_mixture_result()
    passing_result.test_metrics["mape"] = 4.9
    assert_case_passes(passing_result, BenchmarkThresholds(max_test_mape=5.0))

    failing_result = _make_mixture_result()
    failing_result.test_metrics["mape"] = 5.1
    with pytest.raises(AssertionError, match="test_mape=5.100 exceeds 5.000"):
        assert_case_passes(failing_result, BenchmarkThresholds(max_test_mape=5.0))


def test_assert_case_passes_enforces_max_test_crps() -> None:
    passing_result = _make_mixture_result()
    passing_result.test_metrics["crps"] = 0.24
    assert_case_passes(passing_result, BenchmarkThresholds(max_test_crps=0.25))

    failing_result = _make_mixture_result()
    failing_result.test_metrics["crps"] = 0.26
    with pytest.raises(AssertionError, match="test_crps=0.260 exceeds 0.250"):
        assert_case_passes(failing_result, BenchmarkThresholds(max_test_crps=0.25))


def test_assert_case_passes_enforces_max_test_mu_nrmse() -> None:
    passing_result = _make_mixture_result()
    passing_result.latent_test = {
        "mape": 4.0,
        "mae": 0.5,
        "rmse": 0.7,
        "nrmse": 0.09,
        "crps": 0.3,
        "coverage_90": 0.9,
        "coverage_95": 0.95,
    }
    assert_case_passes(passing_result, BenchmarkThresholds(max_test_mu_nrmse=0.10))

    failing_result = _make_mixture_result()
    failing_result.latent_test = dict(passing_result.latent_test)
    failing_result.latent_test["nrmse"] = 0.11
    with pytest.raises(AssertionError, match="test_mu_nrmse=0.110 exceeds 0.100"):
        assert_case_passes(failing_result, BenchmarkThresholds(max_test_mu_nrmse=0.10))


def test_comparison_thresholds_use_mape_delta_and_ratio() -> None:
    baseline = _make_mixture_result()
    baseline.label = "baseline"
    baseline.test_metrics["mape"] = 4.0
    baseline.loo["elpd_loo"] = -10.0
    baseline.loo["se"] = 1.0

    candidate = _make_mixture_result()
    candidate.label = "candidate"
    candidate.test_metrics["mape"] = 4.5
    candidate.loo["elpd_loo"] = -8.5
    candidate.loo["se"] = 1.0

    comparison = compare_case_results(baseline, candidate)
    assert comparison["delta_test_mape"] == 0.5
    assert comparison["candidate_mape_ratio"] == 1.125

    assert_comparison_passes(
        baseline,
        candidate,
        ComparisonThresholds(
            min_delta_loo=1.0,
            max_delta_mape=0.6,
            max_candidate_mape_ratio=1.2,
        ),
    )

    candidate.test_metrics["mape"] = 5.0
    with pytest.raises(AssertionError, match="candidate_mape_ratio=1.250 exceeds 1.200"):
        assert_comparison_passes(
            baseline,
            candidate,
            ComparisonThresholds(max_candidate_mape_ratio=1.2),
        )
