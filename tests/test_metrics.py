"""Tests for benchmark metrics."""

import numpy as np

from hill_mixture_mmm.metrics import (
    compute_across_seed_component_stability,
    compute_component_curve_tv_separation,
    compute_inverse_simpson_effective_count,
    compute_latent_recovery,
    compute_leinster_cobbold_effective_count,
    compute_parameter_recovery,
    compute_permutation_invariant_component_recovery,
    compute_rao_quadratic_entropy_equivalent_count,
    compute_shannon_effective_count,
    compute_similarity_adjusted_effective_count,
    summarize_component_posterior,
    summarize_true_components,
)


class _FakeMCMC:
    """Minimal MCMC stub for metric unit tests."""

    def __init__(self, samples: dict[str, np.ndarray]) -> None:
        self._samples = samples

    def get_samples(self) -> dict[str, np.ndarray]:
        return self._samples


class TestLatentRecovery:
    """Tests for latent mean recovery metrics."""

    def test_perfect_recovery_has_zero_error_and_full_coverage(self):
        """Exact latent mean samples should yield zero error and full coverage."""
        mu_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mu_samples = np.tile(mu_true, (4, 1))

        metrics = compute_latent_recovery(mu_true, mu_samples)

        assert metrics["mape"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["nrmse"] == 0.0
        assert metrics["crps"] == 0.0
        assert metrics["coverage_90"] == 1.0
        assert metrics["coverage_95"] == 1.0

    def test_invalid_shapes_raise_value_error(self):
        """Mismatched latent shapes should raise a clear error."""
        mu_true = np.array([1.0, 2.0], dtype=np.float32)
        mu_samples = np.array([1.0, 2.0], dtype=np.float32)

        try:
            compute_latent_recovery(mu_true, mu_samples)
        except ValueError as exc:
            assert "shape (n_samples, T)" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid mu_samples shape")


class TestComponentMetrics:
    """Tests for component recovery and across-seed stability metrics."""

    def test_true_component_curve_separation_is_bounded_and_positive_for_mixture(self):
        """The true-separation score should lie in [0, 1] and increase above zero for mixtures."""
        meta = {
            "A_true": np.array([12.0, 35.0, 85.0], dtype=np.float32),
            "k_true": np.array([2.0, 6.0, 15.0], dtype=np.float32),
            "n_true": np.array([0.8, 1.4, 2.1], dtype=np.float32),
            "pi_true": np.array([0.4, 0.35, 0.25], dtype=np.float32),
            "s_median": 10.0,
        }

        summary = summarize_true_components(meta)
        separation = compute_component_curve_tv_separation(summary)

        assert separation["num_components"] == 3
        assert separation["pair_count"] == 3
        assert 0.0 < separation["mean_pairwise_tv"] <= 1.0

    def test_similarity_adjusted_effective_count_reaches_component_count_when_separated(self):
        """Separated equal-weight components should count close to their actual number."""
        component_summary = {
            "K_total": 3,
            "K_active": 3,
            "weight_threshold": 0.05,
            "scale_reference": 10.0,
            "components": [
                {
                    "index": 0,
                    "A_mean": 10.0,
                    "A_std": 0.0,
                    "k_mean": 0.5,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.05,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 1,
                    "A_mean": 30.0,
                    "A_std": 0.0,
                    "k_mean": 10.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 1.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 4.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 2,
                    "A_mean": 80.0,
                    "A_std": 0.0,
                    "k_mean": 60.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 6.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 8.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
            ],
        }

        effective = compute_similarity_adjusted_effective_count(component_summary, gamma=0.5)

        assert effective["num_components"] == 3
        assert 2.4 <= effective["effective_count"] <= 3.0

    def test_similarity_adjusted_effective_count_discourages_duplicate_components(self):
        """Near-duplicate components should count for less than distinct separated ones."""
        duplicate_summary = {
            "K_total": 3,
            "K_active": 3,
            "weight_threshold": 0.05,
            "scale_reference": 10.0,
            "components": [
                {
                    "index": 0,
                    "A_mean": 25.0,
                    "A_std": 0.0,
                    "k_mean": 8.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.8,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.4,
                    "n_std": 0.0,
                    "pi_mean": 0.4,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 1,
                    "A_mean": 25.5,
                    "A_std": 0.0,
                    "k_mean": 8.1,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.81,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.39,
                    "n_std": 0.0,
                    "pi_mean": 0.35,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 2,
                    "A_mean": 70.0,
                    "A_std": 0.0,
                    "k_mean": 20.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 2.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 2.0,
                    "n_std": 0.0,
                    "pi_mean": 0.25,
                    "pi_std": 0.0,
                    "active": True,
                },
            ],
        }

        effective = compute_similarity_adjusted_effective_count(duplicate_summary, gamma=0.5)

        assert 1.0 < effective["effective_count"] < 2.5

    def test_weight_hill_counts_match_equal_component_count(self):
        """Weight-only Hill numbers should match the component count for equal weights."""
        component_summary = {
            "K_total": 3,
            "K_active": 3,
            "weight_threshold": 0.05,
            "scale_reference": 10.0,
            "components": [
                {
                    "index": 0,
                    "A_mean": 10.0,
                    "A_std": 0.0,
                    "k_mean": 2.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.2,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 1,
                    "A_mean": 20.0,
                    "A_std": 0.0,
                    "k_mean": 6.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.6,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.5,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 2,
                    "A_mean": 40.0,
                    "A_std": 0.0,
                    "k_mean": 18.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 1.8,
                    "k_ratio_std": 0.0,
                    "n_mean": 2.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
            ],
        }

        shannon = compute_shannon_effective_count(component_summary)
        simpson = compute_inverse_simpson_effective_count(component_summary)

        assert np.isclose(shannon["effective_count"], 3.0, atol=1e-6)
        assert np.isclose(simpson["effective_count"], 3.0, atol=1e-6)

    def test_theory_counts_include_subthreshold_components(self):
        """Shannon and Leinster-Cobbold counts should use all component weights, not only active ones."""
        component_summary = {
            "K_total": 3,
            "K_active": 1,
            "weight_threshold": 0.05,
            "scale_reference": 10.0,
            "components": [
                {
                    "index": 0,
                    "A_mean": 10.0,
                    "A_std": 0.0,
                    "k_mean": 0.5,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.05,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.0,
                    "n_std": 0.0,
                    "pi_mean": 0.92,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 1,
                    "A_mean": 30.0,
                    "A_std": 0.0,
                    "k_mean": 10.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 1.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 4.0,
                    "n_std": 0.0,
                    "pi_mean": 0.04,
                    "pi_std": 0.0,
                    "active": False,
                },
                {
                    "index": 2,
                    "A_mean": 80.0,
                    "A_std": 0.0,
                    "k_mean": 60.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 6.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 8.0,
                    "n_std": 0.0,
                    "pi_mean": 0.04,
                    "pi_std": 0.0,
                    "active": False,
                },
            ],
        }

        shannon = compute_shannon_effective_count(component_summary)
        leinster_q1 = compute_leinster_cobbold_effective_count(
            component_summary,
            q=1.0,
            lambda_=6.0,
        )

        expected_shannon = float(
            np.exp(-(0.92 * np.log(0.92) + 0.04 * np.log(0.04) + 0.04 * np.log(0.04)))
        )

        assert shannon["num_components"] == 3
        assert np.isclose(shannon["effective_count"], expected_shannon, atol=1e-9)
        assert leinster_q1["num_components"] == 3
        assert leinster_q1["effective_count"] > 1.0

    def test_leinster_cobbold_q1_and_q2_reward_separated_components(self):
        """Similarity-sensitive counts should increase toward the true count when curves separate."""
        component_summary = {
            "K_total": 3,
            "K_active": 3,
            "weight_threshold": 0.05,
            "scale_reference": 10.0,
            "components": [
                {
                    "index": 0,
                    "A_mean": 10.0,
                    "A_std": 0.0,
                    "k_mean": 0.5,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.05,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 1,
                    "A_mean": 30.0,
                    "A_std": 0.0,
                    "k_mean": 10.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 1.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 4.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 2,
                    "A_mean": 80.0,
                    "A_std": 0.0,
                    "k_mean": 60.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 6.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 8.0,
                    "n_std": 0.0,
                    "pi_mean": 1 / 3,
                    "pi_std": 0.0,
                    "active": True,
                },
            ],
        }

        lc_q1 = compute_leinster_cobbold_effective_count(component_summary, q=1.0, lambda_=6.0)
        lc_q2 = compute_leinster_cobbold_effective_count(component_summary, q=2.0, lambda_=6.0)
        rao = compute_rao_quadratic_entropy_equivalent_count(component_summary)

        assert 2.0 <= lc_q2["effective_count"] <= lc_q1["effective_count"] <= 3.0
        assert 1.0 < rao["effective_count"] <= 3.0

    def test_leinster_cobbold_discourages_duplicates_more_than_weight_only_counts(self):
        """Near-duplicate components should score below the corresponding weight-only count."""
        duplicate_summary = {
            "K_total": 3,
            "K_active": 3,
            "weight_threshold": 0.05,
            "scale_reference": 10.0,
            "components": [
                {
                    "index": 0,
                    "A_mean": 25.0,
                    "A_std": 0.0,
                    "k_mean": 8.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.8,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.4,
                    "n_std": 0.0,
                    "pi_mean": 0.4,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 1,
                    "A_mean": 25.5,
                    "A_std": 0.0,
                    "k_mean": 8.1,
                    "k_std": 0.0,
                    "k_ratio_mean": 0.81,
                    "k_ratio_std": 0.0,
                    "n_mean": 1.39,
                    "n_std": 0.0,
                    "pi_mean": 0.35,
                    "pi_std": 0.0,
                    "active": True,
                },
                {
                    "index": 2,
                    "A_mean": 70.0,
                    "A_std": 0.0,
                    "k_mean": 20.0,
                    "k_std": 0.0,
                    "k_ratio_mean": 2.0,
                    "k_ratio_std": 0.0,
                    "n_mean": 2.0,
                    "n_std": 0.0,
                    "pi_mean": 0.25,
                    "pi_std": 0.0,
                    "active": True,
                },
            ],
        }

        shannon = compute_shannon_effective_count(duplicate_summary)
        lc_q1 = compute_leinster_cobbold_effective_count(duplicate_summary, q=1.0, lambda_=6.0)

        assert 1.0 < lc_q1["effective_count"] < shannon["effective_count"] < 3.0

    def test_permutation_invariant_component_recovery_handles_swapped_components(self):
        """Matching against the true DGP should be invariant to component ordering."""
        meta = {
            "A_true": np.array([20.0, 40.0], dtype=np.float32),
            "k_true": np.array([5.0, 12.0], dtype=np.float32),
            "n_true": np.array([2.0, 1.2], dtype=np.float32),
            "pi_true": np.array([0.6, 0.4], dtype=np.float32),
            "s_median": 10.0,
        }
        samples = {
            "A": np.array([[40.0, 20.0], [40.5, 19.5]], dtype=np.float32),
            "k": np.array([[12.0, 5.0], [12.1, 4.9]], dtype=np.float32),
            "n": np.array([[1.2, 2.0], [1.18, 2.02]], dtype=np.float32),
            "pis": np.array([[0.4, 0.6], [0.41, 0.59]], dtype=np.float32),
        }

        recovery = compute_permutation_invariant_component_recovery(samples, meta)

        assert recovery is not None
        assert recovery["K_true_active"] == 2
        assert recovery["K_posterior_active"] == 2
        assert recovery["weighted_curve_nrmse"] < 0.05
        assert recovery["weighted_pi_abs_error"] < 0.05
        assert recovery["unmatched_reference_weight"] == 0.0
        assert recovery["unmatched_candidate_weight"] == 0.0

    def test_across_seed_stability_is_low_for_equivalent_component_sets(self):
        """Across-seed stability should treat relabeled-equivalent summaries as stable."""
        summary_seed0 = {
            "label": "case_seed0",
            "seed": 0,
            "component_summary": summarize_component_posterior(
                {
                    "A": np.array([[20.0, 40.0], [20.5, 39.5]], dtype=np.float32),
                    "k": np.array([[5.0, 12.0], [5.1, 11.9]], dtype=np.float32),
                    "n": np.array([[2.0, 1.2], [2.05, 1.18]], dtype=np.float32),
                    "pis": np.array([[0.6, 0.4], [0.59, 0.41]], dtype=np.float32),
                },
                scale_reference=10.0,
            ),
        }
        summary_seed1 = {
            "label": "case_seed1",
            "seed": 1,
            "component_summary": summarize_component_posterior(
                {
                    "A": np.array([[40.0, 20.0], [39.8, 20.1]], dtype=np.float32),
                    "k": np.array([[12.0, 5.0], [12.2, 5.1]], dtype=np.float32),
                    "n": np.array([[1.2, 2.0], [1.19, 1.98]], dtype=np.float32),
                    "pis": np.array([[0.4, 0.6], [0.39, 0.61]], dtype=np.float32),
                },
                scale_reference=10.0,
            ),
        }

        stability = compute_across_seed_component_stability([summary_seed0, summary_seed1])

        assert stability["num_seeds"] == 2
        assert stability["pair_count"] == 1
        assert stability["active_k_consistency"] == 1.0
        assert stability["weighted_curve_nrmse"]["mean"] < 0.05
        assert stability["weighted_pi_abs_error"]["mean"] < 0.05


class TestParameterRecovery:
    """Tests for scalar parameter recovery summaries."""

    def test_parameter_recovery_reports_slope_when_available(self):
        """Slope should be included alongside the other scalar recovery metrics."""
        mcmc = _FakeMCMC(
            {
                "slope": np.array([1.8, 2.0, 2.2, 2.1], dtype=np.float32),
                "intercept": np.array([49.0, 50.0, 51.0, 50.5], dtype=np.float32),
            }
        )
        meta = {
            "slope_true": 2.0,
            "intercept_true": 50.0,
        }

        recovery = compute_parameter_recovery(mcmc, meta)

        assert "slope" in recovery
        assert recovery["slope"]["true"] == 2.0
        assert recovery["slope"]["in_ci"]
        assert "intercept" in recovery
