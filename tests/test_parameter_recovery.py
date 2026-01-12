"""Integration tests for parameter recovery.

Verifies that models can recover known parameters from synthetic data.
These are slower tests that run actual MCMC.
"""

import numpy as np
import pytest

from hill_mixture_mmm.data import DGPConfig, compute_prior_config, generate_data
from hill_mixture_mmm.inference import run_inference
from hill_mixture_mmm.metrics import compute_effective_k, compute_parameter_recovery
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam, model_single_hill

model_hill_mixture_sparse = model_hill_mixture_hierarchical_reparam

TEST_WARMUP = 200
TEST_SAMPLES = 200
TEST_CHAINS = 2


class TestSingleHillRecovery:
    """Test that single Hill model recovers parameters from single Hill DGP."""

    @pytest.mark.slow
    def test_recovers_alpha(self):
        """Alpha (adstock) should be recovered within 95% CI."""
        config = DGPConfig(dgp_type="single", T=200, seed=42)
        x, y, meta = generate_data(config)
        prior_config = compute_prior_config(x, y)

        mcmc = run_inference(
            model_single_hill,
            x,
            y,
            seed=42,
            num_warmup=TEST_WARMUP,
            num_samples=TEST_SAMPLES,
            num_chains=TEST_CHAINS,
            prior_config=prior_config,
        )

        recovery = compute_parameter_recovery(mcmc, meta)
        assert recovery["alpha"]["in_ci"], (
            f"Alpha not recovered: true={meta['alpha_true']}, "
            f"CI=[{recovery['alpha']['ci_low']:.3f}, {recovery['alpha']['ci_high']:.3f}]"
        )

    @pytest.mark.slow
    def test_recovers_sigma(self):
        """Sigma (noise) should be recovered within 95% CI."""
        config = DGPConfig(dgp_type="single", T=200, seed=42)
        x, y, meta = generate_data(config)
        prior_config = compute_prior_config(x, y)

        mcmc = run_inference(
            model_single_hill,
            x,
            y,
            seed=42,
            num_warmup=TEST_WARMUP,
            num_samples=TEST_SAMPLES,
            num_chains=TEST_CHAINS,
            prior_config=prior_config,
        )

        recovery = compute_parameter_recovery(mcmc, meta)
        assert recovery["sigma"]["in_ci"], (
            f"Sigma not recovered: true={meta['sigma_true']}, "
            f"CI=[{recovery['sigma']['ci_low']:.3f}, {recovery['sigma']['ci_high']:.3f}]"
        )


class TestMixtureEffectiveK:
    """Test that mixture models identify correct number of components."""

    @pytest.mark.slow
    def test_sparse_finds_single_component(self):
        """On single Hill DGP, sparse K=5 should prune to ~1 component."""
        config = DGPConfig(dgp_type="single", T=200, seed=42)
        x, y, meta = generate_data(config)
        prior_config = compute_prior_config(x, y)

        mcmc = run_inference(
            model_hill_mixture_sparse,
            x,
            y,
            K=5,
            seed=42,
            num_warmup=TEST_WARMUP,
            num_samples=TEST_SAMPLES,
            num_chains=TEST_CHAINS,
            prior_config=prior_config,
        )

        eff_k = compute_effective_k(mcmc, threshold=0.05)
        assert eff_k["effective_k_mean"] < 3.0, (
            f"Sparse K=5 on single Hill DGP should prune, got {eff_k['effective_k_mean']:.2f}"
        )

    @pytest.mark.slow
    def test_sparse_finds_multiple_components(self):
        """On mixture K=3 DGP, sparse K=5 should find ~3 components."""
        config = DGPConfig(dgp_type="mixture_k3", T=200, seed=42)
        x, y, meta = generate_data(config)
        prior_config = compute_prior_config(x, y)

        mcmc = run_inference(
            model_hill_mixture_sparse,
            x,
            y,
            K=5,
            seed=42,
            num_warmup=TEST_WARMUP,
            num_samples=TEST_SAMPLES,
            num_chains=TEST_CHAINS,
            prior_config=prior_config,
        )

        eff_k = compute_effective_k(mcmc, threshold=0.05)
        assert 2.0 <= eff_k["effective_k_mean"] <= 5.0, (
            f"Sparse K=5 on mixture K=3 DGP should find ~3, got {eff_k['effective_k_mean']:.2f}"
        )


class TestDataGeneration:
    """Test that DGP produces expected data characteristics."""

    @staticmethod
    def _pairwise_curve_summary(hill_mat: np.ndarray) -> tuple[float, float, int]:
        """Summarize the closest pair of component curves on the observed support."""
        min_pairwise_nrmse = np.inf
        min_pairwise_mean_abs_gap = np.inf
        max_pairwise_crossings = 0

        for left_idx in range(hill_mat.shape[1]):
            for right_idx in range(left_idx + 1, hill_mat.shape[1]):
                component_left = hill_mat[:, left_idx]
                component_right = hill_mat[:, right_idx]
                diff = component_right - component_left
                rmse = np.sqrt(np.mean(diff**2))
                scale = max(
                    float(np.max(np.abs(component_left))),
                    float(np.max(np.abs(component_right))),
                    1.0,
                )
                signs = np.sign(diff)
                crossings = int(np.sum(signs[:-1] * signs[1:] < 0))
                min_pairwise_nrmse = min(min_pairwise_nrmse, float(rmse / scale))
                min_pairwise_mean_abs_gap = min(
                    min_pairwise_mean_abs_gap,
                    float(np.mean(np.abs(diff))),
                )
                max_pairwise_crossings = max(max_pairwise_crossings, crossings)

        return min_pairwise_nrmse, min_pairwise_mean_abs_gap, max_pairwise_crossings

    def test_single_dgp_has_one_component(self):
        """Single Hill DGP should have K_true=1."""
        config = DGPConfig(dgp_type="single", T=100, seed=0)
        _, _, meta = generate_data(config)
        assert meta["K_true"] == 1
        assert len(meta["A_true"]) == 1

    def test_mixture_k3_dgp_has_three_components(self):
        """Mixture K=3 DGP should have K_true=3."""
        config = DGPConfig(dgp_type="mixture_k3", T=100, seed=0)
        _, _, meta = generate_data(config)
        assert meta["K_true"] == 3
        assert len(meta["A_true"]) == 3
        assert len(meta["pi_true"]) == 3
        np.testing.assert_allclose(meta["pi_true"].sum(), 1.0)

    @pytest.mark.parametrize("seed", range(5))
    def test_mixture_k2_dgp_is_separated_on_observed_support(self, seed):
        """The K=2 benchmark DGP should avoid near-tied component curves."""
        config = DGPConfig(dgp_type="mixture_k2", T=200, seed=seed)
        _, _, meta = generate_data(config)

        min_pairwise_nrmse, min_pairwise_mean_abs_gap, max_pairwise_crossings = (
            self._pairwise_curve_summary(np.asarray(meta["hill_mat"]))
        )

        assert min_pairwise_nrmse > 0.20, (
            "mixture_k2 curves are too close on the realized support: "
            f"min pairwise nRMSE={min_pairwise_nrmse:.3f}"
        )
        assert min_pairwise_mean_abs_gap > float(meta["sigma_true"]), (
            "mixture_k2 closest pair does not separate beyond the observation noise scale: "
            f"mean abs gap={min_pairwise_mean_abs_gap:.3f}, sigma={meta['sigma_true']:.3f}"
        )
        assert max_pairwise_crossings == 0, (
            "mixture_k2 component curves cross on the realized support, which makes "
            f"the benchmark identifiability hinge on near-ties (crossings={max_pairwise_crossings})"
        )

    def test_different_seeds_produce_different_data(self):
        """Different seeds should produce different y values."""
        config1 = DGPConfig(dgp_type="single", T=100, seed=0)
        config2 = DGPConfig(dgp_type="single", T=100, seed=1)
        _, y1, _ = generate_data(config1)
        _, y2, _ = generate_data(config2)
        assert not np.allclose(y1, y2), "Different seeds should produce different data"

    def test_same_seed_produces_same_data(self):
        """Same seed should produce identical data."""
        config1 = DGPConfig(dgp_type="single", T=100, seed=42)
        config2 = DGPConfig(dgp_type="single", T=100, seed=42)
        x1, y1, _ = generate_data(config1)
        x2, y2, _ = generate_data(config2)
        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(y1, y2)
