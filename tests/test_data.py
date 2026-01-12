"""Tests for synthetic-data helpers and prior configuration."""

import numpy as np

from hill_mixture_mmm.data import (
    A_PRIOR_RANGE_FRACTION,
    ControlledKSpacingConfig,
    DGPConfig,
    compute_prior_config,
    generate_controlled_k_spacing_data,
    generate_data,
)
from hill_mixture_mmm.metrics import (
    compute_component_curve_tv_separation,
    summarize_true_components,
)


class TestComputePriorConfig:
    """Tests for empirical-Bayes prior construction."""

    def test_intercept_prior_is_shifted_below_response_mean(self):
        """Intercept prior should reserve room for positive Hill effects."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([10.0, 12.0, 18.0, 22.0], dtype=np.float32)

        prior = compute_prior_config(x, y)

        y_mean = float(np.mean(y))
        y_min = float(np.min(y))
        y_range = float(np.max(y) - np.min(y))
        expected = max(y_min, y_mean - y_range * A_PRIOR_RANGE_FRACTION)

        assert prior["intercept_loc"] == expected
        assert prior["intercept_loc"] < y_mean
        assert np.isclose(np.exp(prior["A_loc"]), y_range * A_PRIOR_RANGE_FRACTION, atol=1e-6)


class TestSyntheticLatentTargets:
    """Tests that synthetic metadata exposes the correct latent benchmark target."""

    def test_single_dgp_expected_latent_matches_realized_latent(self):
        """Single-Hill data should have identical realized and expected latent means."""
        _, _, meta = generate_data(DGPConfig(dgp_type="single", T=50, seed=0))

        np.testing.assert_allclose(meta["mu_expected_true"], meta["mu_true"])

    def test_mixture_dgp_stores_expected_latent_mean(self):
        """Mixture DGP should store the expectation over latent segments."""
        _, _, meta = generate_data(DGPConfig(dgp_type="mixture_k2", T=200, seed=0))

        expected = meta["baseline"] + np.sum(meta["pi_true"][None, :] * meta["hill_mat"], axis=1)

        np.testing.assert_allclose(meta["mu_expected_true"], expected)
        assert not np.allclose(meta["mu_expected_true"], meta["mu_true"])

    def test_mixture_k3_profiles_increase_true_component_separation(self):
        """Additional K=3 profiles should span progressively larger oracle separation."""
        labels = ["default", "separated", "high_separation"]
        separations = []
        for profile in labels:
            _, _, meta = generate_data(
                DGPConfig(dgp_type="mixture_k3", T=200, seed=0, profile=profile)
            )
            separation = compute_component_curve_tv_separation(summarize_true_components(meta))
            separations.append(float(separation["mean_pairwise_tv"]))

        assert separations[0] < separations[1] < separations[2]
        assert separations[-1] > 0.85

    def test_controlled_k_spacing_family_is_monotone_in_true_separation(self):
        """Controlled spacing sweep should increase oracle separation with delta."""
        deltas = [0.1, 0.3, 0.6]
        separations = []
        for delta in deltas:
            _, _, meta = generate_controlled_k_spacing_data(
                ControlledKSpacingConfig(seed=0, spacing_delta=delta)
            )
            separation = compute_component_curve_tv_separation(summarize_true_components(meta))
            separations.append(float(separation["mean_pairwise_tv"]))

        assert separations[0] < separations[1] < separations[2]
        assert separations[-1] > 0.8

    def test_controlled_k_spacing_supports_k1_and_k2(self):
        """Controlled sweep helper should also support K_true=1 and K_true=2."""
        _, _, meta_single = generate_controlled_k_spacing_data(
            ControlledKSpacingConfig(K_true=1, seed=0, spacing_delta=0.0)
        )
        _, _, meta_mix2 = generate_controlled_k_spacing_data(
            ControlledKSpacingConfig(K_true=2, seed=0, spacing_delta=0.2)
        )

        assert int(meta_single["K_true"]) == 1
        assert meta_single["pi_true"].shape == (1,)
        assert meta_single["k_true"].shape == (1,)

        assert int(meta_mix2["K_true"]) == 2
        assert meta_mix2["pi_true"].shape == (2,)
        assert meta_mix2["k_true"].shape == (2,)

        single_sep = compute_component_curve_tv_separation(summarize_true_components(meta_single))
        mix2_sep = compute_component_curve_tv_separation(summarize_true_components(meta_mix2))
        assert np.isclose(float(single_sep["mean_pairwise_tv"]), 0.0)
        assert float(mix2_sep["mean_pairwise_tv"]) > 0.0

    def test_controlled_k_spacing_honors_active_component_overrides(self):
        """Controlled helper should respect custom K=2/K=3 component parameters."""
        _, _, meta_mix2 = generate_controlled_k_spacing_data(
            ControlledKSpacingConfig(
                K_true=2,
                seed=0,
                spacing_delta=0.2,
                pi_true=(0.7, 0.3, 0.0),
                A_true=(42.0, 68.0, 1.0),
                n_true=(1.5, 3.5, 1.0),
            )
        )

        np.testing.assert_allclose(meta_mix2["pi_true"], np.array([0.7, 0.3], dtype=np.float32))
        np.testing.assert_allclose(meta_mix2["A_true"], np.array([42.0, 68.0], dtype=np.float32))
        np.testing.assert_allclose(meta_mix2["n_true"], np.array([1.5, 3.5], dtype=np.float32))
