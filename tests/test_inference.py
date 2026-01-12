"""Tests for inference utilities."""

import numpy as np
from numpyro.handlers import seed, trace

from hill_mixture_mmm.baseline import linear_baseline, standardized_time_index
from hill_mixture_mmm.inference import (
    compute_mixture_log_likelihood,
    compute_predictions,
)
from hill_mixture_mmm.metrics import compute_predictive_metrics
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam, model_single_hill
from hill_mixture_mmm.transforms import adstock_geometric, hill_matrix


def _reference_mixture_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    alpha_key: str = "alpha",
) -> np.ndarray:
    """Naive loop implementation used as a numerical reference in tests."""
    t_std = standardized_time_index(len(x))
    log_two_pi = np.log(2.0 * np.pi)
    n_samples = samples["A"].shape[0]
    out = np.zeros(n_samples)

    for i in range(n_samples):
        alpha = float(samples[alpha_key][i])
        intercept = float(samples["intercept"][i])
        slope = float(samples["slope"][i])
        sigma = float(samples["sigma"][i])
        A = samples["A"][i]
        k = samples["k"][i]
        n = samples["n"][i]
        pis = samples["pis"][i]

        s = np.asarray(adstock_geometric(x, alpha))
        baseline = linear_baseline(intercept, slope, t_std)
        mu = baseline[:, None] + np.asarray(hill_matrix(s, A, k, n))

        sigma_sq = sigma**2
        log_normal = -0.5 * (log_two_pi + np.log(sigma_sq) + ((y[:, None] - mu) ** 2) / sigma_sq)
        log_probs = np.log(pis + 1e-10) + log_normal
        out[i] = np.logaddexp.reduce(log_probs, axis=1).sum()

    return out


class TestMixtureLogLikelihood:
    """Tests for label-invariant mixture log-likelihood computation."""

    def test_matches_reference_loop(self):
        """Vectorized implementation should match the previous loop-based result."""
        rng = np.random.default_rng(0)
        T = 10
        n_samples = 5
        K = 3

        x = rng.lognormal(mean=1.0, sigma=0.3, size=T).astype(np.float32)
        y = rng.normal(loc=20.0, scale=3.0, size=T).astype(np.float32)
        samples = {
            "alpha": rng.uniform(0.1, 0.8, size=n_samples).astype(np.float32),
            "intercept": rng.normal(18.0, 2.0, size=n_samples).astype(np.float32),
            "slope": rng.normal(0.0, 1.0, size=n_samples).astype(np.float32),
            "sigma": rng.uniform(1.0, 4.0, size=n_samples).astype(np.float32),
            "A": rng.uniform(5.0, 30.0, size=(n_samples, K)).astype(np.float32),
            "k": rng.uniform(1.0, 8.0, size=(n_samples, K)).astype(np.float32),
            "n": rng.uniform(0.8, 2.5, size=(n_samples, K)).astype(np.float32),
            "pis": rng.dirichlet(np.ones(K), size=n_samples).astype(np.float32),
        }

        expected = _reference_mixture_log_likelihood(x, y, samples)
        actual = compute_mixture_log_likelihood(x, y, samples)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_respects_custom_alpha_key(self):
        """Custom alpha_key should produce the same values as the default path."""
        rng = np.random.default_rng(1)
        T = 8
        n_samples = 4
        K = 2

        x = rng.lognormal(mean=1.2, sigma=0.4, size=T).astype(np.float32)
        y = rng.normal(loc=15.0, scale=2.0, size=T).astype(np.float32)
        decay = rng.uniform(0.05, 0.6, size=n_samples).astype(np.float32)
        samples = {
            "decay": decay,
            "intercept": rng.normal(14.0, 1.0, size=n_samples).astype(np.float32),
            "slope": rng.normal(0.0, 0.8, size=n_samples).astype(np.float32),
            "sigma": rng.uniform(0.8, 2.5, size=n_samples).astype(np.float32),
            "A": rng.uniform(4.0, 12.0, size=(n_samples, K)).astype(np.float32),
            "k": rng.uniform(0.7, 5.0, size=(n_samples, K)).astype(np.float32),
            "n": rng.uniform(0.7, 2.0, size=(n_samples, K)).astype(np.float32),
            "pis": rng.dirichlet(np.ones(K), size=n_samples).astype(np.float32),
        }

        expected = _reference_mixture_log_likelihood(x, y, samples, alpha_key="decay")
        actual = compute_mixture_log_likelihood(x, y, samples, alpha_key="decay")

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class _DummyMCMC:
    """Minimal MCMC stub for predictive tests."""

    def __init__(self, samples: dict[str, np.ndarray]):
        self._samples = samples

    def get_samples(self, group_by_chain: bool = False):
        assert not group_by_chain
        return self._samples


class TestSequentialPredictions:
    """Tests for train/test predictions that preserve temporal state."""

    def test_single_hill_preserves_history_and_absolute_time_index(self):
        """Single Hill predictions should continue adstock and trend into the test split."""
        history_x = np.array([2.0, 0.0], dtype=np.float32)
        x_test = np.array([0.0, 1.0], dtype=np.float32)
        total_time = len(history_x) + len(x_test)

        samples = {
            "alpha": np.array([0.5], dtype=np.float32),
            "intercept": np.array([10.0], dtype=np.float32),
            "slope": np.array([3.0], dtype=np.float32),
            "A": np.array([5.0], dtype=np.float32),
            "k": np.array([1.0], dtype=np.float32),
            "n": np.array([1.0], dtype=np.float32),
            "sigma": np.array([0.1], dtype=np.float32),
        }
        mcmc = _DummyMCMC(samples)

        predictions = compute_predictions(
            mcmc,
            model_single_hill,
            x_test,
            history_x=history_x,
            random_seed=0,
        )

        carry = np.asarray(adstock_geometric(history_x, samples["alpha"][0]))[-1]
        s_test = np.asarray(adstock_geometric(x_test, samples["alpha"][0], init=carry))
        t_std = standardized_time_index(total_time)[len(history_x) :]
        baseline = linear_baseline(samples["intercept"][0], samples["slope"][0], t_std)
        effect = samples["A"][0] * s_test / (samples["k"][0] + s_test + 1e-12)
        expected_mu = baseline + effect

        reset_s = np.asarray(adstock_geometric(x_test, samples["alpha"][0]))
        reset_t_std = standardized_time_index(len(x_test))
        reset_mu = linear_baseline(
            samples["intercept"][0], samples["slope"][0], reset_t_std
        ) + samples["A"][0] * reset_s / (samples["k"][0] + reset_s + 1e-12)

        np.testing.assert_allclose(predictions["mu"][0], expected_mu, rtol=1e-6, atol=1e-6)
        assert not np.allclose(predictions["mu"][0], reset_mu)

    def test_mixture_predictions_preserve_history_and_absolute_time_index(self):
        """Mixture predictions should preserve carryover and absolute trend on the test split."""
        history_x = np.array([1.0, 1.0], dtype=np.float32)
        x_test = np.array([0.0, 2.0], dtype=np.float32)
        total_time = len(history_x) + len(x_test)

        samples = {
            "alpha": np.array([0.4], dtype=np.float32),
            "intercept": np.array([8.0], dtype=np.float32),
            "slope": np.array([2.0], dtype=np.float32),
            "A": np.array([[2.0, 6.0]], dtype=np.float32),
            "k": np.array([[0.8, 2.5]], dtype=np.float32),
            "n": np.array([[1.0, 1.0]], dtype=np.float32),
            "pis": np.array([[0.25, 0.75]], dtype=np.float32),
            "sigma": np.array([0.2], dtype=np.float32),
        }
        mcmc = _DummyMCMC(samples)

        predictions = compute_predictions(
            mcmc,
            model_hill_mixture_hierarchical_reparam,
            x_test,
            history_x=history_x,
            K=2,
            random_seed=0,
        )

        carry = np.asarray(adstock_geometric(history_x, samples["alpha"][0]))[-1]
        s_test = np.asarray(adstock_geometric(x_test, samples["alpha"][0], init=carry))
        t_std = standardized_time_index(total_time)[len(history_x) :]
        baseline = linear_baseline(samples["intercept"][0], samples["slope"][0], t_std)
        hills = np.asarray(hill_matrix(s_test, samples["A"][0], samples["k"][0], samples["n"][0]))
        expected_mu = baseline + np.sum(samples["pis"][0] * hills, axis=1)

        np.testing.assert_allclose(predictions["mu_expected"][0], expected_mu, rtol=1e-6, atol=1e-6)


class TestCustomTimeIndex:
    """Tests that models respect externally supplied time indices."""

    def test_single_hill_uses_supplied_time_index(self):
        """Single Hill should use the provided standardized time index during fitting."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        default_t = standardized_time_index(len(x))
        shifted_t = default_t + 5.0
        prior_config = {
            "intercept_loc": 10.0,
            "intercept_scale": 1.0,
            "slope_scale": 1.0,
            "A_loc": np.log(5.0),
            "A_scale": 0.5,
            "k_scale": 0.5,
            "n_loc": np.log(1.5),
            "n_scale": 0.4,
            "sigma_scale": 1.0,
        }

        default_trace = trace(seed(model_single_hill, 0)).get_trace(
            x=x,
            y=None,
            prior_config=prior_config,
            t_std=default_t,
        )
        shifted_trace = trace(seed(model_single_hill, 0)).get_trace(
            x=x,
            y=None,
            prior_config=prior_config,
            t_std=shifted_t,
        )

        assert not np.allclose(default_trace["mu"]["value"], shifted_trace["mu"]["value"])

    def test_mixture_model_uses_supplied_time_index(self):
        """Mixture model should use the provided standardized time index during fitting."""
        x = np.array([1.0, 2.0, 1.5], dtype=np.float32)
        default_t = standardized_time_index(len(x))
        shifted_t = default_t - 3.0
        prior_config = {
            "intercept_loc": 8.0,
            "intercept_scale": 1.0,
            "slope_scale": 1.0,
            "A_loc": np.log(4.0),
            "A_scale": 0.5,
            "k_scale": 0.5,
            "sigma_log_A_loc": -1.2,
            "sigma_log_A_scale": 0.4,
            "sigma_log_n_loc": -1.7,
            "sigma_log_n_scale": 0.4,
            "sigma_scale": 1.0,
        }

        default_trace = trace(seed(model_hill_mixture_hierarchical_reparam, 0)).get_trace(
            x=x,
            y=None,
            prior_config=prior_config,
            t_std=default_t,
            K=2,
        )
        shifted_trace = trace(seed(model_hill_mixture_hierarchical_reparam, 0)).get_trace(
            x=x,
            y=None,
            prior_config=prior_config,
            t_std=shifted_t,
            K=2,
        )

        assert not np.allclose(
            default_trace["mu_expected"]["value"],
            shifted_trace["mu_expected"]["value"],
        )


class TestPredictiveMetrics:
    """Tests for predictive-summary metrics."""

    def test_mape_is_zero_when_posterior_mean_matches_targets(self):
        """Perfect posterior-mean predictions should report zero MAPE."""
        y_true = np.array([20.0, 40.0, 80.0], dtype=np.float32)
        y_samples = np.tile(y_true, (3, 1))

        metrics = compute_predictive_metrics(y_true, y_samples)

        assert metrics["mape"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["nrmse"] == 0.0
        assert metrics["crps"] == 0.0
        assert metrics["coverage_90"] == 1.0

    def test_mape_is_reported_in_percentage_points(self):
        """MAPE should be computed from the posterior predictive mean."""
        y_true = np.array([100.0, 200.0], dtype=np.float32)
        y_samples = np.array(
            [
                [90.0, 150.0],
                [110.0, 180.0],
                [130.0, 210.0],
            ],
            dtype=np.float32,
        )

        metrics = compute_predictive_metrics(y_true, y_samples)

        assert metrics["mape"] == 10.0
        assert metrics["coverage_90"] == 1.0
        assert metrics["rmse"] > 0.0
        assert metrics["nrmse"] > 0.0
        assert metrics["crps"] > 0.0

    def test_mape_handles_zero_targets_without_nan(self):
        """Zero observations should use a finite denominator guard."""
        y_true = np.array([0.0, 100.0], dtype=np.float32)
        y_samples = np.array(
            [
                [0.0, 100.0],
                [0.0, 110.0],
            ],
            dtype=np.float32,
        )

        metrics = compute_predictive_metrics(y_true, y_samples)

        assert np.isfinite(metrics["mape"])
        assert np.isfinite(metrics["nrmse"])
        assert np.isfinite(metrics["crps"])
