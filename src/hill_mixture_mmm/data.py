"""Synthetic data generation for Hill Mixture MMM experiments."""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .baseline import linear_baseline, standardized_time_index
from .transforms import adstock_geometric, hill, hill_matrix


@dataclass
class DGPConfig:
    """Configuration for data generating process."""

    dgp_type: Literal["single", "mixture_k2", "mixture_k3"]
    T: int = 200
    sigma: float = 3.0
    alpha: float = 0.5
    intercept: float = 50.0
    slope: float = 2.0
    seed: int = 42
    profile: str = "default"

    @property
    def name(self) -> str:
        return f"{self.dataset_label}_T{self.T}_seed{self.seed}"

    @property
    def dataset_label(self) -> str:
        if self.profile == "default":
            return self.dgp_type
        return f"{self.dgp_type}_{self.profile}"


@dataclass(frozen=True)
class ControlledKSpacingConfig:
    """Configuration for a controlled component-count/spacing sweep."""

    K_true: int = 3
    T: int = 200
    sigma: float = 3.0
    alpha: float = 0.5
    intercept: float = 50.0
    slope: float = 2.0
    seed: int = 42
    raw_spend_lognormal_mean: float = 1.5
    raw_spend_lognormal_sigma: float = 0.6
    center_k_ratio: float = 0.9
    spacing_delta: float = 0.3
    pi_true: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    A_true: tuple[float, float, float] = (50.0, 50.0, 50.0)
    n_true: tuple[float, float, float] = (6.0, 6.0, 6.0)

    def _slice_component_values(
        self,
        values: tuple[float, float, float],
        *,
        name: str,
    ) -> np.ndarray:
        sliced = np.asarray(values[: self.K_true], dtype=np.float32)
        if sliced.shape != (self.K_true,):
            raise ValueError(f"{name} must provide at least K_true entries")
        return sliced

    @property
    def k_ratio_true(self) -> np.ndarray:
        if self.K_true == 1:
            ratios = np.asarray([self.center_k_ratio], dtype=np.float32)
        elif self.K_true == 2:
            ratios = np.asarray(
                [
                    self.center_k_ratio - self.spacing_delta,
                    self.center_k_ratio + self.spacing_delta,
                ],
                dtype=np.float32,
            )
        elif self.K_true == 3:
            ratios = np.asarray(
                [
                    self.center_k_ratio - self.spacing_delta,
                    self.center_k_ratio,
                    self.center_k_ratio + self.spacing_delta,
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError("ControlledKSpacingConfig supports K_true in {1, 2, 3}")
        if np.any(ratios <= 0.0):
            raise ValueError("center_k_ratio - spacing_delta must stay positive")
        return ratios

    @property
    def resolved_pi_true(self) -> np.ndarray:
        if self.K_true == 1:
            return np.asarray([1.0], dtype=np.float32)
        pi = self._slice_component_values(self.pi_true, name="pi_true")
        if np.any(pi <= 0.0):
            raise ValueError("pi_true must be strictly positive for active components")
        return pi / np.sum(pi)

    @property
    def resolved_A_true(self) -> np.ndarray:
        A = self._slice_component_values(self.A_true, name="A_true")
        if np.any(A <= 0.0):
            raise ValueError("A_true must be strictly positive for active components")
        return A

    @property
    def resolved_n_true(self) -> np.ndarray:
        n = self._slice_component_values(self.n_true, name="n_true")
        if np.any(n <= 0.0):
            raise ValueError("n_true must be strictly positive for active components")
        return n

    @property
    def dataset_label(self) -> str:
        if self.K_true == 1:
            return "single_controlled"
        return f"mixture_k{self.K_true}_spacing_d{self.spacing_delta:.2f}"


DGP_CONFIGS = {
    "single": DGPConfig(dgp_type="single"),
    "mixture_k2": DGPConfig(dgp_type="mixture_k2"),
    "mixture_k3": DGPConfig(dgp_type="mixture_k3"),
}

A_PRIOR_RANGE_FRACTION = 0.5
RAW_SPEND_LOGNORMAL_MEAN = 1.5
RAW_SPEND_LOGNORMAL_SIGMA = 0.6

MIXTURE_K3_PROFILE_PRESETS: dict[str, dict[str, np.ndarray | list[float]]] = {
    "default": {},
    "separated": {
        "pi_true": np.array([0.40, 0.35, 0.25], dtype=np.float32),
        "A_true": np.array([12.0, 35.0, 85.0], dtype=np.float32),
        "k_ratio_true": np.array([0.18, 0.75, 2.2], dtype=np.float32),
        "n_true": np.array([0.9, 2.4, 4.5], dtype=np.float32),
    },
    "high_separation": {
        "pi_true": np.array([0.40, 0.35, 0.25], dtype=np.float32),
        "A_true": np.array([12.0, 35.0, 85.0], dtype=np.float32),
        "k_ratio_true": np.array([0.06, 0.65, 3.2], dtype=np.float32),
        "n_true": np.array([0.9, 3.2, 6.5], dtype=np.float32),
        "spend_lognormal_sigma": 1.2,
    },
    "near_disjoint": {
        "pi_true": np.array([0.40, 0.35, 0.25], dtype=np.float32),
        "A_true": np.array([12.0, 35.0, 85.0], dtype=np.float32),
        "k_ratio_true": np.array([0.02, 0.45, 10.0], dtype=np.float32),
        "n_true": np.array([1.2, 6.0, 12.0], dtype=np.float32),
        "spend_lognormal_sigma": 1.6,
    },
}


def _draw_raw_spend(
    rng: np.random.Generator,
    T: int,
    *,
    mean: float = RAW_SPEND_LOGNORMAL_MEAN,
    sigma: float = RAW_SPEND_LOGNORMAL_SIGMA,
) -> np.ndarray:
    """Draw one synthetic raw spend series."""
    return rng.lognormal(mean=mean, sigma=sigma, size=T).astype(np.float32)


def _support_quantiles(values: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Return selected quantiles on the realized support as float32."""
    return np.asarray(np.quantile(values, quantiles), dtype=np.float32)


def generate_data(config: DGPConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate synthetic MMM data (x, y, meta) from specified DGP."""
    if config.dgp_type == "single":
        return _generate_single_hill(config)
    elif config.dgp_type == "mixture_k2":
        return _generate_mixture(config, K=2)
    elif config.dgp_type == "mixture_k3":
        return _generate_mixture(config, K=3)
    else:
        raise ValueError(f"Unknown DGP type: {config.dgp_type}")


def generate_controlled_k_spacing_data(
    config: ControlledKSpacingConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a controlled K in {1,2,3} synthetic dataset."""
    rng = np.random.default_rng(config.seed)
    T = config.T

    x = _draw_raw_spend(
        rng,
        T,
        mean=float(config.raw_spend_lognormal_mean),
        sigma=float(config.raw_spend_lognormal_sigma),
    )
    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))

    t_std = standardized_time_index(T)
    baseline = linear_baseline(config.intercept, config.slope, t_std)
    s_median = float(np.median(s))

    K_true = int(config.K_true)
    pi_true = config.resolved_pi_true
    A_true = config.resolved_A_true
    n_true = config.resolved_n_true
    k_true = config.k_ratio_true.astype(np.float32) * np.float32(s_median)

    z_true = rng.choice(K_true, size=T, p=pi_true)
    hill_mat = np.array(
        hill_matrix(jnp.array(s), jnp.array(A_true), jnp.array(k_true), jnp.array(n_true))
    )
    effects = hill_mat[np.arange(T), z_true]
    expected_effects = np.sum(pi_true[None, :] * hill_mat, axis=1)

    mu = baseline + effects
    mu_expected = baseline + expected_effects
    y = rng.normal(loc=mu, scale=config.sigma).astype(np.float32)

    meta = {
        "dgp_type": "single_controlled" if K_true == 1 else f"mixture_k{K_true}_controlled_spacing",
        "dataset_label": config.dataset_label,
        "K_true": K_true,
        "pi_true": pi_true,
        "A_true": A_true,
        "k_true": k_true,
        "n_true": n_true,
        "alpha_true": config.alpha,
        "intercept_true": config.intercept,
        "slope_true": config.slope,
        "sigma_true": config.sigma,
        "spacing_delta": float(config.spacing_delta),
        "center_k_ratio": float(config.center_k_ratio),
        "k_ratio_true": config.k_ratio_true.astype(np.float32),
        "raw_spend_lognormal_mean": float(config.raw_spend_lognormal_mean),
        "raw_spend_lognormal_sigma": float(config.raw_spend_lognormal_sigma),
        "s_median": s_median,
        "s_max": float(np.max(s)),
        "s": s,
        "baseline": baseline,
        "mu_true": mu,
        "mu_expected_true": mu_expected,
        "z_true": z_true,
        "hill_mat": hill_mat,
    }

    return x, y, meta


def _generate_single_hill(config: DGPConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate data from a single Hill curve (K=1)."""
    rng = np.random.default_rng(config.seed)
    T = config.T

    x = _draw_raw_spend(rng, T)

    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))

    t_std = standardized_time_index(T)
    baseline = linear_baseline(config.intercept, config.slope, t_std)

    s_median = np.median(s)
    A_true = 30.0
    k_true = s_median
    n_true = 1.5

    effect = np.array(hill(jnp.array(s), A_true, k_true, n_true))
    mu = baseline + effect

    y = rng.normal(loc=mu, scale=config.sigma).astype(np.float32)

    meta = {
        "dgp_type": "single",
        "K_true": 1,
        "pi_true": np.array([1.0]),
        "A_true": np.array([A_true]),
        "k_true": np.array([k_true]),
        "n_true": np.array([n_true]),
        "alpha_true": config.alpha,
        "intercept_true": config.intercept,
        "slope_true": config.slope,
        "sigma_true": config.sigma,
        "s_median": s_median,
        "s_max": np.max(s),
        "s": s,
        "baseline": baseline,
        "mu_true": mu,
        "mu_expected_true": mu,
        "z_true": np.zeros(T, dtype=int),
    }

    return x, y, meta


def _generate_mixture(config: DGPConfig, K: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(config.seed)
    T = config.T
    spend_lognormal_mean = RAW_SPEND_LOGNORMAL_MEAN
    spend_lognormal_sigma = RAW_SPEND_LOGNORMAL_SIGMA
    preset: dict[str, np.ndarray | list[float] | float] | None = None
    if K == 3 and str(config.profile) != "default":
        profile = str(config.profile)
        try:
            preset = MIXTURE_K3_PROFILE_PRESETS[profile]
        except KeyError as exc:
            known = ", ".join(sorted(MIXTURE_K3_PROFILE_PRESETS))
            raise ValueError(
                f"Unknown mixture_k3 profile '{profile}'. Expected one of: {known}"
            ) from exc
        spend_lognormal_mean = float(preset.get("spend_lognormal_mean", RAW_SPEND_LOGNORMAL_MEAN))
        spend_lognormal_sigma = float(
            preset.get("spend_lognormal_sigma", RAW_SPEND_LOGNORMAL_SIGMA)
        )

    x = _draw_raw_spend(
        rng,
        T,
        mean=spend_lognormal_mean,
        sigma=spend_lognormal_sigma,
    )
    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))
    t_std = standardized_time_index(T)
    baseline = linear_baseline(config.intercept, config.slope, t_std)
    s_median = np.median(s)
    k_quantiles = None

    if K == 2:
        pi_true = np.array([0.55, 0.45], dtype=np.float32)
        k_quantiles = [0.25, 0.85]
        k_true = _support_quantiles(s, k_quantiles)
        A_true = np.array([10.0, 50.0], dtype=np.float32)
        n_true = np.array([2.0, 0.8], dtype=np.float32)

    elif K == 3:
        profile = str(config.profile)
        if profile == "default":
            pi_true = np.array([0.40, 0.35, 0.25], dtype=np.float32)
            k_quantiles = [0.15, 0.60, 0.95]
            k_true = _support_quantiles(s, k_quantiles)
            A_true = np.array([12.0, 35.0, 85.0], dtype=np.float32)
            n_true = np.array([0.75, 1.35, 2.1], dtype=np.float32)
        else:
            assert preset is not None
            pi_true = np.asarray(preset["pi_true"], dtype=np.float32)
            A_true = np.asarray(preset["A_true"], dtype=np.float32)
            n_true = np.asarray(preset["n_true"], dtype=np.float32)
            k_ratio_true = np.asarray(preset["k_ratio_true"], dtype=np.float32)
            k_true = (k_ratio_true * s_median).astype(np.float32)
            k_quantiles = None

    else:
        raise ValueError(f"Unsupported K={K}")

    z_true = rng.choice(K, size=T, p=pi_true)

    hill_mat = np.array(
        hill_matrix(jnp.array(s), jnp.array(A_true), jnp.array(k_true), jnp.array(n_true))
    )
    effects = hill_mat[np.arange(T), z_true]
    expected_effects = np.sum(pi_true[None, :] * hill_mat, axis=1)

    mu = baseline + effects
    mu_expected = baseline + expected_effects
    y = rng.normal(loc=mu, scale=config.sigma).astype(np.float32)

    meta = {
        "dgp_type": f"mixture_k{K}",
        "dgp_profile": str(config.profile),
        "dataset_label": config.dataset_label,
        "K_true": K,
        "pi_true": pi_true,
        "A_true": A_true,
        "k_true": k_true,
        "n_true": n_true,
        "k_quantiles_true": None
        if k_quantiles is None
        else np.asarray(k_quantiles, dtype=np.float32),
        "alpha_true": config.alpha,
        "intercept_true": config.intercept,
        "slope_true": config.slope,
        "sigma_true": config.sigma,
        "raw_spend_lognormal_mean": spend_lognormal_mean,
        "raw_spend_lognormal_sigma": spend_lognormal_sigma,
        "s_median": s_median,
        "s_max": np.max(s),
        "s": s,
        "baseline": baseline,
        "mu_true": mu,
        "mu_expected_true": mu_expected,
        "z_true": z_true,
        "hill_mat": hill_mat,
    }

    return x, y, meta


def compute_prior_config(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute empirical Bayes prior hyperparameters from data scale."""
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_min = np.min(y)
    y_range = np.max(y) - y_min
    x_median = np.median(x)
    x_max = np.max(x)
    effect_scale = y_range * A_PRIOR_RANGE_FRACTION
    intercept_loc = max(float(y_min), float(y_mean - effect_scale))

    return {
        "intercept_loc": intercept_loc,
        "intercept_scale": float(y_std * 2),
        "slope_scale": float(y_std),
        "A_loc": float(np.log(effect_scale + 1e-6)),
        "A_scale": 0.8,
        "k_base_loc": float(np.log(x_median + 1e-6)),
        "k_scale": 0.7,
        "n_loc": float(np.log(1.5)),
        "n_scale": 0.4,
        "sigma_log_A_loc": 0.0,
        "sigma_log_A_scale": 0.8,
        "sigma_log_n_loc": -0.5,
        "sigma_log_n_scale": 0.8,
        "sigma_scale": float(y_std),
        "x_median": float(x_median),
        "x_max": float(x_max),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
