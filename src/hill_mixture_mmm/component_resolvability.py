"""Shared component-resolvability benchmark definitions and run configs."""

from __future__ import annotations

from hill_mixture_mmm.benchmark import BenchmarkRunConfig
from hill_mixture_mmm.data import ControlledKSpacingConfig

RESOLVABILITY_PROFILE_LIBRARY: dict[int, list[dict[str, object]]] = {
    1: [
        {
            "profile_id": "tv00_anchor",
            "spacing_delta": 0.0,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (1.0, 0.0, 0.0),
            "A_true": (50.0, 0.0, 0.0),
            "n_true": (2.5, 0.0, 0.0),
        }
    ],
    2: [
        {
            "profile_id": "tv07_low",
            "spacing_delta": 0.05,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.60, 0.40, 0.0),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
        {
            "profile_id": "tv27_mid",
            "spacing_delta": 0.20,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.60, 0.40, 0.0),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
        {
            "profile_id": "tv59_high",
            "spacing_delta": 0.45,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.60, 0.40, 0.0),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
        {
            "profile_id": "tv94_extreme",
            "spacing_delta": 0.80,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.60, 0.40, 0.0),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
    ],
    3: [
        {
            "profile_id": "tv05_low",
            "spacing_delta": 0.05,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.50, 0.30, 0.20),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
        {
            "profile_id": "tv18_mid",
            "spacing_delta": 0.20,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.50, 0.30, 0.20),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
        {
            "profile_id": "tv41_high",
            "spacing_delta": 0.45,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.50, 0.30, 0.20),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
        {
            "profile_id": "tv73_extreme",
            "spacing_delta": 0.80,
            "center_k_ratio": 0.9,
            "raw_spend_lognormal_sigma": 0.6,
            "pi_true": (0.50, 0.30, 0.20),
            "A_true": (50.0, 50.0, 50.0),
            "n_true": (2.5, 2.5, 2.5),
        },
    ],
}

SMOKE_PROFILE_IDS: dict[int, list[str]] = {
    1: ["tv00_anchor"],
    2: ["tv59_high"],
    3: ["tv41_high"],
}


def build_resolvability_config(
    *,
    k_true: int,
    seed: int,
    profile: dict[str, object],
    T: int = 200,
) -> ControlledKSpacingConfig:
    """Build a synthetic component-resolvability config from the shared library."""
    return ControlledKSpacingConfig(
        K_true=int(k_true),
        T=int(T),
        seed=int(seed),
        spacing_delta=float(profile["spacing_delta"]),
        center_k_ratio=float(profile["center_k_ratio"]),
        raw_spend_lognormal_sigma=float(profile["raw_spend_lognormal_sigma"]),
        pi_true=tuple(profile["pi_true"]),
        A_true=tuple(profile["A_true"]),
        n_true=tuple(profile["n_true"]),
    )


def build_resolvability_run_config(
    model_name: str,
    seed: int,
    *,
    quick: bool,
) -> BenchmarkRunConfig:
    """Return the shared benchmark run config for the component-resolvability suite.

    The priors and sampler settings are model-specific but must remain generic:
    they do not depend on the true K/profile being evaluated.
    """

    inference_seed = seed
    prior_config_override = None
    if model_name == "mixture_k2":
        prior_config_override = {
            "stick_alpha": 0.7,
            "stick_beta": 0.7,
            "k_scale": 0.30,
            "k_anchor_scale": 0.06,
            "k_increment_scale": 0.04,
            "sigma_log_A_loc": -1.3,
            "sigma_log_A_scale": 0.20,
            "sigma_log_n_loc": -1.8,
            "sigma_log_n_scale": 0.20,
        }
        inference_seed = seed + (7 if quick else 17)
    elif model_name == "mixture_k3":
        prior_config_override = {
            "stick_alpha": 5.0,
            "stick_beta": 2.5,
            "k_scale": 0.28,
            "k_anchor_scale": 0.06,
            "k_increment_scale": 0.04,
            "sigma_log_A_loc": -1.5,
            "sigma_log_A_scale": 0.18,
            "sigma_log_n_loc": -1.9,
            "sigma_log_n_scale": 0.18,
        }
        inference_seed = seed + (17 if quick else 97)

    if quick:
        if model_name == "mixture_k2":
            return BenchmarkRunConfig(
                seed=inference_seed,
                num_warmup=1400,
                num_samples=1000,
                num_chains=2,
                target_accept_prob=0.997,
                max_tree_depth=16,
                dense_mass=False,
                init_strategy="median",
                progress_bar=False,
                prior_config_override=prior_config_override,
            )
        return BenchmarkRunConfig(
            seed=inference_seed,
            num_warmup=2200,
            num_samples=1800,
            num_chains=2,
            target_accept_prob=0.995,
            max_tree_depth=16,
            dense_mass=False,
            init_strategy="median",
            progress_bar=False,
            prior_config_override=prior_config_override,
        )

    if model_name == "mixture_k2":
        return BenchmarkRunConfig(
            seed=inference_seed,
            num_warmup=2200,
            num_samples=1800,
            num_chains=2,
            target_accept_prob=0.99,
            max_tree_depth=14,
            dense_mass=False,
            init_strategy="median",
            progress_bar=False,
            prior_config_override=prior_config_override,
        )
    return BenchmarkRunConfig(
        seed=inference_seed,
        num_warmup=3200,
        num_samples=2600,
        num_chains=2,
        target_accept_prob=0.997,
        max_tree_depth=17,
        dense_mass=False,
        init_strategy="uniform",
        progress_bar=False,
        prior_config_override=prior_config_override,
    )
