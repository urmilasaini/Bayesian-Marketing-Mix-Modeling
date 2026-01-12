"""NumPyro model definitions for Hill Mixture MMM."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from .baseline import linear_baseline, standardized_time_index
from .transforms import adstock_geometric, hill, hill_matrix

DEFAULT_COMPONENT_ANCHOR_STRENGTH = 0.65
DEFAULT_COMPONENT_ANCHOR_STRENGTH_K2 = 0.8
DEFAULT_COMPONENT_ANCHOR_STRENGTH_K3 = 0.7


def _default_mixture_prior_config() -> dict[str, float]:
    return {
        "intercept_loc": 50.0,
        "intercept_scale": 20.0,
        "slope_scale": 5.0,
        "A_loc": np.log(50.0),
        "A_scale": 0.8,
        "k_scale": 0.7,
        "stick_alpha": 1.0,
        "stick_beta": 1.0,
        "sigma_log_A_loc": 0.0,
        "sigma_log_A_scale": 0.8,
        "sigma_log_n_loc": -0.5,
        "sigma_log_n_scale": 0.8,
        "sigma_scale": 10.0,
    }


def _resolve_mixture_prior_config(
    prior_config: dict[str, float] | None,
    K: int,
) -> dict[str, float | tuple[float, ...]]:
    """Apply conservative K-specific defaults on top of empirical priors."""
    resolved: dict[str, float | tuple[float, ...]] = dict(_default_mixture_prior_config())
    if prior_config is not None:
        resolved.update(prior_config)

    if K == 2:
        if prior_config is None or "stick_alpha" not in prior_config:
            resolved["stick_alpha"] = 3.0
        else:
            resolved["stick_alpha"] = float(resolved["stick_alpha"])
        if prior_config is None or "stick_beta" not in prior_config:
            resolved["stick_beta"] = 1.0
        else:
            resolved["stick_beta"] = float(resolved["stick_beta"])
        resolved["k_scale"] = min(float(resolved["k_scale"]), 0.55)
        resolved["k_anchor_scale"] = min(float(resolved.get("k_anchor_scale", 0.12)), 0.12)
        resolved["k_increment_scale"] = min(float(resolved.get("k_increment_scale", 0.08)), 0.08)
        resolved["k_anchor_quantiles"] = tuple(
            float(q) for q in resolved.get("k_anchor_quantiles", (0.25, 0.85))
        )
        resolved["sigma_log_A_loc"] = min(float(resolved["sigma_log_A_loc"]), -1.2)
        resolved["sigma_log_A_scale"] = min(float(resolved["sigma_log_A_scale"]), 0.35)
        resolved["sigma_log_n_loc"] = min(float(resolved["sigma_log_n_loc"]), -1.7)
        resolved["sigma_log_n_scale"] = min(float(resolved["sigma_log_n_scale"]), 0.35)
    elif K == 3:
        if prior_config is None or "stick_alpha" not in prior_config:
            resolved["stick_alpha"] = 3.0
        else:
            resolved["stick_alpha"] = float(resolved["stick_alpha"])
        if prior_config is None or "stick_beta" not in prior_config:
            resolved["stick_beta"] = 1.5
        else:
            resolved["stick_beta"] = float(resolved["stick_beta"])
        resolved["k_scale"] = min(float(resolved["k_scale"]), 0.45)
        resolved["k_anchor_scale"] = min(float(resolved.get("k_anchor_scale", 0.12)), 0.12)
        resolved["k_increment_scale"] = min(float(resolved.get("k_increment_scale", 0.10)), 0.10)
        resolved["k_anchor_quantiles"] = tuple(
            float(q) for q in resolved.get("k_anchor_quantiles", (0.20, 0.55, 0.90))
        )
        resolved["sigma_log_A_loc"] = min(float(resolved["sigma_log_A_loc"]), -1.1)
        resolved["sigma_log_A_scale"] = min(float(resolved["sigma_log_A_scale"]), 0.40)
        resolved["sigma_log_n_loc"] = min(float(resolved["sigma_log_n_loc"]), -1.4)
        resolved["sigma_log_n_scale"] = min(float(resolved["sigma_log_n_scale"]), 0.40)
    else:
        resolved["stick_alpha"] = float(resolved["stick_alpha"])
        resolved["stick_beta"] = float(resolved["stick_beta"])

    return resolved


def model_single_hill(x, y=None, prior_config=None, t_std=None):
    """Single Hill function MMM: x -> adstock -> hill -> y."""
    T = x.shape[0]
    if t_std is None:
        t_std = standardized_time_index(T, xp=jnp)
    else:
        t_std = jnp.asarray(t_std)

    if prior_config is None:
        prior_config = {
            "intercept_loc": 50.0,
            "intercept_scale": 20.0,
            "slope_scale": 5.0,
            "A_loc": np.log(50.0),
            "A_scale": 0.8,
            "k_scale": 0.7,
            "n_loc": np.log(1.5),
            "n_scale": 0.4,
            "sigma_scale": 10.0,
        }

    alpha = numpyro.sample("alpha", dist.Beta(2, 2))
    s = adstock_geometric(x, alpha)
    numpyro.deterministic("s", s)

    intercept = numpyro.sample(
        "intercept",
        dist.Normal(prior_config["intercept_loc"], prior_config["intercept_scale"]),
    )
    slope = numpyro.sample("slope", dist.Normal(0.0, prior_config["slope_scale"]))
    baseline = linear_baseline(intercept, slope, t_std)

    s_median = jnp.median(s)
    A = numpyro.sample("A", dist.LogNormal(prior_config["A_loc"], prior_config["A_scale"]))  # type: ignore[arg-type]
    k = numpyro.sample("k", dist.LogNormal(jnp.log(s_median + 1e-6), prior_config["k_scale"]))  # type: ignore[arg-type]
    n = numpyro.sample("n", dist.LogNormal(prior_config["n_loc"], prior_config["n_scale"]))  # type: ignore[arg-type]
    sigma = numpyro.sample("sigma", dist.HalfNormal(prior_config["sigma_scale"]))

    effect = hill(s, A, k, n)
    mu = baseline + effect

    with numpyro.plate("time", T):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    numpyro.deterministic("mu", mu)
    numpyro.deterministic("effect", effect)


def _model_hill_mixture_hierarchical_reparam_inner(
    x,
    y=None,
    K=3,
    prior_config=None,
    t_std=None,
    component_anchor_strength: float = 0.0,
):
    T = x.shape[0]
    if t_std is None:
        t_std = standardized_time_index(T, xp=jnp)
    else:
        t_std = jnp.asarray(t_std)

    prior_config = _resolve_mixture_prior_config(prior_config, K)

    alpha = numpyro.sample("alpha", dist.Beta(2, 2))
    s = adstock_geometric(x, alpha)
    numpyro.deterministic("s", s)

    intercept = numpyro.sample(
        "intercept",
        dist.Normal(prior_config["intercept_loc"], prior_config["intercept_scale"]),
    )
    slope = numpyro.sample("slope", dist.Normal(0.0, prior_config["slope_scale"]))
    baseline = linear_baseline(intercept, slope, t_std)

    stick_proportions = numpyro.sample(
        "stick_proportions",
        dist.Beta(prior_config["stick_alpha"], prior_config["stick_beta"]).expand((K - 1,)),  # type: ignore[arg-type]
    )
    remaining = jnp.ones(())
    pis_list = []
    for i in range(K - 1):
        pi_i = stick_proportions[i] * remaining
        pis_list.append(pi_i)
        remaining = remaining * (1.0 - stick_proportions[i])
    pis_list.append(remaining)
    pis = jnp.stack(pis_list)
    numpyro.deterministic("pis", pis)

    mu_log_A = numpyro.sample("mu_log_A", dist.Normal(prior_config["A_loc"], 0.5))
    sigma_log_A = numpyro.sample(
        "sigma_log_A",
        dist.LogNormal(prior_config["sigma_log_A_loc"], prior_config["sigma_log_A_scale"]),  # type: ignore[arg-type]
    )

    mu_log_n = numpyro.sample("mu_log_n", dist.Normal(jnp.log(1.5), 0.3))
    sigma_log_n = numpyro.sample(
        "sigma_log_n",
        dist.LogNormal(prior_config["sigma_log_n_loc"], prior_config["sigma_log_n_scale"]),  # type: ignore[arg-type]
    )

    log_A_raw = numpyro.sample("log_A_raw", dist.Normal(0, 1).expand((K,)))  # type: ignore[arg-type]
    if component_anchor_strength > 0.0:
        anchor_A = jnp.linspace(-1.0, 1.0, K) * component_anchor_strength
        anchor_n_scale = 0.4 if K == 2 else 0.6
        anchor_n_direction = jnp.linspace(-0.8, 0.8, K)
        anchor_n = anchor_n_direction * (component_anchor_strength * anchor_n_scale)
    else:
        anchor_A = jnp.zeros((K,))
        anchor_n = jnp.zeros((K,))

    log_A = mu_log_A + anchor_A + sigma_log_A * log_A_raw
    A = jnp.exp(log_A)
    numpyro.deterministic("log_A", log_A)
    numpyro.deterministic("A", A)

    log_n_raw = numpyro.sample("log_n_raw", dist.Normal(0, 1).expand((K,)))  # type: ignore[arg-type]
    log_n = mu_log_n + anchor_n + sigma_log_n * log_n_raw
    n = jnp.exp(log_n)
    numpyro.deterministic("log_n", log_n)
    numpyro.deterministic("n", n)

    s_median = jnp.median(s)

    if "k_anchor_quantiles" in prior_config and len(prior_config["k_anchor_quantiles"]) == K:
        support_quantiles = jnp.asarray(prior_config["k_anchor_quantiles"])
        k_anchor = jnp.quantile(s, support_quantiles)
        log_k_anchor = jnp.log(k_anchor + 1e-6)
        log_k_base = numpyro.sample(
            "log_k_base",
            dist.Normal(log_k_anchor[0], prior_config["k_anchor_scale"]),
        )
        log_k_increments_raw = numpyro.sample(
            "log_k_increments_raw",
            dist.Normal(0, 1).expand((K - 1,)),  # type: ignore[arg-type]
        )
        anchor_gaps = jnp.maximum(jnp.diff(log_k_anchor), 1e-3)
        if K == 2:
            log_k_increments = anchor_gaps + (
                jnp.abs(log_k_increments_raw) * prior_config["k_increment_scale"]
            )
        else:
            log_k_increments = jnp.maximum(
                anchor_gaps + (log_k_increments_raw * prior_config["k_increment_scale"]),
                1e-3,
            )
    else:
        log_k_base = numpyro.sample(
            "log_k_base", dist.Normal(jnp.log(s_median + 1e-6), prior_config["k_scale"])
        )
        log_k_increments_raw = numpyro.sample(
            "log_k_increments_raw",
            dist.Normal(0, 1).expand((K - 1,)),  # type: ignore[arg-type]
        )
        log_k_increments = jnp.abs(log_k_increments_raw) * prior_config["k_scale"]

    log_k_values = jnp.concatenate(
        [jnp.array([log_k_base]), log_k_base + jnp.cumsum(log_k_increments)]
    )
    k = jnp.exp(log_k_values)
    numpyro.deterministic("log_k", log_k_values)
    numpyro.deterministic("k", k)

    sigma = numpyro.sample("sigma", dist.HalfNormal(prior_config["sigma_scale"]))

    hill_mat = hill_matrix(s, A, k, n)
    mu_components = baseline[:, None] + hill_mat

    with numpyro.plate("time", T):
        numpyro.sample(
            "y",
            dist.MixtureSameFamily(dist.Categorical(pis), dist.Normal(mu_components, sigma)),
            obs=y,
        )

    mu_expected = baseline + jnp.sum(pis * hill_mat, axis=1)
    numpyro.deterministic("mu_expected", mu_expected)
    numpyro.deterministic("hill_components", hill_mat)


def _run_reparameterized_mixture_model(
    x,
    y=None,
    K=3,
    prior_config=None,
    t_std=None,
    component_anchor_strength: float = 0.0,
):
    """Apply the shared reparameterization wrapper to a mixture model."""
    reparam_config = {
        "intercept": LocScaleReparam(centered=0),
        "slope": LocScaleReparam(centered=0),
        "log_k_base": LocScaleReparam(centered=0),
        "mu_log_A": LocScaleReparam(centered=0),
        "mu_log_n": LocScaleReparam(centered=0),
    }

    with reparam(config=reparam_config):
        _model_hill_mixture_hierarchical_reparam_inner(
            x,
            y,
            K,
            prior_config,
            t_std=t_std,
            component_anchor_strength=component_anchor_strength,
        )


def model_hill_mixture_hierarchical_reparam(
    x,
    y=None,
    K=3,
    prior_config=None,
    t_std=None,
    component_anchor_strength: float = DEFAULT_COMPONENT_ANCHOR_STRENGTH,
):
    """Hierarchical Hill Mixture with non-centered parameterization and ordered k."""
    if K == 2 and component_anchor_strength == DEFAULT_COMPONENT_ANCHOR_STRENGTH:
        component_anchor_strength = DEFAULT_COMPONENT_ANCHOR_STRENGTH_K2
    if K == 3 and component_anchor_strength == DEFAULT_COMPONENT_ANCHOR_STRENGTH:
        component_anchor_strength = DEFAULT_COMPONENT_ANCHOR_STRENGTH_K3

    _run_reparameterized_mixture_model(
        x,
        y=y,
        K=K,
        prior_config=prior_config,
        t_std=t_std,
        component_anchor_strength=component_anchor_strength,
    )
