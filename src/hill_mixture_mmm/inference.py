"""Inference utilities for Hill Mixture MMM."""

import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_median, init_to_uniform

from .baseline import linear_baseline, standardized_time_index


def run_inference(
    model_fn: Callable,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    prior_config: dict | None = None,
    target_accept_prob: float = 0.9,
    max_tree_depth: int = 10,
    dense_mass: bool = False,
    init_strategy: str = "uniform",
    progress_bar: bool = True,
    **model_kwargs,
) -> MCMC:
    """Run NUTS MCMC inference and return the fitted MCMC object."""
    rng_key = jax.random.PRNGKey(seed)
    if init_strategy == "median":
        resolved_init_strategy = init_to_median(num_samples=15)
    elif init_strategy == "uniform":
        resolved_init_strategy = init_to_uniform()
    else:
        raise ValueError(f"Unsupported init_strategy: {init_strategy}")

    kernel = NUTS(
        model_fn,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=resolved_init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        rng_key,
        x=jnp.array(x),
        y=jnp.array(y),
        prior_config=prior_config,
        extra_fields=("diverging", "energy", "num_steps", "accept_prob"),
        **model_kwargs,
    )
    return mcmc


def compute_predictions(
    mcmc: MCMC,
    model_fn: Callable,
    x: np.ndarray,
    prior_config: dict | None = None,
    history_x: np.ndarray | None = None,
    time_start: int | None = None,
    total_time: int | None = None,
    random_seed: int = 99,
    **model_kwargs,
) -> dict[str, np.ndarray]:
    """Compute posterior predictive samples, continuing adstock from history_x if provided."""
    samples = mcmc.get_samples()
    n_samples = int(samples["sigma"].shape[0]) if "sigma" in samples else 0

    if history_x is None:
        history_x = np.array([], dtype=np.float32)
    else:
        history_x = np.asarray(history_x, dtype=np.float32)

    if time_start is None:
        time_start = int(len(history_x))
    if total_time is None:
        total_time = int(time_start + len(x))

    t_std_full = standardized_time_index(total_time)
    t_std = np.asarray(t_std_full[time_start : time_start + len(x)], dtype=np.float32)
    adstock_init = _compute_adstock_init(history_x, samples, n_samples)

    if _is_supported_posterior_sample(samples):
        return _compute_predictive_from_samples(
            samples=samples,
            x=np.asarray(x, dtype=np.float32),
            t_std=t_std,
            adstock_init=adstock_init,
            random_seed=random_seed,
        )

    pred = Predictive(model_fn, posterior_samples=samples)
    pred_samples = pred(
        jax.random.PRNGKey(random_seed),
        x=jnp.array(x),
        y=None,
        prior_config=prior_config,
        **model_kwargs,
    )
    return {k: np.array(v) for k, v in pred_samples.items()}


def compute_loo(mcmc: MCMC) -> dict[str, Any]:
    """Compute LOO-CV (PSIS-LOO) using ArviZ."""
    idata = az.from_numpyro(mcmc)
    try:
        loo = az.loo(idata, pointwise=True)
        pareto_k = loo.pareto_k.values
        return {
            "elpd_loo": float(loo.elpd_loo),
            "se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "pareto_k_bad": int(np.sum(pareto_k > 0.7)),
            "pareto_k_very_bad": int(np.sum(pareto_k > 1.0)),
            "n_obs": len(pareto_k),
        }
    except Exception as e:
        return {"elpd_loo": np.nan, "error": str(e)}


def compute_waic(mcmc: MCMC) -> dict[str, Any]:
    """Compute WAIC using ArviZ."""
    idata = az.from_numpyro(mcmc)
    try:
        waic = az.waic(idata)
        return {
            "elpd_waic": float(waic.elpd_waic),
            "se": float(waic.se),
            "p_waic": float(waic.p_waic),
        }
    except Exception as e:
        return {"elpd_waic": np.nan, "error": str(e)}


def compute_convergence_diagnostics(mcmc: MCMC) -> dict[str, Any]:
    """Compute R-hat and ESS diagnostics."""
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata, kind="diagnostics")

    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "converged": bool(summary["r_hat"].max() < 1.05),
        "ess_sufficient": bool(summary["ess_bulk"].min() > 400),
    }


def compute_hmc_diagnostics(
    mcmc: MCMC,
    *,
    max_tree_depth: int = 10,
    bfmi_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compute sampler diagnostics used in stricter HMC convergence checks."""
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    diverging = np.asarray(extra_fields.get("diverging", np.zeros((1, 0), dtype=bool)))
    energy = np.asarray(extra_fields.get("energy", np.zeros((diverging.shape[0], 0))))
    num_steps = np.asarray(extra_fields.get("num_steps", np.zeros_like(diverging, dtype=int)))
    accept_prob = np.asarray(extra_fields.get("accept_prob", np.zeros_like(diverging, dtype=float)))

    tree_depth = np.floor(np.log2(np.maximum(num_steps, 1))).astype(int) + 1
    bfmi_by_chain = np.asarray(az.bfmi(energy), dtype=float) if energy.size else np.array([np.nan])

    return {
        "num_divergences": int(np.sum(diverging)),
        "has_divergence": bool(np.any(diverging)),
        "bfmi_by_chain": bfmi_by_chain.tolist(),
        "min_bfmi": float(np.nanmin(bfmi_by_chain)),
        "bfmi_ok": bool(np.nanmin(bfmi_by_chain) >= bfmi_threshold),
        "max_tree_depth": int(max_tree_depth),
        "tree_depth_hits": int(np.sum(tree_depth >= max_tree_depth)),
        "max_tree_depth_hit": bool(np.any(tree_depth >= max_tree_depth)),
        "max_num_steps": int(np.max(num_steps)) if num_steps.size else 0,
        "mean_accept_prob": float(np.mean(accept_prob)) if accept_prob.size else float("nan"),
    }


def relabel_samples_by_k(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort mixture components by k at each MCMC draw for label-switching robustness."""
    component_params = ["k", "A", "n", "pis", "log_k", "log_A", "log_n"]

    if "k" not in samples:
        raise ValueError("samples must contain 'k' for relabeling")

    k = samples["k"]
    n_samples, K = k.shape

    sort_indices = np.argsort(k, axis=1)

    relabeled = {}

    for key, value in samples.items():
        if key in component_params and value.ndim == 2 and value.shape[1] == K:
            relabeled[key] = np.take_along_axis(value, sort_indices, axis=1)
        else:
            relabeled[key] = value

    return relabeled


def check_label_switching(samples: dict[str, np.ndarray], param: str = "k") -> dict[str, Any]:
    """Diagnose label switching by analyzing component ordering consistency."""
    if param not in samples:
        raise ValueError(f"samples must contain '{param}'")

    values = samples[param]
    n_samples, K = values.shape

    orderings = np.argsort(values, axis=1)

    ordering_tuples = [tuple(o) for o in orderings]

    counts = Counter(ordering_tuples)
    mode_ordering = counts.most_common(1)[0][0]
    mode_count = counts[mode_ordering]

    return {
        "switching_rate": 1.0 - (mode_count / n_samples),
        "n_unique_orderings": len(counts),
        "mode_ordering": mode_ordering,
        "mode_count": mode_count,
        "n_samples": n_samples,
        "top_orderings": counts.most_common(min(5, len(counts))),
    }


def _batched_adstock_geometric(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
    init: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Vectorized geometric adstock for many posterior draws at once."""

    def step(carry, x_t):
        carry = x_t + alpha * carry
        return carry, carry

    if init is None:
        init = jnp.zeros_like(alpha)
    _, s = jax.lax.scan(step, init, x)
    return jnp.swapaxes(s, 0, 1)


def _is_supported_posterior_sample(samples: dict[str, np.ndarray]) -> bool:
    """Return whether predictive samples can be computed analytically."""
    single_required = {"alpha", "intercept", "slope", "A", "k", "n", "sigma"}
    mixture_required = single_required | {"pis"}
    keys = set(samples)
    return single_required.issubset(keys) or mixture_required.issubset(keys)


def _compute_adstock_init(
    history_x: np.ndarray,
    samples: dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Compute posterior-specific adstock carry from the observed history."""
    if n_samples == 0 or len(history_x) == 0 or "alpha" not in samples:
        return np.zeros(n_samples, dtype=np.float32)

    carried = _batched_adstock_geometric(
        x=jnp.asarray(history_x),
        alpha=jnp.asarray(samples["alpha"]),
    )
    return np.asarray(carried[:, -1], dtype=np.float32)


def _compute_predictive_from_samples(
    samples: dict[str, np.ndarray],
    x: np.ndarray,
    t_std: np.ndarray,
    adstock_init: np.ndarray,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Compute posterior predictive samples while preserving sequential state."""
    rng = np.random.default_rng(random_seed)

    alpha = np.asarray(samples["alpha"], dtype=np.float32)
    intercept = np.asarray(samples["intercept"], dtype=np.float32)
    slope = np.asarray(samples["slope"], dtype=np.float32)
    sigma = np.asarray(samples["sigma"], dtype=np.float32)
    A = np.asarray(samples["A"], dtype=np.float32)
    k = np.asarray(samples["k"], dtype=np.float32)
    n = np.asarray(samples["n"], dtype=np.float32)

    s = np.asarray(
        _batched_adstock_geometric(
            x=jnp.asarray(x),
            alpha=jnp.asarray(alpha),
            init=jnp.asarray(adstock_init),
        ),
        dtype=np.float32,
    )
    baseline = linear_baseline(intercept[:, None], slope[:, None], t_std[None, :])

    if "pis" not in samples:
        effect = (
            A[:, None] * (s ** n[:, None]) / (k[:, None] ** n[:, None] + s ** n[:, None] + 1e-12)
        )
        mu = baseline + effect
        y = rng.normal(loc=mu, scale=sigma[:, None]).astype(np.float32)
        return {
            "y": y,
            "mu": mu.astype(np.float32),
            "effect": effect.astype(np.float32),
            "s": s,
        }

    pis = np.asarray(samples["pis"], dtype=np.float32)
    s_expanded = s[:, :, None]
    n_expanded = n[:, None, :]
    s_power = s_expanded**n_expanded
    k_power = k[:, None, :] ** n_expanded
    hill_components = A[:, None, :] * s_power / (k_power + s_power + 1e-12)
    mu_components = baseline[:, :, None] + hill_components
    mu_expected = baseline + np.sum(pis[:, None, :] * hill_components, axis=-1)

    u = rng.random((pis.shape[0], x.shape[0], 1), dtype=np.float32)
    component_idx = (u > np.cumsum(pis[:, None, :], axis=-1)).sum(axis=-1, dtype=np.int32)
    mu_selected = np.take_along_axis(mu_components, component_idx[:, :, None], axis=-1).squeeze(-1)
    y = rng.normal(loc=mu_selected, scale=sigma[:, None]).astype(np.float32)

    return {
        "y": y,
        "mu_expected": mu_expected.astype(np.float32),
        "hill_components": hill_components.astype(np.float32),
        "s": s,
    }


@jax.jit
def _compute_mixture_log_likelihood_vectorized(
    x: jnp.ndarray,
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    intercept: jnp.ndarray,
    slope: jnp.ndarray,
    A: jnp.ndarray,
    k: jnp.ndarray,
    n: jnp.ndarray,
    pis: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Compute total mixture log-likelihood for all draws in a chain."""
    t_std = standardized_time_index(x.shape[0], xp=jnp)
    s = _batched_adstock_geometric(x, alpha)
    baseline = linear_baseline(intercept[:, None], slope[:, None], t_std[None, :])

    s_expanded = s[:, :, None]
    n_expanded = n[:, None, :]
    s_power = s_expanded**n_expanded
    k_power = k[:, None, :] ** n_expanded
    hill_components = A[:, None, :] * s_power / (k_power + s_power + 1e-12)
    mu_components = baseline[:, :, None] + hill_components

    y_expanded = y[None, :, None]
    sigma_expanded = sigma[:, None, None]
    log_normal = -0.5 * (
        jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.log(sigma_expanded)
        + ((y_expanded - mu_components) / sigma_expanded) ** 2
    )
    log_probs = jnp.log(pis[:, None, :] + 1e-10) + log_normal
    return jnp.sum(logsumexp(log_probs, axis=-1), axis=-1)


def compute_mixture_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    samples: dict[str, np.ndarray],
    alpha_key: str = "alpha",
) -> np.ndarray:
    """Compute label-invariant total log-likelihood per MCMC draw."""
    required_keys = {"A", "k", "n", "pis", "sigma", "intercept", "slope", alpha_key}
    missing = sorted(required_keys.difference(samples))
    if missing:
        raise ValueError(f"samples missing keys required for log-likelihood: {missing}")

    log_likelihoods = _compute_mixture_log_likelihood_vectorized(
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        alpha=jnp.asarray(samples[alpha_key]),
        intercept=jnp.asarray(samples["intercept"]),
        slope=jnp.asarray(samples["slope"]),
        A=jnp.asarray(samples["A"]),
        k=jnp.asarray(samples["k"]),
        n=jnp.asarray(samples["n"]),
        pis=jnp.asarray(samples["pis"]),
        sigma=jnp.asarray(samples["sigma"]),
    )
    return np.asarray(log_likelihoods)


def compute_label_invariant_diagnostics(
    mcmc: MCMC,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "rank",
) -> dict[str, Any]:
    """Compute R-hat/ESS on label-invariant quantities (log-likelihood + scalars)."""
    samples = mcmc.get_samples(group_by_chain=True)
    n_chains = samples["sigma"].shape[0]

    log_liks_by_chain = []
    for chain_idx in range(n_chains):
        chain_samples = {k: v[chain_idx] for k, v in samples.items()}
        log_lik = compute_mixture_log_likelihood(x, y, chain_samples)
        log_liks_by_chain.append(log_lik)

    log_liks = np.stack(log_liks_by_chain)

    rhat_log_lik = _compute_rhat(log_liks, method=method)
    ess_bulk_log_lik = _compute_ess(log_liks, method="bulk")
    ess_tail_log_lik = _compute_ess(log_liks, method="tail")

    scalar_params = ["intercept", "slope", "sigma", "alpha"]
    rhat_scalars = {}
    ess_bulk_scalars = {}
    ess_tail_scalars = {}
    for param in scalar_params:
        if param in samples:
            param_values = samples[param]
            if param_values.ndim == 2:
                rhat_scalars[param] = _compute_rhat(param_values, method=method)
                ess_bulk_scalars[param] = _compute_ess(param_values, method="bulk")
                ess_tail_scalars[param] = _compute_ess(param_values, method="tail")

    all_rhats = [rhat_log_lik] + list(rhat_scalars.values())
    all_ess_bulk = [ess_bulk_log_lik] + list(ess_bulk_scalars.values())
    all_ess_tail = [ess_tail_log_lik] + list(ess_tail_scalars.values())
    max_rhat = max(all_rhats)
    converged = max_rhat < 1.01

    return {
        "rhat_log_lik": rhat_log_lik,
        "rhat_scalars": rhat_scalars,
        "ess_bulk_log_lik": ess_bulk_log_lik,
        "ess_tail_log_lik": ess_tail_log_lik,
        "ess_bulk_scalars": ess_bulk_scalars,
        "ess_tail_scalars": ess_tail_scalars,
        "min_ess_bulk": float(np.nanmin(all_ess_bulk)),
        "min_ess_tail": float(np.nanmin(all_ess_tail)),
        "max_rhat": max_rhat,
        "converged": converged,
        "method": method,
        "threshold": 1.01,
    }


def compute_diagnostics_on_relabeled(
    mcmc: MCMC,
    method: str = "rank",
) -> dict[str, Any]:
    """Compute R-hat/ESS on k-relabeled component parameters."""
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    n_chains = list(samples_by_chain.values())[0].shape[0]

    relabeled_by_chain = []
    for chain_idx in range(n_chains):
        chain_samples = {k: v[chain_idx] for k, v in samples_by_chain.items()}
        relabeled = relabel_samples_by_k(chain_samples)
        relabeled_by_chain.append(relabeled)

    relabeled_samples = {}
    for key in relabeled_by_chain[0].keys():
        relabeled_samples[key] = np.stack([r[key] for r in relabeled_by_chain])

    component_params = ["A", "k", "n", "pis"]
    results = {}
    ess_bulk_results = {}
    ess_tail_results = {}

    for param in component_params:
        if param in relabeled_samples:
            values = relabeled_samples[param]
            if values.ndim == 3:
                K = values.shape[2]
                rhats = []
                ess_bulk = []
                ess_tail = []
                for k_idx in range(K):
                    rhat = _compute_rhat(values[:, :, k_idx], method=method)
                    rhats.append(rhat)
                    ess_bulk.append(_compute_ess(values[:, :, k_idx], method="bulk"))
                    ess_tail.append(_compute_ess(values[:, :, k_idx], method="tail"))
                results[param] = {
                    "per_component": rhats,
                    "max": max(rhats),
                }
                ess_bulk_results[param] = {
                    "per_component": ess_bulk,
                    "min": min(ess_bulk),
                }
                ess_tail_results[param] = {
                    "per_component": ess_tail,
                    "min": min(ess_tail),
                }

    max_rhat = max(r["max"] for r in results.values())
    min_ess_bulk = min(r["min"] for r in ess_bulk_results.values())
    min_ess_tail = min(r["min"] for r in ess_tail_results.values())
    converged = max_rhat < 1.01

    return {
        "component_rhats": results,
        "component_ess_bulk": ess_bulk_results,
        "component_ess_tail": ess_tail_results,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "converged": converged,
        "method": method,
        "threshold": 1.01,
    }


def _extract_arviz_scalar(result: Any) -> float:
    """Extract a scalar float from an ArviZ diagnostic result."""
    if isinstance(result, xr.Dataset):
        var_name = list(result.data_vars)[0]
        value = float(result[var_name].values.item())
    elif isinstance(result, xr.DataArray):
        value = float(result.values.item())
    elif isinstance(result, (int, float)):
        value = float(result)
    elif hasattr(result, "item"):
        value = float(result.item())
    else:
        value = float(result)
    return value if np.isfinite(value) else np.nan


def _compute_rhat(values: np.ndarray, method: str = "rank") -> float:
    n_chains, n_samples = values.shape

    if n_chains < 2:
        return np.nan

    if np.any(~np.isfinite(values)):
        return np.nan

    if np.allclose(values, values[0, 0]):
        return 1.0

    try:
        da = xr.DataArray(
            values,
            dims=["chain", "draw"],
            coords={"chain": np.arange(n_chains), "draw": np.arange(n_samples)},
        )

        if method == "rank":
            rhat = az.rhat(da, method="rank")
        else:
            rhat = az.rhat(da, method="split")

        return _extract_arviz_scalar(rhat)

    except (ValueError, TypeError, RuntimeWarning) as e:
        warnings.warn(f"R-hat computation failed: {e}")
        return np.nan


def _compute_ess(values: np.ndarray, method: str = "bulk") -> float:
    """Compute ESS using ArviZ."""
    n_chains, n_samples = values.shape

    if n_chains < 2:
        return np.nan
    if np.any(~np.isfinite(values)):
        return np.nan
    if np.allclose(values, values[0, 0]):
        return float(n_chains * n_samples)

    try:
        da = xr.DataArray(
            values,
            dims=["chain", "draw"],
            coords={"chain": np.arange(n_chains), "draw": np.arange(n_samples)},
        )
        return _extract_arviz_scalar(az.ess(da, method=method))
    except (ValueError, TypeError, RuntimeWarning) as e:
        warnings.warn(f"ESS computation failed: {e}")
        return np.nan


def compute_comprehensive_mixture_diagnostics(
    mcmc: MCMC,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "rank",
) -> dict[str, Any]:
    """Combine standard, label-invariant, relabeled, and label-switching diagnostics."""
    standard = compute_convergence_diagnostics(mcmc)

    label_invariant = compute_label_invariant_diagnostics(mcmc, x, y, method=method)

    relabeled = compute_diagnostics_on_relabeled(mcmc, method=method)

    samples = mcmc.get_samples()
    switching = check_label_switching(samples, param="k")

    primary_converged = label_invariant["converged"]

    relabeled_converged = relabeled["converged"]

    significant_switching = switching["switching_rate"] > 0.1

    if primary_converged and relabeled_converged:
        status = "converged"
        recommendation = "Chains have converged. Safe to use posterior samples."
    elif primary_converged and not relabeled_converged and significant_switching:
        status = "partial"
        recommendation = (
            "Log-likelihood converged but component parameters show high R-hat "
            "with significant label switching (rate={:.1%}). "
            "Use label-invariant summaries (e.g., mixture density, predictive)."
        ).format(switching["switching_rate"])
    elif primary_converged and not relabeled_converged:
        status = "partial"
        recommendation = (
            "Log-likelihood converged but component parameters show high R-hat "
            "without significant label switching. Consider longer chains "
            "or reparameterization."
        )
    else:
        status = "not_converged"
        recommendation = (
            "Log-likelihood R-hat is high, indicating true non-convergence. "
            "Consider: (1) longer warmup/sampling, (2) stronger priors, "
            "(3) reparameterization, or (4) simpler model."
        )

    return {
        "status": status,
        "recommendation": recommendation,
        "standard": standard,
        "label_invariant": label_invariant,
        "relabeled": relabeled,
        "label_switching": switching,
        "summary": {
            "rhat_log_lik": label_invariant["rhat_log_lik"],
            "rhat_relabeled_max": relabeled["max_rhat"],
            "rhat_standard_max": standard["max_rhat"],
            "switching_rate": switching["switching_rate"],
            "primary_converged": primary_converged,
            "relabeled_converged": relabeled_converged,
        },
    }
