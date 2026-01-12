"""Evaluation metrics for Hill Mixture MMM."""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb
from typing import Any

import numpy as np
from numpyro.infer import MCMC


def compute_effective_k(mcmc: MCMC, threshold: float = 0.05) -> dict[str, float]:
    """Count mixture components with weight > threshold."""
    samples = mcmc.get_samples()

    if "pis" not in samples:
        return {
            "effective_k_mean": 1.0,
            "effective_k_std": 0.0,
            "effective_k_samples": np.ones(1),
        }

    pis = np.array(samples["pis"])
    effective_k = (pis > threshold).sum(axis=-1)

    return {
        "effective_k_mean": float(effective_k.mean()),
        "effective_k_std": float(effective_k.std()),
        "effective_k_samples": effective_k,
    }


def _normalize_probability_mass(values: np.ndarray) -> np.ndarray:
    """Normalize a nonnegative discrete mass vector to sum to one."""
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape, 1.0 / max(clipped.size, 1), dtype=np.float64)
    return clipped / total


def compute_total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return the discrete total variation distance between two mass vectors."""
    p_mass = _normalize_probability_mass(p)
    q_mass = _normalize_probability_mass(q)
    if p_mass.shape != q_mass.shape:
        raise ValueError("p and q must have matching shapes")
    return float(0.5 * np.abs(p_mass - q_mass).sum())


def compute_cosine_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return the cosine distance between two mass (or density) vectors.

    cosine_distance = 1 - (p·q) / (‖p‖·‖q‖)

    Bounded in [0, 1] for non-negative vectors.  Scale-invariant: the ratio
    cancels any common factor such as bin width, so the result is the same
    whether *p* and *q* are probability masses or densities.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have matching shapes")
    dot = float(np.sum(p * q))
    norm = float(np.sqrt(np.sum(p**2) * np.sum(q**2)))
    if norm <= 0.0:
        return 0.0
    return float(1.0 - dot / norm)


def compute_nabc_distance(f: np.ndarray, g: np.ndarray) -> float:
    """Return the Normalized Area Between Curves (NABC) distance.

    NABC = sum|f(x_i) - g(x_i)| / sum max(f(x_i), g(x_i))

    Works on raw (unnormalized) curve arrays.  Bounded in [0, 1].
    Captures both shape *and* amplitude differences.
    """
    f = np.asarray(f, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if f.shape != g.shape:
        raise ValueError("f and g must have matching shapes")
    envelope = np.maximum(f, g)
    denom = float(envelope.sum())
    if denom <= 0.0:
        return 0.0
    return float(np.abs(f - g).sum() / denom)


def _scalar_ci(samples_arr: np.ndarray, true_val: float, tail: float) -> dict:
    ci_low = np.percentile(samples_arr, 100 * tail)
    ci_high = np.percentile(samples_arr, 100 * (1 - tail))
    return {
        "true": true_val,
        "mean": float(samples_arr.mean()),
        "std": float(samples_arr.std()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "in_ci": bool(ci_low <= true_val <= ci_high),
    }


_SCALAR_RECOVERY_PARAMS = [
    ("alpha", "alpha_true"),
    ("sigma", "sigma_true"),
    ("intercept", "intercept_true"),
    ("slope", "slope_true"),
]


def compute_parameter_recovery(mcmc: MCMC, meta: dict, ci_level: float = 0.95) -> dict[str, dict]:
    """Check if true parameters fall within posterior credible intervals."""
    samples = mcmc.get_samples()
    tail = (1 - ci_level) / 2
    results = {}

    for param, meta_key in _SCALAR_RECOVERY_PARAMS:
        if param in samples and meta_key in meta:
            results[param] = _scalar_ci(np.array(samples[param]), meta[meta_key], tail)

    if "pis" in samples and "pi_true" in meta:
        pis_samples = np.array(samples["pis"])
        pis_mean = pis_samples.mean(axis=0)
        K_fit = pis_samples.shape[1]
        K_true = len(meta["pi_true"])

        results["pis"] = {
            "K_fit": K_fit,
            "K_true": K_true,
            "pis_mean": pis_mean.tolist(),
            "pi_true": meta["pi_true"].tolist(),
        }

    return results


def compute_latent_recovery(mu_true: np.ndarray, mu_samples: np.ndarray) -> dict[str, float]:
    """Measure recovery of the noise-free latent mean function."""
    mu_true = np.asarray(mu_true, dtype=np.float64)
    mu_samples = np.asarray(mu_samples, dtype=np.float64)

    if mu_samples.ndim != 2:
        raise ValueError("mu_samples must have shape (n_samples, T)")
    if mu_true.shape[0] != mu_samples.shape[1]:
        raise ValueError("mu_true and mu_samples must align on time dimension")

    mu_mean = mu_samples.mean(axis=0)
    q05 = np.quantile(mu_samples, 0.05, axis=0)
    q95 = np.quantile(mu_samples, 0.95, axis=0)
    q025 = np.quantile(mu_samples, 0.025, axis=0)
    q975 = np.quantile(mu_samples, 0.975, axis=0)
    denom = np.maximum(np.abs(mu_true), 1e-8)
    rmse = float(np.sqrt(np.mean((mu_mean - mu_true) ** 2)))
    scale = float(max(np.max(mu_true) - np.min(mu_true), 1e-8))

    return {
        "mape": float(np.mean(np.abs((mu_mean - mu_true) / denom)) * 100.0),
        "mae": float(np.mean(np.abs(mu_mean - mu_true))),
        "rmse": rmse,
        "nrmse": float(rmse / scale),
        "crps": float(np.mean(_crps_ensemble(mu_true, mu_samples))),
        "coverage_90": float(np.mean((mu_true >= q05) & (mu_true <= q95))),
        "coverage_95": float(np.mean((mu_true >= q025) & (mu_true <= q975))),
    }


def _crps_ensemble(y_true: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    """Compute empirical CRPS per observation for posterior samples."""
    if y_samples.ndim != 2:
        raise ValueError("y_samples must have shape (n_samples, T)")
    if y_true.shape[0] != y_samples.shape[1]:
        raise ValueError("y_true and y_samples must align on time dimension")
    n_samples = y_samples.shape[0]
    sorted_samples = np.sort(y_samples, axis=0)
    coeffs = (2 * np.arange(1, n_samples + 1, dtype=np.float64) - n_samples - 1)[:, None]
    pairwise_term = np.sum(coeffs * sorted_samples, axis=0) / (n_samples**2)
    observation_term = np.mean(np.abs(y_samples - y_true[None, :]), axis=0)
    return observation_term - pairwise_term


def compute_predictive_metrics(y_true: np.ndarray, y_samples: np.ndarray) -> dict[str, float]:
    """Compute predictive summary metrics (MAPE, RMSE, nRMSE, CRPS, coverage)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_samples = np.asarray(y_samples, dtype=np.float64)
    y_pred_mean = y_samples.mean(axis=0)
    q05 = np.quantile(y_samples, 0.05, axis=0)
    q95 = np.quantile(y_samples, 0.95, axis=0)

    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_pred_mean - y_true) / denom)) * 100.0)
    rmse = float(np.sqrt(np.mean((y_pred_mean - y_true) ** 2)))
    scale = float(max(np.max(y_true) - np.min(y_true), 1e-8))
    nrmse = float(rmse / scale)
    crps = float(np.mean(_crps_ensemble(y_true, y_samples)))
    coverage = float(np.mean((y_true >= q05) & (y_true <= q95)))

    return {
        "mape": mape,
        "rmse": rmse,
        "nrmse": nrmse,
        "crps": crps,
        "coverage_90": coverage,
        "y_pred_mean": y_pred_mean,
        "q05": q05,
        "q95": q95,
    }


def compute_delta_loo(loo_model: dict, loo_baseline: dict) -> dict[str, float]:
    """Compute LOO-CV improvement relative to baseline (positive = better)."""
    if np.isnan(loo_model.get("elpd_loo", np.nan)) or np.isnan(
        loo_baseline.get("elpd_loo", np.nan)
    ):
        return {"delta_loo": np.nan, "se": np.nan, "significant": False}

    delta = loo_model["elpd_loo"] - loo_baseline["elpd_loo"]
    se = np.sqrt(loo_model["se"] ** 2 + loo_baseline["se"] ** 2)

    return {
        "delta_loo": float(delta),
        "se": float(se),
        "significant": bool(abs(delta) > 2 * se),
    }


def summarize_component_posterior(
    samples: dict[str, np.ndarray],
    *,
    scale_reference: float = 1.0,
    weight_threshold: float = 0.05,
) -> dict[str, Any] | None:
    """Summarize posterior component parameters for recovery and stability checks."""
    if not {"A", "k", "n"}.issubset(samples):
        return None

    scale_reference = float(max(scale_reference, 1e-6))
    A = np.asarray(samples["A"], dtype=np.float64)
    k = np.asarray(samples["k"], dtype=np.float64)
    n = np.asarray(samples["n"], dtype=np.float64)

    if A.ndim == 1:
        A = A[:, None]
    if k.ndim == 1:
        k = k[:, None]
    if n.ndim == 1:
        n = n[:, None]

    if "pis" in samples:
        pis = np.asarray(samples["pis"], dtype=np.float64)
        if pis.ndim == 1:
            pis = pis[:, None]
    else:
        pis = np.ones((A.shape[0], 1), dtype=np.float64)

    components = []
    for idx in range(A.shape[1]):
        pi_mean = float(pis[:, idx].mean())
        k_mean = float(k[:, idx].mean())
        components.append(
            {
                "index": idx,
                "A_mean": float(A[:, idx].mean()),
                "A_std": float(A[:, idx].std()),
                "k_mean": k_mean,
                "k_std": float(k[:, idx].std()),
                "k_ratio_mean": float(k_mean / scale_reference),
                "k_ratio_std": float(k[:, idx].std() / scale_reference),
                "n_mean": float(n[:, idx].mean()),
                "n_std": float(n[:, idx].std()),
                "pi_mean": pi_mean,
                "pi_std": float(pis[:, idx].std()),
                "active": bool(pi_mean > weight_threshold),
            }
        )

    return {
        "K_total": len(components),
        "K_active": int(sum(component["active"] for component in components)),
        "weight_threshold": float(weight_threshold),
        "scale_reference": scale_reference,
        "components": components,
    }


def summarize_true_components(
    meta: dict[str, Any], weight_threshold: float = 0.05
) -> dict[str, Any]:
    """Build a component summary from synthetic DGP metadata."""
    scale_reference = float(max(meta.get("s_median", 1.0), 1e-6))
    A_true = np.asarray(meta["A_true"], dtype=np.float64)
    k_true = np.asarray(meta["k_true"], dtype=np.float64)
    n_true = np.asarray(meta["n_true"], dtype=np.float64)
    pi_true = np.asarray(meta["pi_true"], dtype=np.float64)

    components = []
    for idx, (A_i, k_i, n_i, pi_i) in enumerate(zip(A_true, k_true, n_true, pi_true, strict=True)):
        components.append(
            {
                "index": idx,
                "A_mean": float(A_i),
                "A_std": 0.0,
                "k_mean": float(k_i),
                "k_std": 0.0,
                "k_ratio_mean": float(k_i / scale_reference),
                "k_ratio_std": 0.0,
                "n_mean": float(n_i),
                "n_std": 0.0,
                "pi_mean": float(pi_i),
                "pi_std": 0.0,
                "active": bool(pi_i > weight_threshold),
            }
        )

    return {
        "K_total": len(components),
        "K_active": int(sum(component["active"] for component in components)),
        "weight_threshold": float(weight_threshold),
        "scale_reference": scale_reference,
        "components": components,
    }


def _extract_component_summary(payload: dict[str, Any]) -> tuple[dict[str, Any], str, int | None]:
    """Normalize a case summary or component summary payload."""
    if "component_summary" in payload:
        component_summary = payload["component_summary"]
        label = str(payload.get("label", "case"))
        seed = int(payload["seed"]) if "seed" in payload else None
        return component_summary, label, seed

    if "components" in payload and "K_total" in payload:
        label = str(payload.get("label", "case"))
        seed = int(payload["seed"]) if "seed" in payload else None
        return payload, label, seed

    raise ValueError("payload must contain either component_summary or a component summary itself")


def _select_components(
    component_summary: dict[str, Any], *, active_only: bool
) -> list[dict[str, Any]]:
    """Return selected components, optionally filtering to the active subset."""
    components = [dict(component) for component in component_summary.get("components", [])]
    if not active_only:
        return components
    active = [component for component in components if component.get("active", False)]
    if active:
        return active
    if not components:
        return []
    return [max(components, key=lambda component: float(component.get("pi_mean", 0.0)))]


def _active_components(component_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active components, falling back to the highest-weight component if needed."""
    return _select_components(component_summary, active_only=True)


def _component_weights(component_summary: dict[str, Any], *, active_only: bool) -> np.ndarray:
    """Return normalized component weights."""
    components = _select_components(component_summary, active_only=active_only)
    if not components:
        return np.asarray([1.0], dtype=np.float64)
    weights = np.asarray(
        [float(component["pi_mean"]) for component in components], dtype=np.float64
    )
    return _normalize_probability_mass(weights)


def _component_curve(component: dict[str, Any], u_grid: np.ndarray) -> np.ndarray:
    """Return the normalized Hill effect curve for one component."""
    A_mean = float(component["A_mean"])
    k_ratio = float(max(component["k_ratio_mean"], 1e-6))
    n_mean = float(max(component["n_mean"], 1e-6))
    numerator = np.power(u_grid, n_mean)
    denominator = np.power(k_ratio, n_mean) + numerator + 1e-12
    return A_mean * numerator / denominator


def _component_curve_mass(component: dict[str, Any], u_grid: np.ndarray) -> np.ndarray:
    """Return normalized incremental response mass for one component on the grid."""
    curve = _component_curve(component, u_grid)
    increments = np.diff(curve, prepend=0.0)
    return _normalize_probability_mass(increments)


def compute_component_distance_matrix(
    component_summary: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
    active_only: bool = True,
) -> dict[str, Any]:
    """Return active-component masses, weights, and pairwise TV distances."""
    components = _select_components(component_summary, active_only=active_only)
    if not components:
        return {
            "components": [],
            "weights": np.zeros((0,), dtype=np.float64),
            "masses": np.zeros((0, 0), dtype=np.float64),
            "distance_matrix": np.zeros((0, 0), dtype=np.float64),
        }

    u_grid = np.linspace(0.0, curve_grid_max, grid_size, dtype=np.float64)
    masses = np.stack(
        [_component_curve_mass(component, u_grid) for component in components], axis=0
    )
    weights = _normalize_probability_mass(
        np.asarray([float(component["pi_mean"]) for component in components], dtype=np.float64)
    )
    k = len(components)
    distance_matrix = np.zeros((k, k), dtype=np.float64)
    for left_idx, right_idx in combinations(range(k), 2):
        tv = compute_total_variation_distance(masses[left_idx], masses[right_idx])
        distance_matrix[left_idx, right_idx] = tv
        distance_matrix[right_idx, left_idx] = tv
    return {
        "components": components,
        "weights": weights,
        "masses": masses,
        "distance_matrix": distance_matrix,
    }


def compute_component_nabc_distance_matrix(
    component_summary: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
    active_only: bool = True,
) -> dict[str, Any]:
    """Return active-component raw curves, weights, and pairwise NABC distances."""
    components = _select_components(component_summary, active_only=active_only)
    if not components:
        return {
            "components": [],
            "weights": np.zeros((0,), dtype=np.float64),
            "curves": np.zeros((0, 0), dtype=np.float64),
            "distance_matrix": np.zeros((0, 0), dtype=np.float64),
        }

    u_grid = np.linspace(0.0, curve_grid_max, grid_size, dtype=np.float64)
    curves = np.stack([_component_curve(component, u_grid) for component in components], axis=0)
    weights = _normalize_probability_mass(
        np.asarray([float(component["pi_mean"]) for component in components], dtype=np.float64)
    )
    k = len(components)
    distance_matrix = np.zeros((k, k), dtype=np.float64)
    for left_idx, right_idx in combinations(range(k), 2):
        d = compute_nabc_distance(curves[left_idx], curves[right_idx])
        distance_matrix[left_idx, right_idx] = d
        distance_matrix[right_idx, left_idx] = d
    return {
        "components": components,
        "weights": weights,
        "curves": curves,
        "distance_matrix": distance_matrix,
    }


def compute_component_curve_tv_separation(
    component_summary: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Average pairwise TV separation for active components represented as normalized curves."""
    components = _active_components(component_summary)
    if len(components) < 2:
        return {
            "num_components": len(components),
            "pair_count": 0,
            "pairwise_tv": [],
            "mean_pairwise_tv": 0.0,
        }

    matrix = compute_component_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    masses = list(matrix["masses"])
    pairwise_tv = [
        {
            "left_index": int(left_component["index"]),
            "right_index": int(right_component["index"]),
            "tv": compute_total_variation_distance(left_mass, right_mass),
        }
        for (left_component, left_mass), (right_component, right_mass) in combinations(
            list(zip(components, masses, strict=True)), 2
        )
    ]
    return {
        "num_components": len(components),
        "pair_count": len(pairwise_tv),
        "pairwise_tv": pairwise_tv,
        "mean_pairwise_tv": float(np.mean([item["tv"] for item in pairwise_tv], dtype=np.float64)),
    }


def compute_component_curve_cosine_separation(
    component_summary: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Average pairwise cosine distance for active components as normalized curve masses."""
    components = _active_components(component_summary)
    if len(components) < 2:
        return {
            "num_components": len(components),
            "pair_count": 0,
            "pairwise_cosine": [],
            "mean_pairwise_cosine": 0.0,
        }

    u_grid = np.linspace(0.0, curve_grid_max, grid_size, dtype=np.float64)
    masses = [_component_curve_mass(component, u_grid) for component in components]
    pairwise_cosine = [
        {
            "left_index": int(left_component["index"]),
            "right_index": int(right_component["index"]),
            "cosine_distance": compute_cosine_distance(left_mass, right_mass),
        }
        for (left_component, left_mass), (right_component, right_mass) in combinations(
            list(zip(components, masses, strict=True)), 2
        )
    ]
    return {
        "num_components": len(components),
        "pair_count": len(pairwise_cosine),
        "pairwise_cosine": pairwise_cosine,
        "mean_pairwise_cosine": float(
            np.mean([item["cosine_distance"] for item in pairwise_cosine], dtype=np.float64)
        ),
    }


def compute_component_curve_nabc_separation(
    component_summary: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Average pairwise NABC separation for active components using raw Hill curves."""
    components = _active_components(component_summary)
    if len(components) < 2:
        return {
            "num_components": len(components),
            "pair_count": 0,
            "pairwise_nabc": [],
            "mean_pairwise_nabc": 0.0,
        }

    matrix = compute_component_nabc_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    curves = list(matrix["curves"])
    pairwise_nabc = [
        {
            "left_index": int(left_component["index"]),
            "right_index": int(right_component["index"]),
            "nabc": compute_nabc_distance(left_curve, right_curve),
        }
        for (left_component, left_curve), (right_component, right_curve) in combinations(
            list(zip(components, curves, strict=True)), 2
        )
    ]
    return {
        "num_components": len(components),
        "pair_count": len(pairwise_nabc),
        "pairwise_nabc": pairwise_nabc,
        "mean_pairwise_nabc": float(
            np.mean([item["nabc"] for item in pairwise_nabc], dtype=np.float64)
        ),
    }


def compute_similarity_adjusted_effective_count(
    component_summary: dict[str, Any],
    *,
    gamma: float = 0.5,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Estimate a continuous effective component count using curve similarity and flattened weights."""
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")

    components = _active_components(component_summary)
    if not components:
        return {
            "gamma": float(gamma),
            "num_components": 0,
            "effective_count": 0.0,
            "weights": [],
            "flattened_weights": [],
            "similarity_matrix": [],
        }

    matrix = compute_component_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    curve_masses = np.asarray(matrix["masses"], dtype=np.float64)
    weights = np.asarray(matrix["weights"], dtype=np.float64)
    flattened = np.power(weights, gamma)
    flattened_total = float(flattened.sum())
    if flattened_total <= 0.0:
        flattened = np.full(weights.shape, 1.0 / len(weights), dtype=np.float64)
    else:
        flattened = flattened / flattened_total

    k = len(components)
    similarity = np.eye(k, dtype=np.float64)
    for left_idx, right_idx in combinations(range(k), 2):
        tv = compute_total_variation_distance(curve_masses[left_idx], curve_masses[right_idx])
        similarity_value = float(1.0 - tv)
        similarity[left_idx, right_idx] = similarity_value
        similarity[right_idx, left_idx] = similarity_value

    denominator = float(flattened @ similarity @ flattened)
    effective_count = float(1.0 / max(denominator, 1e-12))
    return {
        "gamma": float(gamma),
        "num_components": k,
        "effective_count": effective_count,
        "weights": weights.tolist(),
        "flattened_weights": flattened.tolist(),
        "similarity_matrix": similarity.tolist(),
    }


def compute_nabc_effective_count(
    component_summary: dict[str, Any],
    *,
    gamma: float = 0.5,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Leinster-Cobbold D2^Z effective count using NABC-based similarity on raw Hill curves."""
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")

    components = _active_components(component_summary)
    if not components:
        return {
            "gamma": float(gamma),
            "num_components": 0,
            "effective_count": 0.0,
            "weights": [],
            "flattened_weights": [],
            "similarity_matrix": [],
        }

    matrix = compute_component_nabc_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    weights = np.asarray(matrix["weights"], dtype=np.float64)
    flattened = np.power(weights, gamma)
    flattened_total = float(flattened.sum())
    if flattened_total <= 0.0:
        flattened = np.full(weights.shape, 1.0 / len(weights), dtype=np.float64)
    else:
        flattened = flattened / flattened_total

    k = len(components)
    nabc_distances = np.asarray(matrix["distance_matrix"], dtype=np.float64)
    similarity = np.eye(k, dtype=np.float64)
    for left_idx, right_idx in combinations(range(k), 2):
        similarity_value = float(1.0 - nabc_distances[left_idx, right_idx])
        similarity[left_idx, right_idx] = similarity_value
        similarity[right_idx, left_idx] = similarity_value

    denominator = float(flattened @ similarity @ flattened)
    effective_count = float(1.0 / max(denominator, 1e-12))
    return {
        "gamma": float(gamma),
        "num_components": k,
        "effective_count": effective_count,
        "weights": weights.tolist(),
        "flattened_weights": flattened.tolist(),
        "similarity_matrix": similarity.tolist(),
    }


def compute_shannon_effective_count(component_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the Hill q=1 effective count from all component weights."""
    weights = _component_weights(component_summary, active_only=False)
    effective_count = float(np.exp(-np.sum(weights * np.log(np.clip(weights, 1e-12, None)))))
    return {
        "q": 1.0,
        "num_components": int(len(weights)),
        "effective_count": effective_count,
        "weights": weights.tolist(),
    }


def compute_inverse_simpson_effective_count(component_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the Hill q=2 effective count from all component weights."""
    weights = _component_weights(component_summary, active_only=False)
    effective_count = float(1.0 / np.sum(weights**2))
    return {
        "q": 2.0,
        "num_components": int(len(weights)),
        "effective_count": effective_count,
        "weights": weights.tolist(),
    }


def _exp_similarity_matrix(distance_matrix: np.ndarray, lambda_: float) -> np.ndarray:
    """Convert a distance matrix to an exponential-kernel similarity matrix."""
    if lambda_ <= 0.0:
        raise ValueError("lambda_ must be positive")
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    similarity = np.exp(-float(lambda_) * distance_matrix)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def compute_rao_quadratic_entropy_equivalent_count(
    component_summary: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Return Rao's quadratic entropy and its equivalent-number transform."""
    matrix = compute_component_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
        active_only=False,
    )
    weights = np.asarray(matrix["weights"], dtype=np.float64)
    if len(weights) == 0:
        return {"num_components": 0, "rao_q": 0.0, "effective_count": 0.0, "weights": []}
    distance_matrix = np.asarray(matrix["distance_matrix"], dtype=np.float64)
    rao_q = float(weights @ distance_matrix @ weights)
    effective_count = float(1.0 / max(1.0 - rao_q, 1e-12))
    return {
        "num_components": int(len(weights)),
        "rao_q": rao_q,
        "effective_count": effective_count,
        "weights": weights.tolist(),
    }


def compute_leinster_cobbold_effective_count(
    component_summary: dict[str, Any],
    *,
    q: float = 1.0,
    lambda_: float = 6.0,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Return a similarity-sensitive effective count in the Leinster-Cobbold family."""
    if q <= 0.0:
        raise ValueError("q must be positive")
    matrix = compute_component_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
        active_only=False,
    )
    weights = np.asarray(matrix["weights"], dtype=np.float64)
    if len(weights) == 0:
        return {
            "q": float(q),
            "lambda": float(lambda_),
            "num_components": 0,
            "effective_count": 0.0,
            "weights": [],
            "similarity_matrix": [],
        }

    similarity = _exp_similarity_matrix(
        np.asarray(matrix["distance_matrix"], dtype=np.float64), lambda_
    )
    zp = similarity @ weights
    if np.isclose(q, 1.0):
        effective_count = float(np.exp(-np.sum(weights * np.log(np.clip(zp, 1e-12, None)))))
    else:
        effective_count = float(
            np.power(
                np.sum(weights * np.power(np.clip(zp, 1e-12, None), q - 1.0)), -1.0 / (q - 1.0)
            )
        )
    return {
        "q": float(q),
        "lambda": float(lambda_),
        "num_components": int(len(weights)),
        "effective_count": effective_count,
        "weights": weights.tolist(),
        "similarity_matrix": similarity.tolist(),
    }


def compute_soft_component_count(
    component_summary: dict[str, Any],
    *,
    lambda_: float = 6.0,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Return a soft pairwise effective-count heuristic from weights and distances."""
    if lambda_ <= 0.0:
        raise ValueError("lambda_ must be positive")
    matrix = compute_component_distance_matrix(
        component_summary,
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    weights = np.asarray(matrix["weights"], dtype=np.float64)
    distance_matrix = np.asarray(matrix["distance_matrix"], dtype=np.float64)
    if len(weights) <= 1:
        return {
            "lambda": float(lambda_),
            "num_components": int(len(weights)),
            "effective_count": 1.0 if len(weights) == 1 else 0.0,
            "weights": weights.tolist(),
        }

    total = 1.0
    for left_idx, right_idx in combinations(range(len(weights)), 2):
        total += (
            2.0
            * min(float(weights[left_idx]), float(weights[right_idx]))
            * (1.0 - float(np.exp(-lambda_ * distance_matrix[left_idx, right_idx])))
        )
    return {
        "lambda": float(lambda_),
        "num_components": int(len(weights)),
        "effective_count": float(total),
        "weights": weights.tolist(),
    }


def _pair_metrics(
    reference_component: dict[str, Any],
    candidate_component: dict[str, Any],
    u_grid: np.ndarray,
) -> dict[str, float]:
    """Compute component-level recovery metrics for one matched pair."""
    reference_curve = _component_curve(reference_component, u_grid)
    candidate_curve = _component_curve(candidate_component, u_grid)
    curve_rmse = float(np.sqrt(np.mean((reference_curve - candidate_curve) ** 2)))
    curve_scale = max(
        float(np.max(np.abs(reference_curve))),
        float(np.max(np.abs(candidate_curve))),
        1.0,
    )
    avg_weight = 0.5 * (
        float(reference_component["pi_mean"]) + float(candidate_component["pi_mean"])
    )
    return {
        "curve_rmse": curve_rmse,
        "curve_nrmse": float(curve_rmse / curve_scale),
        "pi_abs_error": float(
            abs(float(reference_component["pi_mean"]) - float(candidate_component["pi_mean"]))
        ),
        "A_rel_error": float(
            abs(float(reference_component["A_mean"]) - float(candidate_component["A_mean"]))
            / max(abs(float(reference_component["A_mean"])), 1e-6)
        ),
        "k_ratio_rel_error": float(
            abs(
                float(reference_component["k_ratio_mean"])
                - float(candidate_component["k_ratio_mean"])
            )
            / max(abs(float(reference_component["k_ratio_mean"])), 1e-6)
        ),
        "n_abs_error": float(
            abs(float(reference_component["n_mean"]) - float(candidate_component["n_mean"]))
        ),
        "avg_weight": float(avg_weight),
    }


def compute_component_set_alignment(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Align two component sets with a permutation-invariant matching rule."""
    reference_summary, reference_label, reference_seed = _extract_component_summary(reference)
    candidate_summary, candidate_label, candidate_seed = _extract_component_summary(candidate)
    reference_components = _active_components(reference_summary)
    candidate_components = _active_components(candidate_summary)

    u_grid = np.linspace(0.0, curve_grid_max, grid_size, dtype=np.float64)
    reference_weight_total = float(sum(component["pi_mean"] for component in reference_components))
    candidate_weight_total = float(sum(component["pi_mean"] for component in candidate_components))

    if not reference_components and not candidate_components:
        return {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_seed": reference_seed,
            "candidate_seed": candidate_seed,
            "reference_active_k": 0,
            "candidate_active_k": 0,
            "matched_components": [],
            "matched_weight": 0.0,
            "unmatched_reference_weight": 0.0,
            "unmatched_candidate_weight": 0.0,
            "weighted_curve_rmse": 0.0,
            "weighted_curve_nrmse": 0.0,
            "mean_curve_rmse": 0.0,
            "mean_curve_nrmse": 0.0,
            "weighted_pi_abs_error": 0.0,
            "mean_pi_abs_error": 0.0,
            "mean_A_rel_error": 0.0,
            "mean_k_ratio_rel_error": 0.0,
            "mean_n_abs_error": 0.0,
            "assignment_cost": 0.0,
        }

    if len(reference_components) <= len(candidate_components):
        assignment_iter = (
            list(zip(range(len(reference_components)), candidate_order, strict=True))
            for candidate_choice in combinations(
                range(len(candidate_components)), len(reference_components)
            )
            for candidate_order in permutations(candidate_choice)
        )
    else:
        assignment_iter = (
            list(zip(reference_order, range(len(candidate_components)), strict=True))
            for reference_choice in combinations(
                range(len(reference_components)), len(candidate_components)
            )
            for reference_order in permutations(reference_choice)
        )

    best_result: dict[str, Any] | None = None
    for pairs in assignment_iter:
        matched_components = []
        weighted_curve_rmse = 0.0
        weighted_curve_nrmse = 0.0
        weighted_pi_abs_error = 0.0
        mean_curve_rmse = []
        mean_curve_nrmse = []
        mean_pi_abs_error = []
        mean_A_rel_error = []
        mean_k_ratio_rel_error = []
        mean_n_abs_error = []
        matched_reference_weight = 0.0
        matched_candidate_weight = 0.0
        matched_avg_weight = 0.0

        for reference_idx, candidate_idx in pairs:
            reference_component = reference_components[reference_idx]
            candidate_component = candidate_components[candidate_idx]
            metrics = _pair_metrics(reference_component, candidate_component, u_grid)
            matched_components.append(
                {
                    "reference_component_index": int(reference_component["index"]),
                    "candidate_component_index": int(candidate_component["index"]),
                    **metrics,
                }
            )
            matched_reference_weight += float(reference_component["pi_mean"])
            matched_candidate_weight += float(candidate_component["pi_mean"])
            matched_avg_weight += metrics["avg_weight"]
            weighted_curve_rmse += metrics["avg_weight"] * metrics["curve_rmse"]
            weighted_curve_nrmse += metrics["avg_weight"] * metrics["curve_nrmse"]
            weighted_pi_abs_error += metrics["avg_weight"] * metrics["pi_abs_error"]
            mean_curve_rmse.append(metrics["curve_rmse"])
            mean_curve_nrmse.append(metrics["curve_nrmse"])
            mean_pi_abs_error.append(metrics["pi_abs_error"])
            mean_A_rel_error.append(metrics["A_rel_error"])
            mean_k_ratio_rel_error.append(metrics["k_ratio_rel_error"])
            mean_n_abs_error.append(metrics["n_abs_error"])

        unmatched_reference_weight = max(reference_weight_total - matched_reference_weight, 0.0)
        unmatched_candidate_weight = max(candidate_weight_total - matched_candidate_weight, 0.0)
        cost = weighted_curve_nrmse + unmatched_reference_weight + unmatched_candidate_weight

        result = {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_seed": reference_seed,
            "candidate_seed": candidate_seed,
            "reference_active_k": len(reference_components),
            "candidate_active_k": len(candidate_components),
            "matched_components": matched_components,
            "matched_weight": float(matched_avg_weight),
            "unmatched_reference_weight": float(unmatched_reference_weight),
            "unmatched_candidate_weight": float(unmatched_candidate_weight),
            "weighted_curve_rmse": float(weighted_curve_rmse / max(matched_avg_weight, 1e-6)),
            "weighted_curve_nrmse": float(weighted_curve_nrmse / max(matched_avg_weight, 1e-6)),
            "mean_curve_rmse": float(np.mean(mean_curve_rmse)) if mean_curve_rmse else 0.0,
            "mean_curve_nrmse": float(np.mean(mean_curve_nrmse)) if mean_curve_nrmse else 0.0,
            "weighted_pi_abs_error": float(weighted_pi_abs_error / max(matched_avg_weight, 1e-6)),
            "mean_pi_abs_error": float(np.mean(mean_pi_abs_error)) if mean_pi_abs_error else 0.0,
            "mean_A_rel_error": float(np.mean(mean_A_rel_error)) if mean_A_rel_error else 0.0,
            "mean_k_ratio_rel_error": float(np.mean(mean_k_ratio_rel_error))
            if mean_k_ratio_rel_error
            else 0.0,
            "mean_n_abs_error": float(np.mean(mean_n_abs_error)) if mean_n_abs_error else 0.0,
            "assignment_cost": float(cost),
        }
        if best_result is None or result["assignment_cost"] < best_result["assignment_cost"]:
            best_result = result

    if best_result is None:
        return {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_seed": reference_seed,
            "candidate_seed": candidate_seed,
            "reference_active_k": len(reference_components),
            "candidate_active_k": len(candidate_components),
            "matched_components": [],
            "matched_weight": 0.0,
            "unmatched_reference_weight": float(reference_weight_total),
            "unmatched_candidate_weight": float(candidate_weight_total),
            "weighted_curve_rmse": 0.0,
            "weighted_curve_nrmse": 0.0,
            "mean_curve_rmse": 0.0,
            "mean_curve_nrmse": 0.0,
            "weighted_pi_abs_error": 0.0,
            "mean_pi_abs_error": 0.0,
            "mean_A_rel_error": 0.0,
            "mean_k_ratio_rel_error": 0.0,
            "mean_n_abs_error": 0.0,
            "assignment_cost": float(reference_weight_total + candidate_weight_total),
        }
    return best_result


def compute_permutation_invariant_component_recovery(
    samples: dict[str, np.ndarray],
    meta: dict[str, Any],
    *,
    weight_threshold: float = 0.05,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any] | None:
    """Compare recovered mixture components to the true DGP up to permutation."""
    required = {"A_true", "k_true", "n_true", "pi_true"}
    if not required.issubset(meta):
        return None

    posterior_summary = summarize_component_posterior(
        samples,
        scale_reference=float(meta.get("s_median", 1.0)),
        weight_threshold=weight_threshold,
    )
    if posterior_summary is None:
        return None

    true_summary = summarize_true_components(meta, weight_threshold=weight_threshold)
    alignment = compute_component_set_alignment(
        {"component_summary": true_summary, "label": "true_components"},
        {"component_summary": posterior_summary, "label": "posterior_components"},
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    return {
        "K_true": int(true_summary["K_total"]),
        "K_true_active": int(true_summary["K_active"]),
        "K_posterior": int(posterior_summary["K_total"]),
        "K_posterior_active": int(posterior_summary["K_active"]),
        "effective_k_error": float(abs(posterior_summary["K_active"] - true_summary["K_active"])),
        **alignment,
    }


def compute_across_seed_component_stability(
    summaries: list[dict[str, Any]],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Measure how stably a model recovers components across benchmark seeds."""
    if len(summaries) < 2:
        raise ValueError("need at least two summaries to evaluate across-seed stability")

    normalized = []
    for summary in summaries:
        component_summary, label, seed = _extract_component_summary(summary)
        normalized.append(
            {
                "label": label,
                "seed": seed,
                "component_summary": component_summary,
            }
        )

    pairwise = []
    for left, right in combinations(normalized, 2):
        pairwise.append(
            compute_component_set_alignment(
                left,
                right,
                curve_grid_max=curve_grid_max,
                grid_size=grid_size,
            )
        )

    active_ks = np.asarray(
        [entry["component_summary"]["K_active"] for entry in normalized],
        dtype=np.float64,
    )
    active_k_values, active_k_counts = np.unique(active_ks, return_counts=True)
    mode_idx = int(np.argmax(active_k_counts))

    def _aggregate(name: str) -> dict[str, float]:
        values = np.asarray([pair[name] for pair in pairwise], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
        }

    unmatched_totals = np.asarray(
        [
            pair["unmatched_reference_weight"] + pair["unmatched_candidate_weight"]
            for pair in pairwise
        ],
        dtype=np.float64,
    )

    return {
        "num_seeds": len(normalized),
        "expected_pair_count": comb(len(normalized), 2),
        "pair_count": len(pairwise),
        "seed_labels": [entry["label"] for entry in normalized],
        "seeds": [entry["seed"] for entry in normalized],
        "mode_active_k": int(active_k_values[mode_idx]),
        "active_k_mean": float(active_ks.mean()),
        "active_k_std": float(active_ks.std()),
        "active_k_consistency": float(active_k_counts[mode_idx] / len(active_ks)),
        "weighted_curve_nrmse": _aggregate("weighted_curve_nrmse"),
        "weighted_curve_rmse": _aggregate("weighted_curve_rmse"),
        "weighted_pi_abs_error": _aggregate("weighted_pi_abs_error"),
        "mean_k_ratio_rel_error": _aggregate("mean_k_ratio_rel_error"),
        "mean_A_rel_error": _aggregate("mean_A_rel_error"),
        "mean_n_abs_error": _aggregate("mean_n_abs_error"),
        "unmatched_total_weight": {
            "mean": float(unmatched_totals.mean()),
            "std": float(unmatched_totals.std()),
            "max": float(unmatched_totals.max()),
        },
        "pairwise": pairwise,
    }
