#!/usr/bin/env python
"""Run a controlled component-spacing sweep where Hill k-separation is the main axis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Allow NumPyro multi-chain CPU execution before JAX initializes.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    run_prepared_synthetic_benchmark_case,
)
from hill_mixture_mmm.data import (
    ControlledKSpacingConfig,
    generate_controlled_k_spacing_data,
)
from hill_mixture_mmm.metrics import (
    compute_component_curve_tv_separation,
    summarize_true_components,
)


DEFAULT_DELTAS = [0.20, 0.45]
MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3"]
K_TRUE_ORDER = [1, 2, 3]
MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
MODEL_COLORS = {
    "single_hill": "#1f77b4",
    "mixture_k2": "#9467bd",
    "mixture_k3": "#ff7f0e",
}
K_TRUE_MARKERS = {1: "o", 2: "s", 3: "^"}
CONTROLLED_COMPONENT_DEFAULTS = {
    1: {
        "pi_true": (1.0, 1.0, 1.0),
        "A_true": (50.0, 50.0, 50.0),
    },
    2: {
        "pi_true": (0.65, 0.35, 1.0),
        "A_true": (38.0, 78.0, 50.0),
    },
    3: {
        "pi_true": (0.55, 0.30, 0.15),
        "A_true": (30.0, 55.0, 85.0),
    },
}


def _build_run_config(model_name: str, seed: int, *, quick: bool) -> BenchmarkRunConfig:
    """Return inference settings for the controlled sweep."""
    if quick:
        if model_name == "single_hill":
            return BenchmarkRunConfig(
                seed=seed,
                num_warmup=300,
                num_samples=300,
                num_chains=2,
                target_accept_prob=0.9,
                progress_bar=False,
            )
        if model_name == "mixture_k2":
            return BenchmarkRunConfig(
                seed=seed,
                num_warmup=600,
                num_samples=450,
                num_chains=2,
                target_accept_prob=0.992,
                max_tree_depth=14,
                dense_mass=True,
                init_strategy="median",
                progress_bar=False,
            )
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=800,
            num_samples=500,
            num_chains=2,
            target_accept_prob=0.996,
            max_tree_depth=16,
            dense_mass=True,
            init_strategy="median",
            progress_bar=False,
        )

    if model_name == "single_hill":
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=700,
            num_samples=700,
            num_chains=2,
            target_accept_prob=0.9,
            progress_bar=False,
        )
    if model_name == "mixture_k2":
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=1000,
            num_samples=800,
            num_chains=2,
            target_accept_prob=0.995,
            max_tree_depth=14,
            dense_mass=True,
            init_strategy="median",
            progress_bar=False,
        )
    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=1400,
        num_samples=1000,
        num_chains=2,
        target_accept_prob=0.997,
        max_tree_depth=16,
        dense_mass=True,
        init_strategy="median",
        progress_bar=False,
    )


def _build_data_config(
    *,
    k_true: int,
    seed: int,
    spacing_delta: float,
    args: argparse.Namespace,
) -> ControlledKSpacingConfig:
    defaults = CONTROLLED_COMPONENT_DEFAULTS[int(k_true)]
    return ControlledKSpacingConfig(
        K_true=int(k_true),
        T=int(args.T),
        seed=seed,
        spacing_delta=float(spacing_delta),
        center_k_ratio=float(args.center_k_ratio),
        raw_spend_lognormal_sigma=float(args.raw_spend_sigma),
        pi_true=tuple(defaults["pi_true"]),
        A_true=tuple(defaults["A_true"]),
        n_true=(float(args.n_true), float(args.n_true), float(args.n_true)),
    )


def _record_result(result, *, spacing_delta: float, config: ControlledKSpacingConfig) -> dict[str, float | int | str | bool]:
    """Flatten one benchmark result into a CSV row."""
    true_summary = summarize_true_components(result.meta)
    true_separation = compute_component_curve_tv_separation(true_summary)["mean_pairwise_tv"]
    return {
        "dataset_name": result.dataset_name,
        "K_true": int(config.K_true),
        "spacing_delta": float(spacing_delta),
        "seed": int(result.seed),
        "model": result.model_name,
        "converged": bool(result.converged),
        "true_separation": float(true_separation),
        "effective_k_mean": float(result.effective_k["effective_k_mean"]),
        "effective_k_std": float(result.effective_k["effective_k_std"]),
        "latent_test_nrmse": float((result.latent_test or {}).get("nrmse", np.nan)),
        "max_rhat": float(result.convergence.get("max_rhat", np.nan)),
        "label_invariant_max_rhat": float((result.label_invariant or {}).get("max_rhat", np.nan)),
        "relabeled_max_rhat": float((result.relabeled or {}).get("max_rhat", np.nan)),
        "min_ess_bulk": float(result.convergence.get("min_ess_bulk", np.nan)),
        "label_invariant_min_ess_bulk": float(
            (result.label_invariant or {}).get("min_ess_bulk", np.nan)
        ),
        "relabeled_min_ess_bulk": float((result.relabeled or {}).get("min_ess_bulk", np.nan)),
        "num_divergences": int(result.hmc_diagnostics.get("num_divergences", 0)),
        "tree_depth_hits": int(result.hmc_diagnostics.get("tree_depth_hits", 0)),
        "raw_spend_lognormal_sigma": float(config.raw_spend_lognormal_sigma),
        "center_k_ratio": float(config.center_k_ratio),
        "n_true": float(config.n_true[0]),
    }


def _plot_sweep(df: pd.DataFrame, *, output_path: Path) -> None:
    """Plot effective K against true separation."""
    fig, ax_k = plt.subplots(figsize=(9.0, 5.6))

    seed_values = sorted(pd.unique(df["seed"]))
    seed_offsets = {
        int(seed): (idx - (len(seed_values) - 1) / 2) * 0.005
        for idx, seed in enumerate(seed_values)
    }

    for target in [1, 2, 3]:
        ax_k.axhline(target, color="0.88", linestyle="--", linewidth=0.9, zorder=0)

    has_k_true = "K_true" in df.columns
    for model_name in MODEL_ORDER:
        model_panel = df[df["model"] == model_name].copy()
        if model_panel.empty:
            continue

        k_trues = sorted(pd.unique(model_panel["K_true"])) if has_k_true else [None]
        for k_true in k_trues:
            panel = model_panel if k_true is None else model_panel[model_panel["K_true"] == k_true].copy()
            if panel.empty:
                continue

            x = panel["true_separation"].to_numpy(dtype=float) + panel["seed"].map(seed_offsets).to_numpy(
                dtype=float
            )
            ax_k.scatter(
                x,
                panel["effective_k_mean"],
                color=MODEL_COLORS[model_name],
                marker=K_TRUE_MARKERS.get(int(k_true), "o") if k_true is not None else "o",
                s=46,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )
            means = (
                panel.groupby("spacing_delta", as_index=False)
                .agg(
                    true_separation=("true_separation", "mean"),
                    effective_k_mean=("effective_k_mean", "mean"),
                )
                .sort_values("spacing_delta")
            )
            ax_k.plot(
                means["true_separation"],
                means["effective_k_mean"],
                color=MODEL_COLORS[model_name],
                linewidth=1.8,
                alpha=0.95,
                label=MODEL_LABELS[model_name] if k_true in (None, 3) else None,
            )
    ax_k.set_ylabel("Estimated Effective K")
    ax_k.set_ylim(0.8, 3.2)
    ax_k.set_title("Controlled Component Separation Sweep")
    ax_k.grid(True, alpha=0.25)
    model_legend = ax_k.legend(loc="upper left", title="Fitted model")
    ax_k.add_artist(model_legend)
    if has_k_true:
        marker_handles = [
            plt.Line2D(
                [],
                [],
                linestyle="None",
                marker=K_TRUE_MARKERS[k_true],
                markersize=7,
                markerfacecolor="0.35",
                markeredgecolor="white",
                label=f"Data K={k_true}",
            )
            for k_true in K_TRUE_ORDER
            if k_true in set(int(v) for v in pd.unique(df["K_true"]))
        ]
        if marker_handles:
            ax_k.legend(handles=marker_handles, loc="lower right", title="True K")
    ax_k.set_xlabel("True Component Separation (TV)")

    x_min = float(df["true_separation"].min())
    x_max = float(df["true_separation"].max())
    pad = 0.04
    ax_k.set_xlim(max(0.0, x_min - pad), min(1.0, x_max + pad))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled_k_spacing_sweep"))
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, dest="models")
    parser.add_argument("--k-true", type=int, action="append", choices=K_TRUE_ORDER, dest="k_trues")
    parser.add_argument("--delta", type=float, action="append", dest="deltas")
    parser.add_argument("--quick", action="store_true", help="Use a 3-seed preview configuration.")
    parser.add_argument("--n-true", type=float, default=2.5, dest="n_true")
    parser.add_argument("--center-k-ratio", type=float, default=0.9, dest="center_k_ratio")
    parser.add_argument("--raw-spend-sigma", type=float, default=0.4, dest="raw_spend_sigma")
    parser.add_argument("--T", type=int, default=200, dest="T")
    args = parser.parse_args()

    seeds = args.seeds or ([0, 1, 2] if args.quick else [0, 1, 2])
    models = args.models or MODEL_ORDER
    deltas = args.deltas or DEFAULT_DELTAS
    k_trues = args.k_trues or [3]

    rows: list[dict[str, float | int | str | bool]] = []
    for k_true in k_trues:
        active_deltas = [0.0] if k_true == 1 else deltas
        for spacing_delta in active_deltas:
            for seed in seeds:
                data_config = _build_data_config(
                    k_true=int(k_true),
                    seed=seed,
                    spacing_delta=float(spacing_delta),
                    args=args,
                )
                x, y, meta = generate_controlled_k_spacing_data(data_config)
                dataset_name = str(meta["dataset_label"])
                for model_name in models:
                    result = run_prepared_synthetic_benchmark_case(
                        dataset_name=dataset_name,
                        x=x,
                        y=y,
                        meta=meta,
                        model_name=model_name,
                        config=_build_run_config(model_name, seed, quick=args.quick),
                        label=f"{dataset_name}_{model_name}_seed{seed}",
                    )
                    rows.append(_record_result(result, spacing_delta=spacing_delta, config=data_config))

    df = pd.DataFrame(rows).sort_values(["K_true", "spacing_delta", "seed", "model"]).reset_index(drop=True)
    summary = (
        df.groupby(["K_true", "spacing_delta", "model"], as_index=False)
        .agg(
            true_separation=("true_separation", "mean"),
            effective_k_mean=("effective_k_mean", "mean"),
            effective_k_std=("effective_k_mean", "std"),
            latent_test_nrmse=("latent_test_nrmse", "mean"),
            converged_rate=("converged", "mean"),
            max_rhat=("max_rhat", "mean"),
            label_invariant_max_rhat=("label_invariant_max_rhat", "mean"),
            relabeled_max_rhat=("relabeled_max_rhat", "mean"),
            num_divergences=("num_divergences", "mean"),
            tree_depth_hits=("tree_depth_hits", "mean"),
        )
        .sort_values(["spacing_delta", "model"])
        .reset_index(drop=True)
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "controlled_k_spacing_results.csv"
    summary_csv = output_dir / "controlled_k_spacing_summary.csv"
    plot_path = output_dir / "controlled_k_spacing_effective_k.png"
    metadata_json = output_dir / "controlled_k_spacing_metadata.json"

    df.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_sweep(df, output_path=plot_path)
    metadata_json.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "models": models,
                "k_trues": k_trues,
                "spacing_deltas": deltas,
                "n_true": float(args.n_true),
                "T": int(args.T),
                "center_k_ratio": float(args.center_k_ratio),
                "raw_spend_lognormal_sigma": float(args.raw_spend_sigma),
                "quick": bool(args.quick),
            },
            indent=2,
        )
    )

    print(f"results_csv: {results_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"plot: {plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
