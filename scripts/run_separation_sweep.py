#!/usr/bin/env python
"""Run a synthetic mixture-k3 separation sweep and plot effective-count recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hill_mixture_mmm.benchmark import BenchmarkRunConfig, run_synthetic_benchmark_case
from hill_mixture_mmm.data import DGPConfig
from hill_mixture_mmm.metrics import (
    compute_component_curve_tv_separation,
    compute_soft_component_count,
    summarize_true_components,
)


PROFILE_ORDER = ["default", "separated", "high_separation"]
AVAILABLE_PROFILES = PROFILE_ORDER + ["near_disjoint"]
PROFILE_LABELS = {
    "default": "Default",
    "separated": "Separated",
    "high_separation": "High Separation",
    "near_disjoint": "Near-Disjoint",
}
MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3"]
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


def _build_run_config(model_name: str, seed: int, *, quick: bool) -> BenchmarkRunConfig:
    """Return a pragmatic inference config for the sweep experiment."""
    if quick:
        if model_name == "single_hill":
            return BenchmarkRunConfig(
                seed=seed,
                num_warmup=300,
                num_samples=300,
                num_chains=1,
                target_accept_prob=0.9,
                progress_bar=False,
            )
        if model_name == "mixture_k2":
            return BenchmarkRunConfig(
                seed=seed,
                num_warmup=450,
                num_samples=450,
                num_chains=1,
                target_accept_prob=0.95,
                init_strategy="median",
                progress_bar=False,
            )
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=600,
            num_samples=600,
            num_chains=1,
            target_accept_prob=0.97,
            init_strategy="median",
            progress_bar=False,
        )

    if model_name == "single_hill":
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=600,
            num_samples=600,
            num_chains=2,
            target_accept_prob=0.9,
            progress_bar=False,
        )
    if model_name == "mixture_k2":
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=900,
            num_samples=900,
            num_chains=2,
            target_accept_prob=0.95,
            init_strategy="median",
            progress_bar=False,
        )
    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=1200,
        num_samples=900,
        num_chains=2,
        target_accept_prob=0.97,
        init_strategy="median",
        progress_bar=False,
    )


def _record_for_result(result, *, lambda_: float) -> dict[str, float | int | str | bool]:
    """Project one sweep result into a flat record."""
    true_summary = summarize_true_components(result.meta) if result.meta is not None else None
    true_separation = (
        compute_component_curve_tv_separation(true_summary)["mean_pairwise_tv"]
        if true_summary is not None
        else np.nan
    )
    oracle_soft = (
        compute_soft_component_count(true_summary, lambda_=lambda_)["effective_count"]
        if true_summary is not None
        else np.nan
    )
    model_soft = (
        compute_soft_component_count(result.component_summary, lambda_=lambda_)["effective_count"]
        if result.component_summary is not None
        else 1.0
    )
    base = 1.0
    ratio = np.nan
    if np.isfinite(oracle_soft) and oracle_soft > base + 1e-8:
        ratio = float((model_soft - base) / (oracle_soft - base))

    return {
        "dataset_name": result.dataset_name,
        "profile": str(result.meta.get("dgp_profile", "default")) if result.meta else "default",
        "model": result.model_name,
        "seed": int(result.seed),
        "converged": bool(result.converged),
        "true_separation": float(true_separation),
        "oracle_soft_count": float(oracle_soft),
        "model_soft_count": float(model_soft),
        "oracle_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "latent_test_nrmse": float((result.latent_test or {}).get("nrmse", np.nan)),
    }


def _plot_sweep(df: pd.DataFrame, *, output_path: Path, lambda_: float) -> None:
    """Plot true separation vs model/oracle effective component counts."""
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    seed_values = sorted(pd.unique(df["seed"]))
    model_offsets = {
        model_name: (idx - (len(MODEL_ORDER) - 1) / 2) * 0.012
        for idx, model_name in enumerate(MODEL_ORDER)
    }
    seed_offsets = {
        int(seed): (idx - (len(seed_values) - 1) / 2) * 0.0025
        for idx, seed in enumerate(seed_values)
    }
    profile_markers = {
        "default": "o",
        "separated": "s",
        "high_separation": "^",
        "near_disjoint": "D",
    }

    for target in [1, 2, 3]:
        ax.axhline(target, color="0.88", linestyle="--", linewidth=0.9, zorder=0)

    oracle_points = (
        df.groupby(["profile", "seed"], as_index=False)
        .agg(true_separation=("true_separation", "first"), oracle_soft_count=("oracle_soft_count", "first"))
        .sort_values(["true_separation", "seed"])
    )
    for profile, panel in oracle_points.groupby("profile", sort=False):
        ax.scatter(
            panel["true_separation"],
            panel["oracle_soft_count"],
            marker=profile_markers[profile],
            s=92,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=4,
        )

    for model_name in MODEL_ORDER:
        panel = df[df["model"] == model_name].copy()
        if panel.empty:
            continue
        x = (
            panel["true_separation"].to_numpy(dtype=float)
            + model_offsets[model_name]
            + panel["seed"].map(seed_offsets).to_numpy(dtype=float)
        )
        ax.scatter(
            x,
            panel["model_soft_count"],
            color=MODEL_COLORS[model_name],
            marker="o",
            s=52,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )

        means = (
            panel.groupby("profile", as_index=False)
            .agg(true_separation=("true_separation", "mean"), model_soft_count=("model_soft_count", "mean"))
            .set_index("profile")
            .reindex(PROFILE_ORDER)
            .dropna()
            .reset_index()
        )
        ax.plot(
            means["true_separation"] + model_offsets[model_name],
            means["model_soft_count"],
            color=MODEL_COLORS[model_name],
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.8, max(3.2, float(df["oracle_soft_count"].max()) + 0.15))
    ax.set_xlabel("True Component Separation (TV)")
    ax.set_ylabel(f"Soft Effective Count (lambda={lambda_:g})")
    ax.set_title("Mixture-K3 Separation Sweep")
    ax.grid(True, alpha=0.25)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MODEL_COLORS[model_name],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8,
            label=MODEL_LABELS[model_name],
        )
        for model_name in MODEL_ORDER
    ]
    profile_handles = [
        Line2D(
            [0],
            [0],
            marker=profile_markers[profile],
            color="0.25",
            markerfacecolor="0.8",
            markeredgecolor="0.25",
            linestyle="None",
            markersize=8,
            label=PROFILE_LABELS[profile],
        )
        for profile in PROFILE_ORDER
    ]
    oracle_handle = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            linestyle="None",
            markersize=8,
            label="Oracle DGP",
        )
    ]

    legend_models = ax.legend(handles=model_handles, loc="upper left", title="Model", frameon=False)
    ax.add_artist(legend_models)
    ax.legend(handles=profile_handles + oracle_handle, loc="upper right", title="Profile", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/separation_sweep"))
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--profile", action="append", choices=AVAILABLE_PROFILES, dest="profiles")
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, dest="models")
    parser.add_argument("--quick", action="store_true", help="Use a one-seed, low-cost preview config.")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=6.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or ([0] if args.quick else [0, 1, 2])
    profiles = args.profiles or PROFILE_ORDER
    models = args.models or MODEL_ORDER

    records: list[dict[str, float | int | str | bool]] = []
    for profile in profiles:
        for seed in seeds:
            for model_name in models:
                result = run_synthetic_benchmark_case(
                    dgp_config=DGPConfig(dgp_type="mixture_k3", T=200, seed=seed, profile=profile),
                    model_name=model_name,
                    config=_build_run_config(model_name, seed, quick=args.quick),
                    label=f"separation_sweep_{profile}_{model_name}_seed{seed}",
                )
                records.append(_record_for_result(result, lambda_=args.lambda_))

    df = pd.DataFrame.from_records(records).sort_values(["profile", "seed", "model"]).reset_index(drop=True)
    csv_path = output_dir / "separation_sweep_results.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby(["profile", "model"], as_index=False)
        .agg(
            true_separation=("true_separation", "mean"),
            oracle_soft_count=("oracle_soft_count", "mean"),
            model_soft_count=("model_soft_count", "mean"),
            oracle_ratio=("oracle_ratio", "mean"),
            latent_test_nrmse=("latent_test_nrmse", "mean"),
        )
        .sort_values(["profile", "model"])
    )
    summary_path = output_dir / "separation_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = output_dir / "separation_sweep_soft_count.png"
    _plot_sweep(df, output_path=plot_path, lambda_=args.lambda_)

    metadata = {
        "profiles": profiles,
        "models": models,
        "seeds": seeds,
        "quick": bool(args.quick),
        "lambda": float(args.lambda_),
        "run_config_note": "quick mode uses reduced warmup/samples for preview only",
    }
    (output_dir / "separation_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"results_csv: {csv_path}")
    print(f"summary_csv: {summary_path}")
    print(f"plot: {plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
