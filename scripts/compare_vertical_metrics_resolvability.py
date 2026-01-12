#!/usr/bin/env python
"""Compare alternative vertical-axis metrics on the component-resolvability sweep."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hill_mixture_mmm.benchmark import run_prepared_synthetic_benchmark_case
from hill_mixture_mmm.component_resolvability import (
    RESOLVABILITY_PROFILE_LIBRARY,
    build_resolvability_config,
    build_resolvability_run_config,
)
from hill_mixture_mmm.data import generate_controlled_k_spacing_data
from hill_mixture_mmm.metrics import (
    compute_component_curve_tv_separation,
    compute_inverse_simpson_effective_count,
    compute_leinster_cobbold_effective_count,
    compute_rao_quadratic_entropy_equivalent_count,
    compute_shannon_effective_count,
    compute_similarity_adjusted_effective_count,
    summarize_true_components,
)
from run_controlled_k_spacing_sweep import K_TRUE_MARKERS, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER


def _metric_bundle(result) -> dict[str, float]:
    if result.component_summary is None:
        return {
            "active_component_count": 1.0,
            "similarity_adjusted_count": 1.0,
            "shannon_count": 1.0,
            "simpson_count": 1.0,
            "rao_count": 1.0,
            "leinster_q1_count": 1.0,
            "leinster_q2_count": 1.0,
        }
    component_summary = result.component_summary
    return {
        "active_component_count": float(component_summary["K_active"]),
        "similarity_adjusted_count": float(
            compute_similarity_adjusted_effective_count(component_summary)["effective_count"]
        ),
        "shannon_count": float(compute_shannon_effective_count(component_summary)["effective_count"]),
        "simpson_count": float(compute_inverse_simpson_effective_count(component_summary)["effective_count"]),
        "rao_count": float(compute_rao_quadratic_entropy_equivalent_count(component_summary)["effective_count"]),
        "leinster_q1_count": float(
            compute_leinster_cobbold_effective_count(component_summary, q=1.0, lambda_=6.0)["effective_count"]
        ),
        "leinster_q2_count": float(
            compute_leinster_cobbold_effective_count(component_summary, q=2.0, lambda_=6.0)["effective_count"]
        ),
    }


METRIC_SPECS = [
    ("active_component_count", "Active K (pi > 0.05)"),
    ("similarity_adjusted_count", "Similarity-Adjusted Count"),
    ("shannon_count", "Hill q=1 (weights only)"),
]


def _plot(df: pd.DataFrame, *, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(METRIC_SPECS) + 1, figsize=(16.5, 4.8), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    seed_values = sorted(pd.unique(df["seed"]))
    seed_offsets = {
        int(seed): (idx - (len(seed_values) - 1) / 2) * 0.006 for idx, seed in enumerate(seed_values)
    }

    for axis_idx, (metric_key, title) in enumerate(METRIC_SPECS):
        ax = axes_flat[axis_idx]
        for target in [1, 2, 3]:
            ax.axhline(target, color="0.9", linestyle="--", linewidth=0.8, zorder=0)
        for model_name in MODEL_ORDER:
            model_panel = df[df["model"] == model_name]
            for k_true in sorted(pd.unique(model_panel["K_true"])):
                panel = model_panel[model_panel["K_true"] == k_true]
                if panel.empty:
                    continue
                x = panel["true_separation"].to_numpy(dtype=float) + panel["seed"].map(seed_offsets).to_numpy(dtype=float)
                ax.scatter(
                    x,
                    panel[metric_key],
                    color=MODEL_COLORS[model_name],
                    marker=K_TRUE_MARKERS[int(k_true)],
                    s=42,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )
                means = (
                    panel.groupby("profile_id", as_index=False)
                    .agg(true_separation=("true_separation", "mean"), y=(metric_key, "mean"))
                    .sort_values("true_separation")
                )
                ax.plot(
                    means["true_separation"],
                    means["y"],
                    color=MODEL_COLORS[model_name],
                    linewidth=1.6,
                    alpha=0.95,
                )
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0.8, 3.2)
        ax.grid(True, alpha=0.22)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    model_handles = [
        plt.Line2D([], [], color=MODEL_COLORS[m], linewidth=1.8, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    marker_handles = [
        plt.Line2D(
            [], [], linestyle="None", marker=K_TRUE_MARKERS[k], markersize=7, markerfacecolor="0.35", markeredgecolor="white", label=f"Data K={k}"
        )
        for k in sorted(pd.unique(df["K_true"]))
    ]
    legend_ax.legend(handles=model_handles + marker_handles, loc="center", frameon=False)

    for ax in axes_flat[: len(METRIC_SPECS)]:
        ax.set_xlabel("True Component Separation (TV)")
    axes_flat[0].set_ylabel("Estimated Count")
    fig.suptitle("Selected Vertical-Axis Metrics on Resolvability Sweep", y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/resolvability_vertical_metric_comparison"))
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, dest="models")
    parser.add_argument("--k-true", type=int, action="append", choices=[1, 2, 3], dest="k_trues")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--T", type=int, default=200)
    args = parser.parse_args()

    seeds = args.seeds or ([0] if args.quick else [0, 1, 2])
    models = args.models or MODEL_ORDER
    k_trues = args.k_trues or [1, 2, 3]

    rows: list[dict[str, object]] = []
    for k_true in k_trues:
        for profile in RESOLVABILITY_PROFILE_LIBRARY[int(k_true)]:
            for seed in seeds:
                config = build_resolvability_config(
                    k_true=int(k_true), seed=int(seed), profile=profile, T=int(args.T)
                )
                x, y, meta = generate_controlled_k_spacing_data(config)
                true_separation = float(
                    compute_component_curve_tv_separation(summarize_true_components(meta))["mean_pairwise_tv"]
                )
                dataset_name = str(meta["dataset_label"])
                for model_name in models:
                    result = run_prepared_synthetic_benchmark_case(
                        dataset_name=dataset_name,
                        x=x,
                        y=y,
                        meta=meta,
                        model_name=model_name,
                        config=build_resolvability_run_config(
                            model_name,
                            int(seed),
                            quick=bool(args.quick),
                        ),
                        label=f"{profile['profile_id']}_{dataset_name}_{model_name}_seed{seed}",
                    )
                    bundle = _metric_bundle(result)
                    rows.append(
                        {
                            "seed": int(seed),
                            "K_true": int(k_true),
                            "profile_id": str(profile["profile_id"]),
                            "model": model_name,
                            "true_separation": true_separation,
                            **bundle,
                        }
                    )

    df = pd.DataFrame(rows).sort_values(["K_true", "true_separation", "seed", "model"]).reset_index(drop=True)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vertical_metric_comparison_results.csv"
    png_path = out_dir / "vertical_metric_comparison.png"
    df.to_csv(csv_path, index=False)
    _plot(df, output_path=png_path)
    print(f"results_csv: {csv_path}")
    print(f"plot: {png_path}")
    summary = (
        df.groupby(["K_true", "profile_id", "model"], as_index=False)[[key for key, _ in METRIC_SPECS]]
        .mean()
        .sort_values(["K_true", "profile_id", "model"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
