#!/usr/bin/env python
"""Run a component-resolvability benchmark across K-specific profile grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from hill_mixture_mmm.benchmark import run_prepared_synthetic_benchmark_case
from hill_mixture_mmm.component_resolvability import (
    RESOLVABILITY_PROFILE_LIBRARY,
    build_resolvability_config,
    build_resolvability_run_config,
)
from hill_mixture_mmm.data import generate_controlled_k_spacing_data
from run_controlled_k_spacing_sweep import (
    K_TRUE_MARKERS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    _record_result,
)


def _plot_resolvability_sweep(df: pd.DataFrame, *, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    seed_values = sorted(pd.unique(df["seed"]))
    seed_offsets = {
        int(seed): (idx - (len(seed_values) - 1) / 2) * 0.006 for idx, seed in enumerate(seed_values)
    }
    for target in [1, 2, 3]:
        ax.axhline(target, color="0.88", linestyle="--", linewidth=0.9, zorder=0)

    for model_name in MODEL_ORDER:
        model_panel = df[df["model"] == model_name].copy()
        if model_panel.empty:
            continue
        for k_true in sorted(pd.unique(model_panel["K_true"])):
            panel = model_panel[model_panel["K_true"] == k_true].copy()
            x = panel["true_separation"].to_numpy(dtype=float) + panel["seed"].map(seed_offsets).to_numpy(dtype=float)
            ax.scatter(
                x,
                panel["effective_k_mean"],
                color=MODEL_COLORS[model_name],
                marker=K_TRUE_MARKERS[int(k_true)],
                s=48,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )
            means = (
                panel.groupby("profile_id", as_index=False)
                .agg(true_separation=("true_separation", "mean"), effective_k_mean=("effective_k_mean", "mean"))
                .sort_values("true_separation")
            )
            ax.plot(
                means["true_separation"],
                means["effective_k_mean"],
                color=MODEL_COLORS[model_name],
                linewidth=1.8,
                alpha=0.95,
                label=MODEL_LABELS[model_name] if int(k_true) == 3 else None,
            )

    ax.set_title("Component Resolvability Sweep")
    ax.set_xlabel("True Component Separation (TV)")
    ax.set_ylabel("Estimated Effective K")
    ax.set_ylim(0.8, 3.2)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)

    model_legend = ax.legend(loc="upper left", title="Fitted model")
    ax.add_artist(model_legend)
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
        for k_true in sorted(pd.unique(df["K_true"]))
    ]
    ax.legend(handles=marker_handles, loc="lower right", title="True K")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/component_resolvability_sweep"))
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, dest="models")
    parser.add_argument("--k-true", type=int, action="append", choices=[1, 2, 3], dest="k_trues")
    parser.add_argument("--profile-id", action="append", dest="profile_ids")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--T", type=int, default=200)
    args = parser.parse_args()

    seeds = args.seeds or ([0] if args.quick else [0, 1, 2])
    models = args.models or MODEL_ORDER
    k_trues = args.k_trues or [1, 2, 3]

    rows: list[dict[str, object]] = []
    for k_true in k_trues:
        profiles = RESOLVABILITY_PROFILE_LIBRARY[int(k_true)]
        if args.profile_ids:
            profiles = [profile for profile in profiles if str(profile["profile_id"]) in set(args.profile_ids)]
        for profile in profiles:
            for seed in seeds:
                data_config = build_resolvability_config(
                    k_true=int(k_true),
                    seed=int(seed),
                    profile=profile,
                    T=int(args.T),
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
                        config=build_resolvability_run_config(
                            model_name,
                            int(seed),
                            quick=bool(args.quick),
                        ),
                        label=f"{profile['profile_id']}_{dataset_name}_{model_name}_seed{seed}",
                    )
                    row = _record_result(result, spacing_delta=float(data_config.spacing_delta), config=data_config)
                    row["profile_id"] = str(profile["profile_id"])
                    rows.append(row)

    df = pd.DataFrame(rows).sort_values(["K_true", "true_separation", "seed", "model"]).reset_index(drop=True)
    summary = (
        df.groupby(["K_true", "profile_id", "model"], as_index=False)
        .agg(
            true_separation=("true_separation", "mean"),
            effective_k_mean=("effective_k_mean", "mean"),
            effective_k_std=("effective_k_mean", "std"),
            converged_rate=("converged", "mean"),
            label_invariant_max_rhat=("label_invariant_max_rhat", "mean"),
            relabeled_max_rhat=("relabeled_max_rhat", "mean"),
            num_divergences=("num_divergences", "mean"),
        )
        .sort_values(["K_true", "true_separation", "model"])
        .reset_index(drop=True)
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "component_resolvability_results.csv"
    summary_csv = out_dir / "component_resolvability_summary.csv"
    plot_path = out_dir / "component_resolvability_effective_k.png"
    metadata_json = out_dir / "component_resolvability_metadata.json"

    df.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_resolvability_sweep(df, output_path=plot_path)
    metadata_json.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "models": models,
                "k_trues": k_trues,
                "profile_ids": {
                    str(k_true): [profile["profile_id"] for profile in RESOLVABILITY_PROFILE_LIBRARY[int(k_true)]]
                    for k_true in k_trues
                },
                "T": int(args.T),
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
