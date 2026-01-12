"""Compute posterior cosine separation for resolvability and real-data fits.

The resolvability paper section reports the mean pairwise cosine distance
between the *true* Hill component curves on synthetic DGPs. For real data
there is no ground truth, so we recompute the same quantity from each fit's
posterior ``component_summary``. To keep the metric comparable across fits,
we use a fit-specific saturation grid that covers the active components'
half-saturation range, removing the dependence on ``scale_reference``.

The script writes two CSVs under ``results/``:

* ``posterior_separation_real.csv`` — per-fit posterior cosine separation
  and Shannon effective component count for the real-data benchmark.
* ``posterior_separation_resolvability.csv`` — same quantities for the
  90 resolvability fits, with the true cosine separation retained as
  a reference column.

It then renders ``paper/figures/fig_real_resolvability_overlay.png``,
overlaying the real organisations on the resolvability scatter to
visualise where field data lands relative to the resolvability threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hill_mixture_mmm.metrics import compute_component_curve_cosine_separation

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
RESULTS_ROOT = REPO_ROOT / "results"

REAL_DIRS = {
    "single_hill": FIGURES_ROOT / "real" / "single_hill",
    "mixture_k2": FIGURES_ROOT / "real" / "mixture_k2",
    "mixture_k3": FIGURES_ROOT / "real" / "mixture_k3",
}

RESOLV_DIRS = {
    "mixture_k2": FIGURES_ROOT / "synthetic" / "mixture_k2",
    "mixture_k3": FIGURES_ROOT / "synthetic" / "mixture_k3",
}

DATASET_LABELS = {
    "beauty_fitness": "Beauty & Fitness",
    "home_garden": "Home & Garden",
    "toys_hobbies": "Toys & Hobbies",
}
MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture K=2",
    "mixture_k3": "Mixture K=3",
}

# Use the same grid resolution as the resolvability study, but stretch
# curve_grid_max so that the largest active k_ratio sits at u=4 of the
# normalised range. This makes the cosine separation a property of the
# curve shape and independent of how scale_reference was set.
GRID_K_MULT = 4.0
GRID_SIZE = 128
THRESHOLD_LINE = 0.10  # paper-level resolvability threshold


def _adaptive_curve_grid_max(component_summary: dict) -> float:
    components = component_summary.get("components", [])
    active_k = [
        float(c.get("k_ratio_mean", 0.0))
        for c in components
        if c.get("active", False) and float(c.get("pi_mean", 0.0)) > 0.0
    ]
    if not active_k:
        return 4.0
    return float(max(active_k) * GRID_K_MULT)


def _posterior_cosine(component_summary: dict) -> float:
    grid_max = _adaptive_curve_grid_max(component_summary)
    out = compute_component_curve_cosine_separation(
        component_summary,
        curve_grid_max=grid_max,
        grid_size=GRID_SIZE,
    )
    return float(out.get("mean_pairwise_cosine", 0.0))


def _shannon_count(component_summary: dict) -> float:
    pis = np.asarray(
        [
            float(c.get("pi_mean", 0.0))
            for c in component_summary.get("components", [])
        ],
        dtype=np.float64,
    )
    pis = pis / pis.sum() if pis.sum() > 0 else pis
    entropy = -np.nansum(pis * np.log(np.where(pis > 0, pis, 1.0)))
    return float(np.exp(entropy))


def _publication_pass(summary: dict) -> bool:
    return str(summary.get("publication_status", "")).lower() == "pass"


def _dataset_from_label(label: str) -> str:
    # label is like ``real_beauty_fitness_mixture_k3_seed0`` — peel off
    # the leading ``real_`` and trailing ``_<model>_seed<n>``.
    parts = label.split("_")
    for end in range(len(parts) - 1, 0, -1):
        if parts[end].startswith("seed"):
            return "_".join(parts[1 : end - 2])
    return "unknown"


def _load_real_records() -> pd.DataFrame:
    rows = []
    for model, directory in REAL_DIRS.items():
        if model == "single_hill":
            continue
        for path in sorted(directory.glob("real_*_summary.json")):
            with path.open() as fh:
                data = json.load(fh)
            comp = data.get("component_summary") or {}
            if not comp.get("components"):
                continue
            rows.append(
                {
                    "dataset": _dataset_from_label(data["label"]),
                    "model": data["model_name"],
                    "seed": int(data["seed"]),
                    "posterior_cosine": _posterior_cosine(comp),
                    "shannon_count": _shannon_count(comp),
                    "K_active": int(comp.get("K_active", 0)),
                    "publication_pass": _publication_pass(data),
                }
            )
    return pd.DataFrame(rows)


def _load_resolvability_records() -> pd.DataFrame:
    rows = []
    for model, directory in RESOLV_DIRS.items():
        for path in sorted(directory.glob("resolvability_*_summary.json")):
            with path.open() as fh:
                data = json.load(fh)
            comp = data.get("component_summary") or {}
            if not comp.get("components"):
                continue
            label = data["label"]
            tokens = label.split("_")
            K_true = int(tokens[1][1:]) if tokens[1].startswith("k") else None
            profile = "_".join(tokens[2:-2])
            rows.append(
                {
                    "label": label,
                    "model": data["model_name"],
                    "seed": int(data["seed"]),
                    "K_true": K_true,
                    "profile_id": profile,
                    "posterior_cosine": _posterior_cosine(comp),
                    "shannon_count": _shannon_count(comp),
                    "K_active": int(comp.get("K_active", 0)),
                    "publication_pass": _publication_pass(data),
                }
            )
    return pd.DataFrame(rows)


def _attach_true_separation(df: pd.DataFrame) -> pd.DataFrame:
    csv_path = (
        FIGURES_ROOT
        / "component_resolvability"
        / "full"
        / "selected_metric_results.csv"
    )
    true_df = pd.read_csv(csv_path)
    merged = df.merge(
        true_df[["seed", "K_true", "profile_id", "model", "true_cosine_separation"]],
        on=["seed", "K_true", "profile_id", "model"],
        how="left",
    )
    return merged


def _render_overlay(real_df: pd.DataFrame, resolv_df: pd.DataFrame, out_path: Path) -> None:
    """Render the resolvability-plane overlay.

    Design goals (vs. the earlier cluttered version):

    * The *message* is that real organisations sit at different points on the
      resolvability axis **and** differ in how trustworthy their diagnostics
      are. So we foreground the three organisations and demote the synthetic
      cloud to a faint backdrop that merely sketches the resolvable region.
    * Reliability is encoded directly: a filled marker means at least one seed
      reached publication-ready convergence; a hollow marker means none did.
    * The two K settings of the same organisation are joined by a thin arrow so
      the reader sees how adding a component moves the fit, and each
      organisation is labelled in place instead of via a six-entry legend.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    # Shade the resolvable region (right of the threshold) so the dashed line
    # carries meaning on its own.
    ax.axvspan(THRESHOLD_LINE, 1.0, color="#e8f3e8", zorder=0)
    ax.axvspan(-0.05, THRESHOLD_LINE, color="#f4f4f4", zorder=0)

    # Synthetic resolvability cloud as a faint backdrop only.
    converged = resolv_df["publication_pass"]
    ax.scatter(
        resolv_df.loc[converged, "posterior_cosine"],
        resolv_df.loc[converged, "shannon_count"],
        s=14,
        c="#c7c7c7",
        alpha=0.45,
        edgecolors="none",
        zorder=1,
        label="Synthetic fits (converged)",
    )
    ax.scatter(
        resolv_df.loc[~converged, "posterior_cosine"],
        resolv_df.loc[~converged, "shannon_count"],
        s=14,
        facecolors="none",
        edgecolors="#e0a0a0",
        linewidths=0.7,
        alpha=0.7,
        zorder=1,
        label="Synthetic fits (non-converged)",
    )

    # Real data — averaged per (dataset, model) over seeds.
    agg = (
        real_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            posterior_cosine_mean=("posterior_cosine", "mean"),
            posterior_cosine_std=("posterior_cosine", "std"),
            shannon_mean=("shannon_count", "mean"),
            shannon_std=("shannon_count", "std"),
            pub_pass_rate=("publication_pass", "mean"),
        )
        .fillna(0.0)
    )

    marker_for = {"mixture_k2": "s", "mixture_k3": "^"}
    color_for = {
        "beauty_fitness": "#1f77b4",
        "home_garden": "#2ca02c",
        "toys_hobbies": "#9467bd",
    }

    # Join the two K settings of each organisation with a thin arrow.
    for dataset, grp in agg.groupby("dataset"):
        pts = {row["model"]: row for _, row in grp.iterrows()}
        if "mixture_k2" in pts and "mixture_k3" in pts:
            a, b = pts["mixture_k2"], pts["mixture_k3"]
            ax.annotate(
                "",
                xy=(b["posterior_cosine_mean"], b["shannon_mean"]),
                xytext=(a["posterior_cosine_mean"], a["shannon_mean"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color_for.get(dataset, "black"),
                    alpha=0.55,
                    linewidth=1.2,
                    shrinkA=9,
                    shrinkB=9,
                ),
                zorder=2,
            )

    for _, row in agg.iterrows():
        reliable = row["pub_pass_rate"] > 0.0
        color = color_for.get(row["dataset"], "black")
        ax.errorbar(
            row["posterior_cosine_mean"],
            row["shannon_mean"],
            xerr=row["posterior_cosine_std"],
            yerr=row["shannon_std"],
            fmt=marker_for.get(row["model"], "o"),
            markerfacecolor=color if reliable else "white",
            markeredgecolor=color,
            color=color,
            markersize=13,
            markeredgewidth=1.6,
            elinewidth=1.0,
            capsize=3,
            zorder=3,
        )

    # Label each organisation once, near its K=3 marker.
    label_offsets = {
        "beauty_fitness": (0.012, 0.10),
        "home_garden": (0.012, -0.16),
        "toys_hobbies": (0.012, 0.10),
    }
    for dataset, grp in agg.groupby("dataset"):
        anchor = grp[grp["model"] == "mixture_k3"]
        if anchor.empty:
            anchor = grp
        row = anchor.iloc[0]
        dx, dy = label_offsets.get(dataset, (0.012, 0.1))
        ax.text(
            row["posterior_cosine_mean"] + dx,
            row["shannon_mean"] + dy,
            DATASET_LABELS[dataset],
            fontsize=10,
            fontweight="bold",
            color=color_for.get(dataset, "black"),
            va="center",
        )

    ax.axvline(THRESHOLD_LINE, color="#444444", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        THRESHOLD_LINE - 0.008,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
        "← unresolvable",
        fontsize=9,
        color="#777777",
        ha="right",
        va="top",
    )
    ax.text(
        THRESHOLD_LINE + 0.008,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
        "resolvable →",
        fontsize=9,
        color="#4a7a4a",
        ha="left",
        va="top",
    )

    ax.set_xlabel("Posterior mean pairwise cosine distance between active Hill components")
    ax.set_ylabel("Posterior Shannon effective component count $^1\\!D$")
    ax.set_title(
        "Real organisations span the resolvability axis; only the filled markers\n"
        "(at least one publication-ready seed) carry trustworthy diagnostics"
    )
    ax.set_xlim(-0.02, max(0.8, float(agg["posterior_cosine_mean"].max()) + 0.12))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    # Two compact encoding legends: marker shape = K, fill = reliability.
    from matplotlib.lines import Line2D

    encoding_handles = [
        Line2D([0], [0], marker="s", color="0.3", linestyle="none", markersize=10, label="Mixture K=2"),
        Line2D([0], [0], marker="^", color="0.3", linestyle="none", markersize=10, label="Mixture K=3"),
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="0.3", linestyle="none", markersize=10, label="≥1 publication-ready seed"),
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="white", linestyle="none", markersize=10, label="No publication-ready seed"),
    ]
    ax.legend(handles=encoding_handles, fontsize=8, loc="lower right", framealpha=0.92, title="Marker encoding")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    real_df = _load_real_records()
    resolv_df = _attach_true_separation(_load_resolvability_records())

    real_path = RESULTS_ROOT / "posterior_separation_real.csv"
    resolv_path = RESULTS_ROOT / "posterior_separation_resolvability.csv"
    real_df.to_csv(real_path, index=False)
    resolv_df.to_csv(resolv_path, index=False)
    print(f"Wrote {real_path}")
    print(f"Wrote {resolv_path}")

    fig_path = FIGURES_ROOT / "fig_real_resolvability_overlay.png"
    _render_overlay(real_df, resolv_df, fig_path)
    print(f"Wrote {fig_path}")

    summary = (
        real_df.groupby(["dataset", "model"])
        .agg(
            posterior_cosine_mean=("posterior_cosine", "mean"),
            posterior_cosine_std=("posterior_cosine", "std"),
            shannon_mean=("shannon_count", "mean"),
            pub_pass_rate=("publication_pass", "mean"),
        )
        .round(3)
    )
    print("\nReal-data posterior separation summary (mean across seeds):")
    print(summary)


if __name__ == "__main__":
    main()
