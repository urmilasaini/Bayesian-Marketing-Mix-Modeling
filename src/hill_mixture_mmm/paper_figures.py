"""Generate selected paper figures from synthetic benchmark results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import DGPConfig, generate_data
from .metrics import (
    compute_component_curve_tv_separation,
    compute_similarity_adjusted_effective_count,
    summarize_true_components,
)

DGP_ORDER = ["single", "mixture_k2", "mixture_k3"]
DGP_K_TRUE = {"single": 1, "mixture_k2": 2, "mixture_k3": 3}
DGP_LABELS = {
    "single": "Single (K=1)",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3"]
MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
COLORS = {
    "single_hill": "#1f77b4",
    "mixture_k2": "#9467bd",
    "mixture_k3": "#ff7f0e",
}
DEFAULT_FIGURE_IDS = ("fig0", "fig1", "fig2", "fig3", "fig4", "fig5")
FIGURE_FILENAMES = {
    "fig0": "fig0_graphical_model.png",
    "fig1": "fig1_elpd_comparison.png",
    "fig2": "fig2_crps_comparison.png",
    "fig3": "fig3_latent_nrmse_comparison.png",
    "fig4": "fig4_convergence_heatmap.png",
    "fig5": "fig5_component_separation_vs_effective_count.png",
}
RHAT_TEST_PASS_MAX = 1.05

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _safe_float(value: Any) -> float | None:
    """Return a float when possible, else None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_metric(summary: dict[str, Any], section: str, key: str) -> float | None:
    """Read a numeric metric from a nested summary dictionary."""
    payload = summary.get(section)
    if not isinstance(payload, dict):
        return None
    return _safe_float(payload.get(key))


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column as a float series, or all-NaN if missing."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _normalize_bool(series: pd.Series) -> pd.Series:
    """Convert boolean-like benchmark columns into strict bools."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def _compute_rhat_test_pass(df: pd.DataFrame) -> pd.Series:
    """Return whether each row passes the benchmark R-hat rule."""
    if "rhat_test_pass" in df.columns:
        return _normalize_bool(df["rhat_test_pass"])

    standard_rhat = pd.to_numeric(df.get("max_rhat"), errors="coerce")
    if "model" not in df.columns:
        return standard_rhat.le(RHAT_TEST_PASS_MAX).fillna(False)

    mixture_mask = df["model"].astype(str) != "single_hill"
    if "label_invariant_max_rhat" in df.columns:
        label_rhat = pd.to_numeric(df["label_invariant_max_rhat"], errors="coerce")
        rhat_used = standard_rhat.where(~mixture_mask, label_rhat)
    else:
        rhat_used = standard_rhat
    return rhat_used.le(RHAT_TEST_PASS_MAX).fillna(False)


def _summary_to_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Project a seed-level synthetic summary JSON into one raw-results record."""
    dgp_name = str(summary["dataset_name"])
    plot_metrics = _component_count_plot_metrics(summary)
    label_invariant = summary.get("label_invariant") or {}
    relabeled = summary.get("relabeled") or {}
    publication_status = summary.get("publication_status")
    interpretation_status = summary.get("interpretation_status")
    benchmark_pass = summary.get("benchmark_pass")
    if benchmark_pass is None and publication_status is not None:
        benchmark_pass = str(publication_status).lower() != "fail"

    return {
        "dgp": dgp_name,
        "K_true": DGP_K_TRUE.get(dgp_name),
        "model": summary["model_name"],
        "seed": int(summary["seed"]),
        "converged": bool(summary["converged"]),
        "publication_status": publication_status,
        "interpretation_status": interpretation_status,
        "benchmark_pass": bool(benchmark_pass)
        if benchmark_pass is not None
        else bool(summary["converged"]),
        "max_rhat": _nested_metric(summary, "convergence", "max_rhat"),
        "min_ess_bulk": _nested_metric(summary, "convergence", "min_ess_bulk"),
        "min_ess_tail": _nested_metric(summary, "convergence", "min_ess_tail"),
        "label_invariant_max_rhat": _safe_float(label_invariant.get("max_rhat")),
        "rhat_log_lik": _safe_float(label_invariant.get("rhat_log_lik")),
        "relabeled_max_rhat": _safe_float(relabeled.get("max_rhat")),
        "num_divergences": _nested_metric(summary, "hmc_diagnostics", "num_divergences"),
        "min_bfmi": _nested_metric(summary, "hmc_diagnostics", "min_bfmi"),
        "tree_depth_hits": _nested_metric(summary, "hmc_diagnostics", "tree_depth_hits"),
        "elpd_loo": _nested_metric(summary, "loo", "elpd_loo"),
        "pareto_k_bad": _nested_metric(summary, "loo", "pareto_k_bad"),
        "pareto_k_very_bad": _nested_metric(summary, "loo", "pareto_k_very_bad"),
        "train_mape": _nested_metric(summary, "train_metrics", "mape"),
        "test_mape": _nested_metric(summary, "test_metrics", "mape"),
        "train_nrmse": _nested_metric(summary, "train_metrics", "nrmse"),
        "test_nrmse": _nested_metric(summary, "test_metrics", "nrmse"),
        "train_crps": _nested_metric(summary, "train_metrics", "crps"),
        "test_crps": _nested_metric(summary, "test_metrics", "crps"),
        "train_coverage_90": _nested_metric(summary, "train_metrics", "coverage_90"),
        "test_coverage_90": _nested_metric(summary, "test_metrics", "coverage_90"),
        "latent_train_nrmse": _nested_metric(summary, "latent_train", "nrmse"),
        "latent_test_nrmse": _nested_metric(summary, "latent_test", "nrmse"),
        "latent_train_crps": _nested_metric(summary, "latent_train", "crps"),
        "latent_test_crps": _nested_metric(summary, "latent_test", "crps"),
        "latent_train_coverage_90": _nested_metric(summary, "latent_train", "coverage_90"),
        "latent_test_coverage_90": _nested_metric(summary, "latent_test", "coverage_90"),
        "latent_train_coverage_95": _nested_metric(summary, "latent_train", "coverage_95"),
        "latent_test_coverage_95": _nested_metric(summary, "latent_test", "coverage_95"),
        "effective_k_mean": _nested_metric(summary, "effective_k", "mean"),
        "effective_k_std": _nested_metric(summary, "effective_k", "std"),
        "true_component_separation": plot_metrics["true_separation"],
        "estimated_effective_component_count": plot_metrics["estimated_effective_count"],
        "component_count_gamma": plot_metrics["gamma"],
    }


def _component_count_plot_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    """Return the component-separation plot metrics, backfilling old summaries when needed."""
    payload = summary.get("component_count_plot")
    if isinstance(payload, dict):
        return {
            "true_separation": _safe_float(payload.get("true_separation")),
            "estimated_effective_count": _safe_float(payload.get("estimated_effective_count")),
            "gamma": _safe_float(payload.get("gamma")),
        }

    estimated_effective_count = None
    if isinstance(summary.get("component_summary"), dict):
        estimated_effective_count = compute_similarity_adjusted_effective_count(
            summary["component_summary"]
        )["effective_count"]

    true_separation = None
    if isinstance(summary.get("true_component_summary"), dict):
        true_separation = compute_component_curve_tv_separation(summary["true_component_summary"])[
            "mean_pairwise_tv"
        ]
    elif summary.get("domain") == "synthetic":
        dgp_name = str(summary["dataset_name"])
        total_size = int(summary.get("train_size", 0)) + int(summary.get("test_size", 0))
        if dgp_name in DGP_ORDER and total_size > 0 and "seed" in summary:
            _, _, meta = generate_data(
                DGPConfig(
                    dgp_type=dgp_name,
                    T=total_size,
                    seed=int(summary["seed"]),
                )
            )
            true_summary = summarize_true_components(meta)
            true_separation = compute_component_curve_tv_separation(true_summary)[
                "mean_pairwise_tv"
            ]

    return {
        "true_separation": _safe_float(true_separation),
        "estimated_effective_count": _safe_float(estimated_effective_count),
        "gamma": 0.5,
    }


def load_synthetic_results_from_artifacts(
    artifact_root: str | Path,
    *,
    summary_paths: Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    """Load seed-level synthetic benchmark rows from saved case summaries."""
    artifact_root = Path(artifact_root)
    if summary_paths is None:
        paths = sorted(artifact_root.glob("synthetic/*/*_seed*_summary.json"))
    else:
        paths = [Path(path) for path in summary_paths]

    rows_by_case: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("domain") != "synthetic":
            continue
        record = _summary_to_record(summary)
        key = (record["dgp"], record["model"], record["seed"])
        rows_by_case[key] = record

    if not rows_by_case:
        raise ValueError(f"No synthetic seed-level summaries found under {artifact_root}")

    df = pd.DataFrame(rows_by_case.values())
    if "converged" in df.columns:
        df["converged"] = _normalize_bool(df["converged"])
    if "benchmark_pass" in df.columns:
        df["benchmark_pass"] = _normalize_bool(df["benchmark_pass"])
    df["rhat_test_pass"] = _compute_rhat_test_pass(df)
    return df


def load_synthetic_results(
    *,
    results_csv: str | Path | None = None,
    artifact_root: str | Path | None = None,
    summary_paths: Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    """Load synthetic benchmark rows from either a raw CSV or saved summaries."""
    if results_csv is not None:
        df = pd.read_csv(results_csv)
        if "converged" in df.columns:
            df["converged"] = _normalize_bool(df["converged"])
        if "benchmark_pass" in df.columns:
            df["benchmark_pass"] = _normalize_bool(df["benchmark_pass"])
        df["rhat_test_pass"] = _compute_rhat_test_pass(df)
        return df
    if artifact_root is None:
        raise ValueError("Pass either results_csv or artifact_root")
    return load_synthetic_results_from_artifacts(artifact_root, summary_paths=summary_paths)


def _metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate one metric to mean/std per DGP-model cell."""
    metric_frame = (
        df.groupby(["dgp", "model"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
        .fillna({"std": 0.0})
    )
    return metric_frame


def _lookup_metric(
    metric_frame: pd.DataFrame, dgp_name: str, model_name: str
) -> tuple[float, float]:
    """Return mean/std for one DGP-model cell."""
    row = metric_frame[(metric_frame["dgp"] == dgp_name) & (metric_frame["model"] == model_name)]
    if row.empty:
        return np.nan, np.nan
    return float(row["mean"].iloc[0]), float(row["std"].iloc[0])


def generate_graphical_model_figure(output_dir: str | Path) -> Path:
    """Render Figure 0: conceptual overview of the Hill mixture model."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    edge = "#1f2937"
    text = "#111827"
    panel_fill = "#f8fafc"
    latent_fill = "#ffffff"
    observed_fill = "#d1d5db"

    def add_panel(x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.15",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=panel_fill,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.2,
            y + h - 0.25,
            title,
            ha="left",
            va="top",
            fontsize=12,
            color=text,
            weight="bold",
        )
        ax.text(
            x + w - 0.2, y + h - 0.58, subtitle, ha="right", va="top", fontsize=9, color="#4b5563"
        )

    def add_node(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        observed: bool = False,
        fontsize: int = 14,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=observed_fill if observed else latent_fill,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=text
        )

    def add_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        connectionstyle: str | None = None,
    ) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=edge,
            shrinkA=8,
            shrinkB=8,
            connectionstyle=connectionstyle,
        )
        ax.add_patch(arrow)

    add_panel(0.7, 2.0, 4.2, 3.5, "Temporal structure", r"$t = 1, \ldots, T$")
    add_panel(5.4, 2.0, 3.4, 3.5, "Mixture components", r"$k = 1, \ldots, K$")
    add_panel(9.3, 2.0, 3.7, 3.5, "Observation model", "Predictive mixture over components")

    add_node(1.0, 4.0, 0.9, 0.6, r"$\alpha$")
    add_node(1.0, 3.2, 1.0, 0.65, r"$x_t$", observed=True)
    add_node(2.45, 3.2, 1.0, 0.65, r"$s_t$")
    add_node(3.75, 4.0, 1.25, 0.6, r"$\mu_0,\beta$")
    add_node(3.95, 2.75, 0.9, 0.65, r"$b_t$")

    add_node(6.05, 4.0, 1.1, 0.6, r"$\theta_k$")
    add_node(5.9, 3.0, 1.4, 0.75, r"$h_{t,k}$")

    add_node(9.8, 4.0, 1.0, 0.6, r"$\pi_k$")
    add_node(11.35, 4.0, 1.0, 0.6, r"$\sigma$")
    add_node(10.5, 3.0, 1.25, 0.75, r"$y_t$", observed=True)

    add_arrow((1.45, 4.0), (2.95, 3.55), connectionstyle="arc3,rad=-0.12")
    add_arrow((2.0, 3.52), (2.45, 3.52))
    add_arrow((3.45, 3.62), (5.9, 3.58), connectionstyle="arc3,rad=0.03")
    add_arrow((4.38, 4.0), (4.38, 3.35))
    add_arrow((4.85, 3.0), (10.5, 3.18), connectionstyle="arc3,rad=-0.05")
    add_arrow((6.6, 4.0), (6.6, 3.75))
    add_arrow((7.3, 3.38), (10.5, 3.38))
    add_arrow((10.3, 4.0), (11.0, 3.75), connectionstyle="arc3,rad=-0.05")
    add_arrow((11.85, 4.0), (11.25, 3.75), connectionstyle="arc3,rad=0.05")

    ax.text(1.45, 2.55, "spend", ha="center", fontsize=10, color="#374151")
    ax.text(2.95, 2.55, "geometric adstock", ha="center", fontsize=10, color="#374151")
    ax.text(4.4, 2.1, "linear baseline", ha="center", fontsize=10, color="#374151")
    ax.text(
        6.6,
        2.48,
        r"component parameters $(A_k, \lambda_k, n_k)$",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    ax.text(6.6, 2.16, "Hill response for component k", ha="center", fontsize=10, color="#374151")
    ax.text(
        10.2,
        2.66,
        "stick-breaking\nweights",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#374151",
    )
    ax.text(11.95, 2.66, "Gaussian\nnoise", ha="center", va="center", fontsize=9.5, color="#374151")
    ax.text(
        11.1,
        2.04,
        r"$y_t \sim \sum_k \pi_k \,\mathcal{N}(b_t + h_{t,k}, \sigma^2)$",
        ha="center",
        fontsize=10,
        color=text,
    )

    fig.tight_layout()
    output_path = output_dir / "fig0_graphical_model.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_elpd_comparison_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 1: ELPD-LOO comparison across DGPs and models."""
    output_dir = Path(output_dir)
    metric_frame = _metric_frame(df, "elpd_loo")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DGP_ORDER))
    width = min(0.28, 0.8 / max(len(MODEL_ORDER), 1))

    for idx, model_name in enumerate(MODEL_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for dgp_name in DGP_ORDER:
            mean, std = _lookup_metric(metric_frame, dgp_name, model_name)
            means.append(mean)
            stds.append(std)

        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Data Generating Process")
    ax.set_ylabel("ELPD-LOO")
    ax.set_title("Model Comparison: Expected Log Pointwise Predictive Density")
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])
    ax.legend(title="Model")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.annotate(
        "Error bars: ±1 std across random seeds",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    output_path = output_dir / FIGURE_FILENAMES["fig1"]
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_predictive_score_comparison_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 2: holdout test CRPS comparison across DGPs and models."""
    output_dir = Path(output_dir)
    metric_frame = _metric_frame(df, "test_crps")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DGP_ORDER))
    width = min(0.28, 0.8 / max(len(MODEL_ORDER), 1))

    max_height = 0.0
    for idx, model_name in enumerate(MODEL_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for dgp_name in DGP_ORDER:
            mean, std = _lookup_metric(metric_frame, dgp_name, model_name)
            means.append(mean)
            stds.append(std)
            if not np.isnan(mean):
                max_height = max(max_height, mean + (0.0 if np.isnan(std) else std))

        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Data Generating Process")
    ax.set_ylabel("Test CRPS")
    ax.set_title("Model Comparison: Holdout Predictive CRPS (Lower Is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])
    ax.legend(title="Model")
    ax.set_ylim(0, max(1.0, max_height * 1.15))
    ax.annotate(
        "Lower is better; error bars: ±1 std across random seeds",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    output_path = output_dir / FIGURE_FILENAMES["fig2"]
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_convergence_heatmap_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 4: R-hat pass-rate heatmap by DGP and model."""
    output_dir = Path(output_dir)
    convergence = _compute_rhat_test_pass(df)
    conv_rates = (
        df.assign(converged=convergence)
        .groupby(["dgp", "model"])["converged"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(index=DGP_ORDER, columns=MODEL_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(conv_rates.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("R-hat Pass Rate")

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticks(np.arange(len(DGP_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[model_name] for model_name in MODEL_ORDER])
    ax.set_yticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])

    for row_idx, dgp_name in enumerate(DGP_ORDER):
        for col_idx, model_name in enumerate(MODEL_ORDER):
            value = float(conv_rates.loc[dgp_name, model_name])
            text_color = "white" if value < 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Data Generating Process")
    ax.set_title("R-hat Threshold Pass Rate by DGP and Model")

    plt.tight_layout()
    output_path = output_dir / FIGURE_FILENAMES["fig4"]
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_latent_recovery_comparison_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 3: latent test nRMSE across DGPs and models."""
    output_dir = Path(output_dir)
    latent_nrmse = _numeric_series(df, "latent_test_nrmse")
    observed_nrmse = _numeric_series(df, "test_nrmse")

    df = df.assign(
        figure3_nrmse=latent_nrmse.combine_first(observed_nrmse),
    )
    metric_frame = (
        df.groupby(["dgp", "model"], as_index=False)
        .agg(
            mean=("figure3_nrmse", "mean"),
            std=("figure3_nrmse", "std"),
        )
        .fillna({"std": 0.0})
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DGP_ORDER))
    width = min(0.28, 0.8 / max(len(MODEL_ORDER), 1))
    max_height = 0.0

    for idx, model_name in enumerate(MODEL_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for dgp_name in DGP_ORDER:
            mean, std = _lookup_metric(metric_frame, dgp_name, model_name)
            means.append(mean)
            stds.append(std)
            if not np.isnan(mean):
                max_height = max(max_height, mean + (0.0 if np.isnan(std) else std))

        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Data Generating Process")
    ax.set_ylabel("Latent Test nRMSE")
    ax.set_title("Model Comparison: Latent Response Recovery (Lower Is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])
    ax.legend(title="Model")
    ax.set_ylim(0, max(0.05, max_height * 1.15))
    ax.annotate(
        "Lower is better; error bars: ±1 std across random seeds",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    output_path = output_dir / FIGURE_FILENAMES["fig3"]
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_component_separation_effective_count_figure(
    df: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Render Figure 5: one combined scatter over all seeds, models, and DGPs."""
    output_dir = Path(output_dir)
    metric_df = df.copy()
    metric_df["true_component_separation"] = _numeric_series(df, "true_component_separation")
    metric_df["estimated_effective_component_count"] = _numeric_series(
        df, "estimated_effective_component_count"
    )
    metric_df = metric_df.dropna(
        subset=["true_component_separation", "estimated_effective_component_count"]
    )
    marker_map = {"single": "o", "mixture_k2": "s", "mixture_k3": "^"}
    y_max = max(3.2, float(metric_df["estimated_effective_component_count"].max()) + 0.2)
    sorted_seeds = sorted(pd.unique(metric_df["seed"]))
    model_offsets = {
        model_name: (idx - (len(MODEL_ORDER) - 1) / 2) * 0.014
        for idx, model_name in enumerate(MODEL_ORDER)
    }
    seed_offsets = {
        int(seed): (idx - (len(sorted_seeds) - 1) / 2) * 0.003
        for idx, seed in enumerate(sorted_seeds)
    }

    def _plot_metric_frame(ax, panel_df: pd.DataFrame, *, title: str) -> None:
        for target in sorted(set(DGP_K_TRUE.values())):
            ax.axhline(target, color="0.88", linestyle="--", linewidth=0.9, zorder=0)

        for dgp_name in DGP_ORDER:
            dgp_panel = panel_df[panel_df["dgp"] == dgp_name]
            if dgp_panel.empty:
                continue
            for model_name in MODEL_ORDER:
                model_panel = dgp_panel[dgp_panel["model"] == model_name].copy()
                if model_panel.empty:
                    continue
                x_values = (
                    model_panel["true_component_separation"].to_numpy(dtype=float)
                    + model_offsets[model_name]
                    + model_panel["seed"].map(seed_offsets).to_numpy(dtype=float)
                )
                ax.scatter(
                    x_values,
                    model_panel["estimated_effective_component_count"],
                    color=COLORS[model_name],
                    marker=marker_map[dgp_name],
                    s=54,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0.8, y_max)
        ax.set_xlabel("True Component Separation")
        ax.set_ylabel("Estimated Effective Component Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    _plot_metric_frame(
        ax, metric_df, title="True Separation vs Estimated Effective Component Count"
    )
    ax.annotate(
        "All seeds are overlaid in one panel",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[model_name],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8,
            label=MODEL_LABELS[model_name],
        )
        for model_name in MODEL_ORDER
    ]
    dgp_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[dgp_name],
            color="0.25",
            markerfacecolor="0.8",
            markeredgecolor="0.25",
            linestyle="None",
            markersize=8,
            label=DGP_LABELS[dgp_name],
        )
        for dgp_name in DGP_ORDER
    ]
    legend_models = ax.legend(handles=model_handles, loc="upper left", title="Model", frameon=False)
    ax.add_artist(legend_models)
    ax.legend(handles=dgp_handles, loc="upper right", title="DGP", frameon=False)

    fig.tight_layout()
    output_path = output_dir / FIGURE_FILENAMES["fig5"]
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def generate_publication_figures(
    *,
    output_dir: str | Path,
    results_csv: str | Path | None = None,
    artifact_root: str | Path | None = None,
    summary_paths: Sequence[str | Path] | None = None,
    figure_ids: Sequence[str] = DEFAULT_FIGURE_IDS,
) -> dict[str, Path]:
    """Generate the selected benchmark paper figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = tuple(dict.fromkeys(figure_ids))
    unknown = sorted(set(requested) - set(DEFAULT_FIGURE_IDS))
    if unknown:
        raise ValueError(f"Unknown figure ids: {', '.join(unknown)}")

    generated: dict[str, Path] = {}
    if "fig0" in requested:
        generated["fig0"] = generate_graphical_model_figure(output_dir)

    data_figures = [figure_id for figure_id in requested if figure_id != "fig0"]
    if not data_figures:
        return generated

    df = load_synthetic_results(
        results_csv=results_csv,
        artifact_root=artifact_root,
        summary_paths=summary_paths,
    )

    generators = {
        "fig1": generate_elpd_comparison_figure,
        "fig2": generate_predictive_score_comparison_figure,
        "fig3": generate_latent_recovery_comparison_figure,
        "fig4": generate_convergence_heatmap_figure,
        "fig5": generate_component_separation_effective_count_figure,
    }
    for figure_id in data_figures:
        generated[figure_id] = generators[figure_id](df, output_dir)

    return generated


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory to receive fig0/1/2/3/4/5 outputs.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Raw synthetic benchmark CSV to visualize.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Benchmark artifact root containing synthetic/*/*_seed*_summary.json.",
    )
    parser.add_argument(
        "--figure",
        dest="figure_ids",
        action="append",
        choices=DEFAULT_FIGURE_IDS,
        help="Generate only the selected figure id. Repeat to request multiple figures.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    generated = generate_publication_figures(
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        artifact_root=args.artifact_root,
        figure_ids=args.figure_ids or DEFAULT_FIGURE_IDS,
    )
    for figure_id, path in generated.items():
        print(f"{figure_id}: {path}")


if __name__ == "__main__":
    main()
