from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path


def _ensure_xla_host_device_count(device_count: int) -> None:
    """Append a host-device-count flag without clobbering unrelated XLA flags."""
    host_device_flag = f"--xla_force_host_platform_device_count={device_count}"
    tokens = os.environ.get("XLA_FLAGS", "").split()
    if host_device_flag not in tokens:
        tokens.append(host_device_flag)
        os.environ["XLA_FLAGS"] = " ".join(tokens)


_ensure_xla_host_device_count(4)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hill_mixture_mmm.benchmark import (  # noqa: E402
    BenchmarkThresholds,
    assert_case_passes,
    run_prepared_synthetic_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.component_resolvability import (  # noqa: E402
    SMOKE_PROFILE_IDS,
    RESOLVABILITY_PROFILE_LIBRARY,
    build_resolvability_config,
    build_resolvability_run_config,
)
from hill_mixture_mmm.data import generate_controlled_k_spacing_data  # noqa: E402
from hill_mixture_mmm.metrics import (  # noqa: E402
    compute_component_curve_cosine_separation,
    compute_component_curve_nabc_separation,
    compute_component_curve_tv_separation,
    compute_nabc_effective_count,
    compute_shannon_effective_count,
    compute_similarity_adjusted_effective_count,
)

MODELS = ["mixture_k2", "mixture_k3"]
SMOKE_SEEDS = [0]
FULL_SEEDS = [0, 1, 2, 3, 4]

_PLOT_STYLE = {
    "model_labels": {"mixture_k2": "Mixture (K=2)", "mixture_k3": "Mixture (K=3)"},
    "model_colors": {"mixture_k2": "#9467bd", "mixture_k3": "#ff7f0e"},
    "k_true_markers": {1: "o", 2: "s", 3: "^"},
    "k_true_linestyles": {2: "--", 3: "-"},
    "figsize": (10, 5.2),
    "scatter_size": 48,
    "scatter_alpha": 0.75,
    "line_width": 2.0,
    "line_alpha": 0.9,
    "seed_jitter": 0.008,
    "dpi": 200,
}

_RELAXED_BASE = BenchmarkThresholds(
    max_rhat=None,
    min_ess_bulk_per_chain=None,
    min_ess_tail_per_chain=None,
    max_divergences=None,
    min_bfmi=None,
    max_tree_depth_hits=None,
    require_finite_loo_waic=True,
    require_finite_predictive_metrics=True,
)

_DATASET_RE = re.compile(r"^resolvability_k(?P<k_true>\d+)_(?P<profile_id>.+)$")


def _require_full_component_resolvability_benchmark() -> None:
    enabled = os.getenv("HILL_MMM_RUN_FULL_COMPONENT_RESOLVABILITY_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full component-resolvability benchmark is opt-in; set "
            "HILL_MMM_RUN_FULL_COMPONENT_RESOLVABILITY_BENCHMARK=1 to run it"
        )


def _smoke_profile_cases() -> list[tuple[int, dict[str, object]]]:
    cases: list[tuple[int, dict[str, object]]] = []
    for k_true, profile_ids in SMOKE_PROFILE_IDS.items():
        profiles = {str(profile["profile_id"]): profile for profile in RESOLVABILITY_PROFILE_LIBRARY[k_true]}
        for profile_id in profile_ids:
            cases.append((k_true, profiles[profile_id]))
    return cases


def _full_profile_cases() -> list[tuple[int, dict[str, object]]]:
    return [
        (k_true, profile) for k_true, profiles in RESOLVABILITY_PROFILE_LIBRARY.items() for profile in profiles
    ]


def _controlled_thresholds() -> BenchmarkThresholds:
    return replace(
        _RELAXED_BASE,
        require_reportable_diagnostics=True,
        require_truth_metrics=True,
    )


def _dataset_name(k_true: int, profile_id: str) -> str:
    return f"resolvability_k{k_true}_{profile_id}"


def _label(k_true: int, profile_id: str, model_name: str, seed: int) -> str:
    return f"{_dataset_name(k_true, profile_id)}_{model_name}_seed{seed}"


def _case_summary_path(
    benchmark_output_root: Path,
    *,
    k_true: int,
    profile_id: str,
    model_name: str,
    seed: int,
) -> Path:
    return (
        benchmark_output_root
        / "synthetic"
        / model_name
        / f"{_label(k_true, profile_id, model_name, seed)}_summary.json"
    )


def _run_and_assert_controlled_case(
    *,
    k_true: int,
    profile: dict[str, object],
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
    quick: bool,
) -> None:
    profile_id = str(profile["profile_id"])
    config = build_resolvability_config(k_true=k_true, seed=seed, profile=profile)
    x, y, meta = generate_controlled_k_spacing_data(config)
    result = run_prepared_synthetic_benchmark_case(
        dataset_name=_dataset_name(k_true, profile_id),
        x=x,
        y=y,
        meta=meta,
        model_name=model_name,
        config=build_resolvability_run_config(
            model_name,
            seed,
            quick=quick,
        ),
        label=_label(k_true, profile_id, model_name, seed),
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    assert_case_passes(result, _controlled_thresholds())
    for path in artifacts.values():
        assert path.exists(), f"Expected component-resolvability artifact at {path}"


_TRUE_SEPARATION_METRICS: list[tuple[str, object, str]] = [
    ("true_separation", compute_component_curve_tv_separation, "mean_pairwise_tv"),
    ("true_nabc_separation", compute_component_curve_nabc_separation, "mean_pairwise_nabc"),
    ("true_cosine_separation", compute_component_curve_cosine_separation, "mean_pairwise_cosine"),
]

_EFFECTIVE_COUNT_METRICS: list[tuple[str, object]] = [
    ("similarity_adjusted_count", compute_similarity_adjusted_effective_count),
    ("nabc_effective_count", compute_nabc_effective_count),
    ("shannon_count", compute_shannon_effective_count),
]


def _parse_summary_row(summary: dict[str, object]) -> dict[str, object]:
    match = _DATASET_RE.match(str(summary["dataset_name"]))
    assert match, f"Unexpected resolvability dataset_name: {summary['dataset_name']}"
    k_true = int(match.group("k_true"))
    profile_id = match.group("profile_id")

    true_component_summary = summary.get("true_component_summary")
    separation_values = {
        col: float(func(true_component_summary)[key]) if true_component_summary is not None else 0.0
        for col, func, key in _TRUE_SEPARATION_METRICS
    }

    component_summary = summary.get("component_summary")
    if component_summary is None:
        count_values = {"active_component_count": 1.0}
        count_values.update({col: 1.0 for col, _ in _EFFECTIVE_COUNT_METRICS})
    else:
        count_values = {"active_component_count": float(component_summary["K_active"])}
        count_values.update(
            {
                col: float(func(component_summary)["effective_count"])
                for col, func in _EFFECTIVE_COUNT_METRICS
            }
        )

    return {
        "seed": int(summary["seed"]),
        "K_true": k_true,
        "profile_id": profile_id,
        "model": str(summary["model_name"]),
        **separation_values,
        "mcmc_converged": bool(summary["converged"]),
        "benchmark_pass": bool(summary["benchmark_pass"]),
        "publication_status": str(summary["publication_status"]),
        "sampler_status": str(summary["diagnostic_status"]["sampler_status"]),
        "mixing_status": str(summary["diagnostic_status"]["mixing_status"]),
        "interpretation_status": str(summary["diagnostic_status"]["interpretation_status"]),
        **count_values,
    }


def _load_selected_metric_rows(summary_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        rows.append(_parse_summary_row(summary))
    return (
        pd.DataFrame(rows)
        .sort_values(["K_true", "true_cosine_separation", "seed", "model"])
        .reset_index(drop=True)
    )


def _build_legend_handles(df: pd.DataFrame, models: list[str]) -> list[object]:
    style = _PLOT_STYLE
    model_handles = [
        plt.Line2D(
            [], [], color=style["model_colors"][m], linewidth=1.8, label=style["model_labels"][m]
        )
        for m in models
    ]
    marker_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="None",
            marker=style["k_true_markers"][k],
            markersize=7,
            markerfacecolor="0.35",
            markeredgecolor="white",
            label=f"Data K={k}",
        )
        for k in sorted(pd.unique(df["K_true"]))
    ]
    linestyles = style["k_true_linestyles"]
    line_handles = [
        plt.Line2D(
            [], [], color="0.35", linewidth=1.8, linestyle=linestyles[k], label=f"Mean: Data K={k}"
        )
        for k in sorted(k for k in pd.unique(df["K_true"]) if int(k) in linestyles)
    ]
    convergence_handle = [
        plt.Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor="0.7",
            markeredgecolor="red",
            markeredgewidth=1.5,
            label="Convergence issue",
        )
    ]
    return model_handles + marker_handles + line_handles + convergence_handle


def _plot_selected_metrics(df: pd.DataFrame, *, models: list[str], output_path: Path) -> None:
    style = _PLOT_STYLE
    metric_key = "shannon_count"
    metric_title = "Shannon Count (Hill q=1)"

    fig, axes = plt.subplots(1, 2, figsize=style["figsize"], gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    seed_values = sorted(pd.unique(df["seed"]))
    seed_offsets = {
        int(seed): (idx - (len(seed_values) - 1) / 2) * style["seed_jitter"]
        for idx, seed in enumerate(seed_values)
    }
    has_pass_col = "benchmark_pass" in df.columns

    for target in [1, 2, 3]:
        ax.axhline(target, color="0.88", linestyle="--", linewidth=0.8, zorder=0)
    for model_name in models:
        model_panel = df[df["model"] == model_name]
        for k_true in sorted(pd.unique(model_panel["K_true"])):
            panel = model_panel[model_panel["K_true"] == k_true]
            if panel.empty:
                continue
            for _, row in panel.iterrows():
                x_val = row["true_cosine_separation"] + seed_offsets[int(row["seed"])]
                failed = has_pass_col and not row["benchmark_pass"]
                ax.scatter(
                    x_val,
                    row[metric_key],
                    color=style["model_colors"][model_name],
                    marker=style["k_true_markers"][int(k_true)],
                    s=style["scatter_size"],
                    alpha=style["scatter_alpha"],
                    edgecolors="red" if failed else "white",
                    linewidths=1.2 if failed else 0.5,
                    zorder=3,
                )
            means = (
                panel.groupby("profile_id", as_index=False)
                .agg(
                    true_cosine_separation=("true_cosine_separation", "mean"),
                    y=(metric_key, "mean"),
                )
                .sort_values("true_cosine_separation")
            )
            ax.plot(
                means["true_cosine_separation"],
                means["y"],
                color=style["model_colors"][model_name],
                linewidth=style["line_width"],
                alpha=style["line_alpha"],
                linestyle=style["k_true_linestyles"].get(int(k_true), "-"),
            )
    ax.set_title(metric_title, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.7, 3.3)
    ax.set_xlabel("True Component Cosine Distance")
    ax.set_ylabel("Estimated Component Count")
    ax.grid(True, alpha=0.22)

    legend_ax = axes[1]
    legend_ax.axis("off")
    legend_ax.legend(handles=_build_legend_handles(df, models), loc="center", frameon=False)

    fig.suptitle("Component Resolvability Benchmark", y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
    plt.close(fig)


def _collect_summary_paths(
    benchmark_output_root: Path,
    *,
    cases: list[tuple[int, dict[str, object]]],
    models: list[str],
    seeds: list[int],
) -> list[Path]:
    paths = [
        _case_summary_path(
            benchmark_output_root,
            k_true=k_true,
            profile_id=str(profile["profile_id"]),
            model_name=model_name,
            seed=seed,
        )
        for k_true, profile in cases
        for model_name in models
        for seed in seeds
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"{len(missing)} summary file(s) not found (matrix tests not yet run?)")
    return paths


def _summarize_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["K_true", "profile_id", "model"], as_index=False)
        .agg(
            true_separation=("true_separation", "mean"),
            true_nabc_separation=("true_nabc_separation", "mean"),
            true_cosine_separation=("true_cosine_separation", "mean"),
            active_component_count=("active_component_count", "mean"),
            similarity_adjusted_count=("similarity_adjusted_count", "mean"),
            nabc_effective_count=("nabc_effective_count", "mean"),
            shannon_count=("shannon_count", "mean"),
            benchmark_pass_rate=("benchmark_pass", "mean"),
            mcmc_converged_rate=("mcmc_converged", "mean"),
            publication_pass_rate=("publication_status", lambda s: float((s == "Pass").mean())),
            publication_fail_rate=("publication_status", lambda s: float((s == "Fail").mean())),
        )
        .sort_values(["K_true", "true_cosine_separation", "model"])
        .reset_index(drop=True)
    )


def _write_selected_metric_artifacts(
    *,
    benchmark_output_root: Path,
    cases: list[tuple[int, dict[str, object]]],
    seeds: list[int],
    models: list[str],
    artifact_name: str,
) -> None:
    summary_paths = _collect_summary_paths(
        benchmark_output_root,
        cases=cases,
        models=models,
        seeds=seeds,
    )
    df = _load_selected_metric_rows(summary_paths)
    summary = _summarize_metric_rows(df)

    artifact_dir = benchmark_output_root / "component_resolvability" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = artifact_dir / "selected_metric_results.csv"
    summary_csv = artifact_dir / "selected_metric_summary.csv"
    plot_path = artifact_dir / "selected_metric_comparison.png"
    metadata_path = artifact_dir / "metadata.json"

    df.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_selected_metrics(df, models=models, output_path=plot_path)
    metadata_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "models": models,
                "separation_metrics": [col for col, _, _ in _TRUE_SEPARATION_METRICS],
                "effective_count_metrics": [col for col, _ in _EFFECTIVE_COUNT_METRICS],
                "cases": [
                    {"K_true": int(k_true), "profile_id": str(profile["profile_id"])}
                    for k_true, profile in cases
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert raw_csv.exists()
    assert summary_csv.exists()
    assert plot_path.exists()
    assert metadata_path.exists()
    assert len(df) == len(cases) * len(models) * len(seeds)


_TIERS: dict[str, dict[str, object]] = {
    "smoke": {
        "cases_fn": _smoke_profile_cases,
        "seeds": SMOKE_SEEDS,
        "models": MODELS,
        "mark": pytest.mark.benchmark_smoke,
        "guard": lambda: None,
    },
    "full": {
        "cases_fn": _full_profile_cases,
        "seeds": FULL_SEEDS,
        "models": MODELS,
        "mark": pytest.mark.benchmark_full,
        "guard": _require_full_component_resolvability_benchmark,
    },
}


def _matrix_test_params() -> list[object]:
    params: list[object] = []
    for tier_name, cfg in _TIERS.items():
        for k_true, profile in cfg["cases_fn"]():
            pid = str(profile["profile_id"])
            for model_name in cfg["models"]:
                for seed in cfg["seeds"]:
                    params.append(
                        pytest.param(
                            tier_name,
                            k_true,
                            profile,
                            model_name,
                            seed,
                            marks=[cfg["mark"]],
                            id=f"{tier_name}-k{k_true}-{pid}-{model_name}-s{seed}",
                        )
                    )
    return params


@pytest.mark.slow
@pytest.mark.parametrize(
    ("tier", "k_true", "profile", "model_name", "seed"),
    _matrix_test_params(),
)
def test_component_resolvability_matrix(
    tier: str,
    k_true: int,
    profile: dict[str, object],
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _TIERS[tier]["guard"]()
    _run_and_assert_controlled_case(
        k_true=k_true,
        profile=profile,
        model_name=model_name,
        seed=seed,
        benchmark_output_root=benchmark_output_root,
        quick=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "tier",
    [
        pytest.param("smoke", marks=pytest.mark.benchmark_smoke),
        pytest.param("full", marks=pytest.mark.benchmark_full),
    ],
)
def test_component_resolvability_selected_metric_artifacts(
    tier: str,
    benchmark_output_root: Path,
) -> None:
    cfg = _TIERS[tier]
    cfg["guard"]()
    _write_selected_metric_artifacts(
        benchmark_output_root=benchmark_output_root,
        cases=cfg["cases_fn"](),
        seeds=cfg["seeds"],
        models=cfg["models"],
        artifact_name=tier,
    )
