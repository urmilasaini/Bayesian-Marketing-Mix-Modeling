"""Aggregate per-org summary JSONs from the full real-data benchmark.

Reads ``paper/figures/real/{model}/real_{dataset}_{model}_seed{s}_summary.json``
for the three datasets, three models, and three seeds, and writes:

  - ``results/real_benchmark_raw.csv``: one row per case
  - ``results/real_benchmark_summary.csv``: mean/std per (dataset, model)

Prints a compact table for terminal review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = REPO_ROOT / "paper" / "figures" / "real"
RAW_OUT = REPO_ROOT / "results" / "real_benchmark_raw.csv"
SUMMARY_OUT = REPO_ROOT / "results" / "real_benchmark_summary.csv"

DATASETS = ["beauty_fitness", "home_garden", "toys_hobbies"]
MODELS = ["single_hill", "mixture_k2", "mixture_k3"]
SEEDS = [0, 1, 2]


def _extract_row(path: Path) -> dict:
    data = json.loads(path.read_text())
    test_metrics = data.get("test_metrics", {})
    train_metrics = data.get("train_metrics", {})
    loo = data.get("loo", {})
    eff_k = data.get("effective_k", {})
    return {
        "dataset": data.get("dataset_name"),
        "model": data.get("model_name"),
        "seed": data.get("seed"),
        "converged": data.get("converged"),
        "publication_status": data.get("publication_status"),
        "elpd_loo": loo.get("elpd_loo"),
        "loo_se": loo.get("se"),
        "pareto_k_bad": loo.get("pareto_k_bad"),
        "pareto_k_very_bad": loo.get("pareto_k_very_bad"),
        "test_mape": test_metrics.get("mape"),
        "test_nrmse": test_metrics.get("nrmse"),
        "test_crps": test_metrics.get("crps"),
        "test_coverage_90": test_metrics.get("coverage_90"),
        "train_mape": train_metrics.get("mape"),
        "effective_k_mean": eff_k.get("mean"),
    }


def main() -> None:
    rows: list[dict] = []
    missing: list[str] = []

    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                fname = f"real_{dataset}_{model}_seed{seed}_summary.json"
                path = ARTIFACT_ROOT / model / fname
                if not path.exists():
                    missing.append(str(path.relative_to(REPO_ROOT)))
                    continue
                row = _extract_row(path)
                row["dataset_label"] = dataset
                rows.append(row)

    if missing:
        print(f"Warning: {len(missing)} summaries missing:")
        for m in missing:
            print(f"  {m}")
        print()

    df = pd.DataFrame(rows)

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUT, index=False)

    agg_cols = [
        "elpd_loo",
        "test_mape",
        "test_nrmse",
        "test_crps",
        "test_coverage_90",
        "effective_k_mean",
        "pareto_k_bad",
    ]
    summary = (
        df.groupby(["dataset_label", "model"])[agg_cols]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.to_csv(SUMMARY_OUT)

    print(f"Wrote {RAW_OUT.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    print(f"Wrote {SUMMARY_OUT.relative_to(REPO_ROOT)}")
    print()
    print("=" * 80)
    print("Per-org × model summary (mean across 3 seeds)")
    print("=" * 80)

    pivot_cols = ["elpd_loo", "test_mape", "test_nrmse", "test_crps", "test_coverage_90"]
    mean_table = (
        df.groupby(["dataset_label", "model"])[pivot_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(mean_table.to_string(index=False))
    print()

    print("=" * 80)
    print("Convergence and effective K")
    print("=" * 80)
    conv_table = (
        df.groupby(["dataset_label", "model"])
        .agg(
            converged_rate=("converged", "mean"),
            pub_pass_rate=("publication_status", lambda s: (s == "Pass").mean()),
            eff_k_mean=("effective_k_mean", "mean"),
            pareto_k_bad_mean=("pareto_k_bad", "mean"),
        )
        .round(2)
        .reset_index()
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(conv_table.to_string(index=False))


if __name__ == "__main__":
    main()
