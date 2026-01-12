#!/usr/bin/env python
"""Compare benchmark-aligned Hill MMM variants on real Conjura data."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from hill_mixture_mmm.baseline import standardized_time_index
from hill_mixture_mmm.data import compute_prior_config
from hill_mixture_mmm.data_loader import (
    TimeSeriesConfig,
    load_timeseries,
    select_representative_timeseries,
)
from hill_mixture_mmm.inference import (
    compute_convergence_diagnostics,
    compute_loo,
    compute_predictions,
    run_inference,
)
from hill_mixture_mmm.metrics import compute_predictive_metrics
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam, model_single_hill


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    name: str
    fn: object
    kwargs: dict


MODEL_SPECS = [
    ModelSpec("single_hill", model_single_hill, {}),
    ModelSpec("mixture_k2", model_hill_mixture_hierarchical_reparam, {"K": 2}),
    ModelSpec("mixture_k3", model_hill_mixture_hierarchical_reparam, {"K": 3}),
]


def main():
    csv_path = "data/conjura_mmm_data.csv"

    NUM_WARMUP = 200
    NUM_SAMPLES = 400
    NUM_CHAINS = 2
    TRAIN_RATIO = 0.75

    print("=" * 70)
    print("Model Comparison: Single Hill vs Mixture Models")
    print("=" * 70)

    print("\n[1] Selecting organisation...")
    org_ids = select_representative_timeseries(
        csv_path, n=1, min_length=300, min_channels=2, seed=42
    )
    org_id = org_ids[0]
    print(f"    Selected: {org_id[:20]}...")

    print("\n[2] Loading data...")
    config = TimeSeriesConfig(
        organisation_id=org_id,
        target_col="all_purchases",
        aggregate_spend=True,
    )
    data = load_timeseries(csv_path, config)

    T = len(data.y)
    T_train = int(T * TRAIN_RATIO)
    t_std_full = standardized_time_index(T)
    x_train, y_train = data.x[:T_train], data.y[:T_train]
    x_test, y_test = data.x[T_train:], data.y[T_train:]

    print(f"    T={T}, Train={T_train}, Test={T - T_train}")

    prior_config = compute_prior_config(x_train, y_train)

    results = []

    for spec in MODEL_SPECS:
        print(f"\n[3] Running {spec.name}...")

        try:
            mcmc = run_inference(
                spec.fn,
                x_train,
                y_train,
                seed=42,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS,
                prior_config=prior_config,
                t_std=t_std_full[:T_train],
                **spec.kwargs,
            )

            diag = compute_convergence_diagnostics(mcmc)
            print(f"    R-hat: {diag['max_rhat']:.3f}, ESS: {diag['min_ess_bulk']:.0f}")

            loo_result = compute_loo(mcmc)

            preds_train = compute_predictions(
                mcmc,
                spec.fn,
                x_train,
                prior_config,
                total_time=T,
                **spec.kwargs,
            )
            preds_test = compute_predictions(
                mcmc,
                spec.fn,
                x_test,
                prior_config,
                history_x=x_train,
                total_time=T,
                **spec.kwargs,
            )

            metrics_train = compute_predictive_metrics(y_train, preds_train["y"])
            metrics_test = compute_predictive_metrics(y_test, preds_test["y"])

            results.append(
                {
                    "model": spec.name,
                    "elpd_loo": loo_result["elpd_loo"],
                    "p_loo": loo_result["p_loo"],
                    "train_mape": metrics_train["mape"],
                    "test_mape": metrics_test["mape"],
                    "train_coverage": metrics_train["coverage_90"],
                    "test_coverage": metrics_test["coverage_90"],
                    "converged": diag["converged"],
                    "max_rhat": diag["max_rhat"],
                }
            )

            print(f"    ELPD-LOO: {loo_result['elpd_loo']:.1f}")
            print(f"    Test MAPE: {metrics_test['mape']:.2f}%")

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    ERROR: {e}")
            results.append(
                {
                    "model": spec.name,
                    "elpd_loo": np.nan,
                    "p_loo": np.nan,
                    "train_mape": np.nan,
                    "test_mape": np.nan,
                    "train_coverage": np.nan,
                    "test_coverage": np.nan,
                    "converged": False,
                    "max_rhat": np.nan,
                }
            )

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    if len(df) > 1 and not bool(df["elpd_loo"].isna().all()):
        best_idx = df["elpd_loo"].idxmax()
        best_model = df.loc[best_idx, "model"]
        best_elpd = df.loc[best_idx, "elpd_loo"]

        print(f"\nBest model by ELPD-LOO: {best_model}")
        print("\nDelta ELPD from best:")
        for _, row in df.iterrows():
            delta = row["elpd_loo"] - best_elpd
            print(f"  {row['model']}: {delta:+.1f}")

    df.to_csv("results/model_comparison.csv", index=False)
    print("\nResults saved to results/model_comparison.csv")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
