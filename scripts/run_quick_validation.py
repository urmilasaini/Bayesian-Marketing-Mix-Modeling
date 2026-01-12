#!/usr/bin/env python
"""Quick validation script for real data benchmark.

Runs a minimal MCMC test on 1 organisation with reduced samples to verify
the pipeline works end-to-end.

Usage:
    uv run python scripts/run_quick_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
from hill_mixture_mmm.models import model_single_hill


def main():
    csv_path = "data/conjura_mmm_data.csv"

    print("=" * 60)
    print("Quick Validation: Real Conjura Data + Hill MMM")
    print("=" * 60)

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
    print(f"    T={data.meta['T']}, Channels={data.meta['n_channels']}")
    print(f"    Vertical: {data.meta['organisation_vertical']}")

    T = len(data.y)
    T_train = int(T * 0.75)
    t_std_full = standardized_time_index(T)
    x_train, y_train = data.x[:T_train], data.y[:T_train]
    x_test, y_test = data.x[T_train:], data.y[T_train:]
    print(f"    Train: {T_train}, Test: {T - T_train}")

    print("\n[3] Computing priors...")
    prior_config = compute_prior_config(x_train, y_train)
    print(f"    intercept_loc={prior_config['intercept_loc']:.1f}")
    print(f"    x_median={prior_config['x_median']:.1f}")

    print("\n[4] Running MCMC (reduced samples for quick validation)...")
    print("    This may take 1-2 minutes...")
    mcmc = run_inference(
        model_single_hill,
        x_train,
        y_train,
        seed=42,
        num_warmup=200,
        num_samples=400,
        num_chains=2,
        prior_config=prior_config,
        t_std=t_std_full[:T_train],
    )

    print("\n[5] Checking convergence...")
    diag = compute_convergence_diagnostics(mcmc)
    print(f"    max_rhat: {diag['max_rhat']:.3f} (should be < 1.05)")
    print(f"    min_ess_bulk: {diag['min_ess_bulk']:.0f}")
    print(f"    converged: {diag['converged']}")

    print("\n[6] Computing LOO...")
    loo = compute_loo(mcmc)
    print(f"    elpd_loo: {loo.get('elpd_loo', 'N/A'):.1f}")
    print(f"    p_loo: {loo.get('p_loo', 'N/A'):.1f}")

    print("\n[7] Computing predictions...")
    pred_train = compute_predictions(
        mcmc,
        model_single_hill,
        x_train,
        prior_config=prior_config,
        total_time=T,
    )
    pred_test = compute_predictions(
        mcmc,
        model_single_hill,
        x_test,
        prior_config=prior_config,
        history_x=x_train,
        total_time=T,
    )

    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

    print(f"    Train MAPE: {train_metrics['mape']:.2f}%")
    print(f"    Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"    Train Coverage 90%: {train_metrics['coverage_90']:.1%}")
    print(f"    Test Coverage 90%: {test_metrics['coverage_90']:.1%}")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    success = diag["converged"] and test_metrics["coverage_90"] > 0.5

    if success:
        print("\n  Status: SUCCESS")
        print("  - MCMC converged (R-hat < 1.05)")
        print("  - Reasonable predictive coverage")
        print("  - Pipeline works end-to-end")
        print("\n  Ready for full benchmark with:")
        print("  >>> See scripts/run_benchmark.py --real-only")
        print("  >>> results = run_real_benchmark(n_timeseries=5)")
    else:
        print("\n  Status: NEEDS INVESTIGATION")
        if not diag["converged"]:
            print("  - MCMC did not converge (may need more samples)")
        if test_metrics["coverage_90"] <= 0.5:
            print("  - Low predictive coverage")

    print("\n")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
