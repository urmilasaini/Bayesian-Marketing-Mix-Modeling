#!/usr/bin/env python
"""Real data validation: Compare Single Hill vs Mixture Model.

Evaluates both models on real Conjura MMM data using:
1. ELPD-LOO (Expected Log Pointwise Predictive Density via PSIS-LOO)
2. R-hat convergence diagnostics (with mixture-appropriate diagnostics)

Usage:
    python scripts/run_real_data_validation.py
    python scripts/run_real_data_validation.py --quick
    python scripts/run_real_data_validation.py --org ORG_ID
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hill_mixture_mmm.data_loader import (  # noqa: E402
    LoadedData,
    TimeSeriesConfig,
    list_timeseries,
    load_timeseries,
)
from hill_mixture_mmm.inference import (  # noqa: E402
    compute_convergence_diagnostics,
    compute_diagnostics_on_relabeled,
    compute_label_invariant_diagnostics,
    compute_loo,
    relabel_samples_by_k,
    run_inference,
)
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam, model_single_hill  # noqa: E402

DATA_PATH = project_root / "data" / "conjura_mmm_data.csv"
RESULTS_DIR = project_root / "results" / "real_data_validation"


def run_single_hill_model(
    data: LoadedData,
    config: dict,
    seed: int = 42,
) -> dict:
    """Run single Hill model and compute diagnostics."""
    print("\n" + "=" * 60)
    print("Running Single Hill Model")
    print("=" * 60)

    start_time = time.time()

    mcmc = run_inference(
        model_fn=model_single_hill,
        x=data.x,
        y=data.y,
        seed=seed,
        num_warmup=config["num_warmup"],
        num_samples=config["num_samples"],
        num_chains=config["num_chains"],
    )

    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.1f} seconds")

    print("Computing LOO-CV...")
    loo_results = compute_loo(mcmc)

    print("Computing convergence diagnostics...")
    conv_results = compute_convergence_diagnostics(mcmc)

    samples = mcmc.get_samples()
    param_summary = {
        "A_mean": float(np.mean(samples["A"])),
        "A_std": float(np.std(samples["A"])),
        "k_mean": float(np.mean(samples["k"])),
        "k_std": float(np.std(samples["k"])),
        "n_mean": float(np.mean(samples["n"])),
        "n_std": float(np.std(samples["n"])),
        "alpha_mean": float(np.mean(samples["alpha"])),
        "sigma_mean": float(np.mean(samples["sigma"])),
    }

    return {
        "model": "single_hill",
        "elapsed_seconds": elapsed,
        "loo": loo_results,
        "convergence": conv_results,
        "params": param_summary,
    }


def run_mixture_model(
    data: LoadedData,
    config: dict,
    K: int = 3,
    seed: int = 42,
) -> dict:
    """Run hierarchical mixture model with proper diagnostics."""
    print("\n" + "=" * 60)
    print(f"Running Hierarchical Mixture Model (K={K})")
    print("=" * 60)

    start_time = time.time()

    mcmc = run_inference(
        model_fn=model_hill_mixture_hierarchical_reparam,
        x=data.x,
        y=data.y,
        seed=seed,
        num_warmup=config["num_warmup"],
        num_samples=config["num_samples"],
        num_chains=config["num_chains"],
        K=K,
    )

    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.1f} seconds")

    print("Computing LOO-CV...")
    loo_results = compute_loo(mcmc)

    print("Computing standard convergence diagnostics...")
    conv_standard = compute_convergence_diagnostics(mcmc)

    print("Computing label-invariant diagnostics...")
    try:
        conv_invariant = compute_label_invariant_diagnostics(mcmc, data.x, data.y)
    except Exception as e:
        print(f"  Warning: label-invariant diagnostics failed: {e}")
        conv_invariant = {"error": str(e)}

    print("Computing diagnostics on relabeled samples...")
    try:
        conv_relabeled = compute_diagnostics_on_relabeled(mcmc)
    except Exception as e:
        print(f"  Warning: relabeled diagnostics failed: {e}")
        conv_relabeled = {"error": str(e)}

    samples = mcmc.get_samples()
    relabeled = relabel_samples_by_k(samples)

    param_summary = {
        "A_means": [float(x) for x in np.mean(relabeled["A"], axis=0)],
        "k_means": [float(x) for x in np.mean(relabeled["k"], axis=0)],
        "n_means": [float(x) for x in np.mean(relabeled["n"], axis=0)],
        "pis_means": [float(x) for x in np.mean(relabeled["pis"], axis=0)],
        "alpha_mean": float(np.mean(samples["alpha"])),
        "sigma_mean": float(np.mean(samples["sigma"])),
    }

    return {
        "model": f"mixture_k{K}",
        "K": K,
        "elapsed_seconds": elapsed,
        "loo": loo_results,
        "convergence_standard": conv_standard,
        "convergence_label_invariant": conv_invariant,
        "convergence_relabeled": conv_relabeled,
        "params": param_summary,
    }


def format_results_table(single_results: dict, mixture_results: dict) -> str:
    """Format comparison table."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("COMPARISON SUMMARY")
    lines.append("=" * 70)

    lines.append("\n### Model Comparison (ELPD-LOO)")
    lines.append("-" * 50)

    single_elpd = single_results["loo"].get("elpd_loo", np.nan)
    mixture_elpd = mixture_results["loo"].get("elpd_loo", np.nan)
    single_se = single_results["loo"].get("se", np.nan)
    mixture_se = mixture_results["loo"].get("se", np.nan)

    lines.append(f"{'Model':<25} {'ELPD-LOO':>12} {'SE':>10} {'p_loo':>10}")
    lines.append("-" * 57)
    lines.append(
        f"{'Single Hill':<25} {single_elpd:>12.1f} {single_se:>10.1f} "
        f"{single_results['loo'].get('p_loo', np.nan):>10.1f}"
    )
    lines.append(
        f"{mixture_results['model']:<25} {mixture_elpd:>12.1f} {mixture_se:>10.1f} "
        f"{mixture_results['loo'].get('p_loo', np.nan):>10.1f}"
    )

    if not np.isnan(single_elpd) and not np.isnan(mixture_elpd):
        delta = mixture_elpd - single_elpd
        delta_se = np.sqrt(single_se**2 + mixture_se**2) if not np.isnan(single_se) else np.nan
        lines.append(f"\nDelta (Mixture - Single): {delta:+.1f} (SE: {delta_se:.1f})")
        if delta > 0:
            lines.append("  → Mixture model has BETTER predictive performance")
        else:
            lines.append("  → Single model has BETTER predictive performance")

    lines.append("\n### Convergence Diagnostics (R-hat)")
    lines.append("-" * 50)
    lines.append(f"{'Model':<25} {'Max R-hat':>12} {'Converged':>12}")
    lines.append("-" * 49)

    single_rhat = single_results["convergence"].get("max_rhat", np.nan)
    single_conv = single_results["convergence"].get("converged", False)
    lines.append(f"{'Single Hill':<25} {single_rhat:>12.4f} {'Yes' if single_conv else 'No':>12}")

    std_rhat = mixture_results["convergence_standard"].get("max_rhat", np.nan)
    lines.append(f"{'Mixture (standard)':<25} {std_rhat:>12.4f} {'(unreliable)':>12}")

    if "max_rhat" in mixture_results.get("convergence_relabeled", {}):
        rel_rhat = mixture_results["convergence_relabeled"]["max_rhat"]
        rel_conv = mixture_results["convergence_relabeled"].get("converged", False)
        lines.append(
            f"{'Mixture (relabeled)':<25} {rel_rhat:>12.4f} {'Yes' if rel_conv else 'No':>12}"
        )

    if "rhat_log_lik" in mixture_results.get("convergence_label_invariant", {}):
        ll_rhat = mixture_results["convergence_label_invariant"]["rhat_log_lik"]
        lines.append(f"{'Mixture (log-lik)':<25} {ll_rhat:>12.4f}")

    lines.append("\n### Computation Time")
    lines.append("-" * 50)
    lines.append(f"Single Hill: {single_results['elapsed_seconds']:.1f} seconds")
    lines.append(f"Mixture:     {mixture_results['elapsed_seconds']:.1f} seconds")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Real data validation for MMM models")
    parser.add_argument("--quick", action="store_true", help="Use quick settings for testing")
    parser.add_argument("--org", type=str, help="Specific organisation_id to use")
    parser.add_argument("--K", type=int, default=3, help="Number of mixture components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.quick:
        config = {
            "num_warmup": 200,
            "num_samples": 400,
            "num_chains": 2,
        }
        print("Using QUICK settings (for testing only)")
    else:
        config = {
            "num_warmup": 1000,
            "num_samples": 2000,
            "num_chains": 4,
        }
        print("Using FULL settings")

    print(
        f"MCMC config: warmup={config['num_warmup']}, samples={config['num_samples']}, "
        f"chains={config['num_chains']}"
    )

    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please ensure conjura_mmm_data.csv is in the data/ directory.")
        sys.exit(1)

    if args.org:
        org_id = args.org
        print(f"\nUsing specified organisation: {org_id}")
    else:
        print("\nDiscovering available time series...")
        ts_info = list_timeseries(DATA_PATH, min_length=200)
        print(f"Found {len(ts_info)} time series with >= 200 days")

        if len(ts_info) == 0:
            print("ERROR: No suitable time series found")
            sys.exit(1)

        ts_info_sorted = ts_info.sort_values(
            by=["n_active_channels", "n_days"], ascending=[False, False]
        )
        org_id = ts_info_sorted.iloc[0]["organisation_id"]
        org_info = ts_info_sorted.iloc[0]
        print(f"\nSelected organisation: {org_id}")
        print(f"  - Days: {org_info['n_days']}")
        print(f"  - Active channels: {org_info['n_active_channels']}")
        print(f"  - Vertical: {org_info['organisation_vertical']}")

    print(f"\nLoading data for {org_id}...")
    data = load_timeseries(
        DATA_PATH,
        TimeSeriesConfig(organisation_id=org_id, aggregate_spend=True),
    )
    print(f"Loaded: T={data.meta['T']}, spend range=[{data.x.min():.0f}, {data.x.max():.0f}]")
    print(f"Target range: [{data.y.min():.0f}, {data.y.max():.0f}]")

    single_results = run_single_hill_model(data, config, seed=args.seed)
    mixture_results = run_mixture_model(data, config, K=args.K, seed=args.seed)

    summary = format_results_table(single_results, mixture_results)
    print(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": timestamp,
        "config": config,
        "organisation_id": org_id,
        "data_meta": data.meta,
        "single_hill": single_results,
        "mixture": mixture_results,
    }

    json_path = RESULTS_DIR / f"validation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    txt_path = RESULTS_DIR / f"validation_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("Real Data Validation Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Organisation: {org_id}\n")
        f.write(f"Data: T={data.meta['T']}\n")
        f.write(summary)
    print(f"Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
