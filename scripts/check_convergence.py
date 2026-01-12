#!/usr/bin/env python3
"""Test convergence of LogNormal hyperprior model on real data.

Validates that the LogNormal hyperprior improvement addresses the funnel
convergence problem identified by zen-architect.

Usage:
    python scripts/test_convergence.py
    python scripts/test_convergence.py --n-orgs 5 --warmup 1000 --samples 1000

Output:
    Summary table with R-hat, ESS, and convergence status per organization.
"""

import argparse
import sys
import time
from pathlib import Path

import numpyro

numpyro.set_host_device_count(2)

from hill_mixture_mmm.data import compute_prior_config  # noqa: E402
from hill_mixture_mmm.data_loader import (  # noqa: E402
    TimeSeriesConfig,
    load_timeseries,
    select_representative_timeseries,
)
from hill_mixture_mmm.inference import compute_convergence_diagnostics, run_inference  # noqa: E402
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam  # noqa: E402


def run_convergence_test(
    csv_path: str,
    n_orgs: int = 3,
    warmup: int = 500,
    samples: int = 500,
    chains: int = 2,
    K: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Run convergence test on representative organizations.

    Args:
        csv_path: Path to conjura_mmm_data.csv
        n_orgs: Number of organizations to test
        warmup: MCMC warmup iterations
        samples: MCMC sampling iterations
        chains: Number of parallel chains
        K: Number of mixture components
        seed: Random seed for reproducibility

    Returns:
        List of result dicts with convergence diagnostics per org
    """
    print(f"Selecting {n_orgs} representative organizations...")
    org_ids = select_representative_timeseries(
        csv_path,
        n=n_orgs,
        seed=seed,
        min_length=200,
        min_channels=2,
    )
    print(f"Selected: {org_ids}")

    results = []

    for i, org_id in enumerate(org_ids):
        print(f"\n[{i + 1}/{n_orgs}] Testing organization: {org_id[:12]}...")

        try:
            config = TimeSeriesConfig(
                organisation_id=org_id,
                aggregate_spend=True,
            )
            data = load_timeseries(csv_path, config)
            print(f"  Data: T={len(data.y)}, spend_range=[{data.x.min():.1f}, {data.x.max():.1f}]")

            prior_config = compute_prior_config(data.x, data.y)

            start_time = time.time()
            mcmc = run_inference(
                model_fn=model_hill_mixture_hierarchical_reparam,
                x=data.x,
                y=data.y,
                seed=seed + i,
                num_warmup=warmup,
                num_samples=samples,
                num_chains=chains,
                prior_config=prior_config,
                K=K,
            )
            elapsed = time.time() - start_time

            diag = compute_convergence_diagnostics(mcmc)

            result = {
                "org_id": org_id,
                "T": len(data.y),
                "max_rhat": diag["max_rhat"],
                "min_ess": diag["min_ess_bulk"],
                "converged": diag["max_rhat"] <= 1.05 and diag["min_ess_bulk"] >= 100,
                "time_sec": elapsed,
                "error": None,
            }

            status = "✓" if result["converged"] else "✗"
            print(
                f"  Result: R-hat={diag['max_rhat']:.3f}, ESS={diag['min_ess_bulk']:.0f} {status}"
            )

        except Exception as e:
            result = {
                "org_id": org_id,
                "T": None,
                "max_rhat": None,
                "min_ess": None,
                "converged": False,
                "time_sec": None,
                "error": str(e),
            }
            print(f"  ERROR: {e}")

        results.append(result)

    return results


def print_summary(results: list[dict]) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("CONVERGENCE TEST SUMMARY (LogNormal Hyperprior Model)")
    print("=" * 70)
    print(f"{'Org ID':<16} | {'T':>5} | {'R-hat':>6} | {'ESS':>6} | {'Time':>6} | {'Status':>9}")
    print("-" * 70)

    n_converged = 0
    for r in results:
        if r["error"]:
            print(
                f"{r['org_id'][:14]:<16} | {'N/A':>5} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'ERROR':>9}"
            )
        else:
            status = "Converged" if r["converged"] else "FAILED"
            if r["converged"]:
                n_converged += 1
            print(
                f"{r['org_id'][:14]:<16} | {r['T']:>5} | {r['max_rhat']:>6.3f} | {r['min_ess']:>6.0f} | {r['time_sec']:>5.1f}s | {status:>9}"
            )

    print("-" * 70)
    total = len(results)
    pct = 100 * n_converged / total if total > 0 else 0
    print(f"Converged: {n_converged}/{total} ({pct:.0f}%)")
    print("=" * 70)

    if n_converged == total:
        print("\n✓ ALL ORGANIZATIONS CONVERGED - LogNormal hyperprior working as expected")
    elif n_converged > total / 2:
        print(f"\n⚠ PARTIAL CONVERGENCE - {total - n_converged} organizations need investigation")
    else:
        print("\n✗ CONVERGENCE ISSUES PERSIST - Consider additional improvements")


def main():
    parser = argparse.ArgumentParser(description="Test convergence of LogNormal hyperprior model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/conjura_mmm_data.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--n-orgs",
        type=int,
        default=3,
        help="Number of organizations to test",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="MCMC warmup iterations",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="MCMC sampling iterations",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of parallel chains",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Number of mixture components",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print("Testing LogNormal hyperprior convergence")
    print(
        f"Settings: warmup={args.warmup}, samples={args.samples}, chains={args.chains}, K={args.K}"
    )

    results = run_convergence_test(
        csv_path=args.data,
        n_orgs=args.n_orgs,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        K=args.K,
        seed=args.seed,
    )

    print_summary(results)

    n_converged = sum(1 for r in results if r["converged"])
    sys.exit(0 if n_converged == len(results) else 1)


if __name__ == "__main__":
    main()
