#!/usr/bin/env python
"""Search controlled synthetic DGP settings that cover a broad true-TV range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hill_mixture_mmm.data import ControlledKSpacingConfig, generate_controlled_k_spacing_data
from hill_mixture_mmm.metrics import compute_component_curve_tv_separation, summarize_true_components


K2_PI_OPTIONS = [
    (0.55, 0.45, 1.0),
    (0.60, 0.40, 1.0),
    (0.65, 0.35, 1.0),
    (0.70, 0.30, 1.0),
]
K2_A_OPTIONS = [
    (42.0, 68.0, 50.0),
    (38.0, 78.0, 50.0),
    (35.0, 85.0, 50.0),
]
K3_PI_OPTIONS = [
    (0.50, 0.30, 0.20),
    (0.55, 0.30, 0.15),
    (0.60, 0.25, 0.15),
]
K3_A_OPTIONS = [
    (40.0, 55.0, 70.0),
    (30.0, 55.0, 85.0),
    (25.0, 55.0, 95.0),
]


def _tv_for_config(config: ControlledKSpacingConfig) -> float:
    _, _, meta = generate_controlled_k_spacing_data(config)
    summary = summarize_true_components(meta)
    return float(compute_component_curve_tv_separation(summary)["mean_pairwise_tv"])


def _candidate_rows(
    *,
    k_true: int,
    T: int,
    seeds: list[int],
    center_k_ratio: float,
    raw_spend_sigmas: list[float],
    spacing_deltas: list[float],
    n_true: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    if k_true == 1:
        config = ControlledKSpacingConfig(K_true=1, T=T, seed=seeds[0], spacing_delta=0.0)
        rows.append(
            {
                "K_true": 1,
                "seed": seeds[0],
                "spacing_delta": 0.0,
                "raw_spend_sigma": float(config.raw_spend_lognormal_sigma),
                "center_k_ratio": float(config.center_k_ratio),
                "pi_true": str(config.resolved_pi_true.tolist()),
                "A_true": str(config.resolved_A_true.tolist()),
                "n_true": str(config.resolved_n_true.tolist()),
                "tv": 0.0,
            }
        )
        return rows

    pi_options = K2_PI_OPTIONS if k_true == 2 else K3_PI_OPTIONS
    A_options = K2_A_OPTIONS if k_true == 2 else K3_A_OPTIONS
    for seed in seeds:
        for raw_spend_sigma in raw_spend_sigmas:
            for spacing_delta in spacing_deltas:
                for pi_true in pi_options:
                    for A_true in A_options:
                        try:
                            config = ControlledKSpacingConfig(
                                K_true=k_true,
                                T=T,
                                seed=seed,
                                spacing_delta=float(spacing_delta),
                                center_k_ratio=float(center_k_ratio),
                                raw_spend_lognormal_sigma=float(raw_spend_sigma),
                                pi_true=pi_true,
                                A_true=A_true,
                                n_true=(float(n_true), float(n_true), float(n_true)),
                            )
                            tv = _tv_for_config(config)
                        except ValueError:
                            continue
                        rows.append(
                            {
                                "K_true": int(k_true),
                                "seed": int(seed),
                                "spacing_delta": float(spacing_delta),
                                "raw_spend_sigma": float(raw_spend_sigma),
                                "center_k_ratio": float(center_k_ratio),
                                "pi_true": str(list(pi_true[:k_true])),
                                "A_true": str(list(A_true[:k_true])),
                                "n_true": str([float(n_true)] * k_true),
                                "tv": float(tv),
                            }
                        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/resolvability_grid_calibration"))
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--k-true", type=int, action="append", choices=[1, 2, 3], dest="k_trues")
    parser.add_argument("--spacing-delta", type=float, action="append", dest="spacing_deltas")
    parser.add_argument("--raw-spend-sigma", type=float, action="append", dest="raw_spend_sigmas")
    parser.add_argument("--center-k-ratio", type=float, default=0.9)
    parser.add_argument("--n-true", type=float, default=2.5)
    args = parser.parse_args()

    seeds = args.seeds or [0]
    k_trues = args.k_trues or [1, 2, 3]
    spacing_deltas = args.spacing_deltas or [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]
    raw_spend_sigmas = args.raw_spend_sigmas or [0.4, 0.8, 1.2, 1.6]

    rows: list[dict[str, float | int | str]] = []
    for k_true in k_trues:
        rows.extend(
            _candidate_rows(
                k_true=k_true,
                T=int(args.T),
                seeds=seeds,
                center_k_ratio=float(args.center_k_ratio),
                raw_spend_sigmas=[float(v) for v in raw_spend_sigmas],
                spacing_deltas=[float(v) for v in spacing_deltas],
                n_true=float(args.n_true),
            )
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "resolvability_grid_candidates.csv"
    json_path = out_dir / "resolvability_grid_summary.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"T": int(args.T), "seeds": seeds, "ranges": {}}
    for k_true in k_trues:
        matching = [float(row["tv"]) for row in rows if int(row["K_true"]) == int(k_true)]
        summary["ranges"][str(k_true)] = {
            "count": len(matching),
            "tv_min": float(np.min(matching)),
            "tv_max": float(np.max(matching)),
            "tv_quantiles": [float(v) for v in np.quantile(matching, [0.1, 0.25, 0.5, 0.75, 0.9])],
        }
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"candidates_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
