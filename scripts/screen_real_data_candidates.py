"""Screen Conjura organisations as candidates for real-data benchmark.

Extends ``list_timeseries`` with spend-variation and spend-target
correlation statistics, then ranks organisations by a composite quality
score for Hill-mixture fitting. Writes the full ranking to
``results/real_data_candidates.csv`` and prints the top entries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hill_mixture_mmm.data_loader import (
    SPEND_COLUMNS,
    list_timeseries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "conjura_mmm_data.csv"
OUTPUT_PATH = REPO_ROOT / "results" / "real_data_candidates.csv"

MIN_SERIES_LENGTH = 200
MIN_ACTIVE_CHANNELS = 2


def _safe_cv(x: pd.Series) -> float:
    arr = x.fillna(0).to_numpy()
    m = arr.mean()
    if m <= 0:
        return float("nan")
    return float(arr.std() / m)


def _spend_target_correlation(group: pd.DataFrame) -> float:
    spend_cols = [c for c in SPEND_COLUMNS if c in group.columns]
    total_spend = group[spend_cols].fillna(0).sum(axis=1)
    target = group["all_purchases"].fillna(0)
    if total_spend.std() == 0 or target.std() == 0:
        return float("nan")
    return float(np.corrcoef(total_spend, target)[0, 1])


def enrich_candidates(csv_path: Path) -> pd.DataFrame:
    base = list_timeseries(csv_path, min_length=MIN_SERIES_LENGTH)
    if len(base) == 0:
        return base

    df_full = pd.read_csv(csv_path, parse_dates=["date_day"])

    extra_rows = []
    for _, row in base.iterrows():
        mask = (
            (df_full["organisation_id"] == row["organisation_id"])
            & (df_full["territory_name"] == row["territory_name"])
        )
        group = df_full.loc[mask]
        spend_cols = [c for c in SPEND_COLUMNS if c in group.columns]
        total_spend = group[spend_cols].fillna(0).sum(axis=1)
        target = group["all_purchases"].fillna(0)
        extra_rows.append(
            {
                "organisation_id": row["organisation_id"],
                "territory_name": row["territory_name"],
                "spend_cv": _safe_cv(total_spend),
                "target_cv": _safe_cv(target),
                "spend_target_corr": _spend_target_correlation(group),
                "log_spend_range": float(
                    np.log1p(total_spend.max()) - np.log1p(total_spend.min())
                ),
            }
        )

    extras = pd.DataFrame(extra_rows)
    merged = base.merge(extras, on=["organisation_id", "territory_name"], how="left")
    return merged


def score(df: pd.DataFrame) -> pd.Series:
    """Composite score favouring long series with strong, varied spend signal."""
    s_len = df["n_days"].clip(lower=MIN_SERIES_LENGTH) / df["n_days"].max()
    s_var = df["spend_cv"].clip(lower=0, upper=3) / 3
    s_corr = df["spend_target_corr"].clip(lower=0, upper=1)
    s_chan = df["n_active_channels"].clip(lower=0, upper=5) / 5
    return 0.25 * s_len + 0.30 * s_var + 0.30 * s_corr + 0.15 * s_chan


def main() -> None:
    df = enrich_candidates(CSV_PATH)
    if len(df) == 0:
        print("No candidates found.")
        return

    df = df[df["n_active_channels"] >= MIN_ACTIVE_CHANNELS].copy()
    df["score"] = score(df)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    display_cols = [
        "organisation_id",
        "organisation_vertical",
        "n_days",
        "n_active_channels",
        "spend_cv",
        "spend_target_corr",
        "log_spend_range",
        "total_spend",
        "score",
    ]
    print(f"Top 15 of {len(df)} candidates "
          f"(n_days >= {MIN_SERIES_LENGTH}, active channels >= {MIN_ACTIVE_CHANNELS}):")
    print()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(15)[display_cols].to_string(index=False))
    print()
    print(f"Full ranking written to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print()
    print("Vertical distribution among top 15:")
    print(df.head(15)["organisation_vertical"].value_counts().to_string())


if __name__ == "__main__":
    main()
