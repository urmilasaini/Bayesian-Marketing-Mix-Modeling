#!/usr/bin/env python3
"""Generate exploratory visualizations for the Conjura MMM dataset.

Usage:
    python data/visualize_data.py [--output-dir OUTPUT_DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

SPEND_COLUMNS = [
    "google_paid_search_spend",
    "google_shopping_spend",
    "google_pmax_spend",
    "google_display_spend",
    "google_video_spend",
    "meta_facebook_spend",
    "meta_instagram_spend",
    "meta_other_spend",
    "tiktok_spend",
]

TARGET_COLUMNS = [
    "first_purchases",
    "all_purchases",
    "first_purchases_original_price",
    "all_purchases_original_price",
]

TRAFFIC_COLUMNS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
]


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date_day"] = pd.to_datetime(df["date_day"])
    return df


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_data_overview(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        missing_pct.head(20).plot(kind="barh", ax=axes[0], color="coral")
        axes[0].set_xlabel("Missing %")
        axes[0].set_title("Top 20 Columns with Missing Values")
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes[0].set_title("Missing Values")

    info_text = (
        f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n\n"
        f"Date Range: {df['date_day'].min().date()} to {df['date_day'].max().date()}\n\n"
        f"Unique Timeseries: {df['mmm_timeseries_id'].nunique()}\n"
        f"Unique Organizations: {df['organisation_id'].nunique()}\n"
        f"Unique Territories: {df['territory_name'].nunique()}\n\n"
        f"Verticals: {df['organisation_vertical'].nunique()}\n"
        f"Currencies: {df['currency_code'].nunique()}"
    )
    axes[1].text(0.1, 0.5, info_text, fontsize=12, family="monospace", va="center")
    axes[1].axis("off")
    axes[1].set_title("Dataset Overview")

    _save(fig, output_dir / "01_data_overview.png")


def plot_spend_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    spend_totals = df[SPEND_COLUMNS].sum().sort_values(ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(spend_totals)))
    spend_totals.plot(kind="barh", ax=axes[0], color=colors)
    axes[0].set_xlabel("Total Spend")
    axes[0].set_title("Total Advertising Spend by Channel")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x / 1e6:.1f}M"))

    spend_data = df[SPEND_COLUMNS].replace(0, np.nan).melt(var_name="Channel", value_name="Spend")
    spend_data = spend_data.dropna()
    sns.boxplot(data=spend_data, x="Spend", y="Channel", ax=axes[1], orient="h")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Daily Spend (log scale)")
    axes[1].set_title("Daily Spend Distribution by Channel")

    _save(fig, output_dir / "02_spend_distribution.png")


def plot_time_series_sample(df: pd.DataFrame, output_dir: Path) -> None:
    sample_ts = df["mmm_timeseries_id"].iloc[0]
    sample_df = df[df["mmm_timeseries_id"] == sample_ts].sort_values(by="date_day")  # pyright: ignore[reportCallIssue]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(sample_df["date_day"], sample_df["first_purchases"], label="First Purchases", alpha=0.8)
    axes[0].plot(sample_df["date_day"], sample_df["all_purchases"], label="All Purchases", alpha=0.8)
    axes[0].set_ylabel("Purchases")
    axes[0].set_title(f"Sample Timeseries: {sample_ts[:16]}...")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    total_spend = sample_df[SPEND_COLUMNS].sum(axis=1)
    axes[1].fill_between(sample_df["date_day"], total_spend, alpha=0.7, label="Total Spend")
    axes[1].set_ylabel("Total Daily Spend")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    google_spend = sample_df[[c for c in SPEND_COLUMNS if "google" in c]].sum(axis=1)
    meta_spend = sample_df[[c for c in SPEND_COLUMNS if "meta" in c]].sum(axis=1)
    tiktok_spend = sample_df["tiktok_spend"].fillna(0)
    axes[2].stackplot(
        sample_df["date_day"],
        google_spend, meta_spend, tiktok_spend,
        labels=["Google", "Meta", "TikTok"],
        alpha=0.7,
    )
    axes[2].set_ylabel("Spend by Platform")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    _save(fig, output_dir / "03_time_series_sample.png")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    corr_cols = SPEND_COLUMNS + TARGET_COLUMNS
    rename_map = {
        "google_paid_search_spend": "G_Search",
        "google_shopping_spend": "G_Shopping",
        "google_pmax_spend": "G_PMax",
        "google_display_spend": "G_Display",
        "google_video_spend": "G_Video",
        "meta_facebook_spend": "M_Facebook",
        "meta_instagram_spend": "M_Instagram",
        "meta_other_spend": "M_Other",
        "tiktok_spend": "TikTok",
        "first_purchases": "1st_Purch",
        "all_purchases": "All_Purch",
        "first_purchases_original_price": "1st_Revenue",
        "all_purchases_original_price": "All_Revenue",
    }
    corr_df = df[corr_cols].rename(columns=rename_map)  # type: ignore[arg-type]
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, ax=ax, square=True, linewidths=0.5,
    )
    ax.set_title("Correlation Matrix: Spend vs Target Variables")

    _save(fig, output_dir / "04_correlation_matrix.png")


def plot_vertical_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vertical_counts = (
        df.groupby("organisation_vertical")["mmm_timeseries_id"]
        .nunique()
        .sort_values(ascending=True)  # pyright: ignore[reportCallIssue]
    )
    vertical_counts.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Number of Timeseries")
    axes[0].set_title("Timeseries Count by Vertical")

    df_temp = df.assign(total_spend=df[SPEND_COLUMNS].sum(axis=1))
    avg_spend = (
        df_temp.groupby("organisation_vertical")["total_spend"]
        .mean()
        .sort_values(ascending=True)  # pyright: ignore[reportCallIssue]
    )
    avg_spend.plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_xlabel("Average Daily Spend")
    axes[1].set_title("Average Daily Spend by Vertical")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))

    _save(fig, output_dir / "05_vertical_breakdown.png")


def plot_channel_usage(df: pd.DataFrame, output_dir: Path) -> None:
    channel_usage = df.groupby("mmm_timeseries_id")[SPEND_COLUMNS].apply(lambda x: (x > 0).any())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    usage_rate = channel_usage.mean().sort_values(ascending=True) * 100  # pyright: ignore[reportAttributeAccessIssue]
    colors = [
        "#4285F4" if "google" in c else "#0866FF" if "meta" in c else "#000000"
        for c in usage_rate.index
    ]
    usage_rate.plot(kind="barh", ax=axes[0], color=colors)
    axes[0].set_xlabel("% of Timeseries Using Channel")
    axes[0].set_title("Channel Usage Rate")
    axes[0].axvline(x=50, color="red", linestyle="--", alpha=0.5)

    channels_per_ts = channel_usage.sum(axis=1)
    channels_per_ts.hist(bins=range(0, 11), ax=axes[1], color="teal", edgecolor="white", rwidth=0.8)
    axes[1].set_xlabel("Number of Channels Used")
    axes[1].set_ylabel("Number of Timeseries")
    axes[1].set_title("Distribution of Channel Count per Timeseries")
    axes[1].set_xticks(range(0, 10))

    _save(fig, output_dir / "06_channel_usage.png")


def plot_traffic_sources(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    traffic_totals = df[TRAFFIC_COLUMNS].sum().sort_values(ascending=True)
    traffic_totals.plot(kind="barh", ax=axes[0], color="forestgreen")
    axes[0].set_xlabel("Total Clicks")
    axes[0].set_title("Total Traffic by Source")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1e6:.1f}M"))

    traffic_totals.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", startangle=90)
    axes[1].set_ylabel("")
    axes[1].set_title("Traffic Source Composition")

    _save(fig, output_dir / "07_traffic_sources.png")


def plot_spend_vs_purchases(df: pd.DataFrame, output_dir: Path) -> None:
    agg_df = df.groupby("mmm_timeseries_id").agg(
        {"first_purchases": "sum", "all_purchases": "sum", **{col: "sum" for col in SPEND_COLUMNS}}
    )
    agg_df = pd.DataFrame(agg_df)  # Ensure DataFrame type for pyright
    agg_df["total_spend"] = agg_df[SPEND_COLUMNS].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(agg_df["total_spend"], agg_df["first_purchases"], alpha=0.5, s=30)
    axes[0].set_xlabel("Total Spend")
    axes[0].set_ylabel("Total First Purchases")
    axes[0].set_title("Total Spend vs First Purchases (by Timeseries)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(agg_df["total_spend"], agg_df["all_purchases"], alpha=0.5, s=30, color="orange")
    axes[1].set_xlabel("Total Spend")
    axes[1].set_ylabel("Total All Purchases")
    axes[1].set_title("Total Spend vs All Purchases (by Timeseries)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    _save(fig, output_dir / "08_spend_vs_purchases.png")


def plot_org_spend_vs_response(df: pd.DataFrame, output_dir: Path) -> None:
    """Per-channel spend vs first_purchases scatter for each organisation (3+ active channels).

    Zero-spend days are excluded to highlight saturation (Hill-function) patterns.
    """
    channel_nice = {c.replace("_spend", ""): c for c in SPEND_COLUMNS}
    response = "first_purchases"
    org_dir = output_dir / "conjura_eda"
    org_dir.mkdir(parents=True, exist_ok=True)

    for org_id, org_df in df.groupby("organisation_id"):
        active = [
            (nice, col)
            for nice, col in channel_nice.items()
            if (org_df[col].fillna(0) > 0).sum() > 50
        ]
        if len(active) < 3:
            continue

        vert = str(org_df["organisation_vertical"].iloc[0])
        n_channels = len(active)
        ncols = min(4, n_channels)
        nrows = (n_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"Org {org_id[:12]}…  |  {vert}  |  {n_channels} channels  |  {len(org_df)} days",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for idx, (nice, col) in enumerate(active):
            ax = axes.flat[idx]
            x = org_df[col].fillna(0).values
            y = org_df[response].fillna(0).values

            mask = x > 0
            x, y = x[mask], y[mask]

            order = np.argsort(x)
            ax.scatter(x[order], y[order], s=8, alpha=0.5, edgecolors="none")
            ax.set_title(nice.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Spend")
            ax.set_ylabel(response.replace("_", " ").title())
            ax.tick_params(labelsize=8)

        for idx in range(n_channels, nrows * ncols):
            axes.flat[idx].set_visible(False)

        _save(fig, org_dir / f"org_{org_id[:12]}.png", dpi=120)


def plot_vertical_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    """Total spend (bar) vs first_purchases (line) over time, grouped by vertical."""
    org_dir = output_dir / "conjura_eda"
    org_dir.mkdir(parents=True, exist_ok=True)
    response = "first_purchases"

    for vert in df["organisation_vertical"].dropna().unique():
        vert_df = df[df["organisation_vertical"] == vert]
        vert_orgs = vert_df["organisation_id"].unique()

        fig, axes = plt.subplots(
            len(vert_orgs), 1,
            figsize=(12, 3.5 * len(vert_orgs)),
            squeeze=False,
        )
        fig.suptitle(f"Vertical: {vert}  ({len(vert_orgs)} orgs)", fontsize=14, fontweight="bold")

        for i, org_id in enumerate(vert_orgs):
            ax = axes[i, 0]
            org_df = vert_df[vert_df["organisation_id"] == org_id]
            dates = pd.to_datetime(org_df["date_day"])

            total_spend = org_df[SPEND_COLUMNS].fillna(0).sum(axis=1).values
            resp = org_df[response].fillna(0).values

            ax2 = ax.twinx()
            ax.bar(dates, total_spend, color="steelblue", alpha=0.5, label="Total spend", width=1)
            ax2.plot(dates, resp, color="crimson", linewidth=0.8, label=response)

            ax.set_ylabel("Total Spend", fontsize=9)
            ax2.set_ylabel(response.replace("_", " ").title(), fontsize=9, color="crimson")
            ax.set_title(f"Org {org_id[:12]}…  ({len(org_df)} days)", fontsize=10)
            ax.tick_params(labelsize=8)
            ax2.tick_params(labelsize=8)

        safe_vert = vert.replace(" & ", "_").replace(" ", "_").lower()
        _save(fig, org_dir / f"vertical_{safe_vert}.png", dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Conjura MMM dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/figures"))
    parser.add_argument("--data-path", type=Path, default=Path("data/conjura_mmm_data.csv"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    print(f"Loaded {len(df):,} rows\n")

    plot_data_overview(df, args.output_dir)
    plot_spend_distribution(df, args.output_dir)
    plot_time_series_sample(df, args.output_dir)
    plot_correlation_matrix(df, args.output_dir)
    plot_vertical_breakdown(df, args.output_dir)
    plot_channel_usage(df, args.output_dir)
    plot_traffic_sources(df, args.output_dir)
    plot_spend_vs_purchases(df, args.output_dir)
    plot_org_spend_vs_response(df, args.output_dir)
    plot_vertical_timeseries(df, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
