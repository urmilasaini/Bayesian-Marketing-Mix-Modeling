"""Build paper-ready figures from real-data benchmark artifacts.

Produces two figures referenced in the real-data section:

  1. ``paper/figures/fig_real_response_panel.png``: 2x3 panel for one
     representative organisation (home_garden, seed 0) across the three
     models — posterior response curves on top, observed vs posterior
     predictive time series below.

  2. ``paper/figures/fig_real_elpd_bar.png``: grouped bar chart of
     mean test ELPD-LOO per (organisation, model) with seed-level
     standard-deviation error bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing paper_figures applies the shared rcParams (fonts, savefig dpi=300)
# and exposes the canonical model palette/labels, so the real-data bar chart
# matches the synthetic Figures 1-3 instead of drifting to its own style.
from hill_mixture_mmm.paper_figures import COLORS, MODEL_LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
REAL_ROOT = FIGURES_ROOT / "real"
RAW_CSV = REPO_ROOT / "results" / "real_benchmark_raw.csv"

# Pick home_garden as the representative organisation — best
# calibrated across all three models and clean K=2 vs K=3 contrast.
PANEL_ORG = "home_garden"
PANEL_SEED = 0
PANEL_MODELS = [
    ("single_hill", "Single Hill"),
    ("mixture_k2", "Mixture (K=2)"),
    ("mixture_k3", "Mixture (K=3)"),
]

# Models ordered for the bar chart.
BAR_MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3"]
BAR_DATASET_ORDER = ["beauty_fitness", "home_garden", "toys_hobbies"]
BAR_DATASET_LABELS = {
    "beauty_fitness": "Beauty & Fitness",
    "home_garden": "Home & Garden",
    "toys_hobbies": "Toys & Hobbies",
}


def _crop_title(img: np.ndarray, frac: float = 0.07) -> np.ndarray:
    """Remove the top ``frac`` of the image (auto-generated title)."""
    h = img.shape[0]
    top = int(h * frac)
    return img[top:, :, :]


def _place_panel(ax, png: Path, *, crop_frac: float, title: str | None) -> None:
    if not png.exists():
        ax.text(0.5, 0.5, f"missing: {png.name}", ha="center", va="center")
        ax.axis("off")
        return
    img = mpimg.imread(png)
    img = _crop_title(img, frac=crop_frac)
    ax.imshow(img)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.axis("off")


def build_response_panel() -> Path:
    out = FIGURES_ROOT / "fig_real_response_panel.png"
    # Two rows: the latent posterior response curves on top (a latent quantity
    # with no directly-observable target), and the observed-vs-posterior-
    # predictive time series below (black observed line) so the reader can see
    # how well each model actually fits the data, not just the latent shape.
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    for col, (model, label) in enumerate(PANEL_MODELS):
        base = REAL_ROOT / model / f"real_{PANEL_ORG}_{model}_seed{PANEL_SEED}"
        # Top row crops the per-model auto title (replaced by our column title);
        # bottom row keeps the embedded test-metric annotation but crops its title.
        _place_panel(axes[0, col], base.with_name(base.name + "_response.png"), crop_frac=0.07, title=label)
        _place_panel(axes[1, col], base.with_name(base.name + "_predictive.png"), crop_frac=0.10, title=None)

    # Row captions.
    axes[0, 0].text(
        -0.04, 0.5, "Latent response", transform=axes[0, 0].transAxes,
        rotation=90, ha="center", va="center", fontsize=12, fontweight="bold",
    )
    axes[1, 0].text(
        -0.04, 0.5, "Observed vs predictive", transform=axes[1, 0].transAxes,
        rotation=90, ha="center", va="center", fontsize=12, fontweight="bold",
    )

    fig.suptitle(
        f"Posterior response curves and predictive fit — {BAR_DATASET_LABELS[PANEL_ORG]} (seed {PANEL_SEED})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_elpd_bar() -> Path:
    out = FIGURES_ROOT / "fig_real_elpd_bar.png"
    df = pd.read_csv(RAW_CSV)
    agg = (
        df.groupby(["dataset_label", "model"])["elpd_loo"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Match the synthetic Figures 1-3 styling (paper_figures.py) exactly:
    # shared palette, white bar edges, "Model"-titled legend, dashed zero
    # line, and the same bar width / annotation placement.
    fig, ax = plt.subplots(figsize=(10, 6))
    n_models = len(BAR_MODEL_ORDER)
    width = min(0.28, 0.8 / max(n_models, 1))
    x = np.arange(len(BAR_DATASET_ORDER))

    for i, model in enumerate(BAR_MODEL_ORDER):
        means = []
        stds = []
        for dataset in BAR_DATASET_ORDER:
            row = agg[(agg["dataset_label"] == dataset) & (agg["model"] == model)]
            means.append(row["mean"].iloc[0] if len(row) else np.nan)
            stds.append(row["std"].iloc[0] if len(row) else np.nan)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BAR_DATASET_LABELS[d] for d in BAR_DATASET_ORDER])
    ax.set_xlabel("Organisation")
    ax.set_ylabel("ELPD-LOO")
    ax.set_title("Real-data predictive density across organisations and models")
    ax.legend(title="Model")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.annotate(
        "Higher is better; error bars: ±1 std across three seeds",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    panel = build_response_panel()
    bar = build_elpd_bar()
    print(f"Wrote {panel.relative_to(REPO_ROOT)}")
    print(f"Wrote {bar.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
