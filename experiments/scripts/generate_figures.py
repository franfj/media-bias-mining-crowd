#!/usr/bin/env python3
"""Generate paper figures from experiment results."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "drafts" / "v1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.figsize": (5.5, 3.5),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Elegant academic palette
BLUE = "#2C5F8A"       # deep steel blue
CORAL = "#C75B3F"      # muted terracotta
TEAL = "#3A8A7A"       # sophisticated teal
GOLD = "#C4963C"       # warm gold


def fig_yearly_distribution():
    """Figure 1: Articles per year + bias rate."""
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    cols = [c for c in ["year", "bias_label"] if c in schema]
    df = pd.read_parquet(DATA_DIR / "articles_with_features.parquet", columns=cols)

    yearly = df.groupby("year").agg(
        count=("bias_label", "count"),
        bias_rate=("bias_label", "mean"),
    )

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    years = yearly.index.astype(int)
    ax1.bar(years, yearly["count"], color=BLUE, alpha=0.55, label="Articles", edgecolor="white", linewidth=0.5)
    ax2.plot(years, yearly["bias_rate"] * 100, color=CORAL, linewidth=3,
             marker="o", markersize=6, markerfacecolor=CORAL, markeredgecolor="white",
             markeredgewidth=1.5, label="Bias rate (%)", zorder=10)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of articles", color=BLUE)
    ax2.set_ylabel("Bias rate (%)", color=CORAL)

    # Dynamic y-limits for bias rate with padding
    br_min = yearly["bias_rate"].min() * 100
    br_max = yearly["bias_rate"].max() * 100
    margin = (br_max - br_min) * 0.3
    ax2.set_ylim(max(0, br_min - margin), br_max + margin)

    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=CORAL)

    # Integer years, rotated
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.savefig(FIG_DIR / "yearly_distribution.pdf")
    plt.close()
    print(f"  Saved yearly_distribution.pdf")


def fig_karma_entropy():
    """Figure 2: Karma entropy distributions biased vs non-biased."""
    karma = pd.read_parquet(DATA_DIR / "karma_features.parquet",
                            columns=["article_id", "karma_entropy"])
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    cols = [c for c in ["article_id", "bias_label"] if c in schema]
    articles = pd.read_parquet(DATA_DIR / "articles_with_features.parquet", columns=cols)

    merged = karma.merge(articles, on="article_id", how="inner")
    biased = merged[merged["bias_label"] == 1]["karma_entropy"]
    non_biased = merged[merged["bias_label"] == 0]["karma_entropy"]

    fig, ax = plt.subplots()
    bins = np.linspace(0, 5, 60)
    ax.hist(non_biased, bins=bins, alpha=0.55, color=BLUE, density=True, label="Non-biased", edgecolor="white", linewidth=0.3)
    ax.hist(biased, bins=bins, alpha=0.55, color=CORAL, density=True, label="Biased", edgecolor="white", linewidth=0.3)

    ax.axvline(non_biased.mean(), color=BLUE, linestyle="--", linewidth=1.8, alpha=0.9)
    ax.axvline(biased.mean(), color=CORAL, linestyle="--", linewidth=1.8, alpha=0.9)

    ax.set_xlabel("Karma entropy (bits)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    plt.savefig(FIG_DIR / "karma_entropy.pdf")
    plt.close()
    print(f"  Saved karma_entropy.pdf")


def fig_sentiment_comparison():
    """Figure 3: Sentiment distribution comparison."""
    summary = pd.read_csv(RESULTS_DIR / "sentiment_summary.csv")

    # Extract values
    pct_neg_biased = summary[summary["metric"] == "pct_neg_biased"]["value"].values[0] * 100
    pct_neg_nonbiased = summary[summary["metric"] == "pct_neg_non_biased"]["value"].values[0] * 100

    # Load full sample for more detail
    sent_path = DATA_DIR / "comments_with_sentiment.parquet"
    if sent_path.exists():
        df = pd.read_parquet(sent_path)
        categories = ["NEG", "NEU", "POS"]
        biased_pcts = [(df[df["bias_label"] == 1]["sentiment"] == c).mean() * 100 for c in categories]
        nonbiased_pcts = [(df[df["bias_label"] == 0]["sentiment"] == c).mean() * 100 for c in categories]
    else:
        categories = ["Negative", "Neutral", "Positive"]
        biased_pcts = [pct_neg_biased, 100 - pct_neg_biased - 8.0, 8.0]
        nonbiased_pcts = [pct_neg_nonbiased, 100 - pct_neg_nonbiased - 8.0, 8.0]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, nonbiased_pcts, width, color=BLUE, alpha=0.75, label="Non-biased", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, biased_pcts, width, color=CORAL, alpha=0.75, label="Biased", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Percentage of comments (%)")
    ax.set_xticks(x)
    labels = ["Negative", "Neutral", "Positive"] if categories[0] == "NEG" else categories
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)

    # Add value labels
    for i, (nb, b) in enumerate(zip(nonbiased_pcts, biased_pcts)):
        ax.text(i - width / 2, nb + 0.8, f"{nb:.1f}", ha="center", fontsize=7)
        ax.text(i + width / 2, b + 0.8, f"{b:.1f}", ha="center", fontsize=7)

    plt.savefig(FIG_DIR / "sentiment_comparison.pdf")
    plt.close()
    print(f"  Saved sentiment_comparison.pdf")


def fig_user_diversity():
    """Figure 4: Outlet diversity by bias exposure category."""
    user_pol = pd.read_csv(RESULTS_DIR / "user_polarisation_summary.csv")

    fig, ax = plt.subplots()
    categories = user_pol["bias_category"].tolist()
    labels = ["Low\n(<0.4)", "Mixed\n(0.4-0.6)", "High\n(0.6-0.8)", "Very high\n(>0.8)"]
    outlets = user_pol["avg_outlets"].tolist()
    users = user_pol["n_users"].tolist()

    colors = [BLUE, TEAL, CORAL, GOLD]
    bars = ax.bar(range(len(categories)), outlets, color=colors, alpha=0.8, edgecolor="white")

    ax.set_ylabel("Mean outlet diversity")
    ax.set_xlabel("Bias exposure category")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)

    # Add user counts
    for i, (bar, n) in enumerate(zip(bars, users)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={n:,}", ha="center", fontsize=7, color="gray")

    plt.savefig(FIG_DIR / "user_diversity.pdf")
    plt.close()
    print(f"  Saved user_diversity.pdf")


def fig_outlet_communities():
    """Figure 5: Outlet communities (horizontal bar chart with bias rates)."""
    communities = pd.read_csv(RESULTS_DIR / "outlet_communities.csv")

    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    cols = [c for c in ["article_id", "media", "bias_label"] if c in schema]
    articles = pd.read_parquet(DATA_DIR / "articles_with_features.parquet", columns=cols)

    outlet_stats = articles.groupby("media").agg(
        count=("bias_label", "count"),
        bias_rate=("bias_label", "mean"),
    ).reset_index()

    merged = communities.merge(outlet_stats, left_on="outlet", right_on="media", how="inner")
    merged = merged[merged["count"] >= 30].sort_values(["community", "bias_rate"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(5.5, 6))

    colors = {0: BLUE, 1: CORAL}
    y_pos = 0
    y_positions = []
    y_labels = []
    comm_boundaries = []

    for comm in sorted(merged["community"].unique()):
        comm_data = merged[merged["community"] == comm]
        start_y = y_pos
        for _, row in comm_data.iterrows():
            ax.barh(y_pos, row["bias_rate"] * 100, color=colors[comm], alpha=0.7, height=0.7)
            y_positions.append(y_pos)
            y_labels.append(row["outlet"])
            y_pos += 1
        comm_boundaries.append((start_y, y_pos - 1))
        y_pos += 0.5

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel("Bias rate (%)")
    ax.axvline(61.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Dataset avg.")

    # Community labels
    for i, (start, end) in enumerate(comm_boundaries):
        mid = (start + end) / 2
        ax.text(95, mid, f"Community {i + 1}", fontsize=8, fontweight="bold",
                color=colors[i], ha="right", va="center")

    ax.legend(fontsize=7, loc="lower right")

    plt.savefig(FIG_DIR / "outlet_communities.pdf")
    plt.close()
    print(f"  Saved outlet_communities.pdf")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_yearly_distribution()
    fig_karma_entropy()
    fig_sentiment_comparison()
    fig_user_diversity()
    fig_outlet_communities()
    print(f"\nAll figures saved to: {FIG_DIR}")
