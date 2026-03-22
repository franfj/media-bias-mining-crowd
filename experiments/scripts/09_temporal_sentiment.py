#!/usr/bin/env python3
"""
Step 9: Temporal analysis — sentiment and interaction evolution over time.

Combines temporal patterns with sentiment/karma data to explore:
- How comment sentiment evolves within an article's lifetime
- Whether biased articles trigger faster/more negative initial reactions
- Yearly trends in comment toxicity and polarisation
- Event-driven spikes in bias and sentiment

Requires: output from step 08 (comments_with_sentiment.parquet)

Usage:
    python 09_temporal_sentiment.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("temporal_sentiment")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def analyse_intra_article_dynamics(comments_path: Path, articles_path: Path, output_dir: Path):
    """Analyse how karma and comment length evolve within an article's comment thread."""
    print("\n" + "=" * 80)
    print("INTRA-ARTICLE DYNAMICS: Early vs Late Comments")
    print("=" * 80)

    # Load articles metadata
    schema = [f.name for f in pq.read_schema(articles_path)]
    cols = [c for c in ["article_id", "bias_label"] if c in schema]
    articles = pd.read_parquet(articles_path, columns=cols)
    bias_map = dict(zip(articles["article_id"], articles["bias_label"]))

    # Process comments in chunks — compute per-article early vs late stats
    pf = pq.ParquetFile(comments_path)
    article_comments: dict[str, list] = {}

    logger.info("Loading comment timestamps and karma...")
    for batch in pf.iter_batches(batch_size=500_000,
                                  columns=["article_id", "comment_number", "comment_karma",
                                           "comment_text_length", "comment_ts"]):
        df = batch.to_pandas()
        for aid, grp in df.groupby("article_id"):
            if aid not in bias_map:
                continue
            existing = article_comments.get(aid, [])
            existing.extend(grp[["comment_number", "comment_karma", "comment_text_length", "comment_ts"]].to_dict("records"))
            article_comments[aid] = existing

    logger.info("Processing %d articles...", len(article_comments))

    rows = []
    for aid, comments in article_comments.items():
        if len(comments) < 10:
            continue

        cdf = pd.DataFrame(comments).sort_values("comment_number")
        n = len(cdf)
        first_quarter = cdf.head(n // 4)
        last_quarter = cdf.tail(n // 4)

        # Time dynamics
        ts = pd.to_datetime(cdf["comment_ts"])
        if ts.isna().all():
            duration_hours = 0
        else:
            duration_hours = (ts.max() - ts.min()).total_seconds() / 3600

        # Early reactions (first 25%)
        early_karma = first_quarter["comment_karma"].mean()
        early_len = first_quarter["comment_text_length"].mean()

        # Late reactions (last 25%)
        late_karma = last_quarter["comment_karma"].mean()
        late_len = last_quarter["comment_text_length"].mean()

        # Karma trend (slope of linear fit on comment_number vs karma)
        if cdf["comment_karma"].std() > 0:
            slope, _, _, _, _ = sp_stats.linregress(cdf["comment_number"], cdf["comment_karma"])
        else:
            slope = 0.0

        # Speed of first N comments (time between first and 10th comment)
        if len(ts.dropna()) >= 10:
            sorted_ts = ts.sort_values().dropna()
            first_10_hours = (sorted_ts.iloc[9] - sorted_ts.iloc[0]).total_seconds() / 3600
        else:
            first_10_hours = np.nan

        rows.append({
            "article_id": aid,
            "bias_label": bias_map[aid],
            "n_comments": n,
            "duration_hours": duration_hours,
            "early_karma_mean": early_karma,
            "late_karma_mean": late_karma,
            "karma_drift": late_karma - early_karma,
            "early_comment_len": early_len,
            "late_comment_len": late_len,
            "karma_slope": slope,
            "first_10_hours": first_10_hours,
        })

    dynamics_df = pd.DataFrame(rows)

    # Compare biased vs non-biased
    biased = dynamics_df[dynamics_df["bias_label"] == 1]
    non_biased = dynamics_df[dynamics_df["bias_label"] == 0]

    features = ["early_karma_mean", "late_karma_mean", "karma_drift", "karma_slope",
                "early_comment_len", "late_comment_len", "first_10_hours", "duration_hours"]

    print(f"\nArticles with >= 10 comments: {len(dynamics_df)} (biased: {len(biased)}, non-biased: {len(non_biased)})")
    print(f"\n{'Feature':<25} {'Non-biased':>12} {'Biased':>12} {'Diff%':>8} {'p-value':>12} {'Sig':>4}")
    print("-" * 77)

    for feat in features:
        g1 = non_biased[feat].dropna()
        g2 = biased[feat].dropna()
        if len(g1) < 10 or len(g2) < 10:
            continue
        _, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        diff = (g2.mean() - g1.mean()) / max(abs(g1.mean()), 0.001) * 100
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<25} {g1.mean():>12.3f} {g2.mean():>12.3f} {diff:>+7.1f}% {p:>12.2e} {sig}")

    dynamics_df.to_csv(output_dir / "intra_article_dynamics.csv", index=False)
    return dynamics_df


def analyse_yearly_sentiment_trends(comments_path: Path, articles_path: Path, output_dir: Path):
    """Analyse yearly trends in comment karma and engagement patterns."""
    print("\n" + "=" * 80)
    print("YEARLY TRENDS: Karma and Engagement Evolution")
    print("=" * 80)

    # Load articles with year info
    schema = [f.name for f in pq.read_schema(articles_path)]
    cols = [c for c in ["article_id", "bias_label", "year"] if c in schema]
    articles = pd.read_parquet(articles_path, columns=cols)
    articles_dedup = articles.drop_duplicates(subset="article_id")
    article_info = articles_dedup.set_index("article_id")[["year", "bias_label"]].to_dict("index")

    # Aggregate karma stats by year × bias_label in chunks
    yearly_data: dict[tuple, list] = {}  # (year, bias_label) → list of karma values

    pf = pq.ParquetFile(comments_path)
    for batch in pf.iter_batches(batch_size=500_000,
                                  columns=["article_id", "comment_karma"]):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            info = article_info.get(row["article_id"])
            if info is None:
                continue
            key = (info["year"], info["bias_label"])
            yearly_data.setdefault(key, []).append(row["comment_karma"])

    # Compute stats
    rows = []
    for (year, bias_label), karmas in yearly_data.items():
        k = np.array(karmas)
        rows.append({
            "year": int(year),
            "bias_label": int(bias_label),
            "n_comments": len(k),
            "mean_karma": k.mean(),
            "median_karma": np.median(k),
            "std_karma": k.std(),
            "pct_negative": (k < 0).mean(),
            "pct_very_negative": (k <= -5).mean(),
            "mean_abs_karma": np.abs(k).mean(),
        })

    yearly_df = pd.DataFrame(rows).sort_values(["year", "bias_label"])

    # Print comparison by year
    print(f"\n{'Year':>6} {'Type':<12} {'Comments':>10} {'MeanKarma':>10} {'StdKarma':>9} {'%Neg':>6} {'%VeryNeg':>8}")
    print("-" * 70)
    for year in sorted(yearly_df["year"].unique()):
        for bl, name in [(0, "Non-biased"), (1, "Biased")]:
            row = yearly_df[(yearly_df["year"] == year) & (yearly_df["bias_label"] == bl)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"{int(r['year']):>6} {name:<12} {r['n_comments']:>10,} {r['mean_karma']:>+10.2f} {r['std_karma']:>9.2f} {r['pct_negative']*100:>5.1f}% {r['pct_very_negative']*100:>7.2f}%")

    yearly_df.to_csv(output_dir / "yearly_karma_trends.csv", index=False)
    return yearly_df


def analyse_sentiment_temporal(sentiment_path: Path, output_dir: Path):
    """If sentiment data from step 08 is available, analyse temporal patterns."""
    if not sentiment_path.exists():
        print("\n[SKIP] Sentiment data not found — run 08_comment_sentiment.py first")
        return

    print("\n" + "=" * 80)
    print("TEMPORAL SENTIMENT PATTERNS (from step 08 sample)")
    print("=" * 80)

    df = pd.read_parquet(sentiment_path)

    # We need to join with article timestamps
    # For now, use the article_id to get year
    articles_path = sentiment_path.parent / "articles_with_features.parquet"
    schema = [f.name for f in pq.read_schema(articles_path)]
    cols = [c for c in ["article_id", "year"] if c in schema]
    articles = pd.read_parquet(articles_path, columns=cols)

    df = df.merge(articles[["article_id", "year"]], on="article_id", how="left")

    # Yearly sentiment by bias label
    yearly = df.groupby(["year", "bias_label"]).agg(
        n_comments=("polarity_score", "count"),
        mean_polarity=("polarity_score", "mean"),
        pct_negative_sent=("sentiment", lambda x: (x == "NEG").mean()),
        pct_positive_sent=("sentiment", lambda x: (x == "POS").mean()),
        mean_anger=("emo_anger", "mean"),
        mean_joy=("emo_joy", "mean"),
    ).reset_index()

    print(f"\n{'Year':>6} {'Type':<12} {'N':>7} {'Polarity':>9} {'%NEG':>6} {'%POS':>6} {'Anger':>7} {'Joy':>7}")
    print("-" * 65)
    for year in sorted(yearly["year"].dropna().unique()):
        for bl, name in [(0, "Non-bias"), (1, "Biased")]:
            row = yearly[(yearly["year"] == year) & (yearly["bias_label"] == bl)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"{int(r['year']):>6} {name:<12} {r['n_comments']:>7} {r['mean_polarity']:>+8.4f} {r['pct_negative_sent']*100:>5.1f}% {r['pct_positive_sent']*100:>5.1f}% {r['mean_anger']:>7.4f} {r['mean_joy']:>7.4f}")

    yearly.to_csv(output_dir / "yearly_sentiment_trends.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Temporal sentiment analysis")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comments_path = args.data_dir / "comments_raw.parquet"
    articles_path = args.data_dir / "articles_with_features.parquet"
    sentiment_path = args.data_dir / "comments_with_sentiment.parquet"

    # 1. Intra-article dynamics (no ML needed, just karma/time)
    analyse_intra_article_dynamics(comments_path, articles_path, RESULTS_DIR)

    # 2. Yearly karma trends
    analyse_yearly_sentiment_trends(comments_path, articles_path, RESULTS_DIR)

    # 3. Temporal sentiment (if step 08 was run)
    analyse_sentiment_temporal(sentiment_path, RESULTS_DIR)

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
