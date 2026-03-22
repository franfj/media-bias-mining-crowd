#!/usr/bin/env python3
"""
Step 6: Deep karma divergence analysis.

Goes beyond the basic stats in 05_analysis.py to explore:
- Karma distribution shapes (biased vs non-biased)
- Entropy and Gini of karma distributions per article
- Controversy index: proportion of comments with karma in [-5, -1]
- Karma polarisation: bimodality coefficient
- Effect of article score on karma patterns
- Per-outlet karma divergence (KL divergence)

Usage:
    python 06_karma_divergence.py
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
logger = logging.getLogger("karma_divergence")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_article_karma_features(comments_path: Path, batch_size: int = 500_000) -> pd.DataFrame:
    """Compute advanced karma features per article, processing comments in chunks."""
    logger.info("Computing advanced karma features from comments (chunked)...")

    pf = pq.ParquetFile(comments_path)
    accum: dict[str, list] = {}

    for batch in pf.iter_batches(batch_size=batch_size, columns=["article_id", "comment_karma"]):
        df_batch = batch.to_pandas()
        for aid, grp in df_batch.groupby("article_id"):
            accum.setdefault(aid, []).extend(grp["comment_karma"].tolist())

    logger.info("Aggregated karma for %d articles", len(accum))

    rows = []
    for aid, karmas in accum.items():
        k = np.array(karmas, dtype=np.float64)
        n = len(k)
        if n < 5:
            continue

        mean_k = k.mean()
        std_k = k.std()
        median_k = np.median(k)

        # Controversy: fraction of comments with negative karma
        neg_frac = (k < 0).sum() / n
        # Strong controversy: fraction with karma <= -5
        strong_neg_frac = (k <= -5).sum() / n

        # Entropy of karma distribution (binned)
        k_clipped = np.clip(k, -50, 50).astype(int)
        vals, counts = np.unique(k_clipped, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # Gini coefficient
        sorted_k = np.sort(np.abs(k))
        cumsum = np.cumsum(sorted_k)
        total = cumsum[-1]
        if total > 0:
            gini = 1 - 2 * np.sum(cumsum) / (n * total) + 1 / n
        else:
            gini = 0.0

        # Bimodality coefficient: BC = (skewness^2 + 1) / kurtosis_excess + 3
        # BC > 5/9 suggests bimodality
        if std_k > 0:
            skew = float(sp_stats.skew(k))
            kurt = float(sp_stats.kurtosis(k, fisher=True))
            bc = (skew**2 + 1) / (kurt + 3) if (kurt + 3) != 0 else 0
        else:
            skew, kurt, bc = 0.0, 0.0, 0.0

        # Karma polarisation: ratio of extreme karma (|k| > 2*std) to total
        if std_k > 0:
            extreme_frac = (np.abs(k - mean_k) > 2 * std_k).sum() / n
        else:
            extreme_frac = 0.0

        # IQR-based spread
        q25, q75 = np.percentile(k, [25, 75])

        rows.append({
            "article_id": aid,
            "n_comments": n,
            "karma_mean": mean_k,
            "karma_std": std_k,
            "karma_median": median_k,
            "karma_skew": skew,
            "karma_kurtosis": kurt,
            "karma_entropy": entropy,
            "karma_gini": gini,
            "karma_bimodality": bc,
            "karma_neg_frac": neg_frac,
            "karma_strong_neg_frac": strong_neg_frac,
            "karma_extreme_frac": extreme_frac,
            "karma_q25": q25,
            "karma_q75": q75,
            "karma_iqr": q75 - q25,
        })

    return pd.DataFrame(rows)


def analyse_biased_vs_nonbiased(karma_df: pd.DataFrame, articles_df: pd.DataFrame, output_dir: Path):
    """Compare karma features between biased and non-biased articles."""
    print("\n" + "=" * 80)
    print("KARMA DIVERGENCE: Biased vs Non-Biased")
    print("=" * 80)

    merged = karma_df.merge(articles_df[["article_id", "bias_label", "bias_prob", "media"]],
                            on="article_id", how="inner")
    print(f"Merged: {len(merged)} articles")

    features = [c for c in karma_df.columns if c not in ("article_id", "n_comments")]

    biased = merged[merged["bias_label"] == 1]
    non_biased = merged[merged["bias_label"] == 0]

    results = []
    print(f"\n{'Feature':<30} {'Non-biased':>12} {'Biased':>12} {'Diff%':>8} {'p-value':>12} {'Sig':>4}")
    print("-" * 82)

    for feat in features:
        g1 = non_biased[feat].dropna()
        g2 = biased[feat].dropna()
        if len(g1) < 10 or len(g2) < 10:
            continue

        u_stat, p_val = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        diff_pct = (g2.mean() - g1.mean()) / max(abs(g1.mean()), 0.001) * 100
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        results.append({
            "feature": feat,
            "mean_non_biased": g1.mean(),
            "mean_biased": g2.mean(),
            "diff_pct": diff_pct,
            "p_value": p_val,
        })

        print(f"{feat:<30} {g1.mean():>12.4f} {g2.mean():>12.4f} {diff_pct:>+7.1f}% {p_val:>12.2e} {sig:>4}")

    results_df = pd.DataFrame(results).sort_values("p_value")
    results_df.to_csv(output_dir / "karma_divergence.csv", index=False)
    return merged


def analyse_per_outlet(merged_df: pd.DataFrame, output_dir: Path):
    """KL divergence of karma distributions between outlets."""
    print("\n" + "=" * 80)
    print("PER-OUTLET KARMA DIVERGENCE")
    print("=" * 80)

    # Global karma distribution as reference
    all_karma_means = merged_df["karma_mean"].values
    # Bin into 20 bins
    bins = np.linspace(all_karma_means.min(), all_karma_means.max(), 21)

    global_hist, _ = np.histogram(all_karma_means, bins=bins, density=True)
    global_hist = global_hist + 1e-10  # smoothing

    outlet_counts = merged_df["media"].value_counts()
    top_outlets = outlet_counts[outlet_counts >= 30].index

    rows = []
    for outlet in top_outlets:
        outlet_df = merged_df[merged_df["media"] == outlet]
        outlet_hist, _ = np.histogram(outlet_df["karma_mean"].values, bins=bins, density=True)
        outlet_hist = outlet_hist + 1e-10

        kl_div = sp_stats.entropy(outlet_hist, global_hist)

        rows.append({
            "outlet": outlet,
            "n_articles": len(outlet_df),
            "kl_divergence": kl_div,
            "mean_entropy": outlet_df["karma_entropy"].mean(),
            "mean_bimodality": outlet_df["karma_bimodality"].mean(),
            "mean_neg_frac": outlet_df["karma_neg_frac"].mean(),
            "bias_rate": outlet_df["bias_label"].mean(),
        })

    outlet_df = pd.DataFrame(rows).sort_values("kl_divergence", ascending=False)

    print(f"\n{'Outlet':<35} {'N':>5} {'KL Div':>8} {'Entropy':>8} {'NegFrac':>8} {'Bias%':>7}")
    print("-" * 75)
    for _, row in outlet_df.head(25).iterrows():
        print(f"{row['outlet']:<35} {row['n_articles']:>5} {row['kl_divergence']:>8.4f} {row['mean_entropy']:>8.3f} {row['mean_neg_frac']:>8.3f} {row['bias_rate']*100:>6.1f}%")

    outlet_df.to_csv(output_dir / "outlet_karma_divergence.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Deep karma divergence analysis")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load articles metadata (lightweight)
    articles_path = args.data_dir / "articles_with_features.parquet"
    needed = ["article_id", "bias_label", "bias_prob", "media"]
    all_cols = [f.name for f in pq.read_schema(articles_path)]
    load_cols = [c for c in needed if c in all_cols]
    articles_df = pd.read_parquet(articles_path, columns=load_cols)
    logger.info("Loaded %d articles metadata", len(articles_df))

    # Compute advanced karma features from comments
    comments_path = args.data_dir / "comments_raw.parquet"
    karma_df = compute_article_karma_features(comments_path)

    # Save karma features
    karma_df.to_parquet(args.data_dir / "karma_features.parquet", index=False)
    logger.info("Saved karma features for %d articles", len(karma_df))

    # Analyses
    merged = analyse_biased_vs_nonbiased(karma_df, articles_df, RESULTS_DIR)
    analyse_per_outlet(merged, RESULTS_DIR)

    print("\n" + "=" * 80)
    print("KARMA DIVERGENCE ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
