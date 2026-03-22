#!/usr/bin/env python3
"""
Step 4: Extract interaction features for each article.

Computes:
- Comment count, avg comment karma, comment karma variance
- Positive/negative comment karma ratio
- Post score (votes)
- Time to first comment (seconds)
- Comment activity duration (last comment - first comment)
- Comment karma std (polarization signal)
- Average comment length
- Unique commenters count
- Comments per unique commenter (engagement depth)

Uses the labeled articles + raw comments parquet files.

Usage:
    python 04_interaction_features.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_comment_features(comments_df: pd.DataFrame, article_ids: set) -> pd.DataFrame:
    """
    Compute comment-based interaction features grouped by article_id.

    Args:
        comments_df: Raw comments DataFrame
        article_ids: Set of article IDs to filter to

    Returns:
        DataFrame with one row per article_id and computed features
    """
    # Filter to only our subsampled articles
    logger.info("Filtering comments to %d articles...", len(article_ids))
    comments = comments_df[comments_df["article_id"].isin(article_ids)].copy()
    logger.info("Filtered to %d comments", len(comments))

    # Group by article
    logger.info("Computing features per article...")

    features = []

    grouped = comments.groupby("article_id")
    total_groups = len(grouped)

    for i, (article_id, group) in enumerate(grouped):
        if (i + 1) % 2000 == 0:
            logger.info("Processing article %d/%d...", i + 1, total_groups)

        karma_values = group["comment_karma"].values
        timestamps = group["comment_ts"].dropna()
        text_lengths = group["comment_text_length"].values
        authors = group["comment_author"]

        feat = {
            "article_id": article_id,
            # Comment volume
            "comment_count": len(group),
            # Karma statistics
            "avg_comment_karma": float(np.mean(karma_values)) if len(karma_values) > 0 else 0.0,
            "median_comment_karma": float(np.median(karma_values)) if len(karma_values) > 0 else 0.0,
            "std_comment_karma": float(np.std(karma_values, ddof=1)) if len(karma_values) > 1 else 0.0,
            "min_comment_karma": float(np.min(karma_values)) if len(karma_values) > 0 else 0.0,
            "max_comment_karma": float(np.max(karma_values)) if len(karma_values) > 0 else 0.0,
            "total_comment_karma": float(np.sum(karma_values)),
            # Karma polarization: ratio of negative karma comments
            "pct_negative_karma": float(np.mean(karma_values < 0)) if len(karma_values) > 0 else 0.0,
            "pct_positive_karma": float(np.mean(karma_values > 0)) if len(karma_values) > 0 else 0.0,
            "pct_zero_karma": float(np.mean(karma_values == 0)) if len(karma_values) > 0 else 0.0,
            # Karma range (spread = polarization indicator)
            "karma_range": float(np.max(karma_values) - np.min(karma_values)) if len(karma_values) > 0 else 0.0,
            "karma_iqr": float(np.percentile(karma_values, 75) - np.percentile(karma_values, 25)) if len(karma_values) >= 4 else 0.0,
            # Comment text features
            "avg_comment_length": float(np.mean(text_lengths)) if len(text_lengths) > 0 else 0.0,
            "total_comment_text_length": float(np.sum(text_lengths)),
            # User engagement
            "unique_commenters": int(authors.nunique()),
            "comments_per_commenter": float(len(group) / max(authors.nunique(), 1)),
        }

        # Temporal features (if timestamps available)
        if len(timestamps) >= 2:
            ts_sorted = timestamps.sort_values()
            first_comment = ts_sorted.iloc[0]
            last_comment = ts_sorted.iloc[-1]

            # Activity duration in hours
            duration = (last_comment - first_comment).total_seconds() / 3600.0
            feat["comment_activity_duration_hours"] = duration
        else:
            feat["comment_activity_duration_hours"] = 0.0

        features.append(feat)

    return pd.DataFrame(features)


def compute_article_features(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute article-level features from the article metadata.
    """
    df = articles_df.copy()

    # Post score features (already in data)
    # Compute time-of-day and day-of-week from timestamp
    if "timestamp" in df.columns:
        df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Tag count
    df["tag_count"] = df["tags"].fillna("").str.count(r"\|") + 1
    df.loc[df["tags"].fillna("") == "", "tag_count"] = 0

    return df


def merge_features(
    articles_df: pd.DataFrame,
    comment_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge article-level and comment-level features."""
    merged = articles_df.merge(
        comment_features_df,
        on="article_id",
        how="left",
        suffixes=("", "_computed"),
    )

    # Fill NaN for articles with no comments in the comments file
    fill_cols = [c for c in comment_features_df.columns if c != "article_id"]
    for col in fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Compute interaction features")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Load labeled articles
    articles_path = args.data_dir / "articles_labeled.parquet"
    logger.info("Loading labeled articles from %s...", articles_path)
    articles_df = pd.read_parquet(articles_path)
    logger.info("Loaded %d articles", len(articles_df))

    # Load raw comments
    comments_path = args.data_dir / "comments_raw.parquet"
    logger.info("Loading comments from %s...", comments_path)
    comments_df = pd.read_parquet(comments_path)
    logger.info("Loaded %d comments", len(comments_df))

    # Compute article-level features
    logger.info("Computing article-level features...")
    articles_enhanced = compute_article_features(articles_df)

    # Compute comment-level features
    article_ids = set(articles_df["article_id"].values)
    comment_features = compute_comment_features(comments_df, article_ids)
    logger.info("Computed comment features for %d articles", len(comment_features))

    # Merge
    logger.info("Merging features...")
    final_df = merge_features(articles_enhanced, comment_features)

    # Save
    output_path = args.data_dir / "articles_with_features.parquet"
    final_df.to_parquet(output_path, index=False)
    logger.info("Saved features dataset: %s (%d articles, %d columns)",
                output_path, len(final_df), len(final_df.columns))

    # Summary
    print("\n--- Feature Summary ---")
    print(f"Total articles: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")
    print(f"\nColumn list:")
    for col in final_df.columns:
        dtype = final_df[col].dtype
        non_null = final_df[col].notna().sum()
        print(f"  {col:40s} {str(dtype):12s} non-null: {non_null}")

    print(f"\n--- Key Feature Statistics ---")
    numeric_cols = [
        "score", "num_comments", "comment_count", "avg_comment_karma",
        "std_comment_karma", "karma_range", "pct_negative_karma",
        "pct_positive_karma", "unique_commenters", "comments_per_commenter",
        "comment_activity_duration_hours", "avg_comment_length",
        "bias_prob",
    ]
    existing_cols = [c for c in numeric_cols if c in final_df.columns]
    print(final_df[existing_cols].describe().to_string())

    # Bias vs interaction correlation preview
    print(f"\n--- Bias Label vs Interaction Features ---")
    for col in ["score", "num_comments", "avg_comment_karma", "std_comment_karma",
                 "karma_range", "pct_negative_karma", "unique_commenters",
                 "comment_activity_duration_hours"]:
        if col in final_df.columns:
            biased_mean = final_df[final_df["bias_label"] == 1][col].mean()
            non_biased_mean = final_df[final_df["bias_label"] == 0][col].mean()
            diff_pct = ((biased_mean - non_biased_mean) / max(abs(non_biased_mean), 0.001)) * 100
            print(f"  {col:40s} non-biased={non_biased_mean:10.2f}  biased={biased_mean:10.2f}  diff={diff_pct:+.1f}%")


if __name__ == "__main__":
    main()
