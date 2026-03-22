#!/usr/bin/env python3
"""
Step 4: Extract interaction features for each article.

Computes:
- Comment count, avg comment karma, comment karma variance
- Positive/negative comment karma ratio
- Post score (votes)
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

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_comment_features(comments_df: pl.DataFrame, article_ids: list[str]) -> pl.DataFrame:
    """
    Compute comment-based interaction features grouped by article_id.

    Args:
        comments_df: Raw comments DataFrame
        article_ids: List of article IDs to filter to

    Returns:
        DataFrame with one row per article_id and computed features
    """
    # Filter to only our subsampled articles
    logger.info("Filtering comments to %d articles...", len(article_ids))
    comments = comments_df.filter(pl.col("article_id").is_in(article_ids))
    logger.info("Filtered to %d comments", comments.height)

    # Compute all features using polars group_by aggregations
    logger.info("Computing features per article...")

    features = comments.group_by("article_id").agg([
        # Comment volume
        pl.len().alias("comment_count"),

        # Karma statistics
        pl.col("comment_karma").mean().alias("avg_comment_karma"),
        pl.col("comment_karma").median().alias("median_comment_karma"),
        pl.col("comment_karma").std().alias("std_comment_karma"),
        pl.col("comment_karma").min().alias("min_comment_karma"),
        pl.col("comment_karma").max().alias("max_comment_karma"),
        pl.col("comment_karma").sum().alias("total_comment_karma"),

        # Karma polarization
        (pl.col("comment_karma") < 0).mean().alias("pct_negative_karma"),
        (pl.col("comment_karma") > 0).mean().alias("pct_positive_karma"),
        (pl.col("comment_karma") == 0).mean().alias("pct_zero_karma"),

        # Karma range (spread = polarization indicator)
        (pl.col("comment_karma").max() - pl.col("comment_karma").min()).alias("karma_range"),
        # IQR approximation
        (
            pl.col("comment_karma").quantile(0.75) - pl.col("comment_karma").quantile(0.25)
        ).alias("karma_iqr"),

        # Comment text features
        pl.col("comment_text_length").mean().alias("avg_comment_length"),
        pl.col("comment_text_length").sum().alias("total_comment_text_length"),

        # User engagement
        pl.col("comment_author").n_unique().alias("unique_commenters"),

        # Temporal features: activity duration in hours
        (
            (pl.col("comment_ts").max() - pl.col("comment_ts").min())
            .dt.total_seconds()
            / 3600.0
        ).alias("comment_activity_duration_hours"),
    ])

    # Compute comments_per_commenter
    features = features.with_columns(
        (pl.col("comment_count").cast(pl.Float64) / pl.col("unique_commenters").cast(pl.Float64))
        .alias("comments_per_commenter")
    )

    # Fill nulls with 0
    features = features.fill_null(0)

    return features


def compute_article_features(articles_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute article-level features from the article metadata.
    """
    df = articles_df

    # Compute time-of-day and day-of-week from timestamp
    if "timestamp" in df.columns:
        df = df.with_columns([
            pl.col("timestamp").dt.hour().alias("hour_of_day"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.weekday().is_in([6, 7]).cast(pl.Int32).alias("is_weekend"),
        ])

    # Tag count
    df = df.with_columns(
        pl.when(pl.col("tags").is_null() | (pl.col("tags") == ""))
        .then(pl.lit(0))
        .otherwise(pl.col("tags").str.count_matches(r"\|") + 1)
        .alias("tag_count")
    )

    return df


def merge_features(
    articles_df: pl.DataFrame,
    comment_features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge article-level and comment-level features."""
    merged = articles_df.join(
        comment_features_df,
        on="article_id",
        how="left",
    )

    # Fill NaN for articles with no comments in the comments file
    fill_cols = [c for c in comment_features_df.columns if c != "article_id"]
    merged = merged.with_columns([
        pl.col(col).fill_null(0) for col in fill_cols if col in merged.columns
    ])

    return merged


def main():
    parser = argparse.ArgumentParser(description="Compute interaction features")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Load labeled articles
    articles_path = args.data_dir / "articles_labeled.parquet"
    logger.info("Loading labeled articles from %s...", articles_path)
    articles_df = pl.read_parquet(articles_path)
    logger.info("Loaded %d articles", articles_df.height)

    # Load raw comments
    comments_path = args.data_dir / "comments_raw.parquet"
    logger.info("Loading comments from %s...", comments_path)
    comments_df = pl.read_parquet(comments_path)
    logger.info("Loaded %d comments", comments_df.height)

    # Compute article-level features
    logger.info("Computing article-level features...")
    articles_enhanced = compute_article_features(articles_df)

    # Compute comment-level features
    article_ids = articles_df["article_id"].to_list()
    comment_features = compute_comment_features(comments_df, article_ids)
    logger.info("Computed comment features for %d articles", comment_features.height)

    # Merge
    logger.info("Merging features...")
    final_df = merge_features(articles_enhanced, comment_features)

    # Save
    output_path = args.data_dir / "articles_with_features.parquet"
    final_df.write_parquet(output_path)
    logger.info(
        "Saved features dataset: %s (%d articles, %d columns)",
        output_path, final_df.height, final_df.width,
    )

    # Summary
    print("\n--- Feature Summary ---")
    print(f"Total articles: {final_df.height}")
    print(f"Total columns: {final_df.width}")
    print(f"\nColumn list:")
    for col_name in final_df.columns:
        dtype = final_df[col_name].dtype
        non_null = final_df[col_name].drop_nulls().len()
        print(f"  {col_name:40s} {str(dtype):12s} non-null: {non_null}")

    print(f"\n--- Key Feature Statistics ---")
    numeric_cols = [
        "score", "num_comments", "comment_count", "avg_comment_karma",
        "std_comment_karma", "karma_range", "pct_negative_karma",
        "pct_positive_karma", "unique_commenters", "comments_per_commenter",
        "comment_activity_duration_hours", "avg_comment_length",
        "bias_prob",
    ]
    existing_cols = [c for c in numeric_cols if c in final_df.columns]
    stats = final_df.select(existing_cols).describe()
    print(stats)

    # Bias vs interaction correlation preview
    print(f"\n--- Bias Label vs Interaction Features ---")
    interaction_cols = [
        "score", "num_comments", "avg_comment_karma", "std_comment_karma",
        "karma_range", "pct_negative_karma", "unique_commenters",
        "comment_activity_duration_hours",
    ]
    for col in interaction_cols:
        if col not in final_df.columns:
            continue
        biased_mean = final_df.filter(pl.col("bias_label") == 1)[col].mean()
        non_biased_mean = final_df.filter(pl.col("bias_label") == 0)[col].mean()
        if non_biased_mean is not None and biased_mean is not None:
            denom = max(abs(non_biased_mean), 0.001)
            diff_pct = ((biased_mean - non_biased_mean) / denom) * 100
            print(
                f"  {col:40s} non-biased={non_biased_mean:10.2f}  "
                f"biased={biased_mean:10.2f}  diff={diff_pct:+.1f}%"
            )


if __name__ == "__main__":
    main()
