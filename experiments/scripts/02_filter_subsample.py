#!/usr/bin/env python3
"""
Step 2: Filter and subsample the dataset.

Filters:
- Articles with >= 5 comments
- Articles with text_length >= 100 chars
- Articles from actual news outlets (not youtube, twitter, facebook, etc.)

Then creates a stratified subsample (~15K articles) balanced by:
- Year
- Media outlet (top outlets + "other")
- Ensures topic diversity via tags

Usage:
    python 02_filter_subsample.py
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
logger = logging.getLogger("filter")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Non-news domains to exclude
NON_NEWS_DOMAINS = [
    "youtube.com", "twitter.com", "facebook.com", "instagram.com",
    "vimeo.com", "reddit.com", "imgur.com", "flickr.com",
    "dailymotion.com", "soundcloud.com", "twitch.tv",
    "wikipedia.org", "es.wikipedia.org", "en.wikipedia.org",
    "github.com", "gitlab.com", "bitbucket.org",
    "amazon.com", "amazon.es", "ebay.es", "ebay.com",
    "change.org", "avaaz.org",
    "docs.google.com", "drive.google.com",
]

# Top Spanish news outlets to track explicitly
TOP_NEWS_OUTLETS = [
    "elpais.com", "elmundo.es", "eldiario.es", "publico.es",
    "20minutos.es", "europapress.es", "elconfidencial.com",
    "cadenaser.com", "abc.es", "lavanguardia.es", "lavanguardia.com",
    "elespanol.com", "elperiodico.com", "lavozdegalicia.es",
    "vozpopuli.com", "elplural.com", "libertaddigital.com",
    "okdiario.com", "rtve.es", "lasexta.com", "antena3.com",
    "cuatro.com", "telecinco.es", "onda0.com", "cope.es",
    "infolibre.es", "lamarea.com", "ctxt.es",
    "huffingtonpost.es", "eleconomista.es", "expansion.com",
    "cincodias.elpais.com", "ara.cat", "rac1.cat",
    "naiz.eus", "deia.eus", "gara.eus",
    "heraldo.es", "diariodesevilla.es", "elcorreo.com",
    "levante-emv.com", "diarioinformacion.com",
    "noticias.lainformacion.com", "lainformacion.com",
]


def filter_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Apply quality filters to the article dataset."""
    initial = df.height
    logger.info("Starting with %d articles", initial)

    # Filter 1: Minimum comments
    df = df.filter(pl.col("num_comments") >= 5)
    logger.info("After >= 5 comments: %d (removed %d)", df.height, initial - df.height)

    # Filter 2: Minimum text length
    prev = df.height
    df = df.filter(pl.col("text_length") >= 100)
    logger.info("After text_length >= 100: %d (removed %d)", df.height, prev - df.height)

    # Filter 3: Remove non-news domains
    prev = df.height
    df = df.filter(~pl.col("media").is_in(NON_NEWS_DOMAINS))
    logger.info("After removing non-news domains: %d (removed %d)", df.height, prev - df.height)

    # Filter 4: Remove articles with no timestamp
    prev = df.height
    df = df.filter(pl.col("year").is_not_null())
    logger.info("After removing no-timestamp: %d (removed %d)", df.height, prev - df.height)

    logger.info(
        "Final filtered dataset: %d articles (%.1f%% of original)",
        df.height, df.height / initial * 100,
    )

    return df


def stratified_subsample(
    df: pl.DataFrame,
    target_size: int = 15000,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Create a stratified subsample balanced by year and media outlet group.

    Strategy:
    - Group media outlets: top outlets individually, rest as "other"
    - Within each year, sample proportionally but ensure minimum representation
    - Target ~15K articles total
    """
    import numpy as np

    rng = np.random.RandomState(random_state)

    # Create outlet group column
    df = df.with_columns(
        pl.when(pl.col("media").is_in(TOP_NEWS_OUTLETS))
        .then(pl.col("media"))
        .otherwise(pl.lit("other"))
        .alias("outlet_group")
    )

    # Calculate per-year allocation (proportional to data availability)
    year_counts = df.group_by("year").len().sort("year")
    total = df.height

    # Allocate samples per year proportionally, but ensure min 200 per year
    year_allocations = {}
    for row in year_counts.iter_rows():
        year, count = row[0], row[1]
        proportion = count / total
        allocated = max(200, int(target_size * proportion))
        year_allocations[year] = min(allocated, count)

    # Adjust to hit target
    total_allocated = sum(year_allocations.values())
    if total_allocated > target_size:
        scale = target_size / total_allocated
        year_allocations = {y: max(200, int(n * scale)) for y, n in year_allocations.items()}

    logger.info("Year allocations: %s", dict(sorted(year_allocations.items())))

    # Sample within each year, stratified by outlet group
    sampled_parts: list[pl.DataFrame] = []
    for year, n_samples in sorted(year_allocations.items()):
        year_df = df.filter(pl.col("year") == year)

        if year_df.height <= n_samples:
            sampled_parts.append(year_df)
            continue

        # Within this year, try to balance outlet groups
        outlet_counts = year_df.group_by("outlet_group").len().sort("len", descending=True)

        # Allocate per outlet proportionally, with minimum 1
        outlet_allocations = {}
        for row in outlet_counts.iter_rows():
            outlet, count = row[0], row[1]
            proportion = count / year_df.height
            allocated = max(1, int(n_samples * proportion))
            outlet_allocations[outlet] = min(allocated, count)

        # Sample per outlet
        year_samples: list[pl.DataFrame] = []
        sampled_indices: set[int] = set()
        for outlet, n in outlet_allocations.items():
            outlet_df = year_df.filter(pl.col("outlet_group") == outlet)
            if outlet_df.height <= n:
                year_samples.append(outlet_df)
            else:
                indices = rng.choice(outlet_df.height, size=n, replace=False)
                year_samples.append(outlet_df[indices.tolist()])

        year_sampled = pl.concat(year_samples)

        # If under target for this year, fill with random remaining
        if year_sampled.height < n_samples:
            # Use row_index to find remaining rows
            remaining_count = n_samples - year_sampled.height
            extra_indices = rng.choice(year_df.height, size=min(remaining_count, year_df.height), replace=False)
            extra = year_df[extra_indices.tolist()]
            year_sampled = pl.concat([year_sampled, extra]).unique()

        sampled_parts.append(year_sampled)

    result = pl.concat(sampled_parts)

    # Drop the helper column
    result = result.drop("outlet_group")

    logger.info("Subsample: %d articles", result.height)
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter and subsample Meneame dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-size", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load raw articles
    articles_path = args.data_dir / "articles_raw.parquet"
    logger.info("Loading articles from %s...", articles_path)
    df = pl.read_parquet(articles_path)
    logger.info("Loaded %d articles", df.height)

    # Filter
    filtered = filter_dataset(df)

    # Save filtered (full)
    filtered_path = args.data_dir / "articles_filtered.parquet"
    filtered.write_parquet(filtered_path)
    logger.info("Saved filtered dataset: %s (%d articles)", filtered_path, filtered.height)

    # Subsample
    subsample = stratified_subsample(filtered, target_size=args.target_size, random_state=args.seed)

    # Save subsample
    subsample_path = args.data_dir / "articles_subsample.parquet"
    subsample.write_parquet(subsample_path)
    logger.info("Saved subsample: %s (%d articles)", subsample_path, subsample.height)

    # Print summary
    print("\n--- Filtered Dataset Summary ---")
    print(f"Total articles: {filtered.height}")
    year_col = filtered["year"].drop_nulls()
    print(f"Year range: {int(year_col.min())} - {int(year_col.max())}")
    print(f"Unique outlets: {filtered['media'].n_unique()}")
    print(f"Score: mean={filtered['score'].mean():.1f}, median={filtered['score'].median():.1f}")
    print(f"Comments: mean={filtered['num_comments'].mean():.1f}, median={filtered['num_comments'].median():.1f}")

    print("\n--- Subsample Summary ---")
    print(f"Total articles: {subsample.height}")
    sub_year = subsample["year"].drop_nulls()
    print(f"Year range: {int(sub_year.min())} - {int(sub_year.max())}")
    print(f"Unique outlets: {subsample['media'].n_unique()}")
    print(f"Score: mean={subsample['score'].mean():.1f}, median={subsample['score'].median():.1f}")
    print(f"Comments: mean={subsample['num_comments'].mean():.1f}, median={subsample['num_comments'].median():.1f}")

    print(f"\nSubsample year distribution:")
    year_dist = subsample.group_by("year").len().sort("year")
    for row in year_dist.iter_rows():
        print(f"  {row[0]}  {row[1]}")

    print(f"\nSubsample top 20 outlets:")
    top_outlets = subsample.group_by("media").len().sort("len", descending=True).head(20)
    for row in top_outlets.iter_rows():
        print(f"  {row[0]:40s} {row[1]}")

    # Tag diversity check
    tags_exploded = (
        subsample.select("tags")
        .filter(pl.col("tags").is_not_null() & (pl.col("tags") != ""))
        .with_columns(pl.col("tags").str.split("|").alias("tag_list"))
        .explode("tag_list")
    )
    print(f"\nUnique tags in subsample: {tags_exploded['tag_list'].n_unique()}")
    print(f"Top 20 tags:")
    top_tags = tags_exploded.group_by("tag_list").len().sort("len", descending=True).head(20)
    for row in top_tags.iter_rows():
        print(f"  {row[0]:40s} {row[1]}")


if __name__ == "__main__":
    main()
