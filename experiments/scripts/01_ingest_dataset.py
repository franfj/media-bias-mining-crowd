#!/usr/bin/env python3
"""
Step 1: Ingest the raw Meneame JSON files into a single Parquet dataset.

Reads all JSON files from the dataset directory, extracts key fields,
and saves as a Parquet file for efficient downstream processing.

Also extracts comments into a separate Parquet file.

Usage:
    python 01_ingest_dataset.py --input-dir /path/to/json/files --output-dir ../data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")

DEFAULT_INPUT_DIR = Path("/Users/franfj/Desktop/AI Factory/meneame/dataset/_test/")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def safe_int(val, default: int = 0) -> int:
    """Safely convert a value to int, returning default on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def parse_unix_ts(ts_str: str) -> datetime | None:
    """Parse a Unix timestamp string to datetime."""
    try:
        ts = int(ts_str)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def process_single_file(filepath: Path) -> tuple[dict | None, list[dict]]:
    """
    Process a single JSON file and return article record + comment records.

    Returns:
        Tuple of (article_dict_or_None, list_of_comment_dicts)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, PermissionError, OSError) as e:
        logger.warning("Failed to read %s: %s", filepath.name, e)
        return None, []

    # Parse timestamps
    post_ts = parse_unix_ts(data.get("post_ts", ""))

    # Extract article record
    article = {
        "article_id": filepath.stem,  # filename without extension
        "url": data.get("post_url", ""),
        "user": data.get("post_user", ""),
        "title": data.get("post_title", ""),
        "text": data.get("post_text", ""),
        "media": data.get("post_media", ""),
        "media_url": data.get("post_media_url", ""),
        "score": safe_int(data.get("post_score", 0)),
        "tags": "|".join(data.get("post_tags", [])),
        "timestamp": post_ts,
        "year": post_ts.year if post_ts else None,
        "month": post_ts.month if post_ts else None,
        "text_length": len(data.get("post_text", "")),
        "num_tags": len(data.get("post_tags", [])),
        "num_comments": len(data.get("post_comments", [])),
    }

    # Extract comment records
    comments = []
    for comment in data.get("post_comments", []):
        comment_ts = parse_unix_ts(comment.get("comment_ts", ""))
        comments.append({
            "article_id": filepath.stem,
            "comment_number": safe_int(comment.get("comment_number", 0)),
            "comment_text": comment.get("comment_text", ""),
            "comment_author": comment.get("comment_author", ""),
            "comment_karma": safe_int(comment.get("comment_karma", 0)),
            "comment_ts": comment_ts,
            "comment_text_length": len(comment.get("comment_text", "")),
        })

    return article, comments


def ingest_dataset(input_dir: Path, output_dir: Path, batch_size: int = 10000) -> None:
    """
    Read all JSON files and write articles + comments as Parquet.

    Uses batched writing to manage memory with 186K files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    total_files = len(json_files)
    logger.info("Found %d JSON files in %s", total_files, input_dir)

    if total_files == 0:
        logger.error("No JSON files found. Exiting.")
        sys.exit(1)

    articles_path = output_dir / "articles_raw.parquet"
    comments_path = output_dir / "comments_raw.parquet"

    article_batches: list[pl.DataFrame] = []
    comment_batches: list[pl.DataFrame] = []
    articles_buffer: list[dict] = []
    comments_buffer: list[dict] = []

    processed = 0
    skipped = 0

    for i, filepath in enumerate(json_files):
        article, comments = process_single_file(filepath)

        if article is not None:
            articles_buffer.append(article)
            comments_buffer.extend(comments)
            processed += 1
        else:
            skipped += 1

        # Flush buffer periodically
        if len(articles_buffer) >= batch_size:
            article_batches.append(pl.DataFrame(articles_buffer))
            if comments_buffer:
                comment_batches.append(pl.DataFrame(comments_buffer))
            articles_buffer = []
            comments_buffer = []
            logger.info(
                "Progress: %d/%d files (%.1f%%), %d processed, %d skipped",
                i + 1, total_files, (i + 1) / total_files * 100,
                processed, skipped,
            )

    # Flush remaining
    if articles_buffer:
        article_batches.append(pl.DataFrame(articles_buffer))
    if comments_buffer:
        comment_batches.append(pl.DataFrame(comments_buffer))

    # Concatenate and write
    logger.info("Concatenating %d article batches...", len(article_batches))
    articles_df = pl.concat(article_batches)

    logger.info("Concatenating %d comment batches...", len(comment_batches))
    comments_df = pl.concat(comment_batches) if comment_batches else pl.DataFrame()

    # Write Parquet
    logger.info("Writing articles to %s...", articles_path)
    articles_df.write_parquet(articles_path)

    logger.info("Writing comments to %s...", comments_path)
    if comments_df.height > 0:
        comments_df.write_parquet(comments_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion complete.")
    logger.info("  Total files: %d", total_files)
    logger.info("  Processed: %d", processed)
    logger.info("  Skipped: %d", skipped)
    logger.info("  Articles: %d rows", articles_df.height)
    logger.info("  Comments: %d rows", comments_df.height)
    logger.info("  Articles file: %s (%.1f MB)", articles_path, articles_path.stat().st_size / 1e6)
    if comments_df.height > 0:
        logger.info("  Comments file: %s (%.1f MB)", comments_path, comments_path.stat().st_size / 1e6)
    logger.info("=" * 60)

    # Print data overview
    year_col = articles_df["year"].drop_nulls()
    print("\n--- Articles Overview ---")
    print(f"Shape: ({articles_df.height}, {articles_df.width})")
    print(f"Year range: {year_col.min()} - {year_col.max()}")
    print(f"Unique media outlets: {articles_df['media'].n_unique()}")
    print(f"Score: mean={articles_df['score'].mean():.1f}, median={articles_df['score'].median():.1f}")
    print(f"Comments: mean={articles_df['num_comments'].mean():.1f}, median={articles_df['num_comments'].median():.1f}")
    print(f"Text length: mean={articles_df['text_length'].mean():.1f}, median={articles_df['text_length'].median():.1f}")
    print(f"\nTop 20 media outlets:")
    top_media = articles_df.group_by("media").len().sort("len", descending=True).head(20)
    for row in top_media.iter_rows():
        print(f"  {row[0]:40s} {row[1]}")
    print(f"\nYear distribution:")
    year_dist = articles_df.group_by("year").len().sort("year")
    for row in year_dist.iter_rows():
        print(f"  {row[0]}  {row[1]}")


def main():
    parser = argparse.ArgumentParser(description="Ingest Meneame JSON files to Parquet")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    ingest_dataset(args.input_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
