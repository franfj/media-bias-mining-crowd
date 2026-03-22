#!/usr/bin/env python3
"""
Step 3: Automatic bias labeling using the supervised DistilBERT model.

Uses the franfj/fdtd_media_bias_E model (trained on MBBMD) to label
articles as biased/non-biased based on their title + text.

Labels:
  - 0 = non-biased
  - 1 = biased

Also computes the bias probability score (continuous 0-1).

Usage:
    python 03_bias_labeling.py
    python 03_bias_labeling.py --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bias_label")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "franfj/fdtd_media_bias_E"


def load_model(device: str = "cpu"):
    """Load the bias detection model and tokenizer."""
    logger.info("Loading model %s on %s...", MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 512,
) -> tuple[list[int], list[float]]:
    """
    Predict bias labels and probabilities for a batch of texts.

    Returns:
        Tuple of (labels, bias_probabilities)
        - labels: 0 (non-biased) or 1 (biased)
        - bias_probabilities: P(biased) in [0, 1]
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

    # LABEL_1 = biased
    bias_probs = probs[:, 1].cpu().numpy().tolist()
    labels = (probs[:, 1] > 0.5).cpu().numpy().astype(int).tolist()

    return labels, bias_probs


def label_articles(
    df: pl.DataFrame,
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 32,
) -> pl.DataFrame:
    """
    Add bias labels and probabilities to the article DataFrame.

    Uses title + first 400 chars of text as input.
    """
    # Prepare input texts: title + text
    titles = df["title"].to_list()
    texts_col = df["text"].to_list()

    texts = []
    for title, text in zip(titles, texts_col):
        title = str(title) if title is not None else ""
        text = str(text) if text is not None else ""
        combined = f"{title}. {text[:400]}" if text else title
        texts.append(combined)

    logger.info("Labeling %d articles in batches of %d...", len(texts), batch_size)

    all_labels: list[int] = []
    all_probs: list[float] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        labels, probs = predict_batch(batch_texts, tokenizer, model, device)
        all_labels.extend(labels)
        all_probs.extend(probs)

        if (i // batch_size) % 50 == 0:
            logger.info(
                "Progress: %d/%d (%.1f%%)",
                min(i + batch_size, len(texts)),
                len(texts),
                min(i + batch_size, len(texts)) / len(texts) * 100,
            )

    df = df.with_columns([
        pl.Series("bias_label", all_labels, dtype=pl.Int32),
        pl.Series("bias_prob", all_probs, dtype=pl.Float64),
    ])

    return df


def main():
    parser = argparse.ArgumentParser(description="Label articles for bias")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, or mps")
    parser.add_argument("--input-file", type=str, default="articles_subsample.parquet")
    parser.add_argument("--output-file", type=str, default="articles_labeled.parquet")
    args = parser.parse_args()

    # Detect MPS (Apple Silicon) if available
    device = args.device
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple Silicon MPS detected, using MPS backend")

    # Load data
    input_path = args.data_dir / args.input_file
    logger.info("Loading articles from %s...", input_path)
    df = pl.read_parquet(input_path)
    logger.info("Loaded %d articles", df.height)

    # Load model
    tokenizer, model = load_model(device)

    # Label
    labeled_df = label_articles(df, tokenizer, model, device, args.batch_size)

    # Save
    output_path = args.data_dir / args.output_file
    labeled_df.write_parquet(output_path)
    logger.info("Saved labeled dataset: %s", output_path)

    # Summary
    biased = labeled_df.filter(pl.col("bias_label") == 1).height
    non_biased = labeled_df.height - biased

    print("\n--- Bias Labeling Summary ---")
    print(f"Total articles: {labeled_df.height}")
    print(f"Non-biased (0): {non_biased} ({non_biased / labeled_df.height * 100:.1f}%)")
    print(f"Biased (1):     {biased} ({biased / labeled_df.height * 100:.1f}%)")
    print(f"\nBias probability distribution:")
    bp = labeled_df["bias_prob"]
    print(f"  Mean:   {bp.mean():.4f}")
    print(f"  Median: {bp.median():.4f}")
    print(f"  Std:    {bp.std():.4f}")
    print(f"  Min:    {bp.min():.4f}")
    print(f"  Max:    {bp.max():.4f}")

    # Bias by outlet
    print(f"\nBias rate by top outlets:")
    top_outlets = (
        labeled_df.group_by("media").len().sort("len", descending=True).head(15)["media"].to_list()
    )
    outlet_bias = (
        labeled_df.filter(pl.col("media").is_in(top_outlets))
        .group_by("media")
        .agg([
            pl.len().alias("count"),
            pl.col("bias_label").mean().alias("bias_rate"),
            pl.col("bias_prob").mean().alias("avg_bias_prob"),
        ])
        .sort("bias_rate", descending=True)
    )
    for row in outlet_bias.iter_rows():
        print(f"  {row[0]:40s} count={row[1]:5d}  bias_rate={row[2]:.3f}  avg_prob={row[3]:.3f}")

    # Bias by year
    print(f"\nBias rate by year:")
    year_bias = (
        labeled_df.group_by("year")
        .agg([
            pl.len().alias("count"),
            pl.col("bias_label").mean().alias("bias_rate"),
            pl.col("bias_prob").mean().alias("avg_bias_prob"),
        ])
        .sort("year")
    )
    for row in year_bias.iter_rows():
        print(f"  {row[0]}  count={row[1]:5d}  bias_rate={row[2]:.3f}  avg_prob={row[3]:.3f}")


if __name__ == "__main__":
    main()
