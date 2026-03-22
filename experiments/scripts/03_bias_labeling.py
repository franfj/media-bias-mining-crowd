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

import numpy as np
import pandas as pd
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
    df: pd.DataFrame,
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Add bias labels and probabilities to the article DataFrame.

    Uses title + first 400 chars of text as input.
    """
    df = df.copy()

    # Prepare input texts: title + text
    texts = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        # Combine title and text, truncating text if needed
        combined = f"{title}. {text[:400]}" if text else title
        texts.append(combined)

    logger.info("Labeling %d articles in batches of %d...", len(texts), batch_size)

    all_labels = []
    all_probs = []

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

    df["bias_label"] = all_labels
    df["bias_prob"] = all_probs

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
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d articles", len(df))

    # Load model
    tokenizer, model = load_model(device)

    # Label
    labeled_df = label_articles(df, tokenizer, model, device, args.batch_size)

    # Save
    output_path = args.data_dir / args.output_file
    labeled_df.to_parquet(output_path, index=False)
    logger.info("Saved labeled dataset: %s", output_path)

    # Summary
    print("\n--- Bias Labeling Summary ---")
    print(f"Total articles: {len(labeled_df)}")
    biased = labeled_df["bias_label"].sum()
    non_biased = len(labeled_df) - biased
    print(f"Non-biased (0): {non_biased} ({non_biased/len(labeled_df)*100:.1f}%)")
    print(f"Biased (1):     {biased} ({biased/len(labeled_df)*100:.1f}%)")
    print(f"\nBias probability distribution:")
    print(f"  Mean:   {labeled_df['bias_prob'].mean():.4f}")
    print(f"  Median: {labeled_df['bias_prob'].median():.4f}")
    print(f"  Std:    {labeled_df['bias_prob'].std():.4f}")
    print(f"  Min:    {labeled_df['bias_prob'].min():.4f}")
    print(f"  Max:    {labeled_df['bias_prob'].max():.4f}")

    # Bias by outlet
    print(f"\nBias rate by top outlets:")
    top_outlets = labeled_df["media"].value_counts().head(15).index
    outlet_bias = labeled_df[labeled_df["media"].isin(top_outlets)].groupby("media").agg(
        count=("bias_label", "size"),
        bias_rate=("bias_label", "mean"),
        avg_bias_prob=("bias_prob", "mean"),
    ).sort_values("bias_rate", ascending=False)
    print(outlet_bias.to_string())

    # Bias by year
    print(f"\nBias rate by year:")
    year_bias = labeled_df.groupby("year").agg(
        count=("bias_label", "size"),
        bias_rate=("bias_label", "mean"),
        avg_bias_prob=("bias_prob", "mean"),
    ).sort_index()
    print(year_bias.to_string())


if __name__ == "__main__":
    main()
