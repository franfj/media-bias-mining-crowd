#!/usr/bin/env python3
"""
Step 11: Re-label articles using the new DistilBERT multilingual model.

Compares old labels (franfj/fdtd_media_bias_E) with new labels from the
model trained on BEADS + MBBMD. Produces:
- articles_relabeled.parquet with both old and new labels
- articles_with_features_v2.parquet (updated features file for analysis)
- Comparison statistics

Usage:
    python 11_relabel_with_new_model.py
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
logger = logging.getLogger("relabel")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NEW_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xlm-roberta-bias"


def predict_batch(texts, tokenizer, model, device, max_length=256):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    bias_probs = probs[:, 1].cpu().numpy().tolist()
    labels = (probs[:, 1] > 0.5).cpu().numpy().astype(int).tolist()
    return labels, bias_probs


def main():
    parser = argparse.ArgumentParser(description="Re-label articles with new model")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=NEW_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS backend")

    # Load current labeled data
    labeled_path = args.data_dir / "articles_with_features.parquet"
    logger.info("Loading articles from %s", labeled_path)
    df = pl.read_parquet(labeled_path)
    logger.info("Loaded %d articles", df.height)

    # Rename old labels
    df = df.rename({"bias_label": "bias_label_old", "bias_prob": "bias_prob_old"})

    # Load new model
    logger.info("Loading new model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_path))
    model.to(device)
    model.eval()

    # Prepare texts (title + first 400 chars)
    titles = df["title"].to_list()
    texts_col = df["text"].to_list()
    texts = []
    for title, text in zip(titles, texts_col):
        title = str(title) if title is not None else ""
        text = str(text) if text is not None else ""
        combined = f"{title}. {text[:400]}" if text else title
        texts.append(combined)

    # Predict in batches
    logger.info("Predicting with new model (batch_size=%d)...", args.batch_size)
    all_labels, all_probs = [], []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        labels, probs = predict_batch(batch, tokenizer, model, device)
        all_labels.extend(labels)
        all_probs.extend(probs)
        if (i // args.batch_size) % 50 == 0:
            logger.info("  Progress: %d/%d (%.1f%%)",
                        min(i + args.batch_size, len(texts)), len(texts),
                        min(i + args.batch_size, len(texts)) / len(texts) * 100)

    # Add new labels
    df = df.with_columns([
        pl.Series("bias_label", all_labels, dtype=pl.Int32),
        pl.Series("bias_prob", all_probs, dtype=pl.Float64),
    ])

    # Save relabeled dataset
    output_path = args.data_dir / "articles_with_features_v2.parquet"
    df.write_parquet(output_path)
    logger.info("Saved relabeled dataset: %s", output_path)

    # === Comparison ===
    old_biased = df.filter(pl.col("bias_label_old") == 1).height
    new_biased = df.filter(pl.col("bias_label") == 1).height
    total = df.height

    # Agreement
    agree = df.filter(pl.col("bias_label") == pl.col("bias_label_old")).height
    agreement_pct = agree / total * 100

    # Confusion between old and new
    both_biased = df.filter((pl.col("bias_label") == 1) & (pl.col("bias_label_old") == 1)).height
    old_only = df.filter((pl.col("bias_label") == 0) & (pl.col("bias_label_old") == 1)).height
    new_only = df.filter((pl.col("bias_label") == 1) & (pl.col("bias_label_old") == 0)).height
    neither = df.filter((pl.col("bias_label") == 0) & (pl.col("bias_label_old") == 0)).height

    print("\n" + "=" * 70)
    print("  RE-LABELING COMPARISON: Old (fdtd_media_bias_E) vs New (BEADS+MBBMD)")
    print("=" * 70)
    print(f"\n  Old model — Biased: {old_biased} ({old_biased/total*100:.1f}%), Non-biased: {total-old_biased} ({(total-old_biased)/total*100:.1f}%)")
    print(f"  New model — Biased: {new_biased} ({new_biased/total*100:.1f}%), Non-biased: {total-new_biased} ({(total-new_biased)/total*100:.1f}%)")
    print(f"\n  Agreement: {agree}/{total} ({agreement_pct:.1f}%)")

    print(f"\n  Confusion matrix:")
    print(f"  {'':>25} {'New=NonBiased':>15} {'New=Biased':>15}")
    print(f"  {'Old=NonBiased':<25} {neither:>15} {new_only:>15}")
    print(f"  {'Old=Biased':<25} {old_only:>15} {both_biased:>15}")

    # Correlation between probability scores
    import numpy as np
    from scipy import stats as sp_stats
    old_p = df["bias_prob_old"].to_numpy()
    new_p = df["bias_prob"].to_numpy()
    r, p = sp_stats.pearsonr(old_p, new_p)
    rho, _ = sp_stats.spearmanr(old_p, new_p)
    print(f"\n  Probability correlation: Pearson r={r:.4f}, Spearman ρ={rho:.4f}")

    print(f"\n  New bias probability distribution:")
    bp = df["bias_prob"]
    print(f"    Mean:   {bp.mean():.4f}")
    print(f"    Median: {bp.median():.4f}")
    print(f"    Std:    {bp.std():.4f}")

    # Per-outlet comparison (top outlets)
    print(f"\n  Bias rate change by top outlets:")
    print(f"  {'Outlet':<35} {'Old%':>6} {'New%':>6} {'Diff':>7}")
    print("  " + "-" * 56)
    top_outlets = (
        df.group_by("media").len().sort("len", descending=True).head(15)["media"].to_list()
    )
    for outlet in top_outlets:
        odf = df.filter(pl.col("media") == outlet)
        old_rate = odf["bias_label_old"].mean() * 100
        new_rate = odf["bias_label"].mean() * 100
        print(f"  {outlet:<35} {old_rate:>5.1f}% {new_rate:>5.1f}% {new_rate-old_rate:>+6.1f}%")

    print(f"\n  Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
