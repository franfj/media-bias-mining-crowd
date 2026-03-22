#!/usr/bin/env python3
"""
Step 8: Comment sentiment analysis.

Analyses sentiment and emotion in Meneame comments using pysentimiento
(Spanish transformer-based sentiment model). Processes a stratified
sample to keep memory/time manageable (13M comments full would take days).

Outputs:
- Per-article aggregated sentiment features
- Sentiment differences: biased vs non-biased articles
- Emotion distribution by bias label

Usage:
    python 08_comment_sentiment.py
    python 08_comment_sentiment.py --sample-size 200000 --device mps
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentiment")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_SAMPLE_SIZE = 200_000
MAX_TEXT_LEN = 512  # truncate long comments for the model


def load_sample(comments_path: Path, articles_path: Path, sample_size: int) -> pd.DataFrame:
    """Load a stratified sample of comments (balanced by article bias label).

    Memory-efficient: first samples row indices using lightweight columns,
    then loads comment_text only for the selected rows.
    """
    logger.info("Loading articles metadata...")
    schema = [f.name for f in pq.read_schema(articles_path)]
    cols = [c for c in ["article_id", "bias_label", "media"] if c in schema]
    articles = pd.read_parquet(articles_path, columns=cols)
    biased_ids = set(articles[articles["bias_label"] == 1]["article_id"])
    non_biased_ids = set(articles[articles["bias_label"] == 0]["article_id"])

    half = sample_size // 2

    # Phase 1: load only article_id + text_length to pick indices (no text!)
    logger.info("Phase 1: selecting comment indices (lightweight)...")
    index_df = pd.read_parquet(comments_path,
                               columns=["article_id", "comment_text_length"])

    # Tag with bias label
    index_df["bias_label"] = index_df["article_id"].map(
        lambda x: 1 if x in biased_ids else (0 if x in non_biased_ids else -1)
    )
    index_df = index_df[index_df["bias_label"] >= 0]

    # Filter short comments
    index_df = index_df[index_df["comment_text_length"] >= 10]

    # Stratified sample of row positions
    biased_idx = index_df[index_df["bias_label"] == 1].index
    non_biased_idx = index_df[index_df["bias_label"] == 0].index

    rng = np.random.RandomState(42)
    n_biased = min(half, len(biased_idx))
    n_non_biased = min(half, len(non_biased_idx))

    selected_biased = rng.choice(biased_idx, size=n_biased, replace=False)
    selected_non_biased = rng.choice(non_biased_idx, size=n_non_biased, replace=False)
    selected = np.sort(np.concatenate([selected_biased, selected_non_biased]))

    # Build lookup of selected positions
    bias_labels = index_df.loc[selected, "bias_label"].values
    article_ids = index_df.loc[selected, "article_id"].values
    del index_df

    selected_set = set(selected)

    # Phase 2: load text only for selected rows via chunked reading
    logger.info("Phase 2: loading text for %d selected comments (chunked)...", len(selected))
    pf = pq.ParquetFile(comments_path)
    texts = []
    karmas = []
    global_idx = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=["comment_text", "comment_karma"]):
        batch_df = batch.to_pandas()
        batch_end = global_idx + len(batch_df)
        # Find which selected indices fall in this batch
        batch_selected = [i for i in range(global_idx, batch_end) if i in selected_set]
        if batch_selected:
            local_indices = [i - global_idx for i in batch_selected]
            texts.extend(batch_df.iloc[local_indices]["comment_text"].tolist())
            karmas.extend(batch_df.iloc[local_indices]["comment_karma"].tolist())
        global_idx = batch_end
        del batch_df

    sample = pd.DataFrame({
        "article_id": article_ids,
        "comment_text": texts,
        "comment_karma": karmas,
        "bias_label": bias_labels,
    })
    del texts, karmas

    logger.info("Sample: %d comments (%d biased, %d non-biased)",
                len(sample), n_biased, n_non_biased)
    return sample


def run_sentiment_analysis(sample: pd.DataFrame, device: str = "cpu", batch_size: int = 64) -> pd.DataFrame:
    """Run sentiment and emotion analysis using HuggingFace pipelines directly.

    Uses pysentimiento's underlying models via transformers pipeline for
    much faster inference (especially on MPS/CUDA).
    """
    import torch
    from transformers import pipeline

    # Pysentimiento model names on HuggingFace
    SENT_MODEL = "pysentimiento/robertuito-sentiment-analysis"
    EMO_MODEL = "pysentimiento/robertuito-emotion-analysis"

    # Map device string to torch device for pipeline
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = 0  # pipeline uses device index; we'll set device_map
        pipe_device = "mps"
    elif device == "cuda" and torch.cuda.is_available():
        pipe_device = 0
    else:
        pipe_device = -1  # CPU

    logger.info("Loading sentiment pipeline (%s) on %s...", SENT_MODEL, device)
    sent_pipe = pipeline("text-classification", model=SENT_MODEL, top_k=None,
                         device=pipe_device, truncation=True, max_length=MAX_TEXT_LEN)

    logger.info("Loading emotion pipeline (%s) on %s...", EMO_MODEL, device)
    emo_pipe = pipeline("text-classification", model=EMO_MODEL, top_k=None,
                        device=pipe_device, truncation=True, max_length=MAX_TEXT_LEN)

    texts = sample["comment_text"].str[:MAX_TEXT_LEN].tolist()
    n = len(texts)

    # --- Sentiment ---
    logger.info("Running sentiment analysis on %d texts (batch_size=%d)...", n, batch_size)
    sent_labels = []
    sent_scores = []

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        results = sent_pipe(batch, batch_size=len(batch))
        for res in results:
            probs = {r["label"]: r["score"] for r in res}
            label = max(probs, key=probs.get)
            polarity = probs.get("POS", 0) - probs.get("NEG", 0)
            sent_labels.append(label)
            sent_scores.append(polarity)

        if (i // batch_size) % 20 == 0:
            logger.info("  Sentiment progress: %d / %d (%.1f%%)", i + len(batch), n, (i + len(batch)) / n * 100)

    sample["sentiment"] = sent_labels
    sample["polarity_score"] = sent_scores

    # --- Emotion ---
    logger.info("Running emotion analysis on %d texts (batch_size=%d)...", n, batch_size)
    emo_labels = []
    emo_joy = []
    emo_anger = []
    emo_sadness = []
    emo_fear = []

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        results = emo_pipe(batch, batch_size=len(batch))
        for res in results:
            probs = {r["label"]: r["score"] for r in res}
            label = max(probs, key=probs.get)
            emo_labels.append(label)
            emo_joy.append(probs.get("joy", 0))
            emo_anger.append(probs.get("anger", 0))
            emo_sadness.append(probs.get("sadness", 0))
            emo_fear.append(probs.get("fear", 0))

        if (i // batch_size) % 20 == 0:
            logger.info("  Emotion progress: %d / %d (%.1f%%)", i + len(batch), n, (i + len(batch)) / n * 100)

    sample["emotion"] = emo_labels
    sample["emo_joy"] = emo_joy
    sample["emo_anger"] = emo_anger
    sample["emo_sadness"] = emo_sadness
    sample["emo_fear"] = emo_fear

    return sample


def analyse_results(sample: pd.DataFrame, output_dir: Path):
    """Aggregate and compare sentiment between biased and non-biased articles."""

    # --- Sentiment distribution ---
    print("\n" + "=" * 80)
    print("SENTIMENT DISTRIBUTION: Biased vs Non-Biased")
    print("=" * 80)

    for label_val, label_name in [(0, "Non-biased"), (1, "Biased")]:
        subset = sample[sample["bias_label"] == label_val]
        counts = subset["sentiment"].value_counts(normalize=True)
        print(f"\n  {label_name} articles ({len(subset):,} comments):")
        for sent in ["POS", "NEU", "NEG"]:
            pct = counts.get(sent, 0) * 100
            bar = "#" * int(pct)
            print(f"    {sent}: {pct:5.1f}% {bar}")

    # --- Polarity comparison ---
    print("\n" + "=" * 80)
    print("POLARITY SCORE: Biased vs Non-Biased")
    print("=" * 80)

    from scipy import stats as sp_stats

    biased_pol = sample[sample["bias_label"] == 1]["polarity_score"]
    non_biased_pol = sample[sample["bias_label"] == 0]["polarity_score"]

    u_stat, p_val = sp_stats.mannwhitneyu(biased_pol, non_biased_pol, alternative="two-sided")
    print(f"\n  Non-biased: mean polarity = {non_biased_pol.mean():+.4f} (std={non_biased_pol.std():.4f})")
    print(f"  Biased:     mean polarity = {biased_pol.mean():+.4f} (std={biased_pol.std():.4f})")
    print(f"  Mann-Whitney U p-value: {p_val:.2e}")

    # --- Emotion comparison ---
    print("\n" + "=" * 80)
    print("EMOTION DISTRIBUTION: Biased vs Non-Biased")
    print("=" * 80)

    emotions = ["emo_joy", "emo_anger", "emo_sadness", "emo_fear"]
    emotion_names = ["Joy", "Anger", "Sadness", "Fear"]

    print(f"\n{'Emotion':<12} {'Non-biased':>12} {'Biased':>12} {'Diff':>8} {'p-value':>12}")
    print("-" * 60)
    for emo_col, emo_name in zip(emotions, emotion_names):
        g1 = sample[sample["bias_label"] == 0][emo_col]
        g2 = sample[sample["bias_label"] == 1][emo_col]
        _, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        diff = g2.mean() - g1.mean()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{emo_name:<12} {g1.mean():>12.4f} {g2.mean():>12.4f} {diff:>+7.4f} {p:>12.2e} {sig}")

    # --- Per-article aggregation ---
    print("\n" + "=" * 80)
    print("PER-ARTICLE AGGREGATED SENTIMENT")
    print("=" * 80)

    article_sent = sample.groupby(["article_id", "bias_label"]).agg(
        n_comments=("polarity_score", "count"),
        mean_polarity=("polarity_score", "mean"),
        std_polarity=("polarity_score", "std"),
        pct_positive=("sentiment", lambda x: (x == "POS").mean()),
        pct_negative=("sentiment", lambda x: (x == "NEG").mean()),
        mean_anger=("emo_anger", "mean"),
        mean_joy=("emo_joy", "mean"),
        mean_fear=("emo_fear", "mean"),
    ).reset_index()

    # Compare at article level
    biased_art = article_sent[article_sent["bias_label"] == 1]
    non_biased_art = article_sent[article_sent["bias_label"] == 0]

    print(f"\n  Articles with sampled comments: {len(article_sent)}")
    print(f"  Biased: {len(biased_art)}, Non-biased: {len(non_biased_art)}")
    print(f"\n  Biased articles:     mean polarity = {biased_art['mean_polarity'].mean():+.4f}, %NEG = {biased_art['pct_negative'].mean()*100:.1f}%")
    print(f"  Non-biased articles: mean polarity = {non_biased_art['mean_polarity'].mean():+.4f}, %NEG = {non_biased_art['pct_negative'].mean()*100:.1f}%")

    # Save all results
    sample.to_parquet(output_dir.parent / "data" / "comments_with_sentiment.parquet", index=False)
    article_sent.to_csv(output_dir / "article_sentiment.csv", index=False)

    # Summary table
    summary = {
        "metric": ["polarity_biased", "polarity_non_biased", "polarity_p_value",
                    "pct_neg_biased", "pct_neg_non_biased",
                    "anger_biased", "anger_non_biased",
                    "joy_biased", "joy_non_biased"],
        "value": [biased_pol.mean(), non_biased_pol.mean(), p_val,
                  (sample[sample["bias_label"] == 1]["sentiment"] == "NEG").mean(),
                  (sample[sample["bias_label"] == 0]["sentiment"] == "NEG").mean(),
                  sample[sample["bias_label"] == 1]["emo_anger"].mean(),
                  sample[sample["bias_label"] == 0]["emo_anger"].mean(),
                  sample[sample["bias_label"] == 1]["emo_joy"].mean(),
                  sample[sample["bias_label"] == 0]["emo_joy"].mean()],
    }
    pd.DataFrame(summary).to_csv(output_dir / "sentiment_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Comment sentiment analysis")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comments_path = args.data_dir / "comments_raw.parquet"
    articles_path = args.data_dir / "articles_with_features.parquet"

    sample = load_sample(comments_path, articles_path, args.sample_size)
    sample = run_sentiment_analysis(sample, device=args.device, batch_size=args.batch_size)
    analyse_results(sample, RESULTS_DIR)

    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
