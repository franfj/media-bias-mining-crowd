#!/usr/bin/env python3
"""
Step 14: Stance detection in comments — agreement/disagreement with article.

Uses a zero-shot NLI model to classify whether each comment agrees,
disagrees, or is neutral toward the article's claim/framing.
This is more informative for bias detection than generic sentiment.

Usage:
    python 14_stance_detection.py [--sample-size 10000]
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stance")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def load_sample(sample_size: int = 10000) -> pd.DataFrame:
    """Load a stratified sample of comments with their article titles."""
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    cols = [c for c in ["article_id", "bias_label", "title"] if c in schema]
    articles = pd.read_parquet(DATA_DIR / "articles_with_features.parquet", columns=cols)
    biased_ids = set(articles[articles["bias_label"] == 1]["article_id"])
    non_biased_ids = set(articles[articles["bias_label"] == 0]["article_id"])
    title_map = dict(zip(articles["article_id"], articles["title"]))

    # Load comment index (lightweight)
    index_df = pd.read_parquet(DATA_DIR / "comments_raw.parquet",
                               columns=["article_id", "comment_text_length"])
    index_df["bias_label"] = index_df["article_id"].map(
        lambda x: 1 if x in biased_ids else (0 if x in non_biased_ids else -1))
    index_df = index_df[(index_df["bias_label"] >= 0) & (index_df["comment_text_length"] >= 20)]

    half = sample_size // 2
    rng = np.random.RandomState(42)
    biased_idx = rng.choice(index_df[index_df["bias_label"] == 1].index,
                            size=min(half, (index_df["bias_label"] == 1).sum()), replace=False)
    non_biased_idx = rng.choice(index_df[index_df["bias_label"] == 0].index,
                                size=min(half, (index_df["bias_label"] == 0).sum()), replace=False)
    selected = np.sort(np.concatenate([biased_idx, non_biased_idx]))

    bias_labels = index_df.loc[selected, "bias_label"].values
    article_ids = index_df.loc[selected, "article_id"].values
    del index_df

    # Load texts for selected
    pf = pq.ParquetFile(DATA_DIR / "comments_raw.parquet")
    selected_set = set(selected)
    texts, global_idx = [], 0
    for batch in pf.iter_batches(batch_size=500_000, columns=["comment_text"]):
        bdf = batch.to_pandas()
        batch_sel = [i for i in range(global_idx, global_idx + len(bdf)) if i in selected_set]
        if batch_sel:
            local = [i - global_idx for i in batch_sel]
            texts.extend(bdf.iloc[local]["comment_text"].tolist())
        global_idx += len(bdf)
        del bdf

    sample = pd.DataFrame({
        "article_id": article_ids, "comment_text": texts,
        "bias_label": bias_labels,
        "title": [title_map.get(aid, "") for aid in article_ids],
    })
    logger.info("Sample: %d comments", len(sample))
    return sample


def run_stance_detection(sample: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Zero-shot NLI stance detection: does the comment agree with the article title?"""
    from transformers import pipeline
    import torch

    device = 0 if torch.backends.mps.is_available() else -1
    logger.info("Loading zero-shot NLI pipeline...")
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli",
                          device=-1)  # CPU for BART (MPS issues)

    candidate_labels = ["agreement", "disagreement", "neutral"]
    n = len(sample)
    stances, agree_scores, disagree_scores = [], [], []

    logger.info("Classifying stance for %d comments...", n)
    for i in range(0, n, batch_size):
        batch = sample.iloc[i:i+batch_size]
        # Premise: article title. Hypothesis: comment (truncated)
        for _, row in batch.iterrows():
            title = str(row["title"])[:200]
            comment = str(row["comment_text"])[:300]
            text = f"{title} [SEP] {comment}"
            try:
                result = classifier(text, candidate_labels, multi_label=False)
                label_scores = dict(zip(result["labels"], result["scores"]))
                stances.append(result["labels"][0])
                agree_scores.append(label_scores.get("agreement", 0))
                disagree_scores.append(label_scores.get("disagreement", 0))
            except Exception:
                stances.append("neutral")
                agree_scores.append(0.33)
                disagree_scores.append(0.33)

        if (i // batch_size) % 20 == 0:
            logger.info("  Progress: %d/%d (%.1f%%)", min(i+batch_size, n), n, min(i+batch_size, n)/n*100)

    sample["stance"] = stances
    sample["agree_score"] = agree_scores
    sample["disagree_score"] = disagree_scores
    return sample


def analyse(sample: pd.DataFrame):
    """Compare stance distributions between biased and non-biased."""
    print("\n" + "=" * 60)
    print("STANCE DETECTION: Biased vs Non-Biased Articles")
    print("=" * 60)

    for bl, name in [(0, "Non-biased"), (1, "Biased")]:
        sub = sample[sample["bias_label"] == bl]
        counts = sub["stance"].value_counts(normalize=True)
        print(f"\n  {name} ({len(sub)} comments):")
        for label in ["agreement", "disagreement", "neutral"]:
            pct = counts.get(label, 0) * 100
            print(f"    {label:<15} {pct:5.1f}%")

    # Statistical test on disagreement score
    g1 = sample[sample["bias_label"] == 0]["disagree_score"]
    g2 = sample[sample["bias_label"] == 1]["disagree_score"]
    _, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    print(f"\n  Disagreement score: Non-biased={g1.mean():.4f}, Biased={g2.mean():.4f}, p={p:.2e}")

    g1a = sample[sample["bias_label"] == 0]["agree_score"]
    g2a = sample[sample["bias_label"] == 1]["agree_score"]
    _, pa = sp_stats.mannwhitneyu(g1a, g2a, alternative="two-sided")
    print(f"  Agreement score:    Non-biased={g1a.mean():.4f}, Biased={g2a.mean():.4f}, p={pa:.2e}")

    # Save
    sample.to_parquet(DATA_DIR / "comments_with_stance.parquet", index=False)

    summary = pd.DataFrame({
        "metric": ["disagree_biased", "disagree_nonbiased", "disagree_p",
                    "agree_biased", "agree_nonbiased", "agree_p",
                    "pct_disagree_biased", "pct_disagree_nonbiased"],
        "value": [g2.mean(), g1.mean(), p, g2a.mean(), g1a.mean(), pa,
                  (sample[sample["bias_label"]==1]["stance"]=="disagreement").mean(),
                  (sample[sample["bias_label"]==0]["stance"]=="disagreement").mean()],
    })
    summary.to_csv(RESULTS_DIR / "stance_summary.csv", index=False)
    print(f"\n  Results saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_sample(args.sample_size)
    sample = run_stance_detection(sample)
    analyse(sample)

if __name__ == "__main__":
    main()
