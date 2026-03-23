#!/usr/bin/env python3
"""
Step 15: Weak supervision (Snorkel-style) — interaction features as labeling functions.

Uses interaction features to define labeling functions that vote on bias,
then combines them with Snorkel's LabelModel to generate probabilistic labels.
Compares Snorkel labels with the DistilBERT labels to assess agreement
and evaluate whether interaction-based weak supervision is viable.

Usage:
    python 15_weak_supervision.py
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weaksup")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ABSTAIN = -1

def make_labeling_functions(df: pd.DataFrame) -> np.ndarray:
    """Define labeling functions based on interaction feature thresholds.

    Each LF votes: 1 (biased), 0 (non-biased), or -1 (abstain).
    Thresholds are set at the 75th percentile of each feature.
    """
    lf_names = []
    votes_list = []

    def add_lf(name, series, threshold, direction="above"):
        lf_names.append(name)
        if direction == "above":
            votes = np.where(series > threshold, 1, np.where(series < threshold * 0.5, 0, ABSTAIN))
        else:
            votes = np.where(series < threshold, 1, np.where(series > threshold * 1.5, 0, ABSTAIN))
        votes_list.append(votes)

    # Karma-based LFs
    if "avg_comment_karma" in df.columns:
        p75 = df["avg_comment_karma"].quantile(0.75)
        add_lf("high_avg_karma", df["avg_comment_karma"], p75, "above")

    if "std_comment_karma" in df.columns:
        p75 = df["std_comment_karma"].quantile(0.75)
        add_lf("high_karma_std", df["std_comment_karma"], p75, "above")

    if "karma_iqr" in df.columns:
        p75 = df["karma_iqr"].quantile(0.75)
        add_lf("high_karma_iqr", df["karma_iqr"], p75, "above")

    # Text-based LFs
    if "text_length" in df.columns:
        p75 = df["text_length"].quantile(0.75)
        add_lf("long_text", df["text_length"], p75, "above")

    if "avg_comment_length" in df.columns:
        p25 = df["avg_comment_length"].quantile(0.25)
        add_lf("short_comments", df["avg_comment_length"], p25, "below")

    # Engagement LFs
    if "score" in df.columns:
        p75 = df["score"].quantile(0.75)
        add_lf("high_score", df["score"], p75, "above")

    if "is_weekend" in df.columns:
        votes = np.where(df["is_weekend"] == 1, 1, ABSTAIN)
        lf_names.append("weekend_post")
        votes_list.append(votes)

    # Temporal LFs (from karma_features if available)
    karma_path = DATA_DIR / "karma_features.parquet"
    if karma_path.exists():
        karma_df = pd.read_parquet(karma_path, columns=["article_id", "karma_entropy"])
        merged = df[["article_id"]].merge(karma_df, on="article_id", how="left")
        if "karma_entropy" in merged.columns:
            p75 = merged["karma_entropy"].quantile(0.75)
            add_lf("high_entropy", merged["karma_entropy"].fillna(0), p75, "above")

    L = np.column_stack(votes_list).astype(int)
    logger.info("Created %d labeling functions: %s", len(lf_names), lf_names)
    return L, lf_names


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    df = pd.read_parquet(DATA_DIR / "articles_with_features.parquet")
    y_true = df["bias_label"].values

    # Create labeling functions
    L, lf_names = make_labeling_functions(df)

    # LF statistics
    print("\n" + "=" * 60)
    print("LABELING FUNCTION STATISTICS")
    print("=" * 60)
    print(f"\n{'LF Name':<25} {'Coverage':>8} {'Accuracy':>8} {'Votes=1':>8}")
    print("-" * 55)
    for i, name in enumerate(lf_names):
        votes = L[:, i]
        coverage = (votes != ABSTAIN).mean()
        non_abstain = votes != ABSTAIN
        if non_abstain.sum() > 0:
            acc = (votes[non_abstain] == y_true[non_abstain]).mean()
            pct_biased = (votes[non_abstain] == 1).mean()
        else:
            acc, pct_biased = 0, 0
        print(f"  {name:<25} {coverage:>7.1%} {acc:>7.1%} {pct_biased:>7.1%}")

    # Snorkel LabelModel
    logger.info("Training Snorkel LabelModel...")
    from snorkel.labeling.model import LabelModel

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L, n_epochs=500, lr=0.01, log_freq=100, seed=42)

    # Predict
    y_snorkel_prob = label_model.predict_proba(L)[:, 1]
    y_snorkel = (y_snorkel_prob > 0.5).astype(int)

    # Compare with DistilBERT labels
    print("\n" + "=" * 60)
    print("SNORKEL vs DISTILBERT LABELS")
    print("=" * 60)

    agreement = (y_snorkel == y_true).mean()
    auc = roc_auc_score(y_true, y_snorkel_prob)
    f1 = f1_score(y_true, y_snorkel, average="macro")

    print(f"\n  Agreement with DistilBERT: {agreement:.1%}")
    print(f"  AUC (Snorkel probs vs DistilBERT labels): {auc:.4f}")
    print(f"  F1 Macro: {f1:.4f}")
    print(f"\n  Snorkel bias rate: {y_snorkel.mean():.1%}")
    print(f"  DistilBERT bias rate: {y_true.mean():.1%}")

    print(f"\n{classification_report(y_true, y_snorkel, target_names=['Non-biased','Biased'])}")

    # Save
    results = pd.DataFrame({
        "article_id": df["article_id"],
        "bias_label_distilbert": y_true,
        "bias_label_snorkel": y_snorkel,
        "bias_prob_snorkel": y_snorkel_prob,
    })
    results.to_csv(RESULTS_DIR / "weak_supervision_labels.csv", index=False)

    summary = pd.DataFrame({
        "metric": ["agreement", "auc", "f1_macro", "snorkel_bias_rate", "distilbert_bias_rate"],
        "value": [agreement, auc, f1, y_snorkel.mean(), y_true.mean()],
    })
    summary.to_csv(RESULTS_DIR / "weak_supervision_summary.csv", index=False)

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
