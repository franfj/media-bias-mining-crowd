#!/usr/bin/env python3
"""
Step 13: Multimodal fusion — text embeddings + interaction features.

Trains and compares:
1. Interaction-only classifier (baseline from step 05)
2. Text-only classifier (sentence-transformer embeddings)
3. Early fusion: concatenate text embeddings + interaction features
4. Late fusion: ensemble of text-only and interaction-only predictions

This is THE experiment that demonstrates complementarity.

Usage:
    python 13_multimodal_fusion.py
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fusion")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

INTERACTION_FEATURES = [
    "score", "num_comments", "avg_comment_karma", "median_comment_karma",
    "std_comment_karma", "karma_range", "karma_iqr", "pct_negative_karma",
    "pct_positive_karma", "pct_zero_karma", "avg_comment_length",
    "unique_commenters", "comments_per_commenter",
    "comment_activity_duration_hours", "total_comment_karma",
    "text_length", "num_tags", "hour_of_day", "is_weekend",
]


def get_text_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Compute sentence-transformer embeddings for article texts."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading sentence-transformers model on %s...", device)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

    logger.info("Encoding %d texts...", len(texts))
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings


def evaluate_cv(X, y, model_class, model_kwargs, n_splits=5):
    """Run stratified CV and return metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro"))
        aucs.append(roc_auc_score(y_te, y_prob))

    return {
        "accuracy": np.mean(accs), "accuracy_std": np.std(accs),
        "f1_macro": np.mean(f1s), "f1_std": np.std(f1s),
        "auc": np.mean(aucs), "auc_std": np.std(aucs),
    }


def late_fusion_cv(X_inter, X_text, y, n_splits=5):
    """Late fusion: average probabilities from two independent classifiers."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in cv.split(X_inter, y):
        y_tr, y_te = y[train_idx], y[test_idx]

        # Interaction model
        sc1 = StandardScaler()
        X1_tr = sc1.fit_transform(X_inter[train_idx])
        X1_te = sc1.transform(X_inter[test_idx])
        m1 = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        m1.fit(X1_tr, y_tr)
        p1 = m1.predict_proba(X1_te)[:, 1]

        # Text model
        sc2 = StandardScaler()
        X2_tr = sc2.fit_transform(X_text[train_idx])
        X2_te = sc2.transform(X_text[test_idx])
        m2 = LogisticRegression(max_iter=1000, random_state=42)
        m2.fit(X2_tr, y_tr)
        p2 = m2.predict_proba(X2_te)[:, 1]

        # Fuse
        p_fused = (p1 + p2) / 2
        y_pred = (p_fused > 0.5).astype(int)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro"))
        aucs.append(roc_auc_score(y_te, p_fused))

    return {
        "accuracy": np.mean(accs), "accuracy_std": np.std(accs),
        "f1_macro": np.mean(f1s), "f1_std": np.std(f1s),
        "auc": np.mean(aucs), "auc_std": np.std(aucs),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    needed = INTERACTION_FEATURES + ["bias_label", "title", "text"]
    cols = [c for c in needed if c in schema]
    df = pd.read_parquet(DATA_DIR / "articles_with_features.parquet", columns=cols)
    available = [f for f in INTERACTION_FEATURES if f in df.columns]

    y = df["bias_label"].values
    X_inter = df[available].fillna(0).values

    # Prepare text: title + first 400 chars
    texts = []
    for _, row in df.iterrows():
        t = str(row.get("title", "")) if row.get("title") is not None else ""
        b = str(row.get("text", "")) if row.get("text") is not None else ""
        texts.append(f"{t}. {b[:400]}" if b else t)

    # Compute text embeddings
    X_text = get_text_embeddings(texts)
    logger.info("Text embeddings shape: %s", X_text.shape)

    # Save embeddings for reuse
    np.save(DATA_DIR / "text_embeddings.npy", X_text)

    # Early fusion: concatenate
    X_early = np.hstack([X_inter, X_text])
    logger.info("Early fusion shape: %s", X_early.shape)

    # === Run experiments ===
    gb_kwargs = {"n_estimators": 200, "max_depth": 4, "random_state": 42}
    lr_kwargs = {"max_iter": 1000, "random_state": 42}

    print("\n" + "=" * 70)
    print("MULTIMODAL FUSION EXPERIMENT")
    print("=" * 70)

    results = []

    # 1. Interaction only
    logger.info("Evaluating: Interaction features only (GB)...")
    r = evaluate_cv(X_inter, y, GradientBoostingClassifier, gb_kwargs)
    r["model"] = "Interaction only (GB)"
    results.append(r)
    print(f"\n  Interaction only:  AUC={r['auc']:.4f}±{r['auc_std']:.4f}  F1={r['f1_macro']:.4f}")

    # 2. Text only (LR on embeddings)
    logger.info("Evaluating: Text embeddings only (LR)...")
    r = evaluate_cv(X_text, y, LogisticRegression, lr_kwargs)
    r["model"] = "Text only (LR)"
    results.append(r)
    print(f"  Text only (LR):   AUC={r['auc']:.4f}±{r['auc_std']:.4f}  F1={r['f1_macro']:.4f}")

    # 3. Text only (GB on embeddings)
    logger.info("Evaluating: Text embeddings only (GB)...")
    r = evaluate_cv(X_text, y, GradientBoostingClassifier, gb_kwargs)
    r["model"] = "Text only (GB)"
    results.append(r)
    print(f"  Text only (GB):   AUC={r['auc']:.4f}±{r['auc_std']:.4f}  F1={r['f1_macro']:.4f}")

    # 4. Early fusion (GB)
    logger.info("Evaluating: Early fusion (GB)...")
    r = evaluate_cv(X_early, y, GradientBoostingClassifier, gb_kwargs)
    r["model"] = "Early fusion (GB)"
    results.append(r)
    print(f"  Early fusion (GB): AUC={r['auc']:.4f}±{r['auc_std']:.4f}  F1={r['f1_macro']:.4f}")

    # 5. Early fusion (LR)
    logger.info("Evaluating: Early fusion (LR)...")
    r = evaluate_cv(X_early, y, LogisticRegression, lr_kwargs)
    r["model"] = "Early fusion (LR)"
    results.append(r)
    print(f"  Early fusion (LR): AUC={r['auc']:.4f}±{r['auc_std']:.4f}  F1={r['f1_macro']:.4f}")

    # 6. Late fusion
    logger.info("Evaluating: Late fusion (GB+LR)...")
    r = late_fusion_cv(X_inter, X_text, y)
    r["model"] = "Late fusion (GB+LR)"
    results.append(r)
    print(f"  Late fusion:       AUC={r['auc']:.4f}±{r['auc_std']:.4f}  F1={r['f1_macro']:.4f}")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "multimodal_fusion.csv", index=False)

    print(f"\n  Results saved to: {RESULTS_DIR / 'multimodal_fusion.csv'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
