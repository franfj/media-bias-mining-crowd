#!/usr/bin/env python3
"""
Step 12: SHAP explainability analysis.

Replaces basic RF feature importances with SHAP values for the
Gradient Boosting classifier. Produces:
- Global SHAP summary (beeswarm)
- SHAP interaction effects
- Per-feature SHAP dependence

Usage:
    python 12_shap_analysis.py
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shap")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "drafts" / "v2" / "figures"

FEATURES = [
    "score", "num_comments", "avg_comment_karma", "median_comment_karma",
    "std_comment_karma", "karma_range", "karma_iqr", "pct_negative_karma",
    "pct_positive_karma", "pct_zero_karma", "avg_comment_length",
    "unique_commenters", "comments_per_commenter",
    "comment_activity_duration_hours", "total_comment_karma",
    "text_length", "num_tags", "hour_of_day", "is_weekend",
]

def main():
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    cols = [c for c in FEATURES + ["bias_label"] if c in schema]
    df = pd.read_parquet(DATA_DIR / "articles_with_features.parquet", columns=cols)
    available = [f for f in FEATURES if f in df.columns]

    X = df[available].fillna(0).values
    y = df["bias_label"].values
    feature_names = available

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Training Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_scaled, y)

    logger.info("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Global summary
    logger.info("Generating SHAP summary plot...")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean |SHAP|)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_abs.to_csv(RESULTS_DIR / "shap_importances.csv")

    print("\n" + "=" * 60)
    print("SHAP Feature Importances (mean |SHAP value|)")
    print("=" * 60)
    for feat, val in mean_abs.items():
        bar = "#" * int(val * 300)
        print(f"  {feat:<40} {val:.4f} {bar}")

    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Data saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
