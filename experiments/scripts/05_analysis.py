#!/usr/bin/env python3
"""
Step 5: Analysis — Correlation, statistical tests, and exploration.

Runs:
- Correlation matrix between bias and interaction features
- Statistical significance tests (Mann-Whitney U, effect sizes)
- Topic-level analysis
- Media outlet analysis
- Temporal analysis
- Feature importance for bias prediction

Usage:
    python 05_analysis.py
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analysis")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Interaction feature columns for analysis
INTERACTION_FEATURES = [
    "score",
    "num_comments",
    "avg_comment_karma",
    "median_comment_karma",
    "std_comment_karma",
    "karma_range",
    "karma_iqr",
    "pct_negative_karma",
    "pct_positive_karma",
    "pct_zero_karma",
    "avg_comment_length",
    "unique_commenters",
    "comments_per_commenter",
    "comment_activity_duration_hours",
    "total_comment_karma",
    "text_length",
    "num_tags",
    "hour_of_day",
    "is_weekend",
]


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def run_statistical_tests(df: pd.DataFrame, output_dir: Path):
    """Run Mann-Whitney U tests comparing biased vs non-biased articles."""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS: Biased vs Non-Biased Articles")
    print("=" * 80)

    biased = df[df["bias_label"] == 1]
    non_biased = df[df["bias_label"] == 0]
    print(f"N biased: {len(biased)}, N non-biased: {len(non_biased)}")

    results = []
    for feat in INTERACTION_FEATURES:
        if feat not in df.columns:
            continue

        g1 = non_biased[feat].dropna()
        g2 = biased[feat].dropna()

        if len(g1) < 10 or len(g2) < 10:
            continue

        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        # Effect size (Cohen's d)
        d = cohens_d(g2, g1)  # positive d means biased > non-biased

        # Rank-biserial correlation (effect size for Mann-Whitney)
        n1, n2 = len(g1), len(g2)
        r = 1 - (2 * u_stat) / (n1 * n2)

        results.append({
            "feature": feat,
            "mean_non_biased": g1.mean(),
            "mean_biased": g2.mean(),
            "diff_pct": (g2.mean() - g1.mean()) / max(abs(g1.mean()), 0.001) * 100,
            "cohens_d": d,
            "rank_biserial_r": r,
            "u_statistic": u_stat,
            "p_value": p_value,
            "significant_001": p_value < 0.001,
            "significant_01": p_value < 0.01,
            "significant_05": p_value < 0.05,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value")

    print(f"\n{'Feature':<40} {'Non-biased':>12} {'Biased':>12} {'Diff%':>8} {'Cohen d':>8} {'p-value':>12} {'Sig':>4}")
    print("-" * 100)
    for _, row in results_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"{row['feature']:<40} {row['mean_non_biased']:>12.3f} {row['mean_biased']:>12.3f} {row['diff_pct']:>+7.1f}% {row['cohens_d']:>+7.3f} {row['p_value']:>12.2e} {sig:>4}")

    results_df.to_csv(output_dir / "statistical_tests.csv", index=False)
    return results_df


def run_correlation_analysis(df: pd.DataFrame, output_dir: Path):
    """Compute correlations between bias probability and interaction features."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS: Bias Probability vs Interaction Features")
    print("=" * 80)

    available_features = [f for f in INTERACTION_FEATURES if f in df.columns]

    # Pearson and Spearman correlations with bias_prob
    correlations = []
    for feat in available_features:
        col = df[feat].dropna()
        bias = df.loc[col.index, "bias_prob"]

        r_pearson, p_pearson = stats.pearsonr(col, bias)
        r_spearman, p_spearman = stats.spearmanr(col, bias)

        correlations.append({
            "feature": feat,
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_rho": r_spearman,
            "spearman_p": p_spearman,
        })

    corr_df = pd.DataFrame(correlations).sort_values("spearman_rho", key=abs, ascending=False)

    print(f"\n{'Feature':<40} {'Pearson r':>10} {'p-value':>12} {'Spearman rho':>12} {'p-value':>12}")
    print("-" * 90)
    for _, row in corr_df.iterrows():
        print(f"{row['feature']:<40} {row['pearson_r']:>+10.4f} {row['pearson_p']:>12.2e} {row['spearman_rho']:>+12.4f} {row['spearman_p']:>12.2e}")

    corr_df.to_csv(output_dir / "correlations.csv", index=False)

    # Full correlation matrix
    feature_cols = available_features + ["bias_prob", "bias_label"]
    corr_matrix = df[feature_cols].corr(method="spearman")
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")

    return corr_df


def run_outlet_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze bias patterns by media outlet."""
    print("\n" + "=" * 80)
    print("OUTLET-LEVEL ANALYSIS")
    print("=" * 80)

    # Top outlets by article count
    outlet_counts = df["media"].value_counts()
    top_outlets = outlet_counts[outlet_counts >= 30].index  # At least 30 articles

    outlet_stats = []
    for outlet in top_outlets:
        outlet_df = df[df["media"] == outlet]
        outlet_stats.append({
            "outlet": outlet,
            "article_count": len(outlet_df),
            "bias_rate": outlet_df["bias_label"].mean(),
            "avg_bias_prob": outlet_df["bias_prob"].mean(),
            "avg_score": outlet_df["score"].mean(),
            "avg_comments": outlet_df["num_comments"].mean(),
            "avg_comment_karma": outlet_df["avg_comment_karma"].mean(),
            "avg_karma_range": outlet_df["karma_range"].mean(),
            "avg_pct_negative_karma": outlet_df["pct_negative_karma"].mean(),
            "avg_unique_commenters": outlet_df["unique_commenters"].mean(),
        })

    outlet_df = pd.DataFrame(outlet_stats).sort_values("bias_rate", ascending=False)

    print(f"\n{'Outlet':<35} {'N':>5} {'Bias%':>7} {'AvgProb':>8} {'Score':>7} {'Comments':>8} {'KarmaRange':>10}")
    print("-" * 85)
    for _, row in outlet_df.head(30).iterrows():
        print(f"{row['outlet']:<35} {row['article_count']:>5} {row['bias_rate']*100:>6.1f}% {row['avg_bias_prob']:>7.3f} {row['avg_score']:>7.0f} {row['avg_comments']:>8.1f} {row['avg_karma_range']:>10.1f}")

    outlet_df.to_csv(output_dir / "outlet_analysis.csv", index=False)
    return outlet_df


def run_topic_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze bias patterns by topic tags."""
    print("\n" + "=" * 80)
    print("TOPIC-LEVEL ANALYSIS")
    print("=" * 80)

    # Explode tags
    tag_df = df[["article_id", "tags", "bias_label", "bias_prob", "score",
                  "num_comments", "avg_comment_karma", "karma_range"]].copy()
    tag_df["tag_list"] = tag_df["tags"].fillna("").str.split("|")
    tag_exploded = tag_df.explode("tag_list")
    tag_exploded = tag_exploded[tag_exploded["tag_list"].str.len() > 0]

    # Top tags
    tag_counts = tag_exploded["tag_list"].value_counts()
    top_tags = tag_counts[tag_counts >= 50].index

    tag_stats = []
    for tag in top_tags:
        tag_articles = tag_exploded[tag_exploded["tag_list"] == tag]
        tag_stats.append({
            "tag": tag,
            "article_count": len(tag_articles),
            "bias_rate": tag_articles["bias_label"].mean(),
            "avg_bias_prob": tag_articles["bias_prob"].mean(),
            "avg_score": tag_articles["score"].mean(),
            "avg_comments": tag_articles["num_comments"].mean(),
            "avg_comment_karma": tag_articles["avg_comment_karma"].mean(),
            "avg_karma_range": tag_articles["karma_range"].mean(),
        })

    tags_df = pd.DataFrame(tag_stats).sort_values("bias_rate", ascending=False)

    print(f"\nMost biased topics (min 50 articles):")
    print(f"{'Tag':<25} {'N':>5} {'Bias%':>7} {'AvgProb':>8} {'Score':>7} {'Comments':>8}")
    print("-" * 65)
    for _, row in tags_df.head(25).iterrows():
        print(f"{row['tag']:<25} {row['article_count']:>5} {row['bias_rate']*100:>6.1f}% {row['avg_bias_prob']:>7.3f} {row['avg_score']:>7.0f} {row['avg_comments']:>8.1f}")

    print(f"\nLeast biased topics (min 50 articles):")
    print(f"{'Tag':<25} {'N':>5} {'Bias%':>7} {'AvgProb':>8} {'Score':>7} {'Comments':>8}")
    print("-" * 65)
    for _, row in tags_df.tail(25).iterrows():
        print(f"{row['tag']:<25} {row['article_count']:>5} {row['bias_rate']*100:>6.1f}% {row['avg_bias_prob']:>7.3f} {row['avg_score']:>7.0f} {row['avg_comments']:>8.1f}")

    tags_df.to_csv(output_dir / "topic_analysis.csv", index=False)
    return tags_df


def run_temporal_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze bias trends over time."""
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)

    year_stats = df.groupby("year").agg(
        article_count=("bias_label", "size"),
        bias_rate=("bias_label", "mean"),
        avg_bias_prob=("bias_prob", "mean"),
        avg_score=("score", "mean"),
        avg_comments=("num_comments", "mean"),
        avg_comment_karma=("avg_comment_karma", "mean"),
        avg_karma_range=("karma_range", "mean"),
        avg_unique_commenters=("unique_commenters", "mean"),
    ).sort_index()

    print(f"\n{'Year':>6} {'N':>6} {'Bias%':>7} {'AvgProb':>8} {'Score':>7} {'Comments':>8} {'KarmaRange':>10} {'Commenters':>10}")
    print("-" * 75)
    for year, row in year_stats.iterrows():
        print(f"{int(year):>6} {row['article_count']:>6} {row['bias_rate']*100:>6.1f}% {row['avg_bias_prob']:>7.3f} {row['avg_score']:>7.0f} {row['avg_comments']:>8.1f} {row['avg_karma_range']:>10.1f} {row['avg_unique_commenters']:>10.1f}")

    year_stats.to_csv(output_dir / "temporal_analysis.csv")
    return year_stats


def run_predictive_modeling(df: pd.DataFrame, output_dir: Path):
    """Train classifiers to predict bias from interaction features."""
    print("\n" + "=" * 80)
    print("PREDICTIVE MODELING: Can interaction features predict bias?")
    print("=" * 80)

    available_features = [f for f in INTERACTION_FEATURES if f in df.columns]

    X = df[available_features].fillna(0).values
    y = df["bias_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, model in models.items():
        logger.info("Training %s...", name)

        # Cross-validation
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")
        auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

        results.append({
            "model": name,
            "accuracy_mean": acc_scores.mean(),
            "accuracy_std": acc_scores.std(),
            "f1_macro_mean": f1_scores.mean(),
            "f1_macro_std": f1_scores.std(),
            "roc_auc_mean": auc_scores.mean(),
            "roc_auc_std": auc_scores.std(),
        })

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
        print(f"  F1 Macro:  {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        print(f"  ROC AUC:   {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")

    # Feature importance from final Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\nFeature importances (Random Forest):")
    for _, row in importance_df.iterrows():
        bar = "#" * int(row["importance"] * 200)
        print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

    # Logistic regression coefficients
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    coef_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": lr.coef_[0],
    }).sort_values("coefficient", ascending=False)

    print(f"\nLogistic Regression coefficients:")
    for _, row in coef_df.iterrows():
        direction = "+" if row["coefficient"] > 0 else "-"
        print(f"  {direction} {row['feature']:<40} {row['coefficient']:>+.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_results.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importances.csv", index=False)
    coef_df.to_csv(output_dir / "lr_coefficients.csv", index=False)

    # Majority baseline
    majority = max(y.mean(), 1 - y.mean())
    print(f"\nBaseline (majority class): {majority:.4f}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run analysis on labeled dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = args.data_dir / "articles_with_features.parquet"
    logger.info("Loading data from %s...", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d articles with %d columns", len(df), len(df.columns))

    # Run all analyses
    stat_results = run_statistical_tests(df, RESULTS_DIR)
    corr_results = run_correlation_analysis(df, RESULTS_DIR)
    outlet_results = run_outlet_analysis(df, RESULTS_DIR)
    topic_results = run_topic_analysis(df, RESULTS_DIR)
    temporal_results = run_temporal_analysis(df, RESULTS_DIR)
    model_results = run_predictive_modeling(df, RESULTS_DIR)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
