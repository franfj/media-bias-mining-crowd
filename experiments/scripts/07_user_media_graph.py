#!/usr/bin/env python3
"""
Step 7: Bipartite graph analysis — users × media outlets.

Builds a bipartite graph (users → outlets) weighted by comment count,
then analyses:
- Community detection (Louvain on projected user graph)
- Outlet clustering by shared commenter base
- User polarisation: do users who comment on biased outlets form distinct communities?
- Cross-outlet mobility

Usage:
    python 07_user_media_graph.py
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("user_media_graph")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Minimum activity thresholds to filter noise
MIN_USER_COMMENTS = 10      # users with fewer comments are excluded
MIN_OUTLET_ARTICLES = 30    # outlets with fewer articles are excluded


def build_user_outlet_matrix(comments_path: Path, articles_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build user × outlet interaction matrix from comments, processing in chunks."""
    logger.info("Building user-outlet interaction matrix...")

    # Load article → outlet mapping
    schema_cols = [f.name for f in pq.read_schema(articles_path)]
    load = [c for c in ["article_id", "media", "bias_label", "bias_prob"] if c in schema_cols]
    articles = pd.read_parquet(articles_path, columns=load)
    article_to_outlet = dict(zip(articles["article_id"], articles["media"]))
    article_to_bias = dict(zip(articles["article_id"], articles["bias_label"]))

    # Count user-outlet interactions in chunks
    user_outlet_counts: Counter = Counter()
    user_article_bias: dict[str, list] = {}  # user → list of bias labels

    pf = pq.ParquetFile(comments_path)
    total = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=["article_id", "comment_author"]):
        df = batch.to_pandas()
        total += len(df)

        for _, row in df.iterrows():
            aid = row["article_id"]
            user = row["comment_author"]
            outlet = article_to_outlet.get(aid)
            bias = article_to_bias.get(aid)

            if outlet is not None:
                user_outlet_counts[(user, outlet)] += 1
            if bias is not None:
                user_article_bias.setdefault(user, []).append(bias)

        logger.info("  Processed %d comments...", total)

    # Build DataFrame
    rows = [{"user": u, "outlet": o, "comment_count": c} for (u, o), c in user_outlet_counts.items()]
    interactions_df = pd.DataFrame(rows)
    logger.info("Raw interactions: %d user-outlet pairs", len(interactions_df))

    # User bias profile
    user_profiles = []
    for user, labels in user_article_bias.items():
        arr = np.array(labels)
        user_profiles.append({
            "user": user,
            "total_articles_commented": len(arr),
            "bias_exposure_rate": arr.mean(),
            "n_biased_articles": int(arr.sum()),
            "n_non_biased_articles": int((1 - arr).sum()),
        })
    user_profiles_df = pd.DataFrame(user_profiles)

    return interactions_df, user_profiles_df


def analyse_outlet_similarity(interactions_df: pd.DataFrame, output_dir: Path):
    """Compute outlet similarity based on shared commenter base (Jaccard)."""
    print("\n" + "=" * 80)
    print("OUTLET SIMILARITY (Shared Commenter Base)")
    print("=" * 80)

    # Filter
    outlet_counts = interactions_df.groupby("outlet")["user"].nunique()
    valid_outlets = outlet_counts[outlet_counts >= MIN_OUTLET_ARTICLES].index
    filtered = interactions_df[interactions_df["outlet"].isin(valid_outlets)]

    # Build outlet → set of users
    outlet_users: dict[str, set] = {}
    for outlet, grp in filtered.groupby("outlet"):
        outlet_users[outlet] = set(grp["user"].unique())

    outlets = sorted(outlet_users.keys())
    n = len(outlets)
    logger.info("Computing Jaccard similarity for %d outlets...", n)

    # Jaccard similarity matrix
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s1, s2 = outlet_users[outlets[i]], outlet_users[outlets[j]]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            sim = inter / union if union > 0 else 0
            jaccard[i, j] = sim
            jaccard[j, i] = sim

    sim_df = pd.DataFrame(jaccard, index=outlets, columns=outlets)
    sim_df.to_csv(output_dir / "outlet_jaccard_similarity.csv")

    # Print top pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((outlets[i], outlets[j], jaccard[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop 20 most similar outlet pairs:")
    print(f"{'Outlet A':<30} {'Outlet B':<30} {'Jaccard':>8}")
    print("-" * 70)
    for a, b, sim in pairs[:20]:
        print(f"{a:<30} {b:<30} {sim:>8.4f}")

    return sim_df


def analyse_user_polarisation(user_profiles_df: pd.DataFrame, interactions_df: pd.DataFrame, output_dir: Path):
    """Analyse user polarisation based on bias exposure patterns."""
    print("\n" + "=" * 80)
    print("USER POLARISATION ANALYSIS")
    print("=" * 80)

    # Filter active users
    active = user_profiles_df[user_profiles_df["total_articles_commented"] >= MIN_USER_COMMENTS].copy()
    print(f"Active users (>= {MIN_USER_COMMENTS} comments): {len(active):,}")

    # Categorise users by bias exposure
    active["bias_category"] = pd.cut(
        active["bias_exposure_rate"],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=["low_bias", "mixed", "high_bias", "very_high_bias"],
        include_lowest=True,
    )

    print(f"\nUser distribution by bias exposure:")
    for cat, count in active["bias_category"].value_counts().sort_index().items():
        pct = count / len(active) * 100
        print(f"  {cat:<20} {count:>6,} ({pct:.1f}%)")

    # Per-category outlet diversity
    user_outlet_diversity = interactions_df.groupby("user")["outlet"].nunique().reset_index()
    user_outlet_diversity.columns = ["user", "outlet_diversity"]
    active = active.merge(user_outlet_diversity, on="user", how="left")

    print(f"\nOutlet diversity by bias exposure:")
    print(f"{'Category':<20} {'AvgOutlets':>10} {'MedianOutlets':>13} {'AvgArticles':>11}")
    print("-" * 58)
    for cat in ["low_bias", "mixed", "high_bias", "very_high_bias"]:
        subset = active[active["bias_category"] == cat]
        if len(subset) > 0:
            print(f"{cat:<20} {subset['outlet_diversity'].mean():>10.1f} {subset['outlet_diversity'].median():>13.0f} {subset['total_articles_commented'].mean():>11.1f}")

    # Save
    active.to_csv(output_dir / "user_polarisation.csv", index=False)

    # Summary stats
    stats = active.groupby("bias_category").agg(
        n_users=("user", "count"),
        avg_bias_rate=("bias_exposure_rate", "mean"),
        avg_articles=("total_articles_commented", "mean"),
        avg_outlets=("outlet_diversity", "mean"),
    )
    stats.to_csv(output_dir / "user_polarisation_summary.csv")

    return active


def analyse_community_detection(interactions_df: pd.DataFrame, user_profiles_df: pd.DataFrame, output_dir: Path):
    """Detect communities in the user-outlet bipartite graph using Louvain."""
    print("\n" + "=" * 80)
    print("COMMUNITY DETECTION (Louvain on outlet projection)")
    print("=" * 80)

    try:
        import networkx as nx
        from networkx.algorithms import bipartite
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        print("  networkx not installed — skipping community detection")
        return

    # Filter to manageable size: top users and outlets
    outlet_counts = interactions_df.groupby("outlet")["comment_count"].sum()
    top_outlets = outlet_counts.nlargest(50).index

    user_counts = interactions_df.groupby("user")["comment_count"].sum()
    top_users = user_counts.nlargest(5000).index

    filtered = interactions_df[
        (interactions_df["outlet"].isin(top_outlets)) &
        (interactions_df["user"].isin(top_users))
    ]
    logger.info("Filtered graph: %d edges, %d users, %d outlets",
                len(filtered), filtered["user"].nunique(), filtered["outlet"].nunique())

    # Build bipartite graph
    B = nx.Graph()
    outlets_set = set()
    users_set = set()

    for _, row in filtered.iterrows():
        u = f"u_{row['user']}"
        o = f"o_{row['outlet']}"
        B.add_edge(u, o, weight=row["comment_count"])
        users_set.add(u)
        outlets_set.add(o)

    # Project onto outlets
    outlet_graph = bipartite.weighted_projected_graph(B, outlets_set)
    logger.info("Outlet projection: %d nodes, %d edges", outlet_graph.number_of_nodes(), outlet_graph.number_of_edges())

    # Louvain communities
    communities = louvain_communities(outlet_graph, weight="weight", seed=42)

    print(f"\nDetected {len(communities)} outlet communities:")
    for i, comm in enumerate(communities):
        outlets = sorted([n.replace("o_", "") for n in comm])
        print(f"\n  Community {i + 1} ({len(outlets)} outlets):")
        for o in outlets:
            print(f"    - {o}")

    # Save community assignments
    comm_rows = []
    for i, comm in enumerate(communities):
        for node in comm:
            comm_rows.append({"outlet": node.replace("o_", ""), "community": i})
    comm_df = pd.DataFrame(comm_rows)
    comm_df.to_csv(output_dir / "outlet_communities.csv", index=False)

    # Project onto users (smaller sample for speed)
    top_users_small = user_counts.nlargest(2000).index
    filtered_small = interactions_df[
        (interactions_df["outlet"].isin(top_outlets)) &
        (interactions_df["user"].isin(top_users_small))
    ]

    B2 = nx.Graph()
    users_set2 = set()
    for _, row in filtered_small.iterrows():
        u = f"u_{row['user']}"
        o = f"o_{row['outlet']}"
        B2.add_edge(u, o, weight=row["comment_count"])
        users_set2.add(u)

    user_graph = bipartite.weighted_projected_graph(B2, users_set2)
    user_communities = louvain_communities(user_graph, weight="weight", seed=42)

    print(f"\nDetected {len(user_communities)} user communities (top 2000 users)")

    # Characterise user communities by bias exposure
    user_bias = dict(zip(user_profiles_df["user"], user_profiles_df["bias_exposure_rate"]))
    for i, comm in enumerate(user_communities):
        users = [n.replace("u_", "") for n in comm]
        rates = [user_bias.get(u, np.nan) for u in users]
        rates = [r for r in rates if not np.isnan(r)]
        if rates:
            print(f"  Community {i + 1}: {len(comm)} users, avg bias exposure: {np.mean(rates):.3f}")

    print(f"\nCommunity results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="User-media bipartite graph analysis")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comments_path = args.data_dir / "comments_raw.parquet"
    articles_path = args.data_dir / "articles_with_features.parquet"

    interactions_df, user_profiles_df = build_user_outlet_matrix(comments_path, articles_path)

    # Save intermediate data
    interactions_df.to_parquet(args.data_dir / "user_outlet_interactions.parquet", index=False)
    user_profiles_df.to_parquet(args.data_dir / "user_profiles.parquet", index=False)

    analyse_outlet_similarity(interactions_df, RESULTS_DIR)
    analyse_user_polarisation(user_profiles_df, interactions_df, RESULTS_DIR)
    analyse_community_detection(interactions_df, user_profiles_df, RESULTS_DIR)

    print("\n" + "=" * 80)
    print("GRAPH ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
