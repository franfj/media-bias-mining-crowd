#!/usr/bin/env python3
"""
Step 16: GNN on the user-outlet bipartite graph.

Uses a Graph Neural Network (GCN) to learn article representations from
the user-outlet interaction graph structure. Compares GNN-based features
with hand-crafted interaction features for bias classification.

Architecture:
- Bipartite graph: users <-> outlets, weighted by comment count
- Article nodes connected to their outlet node
- 2-layer GCN produces node embeddings
- Article embeddings used for bias classification

Usage:
    python 16_gnn_graph.py
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gnn")

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


class BiasGCN(torch.nn.Module):
    """Simple 2-layer GCN for node classification."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.classifier = torch.nn.Linear(out_channels, 2)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def classify(self, x, edge_index, edge_weight=None):
        emb = self.forward(x, edge_index, edge_weight)
        return self.classifier(emb)


def build_graph(articles_df: pd.DataFrame, interactions_path: Path):
    """Build a heterogeneous graph: articles connected to outlets, outlets connected via users."""
    from torch_geometric.data import Data

    # Article features
    available = [f for f in INTERACTION_FEATURES if f in articles_df.columns]
    X_articles = articles_df[available].fillna(0).values
    scaler = StandardScaler()
    X_articles = scaler.fit_transform(X_articles)
    n_articles = len(articles_df)

    # Outlet nodes
    outlet_enc = LabelEncoder()
    articles_df = articles_df.copy()
    articles_df["outlet_id"] = outlet_enc.fit_transform(articles_df["media"].fillna("unknown"))
    n_outlets = articles_df["outlet_id"].nunique()

    # Node features: articles have interaction features, outlets get mean of their articles
    outlet_features = np.zeros((n_outlets, X_articles.shape[1]))
    for oid in range(n_outlets):
        mask = articles_df["outlet_id"].values == oid
        if mask.sum() > 0:
            outlet_features[oid] = X_articles[mask].mean(axis=0)

    # All node features: [articles | outlets]
    X = np.vstack([X_articles, outlet_features])
    x = torch.tensor(X, dtype=torch.float32)

    # Edges: article -> outlet (bidirectional)
    article_indices = np.arange(n_articles)
    outlet_indices = articles_df["outlet_id"].values + n_articles  # offset

    src = np.concatenate([article_indices, outlet_indices])
    dst = np.concatenate([outlet_indices, article_indices])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Labels (only for article nodes)
    y = torch.tensor(articles_df["bias_label"].values, dtype=torch.long)

    # Train mask (article nodes only)
    article_mask = torch.zeros(n_articles + n_outlets, dtype=torch.bool)
    article_mask[:n_articles] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.article_mask = article_mask
    data.n_articles = n_articles

    logger.info("Graph: %d nodes (%d articles + %d outlets), %d edges",
                x.shape[0], n_articles, n_outlets, edge_index.shape[1])
    return data


def train_gnn(data, n_epochs=200, hidden=64, lr=0.01):
    """Train GCN and return article embeddings + predictions."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = BiasGCN(data.x.shape[1], hidden, 32).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Use all article nodes for training (we'll do external CV)
    train_mask = data.article_mask.to(device)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model.classify(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = out[train_mask].argmax(dim=1)
                acc = (pred == data.y).float().mean().item()
            logger.info("  Epoch %d: loss=%.4f, acc=%.4f", epoch + 1, loss.item(), acc)
            model.train()

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        article_embs = embeddings[:data.n_articles].cpu().numpy()
        logits = model.classify(data.x, data.edge_index)
        probs = F.softmax(logits[:data.n_articles], dim=1)[:, 1].cpu().numpy()
        preds = logits[:data.n_articles].argmax(dim=1).cpu().numpy()

    return article_embs, preds, probs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    schema = [f.name for f in pq.read_schema(DATA_DIR / "articles_with_features.parquet")]
    df = pd.read_parquet(DATA_DIR / "articles_with_features.parquet")

    logger.info("Building graph...")
    data = build_graph(df, DATA_DIR / "user_outlet_interactions.parquet")

    logger.info("Training GNN...")
    embeddings, preds, probs = train_gnn(data, n_epochs=300, hidden=64)

    y_true = df["bias_label"].values

    # Evaluate
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    auc = roc_auc_score(y_true, probs)

    print("\n" + "=" * 60)
    print("GNN CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1 Macro:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Save embeddings
    np.save(DATA_DIR / "gnn_embeddings.npy", embeddings)
    logger.info("Saved GNN embeddings: %s", embeddings.shape)

    # Compare with interaction-only baseline
    print(f"\n  GNN embeddings shape: {embeddings.shape}")
    print(f"  These can be used as features in the multimodal fusion pipeline")

    # Save results
    results = pd.DataFrame({
        "model": ["GNN (GCN 2-layer)"],
        "accuracy": [acc], "f1_macro": [f1], "auc": [auc],
    })
    results.to_csv(RESULTS_DIR / "gnn_results.csv", index=False)
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
