#!/usr/bin/env python3
"""
Upload the Meneame media bias dataset to Zenodo.

Creates a new deposit (or updates an existing one) with:
- articles_with_features.parquet (main analysis dataset)
- articles_labeled.parquet (bias-labeled articles)
- karma_features.parquet (advanced karma features per article)
- comments_with_sentiment.parquet (sentiment-analyzed comment sample)
- user_profiles.parquet (user bias exposure profiles)
- user_outlet_interactions.parquet (user-outlet graph data)

Usage:
    python upload_zenodo.py --token YOUR_ZENODO_TOKEN
    python upload_zenodo.py --token YOUR_ZENODO_TOKEN --deposit-id 12345  # update existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Files to upload (relative to DATA_DIR)
UPLOAD_FILES = [
    "articles_with_features.parquet",
    "articles_labeled.parquet",
    "karma_features.parquet",
    "comments_with_sentiment.parquet",
    "user_profiles.parquet",
    "user_outlet_interactions.parquet",
]

METADATA = {
    "metadata": {
        "title": "Meneame Media Bias Dataset: Interaction Features and Bias Labels",
        "upload_type": "dataset",
        "description": (
            "<p>A processed dataset of news articles submitted to Meneame (Spanish social news aggregator) "
            "with automatic media bias labels and rich interaction features derived from user comments.</p>"
            "<h3>Contents</h3>"
            "<ul>"
            "<li><strong>articles_with_features.parquet</strong>: 14,995 articles with 38 columns including "
            "bias labels (from DistilBERT trained on MBBMD), interaction features (karma statistics, "
            "comment engagement metrics), and metadata (outlet, tags, timestamp).</li>"
            "<li><strong>articles_labeled.parquet</strong>: Articles with bias probability scores.</li>"
            "<li><strong>karma_features.parquet</strong>: Advanced karma distribution features per article "
            "(entropy, Gini, bimodality, skewness) for 183K+ articles.</li>"
            "<li><strong>comments_with_sentiment.parquet</strong>: 20K comment sample with sentiment "
            "(POS/NEG/NEU) and emotion (joy, anger, sadness, fear) scores from pysentimiento/robertuito.</li>"
            "<li><strong>user_profiles.parquet</strong>: User-level bias exposure metrics.</li>"
            "<li><strong>user_outlet_interactions.parquet</strong>: Bipartite graph data (user-outlet comment counts).</li>"
            "</ul>"
            "<h3>Pipeline</h3>"
            "<p>Data was collected from meneame.net (2005-2021), processed through a 5-step pipeline: "
            "ingestion, filtering, automatic bias labeling (franfj/fdtd_media_bias_E), interaction feature "
            "extraction, and statistical analysis. See the GitHub repository for full reproducibility.</p>"
            "<h3>Key Statistics</h3>"
            "<ul>"
            "<li>14,995 articles from 2,868 media outlets</li>"
            "<li>13.2M comments from 96K unique users</li>"
            "<li>61.5% articles labeled as biased (automatic labeling)</li>"
            "<li>Timespan: 2005-2021</li>"
            "</ul>"
        ),
        "creators": [
            {"name": "Rodrigo-Ginés, Francisco-Javier", "affiliation": "UNED", "orcid": "0000-0002-5077-2990"},
            {"name": "Chamorro-Padial, Jorge", "affiliation": "Universitat de Lleida"},
            {"name": "Carrillo-de-Albornoz, Jorge", "affiliation": "UNED"},
            {"name": "Plaza, Laura", "affiliation": "UNED"},
        ],
        "keywords": [
            "media bias",
            "meneame",
            "social news",
            "NLP",
            "user interactions",
            "sentiment analysis",
            "Spanish",
            "disinformation",
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/franfj/media-bias-mining-crowd",
                "relation": "isSupplementTo",
                "scheme": "url",
            },
            {
                "identifier": "10.5281/zenodo.14806064",
                "relation": "isDerivedFrom",
                "scheme": "doi",
            },
        ],
        "license": "cc-by-4.0",
        "language": "spa",
        "version": "0.1.0",
        "notes": (
            "This is a preliminary release. Bias labels are automatic (DistilBERT trained on MBBMD). "
            "Future versions will include improved labels from a cross-lingual XLM-RoBERTa model "
            "trained on BEADS + MBIC + MBBMD."
        ),
    }
}


def create_deposit(token: str) -> dict:
    """Create a new Zenodo deposit."""
    print("[ZENODO] Creating new deposit...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions",
        params={"access_token": token},
        json={},
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 201:
        print(f"[ERROR] Failed to create deposit: {r.status_code} {r.text}")
        sys.exit(1)
    dep = r.json()
    print(f"[ZENODO] Created deposit ID: {dep['id']}")
    print(f"[ZENODO] Preview URL: {dep['links']['html']}")
    return dep


def get_deposit(token: str, deposit_id: int) -> dict:
    """Get existing deposit to update."""
    print(f"[ZENODO] Fetching existing deposit {deposit_id}...")

    # First try to create a new version if it's published
    r = requests.get(
        f"{ZENODO_API}/deposit/depositions/{deposit_id}",
        params={"access_token": token},
    )
    if r.status_code != 200:
        print(f"[ERROR] Failed to get deposit: {r.status_code} {r.text}")
        sys.exit(1)

    dep = r.json()

    # If already published, create new version
    if dep.get("submitted"):
        print("[ZENODO] Deposit is published. Creating new version...")
        r = requests.post(
            dep["links"]["newversion"],
            params={"access_token": token},
        )
        if r.status_code != 201:
            print(f"[ERROR] Failed to create new version: {r.status_code} {r.text}")
            sys.exit(1)
        new_dep = r.json()
        # Get the draft of the new version
        r = requests.get(
            new_dep["links"]["latest_draft"],
            params={"access_token": token},
        )
        dep = r.json()
        print(f"[ZENODO] New version draft ID: {dep['id']}")

    return dep


def upload_file(token: str, bucket_url: str, filepath: Path):
    """Upload a single file to the deposit bucket."""
    filename = filepath.name
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"[UPLOAD] {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)

    with open(filepath, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{filename}",
            params={"access_token": token},
            data=f,
        )

    if r.status_code in (200, 201):
        print("OK")
    else:
        print(f"FAILED ({r.status_code}: {r.text[:200]})")
        return False
    return True


def update_metadata(token: str, deposit_id: int):
    """Update deposit metadata."""
    print("[ZENODO] Updating metadata...")
    r = requests.put(
        f"{ZENODO_API}/deposit/depositions/{deposit_id}",
        params={"access_token": token},
        json=METADATA,
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 200:
        print(f"[ERROR] Failed to update metadata: {r.status_code} {r.text}")
        return False
    print("[ZENODO] Metadata updated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Zenodo")
    parser.add_argument("--token", required=True, help="Zenodo API token")
    parser.add_argument("--deposit-id", type=int, default=None,
                        help="Existing deposit ID to update")
    parser.add_argument("--publish", action="store_true",
                        help="Publish the deposit (makes it permanent!)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Create or get deposit
    if args.deposit_id:
        dep = get_deposit(args.token, args.deposit_id)
    else:
        dep = create_deposit(args.token)

    deposit_id = dep["id"]
    bucket_url = dep["links"]["bucket"]

    # Upload files
    print(f"\n--- Uploading files to deposit {deposit_id} ---\n")
    all_ok = True
    for filename in UPLOAD_FILES:
        filepath = args.data_dir / filename
        if not filepath.exists():
            print(f"[SKIP] {filename} — not found")
            continue
        if not upload_file(args.token, bucket_url, filepath):
            all_ok = False

    # Update metadata
    print()
    update_metadata(args.token, deposit_id)

    # Optionally publish
    if args.publish and all_ok:
        print("\n[ZENODO] Publishing deposit...")
        r = requests.post(
            f"{ZENODO_API}/deposit/depositions/{deposit_id}/actions/publish",
            params={"access_token": args.token},
        )
        if r.status_code == 202:
            doi = r.json().get("doi", "unknown")
            print(f"[ZENODO] Published! DOI: {doi}")
            print(f"[ZENODO] URL: https://doi.org/{doi}")
        else:
            print(f"[ERROR] Publish failed: {r.status_code} {r.text}")
    else:
        print(f"\n[ZENODO] Deposit saved as DRAFT (not published)")
        print(f"[ZENODO] Preview: https://zenodo.org/deposit/{deposit_id}")
        print(f"[ZENODO] To publish, re-run with --publish or publish from the web UI")
        print(f"[ZENODO] Deposit ID: {deposit_id} (use --deposit-id {deposit_id} to update later)")


if __name__ == "__main__":
    main()
