#!/usr/bin/env python3
"""
10_train_bias_model.py — Fine-tune XLM-RoBERTa-base for binary media bias detection.

Combines three datasets for cross-lingual training:
  - MBBMD (Spanish, ~100 articles, majority-vote labels)
  - BEADS/BEAD (English, ~1000 articles, from HuggingFace)
  - MBIC (English, sentence-level, from local Excel file)

The model is trained on all data and evaluated specifically on the
held-out MBBMD test split (target domain: Spanish news).

Usage:
    python 10_train_bias_model.py [--epochs 5] [--batch_size 8] [--lr 2e-5]

Output:
    experiments/models/xlm-roberta-bias/   (saved model + tokenizer)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # meneame-media-bias/
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models" / "xlm-roberta-bias"

MBBMD_PATH = Path(
    "/Users/franfj/Desktop/AI Factory/papers/projects/"
    "eswa-hierarchical-bias/experiments/data/raw/mbbmd/MBBMD_complete.json"
)

# MBIC: try several possible locations
MBIC_SEARCH_PATHS = [
    Path("/Users/franfj/Desktop/AI Factory/papers/projects/"
         "eswa-hierarchical-bias/experiments/data/raw/mbic/labeled_dataset.xlsx"),
    Path("/Users/franfj/Desktop/AI Factory/papers/projects/"
         "eswa-hierarchical-bias/experiments/data/raw/mbic/MBIC.csv"),
    EXPERIMENTS_DIR / "data" / "raw" / "mbic" / "labeled_dataset.xlsx",
]

DEFAULT_MODEL = "distilbert-base-multilingual-cased"
MAX_LENGTH = 512
SEED = 42


# ---------------------------------------------------------------------------
# Dataset loading functions
# ---------------------------------------------------------------------------

def load_mbbmd(path: Path) -> tuple[list[dict], list[dict]]:
    """Load MBBMD and split into train+control vs test based on original splits.

    Returns:
        (trainable_samples, test_samples) where each sample is
        {"text": str, "label": int, "source": "mbbmd", "doc_id": str}
    """
    if not path.exists():
        print(f"[MBBMD] ERROR: File not found at {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    trainable, test = [], []
    for doc in data["documents"]:
        text = doc["full_text"].strip()
        if not text:
            continue

        # Majority vote on is_biased
        votes = [ann["is_biased"] for ann in doc["annotations"]]
        label = 1 if sum(votes) > len(votes) / 2 else 0

        sample = {
            "text": text,
            "label": label,
            "source": "mbbmd",
            "doc_id": doc["text_id"],
        }

        if doc["split"] == "test":
            test.append(sample)
        else:
            # train + control go into trainable pool
            trainable.append(sample)

    print(f"[MBBMD] Loaded {len(trainable)} trainable + {len(test)} test samples")
    _print_label_dist("MBBMD trainable", trainable)
    _print_label_dist("MBBMD test", test)
    return trainable, test


def load_beads() -> list[dict]:
    """Load BEADS/BEAD dataset from HuggingFace.

    HuggingFace dataset: shainar/BEAD, config=Full_Annotations, split=full
    Labels: "Neutral" -> 0, "Slightly Biased"/"Highly Biased" -> 1
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[BEADS] ERROR: `datasets` library not installed. pip install datasets")
        return []

    print("[BEADS] Loading from HuggingFace (shainar/BEAD)...")
    try:
        ds = load_dataset("shainar/BEAD", "Full_Annotations", split="full")
    except Exception as e:
        print(f"[BEADS] WARNING: Could not load dataset: {e}")
        print("[BEADS] Skipping BEADS dataset.")
        return []

    samples = []
    for i, row in enumerate(ds):
        text = row.get("text") or ""
        text = text.strip()
        if not text:
            continue
        label_str = row.get("label", "Neutral")
        if label_str is None:
            continue
        label = 0 if label_str == "Neutral" else 1  # Slightly/Highly Biased -> 1

        samples.append({
            "text": text,
            "label": label,
            "source": "beads",
            "doc_id": f"beads_{i}",
        })

    # Subsample BEADS to avoid overwhelming small datasets (MBBMD)
    MAX_BEADS = 5000
    if len(samples) > MAX_BEADS:
        import random
        random.seed(SEED)
        random.shuffle(samples)
        samples = samples[:MAX_BEADS]
        print(f"[BEADS] Subsampled to {MAX_BEADS} (from {len(ds)} raw)")

    print(f"[BEADS] Loaded {len(samples)} samples")
    _print_label_dist("BEADS", samples)
    return samples


def load_mbic() -> list[dict]:
    """Load MBIC dataset from local Excel/CSV file.

    Expected columns: 'sentence' (or 'text'), 'Label_bias' ("Biased"/"Non-biased")
    """
    mbic_path = None
    for p in MBIC_SEARCH_PATHS:
        if p.exists():
            mbic_path = p
            break

    if mbic_path is None:
        print("[MBIC] WARNING: Dataset not found at any of these paths:")
        for p in MBIC_SEARCH_PATHS:
            print(f"  - {p}")
        print("[MBIC] Skipping MBIC dataset. Download from:")
        print("  https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE")
        return []

    print(f"[MBIC] Loading from {mbic_path}")
    try:
        if mbic_path.suffix == ".xlsx":
            df = pd.read_excel(mbic_path)
        else:
            df = pd.read_csv(mbic_path)
    except Exception as e:
        print(f"[MBIC] ERROR reading file: {e}")
        return []

    # Find text column
    text_col = None
    for col in ["sentence", "text", "content"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        print(f"[MBIC] ERROR: No text column found. Columns: {list(df.columns)}")
        return []

    # Find label column
    label_col = None
    for col in ["Label_bias", "label_bias", "label", "bias"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        print(f"[MBIC] ERROR: No label column found. Columns: {list(df.columns)}")
        return []

    samples = []
    for i, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or text == "nan":
            continue

        label_str = str(row[label_col]).strip()
        if label_str in ("Biased", "biased", "1"):
            label = 1
        elif label_str in ("Non-biased", "non-biased", "Non_biased", "0"):
            label = 0
        else:
            continue  # skip ambiguous labels

        samples.append({
            "text": text,
            "label": label,
            "source": "mbic",
            "doc_id": f"mbic_{i}",
        })

    print(f"[MBIC] Loaded {len(samples)} samples")
    _print_label_dist("MBIC", samples)
    return samples


def _print_label_dist(name: str, samples: list[dict]) -> None:
    """Print label distribution for a dataset."""
    counts = Counter(s["label"] for s in samples)
    total = len(samples)
    if total == 0:
        print(f"  {name}: empty")
        return
    print(f"  {name}: {counts[0]} non-biased ({counts[0]/total:.1%}), "
          f"{counts[1]} biased ({counts[1]/total:.1%})")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_datasets(
    mbbmd_trainable: list[dict],
    mbbmd_test: list[dict],
    beads_samples: list[dict],
    mbic_samples: list[dict],
    tokenizer,
    val_ratio: float = 0.2,
) -> tuple[Dataset, Dataset, Dataset]:
    """Combine all data, create stratified train/val split, and tokenize.

    Returns: (train_dataset, val_dataset, mbbmd_test_dataset)
    """
    # Combine all trainable data
    all_train_samples = mbbmd_trainable + beads_samples + mbic_samples
    print(f"\n[DATA] Total trainable samples: {len(all_train_samples)}")
    _print_label_dist("Combined trainable", all_train_samples)

    if len(all_train_samples) == 0:
        print("[DATA] ERROR: No training data available!")
        sys.exit(1)

    # Stratified train/val split
    texts = [s["text"] for s in all_train_samples]
    labels = [s["label"] for s in all_train_samples]
    sources = [s["source"] for s in all_train_samples]

    train_texts, val_texts, train_labels, val_labels, train_sources, val_sources = (
        train_test_split(
            texts, labels, sources,
            test_size=val_ratio,
            stratify=labels,
            random_state=SEED,
        )
    )

    print(f"[DATA] Train: {len(train_texts)}, Val: {len(val_texts)}, "
          f"MBBMD test: {len(mbbmd_test)}")
    _print_label_dist("Train split", [{"label": l} for l in train_labels])
    _print_label_dist("Val split", [{"label": l} for l in val_labels])

    # Tokenize
    def tokenize(texts_list, labels_list, sources_list=None):
        encodings = tokenizer(
            texts_list,
            truncation=True,
            padding=False,  # dynamic padding via data collator
            max_length=MAX_LENGTH,
        )
        ds_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels_list,
        }
        if sources_list is not None:
            ds_dict["source"] = sources_list
        return Dataset.from_dict(ds_dict)

    train_ds = tokenize(train_texts, train_labels, train_sources)
    val_ds = tokenize(val_texts, val_labels, val_sources)

    test_texts = [s["text"] for s in mbbmd_test]
    test_labels = [s["label"] for s in mbbmd_test]
    test_sources = [s["source"] for s in mbbmd_test]
    test_ds = tokenize(test_texts, test_labels, test_sources)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute F1, accuracy, precision, recall for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    f1_biased = f1_score(labels, preds, average="binary", pos_label=1)
    accuracy = (preds == labels).mean()
    return {
        "f1_macro": f1,
        "f1_biased": f1_biased,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = []
    for i in range(n_classes):
        w = total / (n_classes * counts.get(i, 1))
        weights.append(w)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"[WEIGHTS] Class weights: non-biased={weights[0]:.3f}, biased={weights[1]:.3f}")
    return weights_tensor


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the loss function."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[DEVICE] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("[DEVICE] Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("[DEVICE] Using CPU (training will be slow)")
    return device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_per_source(trainer, dataset: Dataset, dataset_name: str) -> dict:
    """Run evaluation and print classification report."""
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print(f"\n{'='*60}")
    print(f"  Evaluation: {dataset_name}")
    print(f"{'='*60}")
    print(classification_report(
        labels, preds,
        target_names=["Non-biased", "Biased"],
        digits=4,
        zero_division=0,
    ))

    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_biased": f1_score(labels, preds, average="binary", pos_label=1),
        "accuracy": (preds == labels).mean(),
    }


def evaluate_val_by_source(trainer, val_dataset: Dataset) -> None:
    """Break down validation results by source dataset."""
    if "source" not in val_dataset.column_names:
        return

    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    sources = val_dataset["source"]

    unique_sources = sorted(set(sources))
    print(f"\n{'='*60}")
    print("  Per-source validation breakdown")
    print(f"{'='*60}")

    for src in unique_sources:
        mask = [s == src for s in sources]
        src_preds = preds[mask]
        src_labels = np.array(labels)[mask]
        if len(src_labels) == 0:
            continue
        f1 = f1_score(src_labels, src_preds, average="macro", zero_division=0)
        acc = (src_preds == src_labels).mean()
        print(f"  {src:>8s}: n={len(src_labels):>5d}, F1={f1:.4f}, Acc={acc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MAX_LENGTH, MODEL_NAME
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual model for binary media bias detection"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name (default: distilbert-base-multilingual-cased)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR), help="Output dir")
    parser.add_argument("--skip_beads", action="store_true", help="Skip BEADS dataset")
    parser.add_argument("--skip_mbic", action="store_true", help="Skip MBIC dataset")
    args = parser.parse_args()

    MAX_LENGTH = args.max_length
    MODEL_NAME = args.model

    print("=" * 60)
    print("  XLM-RoBERTa Media Bias Detection — Training Script")
    print("=" * 60)

    # Detect device
    device = get_device()

    # ---------------------------------------------------------------
    # 1. Load datasets
    # ---------------------------------------------------------------
    print("\n--- Loading Datasets ---\n")

    mbbmd_trainable, mbbmd_test = load_mbbmd(MBBMD_PATH)

    beads_samples = [] if args.skip_beads else load_beads()
    mbic_samples = [] if args.skip_mbic else load_mbic()

    total = len(mbbmd_trainable) + len(beads_samples) + len(mbic_samples)
    print(f"\n[SUMMARY] Total trainable samples: {total}")
    if total == 0:
        print("[ERROR] No training data! Check dataset paths.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 2. Tokenize and prepare splits
    # ---------------------------------------------------------------
    print("\n--- Tokenizing ---\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, test_ds = prepare_datasets(
        mbbmd_trainable, mbbmd_test, beads_samples, mbic_samples,
        tokenizer, val_ratio=args.val_ratio,
    )

    # Remove 'source' column before training (Trainer doesn't expect it)
    train_ds_clean = train_ds.remove_columns(["source"])
    val_ds_clean = val_ds.remove_columns(["source"])
    test_ds_clean = test_ds.remove_columns(["source"])

    # ---------------------------------------------------------------
    # 3. Compute class weights
    # ---------------------------------------------------------------
    all_train_labels = (
        [s["label"] for s in mbbmd_trainable]
        + [s["label"] for s in beads_samples]
        + [s["label"] for s in mbic_samples]
    )
    class_weights = compute_class_weights(all_train_labels)

    # ---------------------------------------------------------------
    # 4. Load model
    # ---------------------------------------------------------------
    print(f"\n[MODEL] Loading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Non-biased", 1: "Biased"},
        label2id={"Non-biased": 0, "Biased": 1},
    )

    # ---------------------------------------------------------------
    # 5. Training arguments
    # ---------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    # Determine fp16/bf16 based on device
    use_fp16 = device == "cuda"
    # MPS: use fp32 (fp16 on MPS can cause issues with some ops)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        # Evaluation strategy
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        # Performance
        fp16=use_fp16,
        dataloader_num_workers=0,  # safer on macOS
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
        # Reproducibility
        seed=SEED,
        data_seed=SEED,
    )

    # ---------------------------------------------------------------
    # 6. Initialize Trainer
    # ---------------------------------------------------------------
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds_clean,
        eval_dataset=val_ds_clean,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    # ---------------------------------------------------------------
    # 7. Train
    # ---------------------------------------------------------------
    print("\n--- Training ---\n")
    train_result = trainer.train()
    print(f"\n[TRAIN] Completed in {train_result.metrics['train_runtime']:.1f}s")
    print(f"[TRAIN] Train loss: {train_result.metrics['train_loss']:.4f}")

    # ---------------------------------------------------------------
    # 8. Save final model
    # ---------------------------------------------------------------
    print(f"\n[SAVE] Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ---------------------------------------------------------------
    # 9. Evaluation
    # ---------------------------------------------------------------
    print("\n--- Evaluation ---\n")

    # Primary evaluation: MBBMD test set (target domain)
    mbbmd_results = evaluate_per_source(trainer, test_ds_clean, "MBBMD Test (Target Domain)")

    # Validation set overall
    val_results = evaluate_per_source(trainer, val_ds_clean, "Validation Set (All Sources)")

    # Per-source validation breakdown
    evaluate_val_by_source(trainer, val_ds)

    # ---------------------------------------------------------------
    # 10. Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.grad_accum} accum = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Max length:     {MAX_LENGTH}")
    print(f"  Device:         {device}")
    print(f"  Train samples:  {len(train_ds)}")
    print(f"  Val samples:    {len(val_ds)}")
    print(f"  Test samples:   {len(test_ds)} (MBBMD)")
    print(f"  ---")
    print(f"  MBBMD Test F1 (macro):  {mbbmd_results['f1_macro']:.4f}")
    print(f"  MBBMD Test F1 (biased): {mbbmd_results['f1_biased']:.4f}")
    print(f"  MBBMD Test Accuracy:    {mbbmd_results['accuracy']:.4f}")
    print(f"  Val F1 (macro):         {val_results['f1_macro']:.4f}")
    print(f"  ---")
    print(f"  Model saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
