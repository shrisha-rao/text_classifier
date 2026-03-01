#!/usr/bin/env python3
"""
Benchmark script for zero‑shot text classification models.

Usage:
    python benchmark.py --bi_model path/to/bi_model --poly_model path/to/poly_model \
                        --test_data data/test.json [--threshold 0.5]
"""

import argparse
import json
import time
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from dataset import ZeroShotDataset
from model import BiEncoderModel  #, PolyencoderModel
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["labels"] for item in batch]
    targets = [item["targets"] for item in batch]
    return {"texts": texts, "labels": labels, "targets": targets}


def load_test_data(path):
    """Load test samples from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Ensure each sample has 'text' and 'labels' keys
    return data


def get_global_labels(test_data):
    """Extract sorted list of all unique labels in the test set."""
    labels = set()
    for sample in test_data:
        labels.update(sample["labels"])
    return sorted(labels)


import numpy as np
from sklearn.metrics import f1_score


def compute_optimal_threshold(y_true, probs, num_steps=100):
    thresholds = np.linspace(0, 1, num_steps)
    best_thresh = 0.0
    best_f1 = 0.0
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')  # or 'micro'
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Optimal threshold: {best_thresh:.3f}, F1: {best_f1:.3f}")
    return best_thresh, best_f1


import torch
import time
from sklearn.metrics import roc_auc_score


def benchmark_model(model, dataloader, device, threshold=0.5):
    """
    Benchmark model using:
      - Micro Precision
      - Micro Recall
      - Micro F1
      - AUC

    Metrics computed only over each text's candidate labels.
    """

    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        for batch in dataloader:

            texts = batch["texts"]
            labels = batch["labels"]
            targets_list = batch["targets"]

            # ---- Accurate GPU timing ----
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            scores, mask = model(texts, labels)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            inference_times.append(end - start)

            # ---- Build target tensor ----
            target_tensor = torch.zeros_like(scores, device=device)

            for i, tgt_list in enumerate(targets_list):
                max_labels = min(len(tgt_list), model.max_num_labels)
                target_tensor[i, :max_labels] = torch.tensor(
                    tgt_list[:max_labels], device=device)

            # Keep only valid (non-padded) positions
            valid_scores = scores[mask]
            valid_targets = target_tensor[mask]

            probs = torch.sigmoid(valid_scores)

            all_probs.append(probs.cpu())
            all_targets.append(valid_targets.cpu())

    # ---- Concatenate all batches ----
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    opt_t, opt_f1 = compute_optimal_threshold(targets, probs)

    preds = (probs >= opt_t).float()
    # preds = (probs >= threshold).float()

    # ---- Micro metrics ----
    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    micro_precision = TP / (TP + FP + 1e-8)
    micro_recall = TP / (TP + FN + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision +
                                                     micro_recall + 1e-8)

    # ---- AUC ----
    try:
        auc = roc_auc_score(targets.numpy(), probs.numpy())
    except ValueError:
        auc = 0.0

    avg_batch_time_ms = sum(inference_times) / len(inference_times) * 1000

    # print(f"Micro-P={micro_precision:.4f} | "
    #       f"Micro-R={micro_recall:.4f} | "
    #       f"Micro-F1={micro_f1:.4f} | "
    #       f"AUC={auc:.4f} | "
    #       f"Avg batch time={avg_batch_time_ms:.2f} ms")

    return {
        "micro_f1": micro_f1.item(),
        "micro_p": micro_precision.item(),
        "micro_r": micro_recall.item(),
        "auc": auc,
        "avg_time_ms": avg_batch_time_ms
    }


def print_comparison_table(results):
    """Pretty-print comparison table for micro metrics + AUC."""

    print("\n" + "=" * 70)

    header = (f"{'Model':<15}"
              f"{'Micro-F1':>10}"
              f"{'Micro-P':>10}"
              f"{'Micro-R':>10}"
              f"{'AUC':>10}"
              f"{'Time (ms)':>12}")

    print(header)
    print("-" * 70)

    for model_name, metrics in results.items():
        print(f"{model_name:<15}"
              f"{metrics['micro_f1']:>10.3f}"
              f"{metrics['micro_p']:>10.3f}"
              f"{metrics['micro_r']:>10.3f}"
              f"{metrics['auc']:>10.3f}"
              f"{metrics['avg_time_ms']:>12.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark zero‑shot models")
    parser.add_argument("--bi_model",
                        type=str,
                        required=True,
                        help="Path to saved bi‑encoder model directory")
    parser.add_argument("--poly_model",
                        type=str,
                        required=True,
                        help="Path to saved polyencoder model directory")
    parser.add_argument("--test_data",
                        type=str,
                        required=True,
                        help="Path to test data JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for binary prediction (default: 0.5)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Score threshold for binary prediction (default: 0.5)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    global_labels = get_global_labels(test_data)
    print(
        f"Test set: {len(test_data)} samples, {len(global_labels)} unique labels."
    )
    test_dataset = ZeroShotDataset(samples=test_data,
                                   all_labels=global_labels,
                                   max_num_negatives=5,
                                   is_train=True)

    # Create dataloaders
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    # Load models
    models = {}
    try:
        print("\nLoading bi‑encoder model...")

        models["BiEncoder"] = BiEncoderModel.from_pretrained(args.bi_model)
    except Exception as e:
        import traceback
        traceback.print_exc()

        print(f"Failed to load bi‑encoder: {e}")

    try:
        print("Loading polyencoder model...")
        # from polyencoder import PolyencoderModel
        # models["Polyencoder"] = PolyencoderModel.from_pretrained(
        #     args.poly_model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load polyencoder: {e}")

    if not models:
        print("No models could be loaded. Exiting.")
        return

    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = benchmark_model(
            model=model,
            dataloader=test_loader,
            device=args.device,
            threshold=args.threshold,
        )
        results[name] = metrics

    # Print comparison
    print_comparison_table(results)


if __name__ == "__main__":
    main()
