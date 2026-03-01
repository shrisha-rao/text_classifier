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

# Import your model classes (must be in PYTHONPATH)
from model import BiEncoderModel  #, PolyencoderModel


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


def evaluate_model(model, test_data, global_labels, device, threshold=0.5):
    """
    Run inference on test data and compute metrics.

    Args:
        model: PyTorch model (BiEncoderModel or PolyencoderModel)
        test_data: list of dicts with 'text' and 'labels'
        global_labels: list of all possible labels (candidate set)
        device: torch device
        threshold: score threshold for binary prediction

    Returns:
        dict of metrics
    """
    model.eval()
    model.to(device)

    label_to_idx = {lbl: i for i, lbl in enumerate(global_labels)}
    num_labels = len(global_labels)

    all_true = []  # list of multi‑hot vectors (list of ints)
    all_scores = []  # list of score vectors (same order as global_labels)
    inference_times = []

    with torch.no_grad():
        for sample in test_data:
            text = sample["text"]
            true_labels = sample["labels"]  # ground truth list

            # Prepare candidate list (all global labels)
            candidates = global_labels

            # Measure inference time
            start = time.time()
            # forward_predict expects lists: [text], [candidates]
            result = model.forward_predict([text], [candidates])[0]
            end = time.time()
            inference_times.append(end - start)

            # Build true multi‑hot vector
            true_vec = [0] * num_labels
            for lbl in true_labels:
                if lbl in label_to_idx:
                    true_vec[label_to_idx[lbl]] = 1

            # Build score vector (sigmoid probabilities already)
            score_vec = [0.0] * num_labels
            for lbl, score in result["scores"].items():
                if lbl in label_to_idx:
                    score_vec[label_to_idx[lbl]] = score

            # pred_vec = [1 if score >= threshold else 0 for score in score_vec]

            all_true.append(true_vec)
            all_scores.append(score_vec)

    # Convert to numpy arrays
    y_true = np.array(all_true)  # shape (n_samples, n_labels)
    y_scores = np.array(all_scores)  # shape (n_samples, n_labels)
    print(y_scores.min().item(), y_scores.max().item())

    best_thresh, _ = compute_optimal_threshold(y_true, y_scores)
    best_thresh = 0.5
    y_pred = (y_scores >= best_thresh).astype(int)

    # Micro averaged metrics (treat each label independently)
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average='binary', zero_division=0)

    # Macro averaged metrics (average per label scores)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)

    # Compute per label AUC and then average (macro AUC)
    auc_per_label = []
    for i in range(num_labels):
        # Skip if only one class present
        if len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            auc_per_label.append(auc)
        except ValueError:
            # In case of issues (e.g., all scores constant), skip
            pass
    macro_auc = np.mean(auc_per_label) if auc_per_label else 0.0

    avg_time = np.mean(inference_times)

    return {
        "micro_f1": micro_f,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "macro_f1": macro_f,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_auc": macro_auc,
        "avg_time_ms": avg_time * 1000,  # convert to milliseconds
    }


def print_comparison_table(results):
    """Pretty‑print a comparison table including AUC."""
    print("\n" + "=" * 80)
    header = (f"{'Model':<12} {'Micro-F1':>8} {'Micro-P':>8} {'Micro-R':>8} "
              f"{'Macro-F1':>8} {'Macro-P':>8} {'Macro-R':>8} "
              f"{'Macro-AUC':>10} {'Time (ms)':>10}")
    print(header)
    print("-" * 80)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<12} {metrics['micro_f1']:>8.3f} {metrics['micro_p']:>8.3f} "
            f"{metrics['micro_r']:>8.3f} {metrics['macro_f1']:>8.3f} "
            f"{metrics['macro_p']:>8.3f} {metrics['macro_r']:>8.3f} "
            f"{metrics['macro_auc']:>10.3f} {metrics['avg_time_ms']:>10.2f}")
    print("=" * 80)


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
        metrics = evaluate_model(model,
                                 test_data,
                                 global_labels,
                                 device=args.device,
                                 threshold=args.threshold)
        results[name] = metrics

    # Print comparison
    print_comparison_table(results)


if __name__ == "__main__":
    main()
