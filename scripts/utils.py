"""Helper functions for train.py
"""

import json
import random
import torch
from torch.utils.data import DataLoader
from dataset import ZeroShotDataset
from model import BiEncoderModel
from polyencoder import PolyencoderModel
from sklearn.metrics import f1_score, roc_auc_score


# -------- HELPER FUNCTIONS ----------
def set_seed(config):
    """set seed for reproducability"""
    seed = int(config["data"].get("random_seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Collate function for Dataloader     
    """
    texts = [item["text"] for item in batch]
    labels = [item["labels"] for item in batch]
    targets = [item["targets"] for item in batch]
    return {"texts": texts, "labels": labels, "targets": targets}


def load_and_split_data(config):
    """Load JSON and split into train/val/test."""
    data_path = config["data"]["synthetic_data_path"]

    with open(data_path) as f:
        samples = json.load(f)

    print(f"Total raw samples in JSON: {len(samples)}")
    print(f"first sample: {samples[0]['text']} {samples[0]['labels']}")

    # Shuffle deterministically
    random.seed(config["data"].get("random_seed", 42))
    random.shuffle(samples)

    total = len(samples)
    train_ratio = config["data"].get("train_split", 0.8)
    val_ratio = config["data"].get("val_split", 0.1)
    test_ratio = config["data"].get("test_split", 0.1)

    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Build global label set from training set only (to avoid leakage)
    all_labels = list(set(lbl for s in train_samples for lbl in s["labels"]))
    max_num_negatives = int(config["data"]["max_num_negatives"])
    # Create datasets
    train_dataset = ZeroShotDataset(samples=train_samples,
                                    all_labels=all_labels,
                                    max_num_negatives=max_num_negatives,
                                    is_train=True)
    val_dataset = ZeroShotDataset(samples=val_samples,
                                  all_labels=all_labels,
                                  max_num_negatives=max_num_negatives,
                                  is_train=True)
    test_dataset = ZeroShotDataset(samples=test_samples,
                                   all_labels=all_labels,
                                   max_num_negatives=max_num_negatives,
                                   is_train=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=config["training"]["batch_size"],
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=config["training"]["batch_size"],
                             shuffle=False,
                             collate_fn=collate_fn)

    print(f"#train: {len(train_loader)} #val: {len(val_loader)}")
    return train_loader, val_loader, test_loader


def eval_on_test(test_loader, model_type, loss_fn):
    """Evaluate the model on test data 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if model_type == 'bi':
            model = BiEncoderModel.from_pretrained('checkpoints/bi/latest')
        else:
            model = PolyencoderModel.from_pretrained('checkpoints/poly/latest')
        #
        model.to(device)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_batch in test_loader:
                texts = test_batch["texts"]
                labels = test_batch["labels"]
                targets = test_batch["targets"]
                scores, mask = model(texts, labels)
                target_tensor = torch.zeros_like(scores, device=device)
                for i, tgt_list in enumerate(targets):
                    max_labels = min(len(tgt_list), model.max_num_labels)
                    target_tensor[i, :max_labels] = torch.tensor(
                        tgt_list[:max_labels], device=device)
                loss = loss_fn(scores[mask], target_tensor[mask])
                test_loss += loss.item() * len(texts)  # sum losses
        #
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"test loss = {avg_test_loss:.4f}")
    except Exception as e:
        print('Error in eval on test')
        print(e)


def log_neg_sampling_ratio(targets, global_step, writer, device, model_type):
    """log sampling stats for the current target batch"""
    pos_counts = [sum(t == 1.0 for t in sample) for sample in targets]
    neg_counts = [len(sample) - p for sample, p in zip(targets, pos_counts)]
    pos_mean = torch.tensor(pos_counts, dtype=torch.float32,
                            device=device).mean()
    neg_mean = torch.tensor(neg_counts, dtype=torch.float32,
                            device=device).mean()
    ratio = neg_mean / (pos_mean + 1e-8)
    writer.add_scalar(f"sampling/{model_type}/neg_pos_ratio", ratio.item(),
                      global_step)


def log_update_to_data_ratio(model, scheduler, global_step, writer,
                             model_type):
    """update-to-data ratio layer by layer ref. Karpathy"""
    current_lr = scheduler.get_last_lr()[0]
    layer_ratios = {}  # dictionary for per layer scalars
    log_ratios_list = []  # for global average
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                update_norm = (current_lr * p.grad).norm()
                param_norm = p.data.norm()
                if param_norm > 0 and update_norm > 0:
                    log10_ratio = (update_norm /
                                   (param_norm + 1e-8)).log10().item()
                    # Store for per‑layer overlay
                    layer_ratios[name] = log10_ratio
                    log_ratios_list.append(log10_ratio)

    # overlay all per layer ratios in one plot
    if layer_ratios:
        writer.add_scalars(f'diagnostics/{model_type}/update_ratio',
                           layer_ratios, global_step)

    # global average
    if log_ratios_list:
        avg_log_ratio = sum(log_ratios_list) / len(log_ratios_list)
        writer.add_scalar(f'diagnostics/{model_type}/update_ratio_avg',
                          avg_log_ratio, global_step)


def multi_label_softmax_loss(scores, targets, mask):
    # 1. remove padded labels from softmax
    scores = scores.masked_fill(~mask, -1e9)

    # 2. log softmax across labels
    log_probs = torch.nn.functional.log_softmax(scores, dim=1)

    # 3. multi-positive cross entropy
    loss = -(targets * log_probs)

    # 4. ignore padding
    loss = (loss * mask).sum() / mask.sum()

    return loss


def validation_and_log(model, val_loader, global_step, writer, device, loss_fn,
                       model_type):
    model.eval()
    val_loss = 0.0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for val_batch in val_loader:
            val_texts = val_batch["texts"]
            val_labels = val_batch["labels"]
            val_targets_list = val_batch["targets"]

            scores, mask = model(val_texts, val_labels)

            # Build target tensor
            target_tensor = torch.zeros_like(scores, device=device)
            for i, tgt_list in enumerate(val_targets_list):
                max_labels = min(len(tgt_list), model.max_num_labels)
                target_tensor[i, :max_labels] = torch.tensor(
                    tgt_list[:max_labels], device=device)

            # Compute loss (mask padded positions)
            loss = loss_fn(scores[mask], target_tensor[mask])
            val_loss += loss.item() * len(val_texts)

            # Accumulate for metrics
            all_logits.append(scores[mask].cpu())
            all_targets.append(target_tensor[mask].cpu())

    # Average validation loss
    avg_val_loss = val_loss / len(val_loader.dataset)

    # Concatenate all batches
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    # Probabilities and predictions
    val_probs = torch.sigmoid(all_logits).numpy()
    val_preds = (val_probs >= 0.5).astype(int)
    targets_np = all_targets.numpy()

    # Metrics
    micro_f1 = f1_score(targets_np, val_preds, average='micro')
    macro_f1 = f1_score(targets_np, val_preds, average='macro')
    try:
        auc = roc_auc_score(targets_np, val_probs)
    except ValueError:
        auc = 0.0

    # Log

    writer.add_scalars(f"loss/{model_type}", {"val": avg_val_loss},
                       global_step)
    writer.add_scalar("metrics/micro_f1", micro_f1, global_step)
    writer.add_scalar("metrics/macro_f1", macro_f1, global_step)
    writer.add_scalar("metrics/auc", auc, global_step)

    print(f"Step {global_step}: val_loss={avg_val_loss:.4f} | "
          f"micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f} | AUC={auc:.4f}")

    return avg_val_loss
