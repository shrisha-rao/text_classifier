import json
import random

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import ZeroShotDataset
from model2 import BiEncoderModel
from polyencoder import PolyencoderModel
import os
import argparse
from sklearn.metrics import f1_score, roc_auc_score


def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["labels"] for item in batch]
    targets = [item["targets"] for item in batch]
    return {"texts": texts, "labels": labels, "targets": targets}


def load_and_split_data(config):
    """Load JSON and split into train/val/test."""
    data_path = config["data"]["synthetic_data_path"]

    with open(data_path) as f:
        samples = json.load(f)

    # from datasets import load_dataset
    # dataset = load_dataset("reuters21578", "ModHayes", trust_remote_code=True)
    # samples = []
    # for ex in dataset["train"]:
    #     if ex["topics"]:  # Skip empty
    #         samples.append({"text": ex["text"], "labels": ex["topics"]})

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

    print(f"#trainL: {len(train_loader)} #val: {len(val_loader)}")
    return train_loader, val_loader, test_loader


def eval_on_test(test_loader, model_type, loss_fn):
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


def log_neg_sampling_ratio(targets, global_step, writer, device):
    # log sampling stats for the current target batch
    pos_counts = [sum(t == 1.0 for t in sample) for sample in targets]
    neg_counts = [len(sample) - p for sample, p in zip(targets, pos_counts)]
    pos_mean = torch.tensor(pos_counts, dtype=torch.float32,
                            device=device).mean()
    neg_mean = torch.tensor(neg_counts, dtype=torch.float32,
                            device=device).mean()
    ratio = neg_mean / (pos_mean + 1e-8)
    writer.add_scalar("sampling/neg_pos_ratio", ratio.item(), global_step)


def log_update_to_data_ratio(model, scheduler, global_step, writer):
    # update-to-data ratio layer by layer ref. Karpathy
    # If this metric drops down to -5.0, the model is essentially frozen.
    # If it spikes up to -1.0, learning rate is too aggressive
    # the model is super noisy
    current_lr = scheduler.get_last_lr()[0]
    layer_ratios = {}  # dictionary for per layer scalars
    log_ratios_list = []  # for global average
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                update_norm = (current_lr * p.grad).norm()
                param_norm = p.data.norm()

                # update_norm = (current_lr * p.grad).std()
                # param_norm = p.data.std()

                if param_norm > 0 and update_norm > 0:
                    log10_ratio = (update_norm /
                                   (param_norm + 1e-8)).log10().item()
                    # Store for per‑layer overlay
                    layer_ratios[name] = log10_ratio
                    log_ratios_list.append(log10_ratio)

    # overlay all per layer ratios in one plot
    if layer_ratios:
        writer.add_scalars('diagnostics/update_ratio', layer_ratios,
                           global_step)

    # global average
    if log_ratios_list:
        avg_log_ratio = sum(log_ratios_list) / len(log_ratios_list)
        writer.add_scalar('diagnostics/update_ratio_avg', avg_log_ratio,
                          global_step)


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


from sklearn.metrics import f1_score, roc_auc_score


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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_type", choices=["bi", "poly"], default="bi")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config["logging"]["tensorboard_dir"])

    # Dataset
    train_loader, val_loader, test_loader = load_and_split_data(config)
    # Model
    if args.model_type == "bi":
        print(f"loading model: {config['model']['name']}")
        model = BiEncoderModel(
            config["model"]["name"], int(config["model"]["max_num_labels"]),
            int(config['data']['max_seq_length']),
            int(config['training']['layers_to_freeze'])).to(device)
    else:
        model = PolyencoderModel(config["model"]["name"],
                                 int(config["model"]["max_num_labels"]),
                                 int(config['data']['max_seq_length']))
    model.to(device)

    # Inside the train() function, replacing the optimizer/scheduler section:

    # 1. Calculate total steps correctly for the scheduler
    epochs = int(config["training"].get("epochs", 5))
    accumulation_steps = int(config["training"].get(
        "gradient_accumulation_steps", 1))
    total_steps = len(train_loader) * epochs // accumulation_steps
    print(f"total steps: {total_steps}")

    # 2. Add weight decay to the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=float(config["training"]["learning_rate"]),
                      weight_decay=config["training"].get(
                          "weight_decay", 0.01))

    # 3. Use warmup_ratio instead of hardcoded steps
    warmup_steps = int(total_steps *
                       config["training"].get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # Loss function
    pos_weight = torch.tensor([2.8])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    # loss_fn = multi_label_softmax_loss()

    # early stopping parameters
    patience = int(config["training"].get("early_stop_patience",
                                          5))  # eval steps without improvement
    min_delta = float(config["training"].get("early_stop_min_delta",
                                             0.0))  # required improvement

    global_step = 0  # tracks optimizer steps
    best_val_loss = float('inf')
    epochs_without_improve = 0.
    for epoch in range(epochs):
        # while global_step < config["training"]["num_steps"]:
        model.train()
        optimizer.zero_grad()  # start accumulation cycle
        for i, batch in enumerate(train_loader):
            texts = batch["texts"]
            labels = batch["labels"]  # list of lists of strings
            targets = batch["targets"]  # list of lists of floats

            if global_step % int(config['training']['eval_steps']) == 0:
                log_neg_sampling_ratio(targets, global_step, writer, device)

            # ---------- Forward Pass  ----------
            scores, mask = model(texts, labels)  # scores: [B, max_labels]

            # print(f"Scores range: {scores.min():.3f} → {scores.max():.3f}")
            # print(
            #     f"Labels mean:  {torch.tensor([t for sublist in targets for t in sublist]).float().mean():.3f}"
            # )  # should be << 1.0
            # print(f"Mask ratio:  {mask.float().mean():.3f}"
            #       )  # fraction valid labels
            # print(
            #     f"Loss inputs match: {scores.shape == targets.shape == mask.shape}"
            # )
            # # Then compute loss manually:
            # valid_scores = scores[mask]
            # valid_labels = torch.tensor(
            #     [t for sublist in targets for t in sublist])[mask]

            # print(
            #     f"Valid scores: {valid_scores.min():.3f} → {valid_scores.max():.3f}"
            # )

            # Create target tensor of same shape as scores
            target_tensor = torch.zeros_like(scores, device=device)
            for i, tgt_list in enumerate(targets):
                max_labels = min(len(tgt_list), model.max_num_labels)
                target_tensor[i, :max_labels] = torch.tensor(
                    tgt_list[:max_labels], device=device)

            # Compute loss only on masked positions
            loss = loss_fn(scores[mask], target_tensor[mask])

            # loss = multi_label_softmax_loss(scores, targets, mask)

            loss.backward()

            # ---------- optimizer step every accumulation_steps ----------
            if (i + 1) % accumulation_steps == 0:
                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config['training'].get('max_grad_norm', 1.0)))

                if global_step > 20:  # some warmup period
                    log_update_to_data_ratio(model, scheduler, global_step,
                                             writer)

                # update params
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # loss logging
            if global_step % config["training"]["log_steps"] == 0:
                # writer.add_scalar(f"loss/train_{args.model_type}", loss.item(),
                #                   global_step)
                writer.add_scalars(f"loss/{args.model_type}",
                                   {"train": loss.item()}, global_step)
                writer.add_scalar(f"lr",
                                  scheduler.get_last_lr()[0], global_step)

                print(f"Step {global_step}: loss = {loss.item():.4f}")

            # ---------- Validation step  ----------
            # Run validation, compute metrics
            if global_step % config["training"]["eval_steps"] == 0:
                avg_val_loss = validation_and_log(model, val_loader,
                                                  global_step, writer, device,
                                                  loss_fn, args.model_type)

                writer.add_scalars(
                    f"diagnostics/{args.model_type}", {
                        "min_score": scores.min().item(),
                        "max_score": scores.max().item()
                    }, global_step)

                # ---------- early stopping logic ----------
                improvement = best_val_loss - avg_val_loss
                if improvement > min_delta:  # got better
                    best_val_loss = avg_val_loss
                    epochs_without_improve = 0
                    # optional: save the *best* checkpoint immediately
                    best_ckpt_dir = f"checkpoints/{args.model_type}/best"
                    os.makedirs(best_ckpt_dir, exist_ok=True)
                    torch.save(
                        {
                            'step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }, f"{best_ckpt_dir}/checkpoint.pt")
                    #model.shared_encoder.config.save_pretrained(best_ckpt_dir)
                    model.save_pretrained(best_ckpt_dir)
                else:  # no improvement
                    if global_step > 20:  # burn-in period
                        epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        print(
                            f"Early stopping triggered at step {global_step} "
                            f"(no improvement for {patience} evals).")
                        # break out of both loops
                        global_step = config["training"]["num_steps"]
                        break

            if global_step % config["training"]["save_steps"] == 0:
                # Save checkpoint
                checkpoint_dir = f"checkpoints/{args.model_type}/latest"  # -{global_step}
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, f"{checkpoint_dir}/checkpoint.pt")
                # Also save model for HF
                #model.shared_encoder.config.save_pretrained(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)

            if global_step >= config["training"]["num_steps"]:
                break
        if global_step >= config["training"]["num_steps"]:
            break

    # Push best model to hub
    # model.push_to_hub("your-username/bi-encoder-zero-shot")  # after implementing
    writer.close()
    eval_on_test(test_loader, args.model_type, loss_fn)


if __name__ == "__main__":
    train()
