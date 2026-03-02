"""
Training script for zero-shot multi-label text classification.

This module implements the full training and evaluation pipeline for 
BiEncoder and PolyEncoder architectures in a zero-shot learning setting (defined in model.py and polyencoder.py respectively).
The goal is to classify input texts into multiple possible labels, including 
labels not seen during training.

Main features:
-------------
- Deterministic train/validation/test split
- Dynamic negative sampling for zero-shot learning
- Support for Bi-Encoder and Poly-Encoder models
- Multi-label classification with BCE loss
- Gradient accumulation and gradient clipping
- Learning rate warmup and linear decay scheduler
- TensorBoard logging for training diagnostics
- Early stopping based on validation performance
- Checkpoint saving (latest and best models)
- Evaluation on a held-out test set

Pipeline:
--------
1. Load synthetic dataset from a JSON file. (path in config.yaml)
2. Split into train, validation, and test sets.
3. Build global label set from the training data.
4. Train the selected model with negative sampling.
5. Monitor performance using F1-score and AUC.
6. Save checkpoints and apply early stopping.
7. Evaluate the final model on the test set.

Usage:
-----
Run the script from the command line:

    python train.py --config config.yaml --model_type bi

Arguments:
---------
--config : Path to YAML configuration file.
--model_type : Type of encoder model ("bi" or "poly").

Reproducibility:
---------------
Random seeds and dataset shuffling controlled by seed set in config.yaml
"""

import json
import random

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import ZeroShotDataset
from model import BiEncoderModel
from polyencoder import PolyencoderModel
import os
import argparse
from sklearn.metrics import f1_score, roc_auc_score

# helper functions for the training loop
from scripts.utils import set_seed, collate_fn, load_and_split_data, eval_on_test, log_neg_sampling_ratio, log_update_to_data_ratio, validation_and_log


def train():
    # Get training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_type", choices=["bi", "poly"], default="bi")
    args = parser.parse_args()

    # read config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # set random seed
    set_seed(config)

    # set device and setup tensorboard for logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config["logging"]["tensorboard_dir"] +
                           f'/{args.model_type}')

    # load dataset
    train_loader, val_loader, test_loader = load_and_split_data(config)

    # load model
    if args.model_type == "bi":
        print(f"loading model: {config['model']['name']}")
        model = BiEncoderModel(
            config["model"]["name"], int(config["model"]["max_num_labels"]),
            int(config['model']['layers_to_freeze'])).to(device)
    else:
        model = PolyencoderModel(config["model"]["name"],
                                 int(config["model"]["max_num_labels"]),
                                 int(config['model']['layers_to_freeze']),
                                 int(config['model']['num_global_vectors']))
    model.to(device)

    # Calculate total steps correctly for the scheduler
    epochs = int(config["training"].get("epochs", 5))
    accumulation_steps = int(config["training"].get(
        "gradient_accumulation_steps", 1))
    total_steps = len(train_loader) * epochs // accumulation_steps
    print(f"total steps: {total_steps}")

    # Add weight decay to the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=float(config["training"]["learning_rate"]),
                      weight_decay=config["training"].get(
                          "weight_decay", 0.01))

    # Use warmup_ratio
    warmup_steps = int(total_steps *
                       config["training"].get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # Loss function
    pos_weight = torch.tensor([float(config['training']['pos_weight'])])
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
        model.train()
        optimizer.zero_grad()  # start accumulation cycle
        for i, batch in enumerate(train_loader):
            texts = batch["texts"]
            labels = batch["labels"]  # list of lists of strings
            targets = batch["targets"]  # list of lists of floats

            if global_step % int(config['training']['eval_steps']) == 0:
                log_neg_sampling_ratio(targets, global_step, writer, device,
                                       args.model_type)

            # ---------- Forward Pass  ----------
            scores, mask = model(texts, labels)  # scores: [B, max_labels]

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
                                             writer, args.model_type)

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
                writer.add_scalar(f"lr/{args.model_type}",
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

    writer.close()
    eval_on_test(test_loader, args.model_type, loss_fn)


if __name__ == "__main__":
    train()
