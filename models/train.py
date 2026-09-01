"""
train.py
--------
Training and validation loop for the drainage blockage classifier.

Usage:
    python src/train.py                        # uses configs/config.yaml
    python src/train.py --config configs/config.yaml

Key features:
    - Transfer learning with pretrained ResNet-50
    - Class-imbalance correction via BCEWithLogitsLoss pos_weight
    - Early stopping (monitors val loss, patience configurable in config.yaml)
    - Saves best checkpoint (lowest val loss)
    - Per-epoch metrics logged to CSV for later plotting
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

from dataset import get_dataloaders, get_dataloaders_from_manifests
from model import get_model, count_parameters


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available — using CPU (training will be slow)")
    return device


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    is_train: bool,
) -> dict:
    """Run a single training or validation epoch. Returns metric dict."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # (B,1)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().squeeze(1).tolist()
            all_preds  += preds
            all_labels += labels.long().cpu().squeeze(1).tolist()

    n = len(all_labels)
    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
    }


# ── Main training loop ─────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    mode = cfg["data"].get("mode", "folder")
    print(f"\nLoading data (mode={mode})...")
    if mode == "manifest":
        loaders = get_dataloaders_from_manifests(
            manifest_dir=cfg["data"]["manifest_dir"],
            batch_size=cfg["training"]["batch_size"],
            img_size=cfg["data"]["img_size"],
            num_workers=cfg["data"]["num_workers"],
        )
    else:
        loaders = get_dataloaders(
            processed_dir=cfg["data"]["processed_dir"],
            batch_size=cfg["training"]["batch_size"],
            img_size=cfg["data"]["img_size"],
            num_workers=cfg["data"]["num_workers"],
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = get_model(pretrained=cfg["model"]["pretrained"]).to(device)

    stats = count_parameters(model)
    print(f"Parameters — total: {stats['total']:,} | trainable: {stats['trainable']:,}")

    # ── Loss: weight the positive (blocked) class to correct for imbalance ───
    pos_weight = torch.tensor([cfg["training"]["pos_weight"]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiser + LR scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_dir  = Path(cfg["paths"]["logs"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "training_log.csv"
    fieldnames = [
        "epoch",
        "train_loss", "train_acc", "train_prec", "train_rec", "train_f1",
        "val_loss",   "val_acc",   "val_prec",   "val_rec",   "val_f1",
        "lr", "epoch_time_s",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_val_loss   = float("inf")
    epochs          = cfg["training"]["epochs"]
    patience        = cfg["training"].get("early_stopping_patience", 7)
    epochs_no_improve = 0

    print(f"\nStarting training for up to {epochs} epochs "
          f"(early stopping patience={patience})...\n")
    print(f"{'Ep':>3} | {'Tr Loss':>8} {'Tr Acc':>7} {'Tr F1':>7} | "
          f"{'Va Loss':>8} {'Va Acc':>7} {'Va F1':>7} | {'LR':>8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, loaders["train"], criterion,
                                  optimizer, device, is_train=True)
        val_metrics   = run_epoch(model, loaders["val"],   criterion,
                                  None,      device, is_train=False)

        scheduler.step()
        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        # ── Console output ────────────────────────────────────────────────────
        print(
            f"{epoch:>3} | "
            f"{train_metrics['loss']:>8.4f} {train_metrics['accuracy']:>7.4f} "
            f"{train_metrics['f1']:>7.4f} | "
            f"{val_metrics['loss']:>8.4f} {val_metrics['accuracy']:>7.4f} "
            f"{val_metrics['f1']:>7.4f} | "
            f"{lr:>8.2e}   [{elapsed:.0f}s]"
        )

        # ── CSV logging ───────────────────────────────────────────────────────
        row = {
            "epoch": epoch,
            "train_loss":  train_metrics["loss"],
            "train_acc":   train_metrics["accuracy"],
            "train_prec":  train_metrics["precision"],
            "train_rec":   train_metrics["recall"],
            "train_f1":    train_metrics["f1"],
            "val_loss":    val_metrics["loss"],
            "val_acc":     val_metrics["accuracy"],
            "val_prec":    val_metrics["precision"],
            "val_rec":     val_metrics["recall"],
            "val_f1":      val_metrics["f1"],
            "lr":          lr,
            "epoch_time_s": elapsed,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        # ── Checkpoint + early stopping ───────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss     = val_metrics["loss"]
            epochs_no_improve = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   best_val_loss,
                "val_f1":     val_metrics["f1"],
                "config":     cfg,
            }, ckpt_path)
            print(f"    ✓ New best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    · No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\n⚠  Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best_model.pt'}")
    print(f"Log:        {log_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "configs" / "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
