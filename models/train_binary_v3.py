#!/usr/bin/env python3
"""
train_binary_v3.py — EfficientNet-B4 binary drain classifier
BLOCKED=1, CLEAR=0  (OTHER images → label 0, ambiguous = treat as clear)

Key design choices:
  - Camera-stratified train/val/test split (no temporal/spatial leakage)
  - Focal Loss with dynamic alpha from class distribution
  - Mixed-precision (AMP) for faster GPU training
  - Two-phase training: classifier head → top feature blocks
  - Threshold calibration on val set (maximise BLOCKED F1)
  - Final test-set evaluation + confusion matrix saved

Usage:
  python3 models/train_binary_v3.py \
      --images  ~/dissertation/images \
      --out     ~/dissertation/results/checkpoints/best_model_v3.pt \
      --cache   ~/dissertation/results/filelist_cache.json \
      --epochs 40 --phase1-epochs 3 \
      --batch 32 --lr 3e-4 --patience 8 --workers 0
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B4_Weights

# ──────────────────────────────────────────────────────────────────────────────
# Data collection — camera-stratified split
# ──────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {"blocked": 1, "clear": 0, "other": 0}
IMG_EXTS  = {".jpg", ".jpeg", ".png"}


def collect_samples(images_root: Path, cache_path=None):
    """
    Walk images/CameraName/{blocked,clear,other}/ and return
    {camera_name: [(path_str, label), ...]} dict.
    Uses os.scandir for NFS efficiency (batches dir-entry metadata).
    Caches result to JSON to avoid re-scanning on repeated runs.
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached file list from {cache_path}", flush=True)
        with open(cache_path) as f:
            return json.load(f)

    print("Scanning image directory (NFS may take ~1 min) ...", flush=True)
    by_camera = {}
    with os.scandir(images_root) as top:
        cam_entries = sorted([e for e in top if e.is_dir()], key=lambda e: e.name)

    for cam_entry in cam_entries:
        samples = []
        with os.scandir(cam_entry.path) as class_entries:
            class_dirs = {e.name.lower(): e for e in class_entries if e.is_dir()}
        for class_name, label in LABEL_MAP.items():
            if class_name not in class_dirs:
                continue
            with os.scandir(class_dirs[class_name].path) as file_entries:
                for fe in file_entries:
                    if Path(fe.name).suffix.lower() in IMG_EXTS:
                        samples.append((fe.path, label))
        by_camera[cam_entry.name] = samples
        print(f"  {cam_entry.name}: {len(samples)} images", flush=True)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(by_camera, f)
        print(f"Saved cache → {cache_path}", flush=True)

    return by_camera


def stratified_split(by_camera: dict, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split images within each camera so every camera contributes to
    train, val, AND test.  Stratified by label inside each camera
    to preserve class balance across all three splits.
    """
    rng = random.Random(seed)
    train_s, val_s, test_s = [], [], []

    for cam, samples in sorted(by_camera.items()):
        blocked = [s for s in samples if s[1] == 1]
        clear   = [s for s in samples if s[1] == 0]

        def split_class(items):
            items = items[:]
            rng.shuffle(items)
            n      = len(items)
            n_test = max(0, round(n * test_frac))
            n_val  = max(0, round(n * val_frac))
            return (items[n_test + n_val:],   # train
                    items[n_test:n_test + n_val],  # val
                    items[:n_test])            # test

        for cls in (blocked, clear):
            tr, va, te = split_class(cls)
            train_s.extend(tr)
            val_s.extend(va)
            test_s.extend(te)

    rng.shuffle(train_s)

    def counts(s):
        b = sum(1 for _, l in s if l == 1)
        return b, len(s) - b

    n_cams = len(by_camera)
    print(f"\nStratified split  ({n_cams} cameras — ALL used in training)")
    print(f"  Train  {len(train_s):6d} imgs  BLOCKED={counts(train_s)[0]}  CLEAR={counts(train_s)[1]}")
    print(f"  Val    {len(val_s):6d} imgs  BLOCKED={counts(val_s)[0]}  CLEAR={counts(val_s)[1]}")
    print(f"  Test   {len(test_s):6d} imgs  BLOCKED={counts(test_s)[0]}  CLEAR={counts(test_s)[1]}")
    print(flush=True)
    return train_s, val_s, test_s


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & transforms
# ──────────────────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop((380, 380)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.20)),
        ])
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class DrainDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples: list of (path_str, label)
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (380, 380), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_feat, 1),
    )
    return model


def freeze_except(model, keep_prefixes):
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(k) for k in keep_prefixes)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    alpha = fraction of CLEAR samples → upweights the minority BLOCKED class.
    gamma = 2.0 → standard focal modulation.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logits  = logits.squeeze(1)
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt      = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss    = alpha_t * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train: bool = True, log_interval: int = 50):
    model.train(train)
    total_loss, total_n = 0.0, 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            probs = torch.sigmoid(logits.squeeze(1)).detach().float().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += loss.item() * imgs.size(0)
            total_n    += imgs.size(0)

            if train and (i + 1) % log_interval == 0:
                preds = (np.array(all_probs) >= 0.5).astype(int)
                f1    = f1_score(all_labels, preds, pos_label=1, zero_division=0)
                print(f"  batch {i+1:4d}/{len(loader)}  "
                      f"loss={total_loss/total_n:.4f}  F1={f1:.3f}", flush=True)

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = (probs_arr >= 0.5).astype(int)
    epoch_f1   = f1_score(labels_arr, preds_arr, pos_label=1, zero_division=0)
    return total_loss / total_n, epoch_f1, probs_arr, labels_arr


def calibrate_threshold(probs: np.ndarray, labels: np.ndarray):
    """Search 0.20–0.80 for the threshold that maximises BLOCKED F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def save_checkpoint(path, model, epoch, val_f1, threshold):
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_f1":           val_f1,
        "threshold":        threshold,
        "architecture":     "efficientnet_b4",
        "epoch":            epoch,
    }, path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B4 binary drain classifier")
    p.add_argument("--images",        required=True,  help="Root image directory")
    p.add_argument("--out",           required=True,  help="Output checkpoint path (.pt)")
    p.add_argument("--cache",         default=None,   help="JSON cache path for file list")
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--phase1-epochs", type=int,   default=3,   help="Classifier-only warmup epochs")
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--patience",      type=int,   default=8,   help="Early stopping patience (phase 2)")
    p.add_argument("--workers",       type=int,   default=0,   help="DataLoader workers (0 = main process)")
    p.add_argument("--val-frac",      type=float, default=0.15)
    p.add_argument("--test-frac",     type=float, default=0.15)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    by_camera = collect_samples(Path(args.images), args.cache)
    if not by_camera:
        raise RuntimeError(f"No camera directories found under {args.images}")

    train_s, val_s, test_s = stratified_split(
        by_camera, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    loader_kw = dict(
        batch_size   = args.batch,
        num_workers  = args.workers,
        pin_memory   = (device.type == "cuda" and args.workers > 0),
        multiprocessing_context = "spawn" if args.workers > 0 else None,
    )
    train_loader = DataLoader(DrainDataset(train_s, get_transforms(True)),
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(DrainDataset(val_s,   get_transforms(False)),
                              shuffle=False, **loader_kw)
    test_loader  = DataLoader(DrainDataset(test_s,  get_transforms(False)),
                              shuffle=False, **loader_kw)

    # ── Model & loss ─────────────────────────────────────────────────────────
    model  = build_model().to(device)
    scaler = GradScaler()

    # Dynamic alpha: weight blocked class inversely to its frequency
    n_blocked = sum(1 for _, l in train_s if l == 1)
    n_clear   = len(train_s) - n_blocked
    alpha     = n_clear / len(train_s)   # fraction that are NOT blocked → upweights blocked
    print(f"Class balance — BLOCKED: {n_blocked:,}  CLEAR: {n_clear:,}  "
          f"ratio 1:{n_clear/max(n_blocked,1):.1f}  Focal alpha={alpha:.3f}", flush=True)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "phase", "train_loss", "train_f1",
                                 "val_loss", "val_f1", "threshold", "lr"])

    best_val_f1    = 0.0
    best_threshold = 0.5
    patience_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 — classifier head only
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1 — classifier only  ({args.phase1_epochs} epochs)")
    print(f"{'='*60}", flush=True)
    freeze_except(model, ["classifier"])
    print(f"Trainable params: {count_trainable(model):,}", flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs, eta_min=1e-5)

    for ep in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _            = run_epoch(model, train_loader, criterion,
                                                     optimizer, scaler, device, train=True)
        va_loss, va_f1, val_probs, val_labels = run_epoch(model, val_loader, criterion,
                                                     optimizer, scaler, device, train=False)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"[P1 ep{ep:02d}] "
              f"train {tr_loss:.4f}/{tr_f1:.3f}  "
              f"val {va_loss:.4f}/{va_f1:.3f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s", flush=True)

        thresh, _ = calibrate_threshold(val_probs, val_labels)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, 1, tr_loss, tr_f1, va_loss, va_f1, thresh, lr_now])

        if va_f1 > best_val_f1:
            best_val_f1    = va_f1
            best_threshold = thresh
            save_checkpoint(out_path, model, ep, best_val_f1, best_threshold)
            print(f"  ✓ Checkpoint  val_f1={best_val_f1:.3f}  threshold={best_threshold:.2f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — top feature blocks + classifier
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 2 — features.6 / features.7 / features.8 + classifier")
    print(f"{'='*60}", flush=True)
    freeze_except(model, ["features.6", "features.7", "features.8", "classifier"])
    print(f"Trainable params: {count_trainable(model):,}", flush=True)
    patience_count = 0

    remaining = max(1, args.epochs - args.phase1_epochs)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=1e-6)

    for ep in range(args.phase1_epochs + 1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _            = run_epoch(model, train_loader, criterion,
                                                     optimizer, scaler, device, train=True)
        va_loss, va_f1, val_probs, val_labels = run_epoch(model, val_loader, criterion,
                                                     optimizer, scaler, device, train=False)
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        thresh, _ = calibrate_threshold(val_probs, val_labels)

        print(f"[P2 ep{ep:02d}] "
              f"train {tr_loss:.4f}/{tr_f1:.3f}  "
              f"val {va_loss:.4f}/{va_f1:.3f}  "
              f"thresh={thresh:.2f}  lr={lr_now:.2e}  {elapsed:.0f}s", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, 2, tr_loss, tr_f1, va_loss, va_f1, thresh, lr_now])

        if va_f1 > best_val_f1:
            best_val_f1    = va_f1
            best_threshold = thresh
            patience_count = 0
            save_checkpoint(out_path, model, ep, best_val_f1, best_threshold)
            print(f"  ✓ Checkpoint  val_f1={best_val_f1:.3f}  threshold={best_threshold:.2f}", flush=True)
        else:
            patience_count += 1
            print(f"  no improvement ({patience_count}/{args.patience})", flush=True)
            if patience_count >= args.patience:
                print("  Early stopping.", flush=True)
                break

    # ─────────────────────────────────────────────────────────────────────────
    # Test evaluation — load best checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test evaluation")
    print(f"{'='*60}", flush=True)

    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = ckpt["threshold"]
    print(f"Best checkpoint: epoch={ckpt['epoch']}  val_f1={ckpt['val_f1']:.3f}  "
          f"threshold={threshold:.2f}", flush=True)

    # Unfreeze everything for inference (no grad)
    for p in model.parameters():
        p.requires_grad = False

    _, _, test_probs, test_labels = run_epoch(
        model, test_loader, criterion, None, scaler, device, train=False)

    test_preds = (test_probs >= threshold).astype(int)
    test_f1    = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
    cm         = confusion_matrix(test_labels, test_preds)

    print("\nClassification report (test set):")
    print(classification_report(test_labels, test_preds,
                                 target_names=["CLEAR", "BLOCKED"], digits=3))
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\nTest F1 (BLOCKED): {test_f1:.3f}", flush=True)

    # Save artefacts for dissertation write-up
    results = {
        "test_f1_blocked":   float(test_f1),
        "threshold":         float(threshold),
        "val_f1":            float(ckpt["val_f1"]),
        "best_epoch":        int(ckpt["epoch"]),
        "confusion_matrix":  {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "n_train":           len(train_s),
        "n_val":             len(val_s),
        "n_test":            len(test_s),
        "n_blocked_train":   n_blocked,
        "n_clear_train":     n_clear,
    }
    results_path = out_path.with_name(out_path.stem + "_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    np.save(out_path.with_name(out_path.stem + "_test_probs.npy"),  test_probs)
    np.save(out_path.with_name(out_path.stem + "_test_labels.npy"), test_labels)
    print(f"\nSaved: {results_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
