"""
train_binary_v2.py
------------------
Improved binary classifier for drainage blockage detection.

Improvements over the original binary model:
  1. EfficientNet-B4 backbone  — better accuracy than ResNet-50 at similar speed
  2. Focal Loss (gamma=2)      — handles the ~1:6 BLOCKED/CLEAR class imbalance
  3. Stronger augmentation     — ColorJitter, RandomErasing, GaussianBlur, flips
  4. All 80K images used       — CLEAR + OTHER both treated as the negative class
  5. Threshold calibration     — optimal decision threshold found on val set and
                                  saved alongside weights (no more fixed 0.5)
  6. Cosine LR schedule        — smoother decay than step/plateau

Label convention:
    BLOCKED → 1  (positive — what we want to detect)
    CLEAR   → 0
    OTHER   → 0  (ambiguous drain status → safe/negative)

Usage on cheery:
    cd ~/dissertation
    CUDA_VISIBLE_DEVICES=6 nohup python3 models/train_binary_v2.py \\
        --images ~/dissertation/images \\
        --out    ~/dissertation/results/checkpoints/best_model_v2.pt \\
        --epochs 30 --phase1-epochs 5 \\
        --batch 32 --lr 3e-4 --patience 8 --workers 4 \\
        > ~/dissertation/logs/train_binary_v2.log 2>&1 &
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss  (Lin et al., 2017).

    alpha  — weight for the positive (BLOCKED) class.
             Computed dynamically as n_neg / (n_pos + n_neg) so that the loss
             is balanced regardless of the exact dataset split.
    gamma  — focusing exponent.  gamma=2 is the standard recommendation:
             easy examples (high pt) contribute very little; hard examples drive
             learning.  This is particularly useful here because most CLEAR
             images are trivially easy and would otherwise dominate training.
    """
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets.float(), reduction="none"
        )
        pt    = torch.exp(-bce)
        alpha = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss  = alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# ── Data collection ───────────────────────────────────────────────────────────

def _collect_samples(images_root: str) -> list:
    """
    Walk the images directory and assign binary labels.

    Folder layout (matches the EA dataset structure):
        images/
          CameraName1/
            blocked/   ← BLOCKED (positive, label=1)
            clear/     ← CLEAR   (negative, label=0)
            other/     ← OTHER   (negative — ambiguous drain, label=0)
          CameraName2/
            ...
    """
    root      = Path(images_root)
    label_map = {"blocked": 1, "clear": 0, "other": 0}
    exts      = {".jpg", ".jpeg", ".png", ".bmp"}
    counts    = {"blocked": 0, "clear": 0, "other": 0}
    samples   = []

    for camera_dir in sorted(root.iterdir()):
        if not camera_dir.is_dir():
            continue
        for class_name, label in label_map.items():
            class_dir = camera_dir / class_name
            if not class_dir.is_dir():
                continue
            imgs = [f for f in class_dir.iterdir() if f.suffix.lower() in exts]
            samples.extend((str(f), label) for f in imgs)
            counts[class_name] += len(imgs)

    for cls, n in counts.items():
        print(f"  {cls:8s}: {n:6d} images  →  label {label_map[cls]}")

    return samples


def _stratified_split(samples, val_frac=0.15, test_frac=0.15):
    paths, labels = zip(*samples)
    paths, labels = list(paths), list(labels)

    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        paths, labels, test_size=test_frac, stratify=labels, random_state=SEED,
    )
    val_adj = val_frac / (1.0 - test_frac)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv, test_size=val_adj, stratify=labels_tv, random_state=SEED,
    )
    return (
        list(zip(paths_train, labels_train)),
        list(zip(paths_val,   labels_val)),
        list(zip(paths_test,  labels_test)),
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrainageDataset(Dataset):
    def __init__(self, samples: list, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ── Transforms ────────────────────────────────────────────────────────────────

def _get_transforms(img_size: int = 224):
    """
    Train: strong augmentation to improve generalisation across cameras /
           lighting conditions / weather / partial blockages.
    Val / Test: deterministic centre-crop only.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    resize = int(img_size * 256 / 224)   # 256 for 224, 293 for 256 etc.

    train_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        # Simulate varying exposure, weather, lighting across 56 cameras
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.05),
        # Simulate rain / condensation on lens
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # Simulate debris, partial occlusion, or damage to drain screen
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_tf, val_tf


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    EfficientNet-B4 with a single binary output (BLOCKED probability logit).

    EfficientNet-B4 vs ResNet-50:
      - Similar parameter count (~19M vs ~25M)
      - Consistently 1-3% higher accuracy on fine-grained visual tasks
      - More efficient use of depth/width/resolution via compound scaling
      - classifier[1] is the final Linear(1792 → num_classes)
    """
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b4(weights=weights)
    in_feat = model.classifier[1].in_features   # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_feat, 1),
    )
    return model


def freeze_except(model: nn.Module, unfrozen_prefixes: list) -> None:
    """Freeze all params except those whose names start with an unfrozen prefix."""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(p) for p in unfrozen_prefixes)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss    = 0.0
    running_loss  = 0.0
    all_preds, all_labels = [], []

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss   += loss.item()
        running_loss += loss.item()

        preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().numpy())

        if train and (i + 1) % 200 == 0:
            print(f"    batch {i+1}/{len(loader)}  loss={running_loss/200:.4f}")
            running_loss = 0.0

    f1       = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    avg_loss = total_loss / len(loader)
    return avg_loss, f1


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_threshold(model: nn.Module, loader, device) -> float:
    """
    Search the validation set for the decision threshold t in [0.20, 0.80]
    that maximises binary F1 for the BLOCKED class.

    Saving this threshold with the model means inference never needs a
    hardcoded 0.5 — the optimal cut-point is baked into the checkpoint.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.sigmoid(model(imgs.to(device)).squeeze(1))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.long().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        preds = (all_probs >= t).astype(int)
        f1    = f1_score(all_labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    print(f"  Best threshold: {best_t:.2f}  (val BLOCKED F1 = {best_f1:.4f})")
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Scanning images...")
    samples   = _collect_samples(args.images)
    n_blocked = sum(1 for _, l in samples if l == 1)
    n_neg     = len(samples) - n_blocked
    print(f"\n  Total: {len(samples)}  BLOCKED: {n_blocked}  NEG: {n_neg}  "
          f"ratio 1:{n_neg/max(n_blocked, 1):.1f}")

    train_s, val_s, test_s = _stratified_split(samples)
    print(f"  Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    train_tf, val_tf = _get_transforms(img_size=224)
    pin = args.workers > 0
    ctx = "spawn" if args.workers > 0 else None
    pw  = args.workers > 0

    def make_loader(split, tf, shuffle):
        return DataLoader(
            DrainageDataset(split, tf),
            batch_size=args.batch, shuffle=shuffle,
            num_workers=args.workers, pin_memory=pin,
            persistent_workers=pw, multiprocessing_context=ctx,
        )

    train_loader = make_loader(train_s, train_tf, shuffle=True)
    val_loader   = make_loader(val_s,   val_tf,   shuffle=False)
    test_loader  = make_loader(test_s,  val_tf,   shuffle=False)

    # ── Loss ──────────────────────────────────────────────────────────────────
    alpha     = n_neg / (n_blocked + n_neg)   # weight for BLOCKED (positive class)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    print(f"\nFocal Loss  alpha={alpha:.3f}  gamma=2.0")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding EfficientNet-B4...")
    model = build_model().to(device)

    # ── Phase 1: warm-up — classifier head only ───────────────────────────────
    print(f"\n── Phase 1: classifier head ({args.phase1_epochs} epochs) ──────────────────────")
    freeze_except(model, ["classifier"])
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    for epoch in range(1, args.phase1_epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion, opt1, device, train=True)
        vl_loss, vl_f1 = run_epoch(model, val_loader,   criterion, None,  device, train=False)
        print(f"  [{epoch:02d}/{args.phase1_epochs}]  "
              f"train loss={tr_loss:.4f} f1={tr_f1:.4f}  "
              f"val loss={vl_loss:.4f} f1={vl_f1:.4f}")

    # ── Phase 2: fine-tune top blocks + classifier ────────────────────────────
    # Unfreeze the last 3 EfficientNet feature blocks (6, 7, 8) and classifier.
    # blocks 0-5 stay frozen — they learned low-level features that transfer well.
    print(f"\n── Phase 2: fine-tune top layers ({args.epochs} epochs) ─────────────────────")
    freeze_except(model, ["features.6", "features.7", "features.8", "classifier"])
    opt2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr / 5, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=args.epochs, eta_min=1e-6,
    )

    best_f1      = 0.0
    patience_cnt = 0
    out_path     = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion, opt2, device, train=True)
        vl_loss, vl_f1 = run_epoch(model, val_loader,   criterion, None,  device, train=False)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"  [{epoch:02d}/{args.epochs}]  "
              f"train loss={tr_loss:.4f} f1={tr_f1:.4f}  "
              f"val loss={vl_loss:.4f} f1={vl_f1:.4f}  lr={lr_now:.2e}")

        if vl_f1 > best_f1:
            best_f1      = vl_f1
            patience_cnt = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_f1": vl_f1,
                        "architecture": "efficientnet_b4"}, str(out_path))
            print(f"    ✓ checkpoint saved (val_f1={vl_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

    # ── Threshold calibration ─────────────────────────────────────────────────
    print("\nCalibrating decision threshold on validation set...")
    ckpt = torch.load(str(out_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_thresh = calibrate_threshold(model, val_loader, device)

    torch.save({
        "model_state_dict": model.state_dict(),
        "val_f1":           best_f1,
        "threshold":        best_thresh,
        "architecture":     "efficientnet_b4",
    }, str(out_path))
    print(f"  Final checkpoint saved with threshold={best_thresh:.2f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n── Test set evaluation ───────────────────────────────────────────────────")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            probs = torch.sigmoid(model(imgs.to(device)).squeeze(1))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.long().numpy())

    preds = (np.array(all_probs) >= best_thresh).astype(int)
    print(classification_report(
        all_labels, preds,
        target_names=["CLEAR/OTHER", "BLOCKED"],
        digits=4,
    ))

    return str(out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train EfficientNet-B4 binary drain classifier")
    p.add_argument("--images",        required=True,  help="Root folder with blocked/clear/other subfolders")
    p.add_argument("--out",           default="results/checkpoints/best_model_v2.pt")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--phase1-epochs", type=int,   default=5,    dest="phase1_epochs")
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--patience",      type=int,   default=8)
    p.add_argument("--workers",       type=int,   default=4)
    args = p.parse_args()

    best = train(args)
    print(f"\nDone → {best}")
