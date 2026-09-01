"""
train_3class.py
---------------
Fine-tune ResNet-50 for 3-class drainage classification.

Classes:
    0 = BLOCKED  —  drain is blocked, needs dispatch
    1 = CLEAR    —  drain is clear, no action
    2 = OTHER    —  not a drainage image / ambiguous, flag for review

Strategy
--------
Starts from best_model.pt (the existing binary ResNet-50).
All convolutional weights transfer directly; only the FC head (1 → 3 outputs)
is replaced with a fresh Linear(2048, 3).

Phase 1  (default 5 epochs)  : freeze backbone, train FC head only.
Phase 2  (default 25 epochs) : unfreeze layer3 + layer4 + FC, lower LR.

Mixed-precision (AMP) is used automatically when a GPU is available.
Best checkpoint is saved by highest validation macro-F1 (not loss) because
the dataset is class-imbalanced.

Usage on hex
------------
  # via SLURM (recommended):
  sbatch slurm/train_3class.slurm

  # or directly:
  python models/train_3class.py \\
      --images      ~/dissertation/images \\
      --checkpoint  ~/dissertation/results/checkpoints/best_model.pt \\
      --out         ~/dissertation/results/checkpoints/best_model_3class.pt

Expected images/ structure:
  images/
    CameraName1/
      blocked/   *.jpg
      clear/     *.jpg
      other/     *.jpg
    CameraName2/
      ...
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_TO_IDX = {"blocked": 0, "clear": 1, "other": 2}
IDX_TO_NAME  = {0: "BLOCKED", 1: "CLEAR", 2: "OTHER"}

# Inverse-frequency class weights (normalised so CLEAR = 1.0).
# Based on: BLOCKED=11 591  CLEAR=42 803  OTHER=26 058
CLASS_WEIGHTS = [3.69, 1.0, 1.64]   # [blocked, clear, other]

IMG_SIZE = 224
SEED     = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

def _collect_samples(images_root: Path) -> list:
    """
    Walk images/CameraName/{blocked,clear,other}/ and return
    a list of (absolute_path_str, class_idx) tuples.
    """
    samples = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for camera_dir in sorted(images_root.iterdir()):
        if not camera_dir.is_dir():
            continue
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = camera_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    samples.append((str(img_path), class_idx))

    return samples


def _stratified_split(
    samples: list,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> tuple:
    """
    Stratified image-level split preserving per-class proportions.
    Returns (train, val, test) sample lists.
    """
    rng = random.Random(seed)

    by_class: dict = {0: [], 1: [], 2: []}
    for s in samples:
        by_class[s[1]].append(s)

    train_s, val_s, test_s = [], [], []
    for cls_idx in sorted(by_class):
        items = by_class[cls_idx]
        rng.shuffle(items)
        n       = len(items)
        n_test  = int(n * test_frac)
        n_val   = int(n * val_frac)
        test_s  += items[:n_test]
        val_s   += items[n_test : n_test + n_val]
        train_s += items[n_test + n_val :]

    rng.shuffle(train_s)
    return train_s, val_s, test_s


class DrainageDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt image — return black frame with correct label
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


def _get_transforms():
    """Training augmentations + inference-time val/test transforms."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Model ─────────────────────────────────────────────────────────────────────

def build_3class_model(binary_ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Build a 3-class ResNet-50 and copy backbone weights from the binary checkpoint.

    The binary model has fc = Linear(2048, 1).
    We replace it with fc = Linear(2048, 3) and copy all other weights.
    This gives us a backbone that already understands drainage images.
    """
    # Architecture
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features   # 2048
    model.fc = nn.Linear(in_features, 3)

    # Load binary checkpoint
    ckpt  = torch.load(binary_ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Copy weights layer-by-layer, skipping the FC (shape mismatch)
    own_state = model.state_dict()
    loaded    = 0
    skipped   = []
    for name, param in state.items():
        if name.startswith("fc."):
            skipped.append(name)
            continue
        if name in own_state and own_state[name].shape == param.shape:
            own_state[name].copy_(param)
            loaded += 1
        else:
            skipped.append(name)

    model.load_state_dict(own_state)
    print(f"  Loaded {loaded} weight tensors from binary checkpoint.")
    if skipped:
        print(f"  Skipped (shape mismatch / new head): {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))

    return model.to(device)


def freeze_except(model: nn.Module, layers_to_train: list) -> None:
    """Freeze all parameters except those in named layers."""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in layers_to_train)


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer,
    device:    torch.device,
    scaler,
    is_train:  bool,
) -> dict:
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_preds:  list = []
    all_labels: list = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)           # (B, 3)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item() * imgs.size(0)
            preds        = logits.argmax(dim=1).cpu().tolist()
            all_preds   += preds
            all_labels  += labels.cpu().tolist()

            # Print progress every 200 batches
            if (batch_idx + 1) % 200 == 0:
                print(f"    batch {batch_idx+1}/{len(loader)}  "
                      f"loss={loss.item():.4f}", flush=True)

    n        = len(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls_f1 = f1_score(all_labels, all_preds, average=None,
                          labels=[0, 1, 2], zero_division=0)
    return {
        "loss":      total_loss / n,
        "macro_f1":  macro_f1,
        "f1_blocked": float(per_cls_f1[0]),
        "f1_clear":   float(per_cls_f1[1]),
        "f1_other":   float(per_cls_f1[2]),
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(args) -> None:
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nCollecting samples...")
    samples = _collect_samples(Path(args.images))

    counts = {0: 0, 1: 0, 2: 0}
    for _, c in samples:
        counts[c] += 1
    print(f"  BLOCKED={counts[0]:,}  CLEAR={counts[1]:,}  OTHER={counts[2]:,}"
          f"  TOTAL={len(samples):,}")

    train_s, val_s, test_s = _stratified_split(samples)
    print(f"  Split  → Train={len(train_s):,}  Val={len(val_s):,}  Test={len(test_s):,}")

    train_tf, val_tf = _get_transforms()
    train_ds = DrainageDataset(train_s, train_tf)
    val_ds   = DrainageDataset(val_s,   val_tf)
    test_ds  = DrainageDataset(test_s,  val_tf)

    pin    = args.workers > 0   # pin_memory requires workers > 0 on some systems
    mp_ctx = "spawn" if args.workers > 0 else None  # avoid CUDA fork deadlock
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
        multiprocessing_context=mp_ctx,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
        multiprocessing_context=mp_ctx,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        multiprocessing_context=mp_ctx,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nBuilding 3-class model from: {args.checkpoint}")
    model = build_3class_model(args.checkpoint, device)

    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = False   # disabled — GradScaler causes hang on this system; A5000 is fast without it
    scaler  = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "train_3class_log.csv"

    best_f1   = 0.0
    log_rows: list = []

    # ── Phase 1: warm up the new FC head ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: FC head warm-up ({args.phase1_epochs} epochs, lr=1e-3)")
    print("="*60)
    freeze_except(model, ["fc"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Loading first batch (may be slow on network filesystem)...", flush=True)

    p1_optim = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    print(f"Optimizer ready. Starting training loop...", flush=True)

    for epoch in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, p1_optim, device, scaler, True)
        vl = run_epoch(model, val_loader,   criterion, None,      device, None,   False)
        elapsed = time.time() - t0

        print(
            f"  P1 Ep {epoch:02d} | "
            f"tr_loss={tr['loss']:.4f}  tr_f1={tr['macro_f1']:.4f}  "
            f"[blk={tr['f1_blocked']:.3f} clr={tr['f1_clear']:.3f} oth={tr['f1_other']:.3f}] | "
            f"val_loss={vl['loss']:.4f}  val_f1={vl['macro_f1']:.4f}  "
            f"[blk={vl['f1_blocked']:.3f} clr={vl['f1_clear']:.3f} oth={vl['f1_other']:.3f}]  "
            f"[{elapsed:.0f}s]"
        )

        row = {
            "phase": 1, "epoch": epoch,
            "tr_loss": tr["loss"], "tr_macro_f1": tr["macro_f1"],
            "tr_f1_blocked": tr["f1_blocked"], "tr_f1_clear": tr["f1_clear"], "tr_f1_other": tr["f1_other"],
            "val_loss": vl["loss"], "val_macro_f1": vl["macro_f1"],
            "val_f1_blocked": vl["f1_blocked"], "val_f1_clear": vl["f1_clear"], "val_f1_other": vl["f1_other"],
            "lr": 1e-3,
        }
        log_rows.append(row)

        if vl["macro_f1"] > best_f1:
            best_f1 = vl["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":           epoch,
                "phase":           1,
                "val_macro_f1":    best_f1,
                "num_classes":     3,
                "class_to_idx":    CLASS_TO_IDX,
            }, out_path)
            print(f"    ✓ Checkpoint saved (val_macro_f1={best_f1:.4f})")

    # ── Phase 2: fine-tune layer3 + layer4 + FC ───────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: backbone fine-tune ({args.epochs} epochs, lr={args.lr})")
    print("="*60)
    freeze_except(model, ["layer3", "layer4", "fc"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    p2_optim = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        p2_optim,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,         # 10% warmup
        anneal_strategy="cos",
    )

    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, p2_optim, device, scaler, True)
        vl = run_epoch(model, val_loader,   criterion, None,      device, None,   False)
        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(
            f"  P2 Ep {epoch:02d} | "
            f"tr_loss={tr['loss']:.4f}  tr_f1={tr['macro_f1']:.4f}  "
            f"[blk={tr['f1_blocked']:.3f} clr={tr['f1_clear']:.3f} oth={tr['f1_other']:.3f}] | "
            f"val_loss={vl['loss']:.4f}  val_f1={vl['macro_f1']:.4f}  "
            f"[blk={vl['f1_blocked']:.3f} clr={vl['f1_clear']:.3f} oth={vl['f1_other']:.3f}]  "
            f"lr={lr:.2e}  [{elapsed:.0f}s]"
        )

        row = {
            "phase": 2, "epoch": epoch,
            "tr_loss": tr["loss"], "tr_macro_f1": tr["macro_f1"],
            "tr_f1_blocked": tr["f1_blocked"], "tr_f1_clear": tr["f1_clear"], "tr_f1_other": tr["f1_other"],
            "val_loss": vl["loss"], "val_macro_f1": vl["macro_f1"],
            "val_f1_blocked": vl["f1_blocked"], "val_f1_clear": vl["f1_clear"], "val_f1_other": vl["f1_other"],
            "lr": lr,
        }
        log_rows.append(row)

        if vl["macro_f1"] > best_f1:
            best_f1 = vl["macro_f1"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":           epoch,
                "phase":           2,
                "val_macro_f1":    best_f1,
                "num_classes":     3,
                "class_to_idx":    CLASS_TO_IDX,
            }, out_path)
            print(f"    ✓ Checkpoint saved (val_macro_f1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    · No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"\n⚠  Early stopping triggered at epoch {epoch}.")
                break

    # ── Final test evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final Test Evaluation (loading best checkpoint)")
    print("="*60)
    best_ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    te = run_epoch(model, test_loader, criterion, None, device, None, False)

    print(f"Test macro-F1 : {te['macro_f1']:.4f}")
    print(f"Test loss     : {te['loss']:.4f}")
    print()
    print(classification_report(
        te["labels"], te["preds"],
        target_names=["BLOCKED", "CLEAR", "OTHER"],
        digits=4,
    ))

    # ── Save training log ─────────────────────────────────────────────────
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Training log : {log_path}")

    print(f"\nDone.  Best val macro-F1 = {best_f1:.4f}")
    print(f"Model        : {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-class drainage ResNet-50 fine-tuning"
    )
    parser.add_argument(
        "--images", required=True,
        help="Path to images/ root directory (contains one folder per camera)"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the existing binary best_model.pt"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for the 3-class checkpoint (e.g. results/checkpoints/best_model_3class.pt)"
    )
    parser.add_argument("--epochs",        type=int,   default=25,   help="Phase 2 epochs")
    parser.add_argument("--phase1-epochs", type=int,   default=5,    help="Phase 1 head warm-up epochs")
    parser.add_argument("--batch",         type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",            type=float, default=3e-4, help="Phase 2 peak learning rate")
    parser.add_argument("--patience",      type=int,   default=8,    help="Early stopping patience (epochs)")
    parser.add_argument("--workers",       type=int,   default=4,    help="DataLoader worker processes")

    args = parser.parse_args()
    train(args)
