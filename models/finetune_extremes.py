"""
finetune_extremes.py
--------------------
Targeted fine-tuning of the ResNet-50 classifier on the extreme_cases dataset.

Why this is needed
------------------
The base model misses two classes of hard cases that pre-processing alone
cannot fully rescue:

  1.  Off-centre / upstream debris  (e.g. Ashton Colliters splitter screen —
      debris pile on the LEFT while the screen on the RIGHT still looks open).
  2.  Subtle partial blockages  visible in good-quality daylight images but
      with p_blocked < 0.50 because the model hasn't seen similar patterns.

Strategy
--------
Only the final fully-connected layer (fc) is updated, plus optionally the
last residual block (layer4).  All earlier layers stay frozen.  This keeps
the rich ImageNet + drainage features intact while adapting the decision
boundary to the hard cases.

Dataset layout expected
-----------------------
extreme_cases/
    blocked/                    ← BLOCKED  (today's confirmed live captures)
    block/                      ← BLOCKED  (historical confirmed)
    clear instead of flagged:blocked/  ← BLOCKED  (false negatives to fix)
    clear/                      ← CLEAR

Images in  extreme_cases/other/,  extreme_cases/other 2/,  and the root
are skipped because their labels are ambiguous.

Usage
-----
Run from the dissertation root:

    python models/finetune_extremes.py

Or with custom paths / settings:

    python models/finetune_extremes.py \\
        --checkpoint results/checkpoints/best_model.pt \\
        --extreme-dir extreme_cases \\
        --output     results/checkpoints/best_model_finetuned.pt \\
        --epochs 60  --lr 3e-5 --unfreeze-layer4

The finetuned checkpoint is a drop-in replacement for best_model.pt.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "models"))
from model import get_model


# ── Transforms ────────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class HardQualityAugment:
    """
    Randomly simulates the three quality failure modes seen in the
    'under review' folder — dark/night, overexposed/flare, colour fault.

    Applied with probability `p` per image per epoch, so the model sees
    both clean and degraded versions of every labeled BLOCKED/CLEAR image.
    This teaches robustness to these conditions without requiring labels
    for the ambiguous images themselves.

    Probabilities are kept low (each mode fires ~20% of the time) so the
    model is not overwhelmed by degraded examples.
    """
    def __init__(self, p: float = 0.20):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        import random

        # Dark / night simulation
        if random.random() < self.p:
            factor = random.uniform(0.05, 0.25)   # very dark
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Overexposed / lens-flare simulation
        elif random.random() < self.p:
            factor = random.uniform(3.0, 6.0)     # blown out
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Colour sensor fault (purple/tinted cast)
        elif random.random() < self.p:
            r, g, b = img.split()
            # boost one channel, suppress others → unnatural tint
            ch = random.choice(["r", "g", "b"])
            if ch == "r":
                r = ImageEnhance.Brightness(Image.fromarray(
                    (np.array(r) * 2.2).clip(0, 255).astype(np.uint8))).enhance(1.0)
            elif ch == "b":
                b = ImageEnhance.Brightness(Image.fromarray(
                    (np.array(b) * 2.2).clip(0, 255).astype(np.uint8))).enhance(1.0)
            img = Image.merge("RGB", (r, g, b))

        return img


def train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Standard colour/contrast jitter
        transforms.ColorJitter(brightness=0.5, contrast=0.5,
                               saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.10),
        # Hard quality augmentation — simulates dark/flare/colour-fault frames
        # so the model learns robustness to the same conditions seen in the
        # 'under review' folder, without needing labels for those images.
        HardQualityAugment(p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ExtremeDataset(Dataset):
    """
    Builds a labelled dataset from the extreme_cases directory.
    Skips ambiguous folders (other/, other 2/) and root-level images.
    """

    BLOCKED_DIRS = [
        "blocked",
        "block",
        "clear instead of flagged:blocked",  # legacy name kept
        "blocked_night",       # dark / night images of blocked drains
        "blocked_flare",       # overexposed / sun-flare images of blocked drains
        "blocked_offcentre",   # debris pile off-centre (e.g. left of frame, screen clear)
        "blocked_sediment",    # sediment / murky water at screen base
        "blocked_colour",      # sensor colour-fault images of blocked drains
    ]
    CLEAR_DIRS = [
        "clear",
        "clear_vegetation",    # clear drains with nearby vegetation (eliminates false positives)
        "clear_night",         # night images of clear drains
        "clear_flare",         # overexposed images of clear drains
    ]
    # Soft label 0.5 — images that should be FLAGGED (needs review).
    # Training with target=0.5 teaches the model to output p ≈ 0.5 for
    # these visual patterns (flood scenes, IR faults, offline placeholders)
    # so they naturally land in the FLAGGED zone without hard-coded rules.
    # The model architecture stays binary (one sigmoid output) — only the
    # training target changes.
    FLAGGED_DIRS = [
        "under review",
    ]
    SOFT_LABEL   = 0.5
    # Hard-labeled (BLOCKED/CLEAR) examples get 3× more gradient weight
    # than soft-label examples so 102 "under review" images don't overwhelm
    # 16 blocked + 15 clear and push the model toward always outputting 0.5.
    HARD_WEIGHT  = 3.0
    # Cap how many soft-label examples are used per run so they never
    # outnumber the hard-labeled pool by more than 2:1.
    MAX_FLAGGED  = 60

    def __init__(self, extreme_dir: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, float]] = []   # (path, label)
        self.weights:  list[float]             = []   # per-sample loss weight

        for folder in self.BLOCKED_DIRS:
            d = extreme_dir / folder
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXTS:
                        self.samples.append((p, 1.0))
                        self.weights.append(self.HARD_WEIGHT)

        for folder in self.CLEAR_DIRS:
            d = extreme_dir / folder
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXTS:
                        self.samples.append((p, 0.0))
                        self.weights.append(self.HARD_WEIGHT)

        flagged_paths = []
        for folder in self.FLAGGED_DIRS:
            d = extreme_dir / folder
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXTS:
                        flagged_paths.append(p)
        # Subsample so soft-label images don't overwhelm hard-labeled ones
        random.shuffle(flagged_paths)
        for p in flagged_paths[:self.MAX_FLAGGED]:
            self.samples.append((p, self.SOFT_LABEL))
            self.weights.append(1.0)

        # Shuffle together
        combined = list(zip(self.samples, self.weights))
        random.shuffle(combined)
        self.samples, self.weights = zip(*combined) if combined else ([], [])
        self.samples = list(self.samples)
        self.weights = list(self.weights)

        blocked_n = sum(1 for _, l in self.samples if l == 1.0)
        clear_n   = sum(1 for _, l in self.samples if l == 0.0)
        flagged_n = sum(1 for _, l in self.samples if l == self.SOFT_LABEL)
        print(f"  Dataset: {len(self.samples)} images  "
              f"({blocked_n} blocked  {clear_n} clear  {flagged_n} flagged/soft)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        weight      = self.weights[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    model = get_model(pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Freeze / unfreeze layers ──────────────────────────────────────────────
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze final FC
    for p in model.fc.parameters():
        p.requires_grad = True
    print("  Unfrozen: fc")

    # Optionally unfreeze layer4 for more capacity
    if args.unfreeze_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True
        print("  Unfrozen: layer4")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  "
          f"({100*trainable/total:.1f} %)")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    extreme_dir = Path(args.extreme_dir)
    dataset = ExtremeDataset(extreme_dir, transform=train_transform())

    if len(dataset) < 4:
        raise ValueError("Too few extreme-case images found — "
                         "check --extreme-dir path.")

    # For such small datasets use the whole set for training.
    # Validation is done on a held-out 20% split.
    n_val   = max(1, int(0.20 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Use deterministic transform for validation
    val_set.dataset = ExtremeDataset(extreme_dir, transform=val_transform())
    val_set.indices = val_set.indices   # keep same split

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Class imbalance correction ────────────────────────────────────────────
    # pos_weight balances BLOCKED vs CLEAR hard-labeled examples only.
    # Soft-label (FLAGGED) examples are excluded from this ratio.
    n_blocked = sum(1 for _, l in dataset.samples if l == 1.0)
    n_clear   = sum(1 for _, l in dataset.samples if l == 0.0)
    pos_weight = torch.tensor([n_clear / max(n_blocked, 1)],
                              dtype=torch.float32).to(device)
    # reduction="none" so we can apply per-sample weights manually.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning for {args.epochs} epochs  (lr={args.lr})  "
          f"→ {output_path}\n")

    for epoch in range(1, args.epochs + 1):
        # ─ Train ─
        model.train()
        train_loss = 0.0
        for imgs, labels, sample_w in train_loader:
            imgs     = imgs.to(device)
            labels   = labels.unsqueeze(1).to(device)
            sample_w = sample_w.unsqueeze(1).to(device)
            optimizer.zero_grad()
            # Per-sample weighted loss: hard-labeled examples (weight=3)
            # dominate over soft-label FLAGGED examples (weight=1).
            loss = (criterion(model(imgs), labels) * sample_w).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ─ Validate ─
        model.eval()
        val_loss  = 0.0
        correct   = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels, sample_w in val_loader:
                imgs     = imgs.to(device)
                labels   = labels.unsqueeze(1).to(device)
                sample_w = sample_w.unsqueeze(1).to(device)
                logits   = model(imgs)
                val_loss += (criterion(logits, labels) * sample_w).mean().item()
                # Accuracy: soft-label targets are "correct" if p lands in [0.35, 0.65]
                preds = torch.sigmoid(logits)
                hard_mask = (labels != 0.5)
                soft_mask = ~hard_mask
                correct += ((preds[hard_mask] > 0.5).float() == labels[hard_mask]).sum().item()
                correct += ((preds[soft_mask] >= 0.35) & (preds[soft_mask] <= 0.65)).sum().item()
                total_val += labels.numel()
        val_loss /= max(len(val_loader), 1)
        val_acc   = correct / max(total_val, 1)

        scheduler.step()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "finetune_source":  "extreme_cases",
            }, output_path)
            marker = "  ← saved"

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.2%}{marker}")

    print(f"\nDone.  Best val_loss={best_val_loss:.4f}")
    print(f"Saved → {output_path}")
    print("\nTo use the fine-tuned model, update your FastAPI startup to point")
    print(f"checkpoint_path to: {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on extreme cases")
    parser.add_argument("--checkpoint",      default="results/checkpoints/best_model.pt",
                        help="Path to existing best_model.pt checkpoint")
    parser.add_argument("--extreme-dir",     default="extreme_cases",
                        help="Path to the extreme_cases directory")
    parser.add_argument("--output",          default="results/checkpoints/best_model_finetuned.pt",
                        help="Where to save the fine-tuned checkpoint")
    parser.add_argument("--epochs",          type=int,   default=60)
    parser.add_argument("--lr",              type=float, default=3e-5,
                        help="Learning rate (keep very low to avoid catastrophic forgetting)")
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--unfreeze-layer4", action="store_true",
                        help="Also unfreeze the last residual block for more capacity")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    finetune(args)
