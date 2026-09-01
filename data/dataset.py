"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for the drainage blockage
classification task.

Supports two modes:
  1. Manifest mode:
     Reads a CSV with columns [path, label] — images stay in place,
     no file copying needed.

  2. Folder mode:
     Standard ImageFolder layout under processed/train|val|test/blocked|clear.
"""

import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Return the appropriate transform pipeline for a given split.

    Training: random crops + flips + colour jitter.
    Val / Test: deterministic resize only.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet statistics
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Manifest Dataset ───────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """
    Loads images from a CSV manifest (path, label columns).
    Images are read directly from their original location — no copying needed.
    """

    def __init__(self, csv_path: str, transform=None):
        self.transform = transform
        self.records   = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.records.append({
                    "path":  row["path"],
                    "label": int(row["label"]),
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec   = self.records[idx]
        img   = Image.open(rec["path"]).convert("RGB")
        label = rec["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── DataLoader factories ───────────────────────────────────────────────────────

def get_dataloaders_from_manifests(
    manifest_dir: str,
    batch_size:   int = 32,
    img_size:     int = 224,
    num_workers:  int = 4,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders from CSV manifests (manifest mode).
    Use this for the full dataset.

    Args:
        manifest_dir: Directory containing train.csv, val.csv, test.csv.
    """
    manifest_dir = Path(manifest_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        csv_path = manifest_dir / f"{split}.csv"
        dataset  = ManifestDataset(str(csv_path), transform=get_transforms(split, img_size))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        n_blocked = sum(1 for r in dataset.records if r["label"] == 0)
        n_clear   = sum(1 for r in dataset.records if r["label"] == 1)
        print(f"  {split}: {len(dataset)} images | blocked={n_blocked} | clear={n_clear}")

    return loaders


def get_dataloaders(
    processed_dir: str,
    batch_size:    int = 32,
    img_size:      int = 224,
    num_workers:   int = 4,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders from processed/ folder structure (folder mode).
    """
    processed_dir = Path(processed_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = ImageFolder(
            root=str(processed_dir / split),
            transform=get_transforms(split, img_size),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"  {split}: {len(dataset)} images | classes: {dataset.class_to_idx}")

    return loaders


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        open(Path(__file__).parent.parent / "configs" / "config.yaml")
    )

    print("=== Manifest mode (80k) ===")
    loaders = get_dataloaders_from_manifests(
        manifest_dir=cfg["data"]["manifest_dir"],
        batch_size=cfg["training"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    imgs, labels = next(iter(loaders["train"]))
    print(f"Batch shape: {imgs.shape}  Labels: {labels[:8]}")