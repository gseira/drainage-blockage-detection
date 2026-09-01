"""
build_manifest.py
-----------------
Scans the full images/ directory and builds train/val/test CSV manifests.

Each site folder has the structure:
    images/<site_name>/blocked/*.jpg
    images/<site_name>/clear/*.jpg
    images/<site_name>/other/         ← ignored

Output CSVs (saved to data/manifests/):
    train.csv, val.csv, test.csv
    Columns: path (absolute), label (0=blocked, 1=clear)

Usage:
    python src/build_manifest.py
    python src/build_manifest.py --images_dir /homes/ges65/dissertation/images
"""

import argparse
import csv
import random
from pathlib import Path

LABEL_MAP   = {"blocked": 0, "clear": 1}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(images_dir: Path) -> list[dict]:
    """Walk all site subdirs and collect blocked/clear image paths + labels."""
    records = []
    for site_dir in sorted(images_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for cls_name, label in LABEL_MAP.items():
            cls_dir = site_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTS:
                    records.append({"path": str(img_path), "label": label})
    return records


def split_records(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """Stratified split into train / val / test."""
    random.seed(seed)

    blocked = [r for r in records if r["label"] == 0]
    clear   = [r for r in records if r["label"] == 1]

    splits = {}
    for cls_name, subset in [("blocked", blocked), ("clear", clear)]:
        random.shuffle(subset)
        n        = len(subset)
        n_train  = int(n * train_ratio)
        n_val    = int(n * val_ratio)
        splits[cls_name] = {
            "train": subset[:n_train],
            "val":   subset[n_train:n_train + n_val],
            "test":  subset[n_train + n_val:],
        }
        print(f"  {cls_name}: {n} total → "
              f"{n_train} train / {int(n*val_ratio)} val / "
              f"{n - n_train - int(n*val_ratio)} test")

    train = splits["blocked"]["train"] + splits["clear"]["train"]
    val   = splits["blocked"]["val"]   + splits["clear"]["val"]
    test  = splits["blocked"]["test"]  + splits["clear"]["test"]

    random.shuffle(train)
    return train, val, test


def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records):>6} records → {path}")


def main(images_dir: str, manifest_dir: str) -> None:
    images_dir   = Path(images_dir)
    manifest_dir = Path(manifest_dir)

    print(f"Scanning {images_dir} ...")
    records = collect_images(images_dir)
    print(f"Found {len(records):,} images "
          f"({sum(1 for r in records if r['label']==0):,} blocked, "
          f"{sum(1 for r in records if r['label']==1):,} clear)\n")

    print("Splitting (70 / 15 / 15, stratified, seed=42):")
    train, val, test = split_records(records)

    print("\nSaving manifests:")
    save_csv(train, manifest_dir / "train.csv")
    save_csv(val,   manifest_dir / "val.csv")
    save_csv(test,  manifest_dir / "test.csv")

    print("\nDone. Class balance check:")
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        n_blocked = sum(1 for r in split if r["label"] == 0)
        n_clear   = sum(1 for r in split if r["label"] == 1)
        pct = n_blocked / len(split) * 100
        print(f"  {split_name}: {len(split):>6} images | "
              f"blocked={n_blocked} ({pct:.1f}%) | clear={n_clear}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",   default="/homes/ges65/dissertation/images")
    parser.add_argument("--manifest_dir", default="/homes/ges65/dissertation/data/manifests")
    args = parser.parse_args()
    main(args.images_dir, args.manifest_dir)