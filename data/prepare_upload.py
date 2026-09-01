"""
prepare_upload.py
-----------------
Creates a flat folder of upload-ready images with location + label
prefixed to each filename.

Source structure (on HEX):
    images/<LocationName>/blocked/<timestamp>.jpg
    images/<LocationName>/clear/<timestamp>.jpg

Output structure:
    upload_ready/<LocationName>_blocked_<timestamp>.jpg
    upload_ready/<LocationName>_clear_<timestamp>.jpg

Nothing is renamed or deleted — only copies are made.

Usage:
    # All locations, all labels:
    python data/prepare_upload.py

    # Specific location only:
    python data/prepare_upload.py --location Cornwall_PorthlevenScree

    # Specific location + label:
    python data/prepare_upload.py --location Cornwall_PorthlevenScree --label blocked

    # Custom output folder:
    python data/prepare_upload.py --out ~/Desktop/upload_ready
"""

import argparse
import shutil
from pathlib import Path


def prepare(images_dir: Path, out_dir: Path, location: str = None, label: str = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect location folders
    locations = (
        [images_dir / location] if location else sorted(images_dir.iterdir())
    )

    copied = 0
    skipped = 0

    for loc_path in locations:
        if not loc_path.is_dir():
            continue
        loc_name = loc_path.name

        # Collect label subfolders (blocked / clear / other)
        labels = [loc_path / label] if label else sorted(loc_path.iterdir())

        for label_path in labels:
            if not label_path.is_dir():
                continue
            label_name = label_path.name
            if label_name not in ("blocked", "clear", "other"):
                continue

            for img_file in sorted(label_path.glob("*.jpg")):
                new_name = f"{loc_name}_{label_name}_{img_file.name}"
                dest     = out_dir / new_name

                if dest.exists():
                    skipped += 1
                    continue

                shutil.copy2(img_file, dest)
                copied += 1

    print(f"\nDone.")
    print(f"  Copied:  {copied} images  →  {out_dir}")
    print(f"  Skipped: {skipped} (already existed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images",
                        help="Path to images/ directory (default: images/)")
    parser.add_argument("--out",    default="upload_ready",
                        help="Output folder (default: upload_ready/)")
    parser.add_argument("--location", default=None,
                        help="Restrict to one location folder (e.g. Cornwall_PorthlevenScree)")
    parser.add_argument("--label",    default=None,
                        help="Restrict to one label: blocked | clear | other")
    args = parser.parse_args()

    prepare(
        images_dir = Path(args.images),
        out_dir    = Path(args.out),
        location   = args.location,
        label      = args.label,
    )
