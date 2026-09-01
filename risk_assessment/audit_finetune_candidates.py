"""
audit_finetune_candidates.py
------------------------------
Two things in one pass, run on hex against images/:

1. For every camera folder, count labeled blocked/clear/other images — the
   real bottleneck for "fine-tune every camera we have data for" (George's
   request). Only folders with a meaningful count in BOTH blocked and clear
   are worth adding to PROBLEM_CAMERAS; folders with zero in one class can't
   train a binary classifier no matter how many images they have of the
   other class.

2. Native image dimensions per folder (sampling a few files from each
   class) — checking George's hypothesis that different cameras have
   different native resolutions/aspect ratios, which get force-squashed to
   224x224 with no aspect-ratio preservation (see transforms.Resize in
   app/inference.py / data/dataset.py / models/finetune_problem_cameras.py).
   If aspect ratios vary a lot across cameras, that's a real, previously
   uninvestigated source of per-camera inconsistency: two visually similar
   scenes at different native aspect ratios get distorted by different
   amounts before the model ever sees them.

Usage (on hex):
    python3 risk_assessment/audit_finetune_candidates.py --images-root images --min-per-class 15
"""

import argparse
from pathlib import Path

from PIL import Image


def sample_dims(folder: Path, n: int = 3) -> list:
    """Return up to n (width, height) pairs sampled from a folder's images."""
    dims = []
    if not folder.exists():
        return dims
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for p in files[:n]:
        try:
            with Image.open(p) as img:
                dims.append(img.size)
        except Exception:
            pass
    return dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default="images")
    ap.add_argument("--min-per-class", type=int, default=15,
                     help="Minimum images needed in BOTH blocked and clear to be worth fine-tuning on.")
    args = ap.parse_args()

    root = Path(args.images_root)
    if not root.exists():
        print(f"{root} does not exist.")
        return

    rows = []
    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir():
            continue
        blocked_dir = cam_dir / "blocked"
        clear_dir   = cam_dir / "clear"
        other_dir   = cam_dir / "other"

        n_blocked = len(list(blocked_dir.glob("*"))) if blocked_dir.exists() else 0
        n_clear   = len(list(clear_dir.glob("*")))   if clear_dir.exists()   else 0
        n_other   = len(list(other_dir.glob("*")))   if other_dir.exists()   else 0

        dims_blocked = sample_dims(blocked_dir)
        dims_clear   = sample_dims(clear_dir)
        all_dims     = dims_blocked + dims_clear
        unique_ratios = sorted(set(round(w / h, 2) for w, h in all_dims)) if all_dims else []

        ready = n_blocked >= args.min_per_class and n_clear >= args.min_per_class

        rows.append({
            "folder": cam_dir.name, "n_blocked": n_blocked, "n_clear": n_clear,
            "n_other": n_other, "dims": all_dims, "ratios": unique_ratios, "ready": ready,
        })

    print(f"{'folder':42s} {'blocked':8s} {'clear':7s} {'other':7s} {'sample dims (w,h)':30s} {'ready?'}")
    print("-" * 120)
    for r in rows:
        dims_str = ", ".join(f"{w}x{h}" for w, h in r["dims"][:3]) or "(no images found)"
        mark = "YES" if r["ready"] else ""
        print(f"{r['folder']:42s} {r['n_blocked']:<8d} {r['n_clear']:<7d} {r['n_other']:<7d} {dims_str:30s} {mark}")

    ready_folders = [r["folder"] for r in rows if r["ready"]]
    print(f"\n{'='*70}")
    print(f"{len(ready_folders)} folders have >= {args.min_per_class} images in BOTH blocked and clear:")
    for f in ready_folders:
        print(f"  {f}")

    print(f"\n--- Aspect ratio check (George's hypothesis) ---")
    all_ratios = set()
    for r in rows:
        all_ratios.update(r["ratios"])
    print(f"Distinct width/height ratios seen across all sampled images: {sorted(all_ratios)}")
    print(f"(All images get force-resized to a square 224x224 with no aspect-ratio preservation")
    print(f" in transforms.Resize — the more these ratios vary, the more some cameras' images")
    print(f" are being stretched/squished by very different amounts before the model sees them.)")

    print(f"\nPython list literal for PROBLEM_CAMERAS (paste into models/finetune_problem_cameras.py,")
    print(f"then cross-check each one against a real live campath before enabling — folder naming")
    print(f"doesn't always match the live camera name/campath 1:1):")
    print("PROBLEM_CAMERAS = [")
    for f in ready_folders:
        print(f'    "{f}",')
    print("]")


if __name__ == "__main__":
    main()
