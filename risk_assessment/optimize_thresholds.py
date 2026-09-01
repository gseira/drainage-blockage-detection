"""
optimize_thresholds.py
------------------------
Finds clear_below/blocked_above thresholds empirically instead of guessing,
by running the deployed model against every labeled image in images/*/blocked
and images/*/clear (real human-labeled ground truth, not synthetic), then
sweeping threshold combinations to see which ones actually earn their
confidence.

For every candidate (clear_below, blocked_above) pair, reports:
  - confident_blocked_precision: of images the pair would confidently call
    BLOCKED, what fraction are ACTUALLY labeled blocked
  - confident_clear_precision:   of images the pair would confidently call
    CLEAR, what fraction are ACTUALLY labeled clear
  - flagged_rate: what fraction of all images fall in neither confident zone
    (i.e. would go to FLAGGED / human review)

There is no single "optimal" answer — tightening thresholds raises precision
but raises the flagged rate (more manual review work); loosening does the
opposite. This reports the best options at a few different flagged-rate
budgets so the actual tradeoff is visible, rather than picking one number
silently.

Usage (on hex — needs torch):
    python3 risk_assessment/optimize_thresholds.py \\
        --checkpoint results/checkpoints/best_model.pt \\
        --images-root images

Optional:
    --max-per-class 400   (cap images per camera per class, for speed)
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.inference import InferencePipeline  # noqa: E402


def collect_labeled_images(images_root: Path, max_per_class: int) -> list:
    """Returns list of (path, true_label) where true_label is 1=blocked, 0=clear."""
    samples = []
    for cam_dir in sorted(images_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        for label, cls in [(1, "blocked"), (0, "clear")]:
            cls_dir = cam_dir / cls
            if not cls_dir.exists():
                continue
            files = sorted([p for p in cls_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
            files = files[:max_per_class]
            for p in files:
                samples.append((p, label))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints" / "best_model.pt"))
    ap.add_argument("--images-root", default=str(ROOT / "images"))
    ap.add_argument("--max-per-class", type=int, default=400,
                     help="Cap images per camera per class, to keep runtime reasonable.")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    samples = collect_labeled_images(images_root, args.max_per_class)
    n_blocked = sum(1 for _, l in samples if l == 1)
    n_clear   = sum(1 for _, l in samples if l == 0)
    print(f"Evaluating on {len(samples)} labeled images  (blocked={n_blocked}, clear={n_clear})")
    print(f"Checkpoint: {args.checkpoint}\n")

    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint, use_llm=False, use_tta=False, preserve_aspect=False,
    )

    # ── Run inference once per image, keep the raw p_blocked ──────────────
    results = []
    for i, (path, true_label) in enumerate(samples):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}")
            continue
        r = pipeline.run(img, validate_scene=False)
        p_blocked = r.get("p_blocked_raw") if r.get("p_blocked_raw") is not None else r.get("p_blocked")
        results.append((p_blocked, true_label))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(samples)} done")

    print(f"\nDone running inference on {len(results)} images.\n")

    # ── Sweep threshold combinations ───────────────────────────────────────
    candidates = []
    steps = [round(x * 0.05, 2) for x in range(1, 20)]  # 0.05, 0.10, ..., 0.95
    for clear_below in steps:
        for blocked_above in steps:
            if blocked_above <= clear_below:
                continue
            n_conf_blocked = n_correct_blocked = 0
            n_conf_clear   = n_correct_clear   = 0
            n_flagged = 0
            for p, true_label in results:
                if p <= clear_below:
                    n_conf_clear += 1
                    if true_label == 0:
                        n_correct_clear += 1
                elif p >= blocked_above:
                    n_conf_blocked += 1
                    if true_label == 1:
                        n_correct_blocked += 1
                else:
                    n_flagged += 1

            if n_conf_blocked == 0 or n_conf_clear == 0:
                continue  # degenerate — one confident class never fires

            prec_blocked = n_correct_blocked / n_conf_blocked
            prec_clear   = n_correct_clear / n_conf_clear
            flagged_rate = n_flagged / len(results)

            candidates.append({
                "clear_below": clear_below, "blocked_above": blocked_above,
                "prec_blocked": prec_blocked, "prec_clear": prec_clear,
                "flagged_rate": flagged_rate,
                "n_conf_blocked": n_conf_blocked, "n_conf_clear": n_conf_clear,
            })

    # ── Report: best precision achievable at a few flagged-rate budgets ────
    print(f"{'='*90}")
    print("Best threshold pairs at different flagged-rate budgets")
    print("(precision = of images confidently called X, what fraction actually ARE X)")
    print(f"{'='*90}")
    for budget in [0.10, 0.20, 0.30, 0.40, 0.50]:
        pool = [c for c in candidates if c["flagged_rate"] <= budget]
        if not pool:
            print(f"\nFlagged rate <= {budget:.0%}: no valid threshold pair found")
            continue
        # Rank by the WORSE of the two precisions (avoid one-sided wins)
        best = max(pool, key=lambda c: min(c["prec_blocked"], c["prec_clear"]))
        print(f"\nFlagged rate <= {budget:.0%}:")
        print(f"  clear_below={best['clear_below']:.2f}  blocked_above={best['blocked_above']:.2f}")
        print(f"  confident-BLOCKED precision: {best['prec_blocked']:.1%}  (n={best['n_conf_blocked']})")
        print(f"  confident-CLEAR precision:   {best['prec_clear']:.1%}  (n={best['n_conf_clear']})")
        print(f"  actual flagged rate:         {best['flagged_rate']:.1%}")

    # ── Also report the current (0.25, 0.85) setting for direct comparison ──
    current = next((c for c in candidates
                     if c["clear_below"] == 0.25 and c["blocked_above"] == 0.85), None)
    print(f"\n{'='*90}")
    if current:
        print(f"Current deployed setting (clear_below=0.25, blocked_above=0.85):")
        print(f"  confident-BLOCKED precision: {current['prec_blocked']:.1%}  (n={current['n_conf_blocked']})")
        print(f"  confident-CLEAR precision:   {current['prec_clear']:.1%}  (n={current['n_conf_clear']})")
        print(f"  flagged rate:                {current['flagged_rate']:.1%}")
    else:
        print("Current (0.25, 0.85) setting produced a degenerate result on this data "
              "(one confident class never fired) — check the raw sweep above.")


if __name__ == "__main__":
    main()
