"""
test_colour_consistency.py
---------------------------
Measures whether the classifier is brittle to small, realistic colour/
lighting drift — the kind of variation that happens naturally between two
scans of the same physical scene (auto-exposure, JPEG re-encoding, slight
white-balance shifts) even when nothing about the drain actually changed.

For every labeled image in extreme_cases/block and extreme_cases/clear:
  1. Run inference on the original image.
  2. Apply a fixed, modest colour perturbation (brightness/contrast/
     saturation/hue) and run inference again.
  3. Record whether the prediction flipped, and how close the ORIGINAL
     prediction was to the decision boundary (its "margin").

If flips are common and concentrated on low-margin (near-boundary) images,
that confirms the "same scene, different day, different verdict" symptom is
really about decision-boundary sensitivity — supporting the confidence-
margin routing added in InferencePipeline (predictions within `margin` of
the boundary are now routed to FLAGGED instead of confidently asserted).

This does NOT run in the Claude sandbox this was written in — it needs
torch and a real checkpoint. Run it yourself:

    python risk_assessment/test_colour_consistency.py \\
        --checkpoint results/checkpoints/best_model.pt

Optional flags:
    --dirs block clear          (default: both)
    --margin 0.10                (must match the InferencePipeline margin
                                   you're testing, purely for the report —
                                   doesn't change how flips are counted)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.inference import InferencePipeline  # noqa: E402


# ── The perturbation ────────────────────────────────────────────────────────
#
# Deliberately modest and deterministic (not random) so the test is exactly
# reproducible: a small brightness lift, a touch more contrast, a noticeable
# but plausible saturation boost, and a slight hue rotation. This is meant to
# resemble ordinary capture-to-capture drift, not an adversarial worst case.

def perturb(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Color(img).enhance(1.15)

    # Slight hue rotation via HSV
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h).astype(np.int16)
    h_arr = (h_arr + 6) % 256  # ~+8 degrees on a 0-255 hue wheel
    h = Image.fromarray(h_arr.astype("uint8"), mode="L")
    img = Image.merge("HSV", (h, s, v)).convert("RGB")
    return img


def margin_bucket(margin: float) -> str:
    if margin < 0.05:  return "0.00-0.05 (razor's edge)"
    if margin < 0.15:  return "0.05-0.15 (borderline)"
    if margin < 0.30:  return "0.15-0.30 (moderate)"
    return "0.30+ (confident)"


def compute_margin(result: dict, threshold: float, num_classes: int) -> float:
    p_blocked = result.get("p_blocked_raw") if result.get("p_blocked_raw") is not None else result.get("p_blocked")
    p_clear   = result.get("p_clear")
    if num_classes == 1 or p_clear is None:
        return abs((p_blocked or 0.0) - threshold)
    return abs((p_blocked or 0.0) - p_clear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints" / "best_model.pt"))
    ap.add_argument("--extreme-dir", default=str(ROOT / "extreme_cases"))
    ap.add_argument("--dirs", nargs="+", default=["block", "clear"])
    ap.add_argument("--margin", type=float, default=0.10, help="For reporting only — must match the pipeline's own margin to interpret FLAGGED-rate numbers correctly.")
    ap.add_argument("--no-tta", action="store_true",
                     help="Disable test-time averaging (InferencePipeline's use_tta) to get "
                          "the single-pass baseline, for a controlled before/after comparison "
                          "against the default (TTA-on) run.")
    ap.add_argument("--no-llm", action="store_true",
                     help="Skip the Groq LLM explanation call entirely (this test never reads "
                          "the explanation text anyway) — avoids burning API quota and rate-limit "
                          "slowdowns on larger batches.")
    args = ap.parse_args()

    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        use_tta=not args.no_tta,
        use_llm=not args.no_llm,
    )
    print(f"TTA (test-time averaging): {'OFF — single-pass baseline' if args.no_tta else 'ON'}")
    print(f"LLM explanations:          {'OFF' if args.no_llm else 'ON'}")

    rows = []
    for subdir in args.dirs:
        folder = Path(args.extreme_dir) / subdir
        if not folder.exists():
            print(f"  [skip] {folder} does not exist")
            continue
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        print(f"\n=== {subdir} ({len(images)} images) ===")
        for path in images:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:
                print(f"  [error] could not open {path.name}: {exc}")
                continue

            orig_result = pipeline.run(img)
            pert_result = pipeline.run(perturb(img))

            margin = compute_margin(orig_result, pipeline.threshold, pipeline.num_classes)
            flipped = orig_result["prediction"] != pert_result["prediction"]

            rows.append({
                "file":    path.name,
                "label":   subdir,
                "orig":    orig_result["prediction"],
                "pert":    pert_result["prediction"],
                "margin":  margin,
                "flipped": flipped,
            })

            flag = "  <-- FLIPPED" if flipped else ""
            print(f"  {path.name:45s} orig={orig_result['prediction']:8s} "
                  f"pert={pert_result['prediction']:8s} margin={margin:.3f}{flag}")

    if not rows:
        print("\nNo images found — check --extreme-dir and --dirs.")
        return

    # ── Summary ──
    total = len(rows)
    n_flipped = sum(r["flipped"] for r in rows)
    print(f"\n{'='*60}")
    print(f"TOTAL: {n_flipped}/{total} predictions flipped under a mild colour "
          f"perturbation ({n_flipped/total:.1%})")

    print(f"\nFlip rate by original decision margin (this is the key result — "
          f"\nif flips concentrate in the low-margin rows, colour sensitivity "
          f"\nnear the boundary is confirmed as the driver):")
    buckets: dict = {}
    for r in rows:
        b = margin_bucket(r["margin"])
        buckets.setdefault(b, {"total": 0, "flipped": 0})
        buckets[b]["total"] += 1
        buckets[b]["flipped"] += r["flipped"]
    for b in ["0.00-0.05 (razor's edge)", "0.05-0.15 (borderline)", "0.15-0.30 (moderate)", "0.30+ (confident)"]:
        if b in buckets:
            t, f = buckets[b]["total"], buckets[b]["flipped"]
            print(f"  {b:28s}  {f}/{t} flipped  ({f/t:.1%})" if t else f"  {b:28s}  (none)")

    n_would_flag = sum(1 for r in rows if r["margin"] < args.margin)
    print(f"\nWith margin={args.margin}, {n_would_flag}/{total} of these images "
          f"({n_would_flag/total:.1%}) would now be routed to FLAGGED instead of "
          f"a confident BLOCKED/CLEAR call.")


if __name__ == "__main__":
    main()
