"""
test_live_pair_stability.py
-----------------------------
The real (not synthetic) counterpart to test_colour_consistency.py.

For every camera, compares the model's p_blocked on TWO genuinely separate
LIVE captures of the same physical scene (produced by capture_live_pairs.py
on a machine with real network access — e.g. your Mac), under both TTA-off
and TTA-on, to see whether test-time averaging actually reduces the swing
between real successive photos — the same kind of swing monitoring.db showed
directly (e.g. Wadebridge going from p=0.9958 to p=0.0001 in 65 seconds).

Usage (run on hex or wherever torch + GPU live):

    python3 risk_assessment/test_live_pair_stability.py \\
        --checkpoint results/checkpoints/best_model.pt \\
        --captures tta_validation
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.inference import InferencePipeline  # noqa: E402


def _safe_name(campath: str) -> str:
    return campath.replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints" / "best_model.pt"))
    ap.add_argument("--captures", default="tta_validation",
                     help="Folder produced by capture_live_pairs.py (contains manifest.csv "
                          "and one subfolder per camera with t1.jpg / t2.jpg).")
    args = ap.parse_args()

    captures_dir  = Path(args.captures)
    manifest_path = captures_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"No manifest.csv found in {captures_dir} — run capture_live_pairs.py first "
              f"(on a machine with real network access, e.g. your Mac) and copy that folder "
              f"here before running this.")
        return

    # Recover campath -> name from the manifest (one row per capture, so dedupe by campath)
    cams = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            cams[row["campath"]] = row["name"]

    print("Loading TTA-off pipeline (single-pass baseline)...")
    pipeline_no_tta = InferencePipeline(checkpoint_path=args.checkpoint, use_tta=False, use_llm=False)
    print("Loading TTA-on pipeline (same weights, 7 views per call)...")
    pipeline_tta = InferencePipeline(checkpoint_path=args.checkpoint, use_tta=True, use_llm=False)

    rows = []
    for campath, name in cams.items():
        cam_dir  = captures_dir / _safe_name(campath)
        p1_path, p2_path = cam_dir / "t1.jpg", cam_dir / "t2.jpg"
        if not (p1_path.exists() and p2_path.exists()):
            print(f"  [skip] {name} — missing capture(s), likely a fetch failure during capture")
            continue

        img1 = Image.open(p1_path).convert("RGB")
        img2 = Image.open(p2_path).convert("RGB")

        r1_no  = pipeline_no_tta.run(img1, validate_scene=False)
        r2_no  = pipeline_no_tta.run(img2, validate_scene=False)
        r1_tta = pipeline_tta.run(img1, validate_scene=False)
        r2_tta = pipeline_tta.run(img2, validate_scene=False)

        swing_no  = abs((r1_no["p_blocked"]  or 0.0) - (r2_no["p_blocked"]  or 0.0))
        swing_tta = abs((r1_tta["p_blocked"] or 0.0) - (r2_tta["p_blocked"] or 0.0))
        flipped_no  = r1_no["prediction"]  != r2_no["prediction"]
        flipped_tta = r1_tta["prediction"] != r2_tta["prediction"]

        rows.append({
            "campath": campath, "name": name,
            "swing_no_tta": swing_no, "swing_tta": swing_tta,
            "flipped_no_tta": flipped_no, "flipped_tta": flipped_tta,
        })

        marker = ("  <-- IMPROVED" if swing_tta < swing_no
                  else "  <-- WORSE" if swing_tta > swing_no else "")
        print(f"  {name:28s}  no-tta swing={swing_no:.3f} ({r1_no['prediction']}->{r2_no['prediction']})"
              f"   tta swing={swing_tta:.3f} ({r1_tta['prediction']}->{r2_tta['prediction']}){marker}")

    if not rows:
        print("\nNo camera pairs found to evaluate — check --captures points at a folder "
              "produced by capture_live_pairs.py.")
        return

    n            = len(rows)
    n_flip_no    = sum(r["flipped_no_tta"] for r in rows)
    n_flip_tta   = sum(r["flipped_tta"] for r in rows)
    avg_swing_no  = sum(r["swing_no_tta"] for r in rows) / n
    avg_swing_tta = sum(r["swing_tta"] for r in rows) / n
    n_improved   = sum(1 for r in rows if r["swing_tta"] < r["swing_no_tta"])
    n_worse      = sum(1 for r in rows if r["swing_tta"] > r["swing_no_tta"])
    n_same       = n - n_improved - n_worse

    print(f"\n{'='*70}")
    print(f"Cameras evaluated:                   {n}")
    print(f"Predictions flipped, no TTA:          {n_flip_no}/{n}  ({n_flip_no/n:.1%})")
    print(f"Predictions flipped, with TTA:        {n_flip_tta}/{n}  ({n_flip_tta/n:.1%})")
    print(f"Average |p_blocked| swing, no TTA:    {avg_swing_no:.4f}")
    print(f"Average |p_blocked| swing, with TTA:  {avg_swing_tta:.4f}")
    print(f"Cameras where TTA reduced the swing:  {n_improved}/{n}")
    print(f"Cameras where TTA made it worse:      {n_worse}/{n}")
    print(f"Cameras unchanged:                    {n_same}/{n}")

    print(f"\nWorst-remaining swings under TTA (still worth a closer look even if TTA helped):")
    for r in sorted(rows, key=lambda r: -r["swing_tta"])[:10]:
        print(f"  {r['name']:28s}  swing_tta={r['swing_tta']:.3f}  swing_no_tta={r['swing_no_tta']:.3f}")


if __name__ == "__main__":
    main()
