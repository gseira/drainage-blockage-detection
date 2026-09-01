"""
test_specialist_vs_general.py
--------------------------------
Real (not held-out-split) validation of a specialist fine-tune, before it
gets wired into _SPECIALIST_CAMERAS. Compares the GENERAL model against the
new specialist checkpoint on genuinely separate LIVE captures of the same
physical scene (from capture_live_pairs.py), restricted to only the cameras
the specialist model was actually trained on.

This exists because a good val F1 from finetune_problem_cameras.py (e.g.
0.993 for best_model_ft_v2.pt) is accuracy on a held-out slice of the SAME
collected dataset — it says nothing about whether live, in-production scans
actually get more stable. We already learned this lesson once: TTA looked
fine in isolation and turned out not to help in the real live-pair test.
Don't repeat that mistake for a specialist retrain either.

Since these fresh live captures have no ground-truth label, this can't
measure "more correct" directly — but it CAN measure stability: whether
p_blocked swings less between two near-simultaneous real captures of the
same physical scene on the specialist model than on the general model. A
camera that's actually learned its own conditions should behave less
erratically on its own live feed, not just score well on a held-out split
of 2022-era photos.

Usage:
    # 1. On a machine with real network access (e.g. your Mac):
    python3 risk_assessment/capture_live_pairs.py --wait-seconds 90 --out tta_validation2

    # 2. rsync tta_validation2/ to hex, then here (needs torch + GPU):
    python3 risk_assessment/test_specialist_vs_general.py \\
        --general-checkpoint results/checkpoints/best_model.pt \\
        --specialist-checkpoint results/checkpoints/best_model_ft_v2.pt \\
        --captures tta_validation2 \\
        --specialist-cameras cornwall/WadebridgePolmorla/cam1 cornwall/PlymptonChaddlewood/cam1 \\
            cornwall/BodminPetrocsWell/cam1 cornwall/BudeCedarGrove/cam1 cornwall/KingsandCP/cam1 \\
            cornwall/LostwithielUP/cam1 cornwall/Mevagissey/cam1 cornwall/Penryn/cam1 \\
            cornwall/PenzanceCC/cam1 cornwall/PlymptonForSt/cam1 cornwall/PorthlevenScreen/cam1 \\
            cornwall/Portreath/cam4 cornwall/TamertonFoliot/cam2 BarnstapleBradiford \\
            BarnstapleConeyGut/Screen BarnstaplePortmarshLane Buckfastleigh \\
            KenwithValleyChannelScreen/Screen LympstoneScreen SwimbridgeScreen
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
    ap.add_argument("--general-checkpoint", default=str(ROOT / "results" / "checkpoints" / "best_model.pt"))
    ap.add_argument("--specialist-checkpoint", required=True)
    ap.add_argument("--captures", default="tta_validation2",
                     help="Folder produced by capture_live_pairs.py.")
    ap.add_argument("--specialist-cameras", nargs="+", required=True,
                     help="campaths the specialist model was trained on — only these are evaluated, "
                          "since the specialist model has no reason to behave differently from the "
                          "general model on cameras it never saw.")
    args = ap.parse_args()

    captures_dir  = Path(args.captures)
    manifest_path = captures_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"No manifest.csv found in {captures_dir} — run capture_live_pairs.py first "
              f"(on a machine with real network access) and copy that folder here.")
        return

    cams = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            cams[row["campath"]] = row["name"]

    target_set = set(args.specialist_cameras)
    cams = {cp: name for cp, name in cams.items() if cp in target_set}
    missing = target_set - set(cams.keys())
    if missing:
        print(f"[warn] {len(missing)} specialist camera(s) not found in the capture manifest "
              f"(check spelling / that capture_live_pairs.py actually fetched them): {sorted(missing)}")

    print("Loading GENERAL model...")
    pipeline_general = InferencePipeline(
        checkpoint_path=args.general_checkpoint, use_llm=False, preserve_aspect=False,
    )
    print("Loading SPECIALIST model (aspect-preserving preprocessing)...")
    pipeline_specialist = InferencePipeline(
        checkpoint_path=args.specialist_checkpoint, use_llm=False, preserve_aspect=True,
    )

    rows = []
    for campath, name in cams.items():
        cam_dir = captures_dir / _safe_name(campath)
        p1_path, p2_path = cam_dir / "t1.jpg", cam_dir / "t2.jpg"
        if not (p1_path.exists() and p2_path.exists()):
            print(f"  [skip] {name} — missing capture(s)")
            continue

        img1 = Image.open(p1_path).convert("RGB")
        img2 = Image.open(p2_path).convert("RGB")

        r1_gen = pipeline_general.run(img1, validate_scene=False)
        r2_gen = pipeline_general.run(img2, validate_scene=False)
        r1_spec = pipeline_specialist.run(img1, validate_scene=False)
        r2_spec = pipeline_specialist.run(img2, validate_scene=False)

        swing_gen  = abs((r1_gen["p_blocked"]  or 0.0) - (r2_gen["p_blocked"]  or 0.0))
        swing_spec = abs((r1_spec["p_blocked"] or 0.0) - (r2_spec["p_blocked"] or 0.0))
        flipped_gen  = r1_gen["prediction"]  != r2_gen["prediction"]
        flipped_spec = r1_spec["prediction"] != r2_spec["prediction"]

        rows.append({
            "campath": campath, "name": name,
            "swing_gen": swing_gen, "swing_spec": swing_spec,
            "flipped_gen": flipped_gen, "flipped_spec": flipped_spec,
            "gen_preds": f"{r1_gen['prediction']}->{r2_gen['prediction']}",
            "spec_preds": f"{r1_spec['prediction']}->{r2_spec['prediction']}",
            # Full per-capture probabilities — needed to identify and cite the
            # specific case where general and specialist disagreed (added
            # 2026-08-28; previously only the swing/prediction summary was
            # printed and this data was lost once the terminal scrolled).
            "p1_gen": r1_gen["p_blocked"], "p2_gen": r2_gen["p_blocked"],
            "p1_spec": r1_spec["p_blocked"], "p2_spec": r2_spec["p_blocked"],
            "t1_path": str(p1_path), "t2_path": str(p2_path),
        })

        marker = ("  <-- IMPROVED" if swing_spec < swing_gen
                  else "  <-- WORSE" if swing_spec > swing_gen else "  (same)")
        print(f"  {name:28s}  general swing={swing_gen:.3f} ({rows[-1]['gen_preds']}, "
              f"p={r1_gen['p_blocked']:.3f}->{r2_gen['p_blocked']:.3f})"
              f"   specialist swing={swing_spec:.3f} ({rows[-1]['spec_preds']}, "
              f"p={r1_spec['p_blocked']:.3f}->{r2_spec['p_blocked']:.3f}){marker}")

        # Flag any capture where the two models landed on a DIFFERENT label —
        # this is exactly the case the thesis's "confirmed catch" figure needs.
        if r1_gen["prediction"] != r1_spec["prediction"]:
            print(f"      *** DISAGREEMENT on t1 ({p1_path}): "
                  f"general={r1_gen['prediction']} (p={r1_gen['p_blocked']:.3f})  "
                  f"specialist={r1_spec['prediction']} (p={r1_spec['p_blocked']:.3f}) ***")
        if r2_gen["prediction"] != r2_spec["prediction"]:
            print(f"      *** DISAGREEMENT on t2 ({p2_path}): "
                  f"general={r2_gen['prediction']} (p={r2_gen['p_blocked']:.3f})  "
                  f"specialist={r2_spec['prediction']} (p={r2_spec['p_blocked']:.3f}) ***")

    if not rows:
        print("\nNo camera pairs found to evaluate.")
        return

    # Persist full results to CSV next to the captures — so this data survives
    # regardless of terminal scrollback (it didn't, the first time this was run).
    csv_out = captures_dir / "comparison_results.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results saved to: {csv_out}")

    n            = len(rows)
    n_flip_gen   = sum(r["flipped_gen"] for r in rows)
    n_flip_spec  = sum(r["flipped_spec"] for r in rows)
    avg_swing_gen  = sum(r["swing_gen"] for r in rows) / n
    avg_swing_spec = sum(r["swing_spec"] for r in rows) / n
    n_improved   = sum(1 for r in rows if r["swing_spec"] < r["swing_gen"])
    n_worse      = sum(1 for r in rows if r["swing_spec"] > r["swing_gen"])
    n_same       = n - n_improved - n_worse

    print(f"\n{'='*70}")
    print(f"Specialist cameras evaluated:          {n}")
    print(f"Predictions flipped, general model:    {n_flip_gen}/{n}  ({n_flip_gen/n:.1%})")
    print(f"Predictions flipped, specialist model:  {n_flip_spec}/{n}  ({n_flip_spec/n:.1%})")
    print(f"Average |p_blocked| swing, general:    {avg_swing_gen:.4f}")
    print(f"Average |p_blocked| swing, specialist: {avg_swing_spec:.4f}")
    print(f"Cameras where specialist reduced swing: {n_improved}/{n}")
    print(f"Cameras where specialist made it worse: {n_worse}/{n}")
    print(f"Cameras unchanged:                      {n_same}/{n}")

    print(f"\nPer-camera detail, worst-first (specialist swing):")
    for r in sorted(rows, key=lambda r: -r["swing_spec"]):
        print(f"  {r['name']:28s}  general={r['swing_gen']:.3f} ({r['gen_preds']})"
              f"   specialist={r['swing_spec']:.3f} ({r['spec_preds']})")


if __name__ == "__main__":
    main()
