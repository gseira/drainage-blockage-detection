"""
export_reviewed_images.py
--------------------------
Exports human-corrected results (Mark Blocked / Mark Clear, via
POST /results/reclassify) out of monitoring.db into the
images/<camera>/blocked/ and images/<camera>/clear/ folder structure that
models/finetune_problem_cameras.py actually reads from — this is what turns
"a person caught the model being wrong" into real training data for the
next fine-tune, rather than just a one-off UI correction that only helps
that single scan.

Only rows with manually_reviewed=1 AND a stored original_b64 are usable.
That requires the row to have originally been BLOCKED or FLAGGED (see
_save_result in app/main.py, 2026-07-07) — original_b64 is never stored for
a plain LOW/CLEAR scan, so a correction made on a row that was never
flagged in the first place has no image to export. Run this periodically
(e.g. whenever you've done a batch of manual reviews) rather than once.

Usage:
    python3 data/export_reviewed_images.py --db monitoring.db --out images
"""

import argparse
import base64
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from camera_folder_map import folder_for  # noqa: E402

# NOTE: as of 2026-07-07, /results/reclassify in app/main.py exports each
# correction to disk immediately when it happens (see _export_reviewed_image
# there) — this script now mainly matters for backfilling older corrections
# made before that change, or for re-running against a monitoring.db copied
# in from elsewhere. New corrections don't need this script run manually.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="monitoring.db")
    ap.add_argument("--out", default="images")
    args = ap.parse_args()

    out_root = Path(args.out)
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute("""
        SELECT id, campath, risk, original_b64, created_at
        FROM results
        WHERE manually_reviewed = 1 AND original_b64 IS NOT NULL AND original_b64 != ''
        ORDER BY id
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} manually-reviewed rows with a stored original frame.")

    n_saved = 0
    n_skipped = 0
    per_camera: dict = {}

    for row_id, campath, risk, original_b64, created_at in rows:
        if risk not in ("BLOCKED", "LOW"):
            n_skipped += 1
            continue

        label_dir = "blocked" if risk == "BLOCKED" else "clear"
        folder    = folder_for(campath)
        dest_dir  = out_root / folder / label_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f"reviewed_{row_id}.jpg"
        if dest_path.exists():
            continue  # already exported in a previous run

        try:
            dest_path.write_bytes(base64.b64decode(original_b64))
            n_saved += 1
            per_camera.setdefault(folder, {"blocked": 0, "clear": 0})
            per_camera[folder][label_dir] += 1
        except Exception as exc:
            print(f"  [skip] row {row_id} ({campath}): {exc}")
            n_skipped += 1

    print(f"\nExported {n_saved} new images this run "
          f"(skipped {n_skipped} — already exported, unusable risk value, or decode error).")

    if per_camera:
        print("\nNew images by camera this run:")
        for folder, counts in sorted(per_camera.items()):
            print(f"  {folder:45s} blocked+{counts['blocked']:<4d} clear+{counts['clear']}")

    print(f"\nOutput root: {out_root.resolve()}")
    print("Next: once any camera folder has a meaningful number in both blocked/")
    print("and clear/, add its folder name to PROBLEM_CAMERAS in")
    print("models/finetune_problem_cameras.py for the next retrain.")


if __name__ == "__main__":
    main()
