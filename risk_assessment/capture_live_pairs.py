"""
capture_live_pairs.py
----------------------
Fetches TWO live captures of every monitored camera, separated by a fixed
wait, and saves both to disk — the raw material for a real (not synthetic)
test of whether TTA averaging (app/inference.py's use_tta) reduces the
swing in p_blocked between two genuine successive photos of the same
physical scene, instead of a synthetic brightness/contrast tweak.

This MUST run wherever the live monitor already runs successfully — it
needs real network access to eacornwallwebcams.org / eadevonwebcams.org.
That is your Mac (wherever `uvicorn app.main:app` already works), NOT hex —
hex is GPU-only with no route to those sites.

Usage:
    python3 risk_assessment/capture_live_pairs.py --wait-seconds 90 --out tta_validation

    # Or restrict to specific cameras (much faster — skips the other ~53):
    python3 risk_assessment/capture_live_pairs.py --wait-seconds 90 --out tta_validation_v4 \\
        --campaths cornwall/BudeCedarGrove/cam1 cornwall/PlymptonChaddlewood/cam1 BarnstapleBradiford

Then copy the resulting tta_validation/ folder to hex (rsync/scp) and run
risk_assessment/test_live_pair_stability.py there (needs torch + GPU).
"""

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.main import _fetch_latest_image, _WEBCAM_LIST  # noqa: E402


def _safe_name(campath: str) -> str:
    return campath.replace("/", "_").replace(" ", "_")


async def _capture_all(client: httpx.AsyncClient, label: str, out_dir: Path, manifest_rows: list,
                        cameras: list) -> None:
    for cam in cameras:
        safe    = _safe_name(cam["campath"])
        cam_dir = out_dir / safe
        cam_dir.mkdir(parents=True, exist_ok=True)
        try:
            image, captured_at = await _fetch_latest_image(cam, client)
        except Exception as exc:
            print(f"  [{cam['name']:28s}] {label} fetch failed: {exc}")
            manifest_rows.append({
                "campath": cam["campath"], "name": cam["name"], "label": label,
                "captured_at": "", "ok": False,
            })
            continue
        path = cam_dir / f"{label}.jpg"
        image.save(path, format="JPEG", quality=90)
        print(f"  [{cam['name']:28s}] {label} saved  (captured_at={captured_at})")
        manifest_rows.append({
            "campath": cam["campath"], "name": cam["name"], "label": label,
            "captured_at": captured_at, "ok": True,
        })


async def main_async(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list = []

    if args.campaths:
        target_set = set(args.campaths)
        cameras = [cam for cam in _WEBCAM_LIST if cam["campath"] in target_set]
        missing = target_set - {cam["campath"] for cam in cameras}
        if missing:
            print(f"[warn] {len(missing)} campath(s) not found in _WEBCAM_LIST: {sorted(missing)}")
    else:
        cameras = _WEBCAM_LIST

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        print(f"=== Capture 1/2 — {len(cameras)} cameras ===")
        await _capture_all(client, "t1", out_dir, manifest_rows, cameras)

        print(f"\nWaiting {args.wait_seconds}s before the second capture "
              f"(real cameras need real time to actually change)...")
        time.sleep(args.wait_seconds)

        print(f"\n=== Capture 2/2 — {len(cameras)} cameras ===")
        await _capture_all(client, "t2", out_dir, manifest_rows, cameras)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["campath", "name", "label", "captured_at", "ok"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_ok_pairs = sum(
        1 for cam in cameras
        if (out_dir / _safe_name(cam["campath"]) / "t1.jpg").exists()
        and (out_dir / _safe_name(cam["campath"]) / "t2.jpg").exists()
    )
    print(f"\nDone. {n_ok_pairs}/{len(cameras)} cameras have a complete t1+t2 pair.")
    print(f"Manifest: {manifest_path}")
    print(f"Now copy {out_dir}/ to hex (rsync/scp) and run "
          f"risk_assessment/test_live_pair_stability.py there.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-seconds", type=int, default=90,
                     help="Gap between the two captures of each camera. Larger gaps give the "
                          "real world more chance to actually change — but most EA cameras "
                          "refresh on their own schedule regardless, so this mainly controls "
                          "how long the whole run takes, not the odds of catching a difference.")
    ap.add_argument("--out", default="tta_validation")
    ap.add_argument("--campaths", nargs="+", default=None,
                     help="Restrict capture to these campaths only (much faster than all 56). "
                          "Omit to capture every camera in _WEBCAM_LIST.")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
