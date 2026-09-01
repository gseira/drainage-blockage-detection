"""
collect_finetune_frames.py
---------------------------
Pulls the current live frame for each "problem camera" and drops it into
images/<CameraFolder>/unlabeled/ so it can be hand-sorted into blocked/ or
clear/ before running models/finetune_problem_cameras.py.

Why this exists: fine-tuning needs real labeled JPEGs per camera, and the
monitoring DB does NOT store them (only GradCAM overlay/heatmap composites),
so there is currently no labeled data sitting anywhere for these cameras.
This script is the first step of building that dataset from scratch by
sampling the live feed repeatedly over time (different light, weather,
partial-clear states, etc. — exactly the diversity a fine-tune needs).

Run it on a schedule (cron/launchd), NOT continuously in the foreground:

    */20 * * * *  cd ~/dissertation && python3 data/collect_finetune_frames.py --once >> logs/collect_frames.log 2>&1

Or run it interactively in a loop for a quick burst of samples:

    python3 data/collect_finetune_frames.py --interval-min 15 --hours 6

After a few days you should have a spread of frames per camera in
images/<CameraFolder>/unlabeled/. Then manually drag each one into
images/<CameraFolder>/blocked/ or images/<CameraFolder>/clear/ (just look at
it — no tooling needed) before fine-tuning.
"""

import argparse
import asyncio
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.main import _fetch_latest_image, _WEBCAM_LIST  # noqa: E402

# ── Problem cameras → output folder name ────────────────────────────────────
# Folder name is what you'll later add to PROBLEM_CAMERAS in
# models/finetune_problem_cameras.py, so keep this mapping and that list
# in sync (same key on both sides).
#
# NOTE (checked directly on hex, ~/dissertation/images, 2026-07-07):
#   - Cornwall_WadebridgePolmorla:            329 blocked / 132 clear — already fine-tuned on.
#   - Cornwall_PlymptonChaddlewood_MainScree: 167 blocked / 995 clear — DATA ALREADY EXISTS,
#     no collection needed. Folder name on hex has the "_MainScree" suffix — don't run this
#     collector for it unless you specifically want to add more variety later.
#   - Cornwall_StIvesConsols_Scree / _Upstream: plenty of "clear" (1558 / 1194) but ZERO
#     "blocked" in either — the real gap for this camera is blocked examples specifically,
#     not more images in general. Check the "other" bucket (393 in _Upstream) for any
#     mislabeled blocked frames before assuming none exist at all.
#   - Porthallow, LauncestonWooda, AshburtonLower, KingsbridgeDuncombe, NewtonAbbotBakersPark:
#     genuinely no folder on hex at all — these are the ones this collector is actually for.
TARGET_CAMERAS = {
    "cornwall/Porthallow/cam1":           "Cornwall_Porthallow",
    "cornwall/LauncestonWooda/cam1":      "Cornwall_LauncestonWooda",
    "AshburtonLower/Screen":              "Devon_AshburtonLower",
    "KingsbridgeDuncombe":                "Devon_KingsbridgeDuncombe",
    "NewtonAbbotBakersPark":              "Devon_NewtonAbbotBakersPark",
}

IMAGES_ROOT = ROOT / "images"


def _cam_lookup() -> dict:
    return {c["campath"]: c for c in _WEBCAM_LIST}


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


async def collect_once() -> None:
    cams_by_path = _cam_lookup()
    missing = [cp for cp in TARGET_CAMERAS if cp not in cams_by_path]
    if missing:
        print(f"  [warn] not found in _WEBCAM_LIST, skipping: {missing}")

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for campath, folder in TARGET_CAMERAS.items():
            cam = cams_by_path.get(campath)
            if cam is None:
                continue

            out_dir = IMAGES_ROOT / folder / "unlabeled"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                image, captured_at = await _fetch_latest_image(cam, client)
            except Exception as exc:
                print(f"  [{cam['name']:28s}] fetch failed: {exc}")
                continue

            # Serialize once so we can hash it for de-dup.
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            raw = buf.getvalue()
            digest = _md5(raw)

            # Skip if identical to the most recently saved frame for this camera
            # (camera hasn't refreshed yet — no point saving duplicates).
            existing = sorted(out_dir.glob("*.jpg"))
            if existing:
                last = existing[-1]
                if _md5(last.read_bytes()) == digest:
                    print(f"  [{cam['name']:28s}] unchanged since last capture — skipped")
                    continue

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{ts}.jpg"
            out_path.write_bytes(raw)
            print(f"  [{cam['name']:28s}] saved {out_path.relative_to(ROOT)}  (captured_at={captured_at})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Single pass over all target cameras, then exit (use this from cron).")
    ap.add_argument("--interval-min", type=float, default=20.0, help="Minutes between passes when looping.")
    ap.add_argument("--hours", type=float, default=None, help="Stop looping after this many hours (default: run forever until Ctrl-C).")
    args = ap.parse_args()

    if args.once:
        asyncio.run(collect_once())
        return

    start = time.monotonic()
    deadline = start + args.hours * 3600 if args.hours else None
    while True:
        print(f"\n=== pass @ {datetime.now(timezone.utc).isoformat()} ===")
        asyncio.run(collect_once())
        if deadline and time.monotonic() >= deadline:
            print("Reached --hours limit, stopping.")
            break
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
