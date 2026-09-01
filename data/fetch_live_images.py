"""
fetch_live_images.py
--------------------
Downloads the latest image from every camera on eacornwallwebcams.org
and saves them into a flat folder with location-prefixed filenames,
ready to drag into the web app.

Output:
    live_images/<LocationName>_cam<N>_<YYYY-MM-DD>.jpg

Usage:
    python data/fetch_live_images.py
    python data/fetch_live_images.py --out ~/Desktop/live_batch
    python data/fetch_live_images.py --location PorthlevenScree   # one location only
    python data/fetch_live_images.py --screens-only               # only screen cameras

Open Government Licence v3.0 — Environment Agency Cornwall Webcams
"""

import argparse
import os
import re
import time
from datetime import date
from pathlib import Path

import requests
from PIL import Image
import io

BASE_URL  = "https://eacornwallwebcams.org"
TODAY     = date.today().strftime("%Y_%m_%d")
HEADERS   = {"User-Agent": "Mozilla/5.0 (dissertation research tool)"}

# ── Full camera list extracted from eacornwallwebcams.org ────────────────────
# Format: (location_folder, cam_id, human_name)
CAMERAS = [
    ("Angarrack",           "cam1", "Angarrack Screen"),
    ("BodminFlaxmoor",      "cam1", "Bodmin Flaxmoor Screen"),
    ("BodminPetrocsWell",   "cam1", "Bodmin St Petrocs Well Screen"),
    ("BoscastleMarine",     "cam1", "Boscastle Marine Terrace"),
    ("Boscundle",           "cam4", "Boscundle Main Screen"),
    ("BudeCedarGrove",      "cam1", "Bude Cedar Grove Screen"),
    ("CawsandCP",           "cam1", "Cawsand Car Park Screen"),
    ("Crinnis",             "cam1", "Crinnis Screen"),
    ("Dousland",            "cam1", "Dousland Screen"),
    ("GorranHaven",         "cam1", "Gorran Haven Screen"),
    ("HayleMellanear",      "cam1", "Hayle Mellanear Screen"),
    ("HorrabridgeFP",       "cam1", "Horrabridge Fillace Park Screen"),
    ("HorrabridgeSF",       "cam1", "Horrabridge Springfields Screen"),
    ("KingsandCP",          "cam1", "Kingsand Car Park Screen"),
    ("KingsandPP",          "cam1", "Kingsand Porsproder Screen"),
    ("LostwithielUP",       "cam1", "Lostwithiel Uzella Park Screen"),
    ("Mevagissey",          "cam1", "Mevagissey Fire Station Screen"),
    ("MillbrookScreen",     "cam1", "Millbrook Screen"),
    ("PadstowChurchScreen", "cam1", "Padstow Church Screen"),
    ("ParPumps",            "cam1", "Par Pumps Screen"),
    ("PenrynTP",            "cam1", "Penryn Trelawney Park"),
    ("Penryn",              "cam1", "Penryn Screen"),
    ("PentewanScreen",      "cam1", "Pentewan Screen"),
    ("PenzanceCC",          "cam1", "Penzance Coombe Cottage Screen"),
    ("PenzanceCS",          "cam1", "Penzance Chyander Square Screen"),
    ("Perrancoombe",        "cam1", "Perrancoombe Screen"),
    ("PlymptonChaddlewood", "cam1", "Plympton Chaddlewood Main Screen"),
    ("PlymCotHill",         "cam1", "Plympton Cot Hill Screen"),
    ("PlymptonForSt",       "cam1", "Plympton Fore St Screen"),
    ("PlymptonKay",         "cam1", "Plympton Kay Close Screen"),
    ("PlymptonStoggy",      "cam1", "Plympton Stoggy Lane Screen"),
    ("PolperroLangreek",    "cam1", "Polperro Langreek Screen"),
    ("PolperroTS",          "cam1", "Polperro Top Screen"),
    ("Porthallow",          "cam1", "Porthallow Screen"),
    ("PorthlevenScreen",    "cam1", "Porthleven Screen"),
    ("Portloe",             "cam1", "Portloe Screen"),
    ("Portreath",           "cam4", "Portreath Screen"),
    ("StBlazeyStationRd",   "cam1", "St Blazey Station Rd Screen"),
    ("StIvesConsols",       "cam1", "St Ives Consols Farm Screen"),
    ("TamertonFoliot",      "cam2", "Tamerton Foliot Screen"),
    ("WadebridgePolmorla",  "cam1", "Wadebridge Polmorla"),
    ("WalkhamptonScreen",   "cam1", "Walkhampton Screen"),
]


def fetch_image(campath: str, cam_id: str) -> bytes | None:
    """
    Try multiple URL patterns to get the latest JPEG for a camera.
    Returns raw image bytes or None if all attempts fail.
    """
    patterns = [
        # Pattern 1: direct latest.jpg
        f"{BASE_URL}/cornwall/{campath}/{cam_id}/latest.jpg",
        # Pattern 2: gallery PHP page → parse img src
        f"{BASE_URL}/common/gallerylatestpic.php?campath=cornwall/{campath}/{cam_id}",
    ]

    for url in patterns:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15, stream=True)
            if r.status_code != 200:
                continue

            content_type = r.headers.get("content-type", "")

            # Direct image response
            if "image" in content_type:
                return r.content

            # HTML page — parse for img src
            if "html" in content_type:
                html = r.text
                # Look for <img src="...jpg">
                match = re.search(r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']',
                                  html, re.IGNORECASE)
                if match:
                    img_url = match.group(1)
                    if not img_url.startswith("http"):
                        img_url = BASE_URL + "/" + img_url.lstrip("/")
                    img_r = requests.get(img_url, headers=HEADERS, timeout=15)
                    if img_r.status_code == 200:
                        return img_r.content

        except Exception as e:
            print(f"    ✗ {url}  ({e})")
            continue

    return None


def main(out_dir: Path, location_filter: str | None, screens_only: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = CAMERAS
    if location_filter:
        cameras = [c for c in cameras if location_filter.lower() in c[0].lower()]
    if screens_only:
        cameras = [c for c in cameras if "screen" in c[2].lower()]

    print(f"Fetching {len(cameras)} camera(s) → {out_dir}\n")

    success, failed = 0, 0

    for campath, cam_id, name in cameras:
        loc_label = f"Cornwall_{campath}"
        filename  = f"{loc_label}_{cam_id}_{TODAY}.jpg"
        dest      = out_dir / filename

        if dest.exists():
            print(f"  ↷  {name} — already exists, skipping")
            success += 1
            continue

        print(f"  ↓  {name} ...", end=" ", flush=True)
        img_bytes = fetch_image(campath, cam_id)

        if img_bytes:
            # Validate it's a real image
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(dest, format="JPEG", quality=95)
                print(f"✓  ({img.width}×{img.height})")
                success += 1
            except Exception as e:
                print(f"✗  invalid image ({e})")
                failed += 1
        else:
            print("✗  could not fetch")
            failed += 1

        time.sleep(0.5)   # be polite to the server

    print(f"\nDone.  {success} saved · {failed} failed → {out_dir}")
    print("\nNow upload from:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",          default="live_images",
                        help="Output folder (default: live_images/)")
    parser.add_argument("--location",     default=None,
                        help="Filter by location name (e.g. PorthlevenScree)")
    parser.add_argument("--screens-only", action="store_true",
                        help="Only fetch screen cameras (most relevant for blockage detection)")
    args = parser.parse_args()

    main(
        out_dir        = Path(args.out),
        location_filter= args.location,
        screens_only   = args.screens_only,
    )
