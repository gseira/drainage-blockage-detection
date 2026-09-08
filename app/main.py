"""
main.py
-------
FastAPI application for the Drainage Blockage Detection system.

Endpoints:
    GET  /               → API status (frontend is Flood Watcher, run separately)
    POST /predict        → single image analysis (file upload)
    POST /predict/batch  → batch file upload, results sorted HIGH first
    POST /predict/webcam → fetch latest EA Cornwall webcam image + run inference
    GET  /webcams        → list of available screen cameras
    GET  /results/latest → most recent DB result per camera
    GET  /results/history→ time-filtered result log (optional campath filter)
    GET  /results/stats  → daily HIGH/LOW counts for trend chart
    GET  /health         → liveness check

Background:
    APScheduler runs monitor_all_cameras() every 15 minutes automatically.
    All webcam results are persisted to SQLite (monitoring.db).

Run locally:
    uvicorn app.main:app --reload --port 8000

Environment variables:
    CHECKPOINT_PATH   path to model checkpoint  (default: results/checkpoints/best_model.pt)
    CEREBRAS_API_KEY  Cerebras API key        (or set in .env)
    DB_PATH           SQLite database path    (default: monitoring.db in project root)
    MONITOR_INTERVAL  sweep interval minutes  (default: 15)
"""

import asyncio
import base64
import io
import logging
import math
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import aiosqlite
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw
from pydantic import BaseModel

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from app.inference import InferencePipeline
from app.correction_memory import init_corrections_table, add_correction, find_correction_match, get_all_corrections, prune_corrections
from app.live_head import retrain_head, save_live_head, evaluate_against_canary, CANARY_MIN_AGREEMENT

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from camera_folder_map import folder_for as _folder_for  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH          = os.getenv("DB_PATH", str(Path(__file__).parent.parent / "monitoring.db"))
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "15"))   # minutes
BASE_URL         = "https://eacornwallwebcams.org"

# Each row can store up to three base64 images (original/overlay/heatmap)
# plus an embedding BLOB — those are what make monitoring.db grow fast (56+
# cameras swept every MONITOR_INTERVAL minutes). But nothing in this app
# actually needs those heavy columns for more than a day or two:
#   - /results/latest, /results/latest-by-location, and the scan-consistency
#     / correction-memory overrides (_get_previous_scan) only ever read the
#     SINGLE MOST RECENT row per camera.
#   - /results/history and /results/stats (the metrics/trend page) only need
#     the lightweight scalar columns (risk, p_blocked, timestamps) over time
#     -- never the images.
# So FULL_IMAGE_RETENTION_HOURS strips overlay_b64/heatmap_b64/original_b64/
# embedding from a row once it's older than this (see _strip_old_images),
# while keeping the row itself (and its scalar columns) around for the trend
# chart. That keeps the DB's real disk footprint roughly flat regardless of
# how long the app has been running, rather than growing without bound.
# RESULTS_RETENTION_DAYS is just the final full-row cleanup; once rows are
# stripped down to a few hundred bytes each, that can be generous. (30 days
# of full, unstripped rows previously filled a droplet's disk -- 48GB on a
# 58GB volume, 2026-08-28 incident -- and the follow-up fix that only
# shortened RESULTS_RETENTION_DAYS to 7 days was not enough on its own:
# 7 days of full-resolution images across 56+ cameras swept every
# MONITOR_INTERVAL minutes still grew past 38GB on the same 57GB volume.
# Splitting image-stripping from row-deletion, as below, is what actually
# keeps disk usage flat.)
FULL_IMAGE_RETENTION_HOURS = int(os.getenv("FULL_IMAGE_RETENTION_HOURS", "48"))
RESULTS_RETENTION_DAYS     = int(os.getenv("RESULTS_RETENTION_DAYS", "180"))

# ── Camera list ───────────────────────────────────────────────────────────────
# Each entry: name, campath, region, base_url, gallery_prefix
# gallery_prefix: the path before /gallerylatestpic.php on that site
_CORNWALL = "https://eacornwallwebcams.org"
_DEVON    = "https://eadevonwebcams.org"

def _c(name, campath):
    """Cornwall camera shorthand."""
    return {"name": name, "campath": campath, "region": "Cornwall",
            "base_url": _CORNWALL, "gallery_prefix": "common"}

def _d(name, campath):
    """Devon camera shorthand."""
    return {"name": name, "campath": campath, "region": "Devon",
            "base_url": _DEVON, "gallery_prefix": "Devon/common"}

_WEBCAM_LIST = [
    # ── Cornwall ──────────────────────────────────────────────────────────────
    _c("Angarrack",                "cornwall/Angarrack/cam1"),
    _c("Bodmin Flaxmoor Terrace",  "cornwall/BodminFlaxmoor/cam1"),
    _c("Bodmin Petrocs Well",      "cornwall/BodminPetrocsWell/cam1"),
    _c("Boscundle",                "cornwall/Boscundle/cam1"),
    _c("Bude Berries Avenue",      "cornwall/BudeBerries/cam1"),
    _c("Bude Cedar Grove",         "cornwall/BudeCedarGrove/cam1"),
    _c("Cawsand Car Park",         "cornwall/CawsandCP/cam1"),
    _c("Gorran Haven",             "cornwall/GorranHaven/cam1"),
    _c("Hayle Mellanear",          "cornwall/HayleMellanear/cam1"),
    _c("Horrabridge Fillace Park", "cornwall/HorrabridgeFP/cam1"),
    _c("Horrabridge Springfields", "cornwall/HorrabridgeSF/cam1"),
    _c("Kingsand Car Park",        "cornwall/KingsandCP/cam1"),
    _c("Launceston Wooda Lane",    "cornwall/LauncestonWooda/cam1"),
    _c("Loe Bar",                  "cornwall/LoeBar/cam1"),
    _c("Lostwithiel Uzella Park",  "cornwall/LostwithielUP/cam1"),
    _c("Mevagissey Fire Station",  "cornwall/Mevagissey/cam1"),
    _c("Penryn",                   "cornwall/Penryn/cam1"),
    _c("Pentewan",                 "cornwall/PentewanScreen/cam1"),
    _c("Penzance Chyander Sq",     "cornwall/PenzanceCS/cam1"),
    _c("Penzance Coombe Cottage",  "cornwall/PenzanceCC/cam1"),
    _c("Perrancoombe",             "cornwall/Perrancoombe/cam1"),
    _c("Plympton Chaddlewood",     "cornwall/PlymptonChaddlewood/cam1"),
    _c("Plympton Fore St",         "cornwall/PlymptonForSt/cam1"),
    _c("Polperro Top Screen",      "cornwall/PolperroTS/cam1"),
    _c("Port Isaac",               "cornwall/PortIsaac/cam1"),
    _c("Porthallow",               "cornwall/Porthallow/cam1"),
    _c("Porthleven",               "cornwall/PorthlevenScreen/cam1"),
    _c("Portreath",                "cornwall/Portreath/cam4"),
    _c("St Ives Consols",          "cornwall/StIvesConsols/cam1"),
    _c("Tamerton Foliot",          "cornwall/TamertonFoliot/cam2"),
    _c("Wadebridge Polmorla",      "cornwall/WadebridgePolmorla/cam1"),
    _c("Walkhampton",              "cornwall/WalkhamptonScreen/cam1"),

    # ── Devon ─────────────────────────────────────────────────────────────────
    _d("Ashburton Lower Screen",       "AshburtonLower/Screen"),
    _d("Barnstaple Bradiford",         "BarnstapleBradiford"),
    _d("Barnstaple Coney Gut Screen",  "BarnstapleConeyGut/Screen"),
    _d("Barnstaple Newport Road",      "BarnstapleNewportRoad"),
    _d("Barnstaple Portmarsh Lane",    "BarnstaplePortmarshLane"),
    _d("Bideford Elliots Garage",      "BidefordElliotsGarageScreen"),
    _d("Braunton Hordens Bridge",      "BrauntonHordens"),
    _d("Buckfastleigh",                "Buckfastleigh"),
    _d("Cullompton Langlands",         "CullomptonLanglands"),
    _d("Dawlish Warren",               "DawlishWarren"),
    _d("Dulverton",                    "Dulverton"),
    _d("East Budleigh",                "EastBudleigh"),
    _d("Harbertonford Screen",         "HarbertonfordScreen"),
    _d("Ilfracombe Screen",            "IlfracombeScreen"),
    _d("Kenwith Valley Screen",        "KenwithValleyChannelScreen/Screen"),
    _d("Kingsbridge Duncombe",         "KingsbridgeDuncombe"),
    _d("Lympstone Screen",             "LympstoneScreen"),
    _d("Newton Abbot Bakers Park",     "NewtonAbbotBakersPark"),
    _d("Occombe Valley Screen",        "OccombeValley"),
    _d("Ottery SM Chapel Lane",        "OtterySMChapelLane"),
    _d("Ottery SM Kennaway Road",      "OtterySMKenawayRoad"),
    _d("Swimbridge Screen",            "SwimbridgeScreen"),
    _d("Teignmouth First Ave Screen",  "TeignmouthFirstAvenue/cam2"),
    _d("Yalberton",                    "Yalberton"),
]


# ── Camera configuration ──────────────────────────────────────────────────────

# Cameras where the view does not show a drainage structure.
# These are always flagged for manual review — inference is skipped.
_ALWAYS_REVIEW_CAMERAS = {
    "BrauntonHordens",          # view shows a bridge, not a drain
    "cornwall/Boscundle/cam1",  # view does not show a drain
    "cornwall/LoeBar/cam1",     # view does not show a drain
}

def _is_real_explanation(text: str) -> bool:
    """
    True if `text` looks like a genuine model-generated explanation, False if
    it's one of the hardcoded "LLM unavailable/disabled" placeholder strings.
    Used to avoid appending "Model's own read: <error message>" to an
    override's explanation — that's noise, not information, for an operator.
    """
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in (
        "llm explanation unavailable", "llm explanation disabled", "could not be reached",
    ))


# Kill switches — every automatic "learn from corrections" mechanism tried
# in this codebase so far has made real production behaviour WORSE on live
# traffic at least once:
#   1. scan-to-scan consistency: FLAGGED self-perpetuated indefinitely.
#      Still disabled.
#   2. correction-memory embedding match: same-camera photos cluster too
#      tightly regardless of actual state, flooding Needs Review. Still
#      disabled.
#   3. live_head.py's online fc retrain: v1 refit a 2,049-parameter linear
#      layer on only 24-28 corrections with NO regularization and NO
#      validation, hit train_acc=1.000 every time (see
#      logs/live_head_updates.log) — i.e. it perfectly memorized those few
#      points at the direct cost of the decision boundary that worked for
#      every other camera/case. This is what "destroyed the model" — real,
#      observed overfitting, not a hypothetical risk. v2 (current) adds
#      anchor regularization (resist drifting from the original weights
#      unless the correction evidence is strong) and an automatic
#      historical-canary rejection gate (reject any update that would flip
#      too many of the model's own past high-confidence calls) — see
#      live_head.py's docstring. RE-ENABLED at the user's explicit request
#      to test v2 against specific real cases. Watch the [live_head] log
#      lines closely (accept/reject + agreement %) for the first several
#      corrections.
# (2) remains disabled — the model's own raw prediction (routed through
# inference.py's confidence-margin rule, clear_below/blocked_above) is what's
# used for that path, with no per-camera overrides layered on top (removed
# at the user's explicit request — Needs Review should come only from that
# one general rule). All disabled code is kept, not deleted, same as
# models/finetune_problem_cameras.py's v2-v5 record of what didn't work.
#
# (1) RE-ENABLED 2026-07-22 at the user's explicit, repeated request ("it is
# 1000% necessary"). Before flipping this back on, the one real gap found in
# the canary safety check was closed: it used to *accept* a retrained head
# with NO regression check at all if there wasn't yet enough historical data
# to sample a canary set from (see the "no historical canary data" branch a
# few hundred lines down) — a silent hole that could let a bad update
# through unchecked. That branch now REJECTS instead. With 37,000+ scans
# already in the database, canary data should be plentiful going forward, so
# this should rarely even trigger, but the check now actually means what it
# says either way.
_ENABLE_LIVE_HEAD = True
_ENABLE_SCAN_CONSISTENCY  = False
_ENABLE_CORRECTION_MEMORY = False

# Near-duplicate of the camera's own previous scan -> keep that scan's
# verdict. Starting guess, same caveat as correction_memory.py's thresholds:
# not calibrated against real embeddings, logged on every match so it can be
# tuned against real numbers once this has run against live traffic.
SCAN_CONSISTENCY_SIMILARITY = 0.995


async def _get_previous_scan(campath: str) -> Optional[dict]:
    """
    The most recent stored row for this camera that has an embedding (older
    rows predating the embedding column, and OFFLINE/placeholder rows that
    never had a real frame, are skipped automatically since their embedding
    is NULL). Returns None if there's nothing to compare against yet.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT risk, p_blocked, heatmap_coverage, explanation, overlay_b64, "
            "heatmap_b64, embedding FROM results WHERE campath = ? AND embedding IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (campath,),
        )
        row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def _sample_canary_embeddings(pipeline, n: int = 200) -> list:
    """
    A random sample of past scans the CURRENT model was already highly
    confident about (p_blocked <= 0.1 or >= 0.9), excluding rows a human
    has since corrected (manually_reviewed=1) — those reflect a known
    mistake, not "already working", so they'd defeat the point of this
    check. Used by app/live_head.py's evaluate_against_canary() as a
    zero-effort proxy for "does this update still agree with what the
    model was already confidently doing" — see that module's docstring for
    what this can and can't catch. Returns [] if there's nothing usable yet
    (e.g. very early on), which the caller must handle explicitly rather
    than silently treating as "safe to proceed".
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT original_b64 FROM results
               WHERE original_b64 IS NOT NULL
                 AND (manually_reviewed IS NULL OR manually_reviewed = 0)
                 AND ((risk = 'LOW' AND p_blocked <= 0.1) OR (risk = 'BLOCKED' AND p_blocked >= 0.9))
               ORDER BY RANDOM() LIMIT ?""",
            (n,),
        )
        rows = await cursor.fetchall()

    embeddings = []
    for row in rows:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(row["original_b64"]))).convert("RGB")
            embeddings.append(pipeline.extract_embedding(image))
        except Exception:
            continue  # a single unreadable image shouldn't block the whole canary check
    return embeddings


async def _apply_scan_consistency_override(campath: str, embedding, result: dict) -> dict:
    """
    Compares this scan's frame to the immediately previous scan of the same
    camera — not a human correction (see app/correction_memory.py for that),
    just whatever was stored last, corrected or not. If the two frames are
    visually indistinguishable (ordinary capture noise only), the previous
    scan's verdict is kept instead of letting a fresh forward pass re-roll
    the dice on what is essentially the same photo.

    This targets a real, measured problem: risk_assessment/
    test_live_pair_stability.py found real photos of the same physical scene,
    taken moments apart, swinging from p_blocked 0.08 to 0.99 on roughly 25
    of 56 cameras, purely from auto-exposure/JPEG-recompression/lighting
    noise the model was never taught to ignore. TTA was tried against this
    directly and didn't meaningfully help (see InferencePipeline's use_tta
    docstring). This is a different, cheaper fix: it doesn't try to make the
    model itself more robust, it just refuses to let two frames a human
    couldn't tell apart get two different verdicts.

    Unlike correction memory, this needs no human input at all and applies
    to every camera from the very first pair of consecutive scans — but it
    also means an early wrong call can persist across near-duplicate frames
    until either the scene visibly changes or a human corrects it via Mark
    Blocked/Clear (which immediately becomes the new "previous scan" for
    this comparison). That's the same accepted tradeoff as correction
    memory, just automatic rather than requiring a click first.

    Never raises — a lookup hiccup just skips the override, same as the
    other override functions.
    """
    if not _ENABLE_SCAN_CONSISTENCY or embedding is None:
        return result
    try:
        previous = await _get_previous_scan(campath)
    except Exception as exc:
        log.warning("[scan_consistency] %s: lookup failed (%s) — skipping", campath, exc)
        return result
    if previous is None:
        return result

    prev_risk = previous["risk"]
    # Only propagate a CONFIDENT previous verdict (BLOCKED/LOW) forward.
    # FLAGGED must never self-perpetuate: it's meant to be a one-off "too
    # close to call, ask a human" state, not a sticky trap. Propagating it
    # forward turned every camera that ever landed in FLAGGED once into a
    # camera that stays in FLAGGED indefinitely on every visually-similar
    # future scan — even once the model becomes confident again — which is
    # exactly the "too many under review" regression this guard fixes.
    # FLAGGED can still happen fresh on any given scan (the confidence-margin
    # check in inference.py), it just isn't reinforced by this override.
    if prev_risk not in ("BLOCKED", "LOW"):
        return result

    prev_emb = np.frombuffer(previous["embedding"], dtype=np.float32)
    cur_emb  = np.asarray(embedding, dtype=np.float32)
    denom = (np.linalg.norm(cur_emb) * np.linalg.norm(prev_emb)) + 1e-8
    sim = float(np.dot(cur_emb, prev_emb) / denom)

    if sim < SCAN_CONSISTENCY_SIMILARITY:
        return result

    if prev_risk == result.get("risk"):
        log.info("[scan_consistency] %s: matches previous scan (sim=%.4f), already agrees (%s)",
                  campath, sim, prev_risk)
        return result

    log.info("[scan_consistency] %s: overriding %s -> %s (matches previous scan, sim=%.4f)",
              campath, result.get("risk"), prev_risk, sim)

    prev_prediction = {"BLOCKED": "BLOCKED", "LOW": "CLEAR", "FLAGGED": "FLAGGED"}.get(prev_risk, prev_risk)
    prev_label = "blocked" if prev_risk == "BLOCKED" else "clear" if prev_risk == "LOW" else "needing review"
    return {
        **result,
        "risk":         prev_risk,
        "prediction":   prev_prediction,
        "needs_review": prev_risk == "FLAGGED",
        "p_blocked":    previous["p_blocked"] if previous["p_blocked"] is not None else result.get("p_blocked"),
        "heatmap_coverage": previous["heatmap_coverage"],
        "overlay_b64":  previous["overlay_b64"] or result.get("overlay_b64"),
        "heatmap_b64":  previous["heatmap_b64"] or result.get("heatmap_b64"),
        "explanation": (
            f"This frame is visually indistinguishable from this camera's previous "
            f"scan (similarity {sim:.3f}), which was assessed as {prev_label} — "
            "keeping that verdict rather than re-running the classifier's judgement "
            "on what is essentially the same photo."
        ),
    }


async def _apply_correction_memory_override(campath: str, embedding, result: dict) -> dict:
    """
    Check this frame against past human corrections (Mark Blocked/Clear
    clicks) for this same camera. See app/correction_memory.py for the full
    reasoning — this is a safer substitute for fine-tuning the network on
    every click: it never touches the model's weights, it only compares this
    frame's frozen-backbone embedding against past corrections and acts only
    when the match is close.

    - similarity >= SIMILARITY_THRESHOLD: near-duplicate of a corrected frame
      -> reapply that correction's label directly.
    - otherwise: leave `result` untouched — no "flag for review" middle tier
      any more. That used to exist, but real data showed same-camera photos
      routinely land in the 0.97-0.99 range just from sharing the same
      background/framing regardless of actual blockage state, so the middle
      tier was flagging far too much for review without adding real
      confidence either way. See correction_memory.py's docstring.

    `embedding` is computed once per scan by the caller (see the shared
    extract_embedding() call at each live-camera call site) and reused here
    and by _apply_scan_consistency_override, rather than re-running the
    backbone forward pass twice per scan.

    Never raises — a correction-memory hiccup should never take down a live
    scan; it just falls back to trusting the model's own result.
    """
    if not _ENABLE_CORRECTION_MEMORY or embedding is None:
        return result
    try:
        match = await find_correction_match(DB_PATH, campath, embedding)
    except Exception as exc:
        log.warning("[correction_memory] %s: lookup failed (%s) — skipping", campath, exc)
        return result

    if match is None:
        return result

    label, sim = match["label"], match["similarity"]
    log.info("[correction_memory] %s: matched past correction (%s, sim=%.4f)",
              campath, label, sim)

    if result.get("risk") == label:
        return result  # model already agrees — nothing to override
    return {
        **result,
        "prediction": label,
        "risk": label,
        "explanation": (
            f"This frame closely matches one you previously marked {label} "
            f"(similarity {sim:.3f}) — reapplying that correction rather than "
            "the model's fresh read, since the scene appears unchanged."
        ),
    }


def _unknown_result(name: str, reason: str, category: str = "offline") -> dict:
    """
    Return a FLAGGED result for cameras that cannot be assessed at all —
    no image, a blank/placeholder frame, or a view that isn't a drain.

    `category` distinguishes *why* it's flagged, via quality_flags:
      "offline"      — no image / blank frame (hardware or connectivity issue)
      "no_drainage"  — camera view doesn't show a drain (permanently miscategorised camera)
    Both are operational issues, not the model being uncertain about a real
    drainage scene — that's quality_flags=["low_margin"], set separately in
    inference.py's confidence-margin check. Keeping these distinct lets the
    frontend filter "offline / no drainage" apart from genuine model
    uncertainty instead of lumping everything into one FLAGGED pile.
    """
    return {
        "prediction":       "FLAGGED",
        "risk":             "FLAGGED",
        "p_blocked":        None,
        "heatmap_coverage": None,
        "explanation":      reason,
        "overlay_b64":      "",
        "heatmap_b64":      "",
        "quality_flags":    [category],
        "needs_review":     True,
        "camera_name":      name,
    }


# ── Database ──────────────────────────────────────────────────────────────────

async def _init_db() -> None:
    """Create tables and indexes if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Only takes effect on a fresh/empty database file (SQLite ignores an
        # auto_vacuum mode change on a DB that already has pages allocated
        # unless a full VACUUM is run first) — set here so it's in place from
        # the very first write on a rebuilt monitoring.db. Combined with the
        # periodic PRAGMA incremental_vacuum in _db_maintenance() below, this
        # means freed space from the 7-day prune is actually returned to the
        # OS instead of just being reused internally, so the file stops
        # growing once it reaches steady state instead of only being capped
        # by however much data 7 days' worth happens to be at peak.
        await db.execute("PRAGMA auto_vacuum = INCREMENTAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                campath     TEXT    NOT NULL,
                captured_at TEXT    NOT NULL,
                risk        TEXT    NOT NULL,
                p_blocked   REAL    NOT NULL,
                explanation TEXT,
                overlay_b64 TEXT,
                heatmap_b64 TEXT,
                created_at  DATETIME DEFAULT (datetime('now'))
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_campath   ON results(campath)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_created   ON results(created_at)")
        await db.commit()
        # Migration: add heatmap_coverage column if it does not yet exist
        try:
            await db.execute("ALTER TABLE results ADD COLUMN heatmap_coverage REAL DEFAULT NULL")
            await db.commit()
            log.info("[db] Migration: heatmap_coverage column added")
        except Exception:
            pass  # column already present — normal on subsequent startups
        # Migration: add manually_reviewed column if it does not yet exist —
        # marks a row as a human-corrected verdict (via /results/reclassify)
        # rather than a raw model output.
        try:
            await db.execute("ALTER TABLE results ADD COLUMN manually_reviewed INTEGER DEFAULT 0")
            await db.commit()
            log.info("[db] Migration: manually_reviewed column added")
        except Exception:
            pass  # column already present — normal on subsequent startups
        # Migration: add original_b64 column if it does not yet exist — the
        # exact source frame GradCAM ran on, so the frontend can show a
        # genuinely matching Original/Grad-CAM pair instead of re-fetching the
        # live camera (which may have refreshed to a different frame by the
        # time the comparison is displayed).
        try:
            await db.execute("ALTER TABLE results ADD COLUMN original_b64 TEXT DEFAULT NULL")
            await db.commit()
            log.info("[db] Migration: original_b64 column added")
        except Exception:
            pass  # column already present — normal on subsequent startups
        # Migration: add embedding column — the frozen-backbone feature
        # vector for this scan's frame, stored so the NEXT scan of the same
        # camera can be compared against it (see
        # _apply_scan_consistency_override below). Historical rows before
        # this migration simply have NULL here and are skipped when looking
        # back for a previous scan to compare against.
        try:
            await db.execute("ALTER TABLE results ADD COLUMN embedding BLOB DEFAULT NULL")
            await db.commit()
            log.info("[db] Migration: embedding column added")
        except Exception:
            pass  # column already present — normal on subsequent startups
    # corrections table — see app/correction_memory.py
    await init_corrections_table(DB_PATH)
    log.info("[db] Database ready at %s", DB_PATH)


async def _strip_old_images(db) -> None:
    """
    Null out the heavy columns (overlay/heatmap/original images + the
    embedding blob) on rows older than FULL_IMAGE_RETENTION_HOURS, while
    leaving the row itself -- and its lightweight scalar columns (risk,
    p_blocked, explanation, timestamps) -- in place for /results/history and
    /results/stats. See FULL_IMAGE_RETENTION_HOURS's docstring above for why
    this is safe: nothing reads images/embeddings from a row that old.
    Does NOT commit -- caller commits alongside its own writes.
    """
    await db.execute(
        "UPDATE results SET overlay_b64 = '', heatmap_b64 = '', original_b64 = NULL, embedding = NULL "
        "WHERE created_at < datetime('now', ?) "
        "AND (overlay_b64 != '' OR heatmap_b64 != '' OR original_b64 IS NOT NULL OR embedding IS NOT NULL)",
        (f"-{FULL_IMAGE_RETENTION_HOURS} hours",),
    )


async def _save_result(
    name: str, campath: str, captured_at: str, result: dict,
    embedding: Optional[np.ndarray] = None,
) -> None:
    """
    Persist a single inference result. Also runs the two-tier prune inline
    (strip images off old rows, delete very old rows) as a safety net for
    however long it's been since the last save/maintenance run — see
    FULL_IMAGE_RETENTION_HOURS / RESULTS_RETENTION_DAYS above.
    """
    # Store the original frame for BLOCKED, FLAGGED, AND LOW rows now
    # (2026-07-08) — Mark Blocked/Mark Clear is no longer restricted to
    # FLAGGED results; catching a CONFIDENTLY WRONG call (e.g. the model
    # said clear but it's actually blocked) is exactly the case this widened
    # correction feature is for, and that requires the original frame from a
    # LOW-risk row to exist so there's something to run GradCAM on / export
    # as training data. This does grow the database faster than before
    # (LOW is the majority outcome for most cameras) — accepted deliberately
    # for this, given the 30-day pruning below keeps it bounded.
    # _unknown_result() (offline/no-drainage placeholders) never populates a
    # real original_b64 in the first place, so this doesn't start storing
    # anything for those even though they're also tagged risk="FLAGGED".
    original_b64 = result.get("original_b64") if result.get("risk") in ("BLOCKED", "FLAGGED", "LOW") else None

    embedding_bytes = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO results
               (name, campath, captured_at, risk, p_blocked, heatmap_coverage,
                explanation, overlay_b64, heatmap_b64, original_b64, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                campath,
                captured_at,
                result["risk"],
                result.get("p_blocked") if result.get("p_blocked") is not None else 0.0,
                result.get("heatmap_coverage") or 0.0,
                result.get("explanation", ""),
                result.get("overlay_b64", ""),
                result.get("heatmap_b64", ""),
                original_b64,
                embedding_bytes,
            ),
        )
        # Keep the DB lean, on every single save rather than only once a
        # day, so growth stays bounded even if the app is never running at
        # _db_maintenance()'s scheduled time:
        #   1. Strip images/embedding off rows older than
        #      FULL_IMAGE_RETENTION_HOURS -- this is what actually keeps
        #      disk usage flat, since it runs far more often than the final
        #      delete below and image columns are what's heavy.
        #   2. Fully drop rows older than RESULTS_RETENTION_DAYS.
        # Neither shrinks the file on disk by itself — that only frees pages
        # for SQLite to reuse internally. See _db_maintenance() for the
        # periodic incremental_vacuum that actually returns freed space to
        # the OS.
        await _strip_old_images(db)
        await db.execute(
            "DELETE FROM results WHERE created_at < datetime('now', ?)",
            (f"-{RESULTS_RETENTION_DAYS} days",),
        )
        await db.commit()


async def _db_maintenance() -> None:
    """
    Daily housekeeping job. Runs the same two-tier prune as the inline
    safety net in _save_result() (image-stripping + full-row delete — see
    FULL_IMAGE_RETENTION_HOURS above for why this is split in two), then
    PRAGMA incremental_vacuum to actually hand freed pages back to the OS —
    plain DELETE/UPDATE only free pages for SQLite to reuse internally, they
    don't shrink the file. Logs the file size before/after so shrinkage is
    visible in the container logs.

    Added 2026-08-28 after monitoring.db grew to 48GB and filled the
    droplet's disk: 30 days of full-resolution image rows across 56 cameras
    swept every MONITOR_INTERVAL minutes was simply too much data for a
    58GB volume, and nothing was ever returning freed space to the OS. Later
    split into two retention windows (image-stripping vs. full-row delete)
    once it became clear nothing actually needs images beyond a day or two —
    see FULL_IMAGE_RETENTION_HOURS. (The interim fix that only shortened
    RESULTS_RETENTION_DAYS to 7 days was not enough on its own — the DB
    still grew past 38GB on the same 57GB volume, because full-resolution
    images were still being kept for the entire 7-day window.)

    Also prunes the corrections table (app/correction_memory.py) the same
    way, since it was the one other DB table with unbounded growth and no
    retention — much smaller per row than a results row, so it was never
    the cause of the original incident, but there's no reason to leave it
    unbounded either. images/ (the human-reviewed training images
    _export_reviewed_image() saves) is deliberately NOT pruned here: unlike
    results rows or correction embeddings, those files are the real,
    hard-won labelled training data this project's evaluation repeatedly
    identifies as scarce, so deleting them would destroy something valuable
    rather than reclaim disk waste. Its size is only logged, so growth is
    visible rather than silent.
    """
    try:
        before = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        async with aiosqlite.connect(DB_PATH) as db:
            await _strip_old_images(db)
            await db.execute(
                "DELETE FROM results WHERE created_at < datetime('now', ?)",
                (f"-{RESULTS_RETENTION_DAYS} days",),
            )
            await db.commit()
            await db.execute("PRAGMA incremental_vacuum")
            await db.commit()
        after = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        log.info(
            "[db_maintenance] stripped images off rows older than %d hours, pruned rows "
            "older than %d days, incremental_vacuum ran — file size %.1f MB -> %.1f MB",
            FULL_IMAGE_RETENTION_HOURS, RESULTS_RETENTION_DAYS, before / 1e6, after / 1e6,
        )
    except Exception as exc:
        log.warning("[db_maintenance] failed: %s", exc)

    try:
        deleted = await prune_corrections(DB_PATH)
        log.info("[db_maintenance] pruned %d unreachable correction rows", deleted)
    except Exception as exc:
        log.warning("[db_maintenance] corrections prune failed: %s", exc)

    try:
        images_root = Path(__file__).parent.parent / "images"
        if images_root.exists():
            size_mb = sum(f.stat().st_size for f in images_root.rglob("*") if f.is_file()) / 1e6
            log.info(
                "[db_maintenance] images/ (human-reviewed training data, not pruned) "
                "currently %.1f MB — monitor manually if this grows large over time",
                size_mb,
            )
    except Exception as exc:
        log.warning("[db_maintenance] images/ size check failed: %s", exc)


# ── Image fetching ─────────────────────────────────────────────────────────────

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

_MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


async def _fetch_latest_image(
    cam: dict,
    client: httpx.AsyncClient,
) -> tuple[Image.Image, str]:
    """
    Fetch the most recent JPEG for any EA screen camera (Cornwall or Devon).

    Strategy:
      1. Fetch gallerylatestpic.php with browser headers.
      2. Parse the <img src> from the raw HTML — authoritative regardless of path format.
      3. Parse the timestamp text for captured_at.
      4. Fetch the image with a Referer header so the server allows it.

    Returns (PIL Image in RGB, human-readable captured_at string).
    """
    base_url        = cam.get("base_url", BASE_URL)
    gallery_prefix  = cam.get("gallery_prefix", "common")
    campath         = cam["campath"]

    gallery_url = (
        f"{base_url}/{gallery_prefix}/gallerylatestpic.php"
        f"?campath={campath}&name=screen"
    )

    # ── Step 1: fetch gallery page ────────────────────────────────────────────
    try:
        gallery_resp = await client.get(
            gallery_url,
            headers={**_BROWSER_HEADERS, "Referer": base_url + "/"},
        )
        gallery_resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"Gallery unreachable for {campath}: {exc}")

    html = gallery_resp.text
    log.debug("[fetch] gallery HTML for %s:\n%s", campath, html[:800])

    # ── Step 2: find <img src="..."> in the raw HTML ──────────────────────────
    img_match = re.search(
        r'<img\b[^>]+\bsrc=["\']([^"\']+\.jpg)["\']',
        html,
        re.IGNORECASE,
    )

    if img_match:
        raw_src = img_match.group(1)
        if raw_src.startswith("http"):
            img_url = raw_src
        elif raw_src.startswith("/"):
            img_url = base_url + raw_src
        else:
            # Relative to the gallery_prefix directory, e.g. "../devon/..."
            img_url = f"{base_url}/{gallery_prefix}/{raw_src}"
            # Collapse any "prefix/../" segments
            while "/../" in img_url:
                img_url = re.sub(r"/[^/]+/\.\./", "/", img_url)
        log.info("[fetch] %s → %s", campath, img_url)
    else:
        log.warning("[fetch] No <img> in gallery for %s — falling back to timestamp URL", campath)
        ts_match = re.search(
            r"(\d{1,2})\s+(\w{3})\s+(\d{4})\s+(\d{2}):(\d{2})", html
        )
        if not ts_match:
            raise HTTPException(503, f"Camera offline or no image available: {campath}")
        day_s, mon_str, year_s, hour_s, min_s = ts_match.groups()
        month_s = _MONTH_MAP.get(mon_str, "01")
        img_url = (
            f"{base_url}/{campath}"
            f"/{year_s[2:]}/{month_s}/{day_s.zfill(2)}/{hour_s}{min_s}.jpg"
        )

    # ── Step 3: parse captured_at ─────────────────────────────────────────────
    ts_match = re.search(
        r"(\d{1,2})\s+(\w{3})\s+(\d{4})\s+(\d{2}):(\d{2})", html
    )
    if ts_match:
        day_s, mon_str, year_s, hour_s, min_s = ts_match.groups()
        month_s     = _MONTH_MAP.get(mon_str, "01")
        captured_at = f"{day_s.zfill(2)}/{month_s}/{year_s} {hour_s}:{min_s} GMT"
    else:
        captured_at = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M GMT")

    # ── Step 4: fetch image with Referer ─────────────────────────────────────
    try:
        img_resp = await client.get(
            img_url,
            headers={
                **_BROWSER_HEADERS,
                "Accept":  "image/jpeg,image/*,*/*;q=0.8",
                "Referer": gallery_url,
            },
        )
        img_resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"Could not fetch image {img_url}: {exc}")

    ct = img_resp.headers.get("content-type", "")
    if "image" not in ct and len(img_resp.content) < 500:
        raise HTTPException(
            503,
            f"Response from {img_url} does not appear to be an image "
            f"(content-type: {ct!r}, size: {len(img_resp.content)} bytes)",
        )

    try:
        image = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    except Exception:
        raise HTTPException(422, f"Could not decode image from {img_url}")

    return image, captured_at


# ── Background monitor ────────────────────────────────────────────────────────

pipeline: InferencePipeline = None   # single model — set in lifespan

# Specialist per-camera fine-tuning was tried and removed 2026-07-09.
#
# History (kept here for the dissertation record, not as a TODO):
#   - Original 2-camera fine-tune (Wadebridge Polmorla + Chaddlewood,
#     best_model_ft.pt, F1_blocked=0.985) — never reported broken.
#   - Expanded to 20 cameras via best_model_ft_v2.pt (center-crop) — FAILED,
#     confidently predicted CLEAR on a severely-blocked Chaddlewood photo.
#   - Retrained as best_model_ft_v3.pt (letterbox, 20 cameras) — passed
#     spot-checks but then reported predicting too many BLOCKED cases on
#     obviously-clear scenes in real usage. Rolled back entirely.
#   - best_model_ft_v4.pt (3 cameras: BudeCedarGrove, Chaddlewood,
#     BarnstapleBradiford) — live-pair validation looked good (correctly
#     caught a real Barnstaple Bradiford blockage the general model missed),
#     but real production usage the same day showed other mistakes on those
#     same cameras. Reverted.
#   - Wadebridge Polmorla alone re-routed to the original best_model_ft.pt —
#     also reverted the same day at the user's request.
#   - Given every specialist attempt this session ended in rollback, the
#     specialist routing path itself was removed entirely — the single
#     general model (best_model.pt) now serves every camera. The checkpoint
#     files and the training/validation scripts (models/finetune_problem_
#     cameras.py, risk_assessment/test_specialist_vs_general.py) are kept as
#     a record of the methodology and its real-world results, but nothing
#     in production loads or routes to them anymore.


async def monitor_all_cameras() -> None:
    """
    Sweep every camera, run inference, persist to DB.
    Called automatically by APScheduler every MONITOR_INTERVAL minutes.
    """
    if pipeline is None:
        log.warning("[monitor] Model not loaded yet — skipping sweep.")
        return

    log.info("[monitor] Starting sweep of %d cameras…", len(_WEBCAM_LIST))
    ok = err = 0

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for cam in _WEBCAM_LIST:
            try:
                image, captured_at = await _fetch_latest_image(cam, client)
                result              = pipeline.run(image, validate_scene=False)
                embedding           = (
                    pipeline.extract_embedding(image)
                    if (_ENABLE_SCAN_CONSISTENCY or _ENABLE_CORRECTION_MEMORY) else None
                )
                result              = await _apply_scan_consistency_override(cam["campath"], embedding, result)
                result              = await _apply_correction_memory_override(cam["campath"], embedding, result)
                await _save_result(cam["name"], cam["campath"], captured_at, result, embedding)
                log.info(
                    "[monitor] %-28s  %s  (p=%.2f)",
                    cam["name"], result["risk"], result["p_blocked"],
                )
                ok += 1
            except Exception as exc:
                log.warning("[monitor] %-28s  ERROR: %s", cam["name"], exc)
                err += 1

    log.info("[monitor] Sweep done — %d OK / %d errors.", ok, err)


# ── Startup / shutdown ────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    # Load the (single) general model
    checkpoint = os.getenv(
        "CHECKPOINT_PATH",
        str(Path(__file__).parent.parent / "results" / "checkpoints" / "best_model.pt"),
    )
    if not Path(checkpoint).exists():
        log.error("Checkpoint not found: %s — /predict will return 503.", checkpoint)
        pipeline = None
    else:
        log.info("Loading model from %s …", checkpoint)
        try:
            pipeline = InferencePipeline(checkpoint_path=checkpoint)
            log.info("Model ready.")
        except Exception as exc:
            log.error("Model failed to load: %s", exc)
            pipeline = None

    # 2. Initialise database
    await _init_db()

    # 3. Schedule background sweep (first run 2 min after startup)
    first_run = datetime.now(timezone.utc) + timedelta(minutes=2)
    scheduler.add_job(
        monitor_all_cameras,
        trigger="interval",
        minutes=MONITOR_INTERVAL,
        next_run_time=first_run,
        id="webcam_monitor",
        max_instances=1,           # never overlap
        coalesce=True,
    )
    # 4. Schedule daily DB maintenance (prune + incremental_vacuum) at a
    #    quiet hour — see _db_maintenance() docstring for why this exists.
    scheduler.add_job(
        _db_maintenance,
        trigger="cron",
        hour=3,
        minute=15,
        id="db_maintenance",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    log.info(
        "[scheduler] Webcam monitor started — interval=%d min, first run at %s UTC",
        MONITOR_INTERVAL,
        first_run.strftime("%H:%M"),
    )
    log.info(
        "[scheduler] DB maintenance scheduled daily at 03:15 UTC — retention=%d days",
        RESULTS_RETENTION_DAYS,
    )

    yield

    scheduler.shutdown(wait=False)
    pipeline.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drainage Blockage Detection API",
    description=(
        "Upload a CCTV drainage image or pull from live EA Cornwall webcams "
        "and get a risk assessment with Grad-CAM explanation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# EXTRA_CORS_ORIGIN — for a real cloud deployment reachable at a fixed
# http://<server-ip>:5173 (not localhost, not a random trycloudflare.com
# subdomain, neither of which the two rules below already cover). Set this in
# the server's .env, e.g. EXTRA_CORS_ORIGIN=http://203.0.113.5:5173
_extra_origin = os.getenv("EXTRA_CORS_ORIGIN")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        *([_extra_origin] if _extra_origin else []),
    ],
    # Quick Cloudflare Tunnels (`cloudflared tunnel --url ...`) hand out a
    # random *.trycloudflare.com subdomain every time you start one — an
    # exact allow_origins entry would need editing every single session.
    # This regex covers any of those without needing a real, fixed domain.
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Root ──────────────────────────────────────────────────────────────────────
#
# This backend used to also serve a static HTML frontend directly from
# app/static/index.html (mounted at /static, rendered at GET /). That page
# was an early, one-off prototype from before Flood Watcher existed and was
# never touched again once real frontend work moved there — it has been
# removed as dead weight. The actual frontend is Flood Watcher (run
# separately via `npm run dev`, see start.sh); this backend is API-only.

@app.get("/", response_class=JSONResponse)
async def root():
    return JSONResponse({
        "status": "ok",
        "service": "DrainWatch API",
        "frontend": "Run separately — see Flood Watcher/ (start.sh launches both).",
    })


# ── Single upload ─────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyse a drainage CCTV image uploaded as a file.

    Returns JSON:
        prediction  "BLOCKED" | "CLEAR"
        risk        "BLOCKED" | "LOW"
        p_blocked   float
        explanation str
        overlay_b64 str   base64 JPEG — Grad-CAM overlay
        heatmap_b64 str   base64 JPEG — raw heatmap
    """
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check CHECKPOINT_PATH.")
    try:
        # allow_flagged=False: this result has no campath/DB row, so a
        # FLAGGED outcome would have no way to be resolved (Mark Blocked/
        # Clear needs a campath). Always commit to BLOCKED or CLEAR instead.
        result = pipeline.run(image, validate_scene=True, allow_flagged=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return JSONResponse(content=result)


# ── Batch upload ──────────────────────────────────────────────────────────────

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Analyse up to 50 images in one request.
    Returns results sorted HIGH risk first.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check CHECKPOINT_PATH.")
    results = []
    for file in files:
        try:
            contents = await file.read()
            image    = Image.open(io.BytesIO(contents)).convert("RGB")
            # allow_flagged=False: same reasoning as /predict — batch-uploaded
            # images have no campath/DB row, so FLAGGED would be a dead end.
            result   = pipeline.run(image, validate_scene=True, allow_flagged=False)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e), "risk": "ERROR"})

    # Sort: BLOCKED first, LOW, FLAGGED (poor quality), OFFLINE, errors last.
    # (Previously keyed on "HIGH", which the model never actually returns —
    # see the risk-value mismatch fixed across the frontend earlier; this
    # sort silently never put a blocked result first because of it.)
    order = {"BLOCKED": 0, "CAUTION": 0, "LOW": 1, "FLAGGED": 2, "OFFLINE": 3, "ERROR": 4}
    results.sort(key=lambda r: (order.get(r.get("risk", "ERROR"), 4), -(r.get("p_blocked") or 0)))
    return JSONResponse(content={"total": len(results), "results": results})


# ── Live webcam ───────────────────────────────────────────────────────────────

@app.get("/webcam/image")
async def webcam_image(campath: str = Query(...)):
    """
    Proxy the latest JPEG for a camera.
    The browser cannot fetch EA webcam images directly (no CORS headers on their servers),
    so we fetch them server-side and forward the raw bytes.
    """
    cam = next((c for c in _WEBCAM_LIST if c["campath"] == campath), None)
    if cam is None:
        cam = {"campath": campath, "base_url": BASE_URL, "gallery_prefix": "common", "region": "Cornwall"}

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            image, _ = await _fetch_latest_image(cam, client)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Image unavailable for {campath}")


@app.get("/webcams")
async def list_webcams():
    """Return the list of monitored Cornwall EA screen cameras."""
    return JSONResponse(content={"cameras": _WEBCAM_LIST})


@app.post("/predict/webcam")
async def predict_webcam(
    campath: str = Query(..., description="e.g. cornwall/BodminPetrocsWell/cam1"),
    camname: str = Query("", description="Human-readable camera name"),
):
    """
    Fetch the latest image from an EA Cornwall webcam and run blockage detection.
    Result is also persisted to the monitoring database.

    Returns the same fields as /predict plus:
        camera_name  str — human label
        captured_at  str — image timestamp
        source_url   str — camera page URL
    """
    # Find the full cam dict (needed for base_url / gallery_prefix)
    cam = next((c for c in _WEBCAM_LIST if c["campath"] == campath), None)
    if cam is None:
        # Fallback: treat as Cornwall if not in list
        cam = {"campath": campath, "base_url": BASE_URL, "gallery_prefix": "common"}

    name = camname or campath
    captured_at = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M GMT")

    # Try fetching the image — return FLAGGED if unavailable or undecodable
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            image, captured_at = await _fetch_latest_image(cam, client)
    except HTTPException:
        result = _unknown_result(
            name,
            "No image could be retrieved from this camera — it may be temporarily offline. "
            "Please check the EA webcam site directly.",
            category="offline",
        )
        result["captured_at"] = captured_at
        result["source_url"]  = f"{BASE_URL}/{campath}"
        result["filename"]    = name.replace(" ", "_")
        await _save_result(name, campath, captured_at, result)
        return JSONResponse(content=result)

    # Detect "CAMERA OFFLINE" placeholder images — uniform grey frames with no
    # real content.  If the image has very low variance and near-neutral colour,
    # it is almost certainly a placeholder, not a live drainage scene.
    _arr  = np.array(image, dtype=np.float32)
    _r, _g, _b = _arr[:,:,0], _arr[:,:,1], _arr[:,:,2]
    _mx   = np.maximum(np.maximum(_r, _g), _b)
    _mn   = np.minimum(np.minimum(_r, _g), _b)
    _sat  = float(np.where(_mx > 0, (_mx - _mn) / _mx, 0.0).mean())
    _std  = float(_arr.mean(axis=2).std())
    # Detect "CAMERA OFFLINE" placeholder: it is a pure B&W graphic — saturation
    # is essentially 0.  Real drain images (even grey concrete in overcast light)
    # always have at least slight colour variation (sat > 0.01).
    if _sat < 0.01 and _std > 5:
        result = _unknown_result(
            name,
            "The camera appears to be offline or transmitting a blank frame. "
            "Manual inspection is required.",
            category="offline",
        )
        result["captured_at"] = captured_at
        result["source_url"]  = f"{BASE_URL}/{campath}"
        result["filename"]    = name.replace(" ", "_")
        await _save_result(name, campath, captured_at, result)
        return JSONResponse(content=result)

    # Cameras with no visible drain — always flag for manual review
    if campath in _ALWAYS_REVIEW_CAMERAS:
        result = _unknown_result(
            name,
            "This camera does not show a drainage structure. Manual inspection required.",
            category="no_drainage",
        )
        result["captured_at"] = captured_at
        result["source_url"]  = f"{BASE_URL}/{campath}"
        result["filename"]    = name.replace(" ", "_")
        await _save_result(name, campath, captured_at, result)
        return JSONResponse(content=result)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check CHECKPOINT_PATH.")
    try:
        result = pipeline.run(image, validate_scene=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Computed once and reused by both embedding-based overrides below,
    # rather than re-running the frozen backbone forward pass twice per scan.
    # Skipped entirely while both are disabled (see _ENABLE_SCAN_CONSISTENCY
    # / _ENABLE_CORRECTION_MEMORY above).
    embedding = (
        pipeline.extract_embedding(image)
        if (_ENABLE_SCAN_CONSISTENCY or _ENABLE_CORRECTION_MEMORY) else None
    )

    result = await _apply_scan_consistency_override(campath, embedding, result)
    result = await _apply_correction_memory_override(campath, embedding, result)
    await _save_result(name, campath, captured_at, result, embedding)

    result["camera_name"] = name
    result["captured_at"] = captured_at
    result["source_url"]  = f"{BASE_URL}/{campath}"
    result["filename"]    = name.replace(" ", "_")

    return JSONResponse(content=result)


# ── Monitoring history endpoints ──────────────────────────────────────────────

@app.get("/results/latest")
async def results_latest():
    """
    Return the most recent inference result for every camera that has been
    scanned at least once. Used by the frontend to populate camera cards on load.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT r.id, r.name, r.campath, r.captured_at,
                   r.risk, r.p_blocked, r.heatmap_coverage, r.explanation,
                   r.overlay_b64, r.heatmap_b64, r.original_b64, r.created_at
            FROM   results r
            INNER JOIN (
                SELECT campath, MAX(id) AS max_id
                FROM   results
                GROUP  BY campath
            ) latest ON r.id = latest.max_id
            ORDER  BY r.name ASC
        """)
        rows = await cursor.fetchall()
    return JSONResponse(content={"results": [dict(r) for r in rows]})


@app.get("/results/history")
async def results_history(
    campath: Optional[str] = Query(None, description="Filter to one camera"),
    days:    int           = Query(7,    description="How many days back to query"),
    limit:   int           = Query(200,  description="Max rows returned"),
):
    """
    Return a filtered result log — without image blobs to keep responses small.
    Used for per-camera drill-downs and audit trails.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if campath:
            cursor = await db.execute(
                """SELECT id, name, campath, captured_at, risk, p_blocked,
                          heatmap_coverage, created_at
                   FROM   results
                   WHERE  campath = ?
                   AND    created_at >= datetime('now', ? || ' days')
                   ORDER  BY created_at DESC
                   LIMIT  ?""",
                (campath, f"-{days}", limit),
            )
        else:
            cursor = await db.execute(
                """SELECT id, name, campath, captured_at, risk, p_blocked,
                          heatmap_coverage, created_at
                   FROM   results
                   WHERE  created_at >= datetime('now', ? || ' days')
                   ORDER  BY created_at DESC
                   LIMIT  ?""",
                (f"-{days}", limit),
            )
        rows = await cursor.fetchall()
    return JSONResponse(content={"results": [dict(r) for r in rows]})


@app.get("/results/stats")
async def results_stats(days: int = Query(30, description="Window in days")):
    """
    Return daily HIGH / LOW counts for the trend chart on the dashboard.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT DATE(created_at)                                  AS day,
                      COUNT(*)                                          AS total,
                      SUM(CASE WHEN risk = 'BLOCKED' THEN 1 ELSE 0 END)  AS high,
                      SUM(CASE WHEN risk = 'LOW'  THEN 1 ELSE 0 END)  AS low,
                      ROUND(AVG(p_blocked), 3)                         AS avg_p_blocked
               FROM   results
               WHERE  created_at >= datetime('now', ? || ' days')
               GROUP  BY DATE(created_at)
               ORDER  BY day ASC""",
            (f"-{days}",),
        )
        rows = await cursor.fetchall()
    return JSONResponse(content={"stats": [dict(r) for r in rows]})


def _export_reviewed_image(campath: str, risk: str, original_b64: str, row_id: int) -> None:
    """
    Save a human-corrected result straight to images/<camera>/blocked|clear/
    the moment the correction happens — turns "someone caught the model
    being wrong" into real training data immediately, with no separate
    export script to remember to run. See data/export_reviewed_images.py
    for the batch/backfill version of this same logic (kept in sync via
    data/camera_folder_map.py, imported as _folder_for above).

    Best-effort only: a failure here must never break the actual
    reclassify request the user is waiting on, so all errors are caught
    and logged, not raised.
    """
    if risk not in ("BLOCKED", "LOW") or not original_b64:
        return
    try:
        label_dir = "blocked" if risk == "BLOCKED" else "clear"
        images_root = Path(__file__).parent.parent / "images"
        dest_dir = images_root / _folder_for(campath) / label_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"reviewed_{row_id}.jpg"
        if not dest_path.exists():
            dest_path.write_bytes(base64.b64decode(original_b64))
            log.info("[auto-export] %s -> %s", campath, dest_path.relative_to(images_root.parent))
    except Exception as exc:
        log.warning("[auto-export] %s: failed to save training image (%s)", campath, exc)


@app.post("/results/reclassify")
async def reclassify_result(
    campath: str = Query(..., description="Camera path, e.g. cornwall/BodminPetrocsWell/cam1"),
    risk:    str = Query(..., description="Corrected verdict: 'BLOCKED' or 'LOW'"),
):
    """
    Manually resolve an uncertain (FLAGGED) result into a definite BLOCKED or
    LOW verdict. Used by the "Uncertain" review queue — the model's own
    confidence-margin check (see inference.py) routes anything too close to
    the decision boundary to FLAGGED rather than guessing, and this is how a
    human reviewer closes that loop, persisting the correction and marking
    the row so it's traceable later as a human call rather than a raw model
    output. Updates only the most recent stored result for this camera.

    When marking BLOCKED: also generates a real GradCAM overlay from the
    stored original frame. The automatic pipeline never ran GradCAM for this
    scan (it only runs for the model's OWN confident BLOCKED calls, not a
    FLAGGED one a human is now overriding) — without this, the frontend's
    Original/Grad-CAM comparison would just show the plain photo twice, with
    no actual heatmap. Falls back to a plain risk/p_blocked update, no
    overlay, if this row predates the original_b64-for-FLAGGED-rows change
    (main.py, 2026-07-07) and so has nothing to run GradCAM on.
    """
    if risk not in ("BLOCKED", "LOW"):
        raise HTTPException(status_code=400, detail="risk must be 'BLOCKED' or 'LOW'")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, original_b64 FROM results WHERE campath = ? ORDER BY id DESC LIMIT 1",
            (campath,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No stored result found for {campath}")

        result_id   = row["id"]
        original_b64 = row["original_b64"]
        p_blocked   = 1.0 if risk == "BLOCKED" else 0.0

        overlay_fields = {}
        if risk == "BLOCKED" and original_b64:
            if pipeline is not None:
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(original_b64))).convert("RGB")
                    overlay_fields = pipeline.generate_blocked_overlay(image)
                except Exception as exc:
                    log.warning("[reclassify] %s: overlay generation failed (%s) — "
                                "updating risk only.", campath, exc)
        elif risk == "BLOCKED":
            log.info("[reclassify] %s: no stored original_b64 for this row (predates the "
                     "2026-07-07 fix) — updating risk only, no overlay generated.", campath)

        if overlay_fields:
            await db.execute(
                """UPDATE results
                   SET risk = ?, p_blocked = ?, manually_reviewed = 1,
                       overlay_b64 = ?, heatmap_b64 = ?, heatmap_coverage = ?, explanation = ?
                   WHERE id = ?""",
                (risk, p_blocked, overlay_fields["overlay_b64"], overlay_fields["heatmap_b64"],
                 overlay_fields["heatmap_coverage"], overlay_fields["explanation"], result_id),
            )
        else:
            await db.execute(
                "UPDATE results SET risk = ?, p_blocked = ?, manually_reviewed = 1 WHERE id = ?",
                (risk, p_blocked, result_id),
            )
        await db.commit()

    # Every correction becomes real training data immediately — no separate
    # export step to remember. Uses whichever original_b64 was already on
    # the row (the frame this correction is actually about), not a fresh
    # fetch of the live camera, which could show something different by now.
    _export_reviewed_image(campath, risk, original_b64, result_id)

    # Also feed the correction-memory system (app/correction_memory.py):
    # store this frame's frozen-backbone embedding + the corrected label, so
    # a future near-duplicate scan of the same still-unresolved scene reuses
    # this correction instead of repeating the model's mistake. No weight
    # updates — see that module's docstring for why. Best-effort only: never
    # let a correction-memory failure break the reclassify request itself.
    if pipeline is not None and original_b64:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(original_b64))).convert("RGB")
            embedding = pipeline.extract_embedding(image)
            await add_correction(DB_PATH, campath, embedding, risk, source_result_id=result_id)
            log.info("[correction_memory] %s: stored correction (%s)", campath, risk)
        except Exception as exc:
            log.warning("[correction_memory] %s: failed to store correction (%s)", campath, exc)

        # True live learning (app/live_head.py, v2) — retrain the shared
        # online classification head on every correction stored so far
        # (global, across all cameras), but only accept it if it (a) stays
        # anchored close to the original weights when data is scarce and
        # (b) still agrees with the model's own past confident calls on a
        # random historical sample. See that module's docstring for why
        # v1 (neither of those checks) caused real damage. Best-effort
        # only — a failure here must never break the reclassify request.
        if _ENABLE_LIVE_HEAD and pipeline._original_fc_weight is not None:
            try:
                all_corrections = await get_all_corrections(DB_PATH)
                head = retrain_head(pipeline._original_fc_weight, pipeline._original_fc_bias, all_corrections)
                if head is None:
                    log.info("[live_head] %s: not enough corrections yet (need at least one "
                              "BLOCKED and one LOW, globally) — head unchanged", campath)
                else:
                    canary_embeddings = await _sample_canary_embeddings(pipeline)
                    canary = evaluate_against_canary(
                        head["weight"], head["bias"],
                        pipeline._original_fc_weight, pipeline._original_fc_bias,
                        canary_embeddings,
                    )
                    if canary is None:
                        log.warning("[live_head] %s: REJECTED — no historical canary data "
                                    "available to check this update against, so it can't be "
                                    "verified safe (n=%d, train_acc=%.3f). Keeping current head "
                                    "rather than accepting on faith.", campath, head["n"], head["train_acc"])
                        accept = False
                    elif canary["agreement"] < CANARY_MIN_AGREEMENT:
                        log.warning("[live_head] %s: REJECTED — candidate head only agreed with "
                                    "the model's own past confident calls %.1f%% of the time "
                                    "(need >= %.1f%%, n=%d canary cases). Keeping current head.",
                                    campath, canary["agreement"] * 100, CANARY_MIN_AGREEMENT * 100,
                                    canary["n"])
                        accept = False
                    else:
                        log.info("[live_head] %s: candidate passed canary check (%.1f%% agreement, "
                                  "n=%d)", campath, canary["agreement"] * 100, canary["n"])
                        accept = True

                    if accept:
                        pipeline.refresh_live_head(head["weight"], head["bias"])
                        save_live_head(head["weight"], head["bias"],
                                        {"n": head["n"], "train_acc": head["train_acc"]})
                        log.info("[live_head] retrained on %d corrections (train_acc=%.3f) — "
                                  "now live, no restart needed", head["n"], head["train_acc"])
            except Exception as exc:
                log.warning("[live_head] retrain failed (%s) — leaving current head in place", exc)

    log.info("[reclassify] %s -> %s (manual%s)", campath, risk,
              ", overlay generated" if overlay_fields else "")
    return JSONResponse(content={
        "ok": True, "campath": campath, "risk": risk,
        "overlay_generated": bool(overlay_fields),
        # Fresh overlay/heatmap/explanation so the frontend can update its
        # already-loaded result in place, instead of needing a full refetch.
        "overlay_b64":      overlay_fields.get("overlay_b64"),
        "heatmap_b64":      overlay_fields.get("heatmap_b64"),
        "heatmap_coverage": overlay_fields.get("heatmap_coverage"),
        "explanation":      overlay_fields.get("explanation"),
    })


# ── Static map (server-side tile compositing for PDF export) ──────────────────
#
# The PDF report is built client-side with jsPDF, which needs raw pixel data
# to embed an image. A browser can't get that from OpenStreetMap's tiles
# directly — OSM's tile servers don't send CORS headers, so any attempt to
# read them back out of a <canvas> (which is what screenshotting a live
# Leaflet map requires) throws a "tainted canvas" security error. There's no
# client-side way around that short of a paid basemap provider.
#
# Doing the compositing here sidesteps the whole problem: server-to-server
# HTTP requests have no CORS restriction at all — that's purely a browser
# concept. This endpoint fetches the OSM tiles itself, stitches them into one
# image sized to fit the scan's camera locations, draws the routes and pins
# on top with Pillow, and hands the frontend a single ready-to-embed PNG.

_TILE_SIZE      = 256
_OSM_TILE_URL   = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_OSM_USER_AGENT = "DrainageBlockageReport/1.0 (dissertation project; static map for PDF export)"


class StaticMapMarker(BaseModel):
    lat: float
    lng: float
    label: str = ""   # short label drawn INSIDE the pin (usually a stop number)
    name:  str = ""   # camera name drawn NEXT TO the pin, on a legible pill background
    colour: str = "#c0463a"


class StaticMapRoute(BaseModel):
    points: List[StaticMapMarker]
    colour: str = "#c0463a"


class StaticMapRequest(BaseModel):
    markers: List[StaticMapMarker] = []
    routes:  List[StaticMapRoute]  = []
    width:   int = 900
    height:  int = 480


def _lonlat_to_px(lon: float, lat: float, zoom: int) -> tuple:
    """Web Mercator projection → pixel coordinates at a given zoom level."""
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * _TILE_SIZE
    lat_rad = math.radians(max(min(lat, 85.05), -85.05))
    y = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * _TILE_SIZE
    return x, y


def _choose_zoom(min_lat, max_lat, min_lon, max_lon, out_w, out_h, max_zoom=15) -> int:
    """Largest zoom level at which the whole bounding box still fits the output image."""
    for zoom in range(max_zoom, 0, -1):
        x1, y1 = _lonlat_to_px(min_lon, max_lat, zoom)
        x2, y2 = _lonlat_to_px(max_lon, min_lat, zoom)
        if abs(x2 - x1) <= out_w and abs(y2 - y1) <= out_h:
            return zoom
    return 1


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    if len(h) != 6:
        return (192, 70, 58)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _declutter_positions(coords, project_fn, min_dist: float = 13.0) -> dict:
    """
    Two real-world cameras can be very close together (e.g. 400m apart) while
    the map covers an entire region — at that zoom, both project to the same
    pixel, so the second pin drawn completely hides the first with no visible
    connecting line. This nudges any point landing within `min_dist` px of an
    already-placed point outward in a small ring so every pin stays visible.

    coords:      list of (lat, lon), duplicates allowed.
    project_fn:  (lat, lon) -> (x, y) raw pixel position.
    Returns {(lat, lon) rounded to 6dp: (x, y)} — the position to actually
    draw at, for both pins and route line endpoints, so lines still connect
    to where the pin really ends up.
    """
    placed: list = []
    adjusted: dict = {}
    for lat, lon in coords:
        key = (round(lat, 6), round(lon, 6))
        if key in adjusted:
            continue
        x, y = project_fn(lat, lon)
        collisions = [p for p in placed if math.hypot(p[0] - x, p[1] - y) < min_dist]
        if collisions:
            k = len(collisions)
            angle = k * 2.399963  # golden angle — spreads successive collisions apart evenly
            radius = min_dist * (1.0 + 0.65 * k)
            x += radius * math.cos(angle)
            y += radius * math.sin(angle)
        placed.append((x, y))
        adjusted[key] = (x, y)
    return adjusted


@app.post("/staticmap")
async def staticmap(req: StaticMapRequest):
    """Compose a real OSM basemap with routes/pins drawn on top, as one PNG."""
    all_points = [(m.lat, m.lng) for m in req.markers] + \
                 [(p.lat, p.lng) for r in req.routes for p in r.points]

    if not all_points:
        img = Image.new("RGB", (req.width, req.height), (245, 246, 248))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")

    lats = [p[0] for p in all_points]
    lons = [p[1] for p in all_points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Pad the bounding box so markers aren't flush against the image edge.
    lat_pad = max((max_lat - min_lat) * 0.18, 0.02)
    lon_pad = max((max_lon - min_lon) * 0.18, 0.02)
    min_lat -= lat_pad; max_lat += lat_pad
    min_lon -= lon_pad; max_lon += lon_pad

    zoom = _choose_zoom(min_lat, max_lat, min_lon, max_lon, req.width, req.height)

    centre_lat = (min_lat + max_lat) / 2
    centre_lon = (min_lon + max_lon) / 2
    cx, cy = _lonlat_to_px(centre_lon, centre_lat, zoom)
    top_left_x = cx - req.width / 2
    top_left_y = cy - req.height / 2

    tile_x0 = int(top_left_x // _TILE_SIZE)
    tile_y0 = int(top_left_y // _TILE_SIZE)
    tile_x1 = int((top_left_x + req.width) // _TILE_SIZE)
    tile_y1 = int((top_left_y + req.height) // _TILE_SIZE)
    n_tiles = 2 ** zoom

    canvas = Image.new(
        "RGB",
        ((tile_x1 - tile_x0 + 1) * _TILE_SIZE, (tile_y1 - tile_y0 + 1) * _TILE_SIZE),
        (230, 230, 230),
    )

    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": _OSM_USER_AGENT}) as client:
        for tx in range(tile_x0, tile_x1 + 1):
            for ty in range(tile_y0, tile_y1 + 1):
                if tx < 0 or ty < 0 or tx >= n_tiles or ty >= n_tiles:
                    continue
                try:
                    resp = await client.get(_OSM_TILE_URL.format(z=zoom, x=tx, y=ty))
                    if resp.status_code == 200:
                        tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        canvas.paste(tile_img, ((tx - tile_x0) * _TILE_SIZE, (ty - tile_y0) * _TILE_SIZE))
                except Exception as exc:
                    log.warning("[staticmap] tile %d/%d/%d failed: %s", zoom, tx, ty, exc)

    crop_x = int(top_left_x - tile_x0 * _TILE_SIZE)
    crop_y = int(top_left_y - tile_y0 * _TILE_SIZE)
    canvas = canvas.crop((crop_x, crop_y, crop_x + req.width, crop_y + req.height))

    draw = ImageDraw.Draw(canvas)

    def project(lat: float, lon: float) -> tuple:
        x, y = _lonlat_to_px(lon, lat, zoom)
        return x - top_left_x, y - top_left_y

    # Nudge apart any markers/route points that would otherwise land on top
    # of each other (see _declutter_positions docstring). Built from every
    # coordinate that will actually be drawn — markers AND route points —
    # so a route line always ends exactly where its stop's pin is drawn.
    all_coords = [(m.lat, m.lng) for m in req.markers] + \
                 [(p.lat, p.lng) for r in req.routes for p in r.points]
    declutter = _declutter_positions(all_coords, project)

    def project_adj(lat: float, lon: float) -> tuple:
        return declutter.get((round(lat, 6), round(lon, 6)), project(lat, lon))

    # Routes first, so pins draw on top of the lines
    for route in req.routes:
        colour = _hex_to_rgb(route.colour)
        pts = [project_adj(p.lat, p.lng) for p in route.points]
        if len(pts) >= 2:
            draw.line(pts, fill=colour, width=4)
            draw.line(pts, fill=(255, 255, 255), width=1)  # thin highlight for contrast

    for m in req.markers:
        colour = _hex_to_rgb(m.colour)
        x, y = project_adj(m.lat, m.lng)
        r = 9
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline=(255, 255, 255), width=2)
        if m.label:
            bbox = draw.textbbox((0, 0), m.label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2 - 1), m.label, fill=(255, 255, 255))

        # Camera name, on a legible white pill — plain text directly on a
        # basemap disappears over water/forest tiles, so give it a background.
        if m.name:
            pad_x, pad_y = 5, 3
            bbox = draw.textbbox((0, 0), m.name)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_x, label_y = x + r + 5, y - th / 2 - pad_y
            draw.rounded_rectangle(
                (label_x, label_y, label_x + tw + pad_x * 2, label_y + th + pad_y * 2),
                radius=4, fill=(255, 255, 255), outline=(210, 210, 210), width=1,
            )
            draw.text((label_x + pad_x, label_y + pad_y - 1), m.name, fill=(40, 40, 40))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/debug/webcam")
async def debug_webcam(campath: str = Query("cornwall/BodminPetrocsWell/cam1")):
    """
    Returns raw diagnostic info for one camera — gallery HTML, parsed img URL,
    image fetch status. Open in browser to diagnose connectivity issues.
    """
    cam = next((c for c in _WEBCAM_LIST if c["campath"] == campath), None)
    if cam is None:
        cam = {"campath": campath, "base_url": BASE_URL, "gallery_prefix": "common", "region": "unknown"}
    base_url       = cam["base_url"]
    gallery_prefix = cam["gallery_prefix"]
    gallery_url    = f"{base_url}/{gallery_prefix}/gallerylatestpic.php?campath={campath}&name=screen"
    result = {"campath": campath, "region": cam.get("region"), "gallery_url": gallery_url}

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        # Step 1: gallery page
        try:
            r = await client.get(
                gallery_url,
                headers={**_BROWSER_HEADERS, "Referer": base_url + "/"},
            )
            result["gallery_status"] = r.status_code
            result["gallery_content_type"] = r.headers.get("content-type", "")
            result["gallery_html"] = r.text[:2000]
        except Exception as exc:
            result["gallery_error"] = str(exc)
            return JSONResponse(content=result)

        # Step 2: parse img src
        html = r.text
        img_match = re.search(r'<img\b[^>]+\bsrc=["\']([^"\']+\.jpg)["\']', html, re.IGNORECASE)
        if img_match:
            raw_src = img_match.group(1)
            if raw_src.startswith("http"):
                img_url = raw_src
            elif raw_src.startswith("/"):
                img_url = BASE_URL + raw_src
            else:
                img_url = (BASE_URL + "/common/" + raw_src).replace("/common/../", "/")
            result["img_src_found"] = raw_src
            result["img_url_resolved"] = img_url
        else:
            result["img_src_found"] = None
            result["img_url_resolved"] = None
            result["note"] = "No <img src> found in gallery HTML"
            return JSONResponse(content=result)

        # Step 3: try fetching the image
        try:
            ir = await client.get(
                img_url,
                headers={**_BROWSER_HEADERS, "Accept": "image/*", "Referer": gallery_url},
            )
            result["image_status"] = ir.status_code
            result["image_content_type"] = ir.headers.get("content-type", "")
            result["image_size_bytes"] = len(ir.content)
        except Exception as exc:
            result["image_error"] = str(exc)

    return JSONResponse(content=result)


@app.get("/health")
async def health():
    next_run = None
    job = scheduler.get_job("webcam_monitor")
    if job and job.next_run_time:
        next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M UTC")
    return {
        "status":          "ok" if pipeline is not None else "degraded",
        "model_loaded":    pipeline is not None,
        "monitor_running": scheduler.running,
        "next_sweep":      next_run,
        "cameras":         len(_WEBCAM_LIST),
    }
