"""
correction_memory.py
---------------------
"Live learning from clicks" without touching the trained model's weights.

Why not just fine-tune on every correction? models/finetune_problem_cameras.py
already tried that, four times (v2-v5), each with proper batches, held-out
validation splits, weighted sampling, and early stopping — real safeguards a
per-click update would not have. v2 passed validation and then confidently
called a severely blocked site CLEAR in production. v3 regressed live in a
way never root-caused. v4 passed live-pair validation and still made real
mistakes once deployed. A single-image online weight update has none of
those checks, so it would be strictly riskier than four attempts that
already didn't reliably help — specialist per-camera routing was removed
from app/main.py as a direct result.

What this does instead: every human correction (Mark Blocked/Clear) is
stored as (frozen-backbone embedding, label), scoped to the camera it came
from. On each new scan, if the new frame's embedding is a close match to a
stored correction for that same camera, the correction's label is trusted
over the model's fresh prediction. The model's weights never change — this
is a generalised, embedding-based version of the per-camera
_apply_never_auto_clear/block_override already in app/main.py, except it's
learned from real operator clicks instead of hardcoded to specific camera
names.

These are static, fixed-position webcams: a real blockage physically sits
in frame for days or weeks until someone clears it, so consecutive scans of
an uncorrected scene are near-duplicate images. That's exactly the case
this is built to catch cheaply and safely — it does nothing useful for a
genuinely novel-looking frame that doesn't resemble any past correction,
which is the correct, honest limitation to have rather than pretending a
single click can teach the network something general.

Single similarity threshold: similarity >= SIMILARITY_THRESHOLD -> directly
reapply the past correction's label. Below it -> no override at all, trust
the model's raw prediction.

This used to be two tiers (a lower "visually close but not sure" band that
downgraded to FLAGGED instead of asserting). Real production data killed
that: same-camera photos taken at different times routinely land in the
0.97-0.99 range purely from sharing the same background/framing/lighting,
regardless of whether the actual blockage state matches — e.g. a real match
came back at similarity 0.989 against a correction that no longer applied.
With a two-tier design, that whole common range was flagging for review on
every scan, which is what caused "too many under review". A single
threshold is both simpler and produces less unnecessary review load: it
either has real evidence to act on, or it says nothing and gets out of the
way.

IMPORTANT — SIMILARITY_THRESHOLD is a starting point, not calibrated
against real embeddings. This was written without GPU/model access (the
assistant's sandbox has neither torch nor these checkpoints), so the actual
cosine-similarity distribution for "same scene, different scan" vs
"genuinely different scene" on YOUR cameras is unknown. Every match this
runs logs its similarity score (see app/main.py's call site) specifically so
you can look at real numbers after a few days of live traffic and retune
this before fully trusting its auto-assert behaviour.
"""

from typing import Optional

import numpy as np
import aiosqlite

SIMILARITY_THRESHOLD = 0.985  # near-duplicate frame -> trust the past correction directly
MAX_STORED_PER_CAMERA = 200  # most recent corrections considered per camera, per lookup


async def init_corrections_table(db_path: str) -> None:
    """Create the corrections table/index if they don't exist yet."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                campath           TEXT NOT NULL,
                embedding         BLOB NOT NULL,
                label             TEXT NOT NULL,
                source_result_id  INTEGER,
                created_at        DATETIME DEFAULT (datetime('now'))
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_corrections_campath ON corrections(campath)")
        await db.commit()


async def add_correction(
    db_path: str,
    campath: str,
    embedding: np.ndarray,
    label: str,
    source_result_id: Optional[int] = None,
) -> None:
    """
    Record a human correction. `label` should be "BLOCKED" or "LOW" — matches
    the `risk` field convention used everywhere else in this app.
    """
    emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO corrections (campath, embedding, label, source_result_id) "
            "VALUES (?, ?, ?, ?)",
            (campath, emb_bytes, label, source_result_id),
        )
        await db.commit()


async def find_correction_match(
    db_path: str,
    campath: str,
    embedding: np.ndarray,
) -> Optional[dict]:
    """
    Compare `embedding` against this camera's stored corrections (most
    recent MAX_STORED_PER_CAMERA only, so this stays cheap indefinitely).

    Returns None if nothing clears SIMILARITY_THRESHOLD. Otherwise returns:
        {"label": "BLOCKED"|"LOW", "similarity": float}
    """
    query = np.asarray(embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-8)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT embedding, label FROM corrections WHERE campath = ? "
            "ORDER BY id DESC LIMIT ?",
            (campath, MAX_STORED_PER_CAMERA),
        )
        rows = await cursor.fetchall()

    if not rows:
        return None

    best_sim, best_label = -1.0, None
    for row in rows:
        stored = np.frombuffer(row["embedding"], dtype=np.float32)
        stored_norm = stored / (np.linalg.norm(stored) + 1e-8)
        sim = float(np.dot(query_norm, stored_norm))
        if sim > best_sim:
            best_sim, best_label = sim, row["label"]

    if best_sim >= SIMILARITY_THRESHOLD:
        return {"label": best_label, "similarity": best_sim}
    return None


async def prune_corrections(db_path: str, keep_per_camera: int = 500) -> int:
    """
    Bound the corrections table's growth. Only the most recent
    MAX_STORED_PER_CAMERA (200) rows per camera are ever read by
    find_correction_match, and only the most recent 5000 overall are ever
    read by get_all_corrections — anything beyond that is pure disk usage
    with no code path that will ever look at it again. keep_per_camera=500
    is a generous multiple of what's actually used, so this only ever
    removes rows that are already unreachable in practice.

    Each row is small (a single embedding BLOB, a few KB) compared to a
    full-resolution image, so this table was never the cause of the
    disk-full incident documented in app/main.py — this exists as the same
    kind of safety net, not because it's urgent on its own.

    Returns the number of rows deleted.
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            DELETE FROM corrections
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY campath ORDER BY id DESC
                    ) AS rn
                    FROM corrections
                ) WHERE rn > ?
            )
            """,
            (keep_per_camera,),
        )
        await db.commit()
        return cursor.rowcount


async def get_all_corrections(db_path: str, limit: int = 5000) -> list:
    """
    Every stored correction, most recent first, across ALL cameras — the
    global pool app/live_head.py retrains the shared online classification
    head on. Not scoped per camera: the base model is already a single
    classifier shared across every camera, so a shared head update is
    consistent with how it works already, and pooling gives it more to
    learn from than any single camera's corrections alone would.
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT embedding, label FROM corrections ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
    return [
        {"embedding": np.frombuffer(row["embedding"], dtype=np.float32), "label": row["label"]}
        for row in rows
    ]
