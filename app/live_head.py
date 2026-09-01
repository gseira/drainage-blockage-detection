"""
live_head.py
------------
True live learning: the final classification layer (a single nn.Linear over
the frozen ResNet-50's 2048-dim feature vector) is retrained every time a
human correction (Mark Blocked/Clear) comes in, and hot-swapped into the
running server immediately — no restart, no separate training script, no
scheduled job, no curated or balanced dataset required.

VERSION 2 — after version 1 caused real damage: retraining unconstrained on
just 24-28 corrections drove train_acc to 1.000 every time (see
logs/live_head_updates.log from that run), i.e. it perfectly memorized a
handful of points at the direct cost of the decision boundary that worked
for every other camera and case. Two changes directly target that failure
mode, both automatic, neither requiring a single minute of manual labeling:

  1. ANCHOR REGULARIZATION. The fit is no longer "minimize error on the
     corrections, period" — it's "minimize error on the corrections, but
     resist moving far from the ORIGINAL trained weights unless the
     evidence is strong." With few corrections, the anchor term dominates
     and the head barely moves. As corrections accumulate, the data term
     (summed, not averaged, across examples — see retrain_head) naturally
     gets proportionally more say relative to the fixed anchor strength.
     This alone would have stopped 24 examples from reaching train_acc=1.0.

  2. AUTOMATIC HISTORICAL CANARY CHECK. Before any retrained head is
     allowed to go live, it's checked against a random sample of the
     model's OWN past high-confidence predictions (p_blocked near 0 or
     near 1 — cases the current model was already clearly sure about).
     If the candidate head would flip too many of those, it's rejected and
     the current head stays in place. This uses the model's own scan
     history as a zero-effort proxy for "don't break what already worked" —
     see evaluate_against_canary() and app/main.py's _sample_canary_embeddings().

This is still narrower than, and meaningfully different from, other things
tried before this and disabled after making real production behaviour
worse: models/finetune_problem_cameras.py's v2-v5 fine-tuned the WHOLE
network; app/correction_memory.py's embedding-similarity override and the
scan-consistency override never touched weights at all, just compared
embeddings at inference time, and still had to be turned off. The backbone
here still never changes — only ~2049 numbers ever update, and now with a
regularizer plus a rejection gate instead of neither.

What this STILL does not protect against: the canary check uses the
model's own past confident calls as pseudo-ground-truth, not verified human
labels — if the model was already confidently wrong about some category of
case, this check would "protect" that wrongness too, same as it would for
anything genuinely correct. It is a real, load-bearing safety net against
the specific failure that already happened, not a guarantee of correctness
in an absolute sense. Every retrain (accepted or rejected) is logged, and
every accepted head is kept on disk, timestamped, so nothing is silently
lost and any state can be inspected or rolled back by hand.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
LIVE_HEAD_PATH = ROOT / "results" / "checkpoints" / "live_head.pt"
LIVE_HEAD_LOG  = ROOT / "logs" / "live_head_updates.log"

# Need at least one BLOCKED and one LOW correction total (globally, across
# every camera) before there's anything meaningful to fit — a head trained
# on only one class would just learn to always predict that class.
MIN_PER_CLASS = 1
RETRAIN_STEPS = 200
RETRAIN_LR    = 1e-3

# How strongly the fit is pulled back toward the original weights. Higher =
# more conservative (needs more/stronger correction evidence to move the
# boundary at all). This is a starting point, not calibrated against real
# data — watch logs/live_head_updates.log's train_acc and the
# [live_head] canary agreement lines after this runs for a while, and raise
# this further if updates still look too aggressive relative to how little
# data produced them.
ANCHOR_LAMBDA = 8.0

# Historical-canary rejection gate (see evaluate_against_canary below). A
# candidate head must still agree with the CURRENT model's own past
# high-confidence calls at least this often, or it's rejected outright.
CANARY_MIN_AGREEMENT = 0.97


def retrain_head(
    original_weight: torch.Tensor,
    original_bias: torch.Tensor,
    corrections: list,
) -> Optional[dict]:
    """
    corrections: list of {"embedding": np.ndarray, "label": "BLOCKED"|"LOW"}
    — the global correction pool from app/correction_memory.py's
    get_all_corrections(), across every camera. The base model is already a
    single classifier shared across all 56 cameras, so a shared online head
    is consistent with how it works already, and pooling gives it more to
    learn from than any one camera's corrections alone.

    Fits a fresh nn.Linear starting from (original_weight, original_bias) —
    NOT from any previously-live-updated state — via full-batch gradient
    descent on ALL of `corrections`, regularized to stay close to the
    original weights (ANCHOR_LAMBDA) unless the data really pushes back.
    Returns {"weight": Tensor, "bias": Tensor, "train_acc": float, "n": int},
    or None if there isn't at least one example of each class yet.
    """
    # Matches the fc convention used everywhere else in this app (see
    # app/inference.py's _get_p_blocked): sigmoid(logit) = p_clear, so the
    # positive class (label=1) here is LOW/CLEAR, not BLOCKED.
    labels = [1 if c["label"] == "LOW" else 0 for c in corrections]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        return None

    X = torch.tensor(np.stack([c["embedding"] for c in corrections]), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    orig_w = original_weight.reshape(1, -1).clone()
    orig_b = original_bias.reshape(1).clone()

    fc = nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        fc.weight.copy_(orig_w)
        fc.bias.copy_(orig_b)

    optimizer = torch.optim.Adam(fc.parameters(), lr=RETRAIN_LR)
    # reduction="sum", not "mean": the data term's total magnitude grows
    # with how many corrections there are, while the anchor term's
    # magnitude is fixed. That makes the anchor naturally dominate when
    # data is scarce (protecting against exactly the 24-example overfit
    # that already happened) and naturally matter proportionally less as
    # real correction evidence accumulates — without needing to manually
    # retune ANCHOR_LAMBDA as more corrections come in over time.
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for _ in range(RETRAIN_STEPS):
        optimizer.zero_grad()
        data_loss = criterion(fc(X).squeeze(1), y)
        anchor_loss = ANCHOR_LAMBDA * (
            ((fc.weight - orig_w) ** 2).sum() + ((fc.bias - orig_b) ** 2).sum()
        )
        (data_loss + anchor_loss).backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(fc(X).squeeze(1)) >= 0.5).float()
        train_acc = float((preds == y).float().mean())

    return {
        "weight": fc.weight.detach().clone(),
        "bias": fc.bias.detach().clone(),
        "train_acc": train_acc,
        "n": len(corrections),
    }


def evaluate_against_canary(
    candidate_weight: torch.Tensor,
    candidate_bias: torch.Tensor,
    original_weight: torch.Tensor,
    original_bias: torch.Tensor,
    canary_embeddings: list,
) -> Optional[dict]:
    """
    canary_embeddings: list of np.ndarray — a random sample of frames the
    CURRENT (pre-update) model was already highly confident about (see
    app/main.py's _sample_canary_embeddings). No human labels involved: the
    check is "does the candidate still agree with what the model itself was
    already confidently doing", not "is the candidate correct" in any
    absolute sense — see this module's docstring for that limitation.

    Returns {"agreement": float, "n": int}, or None if no canary data was
    available (e.g. very early on, before enough historical scans exist) —
    in which case the caller has to decide whether to proceed without this
    check or wait.
    """
    if not canary_embeddings:
        return None

    X = torch.tensor(np.stack(canary_embeddings), dtype=torch.float32)
    with torch.no_grad():
        orig_pred = (torch.sigmoid(X @ original_weight.reshape(-1) + original_bias) >= 0.5).float()
        cand_pred = (torch.sigmoid(X @ candidate_weight.reshape(-1) + candidate_bias) >= 0.5).float()
        agreement = float((orig_pred == cand_pred).float().mean())

    return {"agreement": agreement, "n": len(canary_embeddings)}


def save_live_head(weight: torch.Tensor, bias: torch.Tensor, meta: dict) -> None:
    """Persists the new head so it survives a restart, and keeps a
    timestamped copy of every version — never silently overwritten/lost,
    same discipline as every checkpoint in this project."""
    LIVE_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fc_weight": weight, "fc_bias": bias, "meta": meta}
    torch.save(payload, str(LIVE_HEAD_PATH))
    backup = LIVE_HEAD_PATH.parent / f"live_head_{int(time.time())}.pt"
    torch.save(payload, str(backup))

    LIVE_HEAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_HEAD_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  n={meta.get('n')}  "
                f"train_acc={meta.get('train_acc'):.3f}\n")


def load_live_head() -> Optional[dict]:
    """
    Returns {"weight": Tensor, "bias": Tensor} if a previously learned live
    head exists on disk (so learning survives a restart), else None — in
    which case the base checkpoint's own original fc layer is used as-is.
    """
    if not LIVE_HEAD_PATH.exists():
        return None
    ck = torch.load(str(LIVE_HEAD_PATH), map_location="cpu", weights_only=False)
    return {"weight": ck["fc_weight"], "bias": ck["fc_bias"]}
