"""
llm_explainer.py
----------------
Uses Cerebras's gemma-4-31b vision model to generate a natural language
explanation of a Grad-CAM overlay image.

Cerebras receives:
    - The Grad-CAM overlay (original image blended with heatmap)
    - The model prediction and confidence

And returns a short plain-English explanation suitable for a field engineer.

No extra pip install needed — this talks to Cerebras's OpenAI-compatible
Chat Completions REST endpoint directly over stdlib urllib, the same way
test_cerebras.py (the standalone script this was validated against) does.

API key stored in .env:
    CEREBRAS_API_KEY=csk-your_key_here

Usage:
    from risk_assessment.llm_explainer import LLMExplainer
    explainer = LLMExplainer()
    explanation = explainer.explain(overlay_image, p_blocked=0.99, risk="HIGH")
    print(explanation)
"""

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

# Load API key from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

CEREBRAS_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"

# Cloudflare (which fronts Cerebras's API) blocks requests carrying Python's
# default urllib User-Agent as bot traffic (HTTP 403, "error code: 1010") —
# a normal browser-looking one gets through fine. Confirmed against the real
# API while debugging test_cerebras.py.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def describe_gradcam(heatmap: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute quantitative, verifiable descriptors directly from the raw
    Grad-CAM array (H, W), values in [0, 1] — the actual interpretability
    signal — rather than relying on a vision LLM's visual impression of the
    colour-blended overlay image.

    Prior to this function, `heatmap_coverage` was computed in
    app/inference.py (`float((heatmap > 0.5).mean())`) and stored against
    the result row, but was never passed to LLMExplainer.explain(), which
    saw only the composited overlay image. In practice this meant the LLM
    was free to invent a plausible-sounding coverage figure ("approximately
    50% obstructed") that had no connection to the number the system had
    actually computed. This function is the fix: it produces the same
    coverage figure plus a coarse location and concentration descriptor, all
    derived only from array arithmetic, so the prompt built in explain()
    can hand the LLM verified facts to phrase rather than values to guess.

    Args:
        heatmap:   raw Grad-CAM array, shape (H, W), values in [0, 1].
        threshold: activation cutoff used to define the "highlighted" region.

    Returns:
        dict with:
            coverage_pct:    float — % of the frame above `threshold`. Kept
                                      for internal verification (e.g.
                                      compare_prompts.py checking whether a
                                      prompt's own stated figure matches
                                      reality) but deliberately NOT surfaced
                                      to the LLM or the operator — a raw
                                      percentage reads as false precision in
                                      an inspection note. Use coverage_bucket
                                      for anything user- or prompt-facing.
            coverage_bucket: str   — qualitative size description derived
                                      from coverage_pct ("a small area",
                                      "a moderate area", "most of the frame", ...).
            location:        str   — coarse 3x3-grid position of the
                                      activation centroid ("centre",
                                      "upper left", "lower right", ...).
            spread:          str   — "concentrated in a single area" or
                                      "spread across multiple areas".
    """
    mask = heatmap > threshold
    coverage_pct = round(float(mask.mean()) * 100, 1)
    coverage_bucket = _coverage_bucket(coverage_pct)

    if not mask.any():
        return {
            "coverage_pct": coverage_pct, "coverage_bucket": coverage_bucket,
            "location": "no distinct region", "spread": "no meaningful activation",
        }

    h, w = heatmap.shape
    ys, xs = np.nonzero(mask)
    weights = heatmap[ys, xs]
    cy = float(np.average(ys, weights=weights)) / h   # 0 = top, 1 = bottom
    cx = float(np.average(xs, weights=weights)) / w   # 0 = left, 1 = right

    row = "upper" if cy < 0.34 else ("lower" if cy > 0.66 else "middle")
    col = "left" if cx < 0.34 else ("right" if cx > 0.66 else "centre")
    location = "centre" if (row, col) == ("middle", "centre") else f"{row} {col}"

    # Concentration: bounding-box area of the highlighted pixels vs the
    # actual highlighted pixel count. Near 1 => a single tight blob fills
    # its own bounding box; markedly higher => activation is scattered
    # across several disconnected regions within that box.
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    concentration_ratio = bbox_area / max(int(mask.sum()), 1)
    spread = "concentrated in a single area" if concentration_ratio < 2.5 else "spread across multiple areas"

    return {
        "coverage_pct": coverage_pct, "coverage_bucket": coverage_bucket,
        "location": location, "spread": spread,
    }


def _coverage_bucket(coverage_pct: float) -> str:
    """Map a raw coverage percentage to a qualitative size phrase — grounded
    in the same computed number, but expressed the way a person would
    describe it rather than as a percentage."""
    if coverage_pct < 5:
        return "a small area"
    if coverage_pct < 20:
        return "a limited area"
    if coverage_pct < 45:
        return "a moderate area"
    if coverage_pct < 70:
        return "a large area"
    return "most of the frame"


class LLMExplainer:
    """
    Wraps Cerebras's gemma-4-31b vision model to explain Grad-CAM overlay
    images. test_cerebras.py is the standalone script this was validated
    against — same endpoint, same headers, same model.

    Args:
        api_key:    Cerebras API key. If None, reads from CEREBRAS_API_KEY
                    env var.
        model_name: Cerebras vision model (default: gemma-4-31b).
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemma-4-31b",
    ):
        key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not key:
            raise ValueError(
                "Cerebras API key not found. Set CEREBRAS_API_KEY in your .env file."
            )
        self.api_key     = key
        self.model_name  = model_name
        self._scene_cache: dict[str, bool] = {}  # md5(image bytes) → is_drainage

    def _chat(self, image_b64: str, prompt_text: str, max_tokens: int, max_retries: int = 4) -> str:
        """
        Single Cerebras Chat Completions call (image + text → text), with
        retry/backoff on rate-limit and transient server errors. Raises on
        anything else (e.g. 402 payment-required — that's a real billing
        problem the caller should see, not silently swallow).
        """
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            CEREBRAS_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # See CEREBRAS_ENDPOINT note above — without this Cloudflare
                # blocks the request as bot traffic before it reaches Cerebras.
                "User-Agent": _BROWSER_USER_AGENT,
            },
            method="POST",
        )
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if attempt < max_retries - 1 and e.code in (429, 500, 502, 503):
                    wait = 2 ** (attempt + 2)   # 4s, 8s, 16s, 32s
                    print(f"  [Cerebras] HTTP {e.code} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise

    def validate_scene(self, image: Image.Image) -> bool:
        """
        Quick scene-domain check: does this image show drainage infrastructure?

        Sends the raw image to the LLM with a single YES/NO prompt.
        Returns True if it looks like a drainage CCTV image (channel, screen,
        pipe, grate, culvert, flood gate, etc.), False otherwise.

        Fails open — returns True on any API/network error so that a transient
        Cerebras outage never blocks legitimate images.

        Args:
            image: PIL Image (the original, unprocessed upload).

        Returns:
            bool — True = looks like drainage, False = not drainage.
        """
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        image_b64   = base64.b64encode(image_bytes).decode("utf-8")

        # Return cached result immediately — same image always gets same answer,
        # no extra LLM call, no rate-limit inconsistency on re-upload.
        img_hash = hashlib.md5(image_bytes).hexdigest()
        if img_hash in self._scene_cache:
            cached = self._scene_cache[img_hash]
            print(f"  [scene_validator] Cache hit ({img_hash[:8]}…) → {cached}")
            return cached, True   # cached result counts as a successful validation

        prompt_text = (
            "Is this image a close-range CCTV monitoring view of a drainage "
            "structure — specifically a drainage channel, screen, grate, "
            "culvert, pipe, weir, or flood gate?\n\n"
            "Answer NO if the image shows: a bridge, road, building, "
            "landscape, vehicle, person, open river from a distance, or any "
            "scene that is NOT the interior or immediate vicinity of a "
            "drainage structure viewed by a fixed monitoring camera.\n\n"
            "Answer with exactly one word: YES or NO."
        )

        try:
            answer = self._chat(image_b64, prompt_text, max_tokens=5).upper()
            result = answer.startswith("YES")
            self._scene_cache[img_hash] = result
            return result, True   # (is_drainage, llm_ran_successfully)
        except Exception as exc:
            print(f"  [scene_validator] API error — failing open: {exc}")
            return True, False   # (pass through, but mark as unvalidated)

    def explain(
        self,
        overlay_image: Image.Image,
        p_blocked: float,
        risk: str,
        heatmap: np.ndarray = None,
        version: str = "v2",
    ) -> str:
        """
        Generate a plain-English explanation of a Grad-CAM overlay.

        Args:
            overlay_image: PIL Image — Grad-CAM heatmap blended onto original.
            p_blocked:     Model's blockage probability (0-1).
            risk:          "HIGH"/"BLOCKED" for a blockage, "LOW" for clear.
                           (app/inference.py's 3-class pipeline passes "BLOCKED";
                           the older HIGH/LOW binary pipeline passes "HIGH" — both
                           must map to the same "blocked" prompt branch below.)
            heatmap:       raw Grad-CAM array (H, W) in [0, 1], if available.
                           Required for version="v2" to be grounded; if omitted
                           under v2 the prompt falls back to the ungrounded
                           wording with a note that no localisation data was
                           available.
            version:       "v1" — original prompt, kept unchanged so it can be
                                   run side-by-side with v2 for comparison.
                           "v2" — new, shorter, grounded prompt (default):
                                   uses describe_gradcam() output as verified
                                   facts instead of letting the LLM guess a
                                   location/coverage figure from the image.

        Returns:
            Short explanation string.
        """
        is_blocked = risk in ("HIGH", "BLOCKED")
        prediction = "BLOCKED" if is_blocked else "CLEAR"
        confidence = (
            int(round(p_blocked * 100)) if is_blocked
            else int(round((1 - p_blocked) * 100))
        )

        if version == "v1":
            # ── Original prompt (unchanged) — kept only for comparison ────────
            if is_blocked:
                prompt = (
                    f"You are a drainage inspection assistant helping field engineers make fast, "
                    f"informed decisions from CCTV footage.\n\n"
                    f"STATUS: BLOCKED | RISK: HIGH\n\n"
                    f"The image shows a CCTV drainage frame. The highlighted region (red/yellow) "
                    f"marks the area of concern. Focus entirely on what is physically visible — "
                    f"do NOT mention AI, models, confidence scores, or heatmaps.\n\n"
                    f"Write a concise inspection report with exactly these three sections:\n\n"
                    f"OBSERVATION: 1-2 sentences. Describe what is physically visible at the "
                    f"highlighted area — debris accumulation, sediment, occlusion of grate bars, "
                    f"standing water. Be specific about location (e.g. lower-right inlet, centre of grate).\n\n"
                    f"RISK ASSESSMENT: One sentence. State HIGH risk and explain why the physical "
                    f"condition justifies immediate attention. No mention of scores or models.\n\n"
                    f"RECOMMENDED ACTION: 1-2 direct sentences. Tell the engineer exactly what to "
                    f"do and which part of the structure to physically inspect or clear.\n\n"
                    f"Use precise drainage engineering terminology. Be direct and practical."
                )
            else:
                prompt = (
                    f"You are a drainage inspection assistant helping field engineers make fast, "
                    f"informed decisions from CCTV footage.\n\n"
                    f"STATUS: CLEAR | RISK: LOW\n\n"
                    f"The image shows a CCTV drainage frame assessed as clear. "
                    f"Focus on what is physically visible — do NOT mention AI, models, scores, or heatmaps.\n\n"
                    f"Write a very brief inspection report with exactly these two sections:\n\n"
                    f"OBSERVATION: One sentence only. Confirm the drainage structure appears clear "
                    f"— note visible flow path, unobstructed grate bars, or absence of debris.\n\n"
                    f"RECOMMENDED ACTION: One sentence. Advise routine monitoring — "
                    f"do NOT suggest specific time intervals or frequencies.\n\n"
                    f"Be concise. No risk assessment section needed."
                )
        else:
            # ── v2: shorter, grounded in the actual Grad-CAM signal ───────────
            # Risk/severity language deliberately dropped: BLOCKED/CLEAR already
            # conveys that (see Chapter 4.5.1 — the earlier severity-tier layer
            # was removed as duplicative). This prompt's only job is to explain
            # the WHERE and describe what's there — the interpretability step
            # that Grad-CAM alone (a heatmap image with no caption) doesn't
            # itself provide, and that existing drainage-monitoring literature
            # (Chapter 2.4) does not address.
            if is_blocked:
                if heatmap is not None:
                    stats = describe_gradcam(heatmap)
                    grounding = (
                        f"Grad-CAM localisation (computed, not your estimate): the model's attention "
                        f"is centred on the {stats['location']} of the frame, over {stats['coverage_bucket']}, "
                        f"{stats['spread']}."
                    )
                else:
                    grounding = "Grad-CAM localisation data was not available for this frame."
                prompt = (
                    f"You are a drainage inspection assistant. This CCTV frame was classified BLOCKED.\n\n"
                    f"{grounding}\n\n"
                    f"Using that location and what is visible there, write exactly two sentences: "
                    f"(1) what is physically obstructing the screen at that location; "
                    f"(2) what the engineer should do about it. "
                    f"Use the location given above exactly as stated — do not restate it as a "
                    f"different part of the frame. Do NOT include any percentage or numeric estimate "
                    f"of coverage anywhere in your answer — describe extent only in plain words "
                    f"(e.g. \"a small patch\", \"much of the screen\"), never as a number. "
                    f"No headers, no bullet points, no mention of AI or models."
                )
            else:
                if heatmap is not None:
                    stats = describe_gradcam(heatmap)
                    grounding = f"Grad-CAM shows negligible activation — no meaningful region of concern."
                else:
                    grounding = "Grad-CAM activation was negligible."
                prompt = (
                    f"You are a drainage inspection assistant. This CCTV frame was classified CLEAR. "
                    f"{grounding}\n\n"
                    f"Write exactly one sentence confirming the screen is clear and flow is unobstructed. "
                    f"No headers, no risk language, no mention of AI or models."
                )

        # Encode image as base64 JPEG
        buf = io.BytesIO()
        overlay_image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 150 is a deliberate hard cap, not a target — the v2 prompt asks for
        # 1-2 plain sentences, so this is mainly a backstop against the model
        # rambling past that. Retry/backoff on rate-limit or transient server
        # errors is handled inside _chat().
        text = self._chat(image_b64, prompt, max_tokens=150)
        # Strip markdown bold markers (**TEXT:**)
        import re
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        return text
