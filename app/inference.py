"""
inference.py
------------
Single entry point for the FastAPI backend.

Supports both:
  - Binary ResNet-50 checkpoint  (best_model.pt,    fc outputs 1 logit)
  - 3-class ResNet-50 checkpoint (best_model_3class.pt, fc outputs 3 logits)

The checkpoint type is detected automatically from the saved state dict.

3-class output classes:
    0 = BLOCKED  →  drain blocked, dispatch required
    1 = CLEAR    →  drain clear, no action
    2 = OTHER    →  drain with ambiguous/unknown status (still a drain — flag for manual review)

New features vs binary inference.py
-------------------------------------
1. OTHER class detection — images that are not drainage scenes are classified
   as OTHER (prediction="OTHER") and skip the blockage pipeline entirely.

2. GradCAM only for BLOCKED — heatmap computation is skipped for CLEAR and
   OTHER predictions, which makes those paths ~2× faster.

This is intentionally the "raw model" ablation: no per-camera threshold
overrides, no pixel-quality gating/ensemble, no GradCAM centroid penalty,
and no severity tiering — just the trained classifier's own probability,
GradCAM for explainability, and the LLM report on top.

One addition on top of the raw model: a confidence-margin check (see the
`margin` constructor arg, default 0.20). Predictions landing within `margin`
of the decision boundary are routed to FLAGGED instead of being confidently
asserted as BLOCKED/CLEAR — this targets the specific failure mode where a
borderline camera flips label between near-identical scans purely from
ordinary capture noise (auto-exposure, JPEG re-encoding, minor lighting
shifts). It's applied identically to every image (not keyed to any specific
camera), so it isn't a re-introduction of the per-camera override system.
Set `margin=0.0` to disable it and get the literal raw-argmax baseline back.

Usage:
    from app.inference import InferencePipeline
    pipeline = InferencePipeline("results/checkpoints/best_model_3class.pt")
    result   = pipeline.run(pil_image)
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# ── Determinism ───────────────────────────────────────────────────────────────
#
# Without this, the SAME image can classify differently between two separate
# calls. In eval() mode there's no dropout/batchnorm randomness, but on GPU
# cuDNN is still free to pick convolution algorithms whose internal reduction
# order isn't guaranteed bit-identical run to run — the output can shift by a
# fraction of a percent with no change to the input or the weights. That's
# invisible for a confidently BLOCKED or CLEAR image, but for a camera sitting
# right on the decision boundary (no p_blocked/p_clear margin — see the
# per-camera threshold discussion), that fraction of a percent is enough to
# flip the label. This forces bit-identical results for identical input.
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only=True: log (not crash) if some op has no deterministic GPU kernel,
# rather than take down a live inference server over it.
torch.use_deterministic_algorithms(True, warn_only=True)

# Allow imports from sibling packages
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "interpretability"))
sys.path.insert(0, str(ROOT / "risk_assessment"))
sys.path.insert(0, str(Path(__file__).parent))  # this directory — for live_head below

from gradcam import GradCAM, overlay_heatmap
from llm_explainer import LLMExplainer
from live_head import load_live_head


# ── Constants ─────────────────────────────────────────────────────────────────

# 3-class label mapping (also used for binary compat)
IDX_TO_NAME = {0: "BLOCKED", 1: "CLEAR", 2: "OTHER"}


# ── Image pre-processing ──────────────────────────────────────────────────────

class _LetterboxResize:
    """
    Resize to fit WITHIN (size, size) preserving aspect ratio (longer edge
    -> size), then pad the shorter edge with mid-grey to make it square.
    Keeps 100% of the original frame — nothing gets cropped out.

    Must exactly match models/finetune_problem_cameras.py's LetterboxResize
    — train and inference preprocessing have to agree, or you get the same
    kind of mismatch preserve_aspect exists to avoid, just in reverse.
    """

    def __init__(self, size: int, fill=(114, 114, 114)):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = ((self.size - new_w) // 2, (self.size - new_h) // 2)
        canvas.paste(resized, offset)
        return canvas


def _build_transform(img_size: int = 224, preserve_aspect: bool = False) -> transforms.Compose:
    """
    preserve_aspect=False (default): Resize((size, size)) — squashes every
    image to a square regardless of its native aspect ratio. This is what
    best_model.pt and the original best_model_ft.pt (Wadebridge/Chaddlewood)
    were trained on, so inference MUST keep using this for those checkpoints
    — switching it globally would feed them differently-shaped input than
    what they learned on, which would make things worse, not better.

    preserve_aspect=True: letterbox padding (see _LetterboxResize) — scales
    the image to fit within a square, keeping proportions, then pads the
    short edge with grey. Real aspect ratios across these cameras range from
    1.22 to 1.78 (see risk_assessment/audit_finetune_candidates.py).

    NOTE: an earlier version of this used CenterCrop instead of padding —
    that trained fine (val F1_blocked=0.993) but failed live validation: on
    a 1920x1080 (1.78 ratio) camera, cropping to square discarded ~44% of
    the frame's width and appears to have cropped the actual blockage out of
    view, causing a confident false CLEAR on a site confirmed to be
    persistently, severely blocked. Letterbox padding replaced it because it
    never loses real content, only adds non-informative border pixels.

    Only use preserve_aspect=True for a checkpoint that was actually TRAINED
    with this exact matching preprocessing (see
    models/finetune_problem_cameras.py's LetterboxResize) — train/inference
    preprocessing must match.
    """
    if preserve_aspect:
        return transforms.Compose([
            _LetterboxResize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device):
    """
    Load a checkpoint and return (model, num_classes, threshold, gradcam_layer).

    Supports three checkpoint types, detected automatically:
      - EfficientNet-B4 binary  (architecture="efficientnet_b4" key in ckpt)
      - ResNet-50 binary        (fc.weight shape [1, 2048])
      - ResNet-50 3-class       (fc.weight shape [3, 2048])
    """
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    architecture = ckpt.get("architecture", None)
    threshold    = float(ckpt.get("threshold", 0.5))

    if architecture == "efficientnet_b4":
        # ── EfficientNet-B4 binary ────────────────────────────────────────────
        print(f"  Checkpoint: EfficientNet-B4 binary  (threshold={threshold:.2f})")
        model = models.efficientnet_b4(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4, inplace=True),
            torch.nn.Linear(in_feat, 1),
        )
        model.load_state_dict(state)
        model.eval()
        return model.to(device), 1, threshold, "features.8"

    else:
        # ── ResNet-50 (binary or 3-class) ─────────────────────────────────────
        fc_weight = state.get("fc.weight")
        if fc_weight is None:
            raise ValueError(f"Cannot find 'fc.weight' or 'architecture' key in: {checkpoint_path}")
        num_classes = fc_weight.shape[0]
        print(f"  Checkpoint: ResNet-50  num_classes={num_classes}  threshold={threshold:.2f}")
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state)
        model.eval()
        return model.to(device), num_classes, threshold, "layer4"


def _tta_views(image: Image.Image) -> list:
    """
    A small, FIXED (not random) set of realistic capture-to-capture
    perturbations layered on top of the original image.

    Motivation (see risk_assessment/test_determinism.py and the DB analysis
    that preceded it): the model itself is provably deterministic — the same
    exact image tensor always gives the same exact output. But across ~25 of
    56 monitored cameras, two REAL photos of the same physical scene, taken
    moments apart, can swing between p_blocked ~0.08 and ~0.99 purely from
    ordinary auto-exposure/JPEG-recompression/lighting noise the model was
    never taught to ignore. Averaging the model's probability over a handful
    of plausible "what this same scene might have looked like a moment
    earlier or later" variants smooths out exactly that axis of brittleness,
    without touching a single model weight.

    Deliberately fixed rather than random (no random.choice/np.random) so the
    pipeline's determinism guarantee still holds: identical input image ->
    identical set of TTA views -> identical averaged output, every time.
    """
    from PIL import ImageEnhance

    views = [image]   # the unperturbed original is always included
    for b in (0.92, 1.08):
        views.append(ImageEnhance.Brightness(image).enhance(b))
    for c in (0.95, 1.05):
        views.append(ImageEnhance.Contrast(image).enhance(c))
    for s in (0.90, 1.10):
        views.append(ImageEnhance.Color(image).enhance(s))
    return views


def _spatial_crops(image: Image.Image) -> list:
    """
    Five overlapping 75% crops covering the full frame.
    Used to catch blockages that appear off-centre (upstream debris pile, etc.)
    Only run for uncertain primary predictions: p ∈ [0.10, 0.72].
    """
    w, h   = image.size
    qw, qh = w // 4, h // 4
    return [
        image.crop((0,      0,      3*qw,      3*qh)),
        image.crop((qw,     0,      w,         3*qh)),
        image.crop((0,      qh,     3*qw,      h)),
        image.crop((qw,     qh,     w,         h)),
        image.crop((qw//2,  qh//2,  w-qw//2,   h-qh//2)),
    ]


# ── p_blocked extraction for both model types ─────────────────────────────────

def _get_p_blocked(model, tensor: torch.Tensor, num_classes: int) -> tuple:
    """
    Run a single forward pass and return (p_blocked, p_clear, p_other).

    For binary model:  p_blocked = 1 - sigmoid(logit),  p_clear = sigmoid(logit), p_other = 0.
    For 3-class model: softmax → [p_blocked, p_clear, p_other].
    """
    with torch.no_grad():
        logits = model(tensor)

    if num_classes == 1:
        p_clear   = float(torch.sigmoid(logits).item())
        p_blocked = 1.0 - p_clear
        p_other   = 0.0
    else:
        probs     = torch.softmax(logits, dim=1)[0]
        p_blocked = float(probs[0].item())
        p_clear   = float(probs[1].item())
        p_other   = float(probs[2].item())

    return p_blocked, p_clear, p_other


# ── Pipeline ──────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Loads the model once at startup and exposes a single .run() method.

    Auto-detects binary vs 3-class from the checkpoint.

    Args:
        checkpoint_path: Path to best_model.pt or best_model_3class.pt.
        device:          'cuda', 'cpu', or None (auto-detect).
        img_size:        Input resolution the model was trained on (default 224).
        clear_below:     p_blocked at or below this is confidently CLEAR.
        blocked_above:   p_blocked at or above this is confidently BLOCKED.
                          Anything strictly between clear_below and
                          blocked_above is routed to FLAGGED instead of a
                          confident call. These replaced a single symmetric
                          `margin` (threshold +/- margin) on 2026-07-07 at
                          the user's request for an ASYMMETRIC zone —
                          originally 0.25-0.85, requiring more confidence
                          before trusting a BLOCKED call than a CLEAR one.
                          Motivated directly by the "too many confidently
                          wrong BLOCKED calls" issue found with the
                          (rolled-back) specialist model: raising the bar
                          for BLOCKED specifically, without also raising it
                          for CLEAR, targets that failure mode instead of
                          just shrinking or growing both sides equally the
                          way the old symmetric `margin` could.
                          Narrowed to 0.40-0.70 on 2026-07-09 — a smaller
                          FLAGGED zone (width 0.30 vs the original 0.60)
                          means more results fall outside it and get
                          auto-classified confidently as CLEAR or BLOCKED,
                          leaving fewer landing in Needs Review.
                          Raised back to 0.40-0.85 on 2026-07-22 after real
                          production BLOCKED calls at p_blocked 0.81-0.997
                          turned out to be wrong on several cameras (Gorran
                          Haven, Ashburton Lower, Barnstaple Portmarsh Lane,
                          Teignmouth First Ave) — 0.70 wasn't a high enough
                          bar for a confident BLOCKED call in practice, so
                          this moves back toward the original 0.25-0.85
                          asymmetric design that specifically required more
                          confidence for BLOCKED than for CLEAR. Raised
                          again to 0.90 the same day — 0.85 still wasn't
                          quite high enough a bar in practice. clear_below
                          raised to 0.50 the same day too, at the user's
                          explicit request that Needs Review come ONLY from
                          this confidence-margin rule (0.50-0.90) — the
                          per-camera "never trust a CLEAR/BLOCKED for this
                          specific site" overrides in app/main.py
                          (_apply_never_auto_clear_override /
                          _apply_never_auto_block_override) were removed
                          the same day for the same reason.
                          Narrowed again to 0.50-0.85 on 2026-07-31, at the
                          user's explicit request to reduce how many results
                          land in Needs Review — chosen as the smaller-risk
                          direction (only lowering blocked_above, not also
                          raising clear_below) since wrong CLEAR calls were
                          never the failure mode that drove this value up in
                          the first place. Worth knowing: 0.85 is exactly the
                          bar that was found insufficient on 2026-07-22 (see
                          above) before being raised to 0.90 the same day —
                          if the same wrong-confident-BLOCKED pattern shows up
                          again in production at this setting, that's why.
                          Old symmetric-margin history, for reference: 0.10
                          -> 0.20 -> 0.35 -> 0.45 -> 0.35 -> 0.30 -> 0.20.
        use_tta:         If True, average the model's probability over the
                          original image plus a small fixed set of
                          brightness/contrast/saturation-perturbed variants
                          of it (see _tta_views), instead of trusting one
                          single forward pass. Tried after test_determinism.py
                          proved the model itself is deterministic but very
                          sensitive to ordinary capture-to-capture noise.
                          Defaults to False: test_live_pair_stability.py
                          (real live capture pairs, 56 cameras, 2026-07-07)
                          showed it does NOT meaningfully help — flip rate
                          was slightly worse (1/56 -> 2/56), average swing
                          essentially unchanged (0.0216 -> 0.0213), and the
                          cameras with the largest real swings (Portreath,
                          Penzance Coombe Cottage) were barely affected at
                          all, for 7x the inference cost. Kept as an option
                          rather than deleted in case a future, more
                          aggressive perturbation set is worth re-testing.
        use_llm:         If True (default), call the Groq LLM for a written
                          explanation on BLOCKED predictions. Set False to
                          skip that entirely and use a placeholder explanation
                          instead — for batch evaluation/testing scripts that
                          only care about the prediction/probability and
                          would otherwise burn through the Groq API quota
                          (and slow down a few-hundred-image test run) for
                          text nothing is reading.
        preserve_aspect: If False (default), squash every image to a square
                          — matches how best_model.pt and the original
                          best_model_ft.pt were trained, so this MUST stay
                          False for those checkpoints. Set True only for a
                          checkpoint trained with matching aspect-preserving
                          preprocessing (see models/finetune_problem_cameras.py,
                          updated 2026-07-07) — real camera aspect ratios
                          range from 1.22 to 1.78, so squashing distorts
                          different cameras' images by different amounts.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        img_size: int = 224,
        clear_below: float = 0.50,
        blocked_above: float = 0.85,
        use_tta: bool = False,
        use_llm: bool = True,
        preserve_aspect: bool = False,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = _build_transform(img_size, preserve_aspect=preserve_aspect)
        # Asymmetric confidence zone — see the docstring above for why this
        # replaced a single symmetric margin. p_blocked <= clear_below is a
        # confident CLEAR, p_blocked >= blocked_above is a confident BLOCKED,
        # anything in between is too close to call and gets FLAGGED instead.
        self.clear_below   = clear_below
        self.blocked_above = blocked_above
        self.use_tta = use_tta
        self.use_llm = use_llm

        print(f"Loading model from: {checkpoint_path}")
        self.model, self.num_classes, self.threshold, gradcam_layer = \
            _load_model(checkpoint_path, self.device)

        # GradCAM — hooks registered once, reused across calls
        self.gradcam = GradCAM(self.model, target_layer=gradcam_layer)

        # LLM explainer (only constructed if actually needed)
        self.explainer = LLMExplainer() if use_llm else None

        # ── Live head learning (see app/live_head.py) ─────────────────────
        # Only meaningful for the binary model (single-logit fc) — the
        # p_blocked/p_clear convention this trains against doesn't apply to
        # the 3-class checkpoint.
        if self.num_classes == 1:
            # The base checkpoint's own fc weights — the FIXED starting
            # point every future live-head retrain begins from, regardless
            # of how many online updates have happened since. Captured
            # before any live head is applied below.
            self._original_fc_weight = self.model.fc.weight.detach().clone()
            self._original_fc_bias   = self.model.fc.bias.detach().clone()

            # If a live head was learned in a previous run of this server,
            # apply it now so that learning survives a restart instead of
            # silently resetting to the base checkpoint every time.
            try:
                saved = load_live_head()
                if saved is not None:
                    self.refresh_live_head(saved["weight"], saved["bias"])
                    print("  Loaded a previously learned live head from disk "
                          "(results/checkpoints/live_head.pt).")
            except Exception as exc:
                print(f"  [live_head] could not load saved live head ({exc}) — "
                      f"using the base checkpoint's own head.")
        else:
            self._original_fc_weight = None
            self._original_fc_bias   = None

        print(f"  Model ready on {self.device}  "
              f"({'3-class' if self.num_classes == 3 else 'binary'})")

    def refresh_live_head(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Hot-swaps the model's fc layer weights in place — takes effect on
        the very next inference call, no restart needed. See
        app/live_head.py for how these weights are produced (retrained from
        the ORIGINAL checkpoint's fc layer on every human correction so
        far, every time a new one comes in).
        """
        with torch.no_grad():
            self.model.fc.weight.copy_(weight.to(self.device))
            self.model.fc.bias.copy_(bias.to(self.device))

    # ── Embedding extraction (for correction memory, no weight updates) ───────

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Frozen ResNet-50 avgpool feature vector (2048-dim) for `image` — a
        single forward pass through the already-trained backbone, no
        gradients, no weight changes. Used by app/correction_memory.py to
        compare a new scan against past human corrections for the same
        camera (see that module's docstring for why this exists instead of
        fine-tuning the network on every click).
        """
        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)

        feats = {}

        def _hook(module, inp, out):
            feats["emb"] = out

        handle = self.model.avgpool.register_forward_hook(_hook)
        with torch.no_grad():
            self.model(tensor)
        handle.remove()

        return feats["emb"].flatten(1).cpu().numpy()[0]

    # ── Main inference method ─────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        validate_scene: bool = False,
        allow_flagged: bool = True,
    ) -> dict:
        """
        Run the full inference pipeline on one PIL image.

        This is the "raw model" path: no per-camera threshold override, no
        pixel-quality gate, and no GradCAM centroid correction. The one
        addition on top of a single raw forward pass is test-time averaging
        (see _tta_views / self.use_tta) — smoothing over ordinary
        capture-to-capture noise, not a per-camera or per-image heuristic.
        GradCAM only runs for BLOCKED predictions (CLEAR returns immediately),
        and always targets the single unperturbed original image.

        Args:
            image:          PIL Image (any size / mode).
            validate_scene: If True, calls the LLM scene validator before
                            inference and rejects non-drainage images.
                            Set False for live EA webcam images.
            allow_flagged:  If False, skips the confidence-margin check
                            entirely and always commits to BLOCKED or CLEAR.
                            Set False for standalone uploads with no campath/
                            DB row: a FLAGGED result there has no correction
                            path (Mark Blocked/Clear needs a campath), so it
                            would be a dead end for the operator. Camera-
                            sourced calls (scheduled sweep, on-demand webcam
                            check) keep the default True, since those results
                            are persisted and can be corrected normally.

        Returns a dict with (at minimum):
            prediction      str   — "BLOCKED", "CLEAR", or "OTHER"
            risk            str   — "BLOCKED" or "LOW"
            p_blocked       float — probability of blockage
            p_other         float — (3-class only) probability of non-drainage
            overlay_b64     str   — base64 JPEG of Grad-CAM overlay (BLOCKED only)
            heatmap_b64     str   — base64 JPEG of raw heatmap (BLOCKED only)
            heatmap_coverage float — fraction of heatmap > 0.5 (BLOCKED only)
            explanation     str   — LLM inspection report
        """
        original = image.convert("RGB")

        # ── Step 1: primary model inference ───────────────────────────────────
        tensor = self.transform(original).unsqueeze(0).to(self.device)

        # Always compute the single-pass, unperturbed-original probability —
        # used for GradCAM's target tensor either way, and kept around as
        # p_blocked_single for transparency/debugging regardless of use_tta.
        p_blocked_single, p_clear_single, p_other_single = _get_p_blocked(
            self.model, tensor, self.num_classes
        )

        if self.use_tta:
            views = _tta_views(original)
            per_view = [
                _get_p_blocked(self.model, self.transform(v).unsqueeze(0).to(self.device), self.num_classes)
                for v in views
            ]
            p_blocked_orig = float(np.mean([p[0] for p in per_view]))
            p_clear_orig   = float(np.mean([p[1] for p in per_view]))
            p_other_orig   = float(np.mean([p[2] for p in per_view]))
            print(f"  [inference/tta] {len(views)} views  "
                  f"single-pass p_blocked={p_blocked_single:.3f}  "
                  f"tta-avg p_blocked={p_blocked_orig:.3f}  "
                  f"(spread across views: {max(p[0] for p in per_view) - min(p[0] for p in per_view):.3f})")
        else:
            p_blocked_orig, p_clear_orig, p_other_orig = p_blocked_single, p_clear_single, p_other_single
            print(f"  [inference] p_blocked={p_blocked_orig:.3f}  p_clear={p_clear_orig:.3f}  "
                  f"p_other={p_other_orig:.3f}")

        # ── Step 1a: confidence-margin check ──────────────────────────────────
        #
        # A prediction sitting right on the decision boundary is close to a
        # coin flip — normal frame-to-frame noise (auto-exposure, JPEG
        # re-encoding, tiny lighting/colour shifts) is enough to push it back
        # and forth across the line, which is what makes a borderline camera
        # look like it's flip-flopping between scans. Rather than confidently
        # asserting BLOCKED or CLEAR that close to the line, be honest that
        # it's a coin flip and route it to FLAGGED for manual review instead.
        #
        # This is NOT a per-camera override (nothing here is keyed on which
        # camera this is) — it's a general uncertainty check applied equally
        # to every image, so it doesn't reintroduce the per-camera threshold
        # hack that was removed earlier.
        #
        # Asymmetric zone: confidently CLEAR at or below clear_below,
        # confidently BLOCKED at or above blocked_above, FLAGGED in between.
        # Applied directly to p_blocked_orig for both binary and 3-class
        # checkpoints — simpler and more predictable than the old formula,
        # which compared against different quantities (threshold for binary,
        # p_clear for 3-class) and could only ever be symmetric.
        #
        # allow_flagged=False skips this block entirely, so callers with no
        # correction path (standalone uploads) always get a definitive
        # BLOCKED/CLEAR call from Step 2 below instead of an unresolvable
        # FLAGGED result — see the allow_flagged docstring above.
        if allow_flagged and self.clear_below < p_blocked_orig < self.blocked_above:
            dist = min(p_blocked_orig - self.clear_below, self.blocked_above - p_blocked_orig)
            print(f"  [margin_check] p_blocked={p_blocked_orig:.3f} is between "
                  f"clear_below={self.clear_below:.2f} and blocked_above={self.blocked_above:.2f} "
                  f"— routing to FLAGGED")
            return self._flagged_result(original, p_blocked_orig, p_clear_orig, p_other_orig, dist)

        # ── Step 2: classify via argmax — trust the model ─────────────────────
        #
        # The model was trained and evaluated using argmax (macro-F1 = 0.9189).
        # No threshold tuning needed: whatever class has the highest softmax
        # probability is the prediction.
        #
        #   0 = BLOCKED  → run GradCAM
        #   1 = CLEAR    → return immediately, no GradCAM
        #   2 = OTHER    → uncertain drain status (not currently gated separately)

        # Binary decision using the calibrated threshold saved in the checkpoint.
        # For EfficientNet-B4: threshold was found on val set (typically 0.40-0.55).
        # For ResNet-50 binary: defaults to 0.5 if no threshold was saved.
        # For ResNet-50 3-class: p_blocked vs p_clear comparison (threshold unused).
        if self.num_classes == 1:
            # Binary model — use calibrated threshold on raw sigmoid probability
            blocked = p_blocked_orig >= self.threshold
        else:
            # 3-class model — argmax between BLOCKED and CLEAR (ignore OTHER)
            blocked = p_blocked_orig >= p_clear_orig

        if not blocked:
            return self._clear_result(original, p_blocked_orig, p_clear_orig, p_other_orig)

        # BLOCKED — fall through to GradCAM
        p_best = p_blocked_orig

        # ── Step 3: GradCAM (BLOCKED predictions only) ────────────────────────
        #
        # For the 3-class model, we differentiate w.r.t. the BLOCKED class
        # logit (class index 0).  For the binary model, we use target="blocked".

        if self.num_classes == 3:
            heatmap = self.gradcam(tensor, target="blocked", target_class_idx=0)
        else:
            heatmap = self.gradcam(tensor, target="blocked")

        overlay = overlay_heatmap(original, heatmap, alpha=0.45)

        import matplotlib
        heatmap_colour = (matplotlib.colormaps["jet"](heatmap)[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil    = Image.fromarray(heatmap_colour).resize(original.size, Image.BILINEAR)

        heatmap_coverage = float((heatmap > 0.5).mean())

        # ── Step 4: LLM explanation ───────────────────────────────────────────
        risk, prediction = "BLOCKED", "BLOCKED"
        import re
        if not self.use_llm:
            explanation = "LLM explanation disabled for this run. Re-run with use_llm=True for a full report."
        else:
            try:
                # heatmap (raw Grad-CAM array, computed above) is passed through
                # so explain()'s v2 prompt can ground its location/coverage
                # description in the same numbers used for heatmap_coverage,
                # rather than letting the LLM estimate them from the overlay
                # image alone.
                explanation = self.explainer.explain(overlay, p_best, risk, heatmap=heatmap)
                explanation = re.sub(r"\*\*(.+?)\*\*", r"\1", explanation)
                # Older (v1) prompts return an OBSERVATION/RISK ASSESSMENT/
                # RECOMMENDED ACTION block and may be preceded by stray LLM
                # preamble lines that need stripping. The current default (v2)
                # prompt has no section headers at all — only strip leading
                # lines when a header actually appears somewhere in the text;
                # otherwise this loop would silently discard the whole v2
                # explanation looking for headers that were never requested.
                section_starts = ("OBSERVATION", "RISK ASSESSMENT", "RECOMMENDED ACTION")
                lines = explanation.splitlines()
                if any(line.strip().upper().startswith(s) for line in lines for s in section_starts):
                    while lines and not any(
                        lines[0].strip().upper().startswith(s) for s in section_starts
                    ):
                        lines = lines[1:]
                    explanation = "\n".join(lines).strip()
            except Exception as llm_err:
                print(f"  [LLM unavailable] {llm_err}")
                explanation = "LLM explanation unavailable — Groq API could not be reached. Retry later."

        return {
            "prediction":       prediction,
            "risk":             risk,
            "p_blocked":        round(p_best, 4),
            "p_blocked_raw":    round(p_blocked_orig, 4),
            "p_clear":          round(p_clear_orig, 4),
            "p_other":          round(p_other_orig, 4),
            "heatmap_coverage": round(heatmap_coverage, 4),
            "explanation":      explanation,
            "original_b64":     _pil_to_base64(original),
            "overlay_b64":      _pil_to_base64(overlay),
            "heatmap_b64":      _pil_to_base64(heatmap_pil),
            "quality_flags":    [],
            "needs_review":     False,
            "sharpness":        None,
            "brightness":       None,
            "colour_cv":        None,
            "mean_sat":         None,
        }

    # ── Private result builders ────────────────────────────────────────────────
    #
    # (The old _flagged_result/_uncertain_result were removed along with the
    # quality-check system — both depended entirely on the `quality` dict and
    # were never actually called from run() even before that cleanup. The
    # _flagged_result below is new: it's driven by decision margin, not by
    # pixel-quality heuristics.)

    def _flagged_result(
        self,
        original: Image.Image,
        p_blocked_orig: float,
        p_clear: float,
        p_other: float,
        dist_to_confident: float,
    ) -> dict:
        """Return a FLAGGED (needs review) result for a too-close-to-call prediction."""
        return {
            "prediction":       "FLAGGED",
            "risk":             "FLAGGED",
            "p_blocked":        round(p_blocked_orig, 4),
            "p_blocked_raw":    round(p_blocked_orig, 4),
            "p_clear":          round(p_clear, 4),
            "p_other":          round(p_other, 4),
            "heatmap_coverage": None,
            # Matches the v2 style used for BLOCKED/CLEAR explanations: plain
            # sentences, no section headers, no raw numbers (p_blocked/
            # thresholds) surfaced to the operator — just what it means and
            # what to do. Headerless text also means the frontend renders
            # this in the same clear, centred box as any other explanation,
            # rather than the old three-section OBSERVATION/RISK ASSESSMENT/
            # RECOMMENDED ACTION template.
            "explanation": (
                "This frame's classification is too close to call confidently. "
                "Asserting BLOCKED or CLEAR here would be more likely to flip on "
                "the next scan than reflect a considered judgement, so a person "
                "should check this image directly."
            ),
            "original_b64":     _pil_to_base64(original),
            "overlay_b64":      _pil_to_base64(original),
            "heatmap_b64":      _pil_to_base64(original),
            "quality_flags":    ["low_margin"],
            "needs_review":     True,
            "sharpness":        None,
            "brightness":       None,
            "colour_cv":        None,
            "mean_sat":         None,
        }

    def _clear_result(
        self,
        original: Image.Image,
        p_blocked_orig: float,
        p_clear: float,
        p_other: float,
    ) -> dict:
        """Return a CLEAR result (no GradCAM)."""
        return {
            "prediction":       "CLEAR",
            "risk":             "LOW",
            "p_blocked":        round(p_blocked_orig, 4),
            "p_blocked_raw":    round(p_blocked_orig, 4),
            "p_clear":          round(p_clear, 4),
            "p_other":          round(p_other, 4),
            "heatmap_coverage": None,
            "explanation":      "Model is confident the drain is clear.",
            "original_b64":     _pil_to_base64(original),
            "overlay_b64":      _pil_to_base64(original),
            "heatmap_b64":      _pil_to_base64(original),
            "quality_flags":    [],
            "needs_review":     False,
            "sharpness":        None,
            "brightness":       None,
            "colour_cv":        None,
            "mean_sat":         None,
        }

    def generate_blocked_overlay(self, image: Image.Image) -> dict:
        """
        Force-generate a GradCAM overlay AND a real LLM inspection report
        targeting the BLOCKED class, regardless of what the model itself
        currently predicts on this image. Used when a human manually
        reclassifies an uncertain (FLAGGED) result to BLOCKED via
        /results/reclassify — the automatic pipeline only ever runs GradCAM
        and the LLM explainer for the model's OWN confident BLOCKED calls
        (see run()'s Steps 3-4), so a manually-corrected result would
        otherwise have no real heatmap and a generic placeholder
        explanation instead of a genuine one. This makes a manually
        confirmed BLOCKED case look and read exactly like one the model
        called BLOCKED itself — same overlay logic, same explanation logic.

        Returns {overlay_b64, heatmap_b64, heatmap_coverage, explanation} —
        the same fields run()'s BLOCKED path produces, ready to overwrite
        onto an existing stored result.
        """
        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)

        if self.num_classes == 3:
            heatmap = self.gradcam(tensor, target="blocked", target_class_idx=0)
        else:
            heatmap = self.gradcam(tensor, target="blocked")

        overlay = overlay_heatmap(original, heatmap, alpha=0.45)

        import matplotlib
        heatmap_colour = (matplotlib.colormaps["jet"](heatmap)[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil    = Image.fromarray(heatmap_colour).resize(original.size, Image.BILINEAR)
        heatmap_coverage = float((heatmap > 0.5).mean())

        # p_blocked purely for the LLM prompt's context — the BLOCKED verdict
        # itself already comes from the human's manual call, not this number.
        p_blocked, _, _ = _get_p_blocked(self.model, tensor, self.num_classes)

        import re
        if not self.use_llm:
            explanation = "LLM explanation disabled for this run. Re-run with use_llm=True for a full report."
        else:
            try:
                explanation = self.explainer.explain(overlay, p_blocked, "BLOCKED", heatmap=heatmap)
                explanation = re.sub(r"\*\*(.+?)\*\*", r"\1", explanation)
                section_starts = ("OBSERVATION", "RISK ASSESSMENT", "RECOMMENDED ACTION")
                lines = explanation.splitlines()
                if any(line.strip().upper().startswith(s) for line in lines for s in section_starts):
                    while lines and not any(
                        lines[0].strip().upper().startswith(s) for s in section_starts
                    ):
                        lines = lines[1:]
                    explanation = "\n".join(lines).strip()
            except Exception as llm_err:
                print(f"  [LLM unavailable] {llm_err}")
                explanation = "LLM explanation unavailable — Groq API could not be reached. Retry later."

        return {
            "overlay_b64":      _pil_to_base64(overlay),
            "heatmap_b64":      _pil_to_base64(heatmap_pil),
            "heatmap_coverage": round(heatmap_coverage, 4),
            "explanation":      explanation,
        }

    def close(self):
        """Release GradCAM hooks."""
        self.gradcam.remove_hooks()
