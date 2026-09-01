"""
gradcam.py
----------
Grad-CAM implementation for the ResNet-50 drainage blockage classifier.

Supports both:
  - Binary model (fc outputs 1 logit, BCEWithLogitsLoss)
  - 3-class model (fc outputs 3 logits, CrossEntropyLoss)
    classes: 0=BLOCKED, 1=CLEAR, 2=OTHER

Grad-CAM highlights the image regions that most influenced a prediction
by computing the gradient of the output score with respect to the last
convolutional layer's feature maps.

Usage:
    from interpretability.gradcam import GradCAM
    cam = GradCAM(model)
    # Binary model:
    heatmap = cam(image_tensor, target="blocked")
    # 3-class model — target the BLOCKED class (index 0):
    heatmap = cam(image_tensor, target="blocked")
    cam.remove_hooks()
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    Grad-CAM for ResNet-50.  Works with both the binary (1-output) and the
    3-class (3-output) variants of the drainage classifier.

    Binary model — single logit:
        logit > 0  →  P(clear)  > 0.5  →  CLEAR
        logit < 0  →  P(blocked) > 0.5  →  BLOCKED
        target="clear"   → differentiates +logit
        target="blocked" → differentiates -logit
        target="pred"    → auto-detect

    3-class model — three logits [blocked, clear, other]:
        target="blocked" → differentiates logits[:, 0]   (BLOCKED class)
        target="clear"   → differentiates logits[:, 1]
        target="other"   → differentiates logits[:, 2]
        target="pred"    → auto-detect (highest logit class)
        target_class_idx → override by integer index (0/1/2)

    In the inference pipeline, GradCAM is only called for BLOCKED predictions.
    """

    # Maps string target name → class index for the 3-class model
    _NAME_TO_IDX = {"blocked": 0, "clear": 1, "other": 2}

    def __init__(self, model: nn.Module, target_layer: str = "layer4"):
        self.model        = model
        self.feature_maps = None   # activations captured at target layer
        self.gradients    = None   # gradients captured at target layer

        # Register hooks on the named target layer
        target_module = dict(model.named_modules())[target_layer]
        self._fwd_hook = target_module.register_forward_hook(self._save_features)
        self._bwd_hook = target_module.register_full_backward_hook(self._save_gradients)

    # ── Hooks ──────────────────────────────────────────────────────────────────

    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Remove hooks to free memory when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Main method ────────────────────────────────────────────────────────────

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target: str = "pred",
        target_class_idx: int = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for one image.

        Args:
            image_tensor:     Pre-processed tensor, shape (1, 3, H, W).
            target:           "blocked", "clear", "other", or "pred" (auto-detect).
                              Ignored when target_class_idx is set.
            target_class_idx: (3-class only) Override target with explicit class index.
                              0=BLOCKED, 1=CLEAR, 2=OTHER.

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        # ── Forward pass ──────────────────────────────────────────────────
        logits = self.model(image_tensor)   # (1, 1) or (1, 3)
        is_binary = (logits.shape[-1] == 1) or (logits.dim() == 1) or (logits.numel() == 1)

        if is_binary:
            # ── Binary model ──────────────────────────────────────────────
            # logit > 0 → CLEAR, logit < 0 → BLOCKED
            logit_val = logits.view(1)
            if target == "pred":
                target = "clear" if logit_val.item() >= 0 else "blocked"
            # Differentiate w.r.t. the requested class
            score = logit_val if target == "clear" else -logit_val

        else:
            # ── 3-class model ─────────────────────────────────────────────
            # Resolve target to a class index
            if target_class_idx is None:
                if target == "pred":
                    target_class_idx = int(logits.argmax(dim=1).item())
                else:
                    target_class_idx = self._NAME_TO_IDX.get(target, 0)
            score = logits[0, target_class_idx]

        # ── Backward pass ─────────────────────────────────────────────────
        self.model.zero_grad()
        score.backward()

        # ── Grad-CAM ──────────────────────────────────────────────────────
        # Global average pool gradients → per-channel weights: (C,)
        weights = self.gradients.squeeze(0).mean(dim=(1, 2))

        # Weighted sum of feature maps → raw CAM: (H_feat, W_feat)
        feature_maps = self.feature_maps.squeeze(0)   # (C, H_feat, W_feat)
        cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32,
                          device=feature_maps.device)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        # ReLU — keep only regions that pushed toward the target class
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


# ── Utility functions ──────────────────────────────────────────────────────────

def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: PIL Image (any size).
        heatmap:        numpy array (H, W) with values in [0, 1].
        alpha:          blending factor (0 = original only, 1 = heatmap only).
        colormap:       matplotlib colormap name.

    Returns:
        Blended PIL Image.
    """
    import matplotlib

    # Resize heatmap to match original image
    h, w   = np.array(original_image).shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply colourmap
    cmap        = matplotlib.colormaps[colormap]
    heatmap_rgb = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Blend
    original_rgb = original_image.convert("RGB")
    blended      = Image.blend(original_rgb, heatmap_pil, alpha=alpha)
    return blended
