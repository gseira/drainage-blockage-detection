"""
risk_indicator.py
-----------------
Binary risk flag derived directly from the model prediction.

Rule:
    Model predicts BLOCKED  →  HIGH risk
    Model predicts CLEAR    →  LOW  risk

Also runs Grad-CAM to produce the heatmap used by the LLM explainer.

Usage:
    from risk_assessment.risk_indicator import RiskIndicator
    ri = RiskIndicator(model)
    result = ri(image_tensor)
    # result: {'risk': 'HIGH', 'p_blocked': 0.99, 'heatmap': ndarray}
    ri.remove_hooks()
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "interpretability"))
from gradcam import GradCAM


COLOURS = {
    "HIGH":  "\033[91m",   # red
    "LOW":   "\033[92m",   # green
    "RESET": "\033[0m",
}


class RiskIndicator:
    """
    Risk assessor for a single drainage image.

    Runs the model and Grad-CAM in one call. Risk is determined purely
    by the model prediction: blocked = HIGH, clear = LOW.

    Args:
        model:        Trained ResNet-50 (eval mode recommended).
        target_layer: Layer to hook for Grad-CAM (default "layer4").
    """

    def __init__(self, model, target_layer: str = "layer4"):
        self.model = model
        self.cam   = GradCAM(model, target_layer=target_layer)

    def __call__(self, image_tensor: torch.Tensor) -> dict:
        """
        Run risk assessment on one image.

        Args:
            image_tensor: Pre-processed tensor, shape (1, 3, H, W).

        Returns:
            dict with keys:
                risk      (str)     — "HIGH" or "LOW"
                p_blocked (float)   — model's blockage probability
                heatmap   (ndarray) — Grad-CAM heatmap (H, W), values in [0, 1]
        """
        self.model.eval()

        with torch.no_grad():
            logit = self.model(image_tensor)
        p_blocked = 1.0 - torch.sigmoid(logit).item()
        risk      = "HIGH" if p_blocked >= 0.5 else "LOW"
        heatmap   = self.cam(image_tensor, target="blocked")

        return {
            "risk":      risk,
            "p_blocked": round(p_blocked, 4),
            "heatmap":   heatmap,
        }

    def remove_hooks(self):
        self.cam.remove_hooks()
