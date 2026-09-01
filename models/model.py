"""
model.py
--------
ResNet-50 classification model for drainage blockage detection.

A pretrained ResNet-50 (ImageNet weights) with its final fully-connected
layer replaced by a configurable output head.

num_classes=1 (default, binary):
    Single logit — pass through sigmoid to get P(clear).
    Loss: BCEWithLogitsLoss

num_classes=3 (3-class):
    Three logits: [blocked_logit, clear_logit, other_logit]
    Loss: CrossEntropyLoss with class weights.
    class 0 = BLOCKED, class 1 = CLEAR, class 2 = OTHER

The last convolutional block (layer4) is the target layer for Grad-CAM.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """
    Build and return the ResNet-50 classifier.

    Args:
        pretrained:  Load ImageNet pretrained weights (default True).
        num_classes: Number of output classes.
                     1 → binary (BCEWithLogitsLoss)
                     3 → 3-class (CrossEntropyLoss)

    Returns:
        ResNet-50 with the requested output head.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    for nc in [1, 3]:
        model = get_model(pretrained=False, num_classes=nc)
        stats = count_parameters(model)
        dummy = torch.randn(4, 3, 224, 224)
        out   = model(dummy)
        print(f"ResNet-50 (num_classes={nc})  |  total: {stats['total']:,}  "
              f"|  trainable: {stats['trainable']:,}  |  output: {out.shape}")
