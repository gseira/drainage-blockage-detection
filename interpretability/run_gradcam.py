"""
run_gradcam.py
--------------
Applies Grad-CAM to a sample of test images and saves a visualisation grid.

For each image shows three panels:
    1. Original image
    2. Grad-CAM heatmap
    3. Overlay (heatmap blended onto original)

Saves to results/figures/gradcam_blocked.png  — correct blocked (TP) + missed blockages (FN)
        results/figures/gradcam_clear.png     — correct clear predictions (TN)

Usage (on HEX):
    CUDA_VISIBLE_DEVICES=4 python interpretability/run_gradcam.py
    CUDA_VISIBLE_DEVICES=4 python interpretability/run_gradcam.py --n 6
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "interpretability"))

from gradcam import GradCAM, overlay_heatmap
from model import get_model

LABEL_NAMES = {0: "BLOCKED", 1: "CLEAR"}


def load_csv(path):
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({"path": row["path"], "label": int(row["label"])})
    return records


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def run_gradcam_grid(samples, model, cam, device, transform, title, out_path):
    """
    For each sample: original | heatmap | overlay.
    Rows = images, Cols = [original, heatmap, overlay].
    """
    n     = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
    if n == 1:
        axes = [axes]

    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=12, fontweight="bold", pad=8)

    for row, rec in enumerate(samples):
        original = Image.open(rec["path"]).convert("RGB")
        tensor   = transform(original).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        pred = 1 if prob >= 0.5 else 0

        # Grad-CAM — show what drove the BLOCKED signal
        heatmap = cam(tensor, target="blocked")
        overlay = overlay_heatmap(original, heatmap, alpha=0.45)

        true_lbl = LABEL_NAMES[rec["label"]]
        pred_lbl = LABEL_NAMES[pred]
        correct  = pred == rec["label"]
        colour   = "#43A047" if correct else "#E53935"
        status   = "✓" if correct else "✗"

        row_label = (
            f"{status}  True: {true_lbl}  |  Pred: {pred_lbl}  "
            f"|  P(blocked)={1-prob:.2f}"
        )

        axes[row][0].imshow(original)
        axes[row][0].set_ylabel(row_label, fontsize=8.5,
                                color=colour, fontweight="bold",
                                rotation=0, labelpad=120, va="center")
        axes[row][1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[row][2].imshow(overlay)

        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            for spine in axes[row][col].spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(cfg_path, n, seed):
    if seed is not None:
        random.seed(seed)

    cfg    = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = get_model(pretrained=False).to(device)
    ckpt  = torch.load(
        Path(cfg["paths"]["checkpoints"]) / "best_model.pt", map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")

    cam       = GradCAM(model, target_layer="layer4")
    transform = get_transform(cfg["data"]["img_size"])
    out_dir   = Path(cfg["paths"]["logs"]).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run inference on test subset to find TP/FN/TN ─────────────────────
    test_records = load_csv(str(Path(cfg["data"]["manifest_dir"]) / "test.csv"))
    sample_pool  = random.sample(test_records, min(500, len(test_records)))

    tp_blocked, fn_blocked, tn_clear = [], [], []

    print("Finding TP/FN/TN samples...")
    for rec in sample_pool:
        tensor = transform(
            Image.open(rec["path"]).convert("RGB")
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(tensor)
        pred = 1 if logit.item() >= 0 else 0

        if rec["label"] == 0 and pred == 0:
            tp_blocked.append(rec)
        elif rec["label"] == 0 and pred == 1:
            fn_blocked.append(rec)
        elif rec["label"] == 1 and pred == 1:
            tn_clear.append(rec)

    print(f"Found: {len(tp_blocked)} correct blocked, "
          f"{len(fn_blocked)} missed blockages, "
          f"{len(tn_clear)} correct clear")

    # ── Figure 1: Blocked (TP + FN) ────────────────────────────────────────
    blocked_samples = (
        random.sample(tp_blocked, min(n // 2, len(tp_blocked))) +
        random.sample(fn_blocked, min(n // 2, len(fn_blocked)))
    )
    random.shuffle(blocked_samples)

    run_gradcam_grid(
        blocked_samples[:n], model, cam, device, transform,
        title="Grad-CAM — Blocked Images  "
              "(green = correctly identified, red = missed blockage)",
        out_path=out_dir / "gradcam_blocked.png",
    )

    # ── Figure 2: Clear (TN) ───────────────────────────────────────────────
    clear_samples = random.sample(tn_clear, min(n, len(tn_clear)))

    run_gradcam_grid(
        clear_samples, model, cam, device, transform,
        title="Grad-CAM — Clear Images  "
              "(what the model focuses on when predicting clear)",
        out_path=out_dir / "gradcam_clear.png",
    )

    cam.remove_hooks()
    print("\nDone. Pull to local:")
    print("  rsync -avz ges65@cheery.cs.bath.ac.uk:~/dissertation/results/figures/ "
          "~/dev/dissertation/results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(
            Path(__file__).parent.parent / "experiments" / "config.yaml"
        )
    )
    parser.add_argument("--n",    type=int, default=6,  help="Images per figure")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.n, args.seed)
