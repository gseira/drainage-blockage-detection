"""
run_risk_demo.py
----------------
Runs the RiskIndicator + LLM explainer on a sample of test images.

For each image:
    - Computes Grad-CAM heatmap
    - Assigns HIGH (blocked) or LOW (clear) risk
    - Sends the overlay image to Gemini 1.5 Flash for a natural language explanation
    - Prints the explanation to the terminal
    - Saves a visual grid: original | heatmap | overlay | risk badge

Output: results/figures/risk_demo.png

Usage (on HEX):
    CUDA_VISIBLE_DEVICES=4 python risk_assessment/run_risk_demo.py
    CUDA_VISIBLE_DEVICES=4 python risk_assessment/run_risk_demo.py --n 6
    CUDA_VISIBLE_DEVICES=4 python risk_assessment/run_risk_demo.py --n 6 --seed 42
"""

import argparse
import csv
import re
import random
import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "interpretability"))
sys.path.insert(0, str(Path(__file__).parent.parent / "risk_assessment"))

from gradcam import overlay_heatmap
from model import get_model
from risk_indicator import RiskIndicator
from llm_explainer import LLMExplainer

RISK_COLOURS = {
    "HIGH": "#E53935",
    "LOW":  "#43A047",
}
RISK_ORDER = {"HIGH": 0, "LOW": 1}


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


def build_risk_grid(samples, model, ri, explainer, device, transform, out_path):
    results = []
    print("Running risk assessment + LLM explanations...\n")

    for rec in samples:
        original = Image.open(rec["path"]).convert("RGB")
        tensor   = transform(original).unsqueeze(0).to(device)
        result   = ri(tensor)
        overlay  = overlay_heatmap(original, result["heatmap"], alpha=0.45)

        # LLM explanation
        print(f"Image: {Path(rec['path']).name}  [{result['risk']}]")
        explanation = explainer.explain(overlay, result["p_blocked"], result["risk"])
        print(explanation)
        print()

        results.append({
            "original":    original,
            "heatmap":     result["heatmap"],
            "overlay":     overlay,
            "risk":        result["risk"],
            "p_blocked":   result["p_blocked"],
            "explanation": explanation,
            "label":       rec["label"],
        })

    results.sort(key=lambda r: RISK_ORDER[r["risk"]])

    n = len(results)
    # height_ratios: image row = 3 units, text row = 5 units
    height_ratios = []
    for _ in range(n):
        height_ratios += [3, 5]

    fig = plt.figure(figsize=(20, sum(height_ratios) * 0.9))
    gs  = fig.add_gridspec(
        n * 2, 4,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.05,
        left=0.08, right=0.98,
        top=0.97,  bottom=0.02,
    )

    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay", "Risk Level"]

    for idx, res in enumerate(results):
        colour   = RISK_COLOURS[res["risk"]]
        true_lbl = "BLOCKED" if res["label"] == 0 else "CLEAR"
        img_row  = idx * 2
        txt_row  = idx * 2 + 1

        # ── Image row: 4 columns ─────────────────────────────────────────
        for col, (img, cmap, title) in enumerate([
            (res["original"], None,  "Original"),
            (res["heatmap"],  "jet", "Grad-CAM Heatmap"),
            (res["overlay"],  None,  "Overlay"),
        ]):
            ax = fig.add_subplot(gs[img_row, col])
            ax.imshow(img, cmap=cmap, **({"vmin": 0, "vmax": 1} if cmap else {}))
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2.5)
            if col == 0:
                ax.set_ylabel(
                    f"True: {true_lbl}", fontsize=9, fontweight="bold",
                    rotation=0, labelpad=70, va="center",
                )

        # ── Risk badge ───────────────────────────────────────────────────
        ax_badge = fig.add_subplot(gs[img_row, 3])

        # Gradient-style background using imshow
        grad = np.linspace(0.85, 1.0, 256).reshape(256, 1)
        badge_rgb = {
            "HIGH": (0.90, 0.18, 0.18),
            "LOW":  (0.20, 0.72, 0.30),
        }[res["risk"]]
        bg = np.ones((256, 1, 3)) * np.array(badge_rgb)
        bg[:, :, 0] *= grad
        bg[:, :, 1] *= grad
        bg[:, :, 2] *= grad
        bg = np.clip(bg, 0, 1)
        ax_badge.imshow(bg, aspect="auto", extent=[0, 1, 0, 1])

        # Icon
        icon = "⚠" if res["risk"] == "HIGH" else "✔"
        ax_badge.text(0.5, 0.78, icon,
                      ha="center", va="center", fontsize=22,
                      color="white", transform=ax_badge.transAxes,
                      alpha=0.9)
        # Risk label
        ax_badge.text(0.5, 0.54, res["risk"],
                      ha="center", va="center", fontsize=26, fontweight="bold",
                      color="white", transform=ax_badge.transAxes,
                      fontfamily="DejaVu Sans")
        # "RISK" subtitle
        ax_badge.text(0.5, 0.33, "R  I  S  K",
                      ha="center", va="center", fontsize=11,
                      color="white", alpha=0.85, transform=ax_badge.transAxes,
                      fontfamily="DejaVu Sans", fontweight="bold")

        ax_badge.set_xlim(0, 1)
        ax_badge.set_ylim(0, 1)
        ax_badge.set_xticks([])
        ax_badge.set_yticks([])
        for spine in ax_badge.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.5)
        if idx == 0:
            ax_badge.set_title("Risk Level", fontsize=11, fontweight="bold", pad=6)

        # ── Explanation panel (spans all 4 columns) ──────────────────────
        ax_text = fig.add_subplot(gs[txt_row, :])
        ax_text.set_facecolor("#1a1a2e")
        ax_text.set_xticks([])
        ax_text.set_yticks([])
        for spine in ax_text.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2)

        # Strip common LLM preambles
        explanation = res["explanation"]
        # Strip any LLM preamble line (e.g. "Here is the inspection report:\n\n")
        lines = explanation.splitlines()
        if lines and re.match(
            r"^here\b.*(report|assessment|summary)",
            lines[0].strip(), re.IGNORECASE
        ):
            explanation = "\n".join(lines[1:]).lstrip("\n ")
        # Also strip markdown bold markers (**TEXT:**)
        explanation = re.sub(r"\*\*(.+?)\*\*", r"\1", explanation)

        # Hard-wrap at 115 chars — safe width for fontsize 8 monospace at 20in
        raw_lines     = explanation.splitlines()
        wrapped_lines = []
        for line in raw_lines:
            if line.strip() == "":
                wrapped_lines.append("")
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=115) or [""])
        wrapped_text = "\n".join(wrapped_lines)

        ax_text.text(
            0.01, 0.97, wrapped_text,
            ha="left", va="top",
            fontsize=8,
            color="white",
            transform=ax_text.transAxes,
            family="monospace",
            linespacing=1.5,
            clip_on=False,
        )
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)

    handles = [
        mpatches.Patch(color=RISK_COLOURS["HIGH"], label="HIGH RISK — immediate inspection required"),
        mpatches.Patch(color=RISK_COLOURS["LOW"],  label="LOW RISK — no obstruction detected"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=10, frameon=True, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Drainage Blockage Risk Indicator  ·  Grad-CAM + LLM Inspection Report",
        fontsize=13, fontweight="bold", color="#222222",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(cfg_path, n, seed):
    if seed is not None:
        random.seed(seed)

    cfg    = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    model = get_model(pretrained=False).to(device)
    ckpt  = torch.load(
        Path(cfg["paths"]["checkpoints"]) / "best_model.pt", map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")

    ri        = RiskIndicator(model, target_layer="layer4")
    explainer = LLMExplainer()
    transform = get_transform(cfg["data"]["img_size"])
    out_dir   = Path(cfg["paths"]["logs"]).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_records = load_csv(str(Path(cfg["data"]["manifest_dir"]) / "test.csv"))
    blocked = [r for r in test_records if r["label"] == 0]
    clear   = [r for r in test_records if r["label"] == 1]
    half    = n // 2
    samples = (random.sample(blocked, min(half, len(blocked))) +
               random.sample(clear,   min(n - half, len(clear))))
    random.shuffle(samples)

    build_risk_grid(samples, model, ri, explainer, device, transform,
                    out_path=out_dir / "risk_demo.png")

    ri.remove_hooks()
    print("\nDone. Pull to local:")
    print("  rsync -avz ges65@cheery.cs.bath.ac.uk:~/dissertation/results/figures/ "
          "~/Desktop/dissertation/results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        Path(__file__).parent.parent / "experiments" / "config.yaml"
    ))
    parser.add_argument("--n",    type=int, default=6,  help="Number of images")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.n, args.seed)
