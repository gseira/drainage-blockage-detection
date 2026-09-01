"""
visualise_predictions.py
------------------------
Loads the best checkpoint and visualises sample predictions on the test set.

Produces two figures:
  1. A grid of correct predictions (TP blocked + TN clear)
  2. A grid of errors — missed blockages (FN) and false alarms (FP)

Saves PNGs to results/figures/.

Usage:
    python src/visualise_predictions.py
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import get_model


LABEL_NAMES = {0: "blocked", 1: "clear"}
COLOURS     = {"TP": "#4CAF50", "TN": "#4CAF50", "FN": "#F44336", "FP": "#FF9800"}


def load_test_records(manifest_path: str) -> list[dict]:
    import csv
    records = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({"path": row["path"], "label": int(row["label"])})
    return records


def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def predict_batch(model, records, device, img_size, batch_size=64):
    """Run inference on all records. Returns list of (record, pred, prob)."""
    transform = get_inference_transform(img_size)
    results   = []

    for i in range(0, len(records), batch_size):
        batch_records = records[i:i + batch_size]
        imgs = torch.stack([
            transform(Image.open(r["path"]).convert("RGB"))
            for r in batch_records
        ]).to(device)

        with torch.no_grad():
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu().squeeze(1).tolist()
            preds  = [int(p >= 0.5) for p in probs]

        for rec, pred, prob in zip(batch_records, preds, probs):
            results.append({"record": rec, "pred": pred, "prob_clear": prob})

    return results


def classify_outcome(true_label, pred):
    if true_label == 0 and pred == 0: return "TP"   # correctly identified blocked
    if true_label == 1 and pred == 1: return "TN"   # correctly identified clear
    if true_label == 0 and pred == 1: return "FN"   # missed blockage — dangerous
    if true_label == 1 and pred == 0: return "FP"   # false alarm


def plot_grid(samples, title, output_path, n_cols=4):
    n      = len(samples)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, item in zip(axes, samples):
        rec     = item["record"]
        outcome = item["outcome"]
        prob    = item["prob_clear"]

        img = Image.open(rec["path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        true_lbl = LABEL_NAMES[rec["label"]]
        pred_lbl = LABEL_NAMES[item["pred"]]
        colour   = COLOURS[outcome]

        ax.set_title(
            f"True: {true_lbl}\nPred: {pred_lbl}  ({1-prob:.2f})",
            fontsize=9, color=colour, fontweight="bold"
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)
            spine.set_visible(True)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main(cfg_path: str, n_samples: int = 8, seed: int = 42) -> None:
    random.seed(seed)

    cfg    = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(cfg["paths"]["logs"]).parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = get_model(cfg["model"]["architecture"]).to(device)
    ckpt  = torch.load(
        Path(cfg["paths"]["checkpoints"]) / "best_model.pt",
        map_location=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Load test records and run inference
    test_csv = Path(cfg["data"]["manifest_dir"]) / "test.csv"
    records  = load_test_records(str(test_csv))
    print(f"Running inference on {len(records)} test images...")

    results = predict_batch(model, records, device, cfg["data"]["img_size"])

    # Separate by outcome
    buckets = {"TP": [], "TN": [], "FN": [], "FP": []}
    for r in results:
        outcome = classify_outcome(r["record"]["label"], r["pred"])
        buckets[outcome].append({**r, "outcome": outcome})

    print(f"\nTest results:")
    print(f"  TP (correct blocked): {len(buckets['TP'])}")
    print(f"  TN (correct clear):   {len(buckets['TN'])}")
    print(f"  FN (missed blockage): {len(buckets['FN'])}  ← critical errors")
    print(f"  FP (false alarm):     {len(buckets['FP'])}")

    # Figure 1: Correct predictions
    correct_samples = (
        random.sample(buckets["TP"], min(n_samples // 2, len(buckets["TP"]))) +
        random.sample(buckets["TN"], min(n_samples // 2, len(buckets["TN"])))
    )
    random.shuffle(correct_samples)
    plot_grid(
        correct_samples,
        "Correct Predictions — Green border = correct",
        out / "correct_predictions.png",
    )

    # Figure 2: Errors
    errors = buckets["FN"] + buckets["FP"]
    if errors:
        error_samples = random.sample(errors, min(n_samples, len(errors)))
        plot_grid(
            error_samples,
            "Model Errors — Red = missed blockage, Orange = false alarm",
            out / "error_predictions.png",
        )
    else:
        print("No errors to visualise — perfect test set performance!")

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "configs" / "config.yaml"),
    )
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()
    main(args.config, args.n_samples)
