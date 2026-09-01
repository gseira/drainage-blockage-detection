"""
confusion_matrix.py
--------------------
Evaluates the trained ResNet-50 checkpoint against a held-out manifest split
(normally test.csv) and produces a confusion matrix figure plus a printed
classification report (precision / recall / F1 / support).

Matches the project's existing conventions: uses get_model() from
models/model.py, get_transforms()/ManifestDataset from data/dataset.py, and
loads a checkpoint saved by models/train.py (dict with "model_state_dict").

Usage:
    python evaluation/confusion_matrix.py \
        --checkpoint results/checkpoints/best_model.pt \
        --manifest   data/manifests/test.csv \
        --out        results/figures/confusion_matrix.png

If your manifest directory differs from data/manifests (e.g. the remote
training server's path), just point --manifest at the right test.csv.
"""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Make sure the project's models/ and data/ folders are importable regardless
# of where this script is run from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "data"))

from model import get_model          # noqa: E402
from dataset import ManifestDataset, get_transforms  # noqa: E402
from torch.utils.data import DataLoader              # noqa: E402


LABEL_NAMES = ["blocked", "clear"]   # matches LABEL_MAP in build_manifest.py (0=blocked, 1=clear)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = get_model(pretrained=False, num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    if "epoch" in ckpt:
        print(f"  epoch={ckpt['epoch']}  val_loss={ckpt.get('val_loss')}  val_f1={ckpt.get('val_f1')}")
    return model


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().squeeze(1).tolist()
        all_preds += preds
        all_labels += labels.tolist()
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — held-out test set")

    # Annotate each cell with its count.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved confusion matrix figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "checkpoints" / "best_model.pt"))
    parser.add_argument("--manifest", required=True, help="Path to test.csv (path,label columns)")
    parser.add_argument("--out", default=str(ROOT / "results" / "figures" / "confusion_matrix.png"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset = ManifestDataset(args.manifest, transform=get_transforms("test", args.img_size))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Loaded {len(dataset)} images from {args.manifest}")

    model = load_model(args.checkpoint, device)
    y_true, y_pred = run_inference(model, loader, device)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"           pred_blocked  pred_clear")
    print(f"true_blocked   {cm[0][0]:>8}     {cm[0][1]:>8}")
    print(f"true_clear     {cm[1][0]:>8}     {cm[1][1]:>8}")

    plot_confusion_matrix(y_true, y_pred, args.out)


if __name__ == "__main__":
    main()
