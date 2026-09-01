"""
plot_results.py
---------------
Generates two figures from the training log CSV:
  1. Training & validation loss over epochs
  2. Training & validation F1 over epochs

Saves PNGs to results/figures/.

Usage:
    python src/plot_results.py
    python src/plot_results.py --log results/logs/training_log_full54k.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def plot_training(log_path: str, output_dir: str) -> None:
    df  = pd.read_csv(log_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]

    style = {
        "train": {"color": "#2196F3", "lw": 2,   "label": "Train"},
        "val":   {"color": "#F44336", "lw": 2,   "label": "Validation"},
        "best":  {"color": "#4CAF50", "lw": 1.5, "ls": "--"},
    }

    # ── Figure 1: Loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(df["epoch"], df["train_loss"], **{k: v for k, v in style["train"].items()})
    ax.plot(df["epoch"], df["val_loss"],   **{k: v for k, v in style["val"].items()})
    ax.axvline(best_epoch, color=style["best"]["color"],
               lw=style["best"]["lw"], ls=style["best"]["ls"],
               label=f"Best epoch ({int(best_epoch)})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    loss_path = out / "loss_curve.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {loss_path}")

    # ── Figure 2: F1 ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(df["epoch"], df["train_f1"], **{k: v for k, v in style["train"].items()})
    ax.plot(df["epoch"], df["val_f1"],   **{k: v for k, v in style["val"].items()})
    ax.axvline(best_epoch, color=style["best"]["color"],
               lw=style["best"]["lw"], ls=style["best"]["ls"],
               label=f"Best epoch ({int(best_epoch)})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Training and Validation F1 Score", fontsize=14, fontweight="bold")
    ax.set_ylim(0.85, 1.01)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    f1_path = out / "f1_curve.png"
    fig.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f1_path}")

    # ── Figure 3: Combined (for dissertation) ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title, ylim in [
        (ax1, ("train_loss", "val_loss"), "Loss (BCE)",
         "Loss Curve", None),
        (ax2, ("train_f1",  "val_f1"),   "F1 Score",
         "F1 Score Curve", (0.85, 1.01)),
    ]:
        ax.plot(df["epoch"], df[metric[0]], **{k: v for k, v in style["train"].items()})
        ax.plot(df["epoch"], df[metric[1]], **{k: v for k, v in style["val"].items()})
        ax.axvline(best_epoch, color=style["best"]["color"],
                   lw=style["best"]["lw"], ls=style["best"]["ls"],
                   label=f"Best epoch ({int(best_epoch)})")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    fig.suptitle("ResNet-50 Classification — Training History (54k images)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    combined_path = out / "training_curves_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="/homes/ges65/dissertation/results/logs/training_log_full54k.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/homes/ges65/dissertation/results/figures",
    )
    args = parser.parse_args()
    plot_training(args.log, args.output_dir)
