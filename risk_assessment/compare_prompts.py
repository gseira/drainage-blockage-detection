"""
compare_prompts.py
-------------------
Runs the v1 (original) and v2 (grounded, Grad-CAM-aware) explanation
prompts side by side on the same sample of real test images, so the two
can be compared directly for the dissertation.

For each image:
    - Computes the Grad-CAM heatmap (same as the deployed pipeline)
    - Computes describe_gradcam(heatmap) — the ground-truth location/coverage
      figures now available to v2
    - Calls LLMExplainer.explain(..., version="v1") — unchanged, ungrounded
    - Calls LLMExplainer.explain(..., version="v2") — new, grounded, concise
    - Records whether v1's own stated coverage figure (if it gives one)
      matches the actual computed coverage, as a concrete measure of how
      often the old prompt was inventing a plausible-but-wrong number

Output:
    results/prompt_comparison.md   — a markdown table, ready to paste into
                                      or cite from the dissertation
    Also printed to stdout.

Usage (on HEX, same environment run_risk_demo.py uses):
    CUDA_VISIBLE_DEVICES=4 python risk_assessment/compare_prompts.py
    CUDA_VISIBLE_DEVICES=4 python risk_assessment/compare_prompts.py --n 6 --seed 42

Requires CEREBRAS_API_KEY in .env and network access to the Cerebras API —
neither is available in the assistant's sandbox, which is why this script
produces no committed output yet. Run it once locally/on hex and the
resulting results/prompt_comparison.md becomes real, citable evidence for
Chapter 3/4.
"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "interpretability"))
sys.path.insert(0, str(Path(__file__).parent.parent / "risk_assessment"))

from gradcam import GradCAM, overlay_heatmap
from model import get_model
from llm_explainer import LLMExplainer, describe_gradcam


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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def extract_stated_coverage(text: str):
    """Best-effort extraction of a percentage the LLM itself stated in its
    explanation (e.g. "approximately 50% obstructed"), so v1's invented
    figure can be checked against the real computed coverage."""
    matches = re.findall(r"(\d{1,3})\s*%", text)
    return int(matches[0]) if matches else None


def main(cfg_path, n, seed):
    if seed is not None:
        random.seed(seed)

    cfg    = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model(pretrained=False).to(device)
    ckpt  = torch.load(Path(cfg["paths"]["checkpoints"]) / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gradcam   = GradCAM(model, target_layer="layer4")
    explainer = LLMExplainer()
    transform = get_transform(cfg["data"]["img_size"])

    test_records = load_csv(str(Path(cfg["data"]["manifest_dir"]) / "test.csv"))
    blocked = [r for r in test_records if r["label"] == 0]
    samples = random.sample(blocked, min(n, len(blocked)))  # BLOCKED only — that's
    # where v1 and v2 actually differ; the CLEAR-side prompt in both versions
    # is trivial and not where the interpretability question lives.

    rows = []
    for rec in samples:
        name = Path(rec["path"]).name
        original = Image.open(rec["path"]).convert("RGB")
        tensor   = transform(original).unsqueeze(0).to(device)

        with torch.enable_grad():
            heatmap = gradcam(tensor, target="blocked")
        overlay = overlay_heatmap(original, heatmap, alpha=0.45)

        logit = model(tensor.detach()).item()
        # binary model: logit < 0 -> blocked. Reuse the same convention as inference.py.
        p_blocked = 1 / (1 + pow(2.718281828, logit))

        stats = describe_gradcam(heatmap)

        print(f"\n=== {name} ===")
        print(f"Ground truth (computed): {stats}")

        v1_text = explainer.explain(overlay, p_blocked, "BLOCKED", heatmap=heatmap, version="v1")
        print(f"\n--- v1 (original) ---\n{v1_text}")

        v2_text = explainer.explain(overlay, p_blocked, "BLOCKED", heatmap=heatmap, version="v2")
        print(f"\n--- v2 (grounded) ---\n{v2_text}")

        v1_claimed = extract_stated_coverage(v1_text)
        v1_correct = (v1_claimed is not None and abs(v1_claimed - stats["coverage_pct"]) <= 5)

        rows.append({
            "name": name,
            "actual_coverage": stats["coverage_pct"],
            "location": stats["location"],
            "v1_text": v1_text,
            "v1_claimed_coverage": v1_claimed,
            "v1_matches_ground_truth": v1_correct,
            "v2_text": v2_text,
        })

    gradcam.remove_hooks()

    # ── Write markdown table ──────────────────────────────────────────────
    out_path = Path(cfg["paths"]["logs"]).parent / "prompt_comparison.md"
    with open(out_path, "w") as f:
        f.write("# Prompt v1 vs v2 comparison (real test images, BLOCKED class)\n\n")
        f.write("| Image | Actual coverage | Actual location | v1 stated coverage | v1 matches ground truth |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['actual_coverage']}% | {r['location']} | "
                     f"{r['v1_claimed_coverage']}% | {'yes' if r['v1_matches_ground_truth'] else 'NO'} |\n")
        f.write("\n")
        for r in rows:
            f.write(f"## {r['name']}\n\n")
            f.write(f"**Ground truth (computed from the raw Grad-CAM array):** "
                     f"{r['actual_coverage']}% coverage, {r['location']}.\n\n")
            f.write(f"**v1 (original prompt):**\n\n{r['v1_text']}\n\n")
            f.write(f"**v2 (grounded prompt):**\n\n{r['v2_text']}\n\n")

    n_matched = sum(1 for r in rows if r["v1_matches_ground_truth"])
    print(f"\n{n_matched}/{len(rows)} v1 outputs stated a coverage figure within 5 points of "
          f"the actual computed value.")
    print(f"Saved: {out_path}")
    print("\nPull to local:")
    print("  rsync -avz ges65@cheery.cs.bath.ac.uk:~/dissertation/results/prompt_comparison.md "
          "~/Desktop/dissertation/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent.parent / "experiments" / "config.yaml"))
    parser.add_argument("--n", type=int, default=6, help="Number of BLOCKED test images to compare")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.n, args.seed)
