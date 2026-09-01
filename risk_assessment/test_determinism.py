"""
test_determinism.py
--------------------
Isolates whether "Scan all" giving different results run-to-run is a code
bug (same exact image bytes in, different prediction out) or the images
genuinely differing between live fetches (a model-robustness issue, not a
determinism bug).

Evidence from monitoring.db motivated this: for ~25 of 56 monitored cameras,
repeated scans that the EA site reported as the *same capture minute* still
produced wildly different p_blocked (e.g. cornwall/WadebridgePolmorla/cam1
went from p=0.9958 to p=0.0001 in two scans 65 seconds apart). But the
capture timestamp is only parsed to minute precision, so "same reported
minute" doesn't prove the two scans actually saw byte-identical images. This
script removes that ambiguity entirely by running the model on ONE loaded
image tensor, in memory, multiple times in a row. There is zero opportunity
for the input to change here — if the output probability varies at all,
that's unambiguous proof of non-determinism in the model/inference code
itself, not the data.

Self-contained on purpose — this loads the checkpoint and preprocesses the
image the same way models/finetune_problem_cameras.py does, rather than
importing app.inference (that module only exists in the deployed app repo,
not in this research repo on hex).

Usage:
    python3 risk_assessment/test_determinism.py \\
        --checkpoint results/checkpoints/best_model.pt \\
        --image extreme_cases/block/Cornwall_WadebridgePolmorla_2022_02_14_07_57.jpg \\
        --runs 10
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = 224


def load_resnet50_binary(checkpoint_path: str, device) -> tuple[nn.Module, int]:
    """Same loading logic as models/finetune_problem_cameras.py."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ck.get("model_state_dict", ck.get("state_dict", ck))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    fc_w = state_dict.get("fc.weight", state_dict.get("fc.1.weight"))
    num_out = fc_w.shape[0] if fc_w is not None else 1

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_out)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys on load: {missing}")
    return model.to(device), num_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    ap.add_argument("--image", required=True, help="Path to a single test image")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image:      {args.image}\n")

    model, num_out = load_resnet50_binary(args.checkpoint, device)
    model.eval()   # if this is missing anywhere in the real pipeline, dropout/batchnorm
                   # would behave randomly at inference time — the single most common
                   # cause of "identical input, different output" bugs like this one.

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    tensor = val_tf(image).unsqueeze(0).to(device)   # preprocessed ONCE, reused every run

    print(f"Running {args.runs} forward passes on the SAME preprocessed tensor "
          f"(no re-fetch, no re-preprocessing, nothing can change between runs).\n")

    results = []
    with torch.no_grad():
        for i in range(args.runs):
            out = model(tensor)
            if out.shape[-1] == 1:
                p_blocked = torch.sigmoid(out.squeeze(1)).item()
            else:
                p_blocked = torch.softmax(out, dim=1)[0, 0].item()   # index 0 = BLOCKED
            results.append(p_blocked)
            print(f"  run {i+1:2d}:  p_blocked={p_blocked:.6f}")

    spread = max(results) - min(results)
    print(f"\np_blocked spread across {args.runs} runs on the identical tensor: {spread:.8f}")

    if spread < 1e-6:
        print("\n=> DETERMINISTIC. Same input, same result every time. The model/pipeline")
        print("   itself is NOT the source of the run-to-run variation you're seeing in")
        print("   the app. That means 'Scan all' really is fetching a genuinely different")
        print("   live photo on each press (expected for a live camera), and the model is")
        print("   simply too sensitive to the small real differences between those photos")
        print("   — a robustness/calibration issue, not a determinism bug.")
    else:
        print("\n=> NON-DETERMINISTIC. Identical input produced different predictions.")
        print("   This is a real bug — something in the model or its inference setup")
        print("   (dropout/batchnorm not fully in eval mode, a non-deterministic CUDA")
        print("   op, or similar) is injecting randomness even with model.eval() and")
        print("   torch.no_grad() set. Needs fixing before any retrain is trustworthy.")


if __name__ == "__main__":
    main()
