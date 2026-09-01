"""
finetune_problem_cameras.py
---------------------------
Fine-tune best_model.pt (ResNet-50 binary) on cameras with real labeled
blocked+clear data on hex, confirmed via risk_assessment/audit_finetune_candidates.py
(2026-07-07). Expanded from 2 to 20 cameras at the user's request ("fine
tune for all cases you can, from the 56 cameras") — every folder below is a
CONFIDENT name match to a real live camera in app/main.py's _WEBCAM_LIST with
>=15 labeled images in both classes. Folders that exist on hex but did NOT
get a confident match to a live campath (Cornwall_BoscastleMarine,
Cornwall_Crinnis, Cornwall_IdlessDam_cam1, Cornwall_PlymptonKa,
Cornwall_PolperroLangreek, Devon_Holbeam_Impound, Devon_PalmersDam_Upstream)
and the generic sites_* folders (not part of the 56-camera live fleet at
all) are deliberately excluded — verify those manually before adding them.

Strategy:
  Phase 1 (5 ep)  — freeze backbone, train head only (LR=1e-3)
  Phase 2 (15 ep) — unfreeze all, full fine-tune at very low LR (5e-6)

Also fixes a real bug found alongside this expansion: train_tf/val_tf used
to Resize((SIZE, SIZE)) — a hard squash to square with NO aspect-ratio
preservation. Real aspect ratios across these cameras range from 1.22 to
1.78 (confirmed via the audit script), so different cameras' images were
being distorted by very different amounts before the model ever saw them.

FIRST attempt at this fix used Resize(SIZE) + CenterCrop(SIZE) (scale the
shorter edge, crop the rest to a square) — this trained (best_model_ft_v2.pt,
val F1_blocked=0.993) but FAILED live validation: on Chaddlewood (1920x1080,
the widest aspect ratio in the set, 1.78) it confidently predicted CLEAR
(p_blocked=0.016) on a live photo of a site independently confirmed to be
persistently, severely blocked — a complete flip from the general model's
correct p_blocked=1.000 on the same photo. Center-cropping a 1.78-ratio
frame down to square discards ~44% of its width; the most likely explanation
is the crop was cutting the actual blockage out of frame on this camera.

FIX: use letterbox padding (LetterboxResize below) instead of center-crop —
scales the image to fit WITHIN size x size (longer edge -> size), then pads
the shorter edge with mid-grey. This keeps the ENTIRE original frame
visible; nothing gets cropped out, at the cost of some non-informative
padding pixels around the edges for non-square cameras.

IMPORTANT: because the preprocessing changed, app/inference.py's
InferencePipeline must load this checkpoint with preserve_aspect=True to
match (that flag now means letterbox, not center-crop — see the comment in
_build_transform) — and the _SPECIALIST_CAMERAS wiring in app/main.py should
NOT be updated to route to whatever --out path this produces until it's
been validated the same way (real live photos of known cases, not just the
held-out F1 number) that caught the center-crop problem in the first place.

v4 (2026-07-09): PROBLEM_CAMERAS cut back down from the 20-camera v2/v3 batch
to just 3 — Cornwall_BudeCedarGrove, Cornwall_PlymptonChaddlewood_MainScree,
Devon_BarnstapleBradiford — the only cameras with a healthy count of BOTH
classes confirmed still-current in images/ (the other 4 synced cameras,
LauncestonWooda/Porthallow/NewtonAbbotBakersPark at 1 image each and
StIvesConsols at 0 blocked ever, aren't usable). This follows the same
smaller/homogeneous-batch reasoning that made the original 2-camera
best_model_ft.pt (F1=0.985) reliable, vs. the two 20-camera batches that
both had real problems in practice (v2 center-crop failure, v3 real-world
"too many blocked" regression never root-caused). Do NOT re-expand this list
back to 20 without re-validating on live photos first, same as v2/v3.

v5 (2026-07-09, later same day): v4 (all available data for BudeCedarGrove +
Chaddlewood + BarnstapleBradiford) passed live-pair validation but still made
real mistakes in production. Specialist routing was fully removed from
app/main.py as a result. This v5 run is a deliberately different experiment,
not a repeat of v4: scoped to just 2 cameras (Chaddlewood, BarnstapleBradiford
— BudeCedarGrove dropped), and trained on a randomly-sampled 100
blocked / 100 clear per camera (models/../finetune_v5_sample/) instead of
every available image, on the hypothesis that the full archival dataset is
mostly near-duplicate consecutive frames that inflate held-out F1 without
adding real diversity. This is UNVALIDATED — do not treat a good F1 here as
a green light; the exact same thing happened with v4. Root-cause v4's actual
production mistakes before trusting this one either, and do not wire
_SPECIALIST_CAMERAS back on without fresh live-pair validation.

Usage:
  CUDA_VISIBLE_DEVICES=6 nohup python3 models/finetune_problem_cameras.py \
    --checkpoint results/checkpoints/best_model.pt \
    --images ~/dissertation/finetune_v5_sample \
    --out results/checkpoints/best_model_ft_v5.pt \
    --workers 2 \
    > logs/finetune_v5.log 2>&1 &

(best_model_ft_v5.pt — v2 was the failed center-crop version, v3 the
20-camera letterbox version that regressed live, v4 the 3-camera version
that also regressed live despite passing validation. All kept on disk as a
record of what not to trust, not deleted. best_model_ft.pt, the original
2-camera version, is the only specialist checkpoint with a track record that
held up — but even it was withdrawn today, not because it broke, just as
part of removing the whole specialist-routing feature.)
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score


class LetterboxResize:
    """
    Resize an image to fit WITHIN (size, size) while preserving its aspect
    ratio — the longer edge becomes exactly `size`, the shorter edge scales
    proportionally — then pads the shorter edge with solid grey to make the
    final result exactly size x size.

    Unlike a center-crop, this keeps 100% of the original frame visible.
    That matters here: a center-crop on a 1.78-ratio camera (the widest in
    this dataset, e.g. Chaddlewood at 1920x1080) discards close to 44% of
    the frame's width, and empirically that crop appears to have cut the
    actual blockage out of view on at least one camera — the center-crop
    version of this script's checkpoint confidently predicted CLEAR on a
    site independently confirmed to be persistently, severely blocked.
    Letterbox padding trades a border of non-informative grey pixels for
    never losing real content, which is the safer failure mode here.
    """

    def __init__(self, size: int, fill=(114, 114, 114)):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = ((self.size - new_w) // 2, (self.size - new_h) // 2)
        canvas.paste(resized, offset)
        return canvas

# ── Problem cameras ────────────────────────────────────────────────────────────
#
# Confidence counts and native image size confirmed via
# risk_assessment/audit_finetune_candidates.py on hex, 2026-07-07:

PROBLEM_CAMERAS = [
    "Cornwall_PlymptonChaddlewood_MainScree",  # v5: 100 blocked / 100 clear curated sample
    "Devon_BarnstapleBradiford",                # v5: 100 blocked / 100 clear curated sample
    # BudeCedarGrove dropped for v5 — not part of this curated 2-camera
    # experiment (it vanished from the local images/ mirror; not in
    # finetune_v5_sample/). Re-add only with its own deliberate sample if
    # revisited later.

    # ── Previous (v2/v3) 20-camera batch — kept here, commented out, as a
    # record. Not currently used: v2 (center-crop) failed live on Chaddlewood,
    # v3 (letterbox, this same list) regressed live with "too many blocked"
    # never root-caused. Re-enable only after re-validating on live photos,
    # camera-by-camera or in small groups — not all 20 at once.
    # "Cornwall_WadebridgePolmorla",             # 329 blocked, 132 clear
    # "Cornwall_BodminPetrocsWell_Scree",        # 120 blocked, 1188 clear
    # "Cornwall_KingsandCP",                     # 337 blocked, 380 clear
    # "Cornwall_LostwithielUP_Scree",             # 50 blocked, 1303 clear
    # "Cornwall_Mevagissey_PreScree",             # 541 blocked, 657 clear
    # "Cornwall_PenrynTP",                        # 22 blocked, 745 clear
    # "Cornwall_PenzanceCC",                      # 512 blocked, 542 clear
    # "Cornwall_PlymptonForSt",                   # 38 blocked, 1203 clear
    # "Cornwall_PorthlevenScree",                 # 527 blocked, 204 clear
    # "Cornwall_Portreath",                       # 241 blocked, 798 clear — the never-auto-block camera
    # "Cornwall_TamertonFoliot",                  # 674 blocked, 231 clear
    # "Devon_BarnstapleConeyGut_Scree",           # 596 blocked, 795 clear
    # "Devon_BarnstaplePortmarshLane",            # 197 blocked, 150 clear
    # "Devon_Buckfastleigh",                      # 419 blocked, 1732 clear
    # "Devon_KenwithValleyChannelScree",          # 193 blocked, 1035 clear
    # "Devon_LympstoneScree",                     # 303 blocked, 751 clear
    # "Devon_SwimbridgeScree",                    # 24 blocked, 593 clear

    # Excluded — no confident match to a live campath in _WEBCAM_LIST, or
    # (sites_*) not part of the 56-camera live fleet at all.
    # "Cornwall_BoscastleMarine", "Cornwall_Crinnis", "Cornwall_IdlessDam_cam1",
    # "Cornwall_PlymptonKa", "Cornwall_PolperroLangreek",
    # "Devon_Holbeam_Impound", "Devon_PalmersDam_Upstream"

    # Still genuinely no usable data (checked 2026-07-07/08):
    #   Cornwall_StIvesConsols_Scree — 1559 clear, 0 blocked ever
    #   Cornwall_Porthallow, Cornwall_LauncestonWooda, Devon_AshburtonLower,
    #   Devon_KingsbridgeDuncombe, Devon_NewtonAbbotBakersPark — 1 image or none
]

# ── Hyperparameters ────────────────────────────────────────────────────────────

IMG_SIZE     = 224
BATCH        = 32
PHASE1_LR    = 1e-3
PHASE2_LR    = 5e-6
PHASE1_EP    = 5
PHASE2_EP    = 15
PATIENCE     = 5
FOCAL_GAMMA  = 2.0

# ── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight, reduction="none"
        )
        p_t   = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()

# ── Dataset ────────────────────────────────────────────────────────────────────

class CameraDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def collect_samples(images_root: Path, cameras: list) -> list:
    samples = []
    for cam in cameras:
        cam_dir = images_root / cam
        if not cam_dir.exists():
            print(f"  [warn] {cam_dir} not found — skipping")
            continue
        for label, cls in [(1, "blocked"), (0, "clear")]:
            cls_dir = cam_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    samples.append((str(p), label))
        b = sum(1 for _, l in samples if l == 1)
        c = sum(1 for _, l in samples if l == 0)
        print(f"  {cam}: {b} blocked, {c} clear so far")
    return samples

# ── Model ──────────────────────────────────────────────────────────────────────

def load_resnet50_binary(checkpoint_path: str, device) -> nn.Module:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ck.get("model_state_dict", ck.get("state_dict", ck))
    # Strip DataParallel prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Detect output size from FC layer
    fc_w = state_dict.get("fc.weight", state_dict.get("fc.1.weight"))
    num_out = fc_w.shape[0] if fc_w is not None else 1
    print(f"  Checkpoint FC output size: {num_out}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_out)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    return model.to(device), num_out

# ── Train / eval ───────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, scaler, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out  = model(imgs)
                # Handle both single-logit (num_out=1) and two-logit (num_out=2) heads
                if out.shape[-1] == 1:
                    logits = out.squeeze(1)
                    probs  = torch.sigmoid(logits)
                else:
                    logits = out[:, 1] - out[:, 0]   # log-odds for BLOCKED
                    probs  = torch.softmax(out, dim=1)[:, 1]
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(labels)
            preds = (probs >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().tolist())

    n   = len(loader.dataset)
    f1b = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1c = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    return total_loss / n, f1b, f1c

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      required=True,        help="Path to best_model.pt")
    parser.add_argument("--images",          required=True,        help="Root images directory")
    parser.add_argument("--out",             required=True,        help="Output checkpoint path")
    parser.add_argument("--phase1-epochs",   type=int, default=PHASE1_EP)
    parser.add_argument("--phase2-epochs",   type=int, default=PHASE2_EP)
    parser.add_argument("--batch",           type=int, default=BATCH)
    parser.add_argument("--workers",         type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    images_root = Path(args.images).expanduser()
    samples     = collect_samples(images_root, PROBLEM_CAMERAS)
    random.seed(42)
    random.shuffle(samples)

    n_blocked = sum(1 for _, l in samples if l == 1)
    n_clear   = sum(1 for _, l in samples if l == 0)
    print(f"\nTotal: {len(samples)}  |  blocked={n_blocked}  clear={n_clear}")

    split        = int(0.8 * len(samples))
    train_samps  = samples[:split]
    val_samps    = samples[split:]

    # Aspect-ratio-PRESERVING resize: Resize() with a single int (not a tuple)
    # scales the shorter edge to that size and keeps the original proportions,
    # then Crop takes a square from the result. This replaces the previous
    # Resize((SIZE, SIZE)) tuple form, which squashed every image to a square
    # regardless of its native shape — real aspect ratios across these
    # cameras range from 1.22 to 1.78 (confirmed via
    # risk_assessment/audit_finetune_candidates.py), so a 16:9 camera and a
    # 4:3 camera were being distorted by very different amounts before the
    # model ever saw them. That's a real, camera-dependent source of
    # inconsistency this fine-tune corrects for its own data.
    # Letterbox, not center-crop — keeps the entire original frame visible
    # (see LetterboxResize docstring for why this replaced the crop version).
    train_tf = transforms.Compose([
        LetterboxResize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        LetterboxResize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Weighted sampler — balance blocked/clear in each batch
    labels_train  = [l for _, l in train_samps]
    class_counts  = [labels_train.count(0), labels_train.count(1)]
    sample_wts    = [1.0 / class_counts[l] for l in labels_train]
    sampler       = WeightedRandomSampler(sample_wts, num_samples=len(train_samps), replacement=True)

    loader_kw = dict(num_workers=args.workers, multiprocessing_context="spawn",
                     persistent_workers=True, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(CameraDataset(train_samps, train_tf),
                              batch_size=args.batch, sampler=sampler, **loader_kw)
    val_loader   = DataLoader(CameraDataset(val_samps, val_tf),
                              batch_size=args.batch, shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, num_out = load_resnet50_binary(args.checkpoint, device)

    # Focal loss: upweight blocked class to counter imbalance
    pos_weight = torch.tensor([n_clear / max(n_blocked, 1)], device=device)
    criterion  = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight)
    scaler     = torch.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = 0.0
    best_sd = None

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print("\n══ Phase 1: head only (backbone frozen) ══")
    for name, p in model.named_parameters():
        p.requires_grad = ("fc" in name)
    opt1     = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR)
    patience = 0

    for ep in range(1, args.phase1_epochs + 1):
        tr_loss, tr_f1b, tr_f1c = run_epoch(model, train_loader, opt1, criterion, device, scaler, True)
        vl_loss, vl_f1b, vl_f1c = run_epoch(model, val_loader,   opt1, criterion, device, scaler, False)
        print(f"  ep{ep:02d}  train loss={tr_loss:.4f} F1_blocked={tr_f1b:.3f} F1_clear={tr_f1c:.3f}"
              f"  |  val loss={vl_loss:.4f} F1_blocked={vl_f1b:.3f} F1_clear={vl_f1c:.3f}")
        if vl_f1b > best_f1:
            best_f1 = vl_f1b
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  Early stopping.")
                break

    if best_sd:
        model.load_state_dict(best_sd)

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    print("\n══ Phase 2: full fine-tune (all layers, LR=5e-6) ══")
    for p in model.parameters():
        p.requires_grad = True
    opt2      = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.phase2_epochs)
    patience  = 0

    for ep in range(1, args.phase2_epochs + 1):
        tr_loss, tr_f1b, tr_f1c = run_epoch(model, train_loader, opt2, criterion, device, scaler, True)
        vl_loss, vl_f1b, vl_f1c = run_epoch(model, val_loader,   opt2, criterion, device, scaler, False)
        scheduler.step()
        print(f"  ep{ep:02d}  train loss={tr_loss:.4f} F1_blocked={tr_f1b:.3f} F1_clear={tr_f1c:.3f}"
              f"  |  val loss={vl_loss:.4f} F1_blocked={vl_f1b:.3f} F1_clear={vl_f1c:.3f}")
        if vl_f1b > best_f1:
            best_f1 = vl_f1b
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  Early stopping.")
                break

    # ── Save ──────────────────────────────────────────────────────────────────
    model.load_state_dict(best_sd)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "num_classes": num_out}, str(out_path))
    print(f"\n✓ Saved → {out_path}  (best val F1_blocked = {best_f1:.3f})")


if __name__ == "__main__":
    main()
