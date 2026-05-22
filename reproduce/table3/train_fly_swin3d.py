"""
train_fly_swin3d.py — Fly Copulation Video Swin-T Training
Stratified Val Split + MLP Head + CE Loss (No Data Efficiency Ratio)

Usage:
    python train_fly_swin3d.py --seed 123
    python train_fly_swin3d.py --seed 1337
    python train_fly_swin3d.py --seed 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from tqdm.auto import tqdm
from collections import Counter
import decord
from decord import VideoReader, cpu
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import ToTensor
import random
import json
import time
import argparse
import matplotlib
matplotlib.use("Agg")

# ============================== Configuration ==============================

ALL_BEHAVIOR_NAMES = [
    "wing_extension", "circle", "copul_attempt", "copulation", "others",
]
SELECTED_BEHAVIORS = ["wing_extension", "circle", "copulation", "others"]
RARE_BEHAVIOR_NAMES = ["copulation"]

# Build remapping: unselected behaviors → others
OTHERS_IDX = SELECTED_BEHAVIORS.index("others")
ORIGINAL_TO_NEW = {}
for oi, name in enumerate(ALL_BEHAVIOR_NAMES):
    if name in SELECTED_BEHAVIORS:
        ORIGINAL_TO_NEW[oi] = SELECTED_BEHAVIORS.index(name)
    else:
        ORIGINAL_TO_NEW[oi] = OTHERS_IDX

NUM_CLASSES = len(SELECTED_BEHAVIORS)
RARE_CLASSES = [SELECTED_BEHAVIORS.index(b) for b in RARE_BEHAVIOR_NAMES if b in SELECTED_BEHAVIORS]

DEFAULTS = dict(
    train_video_dir="data/fly/videos/train",
    train_label_dir="data/fly/labels/train",
    batch_size=8,
    accumulation_steps=2,
    num_epochs=5,
    base_lr=3.8e-5,
    weight_decay=0.01,
    num_workers=8,
    use_class_weights=False,
    validation_ratio=0.15,
    min_behavior_threshold=50,
    aug_blur=True,
    aug_blur_frac=0.35,
    aug_td=True,
    aug_td_frac=0.15,
    window_size=16,
    stride=4,
    skip=0,
    T=8,
    mlp_hidden_dim=512,
    mlp_dropout=0.3,
    seed=123,
    val_split_seed=123,
    model_save_dir="checkpoints/fly_swin3d",
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Video Swin-T on Fly Copulation")
    p.add_argument("--train_video_dir", type=str, default=DEFAULTS["train_video_dir"])
    p.add_argument("--train_label_dir", type=str, default=DEFAULTS["train_label_dir"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--accumulation_steps", type=int, default=DEFAULTS["accumulation_steps"])
    p.add_argument("--num_epochs", type=int, default=DEFAULTS["num_epochs"])
    p.add_argument("--base_lr", type=float, default=DEFAULTS["base_lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--validation_ratio", type=float, default=DEFAULTS["validation_ratio"])
    p.add_argument("--min_behavior_threshold", type=int, default=DEFAULTS["min_behavior_threshold"])
    p.add_argument("--window_size", type=int, default=DEFAULTS["window_size"])
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"])
    p.add_argument("--skip", type=int, default=DEFAULTS["skip"])
    p.add_argument("--mlp_hidden_dim", type=int, default=DEFAULTS["mlp_hidden_dim"])
    p.add_argument("--mlp_dropout", type=float, default=DEFAULTS["mlp_dropout"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                   help="General seed (model init, augmentation, shuffle)")
    p.add_argument("--val_split_seed", type=int, default=DEFAULTS["val_split_seed"],
                   help="Validation split seed (FIXED across experiments)")
    p.add_argument("--model_save_dir", type=str, default=DEFAULTS["model_save_dir"])
    return p.parse_args()


# ============================== Utilities ==============================

def filter_and_remap_labels(labels_oh):
    orig = np.argmax(labels_oh, axis=1)
    return np.array([ORIGINAL_TO_NEW[l] for l in orig], dtype=np.int64)


def get_video_and_label_paths(video_dir, label_dir):
    vps, lps = [], []
    for name in sorted(os.listdir(video_dir)):
        if not name.lower().endswith(".mp4"):
            continue
        vp = os.path.join(video_dir, name)
        lp = os.path.join(label_dir, name.replace(".mp4", ".csv"))
        if os.path.exists(lp):
            vps.append(vp)
            lps.append(lp)
        else:
            print(f"[WARN] Label not found: {name}")
    return vps, lps


def analyze_and_split(video_paths, label_paths, test_size, min_thresh, random_state):
    """Stratified video split by rare behavior presence."""
    profiles = []
    for vp, lp in zip(video_paths, label_paths):
        try:
            df = pd.read_csv(lp)
            labels = filter_and_remap_labels(df.iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values)
            counts = {c: int(np.sum(labels == c)) for c in range(NUM_CLASSES)}
            has_rare = any(counts[c] >= min_thresh for c in RARE_CLASSES)
            profiles.append({"vp": vp, "lp": lp, "counts": counts,
                             "has_rare": has_rare, "name": os.path.basename(vp)})
        except Exception as e:
            print(f"⚠️ {vp}: {e}")
    strat = [1 if p["has_rare"] else 0 for p in profiles]
    _, cnts = np.unique(strat, return_counts=True)
    use_strat = min(cnts) >= 2
    indices = list(range(len(profiles)))
    try:
        if use_strat:
            tr, va = train_test_split(indices, test_size=test_size,
                                      random_state=random_state, stratify=strat)
        else:
            tr, va = train_test_split(indices, test_size=test_size,
                                      random_state=random_state)
    except ValueError:
        tr, va = train_test_split(indices, test_size=test_size,
                                  random_state=random_state)
    print(f"  Split: {len(tr)} train / {len(va)} val")
    for c in range(NUM_CLASSES):
        tc = sum(profiles[i]["counts"][c] for i in tr)
        vc = sum(profiles[i]["counts"][c] for i in va)
        print(f"    {SELECTED_BEHAVIORS[c]:>15}: train={tc}, val={vc}")
    print()
    return ([profiles[i]["vp"] for i in tr], [profiles[i]["lp"] for i in tr],
            [profiles[i]["vp"] for i in va], [profiles[i]["lp"] for i in va])


def random_blur_frames(frames, frac=0.35, radius_range=(0.8, 2.2), rng=None):
    if frac <= 0 or not frames:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = rng.sample(range(n), k)
    rmin, rmax = radius_range
    return [frames[i].filter(ImageFilter.GaussianBlur(radius=rng.uniform(rmin, rmax)))
            if i in idxs else frames[i] for i in range(n)]


def random_temporal_dropout(frames, frac=0.15, rng=None):
    if frac <= 0 or len(frames) < 3:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = rng.sample(range(1, n - 1), min(k, max(1, n - 2)))
    out = frames[:]
    for i in idxs:
        out[i] = out[i - 1] if rng.random() < 0.5 else out[i + 1]
    return out


def custom_video_transform(frames, target_size=(224, 224)):
    frames = [f.resize(target_size, Image.BILINEAR) for f in frames]
    frames = [ToTensor()(f) for f in frames]
    video = torch.stack(frames, dim=1)  # (C, T, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
    return (video - mean) / std


# ============================== Dataset ==============================

class SlidingWindowVideoDataset(Dataset):
    def __init__(self, vps, lps, window_size, stride, transform, skip=0, augment=None):
        self.vps, self.lps = vps, lps
        self.window_size, self.stride, self.skip = window_size, stride, skip
        self.transform, self.augment = transform, augment
        self.samples, self.sample_labels = self._gen()

    def _gen(self):
        samples, labels = [], []
        for vp, lp in zip(self.vps, self.lps):
            df = pd.read_csv(lp)
            oh = df.iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(oh) != T:
                print(f"⚠️ Mismatch: {os.path.basename(vp)}")
                continue
            remapped = filter_and_remap_labels(oh)
            sel = list(range(0, T, self.skip + 1))
            fcls = remapped[sel]
            for s in range(0, len(sel) - self.window_size + 1, self.stride):
                wi = sel[s:s + self.window_size]
                wl = Counter(fcls[s:s + self.window_size]).most_common(1)[0][0]
                samples.append((vp, wi, wl))
                labels.append(wl)
        return samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, fi, label = self.samples[idx]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(fi).asnumpy()]
        if len(frames) < self.window_size:
            frames.extend([frames[-1]] * (self.window_size - len(frames)))
        if self.augment:
            frames = self.augment(frames)
        return self.transform(frames), torch.tensor(label, dtype=torch.long)


# ============================== Model ==============================

class MLPHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class CustomSwin3D(nn.Module):
    def __init__(self, pretrained=True, T=8):
        super().__init__()
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.model = swin3d_t(weights=weights)
        self.T = T
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.features(x)
        x = self.model.norm(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def build_model(num_classes, hidden_dim=512, dropout=0.3):
    model = CustomSwin3D(pretrained=True, T=DEFAULTS["T"])
    model.head = MLPHead(768, num_classes, hidden_dim, dropout)
    return model


# ============================== Evaluation ==============================

def evaluate_window_level(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Validation", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            with autocast():
                logits = model(videos)
                probs = torch.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    f1_pc = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    f1_m = f1_score(all_labels, all_preds, average="macro")

    oh = np.zeros((len(all_labels), NUM_CLASSES))
    for i, l in enumerate(all_labels):
        oh[i, l] = 1.0
    all_probs = np.array(all_probs)
    ap_pc = np.array([
        average_precision_score(oh[:, c], all_probs[:, c]) if oh[:, c].sum() > 0 else 0.0
        for c in range(NUM_CLASSES)])

    return {"f1_per_class": f1_pc, "f1_macro": f1_m,
            "ap_per_class": ap_pc, "mAP": np.mean(ap_pc)}


# ============================== Main ==============================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_save_dir, exist_ok=True)
    run_tag = f"fly_swin3d_seed{args.seed}_vsplit{args.val_split_seed}"

    print(f"\n{'='*70}")
    print(f"Fly Copulation — Swin3D + MLP Head")
    print(f"{'='*70}")
    print(f"  Classes: {SELECTED_BEHAVIORS} ({NUM_CLASSES})")
    print(f"  Excluded → others: {[b for b in ALL_BEHAVIOR_NAMES if b not in SELECTED_BEHAVIORS]}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed} | Val split: {args.val_split_seed} (FIXED)")
    print(f"{'='*70}\n")

    # Load + split
    all_vids, all_labs = get_video_and_label_paths(args.train_video_dir, args.train_label_dir)
    print(f"Total videos: {len(all_vids)}")
    train_v, train_l, val_v, val_l = analyze_and_split(
        all_vids, all_labs, args.validation_ratio, args.min_behavior_threshold, args.val_split_seed)
    print(f"Training videos: {len(train_v)}")
    print(f"Validation videos: {len(val_v)} (FIXED)\n")

    # Datasets
    aug_rng = random.Random(args.seed)
    def augment(frames):
        if DEFAULTS["aug_blur"]:
            frames = random_blur_frames(frames, DEFAULTS["aug_blur_frac"], rng=aug_rng)
        if DEFAULTS["aug_td"]:
            frames = random_temporal_dropout(frames, DEFAULTS["aug_td_frac"], rng=aug_rng)
        return frames

    train_ds = SlidingWindowVideoDataset(train_v, train_l, args.window_size, args.stride,
                                         custom_video_transform, args.skip, augment)
    val_ds = SlidingWindowVideoDataset(val_v, val_l, args.window_size, args.stride,
                                       custom_video_transform, args.skip, None)

    counts = Counter(train_ds.sample_labels)
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")
    print(f"  {'Behavior':>15}  {'Windows':>8}  {'%':>7}")
    total_win = sum(counts.values())
    for c in range(NUM_CLASSES):
        wc = counts.get(c, 0)
        print(f"  {SELECTED_BEHAVIORS[c]:>15}  {wc:>8}  {100*wc/total_win if total_win else 0:>6.2f}%")
    print()

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(NUM_CLASSES, args.mlp_hidden_dim, args.mlp_dropout).to(device)

    if args.use_class_weights:
        w = np.array([counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=np.float32)
        w = w.sum() / (NUM_CLASSES * w); w /= w.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler()

    log, best_f1, best_path = [], -1.0, None

    for epoch in range(args.num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for bi, (vids, tgts) in enumerate(pbar):
            vids, tgts = vids.to(device), tgts.to(device)
            with autocast():
                loss = criterion(model(vids), tgts) / args.accumulation_steps
            scaler.scale(loss).backward()
            if (bi + 1) % args.accumulation_steps == 0 or (bi + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.accumulation_steps * vids.size(0)
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)

        scheduler.step()
        train_time = time.time() - t0
        epoch_loss = running_loss / len(train_ds)

        t1 = time.time()
        m = evaluate_window_level(model, val_loader, device)
        val_time = time.time() - t1

        print(f"\nEpoch {epoch+1} | Loss: {epoch_loss:.4f}")
        print(f"  Val F1-macro: {m['f1_macro']:.4f} | Val mAP: {m['mAP']:.4f}")
        for n, v in zip(SELECTED_BEHAVIORS, m["f1_per_class"]):
            print(f"    {n:>15} F1: {v:.4f}")
        print(f"  Train: {train_time:.1f}s | Val: {val_time:.1f}s")

        ckpt = os.path.join(args.model_save_dir,
                            f"{run_tag}_ep{epoch+1}_f1_{m['f1_macro']:.4f}_map_{m['mAP']:.4f}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"  💾 {ckpt}")

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_path = ckpt
            print(f"  🏆 New best! F1={best_f1:.4f}")

        log.append({
            "epoch": epoch + 1, "train_loss": epoch_loss,
            "val_f1": m["f1_macro"], "val_map": m["mAP"],
            "val_f1_per_class": m["f1_per_class"].tolist(),
            "val_ap_per_class": m["ap_per_class"].tolist(),
            "train_time": train_time, "val_time": val_time,
        })
        print()

    best_ep = max(log, key=lambda x: x["val_f1"])
    print(f"\n✅ Best: Epoch {best_ep['epoch']} | F1={best_ep['val_f1']:.4f} | mAP={best_ep['val_map']:.4f}")
    print(f"   Model: {best_path}")

    log_path = os.path.join(args.model_save_dir, f"training_log_{run_tag}.json")
    with open(log_path, "w") as f:
        json.dump({
            "training_log": log, "best_epoch": best_ep, "best_model_path": best_path,
            "config": {**vars(args), "num_classes": NUM_CLASSES,
                       "selected_behaviors": SELECTED_BEHAVIORS,
                       "all_behavior_names": ALL_BEHAVIOR_NAMES,
                       "others_class_idx": OTHERS_IDX,
                       "model": "Video Swin-T + MLP Head"},
            "split_info": {
                "train_videos": [os.path.basename(v) for v in train_v],
                "val_videos": [os.path.basename(v) for v in val_v],
            },
        }, f, indent=2)
    print(f"💾 Log: {log_path}\n")


if __name__ == "__main__":
    main()
