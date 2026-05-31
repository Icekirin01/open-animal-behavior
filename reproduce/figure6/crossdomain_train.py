"""
crossdomain_train.py — Cross-Domain Fine-tuning (Swin3D)
Fine-tune on a new dataset using either:
  1. Kinetics-400 pretrained backbone (default)
  2. Custom pretrained checkpoint (--pretrained_model_path)

Supports data efficiency experiments via --train_data_ratio.
Training only — use crossdomain_eval.py separately for test evaluation.

Usage:
    # From Kinetics-400 pretrained (no custom model)
    python crossdomain_train.py --seed 123 \
        --train_video_dir data/new/videos/train \
        --train_label_dir data/new/labels/train

    # From custom pretrained checkpoint
    python crossdomain_train.py --seed 123 \
        --pretrained_model_path checkpoints/fulltrain.pth \
        --base_lr 1e-5 \
        --train_video_dir data/new/videos/train \
        --train_label_dir data/new/labels/train

    # Few-shot: only use 50% of training videos
    python crossdomain_train.py --seed 123 \
        --pretrained_model_path checkpoints/fulltrain.pth \
        --train_data_ratio 0.5 --video_split_seed 42 \
        --train_video_dir data/new/videos/train \
        --train_label_dir data/new/labels/train
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
from sklearn.metrics import (f1_score, confusion_matrix, average_precision_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import ToTensor
import random
import argparse
import json
import time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================== Configuration ==============================

ALL_BEHAVIOR_NAMES = [
    'Aggression', 'Investigation', 'Allo-groom', 'Self-groom',
    'Standing', 'Chasing', 'Other'
]
SELECTED_BEHAVIORS = ['Aggression', 'Investigation', 'Allo-groom', 'Standing', 'Other']
RARE_BEHAVIOR_NAMES = ['Aggression', 'Allo-groom']


def parse_args():
    p = argparse.ArgumentParser(description="Cross-domain fine-tuning: Swin3D on a new dataset")

    # Model init
    p.add_argument("--pretrained_model_path", type=str, default=None,
                    help="Path to custom pretrained .pth. If omitted, uses Kinetics-400 weights.")

    # Seeds
    p.add_argument("--seed", type=int, default=2025, help="General seed")
    p.add_argument("--val_split_seed", type=int, default=1337,
                    help="Val split seed (FIXED across experiments)")
    p.add_argument("--video_split_seed", type=int, default=42,
                    help="Seed for subsampling training videos (used with --train_data_ratio)")

    # Data paths
    p.add_argument("--train_video_dir", type=str, required=True)
    p.add_argument("--train_label_dir", type=str, required=True)
    p.add_argument("--model_save_dir", type=str, default="checkpoints/crossdomain")

    # Data efficiency
    p.add_argument("--train_data_ratio", type=float, default=1.0,
                    help="Fraction of training videos to use (1.0=all, 0.5=50%%)")
    p.add_argument("--validation_ratio", type=float, default=0.2,
                    help="Fraction of training videos for validation")
    p.add_argument("--min_behavior_threshold", type=int, default=30)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--base_lr", type=float, default=3.8e-5,
                    help="Learning rate (use ~1e-5 for custom pretrained, ~3.8e-5 for K400)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_class_weights", action="store_true")

    # Model
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_dropout", type=float, default=0.3)
    p.add_argument("--smooth_window_size", type=int, default=1)
    return p.parse_args()


# ============================== Behavior Mapping ==============================

def build_behavior_mapping(selected):
    o2n = {}
    for oi, name in enumerate(ALL_BEHAVIOR_NAMES):
        if name in selected:
            o2n[oi] = selected.index(name)
        else:
            o2n[oi] = None
    return o2n


def filter_and_remap_labels(labels_oh, o2n, num_classes):
    orig = np.argmax(labels_oh, axis=1)
    remapped = np.full(len(orig), -1, dtype=np.int64)
    valid = np.zeros(len(orig), dtype=bool)
    for i, o in enumerate(orig):
        new = o2n.get(o, None)
        if new is not None:
            remapped[i] = new
            valid[i] = True
    return remapped, valid


# ============================== Utilities ==============================

def get_video_and_label_paths(video_dir, label_dir):
    vps, lps = [], []
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(".mp4"):
            lp = os.path.join(label_dir, f.replace(".mp4", ".csv"))
            if os.path.exists(lp):
                vps.append(os.path.join(video_dir, f))
                lps.append(lp)
            else:
                print(f"[WARN] No label for {f}")
    return vps, lps


def subsample_videos(vps, lps, ratio, rng_seed=42):
    if ratio >= 1.0:
        return vps, lps
    n = len(vps)
    k = max(1, int(round(n * ratio)))
    rng = np.random.RandomState(rng_seed)
    idxs = sorted(rng.choice(n, size=k, replace=False).tolist())
    return [vps[i] for i in idxs], [lps[i] for i in idxs]


def custom_video_transform(frames):
    tensors = [ToTensor()(f) for f in frames]
    video = torch.stack(tensors, dim=1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
    return (video - mean) / std


def random_blur_frames(frames, frac=0.35, radius_range=(0.8, 2.2), rng=None):
    if frac <= 0 or not frames:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = set(rng.sample(range(n), k))
    rmin, rmax = radius_range
    return [im.filter(ImageFilter.GaussianBlur(rng.uniform(rmin, rmax))) if i in idxs else im
            for i, im in enumerate(frames)]


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


def plot_confusion_matrix(cm, names, path, title="CM", normalize=False):
    if normalize:
        cm_disp = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_disp = np.nan_to_num(cm_disp)
        fmt = ".1%"
    else:
        cm_disp, fmt = cm, "d"
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=names, yticklabels=names, square=True)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")


# ============================== Stratified Split ==============================

def stratified_video_split(vps, lps, o2n, num_classes, behavior_names, rare_classes,
                           test_size=0.2, min_threshold=30, random_state=1337):
    print(f"\n{'='*60}")
    print(f"Stratified Video Split (val_seed={random_state}, FIXED)")
    print(f"{'='*60}")

    profiles = []
    for vp, lp in tqdm(zip(vps, lps), total=len(vps), desc="Profiling"):
        df = pd.read_csv(lp)
        labels_oh = df.iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values
        remapped, valid = filter_and_remap_labels(labels_oh, o2n, num_classes)
        counts = {c: int(np.sum((remapped == c) & valid)) for c in range(num_classes)}
        has_rare = rare_classes and any(counts[c] >= min_threshold for c in rare_classes)
        profiles.append({"vp": vp, "lp": lp, "name": os.path.basename(vp),
                         "has_rare": has_rare, "counts": counts})

    for c in range(num_classes):
        tf = sum(p["counts"][c] for p in profiles)
        vf = sum(1 for p in profiles if p["counts"][c] >= min_threshold)
        print(f"  {behavior_names[c]:>15}  {tf:>8} frames, {vf:>3} videos (>={min_threshold})")

    strat = [1 if p["has_rare"] else 0 for p in profiles]
    idxs = list(range(len(profiles)))
    _, cnts = np.unique(strat, return_counts=True)
    use_strat = min(cnts) >= 2

    try:
        train_idx, val_idx = train_test_split(
            idxs, test_size=test_size, random_state=random_state,
            stratify=strat if use_strat else None)
    except ValueError:
        train_idx, val_idx = train_test_split(idxs, test_size=test_size, random_state=random_state)

    tv = [profiles[i]["vp"] for i in train_idx]
    tl = [profiles[i]["lp"] for i in train_idx]
    vv = [profiles[i]["vp"] for i in val_idx]
    vl = [profiles[i]["lp"] for i in val_idx]
    print(f"\n  Train: {len(tv)} videos | Val: {len(vv)} videos")
    print(f"{'='*60}\n")
    return tv, tl, vv, vl


# ============================== Datasets ==============================

class SlidingWindowVideoDataset(Dataset):
    def __init__(self, vps, lps, window_size, stride, transform, skip, augment,
                 o2n, num_classes):
        self.vps, self.lps = vps, lps
        self.window_size, self.stride, self.skip = window_size, stride, skip
        self.transform, self.augment = transform, augment
        self.o2n, self.num_classes = o2n, num_classes
        self.samples, self.sample_labels = self._gen()

    def _gen(self):
        samples, labels = [], []
        for vp, lp in zip(self.vps, self.lps):
            df = pd.read_csv(lp)
            oh = df.iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(oh) != T:
                continue
            rem, valid = filter_and_remap_labels(oh, self.o2n, self.num_classes)
            sel = [i for i in range(0, T, self.skip + 1) if i < len(valid) and valid[i]]
            if len(sel) < self.window_size:
                continue
            for s in range(0, len(sel) - self.window_size + 1, self.stride):
                win = sel[s:s + self.window_size]
                lbl = Counter(rem[win]).most_common(1)[0][0]
                samples.append((vp, win, lbl))
                labels.append(lbl)
        return samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, fi, lbl = self.samples[idx]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(fi).asnumpy()]
        if len(frames) < self.window_size:
            frames.extend([frames[-1]] * (self.window_size - len(frames)))
        if self.augment:
            frames = self.augment(frames)
        return self.transform(frames), torch.tensor(lbl, dtype=torch.long)


class WindowPredictionDataset(Dataset):
    def __init__(self, vps, lps, window_size, stride, transform, skip, o2n, num_classes):
        self.vps, self.lps = vps, lps
        self.window_size, self.stride, self.skip = window_size, stride, skip
        self.transform = transform
        self.o2n, self.num_classes = o2n, num_classes
        self.windows, self.frame_mappings = self._gen()

    def _gen(self):
        windows, mappings = [], []
        for vp, lp in zip(self.vps, self.lps):
            df = pd.read_csv(lp)
            oh = df.iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(oh) != T:
                continue
            rem, valid = filter_and_remap_labels(oh, self.o2n, self.num_classes)
            sel = [i for i in range(0, T, self.skip + 1) if i < len(valid) and valid[i]]
            if len(sel) < self.window_size:
                continue
            sel_oh = np.zeros((len(sel), self.num_classes))
            for i, fi in enumerate(sel):
                sel_oh[i, rem[fi]] = 1.0
            f2w = [[] for _ in range(len(sel))]
            for s in range(0, len(sel) - self.window_size + 1, self.stride):
                win = sel[s:s + self.window_size]
                windows.append((vp, win))
                widx = len(windows) - 1
                for fi in range(s, s + self.window_size):
                    if fi < len(f2w):
                        f2w[fi].append(widx)
            mappings.append({"labels": sel_oh, "frame_to_windows": f2w})
        return windows, mappings

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        vp, fi = self.windows[idx]
        vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(fi).asnumpy()]
        if len(frames) < self.window_size:
            frames.extend([frames[-1]] * (self.window_size - len(frames)))
        return self.transform(frames), idx


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
        return self.fc2(self.dropout(self.relu(self.fc1(self.norm(x)))))


class CustomSwin3D(nn.Module):
    def __init__(self, pretrained=False, T=8):
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


def build_model(num_classes, pretrained_k400=False, hidden_dim=512, dropout=0.3):
    model = CustomSwin3D(pretrained=pretrained_k400, T=8)
    model.head = MLPHead(768, num_classes, hidden_dim, dropout)
    return model


# ============================== Evaluation ==============================

def evaluate_framewise(model, loader, mappings, num_classes, smooth_k, device):
    model.eval()
    wp = []
    with torch.no_grad():
        for vids, _ in tqdm(loader, desc="Evaluating", leave=False):
            vids = vids.to(device)
            with autocast():
                wp.extend(torch.softmax(model(vids), dim=1).cpu().numpy())
    wp = np.array(wp)

    all_labels, all_fp = [], []
    for m in mappings:
        sl, f2w = m["labels"], m["frame_to_windows"]
        F = len(sl)
        fp = np.full((F, num_classes), 1.0 / num_classes, dtype=np.float32)
        for f in range(F):
            if f2w[f]:
                fp[f] = np.mean(wp[f2w[f]], axis=0)
        all_fp.append(fp)
        all_labels.extend(np.argmax(sl, axis=1).tolist())

    def smooth(p, k):
        if k <= 1:
            return p
        h = k // 2
        o = np.zeros_like(p)
        for i in range(len(p)):
            o[i] = np.mean(p[max(0, i - h):min(len(p), i + h + 1)], axis=0)
        return o

    preds, raw = [], []
    for fp in all_fp:
        raw.extend(fp.tolist())
        preds.extend(np.argmax(smooth(fp, smooth_k), axis=1).tolist())

    lbls = list(range(num_classes))
    f1_pc = f1_score(all_labels, preds, average=None, labels=lbls)
    f1_m = f1_score(all_labels, preds, average="macro")
    cm = confusion_matrix(all_labels, preds, labels=lbls)
    acc = np.mean(np.array(all_labels) == np.array(preds))

    oh = np.zeros((len(all_labels), num_classes))
    for i, l in enumerate(all_labels):
        oh[i, l] = 1.0
    raw = np.array(raw)
    ap_pc = np.array([
        average_precision_score(oh[:, c], raw[:, c]) if oh[:, c].sum() > 0 else float("nan")
        for c in range(num_classes)
    ])

    return {
        "f1_per_class": f1_pc, "f1_macro": f1_m, "cm": cm,
        "ap_per_class": ap_pc, "mAP": float(np.nanmean(ap_pc)),
        "accuracy": acc, "all_preds": preds, "all_labels": all_labels,
    }


# ============================== Main ==============================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    o2n = build_behavior_mapping(SELECTED_BEHAVIORS)
    num_classes = len(SELECTED_BEHAVIORS)
    rare_classes = [SELECTED_BEHAVIORS.index(b) for b in RARE_BEHAVIOR_NAMES if b in SELECTED_BEHAVIORS]
    os.makedirs(args.model_save_dir, exist_ok=True)

    use_custom = args.pretrained_model_path is not None
    init_source = args.pretrained_model_path if use_custom else "Kinetics-400 (torchvision)"

    ratio_tag = f"ratio{int(args.train_data_ratio * 100)}"
    split_tag = f"_vseed{args.video_split_seed}" if args.train_data_ratio < 1.0 else ""
    mode_tag = "crossdomain" if use_custom else "kinetics400"
    tag = f"{mode_tag}_{ratio_tag}{split_tag}_seed{args.seed}"

    print(f"\n{'='*70}")
    print(f"Cross-Domain Fine-tuning — {tag}")
    print(f"{'='*70}")
    print(f"  Init:       {init_source}")
    print(f"  Behaviors:  {SELECTED_BEHAVIORS}")
    print(f"  LR:         {args.base_lr:.2e}")
    print(f"  Epochs:     {args.num_epochs}")
    print(f"  Data ratio: {args.train_data_ratio*100:.0f}%")
    print(f"  Seeds:      general={args.seed}, val_split={args.val_split_seed}, video_split={args.video_split_seed}")
    print(f"  Device:     {device}")
    print(f"{'='*70}\n")

    # Load data
    all_train_vids, all_train_labs = get_video_and_label_paths(args.train_video_dir, args.train_label_dir)
    print(f"  Train+val: {len(all_train_vids)} videos\n")

    # Stratified split
    train_pool, train_pool_l, val_vids, val_labs = stratified_video_split(
        all_train_vids, all_train_labs, o2n, num_classes, SELECTED_BEHAVIORS, rare_classes,
        test_size=args.validation_ratio, min_threshold=args.min_behavior_threshold,
        random_state=args.val_split_seed)

    # Apply ratio
    train_vids, train_labs = subsample_videos(train_pool, train_pool_l,
                                              args.train_data_ratio, args.video_split_seed)
    print(f"  Train: {len(train_vids)}/{len(train_pool)} videos ({args.train_data_ratio*100:.0f}%)")
    if args.train_data_ratio < 1.0:
        print(f"  Selected: {[os.path.basename(v) for v in train_vids]}")
    print()

    # Augmentation
    aug_rng = random.Random(args.seed)
    def augment(frames):
        frames = random_blur_frames(frames, 0.35, (0.8, 2.2), rng=aug_rng)
        frames = random_temporal_dropout(frames, 0.15, rng=aug_rng)
        return frames

    # Datasets
    train_ds = SlidingWindowVideoDataset(train_vids, train_labs, args.window_size, args.stride,
                                         custom_video_transform, args.skip, augment, o2n, num_classes)
    val_ds = WindowPredictionDataset(val_vids, val_labs, args.window_size, args.stride,
                                     custom_video_transform, args.skip, o2n, num_classes)

    print(f"  Train windows: {len(train_ds)} | Val: {len(val_ds)}")
    counts = Counter(train_ds.sample_labels)
    for c in range(num_classes):
        n = counts.get(c, 0)
        print(f"    {SELECTED_BEHAVIORS[c]:>15}: {n:>6} ({100*n/max(1,len(train_ds)):>5.2f}%)")
    print()

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Build model
    if use_custom:
        model = build_model(num_classes, pretrained_k400=False,
                            hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)
        print(f"  Loading custom pretrained: {args.pretrained_model_path}")
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
        print("  ✅ Custom pretrained weights loaded\n")
    else:
        model = build_model(num_classes, pretrained_k400=True,
                            hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)
        print("  ✅ Kinetics-400 pretrained backbone (MLP head randomly initialized)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler()

    # Training loop
    print(f"{'='*70}")
    print("Fine-tuning")
    print(f"{'='*70}\n")

    log = []
    best_f1, best_path, best_ep = -1.0, None, -1

    for epoch in range(args.num_epochs):
        t0 = time.time()
        model.train()
        rl = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for bi, (vids, tgts) in enumerate(pbar):
            vids, tgts = vids.to(device), tgts.to(device)
            with autocast():
                loss = criterion(model(vids), tgts) / args.accumulation_steps
            scaler.scale(loss).backward()
            if (bi + 1) % args.accumulation_steps == 0 or (bi + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            rl += loss.item() * args.accumulation_steps * vids.size(0)
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)

        scheduler.step()
        tt = time.time() - t0
        el = rl / len(train_ds)

        t1 = time.time()
        vm = evaluate_framewise(model, val_loader, val_ds.frame_mappings, num_classes,
                                args.smooth_window_size, device)
        vt = time.time() - t1
        vf1, vmap, vacc = vm["f1_macro"], vm["mAP"], vm["accuracy"]

        print(f"\nEpoch {epoch+1} | Loss: {el:.4f} | Acc: {vacc:.4f} | F1: {vf1:.4f} | mAP: {vmap:.4f}")
        for n, v in zip(SELECTED_BEHAVIORS, vm["f1_per_class"]):
            print(f"  {n:>15} F1: {v:.4f}")
        print(f"  Train: {tt:.1f}s | Val: {vt:.1f}s")

        ckpt = os.path.join(args.model_save_dir, f"{tag}_ep{epoch+1}_f1_{vf1:.4f}_map_{vmap:.4f}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"  💾 {ckpt}")

        if vf1 > best_f1:
            best_f1, best_path, best_ep = vf1, ckpt, epoch + 1
            print(f"  🏆 New best! Val F1={vf1:.4f}")

        log.append({"epoch": epoch + 1, "train_loss": el, "val_accuracy": vacc,
                     "val_f1": vf1, "val_map": vmap,
                     "val_f1_per_class": vm["f1_per_class"].tolist(),
                     "val_ap_per_class": vm["ap_per_class"].tolist()})
        print()

    # Save training log
    results = {
        "config": vars(args),
        "init_source": init_source,
        "best_epoch": best_ep, "best_val_f1": best_f1,
        "best_checkpoint": best_path,
        "training_log": log,
    }
    rp = os.path.join(args.model_save_dir, f"training_log_{tag}.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Training log: {rp}")

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Epoch':<8} {'Loss':<10} {'Acc':<10} {'F1':<10} {'mAP':<10}")
    for l in log:
        m = " ← best" if l["epoch"] == best_ep else ""
        print(f"  {l['epoch']:<8} {l['train_loss']:<10.4f} {l['val_accuracy']:<10.4f} "
              f"{l['val_f1']:<10.4f} {l['val_map']:<10.4f}{m}")
    print(f"\n  🏆 Best: Epoch {best_ep} (Val F1={best_f1:.4f})")
    print(f"  💾 Best checkpoint: {best_path}")
    print(f"\n  → To evaluate on test set, run:")
    print(f"    python crossdomain_eval.py --model_path {best_path} --test_video_dir <TEST_DIR> --test_label_dir <LABEL_DIR>\n")


if __name__ == "__main__":
    main()
