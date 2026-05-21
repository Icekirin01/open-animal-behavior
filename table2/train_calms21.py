"""
train_calms21.py — CalMS21 Video Swin-T Training
Stratified Video-Level Split (No Data Leakage) + MLP Head + Cross Entropy Loss

Usage:
    python train_calms21.py --seed 123
    python train_calms21.py --seed 1337
    python train_calms21.py --seed 2025
    python train_calms21.py --seed 123 --use_class_weights
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
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== Configuration ==============================

NUM_CLASSES = 4
BEHAVIOR_NAMES = ["Attack", "Investigation", "Mount", "Other"]

DEFAULTS = dict(
    train_video_dir="data/calms21/videos/train",
    train_label_dir="data/calms21/labels/train",
    batch_size=8,
    accumulation_steps=2,
    num_epochs=5,
    base_lr=3.8e-5,
    weight_decay=0.01,
    num_workers=8,
    use_class_weights=False,
    validation_ratio=0.15,
    min_behavior_threshold=50,
    rare_behavior_classes=[0],  # Attack
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
    smooth_window_size=1,
    seed=123,
    val_split_seed=1337,
    model_save_dir="checkpoints/calms21",
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Video Swin-T on CalMS21")
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
    p.add_argument("--smooth_window_size", type=int, default=DEFAULTS["smooth_window_size"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                   help="General seed (model init, augmentation, shuffle). Run with 123/1337/2025.")
    p.add_argument("--val_split_seed", type=int, default=DEFAULTS["val_split_seed"],
                   help="Validation split seed (FIXED across experiments for fair comparison)")
    p.add_argument("--model_save_dir", type=str, default=DEFAULTS["model_save_dir"])
    return p.parse_args()


# ============================== Utilities ==============================

def get_video_and_label_paths(video_dir, label_dir):
    video_paths, label_paths = [], []
    for name in sorted(os.listdir(video_dir)):
        if not name.lower().endswith(".mp4"):
            continue
        vp = os.path.join(video_dir, name)
        lp = os.path.join(label_dir, name.replace(".mp4", ".csv"))
        if os.path.exists(lp):
            video_paths.append(vp)
            label_paths.append(lp)
        else:
            print(f"[WARN] Label not found: {name}")
    return video_paths, label_paths


def analyze_video_behaviors(video_paths, label_paths, min_threshold, rare_classes):
    """Analyze per-video behavior distribution for stratified splitting."""
    profiles = []
    for vp, lp in zip(video_paths, label_paths):
        try:
            df = pd.read_csv(lp)
            labels = np.argmax(df.iloc[:, 0:NUM_CLASSES].values, axis=1)
            counts = {c: int(np.sum(labels == c)) for c in range(NUM_CLASSES)}
            has_rare = any(counts[c] >= min_threshold for c in rare_classes)
            profiles.append({
                "video_path": vp, "label_path": lp,
                "video_name": os.path.basename(vp),
                "has_rare_behavior": has_rare,
                "behavior_counts": counts,
                "total_frames": len(labels),
            })
        except Exception as e:
            print(f"⚠️ Error: {vp}: {e}")
    return profiles


def stratified_video_split(video_paths, label_paths, test_size, min_threshold,
                           rare_classes, random_state):
    """Stratified video-level split ensuring rare behaviors in both sets."""
    profiles = analyze_video_behaviors(video_paths, label_paths, min_threshold, rare_classes)

    stratify_labels = [1 if p["has_rare_behavior"] else 0 for p in profiles]
    _, counts = np.unique(stratify_labels, return_counts=True)
    use_stratify = min(counts) >= 2

    indices = list(range(len(profiles)))
    try:
        if use_stratify:
            train_idx, val_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state,
                stratify=stratify_labels)
        else:
            train_idx, val_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state)
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state)

    train_vids = [profiles[i]["video_path"] for i in train_idx]
    train_labs = [profiles[i]["label_path"] for i in train_idx]
    val_vids = [profiles[i]["video_path"] for i in val_idx]
    val_labs = [profiles[i]["label_path"] for i in val_idx]

    # Report
    print(f"\n  Split: {len(train_idx)} train / {len(val_idx)} val videos")
    print(f"  {'Behavior':>15}  {'Train':>8}  {'Val':>8}")
    print(f"  {'-'*35}")
    for c in range(NUM_CLASSES):
        tc = sum(profiles[i]["behavior_counts"][c] for i in train_idx)
        vc = sum(profiles[i]["behavior_counts"][c] for i in val_idx)
        print(f"  {BEHAVIOR_NAMES[c]:>15}  {tc:>8}  {vc:>8}")
        if vc < min_threshold:
            print(f"  ⚠️ {BEHAVIOR_NAMES[c]} has only {vc} val frames!")
    print()

    return train_vids, train_labs, val_vids, val_labs, profiles


def random_blur_frames(frames, frac=0.35, radius_range=(0.8, 2.2), rng=None):
    if frac <= 0 or len(frames) == 0:
        return frames
    rng = rng or random
    n = len(frames)
    k = max(1, int(round(n * frac)))
    idxs = rng.sample(range(n), k)
    rmin, rmax = radius_range
    out = []
    for i, im in enumerate(frames):
        if i in idxs:
            out.append(im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(rmin, rmax))))
        else:
            out.append(im)
    return out


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


def custom_video_transform(frames):
    frames = [ToTensor()(f) for f in frames]
    video = torch.stack(frames, dim=1)  # (C, T, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
    return (video - mean) / std


def plot_confusion_matrix(cm, names, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    pct = np.nan_to_num(pct)
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax, vmin=0, vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="gray")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {path}")


# ============================== Datasets ==============================

class SlidingWindowVideoDataset(Dataset):
    def __init__(self, video_paths, label_paths, window_size, stride, transform,
                 skip=0, augment=None):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.skip = skip
        self.augment = augment
        self.samples, self.sample_labels = self._generate_samples()

    def _generate_samples(self):
        samples, labels = [], []
        for vp, lp in zip(self.video_paths, self.label_paths):
            df = pd.read_csv(lp)
            labels_oh = df.iloc[:, 0:NUM_CLASSES].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(labels_oh) != T:
                print(f"⚠️ Mismatch: {os.path.basename(vp)}")
                continue
            selected = list(range(0, T, self.skip + 1))
            frame_cls = np.argmax(labels_oh[selected], axis=1)
            for start in range(0, len(selected) - self.window_size + 1, self.stride):
                win_idx = selected[start:start + self.window_size]
                win_label = Counter(frame_cls[start:start + self.window_size]).most_common(1)[0][0]
                samples.append((vp, win_idx, win_label))
                labels.append(win_label)
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
        x = torch.mean(x, dim=1)  # (B, T, D) → (B, D)
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
        x = x.mean(dim=(2, 3))  # (B, T, C)
        return self.head(x)


def build_model(num_classes, hidden_dim=512, dropout=0.3):
    model = CustomSwin3D(pretrained=True, T=DEFAULTS["T"])
    model.head = MLPHead(768, num_classes, hidden_dim, dropout)
    return model


# ============================== Evaluation ==============================

def evaluate_on_validation(model, val_loader, device):
    """Window-level validation (F1 + mAP)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Validation", leave=False):
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
        average_precision_score(oh[:, c], all_probs[:, c])
        if oh[:, c].sum() > 0 else 0.0
        for c in range(NUM_CLASSES)
    ])

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
    run_tag = f"calms21_seed{args.seed}_vsplit{args.val_split_seed}"

    print(f"\n{'='*70}")
    print(f"CalMS21 Training — Swin3D + MLP Head + CE Loss")
    print(f"{'='*70}")
    print(f"  Behaviors: {BEHAVIOR_NAMES} ({NUM_CLASSES} classes)")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed} (general)  |  {args.val_split_seed} (val split, FIXED)")
    print(f"  Epochs: {args.num_epochs}, LR: {args.base_lr:.2e}")
    print(f"{'='*70}\n")

    # Load data
    all_vids, all_labs = get_video_and_label_paths(args.train_video_dir, args.train_label_dir)
    print(f"Total videos: {len(all_vids)}")

    # Stratified video split
    train_vids, train_labs, val_vids, val_labs, profiles = stratified_video_split(
        all_vids, all_labs, args.validation_ratio, args.min_behavior_threshold,
        DEFAULTS["rare_behavior_classes"], args.val_split_seed)

    # Augmentation
    aug_rng = random.Random(args.seed)
    def augment(frames):
        if DEFAULTS["aug_blur"]:
            frames = random_blur_frames(frames, DEFAULTS["aug_blur_frac"], rng=aug_rng)
        if DEFAULTS["aug_td"]:
            frames = random_temporal_dropout(frames, DEFAULTS["aug_td_frac"], rng=aug_rng)
        return frames

    train_ds = SlidingWindowVideoDataset(
        train_vids, train_labs, args.window_size, args.stride,
        custom_video_transform, args.skip, augment)
    val_ds = SlidingWindowVideoDataset(
        val_vids, val_labs, args.window_size, args.stride,
        custom_video_transform, args.skip, None)

    counts = Counter(train_ds.sample_labels)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")
    for c in range(NUM_CLASSES):
        print(f"  {BEHAVIOR_NAMES[c]:>15}: train={counts.get(c,0)}, val={Counter(val_ds.sample_labels).get(c,0)}")
    print()

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(NUM_CLASSES, args.mlp_hidden_dim, args.mlp_dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params/1e6:.1f}M\n")

    if args.use_class_weights:
        w = np.array([counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=np.float32)
        w = w.sum() / (NUM_CLASSES * w)
        w /= w.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler()

    log = []

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
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.accumulation_steps * vids.size(0)
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)

        scheduler.step()
        train_time = time.time() - t0
        epoch_loss = running_loss / len(train_ds)

        t1 = time.time()
        metrics = evaluate_on_validation(model, val_loader, device)
        val_time = time.time() - t1

        f1, mAP = metrics["f1_macro"], metrics["mAP"]
        print(f"\nEpoch {epoch+1} | Loss: {epoch_loss:.4f} | Val F1: {f1:.4f} | Val mAP: {mAP:.4f}")
        for name, v in zip(BEHAVIOR_NAMES, metrics["f1_per_class"]):
            print(f"  {name:>15} F1: {v:.4f}")
        for name, v in zip(BEHAVIOR_NAMES, metrics["ap_per_class"]):
            print(f"  {name:>15} AP: {v:.4f}")
        print(f"  Train: {train_time:.1f}s | Val: {val_time:.1f}s")

        ckpt = os.path.join(args.model_save_dir,
                            f"{run_tag}_ep{epoch+1}_f1_{f1:.4f}_map_{mAP:.4f}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"💾 {ckpt}")

        log.append({
            "epoch": epoch + 1, "train_loss": epoch_loss,
            "val_f1": f1, "val_map": mAP,
            "val_f1_per_class": metrics["f1_per_class"].tolist(),
            "val_ap_per_class": metrics["ap_per_class"].tolist(),
            "train_time": train_time, "val_time": val_time,
            "train_videos": len(train_vids), "val_videos": len(val_vids),
            "train_windows": len(train_ds), "val_windows": len(val_ds),
        })
        print()

    best = max(log, key=lambda x: x["val_f1"])
    print(f"\n✅ Best: Epoch {best['epoch']} | F1={best['val_f1']:.4f} | mAP={best['val_map']:.4f}")

    log_path = os.path.join(args.model_save_dir, f"training_log_{run_tag}.json")
    with open(log_path, "w") as f:
        json.dump({
            "training_log": log, "best_epoch": best,
            "config": {**vars(args), "num_classes": NUM_CLASSES,
                       "behavior_names": BEHAVIOR_NAMES,
                       "model": "Video Swin-T + MLP Head"},
            "split_info": {
                "train_videos": [os.path.basename(v) for v in train_vids],
                "val_videos": [os.path.basename(v) for v in val_vids],
            },
        }, f, indent=2)
    print(f"💾 Log: {log_path}\n")


if __name__ == "__main__":
    main()
