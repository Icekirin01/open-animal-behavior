"""
eval_calms21.py — CalMS21 Test Evaluation: Frame-wise with Confusion Matrix

Usage:
    python eval_calms21.py --model_path checkpoints/calms21/model.pth
    python eval_calms21.py --model_path model.pth --test_video_dir data/test/videos --save_cm cm.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
import decord
from decord import VideoReader, cpu
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, classification_report
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import ToTensor
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== Configuration ==============================

NUM_CLASSES = 4
BEHAVIOR_NAMES = ["Attack", "Investigation", "Mount", "Other"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Video Swin-T on CalMS21 test set")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_video_dir", type=str, default="data/calms21/videos/test")
    p.add_argument("--test_label_dir", type=str, default="data/calms21/labels/test")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_dropout", type=float, default=0.3)
    p.add_argument("--smooth_window_size", type=int, default=1)
    p.add_argument("--save_cm", type=str, default=None, help="Path to save confusion matrix PNG")
    p.add_argument("--save_results", type=str, default=None, help="Path to save results JSON")
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


def custom_video_transform(frames):
    frames = [ToTensor()(f) for f in frames]
    video = torch.stack(frames, dim=1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
    return (video - mean) / std


def plot_confusion_matrix(cm, names, path, title="Confusion Matrix", normalize=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    if normalize:
        cm_disp = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        cm_disp = np.nan_to_num(cm_disp)
        fmt = ".1f"
    else:
        cm_disp = cm
        fmt = "d"
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax, square=True)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {path}")


# ============================== Dataset ==============================

class WindowPredictionDataset(Dataset):
    def __init__(self, video_paths, label_paths, window_size, stride, transform, skip=0):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.skip = skip
        self.windows, self.frame_mappings = self._generate_windows()

    def _generate_windows(self):
        windows, mappings = [], []
        for vp, lp in zip(self.video_paths, self.label_paths):
            df = pd.read_csv(lp)
            labels_oh = df.iloc[:, 0:NUM_CLASSES].values
            vr = VideoReader(vp, ctx=cpu(0))
            T = len(vr)
            if len(labels_oh) != T:
                print(f"⚠️ Mismatch: {os.path.basename(vp)}")
                continue
            selected = list(range(0, T, self.skip + 1))
            sel_labels = labels_oh[selected]
            f2w = [[] for _ in range(len(selected))]
            for start in range(0, len(selected) - self.window_size + 1, self.stride):
                win_idx = selected[start:start + self.window_size]
                windows.append((vp, win_idx))
                widx = len(windows) - 1
                for fi in range(start, start + self.window_size):
                    if fi < len(f2w):
                        f2w[fi].append(widx)
            mappings.append({"labels": sel_labels, "frame_to_windows": f2w})
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
        x = self.norm(x)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


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


def build_model(num_classes, hidden_dim=512, dropout=0.3):
    model = CustomSwin3D(pretrained=False, T=8)
    model.head = MLPHead(768, num_classes, hidden_dim, dropout)
    return model


# ============================== Evaluation ==============================

def evaluate_framewise(model, loader, mappings, smooth_k, device):
    model.eval()
    wp = []
    with torch.no_grad():
        for videos, _ in tqdm(loader, desc="Testing", leave=False):
            videos = videos.to(device)
            with autocast():
                wp.extend(torch.softmax(model(videos), dim=1).cpu().numpy())
    wp = np.array(wp)

    all_labels, all_fp = [], []
    for m in mappings:
        labels, f2w = m["labels"], m["frame_to_windows"]
        F = len(labels)
        fp = np.zeros((F, NUM_CLASSES), dtype=np.float32)
        for f in range(F):
            if f2w[f]:
                fp[f] = np.mean(wp[f2w[f]], axis=0)
            else:
                fp[f, -1] = 1.0  # default to last class
        all_fp.append(fp)
        all_labels.extend(np.argmax(labels, axis=1))

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

    lbls = list(range(NUM_CLASSES))
    f1_pc = f1_score(all_labels, preds, average=None, labels=lbls)
    f1_m = f1_score(all_labels, preds, average="macro")
    cm = confusion_matrix(all_labels, preds, labels=lbls)

    oh = np.zeros((len(all_labels), NUM_CLASSES))
    for i, l in enumerate(all_labels):
        oh[i, l] = 1.0
    raw = np.array(raw)
    ap_pc = np.array([
        average_precision_score(oh[:, c], raw[:, c]) if oh[:, c].sum() > 0 else 0.0
        for c in range(NUM_CLASSES)
    ])

    return {
        "f1_per_class": f1_pc, "f1_macro": f1_m, "cm": cm,
        "ap_per_class": ap_pc, "mAP": np.mean(ap_pc),
        "all_preds": preds, "all_labels": all_labels,
    }


# ============================== Main ==============================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"\n{'='*70}")
    print("CalMS21 Test Evaluation — Frame-wise (MLP Head)")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path}")
    print(f"  Behaviors: {BEHAVIOR_NAMES} ({NUM_CLASSES} classes)\n")

    vids, labs = get_video_and_label_paths(args.test_video_dir, args.test_label_dir)
    print(f"  Test videos: {len(vids)}")

    ds = WindowPredictionDataset(vids, labs, args.window_size, args.stride,
                                 custom_video_transform, args.skip)
    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"  Test windows: {len(ds)}\n")

    model = build_model(NUM_CLASSES, args.mlp_hidden_dim, args.mlp_dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("  ✅ Model loaded\n")

    metrics = evaluate_framewise(model, loader, ds.frame_mappings,
                                 args.smooth_window_size, device)

    # Results
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"  mAP:       {metrics['mAP']:.4f}\n")

    print("  Per-class F1:")
    for n, v in zip(BEHAVIOR_NAMES, metrics["f1_per_class"]):
        print(f"    {n:>15}: {v:.4f}")
    print("\n  Per-class AP:")
    for n, v in zip(BEHAVIOR_NAMES, metrics["ap_per_class"]):
        print(f"    {n:>15}: {v:.4f}")

    # Confusion matrix text
    print(f"\n  Confusion Matrix:")
    cw = 12
    print(f"  {'':>15}" + "".join(f"{n:>{cw}}" for n in BEHAVIOR_NAMES))
    print(f"  {'-'*(15 + cw * NUM_CLASSES)}")
    for i, n in enumerate(BEHAVIOR_NAMES):
        print(f"  {n:>15}" + "".join(f"{metrics['cm'][i,j]:>{cw}}" for j in range(NUM_CLASSES)))

    # Per-class accuracy
    print(f"\n  Per-class Accuracy:")
    for i, n in enumerate(BEHAVIOR_NAMES):
        total = metrics["cm"][i].sum()
        correct = metrics["cm"][i, i]
        acc = correct / total if total > 0 else 0
        print(f"    {n:>15}: {acc:.4f} ({correct}/{total})")

    # Save confusion matrix
    if args.save_cm:
        plot_confusion_matrix(metrics["cm"], BEHAVIOR_NAMES, args.save_cm,
                              f"CalMS21 Test (F1={metrics['f1_macro']:.4f})")
        # Also save normalized
        norm_path = args.save_cm.replace(".png", "_normalized.png")
        plot_confusion_matrix(metrics["cm"], BEHAVIOR_NAMES, norm_path,
                              f"CalMS21 Test Normalized (F1={metrics['f1_macro']:.4f})",
                              normalize=True)

    # Save results JSON
    if args.save_results:
        results = {
            "model_path": args.model_path,
            "f1_macro": float(metrics["f1_macro"]),
            "mAP": float(metrics["mAP"]),
            "f1_per_class": metrics["f1_per_class"].tolist(),
            "ap_per_class": metrics["ap_per_class"].tolist(),
            "confusion_matrix": metrics["cm"].tolist(),
            "behavior_names": BEHAVIOR_NAMES,
            "num_videos": len(ds.frame_mappings),
            "num_frames": len(metrics["all_labels"]),
        }
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results: {args.save_results}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
