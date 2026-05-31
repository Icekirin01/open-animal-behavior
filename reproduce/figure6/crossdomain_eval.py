"""
crossdomain_eval.py — Evaluate a fine-tuned Swin3D checkpoint on a test set
Standalone evaluation (no training). Use after crossdomain_train.py.

Usage:
    python crossdomain_eval.py --model_path checkpoints/crossdomain/model.pth \
        --test_video_dir data/new/videos/test \
        --test_label_dir data/new/labels/test

    python crossdomain_eval.py --model_path checkpoints/crossdomain/model.pth \
        --test_video_dir data/new/videos/test \
        --test_label_dir data/new/labels/test \
        --save_cm --save_results results.json
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


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned Swin3D checkpoint on test set")
    p.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--test_video_dir", type=str, required=True)
    p.add_argument("--test_label_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: same as model_path directory)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_dropout", type=float, default=0.3)
    p.add_argument("--smooth_window_size", type=int, default=1)
    p.add_argument("--save_cm", action="store_true", help="Save confusion matrix images")
    p.add_argument("--save_results", type=str, default=None, help="Path to save JSON results")
    return p.parse_args()


# ============================== Utilities ==============================

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


def custom_video_transform(frames):
    tensors = [ToTensor()(f) for f in frames]
    video = torch.stack(tensors, dim=1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
    return (video - mean) / std


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


def print_confusion_matrix_text(cm, names):
    cw = max(max(len(n) for n in names) + 2, 12)
    print("\n" + "=" * (cw * (len(names) + 1)))
    print(" " * cw + "Predicted →")
    print(" " * cw + "".join(f"{n:>{cw}}" for n in names))
    print("Actual ↓" + " " * (cw - 8) + "-" * (cw * len(names)))
    for i, n in enumerate(names):
        print(f"{n:>{cw}}" + "".join(f"{cm[i, j]:>{cw}}" for j in range(len(names))))
    print("=" * (cw * (len(names) + 1)) + "\n")


# ============================== Dataset ==============================

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
    def __init__(self, T=8):
        super().__init__()
        self.model = swin3d_t(weights=None)
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
    model = CustomSwin3D(T=8)
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
    o2n = build_behavior_mapping(SELECTED_BEHAVIORS)
    num_classes = len(SELECTED_BEHAVIORS)
    output_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"\n{'='*80}")
    print("MODEL EVALUATION — Swin3D (Cross-Domain)")
    print(f"{'='*80}")
    print(f"  Model:     {args.model_path}")
    print(f"  Test data: {args.test_video_dir}")
    print(f"  Behaviors: {SELECTED_BEHAVIORS}\n")

    vids, labs = get_video_and_label_paths(args.test_video_dir, args.test_label_dir)
    print(f"  Test videos: {len(vids)}\n")

    ds = WindowPredictionDataset(vids, labs, args.window_size, args.stride,
                                 custom_video_transform, args.skip, o2n, num_classes)
    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"  Test windows: {len(ds)}\n")

    model = build_model(num_classes, args.mlp_hidden_dim, args.mlp_dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("  ✅ Model loaded\n")

    metrics = evaluate_framewise(model, loader, ds.frame_mappings, num_classes,
                                 args.smooth_window_size, device)

    print(f"{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"  mAP:       {metrics['mAP']:.4f}\n")
    print("  Per-class F1:")
    for n, v in zip(SELECTED_BEHAVIORS, metrics["f1_per_class"]):
        print(f"    {n:>15}: {v:.4f}")
    print("\n  Per-class AP:")
    for n, v in zip(SELECTED_BEHAVIORS, metrics["ap_per_class"]):
        print(f"    {n:>15}: {v:.4f}")
    print_confusion_matrix_text(metrics["cm"], SELECTED_BEHAVIORS)

    model_file = os.path.basename(args.model_path).replace(".pth", "")

    if args.save_cm:
        plot_confusion_matrix(metrics["cm"], SELECTED_BEHAVIORS,
                              os.path.join(output_dir, f"cm_{model_file}.png"),
                              f"CM (F1={metrics['f1_macro']:.4f})")
        plot_confusion_matrix(metrics["cm"], SELECTED_BEHAVIORS,
                              os.path.join(output_dir, f"cm_{model_file}_norm.png"),
                              f"CM Normalized", normalize=True)

    if args.save_results:
        results = {
            "model_path": args.model_path,
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "mAP": float(metrics["mAP"]),
            "f1_per_class": metrics["f1_per_class"].tolist(),
            "ap_per_class": metrics["ap_per_class"].tolist(),
            "confusion_matrix": metrics["cm"].tolist(),
        }
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results: {args.save_results}")

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
