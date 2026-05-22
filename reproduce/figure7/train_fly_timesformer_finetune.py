"""
train_fly_timesformer_finetune.py — Fine-tune TimeSformer from Open Dataset → Lab Dataset
Loads a model pretrained on Open Dataset, then fine-tunes on Lab Dataset.
Supports video-level data efficiency (ratio) and three fine-tuning strategies.

This corresponds to Figure 5 experiments: comparing K400-pretrained vs Open-Dataset-pretrained.

Fine-tuning strategies:
    full       — all parameters trainable (recommended for small data)
    head_only  — freeze backbone, only train MLP head
    gradual    — freeze backbone for epoch 1, then unfreeze all

Usage:
    python train_fly_timesformer_finetune.py \\
      --pretrained_model_path checkpoints/fly_timesformer_open/model.pth \\
      --train_data_ratio 0.75 --video_split_seed 42 --finetune_strategy full

    python train_fly_timesformer_finetune.py \\
      --pretrained_model_path model.pth \\
      --train_data_ratio 0.5 --finetune_strategy head_only --reinit_head
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
from torchvision.transforms import ToTensor
import random
import json
import time
import argparse
import matplotlib
matplotlib.use("Agg")

# ============================== Configuration ==============================

ALL_BEHAVIOR_NAMES = ["wing_extension", "circle", "copul_attempt", "copulation", "others"]
SELECTED_BEHAVIORS = ["wing_extension", "circle", "copulation", "others"]
RARE_BEHAVIOR_NAMES = ["copulation"]

OTHERS_IDX = SELECTED_BEHAVIORS.index("others")
ORIGINAL_TO_NEW = {}
for oi, name in enumerate(ALL_BEHAVIOR_NAMES):
    ORIGINAL_TO_NEW[oi] = SELECTED_BEHAVIORS.index(name) if name in SELECTED_BEHAVIORS else OTHERS_IDX

NUM_CLASSES = len(SELECTED_BEHAVIORS)
RARE_CLASSES = [SELECTED_BEHAVIORS.index(b) for b in RARE_BEHAVIOR_NAMES if b in SELECTED_BEHAVIORS]
TIMESFORMER_NUM_FRAMES = 8

DEFAULTS = dict(
    train_video_dir="data/fly/videos/train",
    train_label_dir="data/fly/labels/train",
    batch_size=8, accumulation_steps=2, num_epochs=5,
    base_lr=3.8e-5, weight_decay=0.01, num_workers=8,
    use_class_weights=False, validation_ratio=0.15, min_behavior_threshold=50,
    aug_blur=True, aug_blur_frac=0.35, aug_td=True, aug_td_frac=0.15,
    window_size=16, stride=4, skip=0,
    mlp_hidden_dim=512, mlp_dropout=0.3, smooth_window_size=1,
    seed=2025, val_split_seed=123, train_data_ratio=0.75, video_split_seed=42,
    model_save_dir="checkpoints/fly_timesformer_finetune",
    hf_model="facebook/timesformer-base-finetuned-k400",
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TimeSformer (Open → Lab)")
    p.add_argument("--pretrained_model_path", type=str, required=True,
                   help="Path to Open-Dataset-pretrained .pth checkpoint")
    p.add_argument("--finetune_strategy", type=str, default="full",
                   choices=["full", "head_only", "gradual"],
                   help="full=all params, head_only=freeze backbone, gradual=freeze then unfreeze")
    p.add_argument("--reinit_head", action="store_true",
                   help="Reinitialize MLP head (only load backbone weights)")
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
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--val_split_seed", type=int, default=DEFAULTS["val_split_seed"])
    p.add_argument("--train_data_ratio", type=float, default=DEFAULTS["train_data_ratio"])
    p.add_argument("--video_split_seed", type=int, default=DEFAULTS["video_split_seed"])
    p.add_argument("--model_save_dir", type=str, default=DEFAULTS["model_save_dir"])
    p.add_argument("--hf_model", type=str, default=DEFAULTS["hf_model"])
    return p.parse_args()


# ============================== Utilities ==============================

def filter_and_remap_labels(oh):
    return np.array([ORIGINAL_TO_NEW[l] for l in np.argmax(oh, axis=1)], dtype=np.int64)

def get_video_and_label_paths(vd, ld):
    vps, lps = [], []
    for n in sorted(os.listdir(vd)):
        if not n.lower().endswith(".mp4"): continue
        lp = os.path.join(ld, n.replace(".mp4", ".csv"))
        if os.path.exists(lp): vps.append(os.path.join(vd, n)); lps.append(lp)
        else: print(f"[WARN] Label not found: {n}")
    return vps, lps

def analyze_and_split(vps, lps, test_size, min_thresh, random_state):
    profiles = []
    for vp, lp in zip(vps, lps):
        try:
            labels = filter_and_remap_labels(pd.read_csv(lp).iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values)
            counts = {c: int(np.sum(labels == c)) for c in range(NUM_CLASSES)}
            has_rare = any(counts[c] >= min_thresh for c in RARE_CLASSES)
            profiles.append({"vp": vp, "lp": lp, "counts": counts, "has_rare": has_rare, "name": os.path.basename(vp)})
        except Exception as e:
            print(f"⚠️ {vp}: {e}")
    strat = [1 if p["has_rare"] else 0 for p in profiles]
    _, cnts = np.unique(strat, return_counts=True); use_s = min(cnts) >= 2
    idx = list(range(len(profiles)))
    try:
        tr, va = train_test_split(idx, test_size=test_size, random_state=random_state,
                                   stratify=strat if use_s else None)
    except ValueError:
        tr, va = train_test_split(idx, test_size=test_size, random_state=random_state)
    print(f"  Split: {len(tr)} train / {len(va)} val")
    return ([profiles[i]["vp"] for i in tr], [profiles[i]["lp"] for i in tr],
            [profiles[i]["vp"] for i in va], [profiles[i]["lp"] for i in va])

def subsample_videos_by_ratio(vps, lps, ratio, seed=42):
    if ratio >= 1.0: return vps, lps
    n = len(vps); k = max(1, int(round(n * ratio)))
    sel = sorted(np.random.RandomState(seed).choice(n, size=k, replace=False).tolist())
    return [vps[i] for i in sel], [lps[i] for i in sel]

def random_blur_frames(frames, frac=0.35, radius_range=(0.8, 2.2), rng=None):
    if frac <= 0 or not frames: return frames
    rng = rng or random; n = len(frames); k = max(1, int(round(n * frac))); idxs = rng.sample(range(n), k)
    rmin, rmax = radius_range
    return [frames[i].filter(ImageFilter.GaussianBlur(radius=rng.uniform(rmin, rmax))) if i in idxs else frames[i] for i in range(n)]

def random_temporal_dropout(frames, frac=0.15, rng=None):
    if frac <= 0 or len(frames) < 3: return frames
    rng = rng or random; n = len(frames); k = max(1, int(round(n * frac)))
    idxs = rng.sample(range(1, n - 1), min(k, max(1, n - 2))); out = frames[:]
    for i in idxs: out[i] = out[i - 1] if rng.random() < 0.5 else out[i + 1]
    return out

def uniform_sample_frames(frames, target):
    n = len(frames)
    if n == target: return frames
    if n < target: return frames + [frames[-1]] * (target - n)
    return [frames[i] for i in np.linspace(0, n - 1, target, dtype=int)]

def custom_video_transform(frames, target_size=(224, 224)):
    frames = [f.resize(target_size, Image.BILINEAR) for f in frames]
    if len(frames) != TIMESFORMER_NUM_FRAMES:
        frames = uniform_sample_frames(frames, TIMESFORMER_NUM_FRAMES)
    t = [ToTensor()(f) for f in frames]
    v = torch.stack(t, dim=0)
    m = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return ((v - m) / s).permute(1, 0, 2, 3)


# ============================== Dataset ==============================

class SlidingWindowVideoDataset(Dataset):
    def __init__(self, vps, lps, ws, stride, transform, skip=0, augment=None):
        self.vps, self.lps, self.ws, self.stride, self.skip = vps, lps, ws, stride, skip
        self.transform, self.augment = transform, augment
        self.samples, self.sample_labels = self._gen()
    def _gen(self):
        samples, labels = [], []
        for vp, lp in zip(self.vps, self.lps):
            oh = pd.read_csv(lp).iloc[:, 0:len(ALL_BEHAVIOR_NAMES)].values
            vr = VideoReader(vp, ctx=cpu(0)); T = len(vr)
            if len(oh) != T: print(f"⚠️ Mismatch: {os.path.basename(vp)}"); continue
            rem = filter_and_remap_labels(oh); sel = list(range(0, T, self.skip + 1)); fcls = rem[sel]
            for s in range(0, len(sel) - self.ws + 1, self.stride):
                wi = sel[s:s + self.ws]; wl = Counter(fcls[s:s + self.ws]).most_common(1)[0][0]
                samples.append((vp, wi, wl)); labels.append(wl)
        return samples, labels
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vp, fi, label = self.samples[idx]; vr = VideoReader(vp, ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(fi).asnumpy()]
        if len(frames) < self.ws: frames.extend([frames[-1]] * (self.ws - len(frames)))
        if self.augment: frames = self.augment(frames)
        return self.transform(frames), torch.tensor(label, dtype=torch.long)


# ============================== Model ==============================

class MLPHead(nn.Module):
    def __init__(self, inf, nc, hd=512, do=0.3):
        super().__init__(); self.norm = nn.LayerNorm(inf); self.fc1 = nn.Linear(inf, hd)
        self.relu = nn.ReLU(True); self.dropout = nn.Dropout(do); self.fc2 = nn.Linear(hd, nc)
    def forward(self, x):
        x = self.norm(x); return self.fc2(self.dropout(self.relu(self.fc1(x))))

class CustomTimeSformer(nn.Module):
    def __init__(self, nc, hf_model, hd=512, do=0.3):
        super().__init__()
        from transformers import TimesformerModel
        print(f"   Loading TimeSformer: {hf_model}")
        self.backbone = TimesformerModel.from_pretrained(hf_model)
        self.head = MLPHead(self.backbone.config.hidden_size, nc, hd, do)
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.head(self.backbone(pixel_values=x).last_hidden_state[:, 0])


def load_pretrained_and_prepare(model, path, reinit_head, strategy, device):
    """Load Open-Dataset-pretrained weights and configure fine-tuning strategy."""
    print(f"\n🔄 Loading pretrained: {path}")
    sd = torch.load(path, map_location=device)

    if reinit_head:
        bb = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        model.load_state_dict(bb, strict=False)
        print(f"   Loaded backbone only ({len(bb)} keys), head reinitialized")
    else:
        model.load_state_dict(sd)
        print("   Loaded complete model (backbone + head)")

    if strategy == "head_only":
        print("   🧊 Freezing backbone (head_only)")
        for p in model.backbone.parameters(): p.requires_grad = False
    elif strategy == "gradual":
        print("   🧊 Freezing backbone (gradual — will unfreeze after epoch 1)")
        for p in model.backbone.parameters(): p.requires_grad = False
    else:
        print("   🔥 Full fine-tuning")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)\n")
    return model


# ============================== Evaluation ==============================

def evaluate_window_level(model, loader, device):
    model.eval(); all_p, all_l, all_pr = [], [], []
    with torch.no_grad():
        for v, l in tqdm(loader, desc="Validation", leave=False):
            v, l = v.to(device), l.to(device)
            with autocast(): lo = model(v); pr = torch.softmax(lo, dim=1)
            all_p.extend(torch.argmax(lo, dim=1).cpu().numpy())
            all_l.extend(l.cpu().numpy()); all_pr.extend(pr.cpu().numpy())
    f1_pc = f1_score(all_l, all_p, average=None, labels=list(range(NUM_CLASSES)))
    f1_m = f1_score(all_l, all_p, average="macro")
    non_o = [i for i in range(NUM_CLASSES) if i != OTHERS_IDX]
    f1_no = float(np.mean(f1_pc[non_o]))
    oh = np.zeros((len(all_l), NUM_CLASSES))
    for i, l in enumerate(all_l): oh[i, l] = 1.0
    all_pr = np.array(all_pr)
    ap_pc = np.array([average_precision_score(oh[:, c], all_pr[:, c]) if oh[:, c].sum() > 0 else 0.0 for c in range(NUM_CLASSES)])
    return {"f1_per_class": f1_pc, "f1_macro": f1_m, "f1_no_others": f1_no,
            "ap_per_class": ap_pc, "mAP": np.mean(ap_pc)}


# ============================== Main ==============================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.pretrained_model_path):
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model_path}")

    os.makedirs(args.model_save_dir, exist_ok=True)
    ratio_tag = f"ratio{int(args.train_data_ratio * 100)}"
    split_tag = f"_vseed{args.video_split_seed}" if args.train_data_ratio < 1.0 else ""
    ft_tag = f"_ft_{args.finetune_strategy}" + ("_reinit" if args.reinit_head else "")
    run_tag = f"fly_timesformer_{ratio_tag}{split_tag}{ft_tag}_seed{args.seed}"

    print(f"\n{'='*70}")
    print(f"Fly Copulation — Fine-tune TimeSformer (Open Dataset → Lab Dataset)")
    print(f"{'='*70}")
    print(f"  Pretrained: {args.pretrained_model_path}")
    print(f"  Strategy: {args.finetune_strategy} | Reinit head: {args.reinit_head}")
    print(f"  Ratio: {args.train_data_ratio * 100:.0f}% | Video split seed: {args.video_split_seed}")
    print(f"  Seed: {args.seed} | Val split: {args.val_split_seed} (FIXED)")
    print(f"  Best model: by val F1-macro excluding 'others'")
    print(f"{'='*70}\n")

    # Load + split
    all_v, all_l = get_video_and_label_paths(args.train_video_dir, args.train_label_dir)
    print(f"Total videos: {len(all_v)}")
    pool_v, pool_l, val_v, val_l = analyze_and_split(all_v, all_l, args.validation_ratio, args.min_behavior_threshold, args.val_split_seed)

    train_v, train_l = subsample_videos_by_ratio(pool_v, pool_l, args.train_data_ratio, args.video_split_seed)
    print(f"Training videos: {len(train_v)}/{len(pool_v)} ({args.train_data_ratio * 100:.0f}%)")
    if args.train_data_ratio < 1.0:
        print(f"  Selected: {[os.path.basename(v) for v in train_v]}")
    print(f"Validation videos: {len(val_v)} (FIXED)\n")

    # Datasets
    aug_rng = random.Random(args.seed)
    def augment(frames):
        if DEFAULTS["aug_blur"]: frames = random_blur_frames(frames, DEFAULTS["aug_blur_frac"], rng=aug_rng)
        if DEFAULTS["aug_td"]: frames = random_temporal_dropout(frames, DEFAULTS["aug_td_frac"], rng=aug_rng)
        return frames

    train_ds = SlidingWindowVideoDataset(train_v, train_l, args.window_size, args.stride, custom_video_transform, args.skip, augment)
    val_ds = SlidingWindowVideoDataset(val_v, val_l, args.window_size, args.stride, custom_video_transform, args.skip, None)
    counts = Counter(train_ds.sample_labels)
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Build model + load pretrained
    model = CustomTimeSformer(NUM_CLASSES, args.hf_model, args.mlp_hidden_dim, args.mlp_dropout).to(device)
    model = load_pretrained_and_prepare(model, args.pretrained_model_path, args.reinit_head, args.finetune_strategy, device)

    if args.use_class_weights:
        w = np.array([counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=np.float32)
        w = w.sum() / (NUM_CLASSES * w); w /= w.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs); scaler = GradScaler()

    log, best_f1no, best_path = [], -1.0, None

    for epoch in range(args.num_epochs):
        # Gradual unfreezing
        if args.finetune_strategy == "gradual" and epoch == 1:
            print("🔥 Unfreezing backbone (gradual, epoch 2+)")
            for p in model.backbone.parameters(): p.requires_grad = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=args.base_lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - 1)
            print(f"   Trainable: {sum(p.numel() for p in trainable_params):,}\n")

        t0 = time.time(); model.train(); rl = 0.0; optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for bi, (vids, tgts) in enumerate(pbar):
            vids, tgts = vids.to(device), tgts.to(device)
            with autocast(): loss = criterion(model(vids), tgts) / args.accumulation_steps
            scaler.scale(loss).backward()
            if (bi + 1) % args.accumulation_steps == 0 or (bi + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            rl += loss.item() * args.accumulation_steps * vids.size(0)
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)

        scheduler.step(); tt = time.time() - t0; el = rl / len(train_ds)
        t1 = time.time(); m = evaluate_window_level(model, val_loader, device); vt = time.time() - t1

        print(f"\nEpoch {epoch+1} | Loss: {el:.4f}")
        print(f"  Val F1 (all): {m['f1_macro']:.4f} | F1 (no others): {m['f1_no_others']:.4f} ← best selection")
        print(f"  Val mAP: {m['mAP']:.4f}")
        for n, v in zip(SELECTED_BEHAVIORS, m["f1_per_class"]):
            print(f"    {n:>15} F1: {v:.4f}{' ← excluded' if n == 'others' else ''}")
        print(f"  Train: {tt:.1f}s | Val: {vt:.1f}s")

        ckpt = os.path.join(args.model_save_dir, f"{run_tag}_ep{epoch+1}_f1no_{m['f1_no_others']:.4f}_map_{m['mAP']:.4f}.pth")
        torch.save(model.state_dict(), ckpt); print(f"  💾 {ckpt}")

        if m["f1_no_others"] > best_f1no:
            best_f1no = m["f1_no_others"]; best_path = ckpt
            print(f"  🏆 New best! F1-no-others={best_f1no:.4f}")

        log.append({"epoch": epoch + 1, "train_loss": el, "val_f1": m["f1_macro"],
                     "val_f1_no_others": m["f1_no_others"], "val_map": m["mAP"],
                     "val_f1_per_class": m["f1_per_class"].tolist(),
                     "val_ap_per_class": m["ap_per_class"].tolist(),
                     "train_time": tt, "val_time": vt,
                     "train_data_ratio": args.train_data_ratio, "finetune_strategy": args.finetune_strategy})
        print()

    best_ep = max(log, key=lambda x: x["val_f1_no_others"])
    print(f"\n✅ Best: Epoch {best_ep['epoch']} | F1-no-others={best_ep['val_f1_no_others']:.4f} | mAP={best_ep['val_map']:.4f}")
    print(f"   Model: {best_path}")

    lp = os.path.join(args.model_save_dir, f"training_log_{run_tag}.json")
    with open(lp, "w") as f:
        json.dump({"training_log": log, "best_epoch": best_ep, "best_model_path": best_path,
                    "config": {**vars(args), "num_classes": NUM_CLASSES, "selected_behaviors": SELECTED_BEHAVIORS,
                               "all_behavior_names": ALL_BEHAVIOR_NAMES, "others_class_idx": OTHERS_IDX,
                               "model": "TimeSformer (ViT-Base) fine-tuned from Open Dataset"},
                    "split_info": {"train_pool": [os.path.basename(v) for v in pool_v],
                                   "train_used": [os.path.basename(v) for v in train_v],
                                   "val": [os.path.basename(v) for v in val_v]}}, f, indent=2)
    print(f"💾 Log: {lp}\n")


if __name__ == "__main__":
    main()
