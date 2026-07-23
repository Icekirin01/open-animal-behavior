# @title gui (with BORIS label support)

# ==================== EDIT BELOW ====================
HF_REPO_ID = "yiheng266/animal-social-models"
DEFAULT_VIDEO_DIR = "/content/drive/My Drive/videos/train/"
DEFAULT_LABEL_DIR = "/content/drive/My Drive/labels/train/"
DEFAULT_OUTPUT_DIR = "/content/drive/My Drive/trained_models/"
MAX_LABELS = 15  # pre-built dropdown slots
# ==================== END OF EDITS ====================

import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
import gradio as gr, pandas as pd, random, shutil, time, traceback
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor
from collections import Counter
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import Dataset, DataLoader
# GradScaler / autocast: the new torch.amp API (device-aware) replaces the deprecated
# torch.cuda.amp API in PyTorch 2.3+. Fall back to the old one on older versions.
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast
    def GradScaler(): return _GradScaler("cuda")
    def autocast(): return _autocast("cuda")
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # pragma: no cover
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ====================== Google Drive Mount Check ======================

def ensure_drive_mounted():
    if os.path.exists("/content") and not os.path.exists("/content/drive/My Drive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("✅ Google Drive mounted.")
        except Exception as e:
            print(f"⚠️ Could not mount Google Drive: {e}")
    elif os.path.exists("/content/drive/My Drive"):
        print("✅ Google Drive already mounted.")
    else:
        print("ℹ️ Not running in Colab — skipping Drive mount.")

ensure_drive_mounted()

# ====================== Models ======================

class MLPHead_CLS(nn.Module):
    def __init__(self, inf, nc, hd, dr):
        super().__init__()
        self.norm=nn.LayerNorm(inf); self.fc1=nn.Linear(inf,hd)
        self.relu=nn.ReLU(True); self.drop=nn.Dropout(dr); self.fc2=nn.Linear(hd,nc)
    def forward(self,x):
        return self.fc2(self.drop(self.relu(self.fc1(self.norm(x)))))

class MLPHead_TM(nn.Module):
    def __init__(self, inf, nc, hd, dr):
        super().__init__()
        self.norm=nn.LayerNorm(inf); self.fc1=nn.Linear(inf,hd)
        self.relu=nn.ReLU(True); self.drop=nn.Dropout(dr); self.fc2=nn.Linear(hd,nc)
    def forward(self,x):
        x=torch.mean(x,dim=1)
        return self.fc2(self.drop(self.relu(self.fc1(self.norm(x)))))

class CustomTimeSformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import TimesformerModel
        self.backbone=TimesformerModel.from_pretrained(cfg["backbone"]["pretrained"])
        self.head=MLPHead_CLS(cfg["head"]["in_features"],cfg["num_classes"],cfg["head"]["hidden_dim"],cfg["head"]["dropout"])
    def forward(self,x):
        x=x.permute(0,2,1,3,4)
        return self.head(self.backbone(pixel_values=x).last_hidden_state[:,0])

class CustomSwin3D(nn.Module):
    """Swin3D — supports both swin3d_t and swin3d_b via cfg["backbone"]["variant"]."""
    def __init__(self, cfg):
        super().__init__()
        variant = cfg["backbone"].get("variant", "t").lower()
        if variant == "b":
            from torchvision.models.video import swin3d_b, Swin3D_B_Weights
            self.model = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)
        else:
            from torchvision.models.video import swin3d_t, Swin3D_T_Weights
            self.model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        self.model.head=nn.Identity(); self.model.avgpool=nn.Identity()
        self.head=MLPHead_TM(cfg["head"]["in_features"],cfg["num_classes"],cfg["head"]["hidden_dim"],cfg["head"]["dropout"])
    def forward(self,x):
        x=self.model.patch_embed(x); x=self.model.pos_drop(x)
        x=self.model.features(x); x=self.model.norm(x); x=x.mean(dim=(2,3))
        return self.head(x)

class CustomVideoMAE(nn.Module):
    """VideoMAE backbone (ViT-S / ViT-B / ViT-L) via HuggingFace.
    Input: (B, C, T, H, W). VideoMAE expects (B, T, C, H, W) so we permute first.
    Output is (B, seq_len, hidden_size); MLPHead_TM averages over seq_len."""
    def __init__(self, cfg):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(cfg["backbone"]["pretrained"])
        self.head = MLPHead_TM(
            cfg["head"]["in_features"], cfg["num_classes"],
            cfg["head"]["hidden_dim"], cfg["head"]["dropout"]
        )
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W) -> (B,T,C,H,W)
        out = self.backbone(pixel_values=x).last_hidden_state  # (B, seq_len, hidden_size)
        return self.head(out)  # MLPHead_TM averages over dim=1

def build_model(cfg):
    n=cfg["backbone"]["name"]
    if n=="TimesformerModel": return CustomTimeSformer(cfg)
    elif n=="CustomSwin3D": return CustomSwin3D(cfg)
    elif n=="VideoMAEModel": return CustomVideoMAE(cfg)
    else: raise ValueError(f"Unknown backbone: {n}")

def rebuild_head(model, cfg, new_nc):
    hd=cfg["head"]["hidden_dim"]; dr=cfg["head"]["dropout"]; inf=cfg["head"]["in_features"]
    pool=cfg["head"].get("pool","cls_token")
    # temporal_mean and sequence_mean both use MLPHead_TM (mean over dim=1).
    # The names differ for documentation (Swin3D averages temporal+spatial inside
    # the backbone leaving a (B, hidden) tensor before head, so its "pool" is
    # cosmetic; VideoMAE outputs (B, seq, hidden) so MLPHead_TM averages seq).
    if pool in ("temporal_mean", "sequence_mean"):
        model.head=MLPHead_TM(inf,new_nc,hd,dr)
    else:
        model.head=MLPHead_CLS(inf,new_nc,hd,dr)
    return model

# ====================== Preprocess + Augmentation ======================

def uniform_sample(frames,t):
    n=len(frames)
    if n==t: return frames
    if n<t: return frames+[frames[-1]]*(t-n)
    return [frames[i] for i in np.linspace(0,n-1,t,dtype=int)]

def preprocess(frames,cfg,skip_resize=False):
    """Convert PIL frames → normalized tensor (C, T, H, W).

    If skip_resize=True, assumes frames are already at the target size (faster,
    when augmentation has already resized them).
    """
    sz=cfg["backbone"]["input_size"]; nf=cfg["backbone"]["num_frames"]
    m=cfg["input_format"]["normalize"]["mean"]; s=cfg["input_format"]["normalize"]["std"]
    r=frames if skip_resize else [f.resize((sz,sz),Image.BILINEAR) for f in frames]
    if len(r)!=nf: r=uniform_sample(r,nf)
    v=torch.stack([ToTensor()(f) for f in r],0)
    return ((v-torch.tensor(m).view(1,-1,1,1))/torch.tensor(s).view(1,-1,1,1)).permute(1,0,2,3)

def random_blur(frames,frac=0.35,rng=None):
    if frac<=0 or not frames: return frames
    rng=rng or random; n=len(frames); k=max(1,int(round(n*frac))); idxs=rng.sample(range(n),k)
    return [f.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8,2.2))) if i in idxs else f for i,f in enumerate(frames)]

def temporal_dropout(frames,frac=0.15,rng=None):
    if frac<=0 or len(frames)<3: return frames
    rng=rng or random; n=len(frames); k=max(1,int(round(n*frac)))
    idxs=rng.sample(range(1,n-1),min(k,max(1,n-2))); out=frames[:]
    for i in idxs: out[i]=out[i-1] if rng.random()<0.5 else out[i+1]
    return out

def horizontal_flip(frames,prob=0.5,rng=None):
    """Roboflow-style horizontal flip. Applied to the whole window (all frames) consistently."""
    if prob<=0 or not frames: return frames
    rng=rng or random
    if rng.random()<prob:
        return [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
    return frames

def vertical_flip(frames,prob=0.5,rng=None):
    """Roboflow-style vertical flip. Applied to the whole window (all frames) consistently."""
    if prob<=0 or not frames: return frames
    rng=rng or random
    if rng.random()<prob:
        return [f.transpose(Image.FLIP_TOP_BOTTOM) for f in frames]
    return frames

def random_rotation(frames,max_deg=0.0,rng=None):
    """Rotate the whole window by the same random angle in [-max_deg, +max_deg]."""
    if max_deg<=0 or not frames: return frames
    rng=rng or random
    ang=rng.uniform(-max_deg,max_deg)
    if abs(ang)<0.1: return frames
    return [f.rotate(ang,resample=Image.BILINEAR) for f in frames]

def color_jitter(frames,brightness=0.0,contrast=0.0,saturation=0.0,rng=None):
    """Roboflow-style brightness/contrast/saturation jitter. Same factor applied to every frame in the window."""
    if not frames: return frames
    if brightness<=0 and contrast<=0 and saturation<=0: return frames
    from PIL import ImageEnhance
    rng=rng or random
    out=frames
    if brightness>0:
        f=1.0+rng.uniform(-brightness,brightness)
        out=[ImageEnhance.Brightness(x).enhance(f) for x in out]
    if contrast>0:
        f=1.0+rng.uniform(-contrast,contrast)
        out=[ImageEnhance.Contrast(x).enhance(f) for x in out]
    if saturation>0:
        f=1.0+rng.uniform(-saturation,saturation)
        out=[ImageEnhance.Color(x).enhance(f) for x in out]
    return out

# ====================== BORIS Label Parsing ======================
# BORIS export format: one row per START/STOP event.
# Key columns: Behavior, Behavior type (START/STOP), Time (seconds), FPS.
# Strategy: use actual video FPS from VideoReader (most accurate),
#           fall back to BORIS CSV FPS column, then finally to int(csv_fps).
# Image index column is intentionally ignored — it is often inaccurate.

def is_boris_csv(lp):
    """Return True if the CSV looks like a BORIS event export."""
    try:
        df = pd.read_csv(lp, nrows=2)
        required = {"Behavior", "Behavior type", "Time"}
        return required.issubset(set(df.columns))
    except Exception:
        return False


def boris_to_onehot(lp, total_frames, video_fps):
    """
    Convert a BORIS CSV to a per-frame one-hot (multi-hot) numpy array.

    Returns
    -------
    onehot : np.ndarray, shape (total_frames, n_behaviors + 1)
        Last column is "Other" (1 when no behaviour is active).
    behavior_names : list[str]
        Column names in order, with "Other" appended last.

    Notes
    -----
    - FPS priority: video_fps (from VideoReader) > BORIS 'FPS' column > fallback 25.
    - Frame index = int(time_seconds * fps), clipped to [0, total_frames).
    - START/STOP count mismatch for a behaviour → that behaviour is skipped with a warning.
    - Overlapping behaviours are allowed (multi-hot).
    """
    df = pd.read_csv(lp)

    # ---- resolve FPS ----
    fps = video_fps  # prefer real video fps
    if fps <= 0 and "FPS" in df.columns:
        try:
            fps_vals = pd.to_numeric(df["FPS"], errors="coerce").dropna()
            if len(fps_vals) > 0:
                fps = float(fps_vals.iloc[0])
        except Exception:
            pass
    if fps <= 0:
        fps = 25.0
        print(f"⚠️  Could not determine FPS for {lp}, defaulting to 25")

    # ---- collect unique behaviors (preserve CSV order) ----
    all_behaviors = list(dict.fromkeys(df["Behavior"].dropna().tolist()))

    n_beh = len(all_behaviors)
    onehot = np.zeros((total_frames, n_beh), dtype=np.int8)

    for bi, beh in enumerate(all_behaviors):
        bdf = df[df["Behavior"] == beh]
        starts = bdf[bdf["Behavior type"] == "START"]["Time"].values.astype(float)
        stops  = bdf[bdf["Behavior type"] == "STOP"]["Time"].values.astype(float)

        if len(starts) != len(stops):
            print(f"⚠️  {beh}: START/STOP mismatch ({len(starts)}/{len(stops)}) — skipping")
            continue

        for t_start, t_stop in zip(starts, stops):
            f_start = int(t_start * fps)
            f_stop  = min(int(t_stop  * fps), total_frames)  # inclusive end → clip
            if f_start >= total_frames:
                continue
            onehot[f_start:f_stop, bi] = 1

    # ---- build "Other" column ----
    other_col = (onehot.sum(axis=1) == 0).astype(np.int8).reshape(-1, 1)
    onehot_full = np.concatenate([onehot, other_col], axis=1)
    behavior_names = all_behaviors + ["Other"]

    return onehot_full, behavior_names


def align_onehot_to_global(oh, col_names, global_names):
    """
    Re-arrange a (T, n_local) onehot to match a global column order (T, n_global).
    Columns present locally are copied to their global slot; missing columns are
    filled with zeros. After realignment, frames where all behaviour columns are
    zero are re-routed to the global "Other" column (so the file still
    contributes valid labels even if it has fewer behaviours than the dataset).

    This fixes the silent bug where np.argmax on an unaligned per-file onehot
    points to the wrong class in the global namespace.
    """
    T = oh.shape[0]
    n_global = len(global_names)
    out = np.zeros((T, n_global), dtype=oh.dtype)

    # Map each local column to its global index (if any)
    for local_i, name in enumerate(col_names):
        if name in global_names:
            g = global_names.index(name)
            out[:, g] = oh[:, local_i]
        # else: this file has a behaviour the global set doesn't know — drop it

    # If the global set has an "Other" column, make sure rows that are
    # all-zero in the behaviour cols (i.e. "no behaviour active") still mark
    # Other = 1, even if this file's local oh didn't have an Other column.
    if "Other" in global_names:
        other_g = global_names.index("Other")
        # behaviour mask = global cols except Other
        beh_mask = [i for i in range(n_global) if i != other_g]
        if beh_mask:
            no_beh = (out[:, beh_mask].sum(axis=1) == 0)
            out[no_beh, other_g] = 1
        else:
            out[:, other_g] = 1
    return out


# In-memory cache: lp -> (onehot_array, behavior_names)
# Avoids re-parsing on every window during training.
_BORIS_CACHE: dict = {}


def load_label_data(lp, total_frames, video_fps):
    """
    Unified label loader.  Returns (onehot_array, behavior_names).

    - Standard one-hot CSV  → read directly with pandas.
    - BORIS event CSV       → convert via boris_to_onehot() with caching.
    """
    if is_boris_csv(lp):
        cache_key = (lp, total_frames, round(video_fps, 4))
        if cache_key not in _BORIS_CACHE:
            onehot, names = boris_to_onehot(lp, total_frames, video_fps)
            _BORIS_CACHE[cache_key] = (onehot, names)
        return _BORIS_CACHE[cache_key]
    else:
        # Original one-hot format
        df = pd.read_csv(lp)
        return df.values.astype(np.int8), list(df.columns)


# ====================== Dataset ======================


class SlidingWindowDataset(Dataset):
    def __init__(self,video_paths,label_paths,ws,stride,cfg,nc,label_map,skip=0,augment=None):
        self.cfg=cfg; self.ws=ws; self.augment=augment; self.samples=[]; self.sample_labels=[]
        self._input_size=cfg["backbone"]["input_size"]  # cache for fast path
        # Global label order from the scan stage. Per-file onehots will be
        # realigned to this order so that argmax produces correct global indices.
        global_names = S.get("label_names", []) or []
        for vp,lp in zip(video_paths,label_paths):
            try:
                vr=VideoReader(vp,ctx=cpu(0)); T=len(vr); fps=vr.get_avg_fps()
                del vr  # free decord's frame cache (see scan stage comment)

                # ---- unified label loading (handles both BORIS and one-hot) ----
                oh, col_names = load_label_data(lp, T, fps)
                if global_names and col_names != global_names:
                    oh = align_onehot_to_global(oh, col_names, global_names)

                if T!=len(oh): print(f"⚠️ Length mismatch {vp}"); continue
                raw=np.argmax(oh,axis=1); mapped=np.array([label_map.get(int(l),-1) for l in raw])
                sel=list(range(0,T,skip+1)); valid=[i for i in sel if i<len(mapped) and mapped[i]>=0]
                if len(valid)<ws: continue
                for s in range(0,len(valid)-ws+1,stride):
                    idx=valid[s:s+ws]; lbl=Counter(mapped[idx]).most_common(1)[0][0]
                    self.samples.append((vp,idx,int(lbl))); self.sample_labels.append(int(lbl))
            except Exception as e: print(f"⚠️ Skipped {vp}: {e}"); continue
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        vp,idx,lbl=self.samples[i]
        # NOTE: we open a fresh VideoReader per call. A per-worker cache was tried
        # but decord + torch DataLoader workers have known memory-accumulation issues
        # (see dmlc/decord#280) — caching makes Colab OOM / workers crash silently.
        vr=VideoReader(vp,ctx=cpu(0))
        frames=[Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
        if len(frames)<self.ws: frames+=[frames[-1]]*(self.ws-len(frames))
        # Resize EARLY — augmentation runs on 224x224 instead of e.g. 1920x1080.
        # Rotation / color jitter / blur are ~50-70x faster at small resolution.
        sz=self._input_size
        frames=[f.resize((sz,sz),Image.BILINEAR) for f in frames]
        if self.augment: frames=self.augment(frames)
        return preprocess(frames,self.cfg,skip_resize=True),torch.tensor(lbl,dtype=torch.long)

# ====================== State ======================

S = {"model":None,"cfg":None,"scan_data":None,"label_names":[],"cur_vf":None,"cur_vr":None,
     "_cursor_data":json.dumps({"T":0,"names":[],"labels":[]}),"train_log":[],"split_indices":{"train":[],"val":[]},
     "_cancel_training":False}

CLR_PAL=["#378ADD","#D85A30","#E24B4A","#7F77DD","#1D9E75","#BA7517",
         "#534AB7","#993C1D","#639922","#D4537E","#185FA5","#854F0B","#A32D2D"]
U=gr.update()

def get_clr(i,name):
    if name.lower() in ("other","others"): return "#FFFFFF","rgba(180,180,180,0.9)"
    c=CLR_PAL[i%len(CLR_PAL)]; r,g,b=int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
    return c,f"rgba({r},{g},{b},0.9)"

# ====================== Model Management ======================

# Built-in backbones — no HF repo download needed.
# Weights are fetched from torchvision / huggingface-transformers cache the first
# time the backbone class is instantiated, then reused from local cache.
BUILTIN_MODELS = {
    "[BUILTIN] Swin3D-T (K400)": {
        "backbone": {
            "name": "CustomSwin3D",
            "variant": "t",
            "source": "torchvision",
            "pretrained": "Swin3D_T_Weights.DEFAULT",
            "hidden_size": 768,
            "num_frames": 16,
            "input_size": 224,
        },
        "head": {
            "type": "MLPHead", "in_features": 768, "hidden_dim": 512,
            "dropout": 0.3, "activation": "ReLU", "norm": "LayerNorm",
            "pool": "temporal_mean",
        },
        "num_classes": 0,
        "class_names": [],
        "input_format": {
            "shape": "(B, C, T, H, W)", "C": 3, "T": 16, "H": 224, "W": 224,
            "normalize": {"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
        },
    },
    "[BUILTIN] TimeSformer-B (K400)": {
        "backbone": {
            "name": "TimesformerModel",
            "source": "huggingface",
            "pretrained": "facebook/timesformer-base-finetuned-k400",
            "hidden_size": 768,
            "num_frames": 8,
            "input_size": 224,
        },
        "head": {
            "type": "MLPHead", "in_features": 768, "hidden_dim": 512,
            "dropout": 0.3, "activation": "ReLU", "norm": "LayerNorm",
            "pool": "cls_token",
        },
        "num_classes": 0,
        "class_names": [],
        "input_format": {
            "shape": "(B, C, T, H, W)", "C": 3, "T": 8, "H": 224, "W": 224,
            "normalize": {"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
        },
    },
    "[BUILTIN] VideoMAE ViT-S (K400)": {
        "backbone": {
            "name": "VideoMAEModel",
            "source": "huggingface",
            "pretrained": "MCG-NJU/videomae-small-finetuned-kinetics",
            "hidden_size": 384,
            "num_frames": 16,
            "input_size": 224,
        },
        "head": {
            "type": "MLPHead", "in_features": 384, "hidden_dim": 512,
            "dropout": 0.1, "activation": "ReLU", "norm": "LayerNorm",
            "pool": "sequence_mean",
        },
        "num_classes": 0,
        "class_names": [],
        "input_format": {
            "shape": "(B, C, T, H, W)", "C": 3, "T": 16, "H": 224, "W": 224,
            "normalize": {"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
        },
    },
}

def list_models(repo):
    builtin_names = list(BUILTIN_MODELS.keys())
    hf_names = []
    hf_err = None
    try:
        files = list_repo_files(repo)
        pths = [f for f in files if f.endswith("/model.pth") or f == "model.pth"]
        if not pths: pths = [f for f in files if f.endswith(".pth")]
        hf_names = [os.path.dirname(p) if "/" in p else p for p in pths]
    except Exception as e:
        hf_err = str(e)
    names = builtin_names + hf_names
    msg = f"✅ {len(builtin_names)} builtin + {len(hf_names)} HF model(s)"
    if hf_err: msg += f"  (HF list failed: {hf_err})"
    return gr.update(choices=names, value=names[0] if names else None), msg

def load_pretrained(repo, mname):
    """Returns status_text. Window is fixed at 16, stride at 4."""
    if not mname:
        return "❌ Select a model first"
    try:
        # ----- Built-in path: build fresh from local torchvision / HF cache -----
        if mname in BUILTIN_MODELS:
            import copy as _copy
            cfg = _copy.deepcopy(BUILTIN_MODELS[mname])
            cfg["num_classes"] = 1
            model = build_model(cfg)
            model.to(device)
            cfg["num_classes"] = 0
            cfg["class_names"] = []
            S.update({"model": model, "cfg": cfg, "train_log": []})
            nf = cfg["backbone"]["num_frames"]
            return (f"✅ Loaded: {mname}\n"
                    f"  Backbone: {cfg['backbone']['name']}\n"
                    f"  Window: 16 frames · Stride: 4 · Model frames: {nf}\n"
                    f"  No pretrained head — use 'New head' mode\n"
                    f"  Device: {device}")

        # ----- HF repo path (existing behaviour) -----
        if not repo:
            return "❌ Specify repo for HF model"
        if not mname.endswith(".pth"): cf=f"{mname}/config.json"; pf=f"{mname}/model.pth"
        else: cf="config.json"; pf=mname
        with open(hf_hub_download(repo_id=repo,filename=cf)) as f: cfg=json.load(f)
        model=build_model(cfg)
        model.load_state_dict(torch.load(hf_hub_download(repo_id=repo,filename=pf),map_location=device,weights_only=True))
        model.to(device); S.update({"model":model,"cfg":cfg,"train_log":[]})
        nf = cfg["backbone"].get("num_frames", 16)
        return (f"✅ Loaded: {mname}\n  Backbone: {cfg['backbone']['name']}\n"
                f"  Classes: {cfg['class_names']}\n  Window: 16 · Stride: 4 · Model frames: {nf}\n"
                f"  Device: {device}")
    except Exception as e:
        traceback.print_exc()
        return f"❌ {e}"

# ====================== Train/Val Split ======================

def compute_split(val_pct, seed=1337):
    if not S["scan_data"]: S["split_indices"]={"train":[],"val":[]}; return
    data=S["scan_data"]; n=len(data); val_ratio=val_pct/100.0
    if val_ratio>0 and n>=4: tidx,vidx=train_test_split(list(range(n)),test_size=val_ratio,random_state=int(seed))
    elif val_ratio>0 and n>=2: vidx=[n-1]; tidx=list(range(n-1))
    else: tidx=list(range(n)); vidx=[]
    S["split_indices"]={"train":tidx,"val":vidx}

def build_video_list_html(active_vf=None):
    if not S["scan_data"]: return "<p style='color:#aaa;font-size:12px;'>Load data first</p>"
    data=S["scan_data"]; tidx_set=set(S["split_indices"].get("train",[])); vidx_set=set(S["split_indices"].get("val",[]))
    html="<div style='max-height:200px;overflow-y:auto;border:1px solid var(--color-border-secondary);border-radius:8px;padding:4px;'>"
    for i,d in enumerate(data):
        vf=d["vf"]; T=d["T"]; fps=d["fps"]; dur=T/fps if fps>0 else 0
        fmt_tag = "<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#EDE9FE;color:#5B21B6;font-weight:600;margin-left:4px;'>BORIS</span>" if d.get("is_boris") else ""
        is_active=(vf==active_vf); is_val=(i in vidx_set)
        bg="background:rgba(220,38,38,0.12);border-left:3px solid #dc2626;" if is_active else "background:transparent;border-left:3px solid transparent;"
        if is_val: role_tag="<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#FEF3C7;color:#92400E;font-weight:600;margin-left:6px;'>VAL</span>"
        elif i in tidx_set: role_tag="<span style='font-size:10px;padding:1px 5px;border-radius:3px;background:#D1FAE5;color:#065F46;font-weight:600;margin-left:6px;'>TRAIN</span>"
        else: role_tag=""
        nc="#dc2626" if is_active else "var(--color-text-primary)"; nw="700" if is_active else "500"
        html+=f"<div style='{bg}border-radius:4px;padding:4px 8px;margin-bottom:1px;'><div style='display:flex;justify-content:space-between;align-items:center;'><span style='font-size:12px;color:{nc};font-weight:{nw};'>{vf}{fmt_tag}{role_tag}</span><span style='font-size:10px;color:var(--color-text-secondary);white-space:nowrap;'>{T} fr · {dur:.1f}s</span></div></div>"
    html+="</div>"
    nt=len(tidx_set); nv=len(vidx_set)
    html+=f"<div style='display:flex;gap:14px;margin-top:4px;font-size:11px;color:var(--color-text-secondary);'><span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#D1FAE5;border:1px solid #065F46;vertical-align:middle;margin-right:3px;'></span>Train: {nt}</span><span><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#FEF3C7;border:1px solid #92400E;vertical-align:middle;margin-right:3px;'></span>Val: {nv}</span><span>Total: {len(data)}</span></div>"
    return html

# ====================== Label Mapping Logic ======================

def fuzzy_match(data_name, pretrained_names):
    """Find best fuzzy match for a data label among pretrained class names."""
    dn = data_name.lower().replace("_"," ")
    best_score, best_match = 0, None
    for pn in pretrained_names:
        pnl = pn.lower().replace("_"," ")
        if dn == pnl: return pn  # exact
        score = SequenceMatcher(None, dn, pnl).ratio()
        # Also check substring containment
        if dn in pnl or pnl in dn: score = max(score, 0.75)
        if score > best_score: best_score = score; best_match = pn
    return best_match if best_score >= 0.55 else None

def build_mapping_choices_pt(idx, data_labels, pretrained_names):
    """Pretrain head: choices = pretrained classes + Exclude (+ → Other only if no other/others in data)."""
    has_other = any(n.lower() in ("other","others") for n in data_labels)
    choices = list(pretrained_names)
    if not has_other: choices.append("→ Other")
    choices.append("→ Exclude")
    # Default: fuzzy match or pretrained others
    default = fuzzy_match(data_labels[idx], pretrained_names)
    if default is None:
        if "others" in pretrained_names: default = "others"
        elif not has_other: default = "→ Other"
        else: default = choices[0]
    return choices, default

def build_mapping_choices_new(idx, data_labels, all_mappings):
    """New head: choices = keep / merge into available / (→ Other if needed) / Exclude."""
    has_other = any(n.lower() in ("other","others") for n in data_labels)
    consumed = set()
    for i_str, val in all_mappings.items():
        i = int(i_str)
        if i == idx: continue
        if val not in (None, "", "keep", "→ Other", "→ Exclude"): consumed.add(i)
    choices = [f"{data_labels[idx]} (keep)"]
    for j, nm in enumerate(data_labels):
        if j == idx or j in consumed: continue
        choices.append(f"→ merge into {nm}")
    if not has_other: choices.append("→ Other")
    choices.append("→ Exclude")
    return choices

def parse_mapping_value(val, data_labels):
    """Parse a dropdown value to a mapping dict entry."""
    if val is None or val == "" or "(keep)" in str(val): return "keep"
    if "merge into" in str(val): return val
    if val == "→ Other": return "→ Other"
    if val == "→ Exclude": return "→ Exclude"
    return val  # pretrain mode: val is a pretrained class name

def compute_label_map_from_dropdowns(mode, dd_values, data_labels, pretrained_names):
    """Convert dropdown values → (new_class_names, label_map{old_idx: new_idx or None}).
    Excluded labels → None (frames skipped in training & evaluation).
    new_names follows CSV column order for consistency with test code."""
    N = len(data_labels)

    if mode == "Pretrain head" and pretrained_names:
        mapping = {}
        other_list = []
        exclude_list = []
        for i in range(N):
            v = dd_values[i] if i < len(dd_values) else "→ Other"
            if v == "→ Exclude":
                exclude_list.append(i)
            elif v == "→ Other":
                other_list.append(i)
            else:
                mapping[i] = v
        if other_list:
            for i in other_list: mapping[i] = "others"
        used = set(mapping.values())
        new_names = [n for n in pretrained_names if n in used]
        if "others" in used and "others" not in new_names:
            new_names.append("others")
        label_map = {}
        for i in range(N):
            if i in exclude_list:
                label_map[i] = None
            elif i in mapping:
                label_map[i] = new_names.index(mapping[i])
        return new_names, label_map

    # New head mode — follow CSV column order
    kept_set = set()
    merge_targets = {}
    other_list = []
    exclude_list = []
    for i in range(N):
        v = dd_values[i] if i < len(dd_values) else "keep"
        if "(keep)" in str(v) or v == "keep":
            kept_set.add(data_labels[i])
        elif "merge into" in str(v):
            merge_targets[i] = v.replace("→ merge into ", "")
        elif v == "→ Other":
            other_list.append(i)
        elif v == "→ Exclude":
            exclude_list.append(i)

    new_names = [nm for nm in data_labels if nm in kept_set]
    if other_list:
        has_o = any(c.lower() in ("other","others") for c in new_names)
        if not has_o: new_names.append("Other")

    label_map = {}
    for i in range(N):
        if i in exclude_list:
            label_map[i] = None
            continue
        nm = data_labels[i]
        if nm in new_names:
            label_map[i] = new_names.index(nm)
        elif i in merge_targets:
            t = merge_targets[i]
            if t in new_names: label_map[i] = new_names.index(t)
            else: label_map[i] = new_names.index("Other") if "Other" in new_names else 0
        elif i in other_list:
            oidx = next((j for j,n in enumerate(new_names) if n.lower() in ("other","others")), len(new_names)-1)
            label_map[i] = oidx
        else:
            label_map[i] = 0

    return new_names, label_map

# ====================== Mapped Timeline ======================

def build_mapped_timeline(vf, mapped_names, label_map):
    """Build timeline HTML using mapped labels. None = excluded (shown as dim gray)."""
    if not S["scan_data"] or not vf: return "", S["_cursor_data"]
    d = next((x for x in S["scan_data"] if x["vf"]==vf), None)
    if not d: return "", S["_cursor_data"]

    T=d["T"]; fps=d["fps"]; raw_labels=d["labels"]
    # Map raw labels to new indices; None → -1 (excluded)
    mapped_labels = [label_map.get(l, 0) if label_map.get(l) is not None else -1 for l in raw_labels]
    names = mapped_names

    # Build bar
    segs=[]; cur,cnt=mapped_labels[0],1
    for i in range(1,T):
        if mapped_labels[i]==cur: cnt+=1
        else: segs.append((cur,cnt)); cur,cnt=mapped_labels[i],1
    segs.append((cur,cnt))

    bar=""
    for li,c in segs:
        if li == -1:
            nm="excluded"; clr="#D0D0D0"; bdr=""
        else:
            nm=names[li] if li<len(names) else "?"
            clr,_=get_clr(li,nm)
            bdr="border-top:1px solid #ccc;border-bottom:1px solid #ccc;" if clr=="#FFFFFF" else ""
        pct=(c/T)*100
        bar+=f"<div style='width:{pct:.3f}%;height:100%;background:{clr};{bdr}display:inline-block;box-sizing:border-box;opacity:{0.35 if li==-1 else 1};' title='{nm}'></div>"

    leg=""
    for i,nm in enumerate(names):
        clr,_=get_clr(i,nm); bdr="border:1px solid #ccc;" if clr=="#FFFFFF" else ""
        leg+=f"<span style='display:inline-flex;align-items:center;gap:3px;margin-right:10px;font-size:11px;color:var(--color-text-secondary);'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:{clr};{bdr}'></span>{nm}</span>"
    # Add excluded legend if any excluded frames
    if -1 in mapped_labels:
        leg+=f"<span style='display:inline-flex;align-items:center;gap:3px;margin-right:10px;font-size:11px;color:var(--color-text-tertiary);'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:#D0D0D0;opacity:0.5;'></span>excluded</span>"

    ml0=mapped_labels[0]
    nm0=names[ml0] if ml0>=0 and ml0<len(names) else "excluded"
    tl=f"""<div style='width:100%;padding:4px 0;'>
      <div style='position:relative;display:flex;height:16px;border-radius:4px;overflow:hidden;border:1px solid #ccc;'>
        {bar}
        <div id='tl-cursor' style='position:absolute;top:-2px;bottom:-2px;width:2px;background:#000;box-shadow:0 0 0 1px rgba(255,255,255,0.8);left:0%;pointer-events:none;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-top:3px;'>
        <div>{leg}</div>
        <span id='tl-frame-label' style='font-size:11px;font-weight:500;color:var(--color-text-secondary);'>F:0 {nm0}</span>
      </div></div>"""

    cursor_data = json.dumps({"T":T, "names":names, "labels":mapped_labels})
    S["_cursor_data"] = cursor_data
    return tl, cursor_data

def build_mapping_summary_html(mode, dd_values, data_labels, pretrained_names):
    """Build HTML summary of the mapping result."""
    new_names, label_map = compute_label_map_from_dropdowns(mode, dd_values, data_labels, pretrained_names)
    html = f"<div style='padding:6px 10px;background:var(--color-background-secondary);border-radius:8px;font-size:11px;color:var(--color-text-secondary);line-height:1.6;'>"
    html += f"<span style='font-weight:500;'>Training classes ({len(new_names)}):</span> {', '.join(new_names)}"
    html += "</div>"
    return html

# ====================== Label Distribution HTML ======================

def build_label_dist_html():
    if not S["scan_data"] or not S["label_names"]: return "<p style='color:#aaa;'>Load data to see labels</p>"
    all_label_names=S["label_names"]; matched=S["scan_data"]; total_frames=sum(d["T"] for d in matched)
    gcounts=Counter()
    for d in matched:
        for k,v in d["counts"].items(): gcounts[k]+=v
    html="<div style='padding:4px 0;'>"
    html+="<p style='font-size:13px;font-weight:500;margin:0 0 6px;'>Label distribution</p>"
    for i,nm in enumerate(all_label_names):
        c=gcounts.get(i,0); pct=100*c/max(total_frames,1); clr,_=get_clr(i,nm)
        bar_clr="#ddd" if clr=="#FFFFFF" else clr
        html+=f"<div style='margin-bottom:5px;'><div style='display:flex;align-items:center;gap:6px;margin-bottom:1px;'><span style='display:inline-block;width:8px;height:8px;border-radius:2px;background:{bar_clr};flex-shrink:0;{("border:1px solid #ccc;" if clr=="#FFFFFF" else "")}'></span><span style='font-size:12px;font-weight:500;flex:1;'>{nm}</span><span style='font-size:11px;color:#888;flex-shrink:0;'>{c:,} fr · {pct:.1f}%</span></div><div style='height:5px;background:#f0f0f0;border-radius:3px;overflow:hidden;margin-left:14px;'><div style='width:{max(pct,0.3):.1f}%;height:100%;background:{bar_clr};border-radius:3px;'></div></div></div>"
    html+="</div>"
    return html

# ====================== Local video cache ======================

# When videos live on Google Drive (FUSE-mounted in Colab), every VideoReader
# open + every frame seek goes through the network. For training/inference
# this is 5-30x slower than reading from local disk and also more prone to
# DataLoader worker timeouts. Caching the videos to /content/oab_video_cache/
# at scan time avoids both problems. Skipped on re-runs if size matches.

VIDEO_CACHE_DIR = "/content/oab_video_cache"

def cache_video_to_local(src_path, cache_dir=VIDEO_CACHE_DIR):
    """Copy src_path to cache_dir and return the local path. If a same-sized
    file is already there, skip the copy. Returns the original path if copy
    fails (so the caller can still proceed with the Drive path)."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, os.path.basename(src_path))
        # Skip if cached copy exists and size matches the source
        if os.path.exists(dst):
            try:
                if os.path.getsize(dst) == os.path.getsize(src_path):
                    return dst
            except Exception:
                pass  # fall through and re-copy
        shutil.copy2(src_path, dst)
        return dst
    except Exception as e:
        print(f"⚠️ Failed to cache {src_path}: {e}; falling back to original path")
        return src_path

# ====================== Data Scanning ======================

def do_scan_and_preview(vdir, ldir, val_pct, val_seed, head_mode, *dd_vals):
    N = MAX_LABELS
    empty = lambda msg: (msg,"","*Load data first*",
                         *[gr.update(visible=False,choices=[],value=None) for _ in range(N)],
                         gr.update(choices=[],value=None),
                         None,"","",gr.update(maximum=1,value=0),S["_cursor_data"],"","")

    if not vdir or not os.path.isdir(vdir): return empty(f"❌ Video dir not found")
    if not ldir or not os.path.isdir(ldir): return empty(f"❌ Label dir not found")

    vfiles=sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4",".avi",".mov"))])
    if not vfiles: return empty("❌ No videos found")

    # Build a lookup of all CSV files in label dir for flexible matching
    csv_files = {os.path.splitext(f)[0].lower(): os.path.join(ldir, f)
                 for f in os.listdir(ldir) if f.lower().endswith(".csv")}

    matched=[]; all_label_names=None
    boris_count=0; onehot_count=0

    for vf in vfiles:
        base=os.path.splitext(vf)[0]; lp=None
        # Try exact match first, then flexible matching
        for candidate in [base, base.replace("-",""), base.replace("_",""), base.replace("-","_"), base.replace("_","-")]:
            # Try candidate.csv
            fp=os.path.join(ldir, candidate+".csv")
            if os.path.exists(fp): lp=fp; break
            # Try candidate_one_hot.csv
            fp=os.path.join(ldir, candidate+"_one_hot.csv")
            if os.path.exists(fp): lp=fp; break
            # Try case-insensitive lookup
            if candidate.lower() in csv_files:
                lp=csv_files[candidate.lower()]; break
        if lp is None: continue
        vp = os.path.join(vdir, vf)
        try:
            vr=VideoReader(vp,ctx=cpu(0)); T=len(vr); fps=vr.get_avg_fps()
            # Free decord's frame cache as soon as we have what we need.
            # Without this, scanning many videos accumulates ~tens of MB each
            # in cached decoded frames and can OOM on Colab.
            del vr

            # ---- detect format and load ----
            boris = is_boris_csv(lp)
            oh, col_names = load_label_data(lp, T, fps)

            if boris:
                boris_count += 1
            else:
                onehot_count += 1

            # sanity check: one-hot CSVs must match frame count exactly
            if not boris and T != len(oh):
                print(f"⚠️ Length mismatch (one-hot) {vf}: video={T}, csv={len(oh)}")
                continue

            # Defer label computation until we know the global column union.
            # Storing oh + col_names per file lets us realign correctly later.
            matched.append({
                "vp": vp, "lp": lp, "vf": vf,
                "T": T, "fps": fps,
                "_oh_raw": oh, "_cols_raw": col_names,
                "is_boris": boris,
            })
        except Exception as e:
            print(f"⚠️ Skipped {vf}: {e}")
            continue

    if not matched: return empty("❌ No matched pairs")

    # ---- compute global label set as the UNION of all files' columns ----
    # This is critical: if file A has [dark eye patch, extended tentacle, Other]
    # and file B has only [dark eye patch, Other], we cannot just adopt the
    # first file's columns — we need every behaviour seen across the dataset.
    # Order = first-appearance order, then "Other" pushed to last if present.
    global_names = []
    for d in matched:
        for n in d["_cols_raw"]:
            if n not in global_names:
                global_names.append(n)
    if "Other" in global_names:
        global_names = [n for n in global_names if n != "Other"] + ["Other"]
    all_label_names = global_names

    # Now realign each file's onehot to the global column order, then compute
    # per-frame argmax labels in the global namespace.
    n_misaligned = 0
    for d in matched:
        oh_local = d.pop("_oh_raw")
        cols_local = d.pop("_cols_raw")
        if cols_local != all_label_names:
            n_misaligned += 1
            print(f"⚠️ Column set differs in {d['vf']} (has {cols_local}); realigning to global order")
        oh_global = align_onehot_to_global(oh_local, cols_local, all_label_names)
        labels = np.argmax(oh_global, axis=1)
        d["counts"] = Counter(labels.tolist())
        d["labels"] = labels.tolist()
    if n_misaligned:
        print(f"ℹ️ Realigned {n_misaligned}/{len(matched)} file(s) to global label order: {all_label_names}")

    S["scan_data"]=matched; S["label_names"]=all_label_names
    compute_split(val_pct, val_seed)
    dist=build_label_dist_html()

    # format summary for status bar
    fmt_parts = []
    if boris_count:   fmt_parts.append(f"{boris_count} BORIS")
    if onehot_count:  fmt_parts.append(f"{onehot_count} one-hot")
    fmt_str = f" ({', '.join(fmt_parts)})" if fmt_parts else ""

    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []

    # Build mapping dropdown updates
    dd_updates = []
    for i in range(N):
        if i < len(all_label_names):
            if head_mode == "Pretrain head" and pretrained_names:
                choices, default = build_mapping_choices_pt(i, all_label_names, pretrained_names)
            else:
                choices = build_mapping_choices_new(i, all_label_names, {})
                default = choices[0]
            dd_updates.append(gr.update(visible=False, choices=choices, value=default, label=all_label_names[i]))
        else:
            dd_updates.append(gr.update(visible=False, choices=[], value=None))

    # Video dropdown
    vnames=[d["vf"] for d in matched]
    vid_dd_update=gr.update(choices=vnames,value=vnames[0])

    # Preview first video with initial mapping
    vf=matched[0]["vf"]
    dd_values = [u["value"] for u in dd_updates[:len(all_label_names)]]
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, dd_values, all_label_names, pretrained_names)
    tl, cdata = build_mapped_timeline(vf, new_names, label_map)
    summary = build_mapping_summary_html(head_mode, dd_values, all_label_names, pretrained_names)

    img = _get_frame(vf, 0)
    d0 = next(x for x in matched if x["vf"]==vf)
    T=d0["T"]; fps=d0["fps"]
    ml = label_map.get(d0["labels"][0], 0)
    nm0 = new_names[ml] if ml is not None and ml < len(new_names) else "?"
    _,bg = get_clr(ml if ml is not None else 0, nm0)
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm0}</span><span style='font-size:12px;color:var(--color-text-secondary);'>F: 0 / {T} | 0.00s / {T/fps:.2f}s</span></div>"

    nav_t=f"**{vf}** — 1 / {len(matched)} videos"
    vid_list=build_video_list_html(active_vf=vf)
    status=f"✅ {len(matched)} matched (of {len(vfiles)} videos){fmt_str}"

    return (status, dist, nav_t, *dd_updates, vid_dd_update,
            img, info, tl, gr.update(maximum=max(T-1,1),value=0), cdata, vid_list, summary)


MAPPER_HTML = r"""
<div id="lm-wrap">
<style>
  #lm-wrap{--line:#5B7FC7;--line-hi:#D85A30;--other:#8a8a8a;--excl:#c0392b;
           font-family:inherit;color:#222}
  #lm-stage{position:relative;background:#fff;border:1px solid #e4e4e4;
            border-radius:10px;padding:14px;overflow:hidden}
  #lm-wires{position:absolute;inset:0;width:100%;height:100%;
            pointer-events:none;z-index:1}
  #lm-cols{position:relative;display:flex;gap:56px;justify-content:space-between;z-index:2}
  .lm-col{flex:1;max-width:260px}
  .lm-head{font-size:11px;font-weight:700;color:#666;letter-spacing:.04em;
           text-transform:uppercase;margin-bottom:8px;text-align:center}
  .lm-panel{border:1px solid #e5e7eb;background:#f9fafb;border-radius:8px;padding:10px;
            min-height:120px;display:flex;flex-direction:column;gap:8px}
  .lm-node{position:relative;background:#fff;border:1px solid #d1d5db;border-radius:8px;
           min-height:40px;display:flex;align-items:center;justify-content:center;
           font-size:13px;padding:8px 30px;text-align:center;cursor:pointer;
           user-select:none;transition:box-shadow .12s,border-color .12s}
  .lm-node:hover{border-color:#9ca3af;box-shadow:0 1px 4px rgba(0,0,0,.08)}
  .lm-node.sel{border-color:var(--line-hi);box-shadow:0 0 0 3px rgba(216,90,48,.16)}
  .lm-node.dim{opacity:.45}
  .lm-node.drop{border-color:#2e7d32;background:#f0f7f0;
                box-shadow:0 0 0 3px rgba(46,125,50,.15)}
  .lm-node.dragging{border-color:var(--line-hi);box-shadow:0 0 0 3px rgba(216,90,48,.16)}
  .lm-node.special{border-style:dashed;color:#777}
  .lm-node[data-kind="other"]{border-color:var(--other)}
  .lm-node[data-kind="exclude"]{border-color:var(--excl);color:var(--excl)}
  .lm-port{position:absolute;top:50%;transform:translateY(-50%);width:12px;height:12px;
           border-radius:50%;background:#fff;border:2px solid var(--line);z-index:3}
  .lm-port.out{right:-7px;cursor:grab}
  .lm-port.in{left:-7px}
  body.lm-dragging .lm-port.out{cursor:grabbing}
  .lm-del{position:absolute;top:50%;right:8px;transform:translateY(-50%);font-size:15px;
          color:#bbb;cursor:pointer;line-height:1;width:18px;height:18px;display:flex;
          align-items:center;justify-content:center;border-radius:4px}
  .lm-del:hover{color:#c0392b;background:#fde8e6}
  .lm-add{border:1px dashed #9ca3af;background:#fff;border-radius:8px;min-height:38px;
          display:flex;align-items:center;justify-content:center;gap:6px;font-size:13px;
          color:#6b7280;cursor:pointer}
  .lm-add:hover{border-color:#2e7d32;color:#2e7d32;background:#f0f7f0}
  input.lm-rename{font:inherit;font-size:13px;text-align:center;border:1px solid #2e7d32;
                  border-radius:5px;padding:3px 6px;width:82%;outline:none}
  #lm-hint{font-size:12px;color:#999;margin-top:9px;line-height:1.6}
  .lm-warn{color:#c07a1d}
</style>

<div id="lm-stage">
  <svg id="lm-wires"></svg>
  <div id="lm-cols">
    <div class="lm-col"><div class="lm-head">CSV labels</div>
      <div class="lm-panel" id="lm-left"></div></div>
    <div class="lm-col"><div class="lm-head">Training classes</div>
      <div class="lm-panel" id="lm-right"></div></div>
  </div>
  <div id="lm-hint"></div>
</div>
</div>
"""

# Behaviour for the mapper. Injected through an event's js= argument,
# because gr.HTML does not execute inline <script> tags in Gradio 6.
MAPPER_JS = r"""
() => {
  if (window.__lmReady) return;
  window.__lmReady = true;
  (function(){
  let left=[], right=[], links={}, pending=null, seq=100, MODE="new", LOCKED=false;
  let dragFrom=null, justDragged=false;

  function findBridge(){
    let el=document.querySelector("#lm_bridge textarea, #lm_bridge input");
    if(!el && window.parent && window.parent.document){
      el=window.parent.document.querySelector("#lm_bridge textarea, #lm_bridge input");
    }
    return el;
  }
  function push(){
    const el=findBridge();
    if(!el){ console.warn("[lm] bridge textarea not found — mapping not sent"); return; }
    const nameOf=id=>(right.find(r=>r.id===id)||{}).name;
    const kindOf=id=>(right.find(r=>r.id===id)||{}).kind;
    const payload={mode:MODE,
      classes:right.filter(r=>r.kind==="class").map(r=>r.name),
      links:{}};
    left.forEach((nm,i)=>{
      const rid=links[i];
      payload.links[nm]= rid===undefined ? null
        : (kindOf(rid)==="other" ? "__OTHER__"
        : (kindOf(rid)==="exclude" ? "__EXCLUDE__" : nameOf(rid)));
    });
    const proto = el.tagName==="TEXTAREA"
      ? Object.getPrototypeOf(el)           // works across realms/iframes
      : Object.getPrototypeOf(el);
    const desc = Object.getOwnPropertyDescriptor(proto, "value")
      || Object.getOwnPropertyDescriptor(
           el.ownerDocument.defaultView.HTMLTextAreaElement.prototype, "value");
    try { desc.set.call(el, JSON.stringify(payload)); }
    catch(e){ el.value = JSON.stringify(payload); }
    el.dispatchEvent(new Event("input",{bubbles:true}));
    el.dispatchEvent(new Event("change",{bubbles:true}));
  }

  function addNode(){
    const id="r"+(++seq);
    const ex=right.findIndex(r=>r.kind==="exclude");
    right.splice(ex<0?right.length:ex,0,{id,name:"NewClass",kind:"class"});
    render(); edit(id);
  }
  function delNode(id){
    const n=right.find(r=>r.id===id);
    if(!n||n.kind!=="class") return;
    right=right.filter(r=>r.id!==id);
    Object.keys(links).forEach(k=>{ if(links[k]===id) delete links[k]; });
    render();
  }
  function edit(id){
    const n=right.find(r=>r.id===id); if(!n||n.kind!=="class"||LOCKED) return;
    const box=document.querySelector('.lm-node[data-rid="'+id+'"]');
    const span=box&&box.querySelector(".lm-nm"); if(!span) return;
    const inp=document.createElement("input");
    inp.className="lm-rename"; inp.value=n.name;
    span.replaceWith(inp); inp.focus(); inp.select();
    const commit=()=>{ const v=inp.value.trim(); if(v) n.name=v; render(); };
    inp.onblur=commit;
    inp.onkeydown=e=>{ e.stopPropagation();
      if(e.key==="Enter") inp.blur();
      if(e.key==="Escape"){ inp.onblur=null; render(); } };
    inp.onclick=e=>e.stopPropagation();
    inp.onpointerdown=e=>e.stopPropagation();
  }
  function pick(i){ pending=(pending===i)?null:i; render(); }
  function connect(rid){ if(pending===null) return; links[pending]=rid; pending=null; render(); }

  function startDrag(e,i){
    if(e.button!==undefined&&e.button!==0) return;
    e.preventDefault(); dragFrom=i; justDragged=false; pending=null;
    document.body.classList.add("lm-dragging");
    window.addEventListener("pointermove",onDrag);
    window.addEventListener("pointerup",endDrag,{once:true});
  }
  function nodeUnder(e){
    const el=document.elementFromPoint(e.clientX,e.clientY);
    return el?el.closest(".lm-node[data-rid]"):null;
  }
  function onDrag(e){
    justDragged=true;
    document.querySelectorAll(".lm-node[data-rid]").forEach(n=>n.classList.remove("drop"));
    const t=nodeUnder(e); if(t) t.classList.add("drop");
    draw(e);
  }
  function endDrag(e){
    window.removeEventListener("pointermove",onDrag);
    document.body.classList.remove("lm-dragging");
    const t=nodeUnder(e), from=dragFrom; dragFrom=null;
    document.querySelectorAll(".lm-node").forEach(n=>n.classList.remove("drop","dragging"));
    if(t&&from!==null&&justDragged) links[from]=t.dataset.rid;
    render(); setTimeout(()=>{justDragged=false;},0);
  }

  function render(){
    const L=document.getElementById("lm-left"), R=document.getElementById("lm-right");
    if(!L||!R) return;
    L.innerHTML=""; R.innerHTML="";
    left.forEach((nm,i)=>{
      const d=document.createElement("div");
      d.className="lm-node"+(pending===i?" sel":"")+((pending!==null&&pending!==i)?" dim":"");
      d.dataset.li=i;
      d.innerHTML='<span>'+nm+'</span><span class="lm-port out"></span>';
      d.onpointerdown=e=>startDrag(e,i);
      d.onclick=()=>{ if(!justDragged) pick(i); };
      L.appendChild(d);
    });
    right.forEach(r=>{
      const d=document.createElement("div");
      d.className="lm-node"+(r.kind!=="class"?" special":"");
      d.dataset.rid=r.id; d.dataset.kind=r.kind;
      d.innerHTML='<span class="lm-port in"></span><span class="lm-nm">'+r.name+'</span>'
                + ((r.kind==="class"&&!LOCKED)?'<span class="lm-del" title="Delete">×</span>':'');
      d.onclick=e=>{ if(e.target.classList.contains("lm-del")){delNode(r.id);return;}
                     if(!justDragged) connect(r.id); };
      d.ondblclick=()=>edit(r.id);
      R.appendChild(d);
    });
    if(!LOCKED){
      const a=document.createElement("div");
      a.className="lm-add"; a.innerHTML="<span>+</span><span>Add class</span>";
      a.onclick=addNode; R.appendChild(a);
    }
    requestAnimationFrame(()=>draw());
    const unset=left.filter((n,i)=>links[i]===undefined);
    const rightClasses=right.filter(r=>r.kind==="class").length;
    let hint;
    if(LOCKED && rightClasses===0){
      hint='<span class="lm-warn">⚠ Pretrain head selected but no model loaded yet — '
           +'load a pretrained model to see its classes.</span>';
    } else {
      hint=(pending!==null?'<b style="color:var(--line-hi)">Selected "'+left[pending]
             +'" — now click a box on the right</b><br>':'')
           +(unset.length?'<span class="lm-warn">⚠ Not connected: '+unset.join(", ")
             +' (treated as unassigned)</span>'
             :'<span style="color:#2e7d32">✓ All labels mapped</span>');
    }
    document.getElementById("lm-hint").innerHTML=hint;
    push();
  }

  function draw(dragEvt){
    const svg=document.getElementById("lm-wires");
    const stage=document.getElementById("lm-stage");
    if(!svg||!stage) return;
    const st=stage.getBoundingClientRect(); svg.innerHTML="";
    Object.entries(links).forEach(([li,rid])=>{
      const a=document.querySelector('.lm-node[data-li="'+li+'"] .lm-port.out');
      const b=document.querySelector('.lm-node[data-rid="'+rid+'"] .lm-port.in');
      if(!a||!b) return;
      const ra=a.getBoundingClientRect(), rb=b.getBoundingClientRect();
      const x1=ra.left+ra.width/2-st.left, y1=ra.top+ra.height/2-st.top;
      const x2=rb.left+rb.width/2-st.left, y2=rb.top+rb.height/2-st.top;
      const p=document.createElementNS("http://www.w3.org/2000/svg","path");
      const mx=(x1+x2)/2;
      p.setAttribute("d","M"+x1+","+y1+" C"+mx+","+y1+" "+mx+","+y2+" "+x2+","+y2);
      p.setAttribute("stroke","var(--line)"); p.setAttribute("stroke-width","1.6");
      p.setAttribute("fill","none");
      p.style.pointerEvents="stroke"; p.style.cursor="pointer";
      p.onmouseenter=()=>p.setAttribute("stroke","var(--line-hi)");
      p.onmouseleave=()=>p.setAttribute("stroke","var(--line)");
      p.onclick=()=>{ delete links[li]; render(); };
      svg.appendChild(p);
    });
    if(dragFrom!==null&&dragEvt){
      const a=document.querySelector('.lm-node[data-li="'+dragFrom+'"] .lm-port.out');
      if(a){
        const ra=a.getBoundingClientRect();
        const x1=ra.left+ra.width/2-st.left, y1=ra.top+ra.height/2-st.top;
        const x2=dragEvt.clientX-st.left,    y2=dragEvt.clientY-st.top;
        const p=document.createElementNS("http://www.w3.org/2000/svg","path");
        const mx=(x1+x2)/2;
        p.setAttribute("d","M"+x1+","+y1+" C"+mx+","+y1+" "+mx+","+y2+" "+x2+","+y2);
        p.setAttribute("stroke","var(--line-hi)"); p.setAttribute("stroke-width","1.8");
        p.setAttribute("stroke-dasharray","5,4"); p.setAttribute("fill","none");
        svg.appendChild(p);
      }
    }
  }

  // Python → JS: seed the widget from the current scan / head mode
  window.lmInit=function(cfg){
    left=cfg.left||[]; MODE=cfg.mode||"new"; LOCKED=!!cfg.locked;
    right=(cfg.right||[]).map((o,k)=>({id:"r"+k,name:o.name,kind:o.kind}));
    links={};
    left.forEach((nm,i)=>{
      const want=(cfg.links||{})[nm];
      if(want===null||want===undefined) return;
      let hit;
      if(want==="__OTHER__")        hit=right.find(r=>r.kind==="other");
      else if(want==="__EXCLUDE__") hit=right.find(r=>r.kind==="exclude");
      else                          hit=right.find(r=>r.kind==="class"&&r.name===want);
      if(hit) links[i]=hit.id;
    });
    seq=right.length+10; pending=null;
    render();
  };
  window.addEventListener("resize",()=>draw());
  })();
}
"""



def _mapper_init_js(data_labels, head_mode, pretrained_names, dd_values):
    """Build the JS call that seeds the mapper from current dropdown state."""
    import json as _json
    # Pretrain head always locks the right column (its classes come from the
    # model, not the user). If the model isn't loaded yet there are no classes
    # to show — still locked, so the user can't add bogus ones.
    is_pre = (head_mode == "Pretrain head")
    locked = is_pre
    if is_pre:
        classes = [n for n in pretrained_names if n.lower() not in ("other", "others")]
    else:
        keep = []
        for i, nm in enumerate(data_labels):
            v = str(dd_values[i]) if i < len(dd_values) else "keep"
            if "(keep)" in v or v == "keep":
                keep.append(nm)
        classes = keep or [n for n in data_labels
                           if n.lower() not in ("other", "others")]

    # "Other" is a fixed box below, so never emit it as a normal class too
    classes = [c for c in classes if c.lower() not in ("other", "others")]

    right = [{"name": c, "kind": "class"} for c in classes]
    right.append({"name": "Other", "kind": "other"})
    right.append({"name": "Exclude", "kind": "exclude"})

    links = {}
    for i, nm in enumerate(data_labels):
        v = str(dd_values[i]) if i < len(dd_values) else ""
        if v == "→ Exclude":
            links[nm] = "__EXCLUDE__"
        elif v == "→ Other":
            links[nm] = "__OTHER__"
        elif "merge into" in v:
            tgt = v.replace("→ merge into ", "")
            links[nm] = "__OTHER__" if tgt.lower() in ("other", "others") else tgt
        elif "(keep)" in v or v == "keep":
            # an "Other (keep)" label belongs on the fixed Other box
            links[nm] = "__OTHER__" if nm.lower() in ("other", "others") else nm
        elif v and v not in ("", "None"):
            links[nm] = "__OTHER__" if v.lower() in ("other", "others") else v
        else:
            links[nm] = None

    cfg = {"left": list(data_labels), "right": right, "links": links,
           "mode": "pre" if locked else "new", "locked": locked}
    return _json.dumps(cfg)


def apply_mapper_bridge(bridge_json, head_mode, *dd_vals):
    """Bridge JSON (from the visual mapper) → dropdown values.

    The dropdowns remain the single source of truth for the rest of the app;
    the mapper is just a nicer way to edit them. Returns updates for all
    MAX_LABELS dropdown slots.
    """
    import json as _json
    N = MAX_LABELS
    data_labels = S["label_names"]
    n = len(data_labels)
    if not bridge_json or n == 0:
        return tuple(gr.update() for _ in range(N))
    try:
        payload = _json.loads(bridge_json)
    except Exception:
        return tuple(gr.update() for _ in range(N))

    links = payload.get("links", {})

    # Safety: a payload that carries no usable links must be IGNORED, not
    # applied. An empty/stale push (e.g. the mapper rendering before the scan
    # populated its labels) would otherwise route every label to Other and
    # silently destroy the whole class list.
    if not isinstance(links, dict) or not links:
        return tuple(gr.update() for _ in range(N))
    known = [nm for nm in data_labels if nm in links]
    if not known:
        return tuple(gr.update() for _ in range(N))
    if all(links.get(nm) is None for nm in known):
        return tuple(gr.update() for _ in range(N))

    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    is_pre = (head_mode == "Pretrain head" and bool(pretrained_names))

    # Data labels that are literally "Other" change how "route to Other" is
    # expressed: when an Other class exists, New head has NO "→ Other" choice —
    # you route there via "→ merge into Other" instead.
    has_other = any(nm.lower() in ("other", "others") for nm in data_labels)
    other_name = next((nm for nm in data_labels if nm.lower() in ("other", "others")), "Other")

    def route_to_other():
        if is_pre:
            return "→ Other"                       # pretrain always has this
        return "→ Other" if not has_other else f"→ merge into {other_name}"

    out = []
    for i in range(N):
        if i >= n:
            out.append(gr.update()); continue
        nm = data_labels[i]
        tgt = links.get(nm)

        if tgt is None:
            # Not connected: keep the label as its own class rather than
            # silently folding it into Other (which would delete the class).
            if is_pre:
                out.append(gr.update(value=route_to_other()))
            elif nm.lower() in ("other", "others"):
                out.append(gr.update(value=f"{nm} (keep)"))
            else:
                out.append(gr.update(value=f"{nm} (keep)"))
            continue
        if tgt == "__EXCLUDE__":
            out.append(gr.update(value="→ Exclude")); continue
        if tgt == "__OTHER__":
            # a label literally named "Other" keeps its own (keep) value in
            # New head mode; anything else routes to Other via the valid choice
            if not is_pre and nm.lower() in ("other", "others"):
                out.append(gr.update(value=f"{nm} (keep)"))
            else:
                out.append(gr.update(value=route_to_other()))
            continue

        if is_pre:
            out.append(gr.update(value=tgt if tgt in pretrained_names else "→ Other"))
        else:
            if tgt == nm:
                out.append(gr.update(value=f"{nm} (keep)"))
            elif tgt in data_labels or tgt in payload.get("classes", []):
                out.append(gr.update(value=f"→ merge into {tgt}"))
            else:
                # target no longer exists (box was deleted/renamed) — fall back
                out.append(gr.update(value=route_to_other()))
    return tuple(out)


# ====================== Mapping Change Handler ======================

def on_mapping_change(head_mode, *dd_vals):
    """When any mapping dropdown or head mode changes → rebuild all dropdowns + timeline + summary."""
    data_labels = S["label_names"]
    pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
    N = MAX_LABELS
    n = len(data_labels)

    if n == 0:
        # must match map_change_outputs: [*map_dds, timeline_html, cursor_state, mapping_summary]
        return (*[gr.update() for _ in range(N)], "", S["_cursor_data"], "")

    # Parse current values
    cur_vals = list(dd_vals[:N])

    if head_mode == "Pretrain head" and pretrained_names:
        # Pretrain: choices are static, no need to rebuild
        dd_updates = []
        for i in range(N):
            if i < n:
                choices, default = build_mapping_choices_pt(i, data_labels, pretrained_names)
                current = cur_vals[i]
                if current in choices:
                    dd_updates.append(gr.update(visible=False, choices=choices, value=current, label=data_labels[i]))
                else:
                    dd_updates.append(gr.update(visible=False, choices=choices, value=default, label=data_labels[i]))
            else:
                dd_updates.append(gr.update())
    else:
        # New head: rebuild choices dynamically (consumed labels disappear)
        mappings = {}
        for i in range(n):
            v = cur_vals[i] if i < len(cur_vals) else "keep"
            mappings[str(i)] = parse_mapping_value(v, data_labels)

        dd_updates = []
        for i in range(N):
            if i < n:
                choices = build_mapping_choices_new(i, data_labels, mappings)
                current = cur_vals[i]
                if current in choices:
                    dd_updates.append(gr.update(visible=False, choices=choices, value=current))
                else:
                    dd_updates.append(gr.update(visible=False, choices=choices, value=choices[0]))
            else:
                dd_updates.append(gr.update())

    # Compute final mapping + rebuild timeline
    final_vals = [dd_updates[i].get("value", cur_vals[i]) if isinstance(dd_updates[i], dict) and "value" in dd_updates[i] else cur_vals[i] for i in range(n)]
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, final_vals, data_labels, pretrained_names)

    vf = S["cur_vf"]
    tl, cdata = build_mapped_timeline(vf, new_names, label_map) if vf else ("", S["_cursor_data"])
    summary = build_mapping_summary_html(head_mode, final_vals, data_labels, pretrained_names)

    return (*dd_updates, tl, cdata, summary)

def on_head_mode_change(head_mode, *dd_vals):
    """When head mode toggles, rebuild all dropdown choices for the new mode."""
    return on_mapping_change(head_mode, *dd_vals)

# ====================== Video Preview ======================

def _get_frame(vf, fi):
    if not S["scan_data"]: return None
    d=next((x for x in S["scan_data"] if x["vf"]==vf),None)
    if not d: return None
    try:
        if S["cur_vf"]!=vf or S["cur_vr"] is None:
            # IMPORTANT: explicitly release the old VideoReader before opening a
            # new one. decord's VideoReader holds a C++ frame cache + open file
            # descriptor that Python's GC may not free promptly — without this,
            # switching between videos in the preview will leak hundreds of MB
            # per switch and crash Colab on RAM.
            old = S.get("cur_vr")
            if old is not None:
                S["cur_vr"] = None
                del old
                import gc; gc.collect()
            S["cur_vr"]=VideoReader(d["vp"],ctx=cpu(0)); S["cur_vf"]=vf
        T=len(S["cur_vr"]); fi=max(0,min(int(fi),T-1))
        return S["cur_vr"][fi].asnumpy()
    except: S["cur_vr"]=None; S["cur_vf"]=None; return None

def _preview_video_mapped(vf, head_mode, dd_vals):
    """Preview video with current mapping applied."""
    if not S["scan_data"] or not vf: return None,"","",U,S["_cursor_data"]
    d=next((x for x in S["scan_data"] if x["vf"]==vf),None)
    if not d: return None,"","",U,S["_cursor_data"]

    data_labels=S["label_names"]; pretrained_names=S["cfg"]["class_names"] if S["cfg"] else []
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, list(dd_vals[:len(data_labels)]), data_labels, pretrained_names)

    T=d["T"]; fps=d["fps"]
    tl, cdata = build_mapped_timeline(vf, new_names, label_map)

    ml = label_map.get(d["labels"][0], 0)
    nm0 = new_names[ml] if ml is not None and ml < len(new_names) else "?"
    _,bg = get_clr(ml if ml is not None else 0, nm0)
    info = f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm0}</span><span style='font-size:12px;color:var(--color-text-secondary);'>F: 0 / {T} | 0.00s / {T/fps:.2f}s</span></div>"

    img = _get_frame(vf, 0)
    return img, info, tl, gr.update(maximum=max(T-1,1),value=0), cdata

def on_scrub(fi, head_mode, *dd_vals):
    vf=S["cur_vf"]
    if not vf or not S["scan_data"]: return None,"<p style='color:#aaa;'>No data</p>"
    d=next((x for x in S["scan_data"] if x["vf"]==vf),None)
    if not d: return None,""

    data_labels=S["label_names"]; pretrained_names=S["cfg"]["class_names"] if S["cfg"] else []
    new_names,label_map=compute_label_map_from_dropdowns(head_mode,list(dd_vals[:len(data_labels)]),data_labels,pretrained_names)

    T=d["T"]; fps=d["fps"]; fi=max(0,min(int(fi),T-1))
    ml=label_map.get(d["labels"][fi],0)
    nm=new_names[ml] if ml is not None and ml<len(new_names) else "?"
    _,bg=get_clr(ml if ml is not None else 0,nm)
    info=f"<div style='display:flex;justify-content:space-between;align-items:center;'><span style='padding:3px 10px;border-radius:6px;background:{bg};color:white;font-size:12px;font-weight:500;'>{nm}</span><span style='font-size:12px;color:var(--color-text-secondary);'>F: {fi} / {T} | {fi/fps:.2f}s / {T/fps:.2f}s</span></div>"
    return _get_frame(vf,fi), info

def do_nav(direction, head_mode, *dd_vals):
    if not S["scan_data"]: return None,"","",U,S["_cursor_data"],"*No data*",""
    vnames=[d["vf"] for d in S["scan_data"]]; cur=S["cur_vf"]
    idx=vnames.index(cur) if cur in vnames else 0
    if direction=="prev": idx=max(0,idx-1)
    else: idx=min(len(vnames)-1,idx+1)
    vf=vnames[idx]
    img,info,tl,scrub,cdata=_preview_video_mapped(vf,head_mode,dd_vals)
    vid_list=build_video_list_html(active_vf=vf)
    return img,info,tl,scrub,cdata,f"**{vf}** — {idx+1} / {len(vnames)} videos",vid_list

def on_vid_change(vf, head_mode, *dd_vals):
    img,info,tl,scrub,cdata=_preview_video_mapped(vf,head_mode,dd_vals)
    vid_list=build_video_list_html(active_vf=vf)
    idx=0; total=0
    if S["scan_data"]:
        vnames=[d["vf"] for d in S["scan_data"]]
        total=len(vnames)
        if vf in vnames: idx=vnames.index(vf)
    nav_txt=f"**{vf}** — {idx+1} / {total} videos" if total else "*Load data first*"
    return img,info,tl,scrub,cdata,vid_list,nav_txt

def on_val_ratio_change(val_pct, val_seed):
    compute_split(val_pct, val_seed)
    return build_video_list_html(active_vf=S["cur_vf"])

# ====================== Progress + Validation HTML ======================

def html_progress(ep_done,ep_total,win_done,win_total,phase="training",ws=None,elapsed=None,stride=4):
    if ep_total==0: return ""
    ep_pct=(ep_done/ep_total)*100; wp=(win_done/max(win_total,1))*100
    ec="#1D9E75" if ep_done==ep_total else "#D85A30"
    st="✅ Complete" if ep_done==ep_total else "Training..."
    # Throughput: win/s and fps = win_done * stride / elapsed
    rate_str=""
    if elapsed and elapsed>0.1 and win_done>0:
        wps=win_done/elapsed
        fps=win_done*stride/elapsed
        rate_str=f" · {wps:.1f} win/s · {fps:.1f} fps"
    return f"<div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;'><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='font-size:13px;font-weight:500;'>Epoch — {st}</span><span style='font-size:12px;color:#888;'>{ep_done}/{ep_total} epochs</span></div><div style='height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px;'><div style='width:{ep_pct:.1f}%;height:100%;background:{ec};border-radius:4px;transition:width 0.3s;'></div></div><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='font-size:12px;font-weight:500;'>Epoch {min(ep_done+1,ep_total)} — {phase}</span><span style='font-size:12px;color:#888;'>{win_done}/{win_total} windows{rate_str}</span></div><div style='height:6px;background:#eee;border-radius:3px;overflow:hidden;'><div style='width:{wp:.1f}%;height:100%;background:#1D9E75;border-radius:3px;transition:width 0.15s;'></div></div></div>"

def html_cache_progress(done, total, current_name, mb_done=0, elapsed=None):
    """Pre-training video caching progress. Shares the same widget as
    html_progress so the user sees one unified status area in the centre column."""
    pct = (done / max(total, 1)) * 100
    color = "#1D9E75" if done == total else "#D85A30"
    label = "✅ Cached" if done == total else "Caching to local disk..."
    rate = ""
    if elapsed and elapsed > 0.5 and mb_done > 0:
        rate = f" · {mb_done/elapsed:.1f} MB/s"
    cur_str = f"Current: {current_name}" if current_name and done < total else "Ready"
    return f"<div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;'><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='font-size:13px;font-weight:500;'>Preparing — {label}</span><span style='font-size:12px;color:#888;'>{done}/{total} videos{rate}</span></div><div style='height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px;'><div style='width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;transition:width 0.3s;'></div></div><div style='font-size:12px;color:#666;'>{cur_str}</div></div>"

def html_val_card(epoch,loss,f1,mAP,f1_per,ap_per,names,prec_per=None,rec_per=None,is_best=False):
    brd="border:2px solid var(--color-border-info);" if is_best else "border:0.5px solid var(--color-border-tertiary);"
    badge="<span style='font-size:10px;padding:2px 6px;background:var(--color-background-info);color:var(--color-text-info);border-radius:var(--border-radius-md);margin-left:6px;'>best</span>" if is_best else ""
    fc="color:var(--color-text-info);" if is_best else ""
    prec_per=prec_per or []; rec_per=rec_per or []
    def _row(i,nm):
        p=prec_per[i] if i<len(prec_per) else 0.0
        r=rec_per[i] if i<len(rec_per) else 0.0
        f=f1_per[i] if i<len(f1_per) else 0.0
        a=ap_per[i] if i<len(ap_per) else 0.0
        return f"<div style='display:flex;justify-content:space-between;'><span>{nm}</span><span style='font-variant-numeric:tabular-nums;'>P: {p:.2f} · R: {r:.2f} · F1: {f:.2f} · AP: {a:.2f}</span></div>"
    rows="".join(_row(i,nm) for i,nm in enumerate(names))
    return f"<div style='background:var(--color-background-primary);{brd}border-radius:var(--border-radius-lg);padding:14px;margin-bottom:12px;'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'><div style='display:flex;align-items:center;'><span style='font-size:13px;font-weight:500;'>Epoch {epoch}</span>{badge}</div><span style='font-size:11px;color:var(--color-text-secondary);'>loss: {loss:.4f}</span></div><div style='display:flex;gap:8px;margin-bottom:8px;'><div style='flex:1;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:6px;text-align:center;'><div style='font-size:11px;color:var(--color-text-secondary);'>F1-macro</div><div style='font-size:16px;font-weight:500;{fc}'>{f1:.4f}</div></div><div style='flex:1;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:6px;text-align:center;'><div style='font-size:11px;color:var(--color-text-secondary);'>mAP</div><div style='font-size:16px;font-weight:500;{fc}'>{mAP:.4f}</div></div></div><div style='font-size:11px;color:var(--color-text-secondary);line-height:1.6;'>{rows}</div></div>"

def build_val_html(log,names):
    if not log: return "<p style='color:#aaa;'>Training not started</p>"
    best=max(range(len(log)),key=lambda i:log[i]["f1"])
    return "".join(html_val_card(e["epoch"],e["loss"],e["f1"],e["mAP"],e["f1_per"],e["ap_per"],names,e.get("prec_per"),e.get("rec_per"),i==best) for i,e in enumerate(log))

def scan_val_folder(val_vdir, val_ldir):
    """Scan a separate validation folder into the same entry format as
    S["scan_data"] (dicts with vf/vp/lp). Returns (entries, message).

    Uses the same flexible video↔CSV matching as the main scan so a val folder
    laid out like the training folder just works."""
    if not val_vdir or not os.path.isdir(val_vdir):
        return [], f"❌ Val video dir not found: {val_vdir}"
    if not val_ldir or not os.path.isdir(val_ldir):
        return [], f"❌ Val label dir not found: {val_ldir}"

    vfiles = sorted([f for f in os.listdir(val_vdir)
                     if f.lower().endswith((".mp4", ".avi", ".mov"))])
    if not vfiles:
        return [], "❌ No videos in val folder"

    csv_files = {os.path.splitext(f)[0].lower(): os.path.join(val_ldir, f)
                 for f in os.listdir(val_ldir) if f.lower().endswith(".csv")}

    entries = []
    for vf in vfiles:
        base = os.path.splitext(vf)[0]; lp = None
        for cand in [base, base.replace("-", ""), base.replace("_", ""),
                     base.replace("-", "_"), base.replace("_", "-")]:
            fp = os.path.join(val_ldir, cand + ".csv")
            if os.path.exists(fp): lp = fp; break
            fp = os.path.join(val_ldir, cand + "_one_hot.csv")
            if os.path.exists(fp): lp = fp; break
            if cand.lower() in csv_files:
                lp = csv_files[cand.lower()]; break
        if lp is None:
            continue
        vp = os.path.join(val_vdir, vf)
        try:
            vr = VideoReader(vp, ctx=cpu(0)); T = len(vr); fps = vr.get_avg_fps()
            del vr
        except Exception:
            continue
        entries.append({"vf": vf, "vp": vp, "lp": lp, "T": T, "fps": fps})

    if not entries:
        return [], "❌ No video/label pairs matched in val folder"
    return entries, f"✅ {len(entries)} val videos"


def build_threshold_table(epoch, n_steps=10):
    """Plot precision & recall vs threshold for one epoch (one subplot/class).

    For each class c and threshold t a window counts as a positive prediction
    when softmax prob for c >= t (one-vs-rest), so precision and recall move
    independently of argmax — that is the point of sweeping the threshold.

    Returns a PNG path for gr.Image (or None when there is nothing to plot).
    """
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vp = S.get("val_probs") or {}
    if not vp:
        return None
    try:
        epoch = int(epoch)
    except Exception:
        return None
    if epoch not in vp:
        return None

    d = vp[epoch]
    y = np.asarray(d["y_true"]); P = np.asarray(d["probs"]); names = d["names"]
    try:
        n_steps = max(2, int(n_steps))
    except Exception:
        n_steps = 10
    ths = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]   # exclude 0 and 1 (degenerate)
    if len(ths) == 0:
        ths = np.array([0.5])

    # Skip Other/Others: it means "no target behaviour", so a threshold sweep
    # on it is not meaningful
    show = [i for i, n in enumerate(names)
            if n.lower() not in ("others", "other")]
    if not show:
        return None
    nc = len(show)
    ncol = min(3, nc)
    nrow = int(np.ceil(nc / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             dpi=130, squeeze=False)

    C_PREC = "#378ADD"   # blue = precision
    C_REC  = "#D85A30"   # orange = recall

    for k, ci in enumerate(show):
        nm = names[ci]
        ax = axes[k // ncol][k % ncol]
        pos = (y == ci)
        n_pos = int(pos.sum())

        precs, recs, f1s = [], [], []
        for t in ths:
            pred = P[:, ci] >= t
            tp = int(np.sum(pred & pos))
            fp = int(np.sum(pred & ~pos))
            fn = int(np.sum(~pred & pos))
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            precs.append(pr); recs.append(rc)
            f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)

        ax.plot(ths, precs, "-o", color=C_PREC, ms=3.5, lw=1.8, label="Precision")
        ax.plot(ths, recs,  "-o", color=C_REC,  ms=3.5, lw=1.8, label="Recall")

        # mark the threshold with the best F1
        if f1s:
            bi = int(np.argmax(f1s))
            ax.axvline(ths[bi], color="#888", ls="--", lw=1, alpha=0.8)
            ax.annotate(f"best F1 {f1s[bi]*100:.0f}% @ {ths[bi]:.2f}",
                        xy=(ths[bi], 1.02), fontsize=7.5, color="#555",
                        ha="center", va="bottom")

        ax.set_title(f"{nm}  ({n_pos} pos)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(-0.02, 1.16)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, lw=0.7)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=1)

    # hide unused grid cells
    for k in range(nc, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"Epoch {epoch} · {len(y)} validation windows · one-vs-rest",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(tempfile.gettempdir(), f"threshold_epoch{epoch}.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def _runs_of(seq):
    """Compress per-frame labels into (label, start, length) runs."""
    out = []
    cur, st = seq[0], 0
    for i in range(1, len(seq)):
        if seq[i] != cur:
            out.append((cur, st, i - st)); cur, st = seq[i], i
    out.append((cur, st, len(seq) - st))
    return out


def list_val_videos(epoch):
    """List of validation videos (for the dropdown)."""
    vw = (S.get("val_windows") or {})
    try:
        epoch = int(epoch)
    except Exception:
        return gr.update(choices=[], value=None)
    if epoch not in vw:
        return gr.update(choices=[], value=None)
    vids = sorted({os.path.basename(vp) for vp, _ in vw[epoch]["samples"]})
    return gr.update(choices=vids, value=vids[0] if vids else None)


def build_val_ethogram(epoch, vf):
    """Ethogram for one validation video: prediction (yellow, top) over
    ground truth (blue, bottom).

    Window predictions (ws=16, stride=4) are scattered back onto every frame
    they cover; overlaps take a majority vote, matching how inference
    aggregates. Ground truth is read per-frame from the original label file.
    """
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    vw = (S.get("val_windows") or {})
    try:
        epoch = int(epoch)
    except Exception:
        return None
    if epoch not in vw or not vf:
        return None

    d = vw[epoch]
    names = d["names"]; label_map = d["label_map"]
    samples = d["samples"]; preds = d["pred"]

    # collect every window belonging to this video
    vp = None
    win = []
    for k, (p_, idx_) in enumerate(samples):
        if os.path.basename(p_) == vf:
            vp = p_
            win.append((idx_, int(preds[k])))
    if vp is None or not win:
        return None

    # ---- ground truth: per-frame, passed through label_map ----
    try:
        vr = VideoReader(vp, ctx=cpu(0)); T = len(vr); fps = vr.get_avg_fps(); del vr
    except Exception:
        return None

    lp = None
    for entry in (S.get("scan_data") or []):
        if entry.get("vp") == vp:
            lp = entry.get("lp"); break
    if lp is None:
        # separate val folder: not in scan_data, so fall back to a same-named csv
        base = os.path.splitext(vf)[0]
        cand = os.path.join(os.path.dirname(vp), base + ".csv")
        lp = cand if os.path.exists(cand) else None

    gt = np.full(T, -1, dtype=int)
    if lp:
        try:
            oh, col_names = load_label_data(lp, T, fps)
            global_names = S.get("label_names", []) or []
            if global_names and col_names != global_names:
                oh = align_onehot_to_global(oh, col_names, global_names)
            raw = np.argmax(oh, axis=1)
            gt = np.array([label_map.get(int(l), -1) for l in raw])
        except Exception as e:
            print(f"⚠️ GT load failed for {vf}: {e}")

    # ---- prediction: scatter windows back onto frames, majority vote ----
    votes = [[] for _ in range(T)]
    for idx_, lbl in win:
        for i in idx_:
            if 0 <= i < T:
                votes[i].append(lbl)
    pred = np.array([Counter(v).most_common(1)[0][0] if v else -1 for v in votes])

    # ---- plot (excluding Other and excluded frames marked -1) ----
    show = [i for i, n in enumerate(names)
            if n.lower() not in ("other", "others")]
    if not show:
        return None
    row_of = {c: k for k, c in enumerate(show)}
    n_show = len(show)

    C_PRED = "#E8B84B"   # yellow = prediction (top)
    C_GT   = "#378ADD"   # blue = ground truth (bottom)

    fig, ax = plt.subplots(figsize=(13, max(2.4, 0.62 * n_show + 1.5)), dpi=140)
    ax.set_facecolor("white")

    H, GAP = 0.34, 0.03
    for seq, color, on_top in ((pred, C_PRED, True), (gt, C_GT, False)):
        for cls, s, w in _runs_of(seq):
            if cls not in row_of:
                continue
            ybase = n_show - 1 - row_of[cls]
            y0 = ybase + (GAP / 2) if on_top else ybase - H - (GAP / 2)
            ax.broken_barh([(s, w)], (y0, H), facecolors=color, edgecolors="none")

    ax.set_yticks(range(n_show))
    ax.set_yticklabels([names[c] for c in reversed(show)], fontsize=10)
    ax.set_ylim(-0.75, n_show - 0.25)
    ax.set_xlim(0, T)
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_title(f"{vf}  ·  epoch {epoch}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")

    if fps:
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / fps, lambda t: t * fps))
        sec.set_xlabel("Time (s)", fontsize=9)
        sec.tick_params(labelsize=8)

    # agreement: only frames where at least one side is a non-Other behaviour
    m = np.isin(gt, show) | np.isin(pred, show)
    agree = float(np.mean(gt[m] == pred[m])) * 100 if m.any() else 0.0
    ax.annotate(f"frame agreement (non-Other): {agree:.1f}%",
                xy=(0.99, 1.14), xycoords="axes fraction",
                ha="right", fontsize=8, color="#666")

    ax.legend(handles=[Patch(facecolor=C_PRED, label="Prediction"),
                       Patch(facecolor=C_GT, label="Ground truth")],
              loc="upper center", bbox_to_anchor=(0.5, -0.38),
              ncol=2, fontsize=9, frameon=False)

    fig.subplots_adjust(bottom=0.34, top=0.80, left=0.13, right=0.97)
    out = os.path.join(tempfile.gettempdir(),
                       f"val_eth_ep{epoch}_{os.path.splitext(vf)[0]}.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def refresh_threshold_epochs():
    """Populate the epoch dropdown from whatever epochs have val data."""
    eps = sorted((S.get("val_probs") or {}).keys())
    if not eps:
        return gr.update(choices=[], value=None)
    return gr.update(choices=eps, value=eps[-1])


# ====================== Training ======================

def cancel_training():
    S["_cancel_training"] = True
    return "<p style='color:#e74c3c;font-weight:600;'>⛔ Cancelling… will stop after current batch.</p>"

def run_training(repo,mname,vdir,ldir,odir,head_mode,
                 n_epochs,batch_sz,lr_str,val_pct,val_seed,train_seed,
                 num_workers,cache_local,
                 use_sep_val,val_vdir,val_ldir,
                 aug_blur_frac,aug_tdrop_frac,aug_hflip_p,aug_vflip_p,
                 aug_rot_deg,aug_brightness,aug_contrast,aug_saturation,
                 aug_mult,aug_excluded_classes,
                 *dd_vals):
    S["_cancel_training"] = False
    try:
        lr=float(lr_str); val_ratio=float(val_pct)/100.0; n_epochs=int(n_epochs)
        batch_sz=int(batch_sz)
        ws=16; stride_val=4  # fixed window and stride
        val_seed=int(val_seed); train_seed=int(train_seed)
        val_seed=int(val_seed); train_seed=int(train_seed)
        num_workers=int(num_workers)
        aug_blur_frac=float(aug_blur_frac); aug_tdrop_frac=float(aug_tdrop_frac)
        aug_hflip_p=float(aug_hflip_p); aug_vflip_p=float(aug_vflip_p)
        aug_rot_deg=float(aug_rot_deg); aug_brightness=float(aug_brightness)
        aug_contrast=float(aug_contrast); aug_saturation=float(aug_saturation)
        aug_mult=int(aug_mult)
        aug_excluded_classes=list(aug_excluded_classes) if aug_excluded_classes else []
    except Exception as e: yield f"❌ Invalid params: {e}",U; return

    if not S["model"] or not S["cfg"]: yield "❌ Load model first",U; return
    if not S["scan_data"]: yield "❌ Load data first",U; return

    cfg=S["cfg"]; data_labels=S["label_names"]
    pretrained_names=cfg["class_names"]
    os.makedirs(odir,exist_ok=True)

    # Compute label map from dropdown values
    vals = list(dd_vals[:len(data_labels)])
    new_names, label_map = compute_label_map_from_dropdowns(head_mode, vals, data_labels, pretrained_names)
    new_nc = len(new_names)

    print(f"🏷️ Label mapping: {label_map}")
    print(f"🏷️ Training classes ({new_nc}): {new_names}")

    # Deep copy pretrained model so S["model"] stays pristine for re-training
    import copy
    model = copy.deepcopy(S["model"])
    model = rebuild_head(model, cfg, new_nc).to(device)

    data=S["scan_data"]
    tidx=S["split_indices"]["train"]; vidx=S["split_indices"]["val"]

    if use_sep_val:
        # Separate validation folder: use all training data for training (no
        # split). Append val videos to `data` and point vidx at those entries.
        eff_val_ldir = val_ldir if val_ldir else ldir
        val_entries, vmsg = scan_val_folder(val_vdir, eff_val_ldir)
        if not val_entries:
            yield f"<p style='color:#e74c3c;font-weight:600;'>❌ {vmsg}</p>", U
            return
        data = list(data) + val_entries
        tidx = list(range(len(S["scan_data"])))
        vidx = list(range(len(S["scan_data"]), len(data)))
        print(f"📁 Separate val folder: {len(tidx)} train / {len(vidx)} val videos")
    elif not tidx and not vidx:
        if val_ratio>0 and len(data)>=4: tidx,vidx=train_test_split(list(range(len(data))),test_size=val_ratio,random_state=val_seed)
        else: tidx=list(range(len(data))); vidx=[]

    # ---- Cache videos to local disk if requested ----
    # We do this here (not at scan time) so the user gets immediate feedback
    # when "Load folder" is clicked, and the slow copy step happens visibly
    # under the same progress widget once they press Train.
    used_idxs = sorted(set(tidx) | set(vidx))  # only cache videos that will be used
    if cache_local and used_idxs:
        n_total = len(used_idxs)
        cache_t0 = time.perf_counter()
        mb_done = 0.0
        # Show 0/N immediately so user knows something started
        yield html_cache_progress(0, n_total, data[used_idxs[0]]["vf"]),"<p style='color:#aaa;'>Caching...</p>"
        for i, di in enumerate(used_idxs):
            if S["_cancel_training"]:
                yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Training cancelled by user.</p>", U
                print("⛔ Training cancelled by user.")
                return
            d = data[di]
            src_path = d["vp"]
            # Only cache if source is not already inside the cache dir (avoids
            # re-cache when user trains again on the same data without re-loading).
            if not src_path.startswith(VIDEO_CACHE_DIR):
                local_path = cache_video_to_local(src_path)
                if local_path != src_path:
                    d["vp"] = local_path  # mutate scan_data so all future reads use local
                    try:
                        mb_done += os.path.getsize(local_path) / (1024*1024)
                    except Exception:
                        pass
            yield html_cache_progress(i+1, n_total,
                                      data[used_idxs[i+1]]["vf"] if i+1 < n_total else "",
                                      mb_done=mb_done,
                                      elapsed=time.perf_counter()-cache_t0), U

    # Now (possibly cached) paths can be collected
    vps=[d["vp"] for d in data]; lps=[d["lp"] for d in data]

    yield html_progress(0,n_epochs,0,0,"building dataset...",ws=ws),"<p style='color:#aaa;'>Building...</p>"

    # Online augmentation: uses the `random` module globally so each DataLoader
    # worker (seeded via worker_init_fn below) gets its own stream, and every
    # epoch re-draws independently. This is the standard PyTorch pattern.
    def aug(fr):
        # Spatial (whole-window consistent) first, then photometric, then per-frame
        fr=horizontal_flip(fr,prob=aug_hflip_p)
        fr=vertical_flip(fr,prob=aug_vflip_p)
        fr=random_rotation(fr,max_deg=aug_rot_deg)
        fr=color_jitter(fr,brightness=aug_brightness,contrast=aug_contrast,saturation=aug_saturation)
        fr=random_blur(fr,frac=aug_blur_frac)
        fr=temporal_dropout(fr,frac=aug_tdrop_frac)
        return fr

    train_ds=SlidingWindowDataset([vps[i] for i in tidx],[lps[i] for i in tidx],ws,stride_val,cfg,new_nc,label_map,augment=aug)
    val_ds=SlidingWindowDataset([vps[i] for i in vidx],[lps[i] for i in vidx],ws,stride_val,cfg,new_nc,label_map) if vidx else None

    if len(train_ds)==0: yield "❌ No training windows created.",""; return

    # ----- Class balancing: duplicate sample-list entries for selected classes -----
    # Note: only the (video_path, frame_indices, label) tuples are duplicated —
    # not the actual frames. Each time the same window is drawn by the DataLoader,
    # online augmentation re-rolls independently, producing different augmented
    # results. Effect is equivalent to offline augmentation but costs no disk
    # and almost no RAM; only cost is that epoch time scales with the multiplier.
    if aug_mult > 1:
        excluded_idx = set()
        for cname in aug_excluded_classes:
            if cname in new_names:
                excluded_idx.add(new_names.index(cname))
        orig_samples = train_ds.samples[:]
        orig_labels = train_ds.sample_labels[:]
        n_before = len(orig_samples)
        extra_samples = []
        extra_labels = []
        n_duplicated = 0
        for s, l in zip(orig_samples, orig_labels):
            if l in excluded_idx:
                continue
            # add (mult - 1) extra copies; the original already counts as 1
            for _ in range(aug_mult - 1):
                extra_samples.append(s)
                extra_labels.append(l)
            n_duplicated += 1
        train_ds.samples = orig_samples + extra_samples
        train_ds.sample_labels = orig_labels + extra_labels
        print(f"📈 Class balancing: {n_before} → {len(train_ds.samples)} windows "
              f"(×{aug_mult} for {n_duplicated} non-excluded windows, "
              f"excluded classes: {aug_excluded_classes or 'none'})")

    # Reproducible shuffle + per-worker seeding for true per-epoch randomness
    g = torch.Generator(); g.manual_seed(train_seed)
    def _worker_init(worker_id):
        seed = train_seed + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32))

    # worker_init_fn / persistent_workers are only meaningful when num_workers > 0
    train_loader_kwargs = dict(batch_size=batch_sz, shuffle=True, pin_memory=True,
                               num_workers=num_workers, generator=g)
    if num_workers > 0:
        train_loader_kwargs["worker_init_fn"] = _worker_init
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_loader = DataLoader(val_ds, batch_sz, shuffle=False, num_workers=num_workers,
                            pin_memory=True) if val_ds and len(val_ds) > 0 else None
    total_win=len(train_ds)

    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    scheduler=CosineAnnealingLR(optimizer,T_max=n_epochs)
    scaler=GradScaler(); criterion=nn.CrossEntropyLoss(); accum=2
    S["train_log"]=[]

    for ep in range(n_epochs):
        if S["_cancel_training"]:
            yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Training cancelled by user.</p>", U
            print("⛔ Training cancelled by user.")
            return
        model.train(); rl=0.0; optimizer.zero_grad(set_to_none=True); nb=len(train_loader)
        ep_t0=time.perf_counter()
        for bi,(vids,tgts) in enumerate(train_loader):
            if S["_cancel_training"]:
                yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Training cancelled by user.</p>", U
                print("⛔ Training cancelled by user.")
                return
            vids=vids.to(device); tgts=tgts.to(device)
            with autocast(): loss=criterion(model(vids),tgts)/accum
            scaler.scale(loss).backward()
            if (bi+1)%accum==0 or (bi+1)==nb: scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            rl+=loss.item()*accum*vids.size(0)
            if (bi+1)%5==0 or bi==nb-1:
                wd=min((bi+1)*batch_sz,total_win)
                yield html_progress(ep,n_epochs,wd,total_win,"training",ws=ws,elapsed=time.perf_counter()-ep_t0),U

        scheduler.step(); ep_loss=rl/len(train_ds)
        f1m=0; mAP=0; f1p=[]; app=[]; precp=[]; recp=[]
        if val_loader:
            model.eval(); ap_=[]; al_=[]; apr_=[]; val_total=len(val_ds); nb_val=len(val_loader)
            val_t0=time.perf_counter()
            with torch.no_grad():
                for vi,(v,t) in enumerate(val_loader):
                    v=v.to(device)
                    with autocast(): o=model(v); pr=torch.softmax(o,dim=1)
                    # .detach() defends against edge cases where autocast or PyTorch
                    # version leaves tensors in a grad-tracking state even inside no_grad
                    ap_.extend(torch.argmax(o,1).detach().cpu().numpy())
                    al_.extend(t.detach().cpu().numpy())
                    apr_.extend(pr.detach().cpu().numpy())
                    if (vi+1)%5==0 or vi==nb_val-1:
                        vd=min((vi+1)*batch_sz,val_total)
                        yield html_progress(ep,n_epochs,vd,val_total,"validating",ws=ws,elapsed=time.perf_counter()-val_t0),U
            f1p=f1_score(al_,ap_,average=None,labels=list(range(new_nc)),zero_division=0).tolist()
            f1m=f1_score(al_,ap_,average="macro",zero_division=0)
            precp=precision_score(al_,ap_,average=None,labels=list(range(new_nc)),zero_division=0).tolist()
            recp=recall_score(al_,ap_,average=None,labels=list(range(new_nc)),zero_division=0).tolist()
            oh=np.zeros((len(al_),new_nc))
            for i,l in enumerate(al_): oh[i,l]=1
            pr_arr=np.array(apr_)
            for ci in range(new_nc):
                try: app.append(average_precision_score(oh[:,ci],pr_arr[:,ci]))
                except: app.append(0.0)
            mAP=np.mean(app)

        mp=os.path.join(odir,f"epoch_{ep+1}_f1_{f1m:.4f}_map_{mAP:.4f}.pth")
        torch.save(model.state_dict(),mp)

        # Save config.json alongside .pth — compatible with test code
        cfg_out = {
            "model_info": {
                "backbone": cfg["backbone"]["name"],
                "head": {"in_features": cfg["head"]["in_features"], "hidden_dim": cfg["head"]["hidden_dim"],
                         "dropout": cfg["head"]["dropout"], "pool": cfg["head"].get("pool","cls_token")},
                "input_format": cfg.get("input_format", {}),
                "backbone_config": {"input_size": cfg["backbone"].get("input_size",224),
                                    "num_frames": cfg["backbone"].get("num_frames",8)},
            },
            # Fields the test code needs directly:
            "ALL_BEHAVIOR_NAMES": list(data_labels),
            "SELECTED_BEHAVIORS": list(new_names),
            "num_classes": new_nc,
            "class_names": list(new_names),
            # original_to_new: same format as test code
            # selected → sequential index, unselected → null (test code treats as None → exclude)
            "original_to_new": {
                str(i): label_map[i] if i in label_map else None
                for i in range(len(data_labels))
            },
            # Full mapping details for reference
            "mapping_mode": head_mode,
            "mapping_detail": {
                data_labels[i]: {"target": new_names[label_map[i]] if i in label_map and label_map[i] is not None else "excluded",
                                 "target_idx": label_map.get(i, None)}
                for i in range(len(data_labels))
            },
            "training_params": {
                "epochs": n_epochs, "batch_size": batch_sz, "lr": lr,
                "window_size": ws, "stride": stride_val,
                "val_seed": val_seed, "train_seed": train_seed,
            },
            "augmentation": {
                "blur_frac": aug_blur_frac,
                "temporal_dropout_frac": aug_tdrop_frac,
                "horizontal_flip_prob": aug_hflip_p,
                "vertical_flip_prob": aug_vflip_p,
                "rotation_deg": aug_rot_deg,
                "brightness": aug_brightness,
                "contrast": aug_contrast,
                "saturation": aug_saturation,
                "class_multiplier": aug_mult,
                "class_multiplier_excluded": aug_excluded_classes,
            },
        }
        cfg_path = mp.replace(".pth", "_config.json")
        with open(cfg_path, "w") as f: json.dump(cfg_out, f, indent=2)

        S["train_log"].append({"epoch":ep+1,"loss":ep_loss,"f1":f1m,"mAP":mAP,"f1_per":f1p,"ap_per":app,"prec_per":precp,"rec_per":recp,"path":mp,"config_path":cfg_path})
        # keep this epoch's val probabilities + true labels for the threshold
        # analysis (deliberately NOT written to json - it would bloat the file)
        if len(al_) > 0:
            S.setdefault("val_probs", {})[ep+1] = {
                "y_true": np.asarray(al_, dtype=np.int16),
                "probs": np.asarray(apr_, dtype=np.float32),
                "names": list(new_names),
            }
            # val_loader uses shuffle=False, so ap_ is in the same order as
            # val_ds.samples. Storing (video, frame indices, prediction) per
            # window lets us scatter predictions back onto frames later.
            try:
                S.setdefault("val_windows", {})[ep+1] = {
                    "samples": [(vp_, list(idx_)) for vp_, idx_, _ in val_ds.samples],
                    "pred": np.asarray(ap_, dtype=np.int16),
                    "names": list(new_names),
                    "label_map": dict(label_map),
                }
            except Exception as e:
                print(f"⚠️ Could not store val windows: {e}")
        yield html_progress(ep+1,n_epochs,total_win,total_win,"done",ws=ws),build_val_html(S["train_log"],new_names)

    with open(os.path.join(odir,"training_log.json"),"w") as f: json.dump(S["train_log"],f,indent=2)
    print("✅ Training complete!")

# ====================== Cursor JS ======================

CURSOR_JS = """
(fi, labels_json) => {
    let T=0, names=[], labels=[];
    try { const d=JSON.parse(labels_json); T=d.T; names=d.names; labels=d.labels; } catch(e) { return fi; }
    if (T===0) return fi;
    fi = Math.max(0, Math.min(Math.floor(fi), T-1));
    const c = document.getElementById('tl-cursor');
    if (c) c.style.left = ((fi/T)*100)+'%';
    const l = document.getElementById('tl-frame-label');
    if (l) { const cls=labels[fi]; l.textContent='F:'+fi+' '+(names[cls]||'?'); }
    return fi;
}
"""

YELLOW_THEME = gr.themes.Soft(primary_hue=gr.themes.colors.amber, secondary_hue=gr.themes.colors.yellow, neutral_hue=gr.themes.colors.gray)

# ====================== Custom CSS ======================
# Global style overrides: black text, larger headings, stronger borders and
# a slightly darker page background for contrast.
CUSTOM_CSS = """
/* keep the mapper bridge in the DOM but invisible (visible=False would drop
   its <textarea>, breaking the JS bridge) */
.lm-hidden{position:absolute!important;width:1px!important;height:1px!important;
           padding:0!important;margin:-1px!important;overflow:hidden!important;
           clip:rect(0 0 0 0)!important;border:0!important;opacity:0!important;
           pointer-events:none!important}
/* darker page background, blocks stay white -> stronger contrast */
.gradio-container { background: #d8d4c4 !important; }
.gr-box, .block, .form, .gr-panel, .gr-accordion,
.gradio-container .prose { background: #ffffff !important; }

/* force black text (overrides Gradio grey secondary/tertiary vars) */
.gradio-container, .gradio-container * {
    color: #000000 !important;
    --color-text-primary: #000000;
    --color-text-secondary: #000000;
    --color-text-tertiary: #000000;
}

/* Border strategy: do NOT outline every .block / .form (that
   would box every child of a gr.Group separately). Only real inputs and
   accordions get a border. */
input, textarea, select,
.gr-input, .gr-dropdown, .gr-accordion {
    border: 1.5px solid #555555 !important;
}
.gradio-container [class*="dropdown"] > label > div,
.gradio-container [data-testid="dropdown"] {
    border: 1.5px solid #555555 !important;
    border-radius: 6px !important;
}

/* slightly larger section headings (Markdown h1/h2/h3) */
.gradio-container h1 { font-size: 30px !important; font-weight: 700 !important; }
.gradio-container h2 { font-size: 22px !important; font-weight: 700 !important; }
.gradio-container h3 { font-size: 18px !important; font-weight: 700 !important; }

/* component labels: black and bold */
label, .gr-input-label, span[data-testid="block-info"] {
    color: #000000 !important; font-weight: 600 !important;
}
"""

# ====================== Demo Loading ======================

DEMO_LOCAL_DIR = os.path.join(os.path.expanduser("~"), "demo_data")

def load_demo_training(repo, val_pct, val_seed, head_mode, *dd_vals):
    """Download ALL files from HF repo demo/ folder, then auto-scan.
    Returns same outputs as do_scan_and_preview so everything loads in one click."""
    N = MAX_LABELS
    empty = lambda msg: (msg, "", "*Load data first*",
                         *[gr.update(visible=False, choices=[], value=None) for _ in range(N)],
                         gr.update(choices=[], value=None),
                         None, "", "", gr.update(maximum=1, value=0), S["_cursor_data"], "", "")
    if not repo:
        return empty("❌ Specify repo")
    try:
        all_files = list_repo_files(repo)
        demo_files = [f for f in all_files if f.startswith("demo/") and f != "demo/"]
        if not demo_files:
            return empty("❌ No files in demo/ folder on HuggingFace")
        os.makedirs(DEMO_LOCAL_DIR, exist_ok=True)
        for f in demo_files:
            local = hf_hub_download(repo_id=repo, filename=f)
            fname = os.path.basename(f)
            dest = os.path.join(DEMO_LOCAL_DIR, fname)
            if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(local):
                shutil.copy2(local, dest)
        # Now run the same scan logic with demo dir.
        return do_scan_and_preview(DEMO_LOCAL_DIR, DEMO_LOCAL_DIR, val_pct, val_seed, head_mode, *dd_vals)
    except Exception as e:
        return empty(f"❌ {e}")

# ====================== GUI ======================

with gr.Blocks(title="Training", theme=YELLOW_THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown("# Animal Behavior Model Training\nFine-tune from pretrained — preview labels & configure mapping before training")

    cursor_state = gr.Textbox(value=S["_cursor_data"], visible=False)
    repo_in = gr.Textbox(value=HF_REPO_ID, visible=False)

    with gr.Row():
        # ===== LEFT =====
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### ① Select model")
            with gr.Group():
                model_dd=gr.Dropdown(label="Base model",choices=[],interactive=True)
                model_st=gr.Textbox(label="Status",interactive=False,lines=4)
            load_btn=gr.Button("📥 Load pretrained",variant="primary")
            gr.Markdown("---")
            gr.Markdown("### ② Load data")
            vdir_in=gr.Textbox(label="Video directory",value=DEFAULT_VIDEO_DIR,
                placeholder="e.g. /content/drive/My Drive/videos/train/")
            with gr.Group():
                ldir_in=gr.Textbox(label="Label directory",value=DEFAULT_LABEL_DIR,
                    placeholder="e.g. /content/drive/My Drive/labels/")
                odir_in=gr.Textbox(label="Output directory",value=DEFAULT_OUTPUT_DIR,
                    placeholder="e.g. /content/drive/My Drive/trained_models/")
                scan_st=gr.Textbox(label="Folder status",interactive=False,lines=1)

            # Validation data in a separate folder (when ticked, no split of train data)
            sep_val_cb=gr.Checkbox(label="✅ Validation data is in a separate folder",
                                   value=False)
            with gr.Group(visible=False) as sep_val_grp:
                val_vdir_in=gr.Textbox(label="Val video directory",
                    placeholder="e.g. /content/drive/My Drive/videos/val/")
                val_ldir_in=gr.Textbox(label="Val label directory (leave blank = use Label directory above)",
                    placeholder="e.g. /content/drive/My Drive/labels_val/")
                gr.Markdown("<p style='font-size:12px;color:#888;'>When enabled, the "
                            "Validation ratio below is ignored and the training data is not split.</p>")
            demo_btn=gr.Button("🎯 Load Demo",variant="secondary",size="sm")
            scan_d=gr.Button("📂 Load folder",variant="secondary")

        # ===== CENTER =====
        with gr.Column(scale=2, min_width=400):
            with gr.Group():
                progress_html=gr.HTML("")
                info_html=gr.HTML("<p style='color:#aaa;'>Load data to preview</p>")
                frame_img=gr.Image(label="Frame preview",type="numpy",interactive=False)
                timeline_html=gr.HTML("")
                scrubber=gr.Slider(minimum=0,maximum=100,step=1,value=0,label="Frame",interactive=True)
            with gr.Row():
                prev_btn=gr.Button("◀ Prev",size="sm")
                nav_md=gr.Markdown("*Load data first*")
                next_btn=gr.Button("Next ▶",size="sm")
            gr.Markdown("---")
            with gr.Accordion("📹 Videos", open=False):
                vid_list_html=gr.HTML("<p style='color:#aaa;font-size:12px;'>Load data first</p>")
            gr.Markdown("---")

            # Label distribution
            with gr.Accordion("📊 Label distribution", open=False):
                label_dist_html=gr.HTML("<p style='color:#aaa;'>Load data to see labels</p>")

            # Head type + mapping dropdowns
            gr.Markdown("### 🏷️ Label mapping")
            with gr.Group():
                head_mode_dd=gr.Dropdown(label="Head type",choices=["Pretrain head","New head"],value="Pretrain head",interactive=True)

                # Visual label mapper (drag to connect). The dropdowns below
                # stay the source of truth for the rest of the app — the mapper
                # just writes into them through a hidden JSON bridge.
                gr.HTML(MAPPER_HTML)
                # Bridge must stay in the DOM (not visible=False, which drops
                # its <textarea> entirely) so the mapper JS can write into it.
                lm_bridge = gr.Textbox(elem_id="lm_bridge",
                                       elem_classes=["lm-hidden"])

                # Pre-build MAX_LABELS dropdown slots (hidden — driven by the mapper)
                map_dds = []
                for i in range(MAX_LABELS):
                    dd = gr.Dropdown(label=f"label_{i}", choices=[], value=None, interactive=True, visible=False)
                    map_dds.append(dd)

                mapping_summary=gr.HTML("")

            # Hidden video dropdown
            vid_dd=gr.Dropdown(label="Video",choices=[],interactive=True,visible=False)

        # ===== RIGHT =====
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ③ Train")
            with gr.Group():
                vr_in=gr.Slider(minimum=0,maximum=50,step=5,value=15,label="Validation ratio (%)",interactive=True)
                with gr.Row():
                    ep_in=gr.Number(label="Epochs",value=5,precision=0)
                    bs_in=gr.Number(label="Batch",value=8,precision=0)
                with gr.Row():
                    lr_in=gr.Textbox(label="LR",value="3.8e-5")
                gr.HTML("<div style='font-size:12px;color:#555;padding:6px 10px;background:#f7f7f7;border-radius:6px;margin:4px 0;'><b>Window:</b> 16 frames &nbsp;·&nbsp; <b>Stride:</b> 4 frames (fixed)</div>")
                with gr.Row():
                    val_seed_in=gr.Number(label="Val seed",value=1337,precision=0,info="Split reproducibility")
                    train_seed_in=gr.Number(label="Train seed",value=2025,precision=0,info="Augmentation reproducibility")
                nw_in=gr.Slider(minimum=0,maximum=8,step=1,value=2,
                                label="DataLoader workers",
                                info="0 = single process (safest, slowest). 2 = good for Colab. 4+ may crash with large videos.")
                cache_local_cb=gr.Checkbox(label="Cache videos to local disk before training",value=True,
                    info="Copy from Drive to /content first (5-30× faster reads). Progress shown when training starts.")

            with gr.Accordion("🎨 Augmentation", open=False):
                gr.Markdown("**Spatial** — applied to the whole window consistently")
                aug_hflip_in=gr.Slider(minimum=0,maximum=1,step=0.05,value=0.5,
                                       label="Horizontal flip probability",
                                       info="0 = off, 0.5 = flip half the windows")
                aug_vflip_in=gr.Slider(minimum=0,maximum=1,step=0.05,value=0.0,
                                       label="Vertical flip probability",
                                       info="Usually 0 for animal behavior (up/down matters)")
                aug_rot_in=gr.Slider(minimum=0,maximum=30,step=1,value=0,
                                     label="Rotation (± degrees)",
                                     info="0 = off. Each window rotated by a random angle in this range")

                gr.Markdown("**Photometric** — same factor per window")
                aug_brightness_in=gr.Slider(minimum=0,maximum=0.5,step=0.05,value=0.0,
                                            label="Brightness jitter (±)",info="0 = off")
                aug_contrast_in=gr.Slider(minimum=0,maximum=0.5,step=0.05,value=0.0,
                                          label="Contrast jitter (±)",info="0 = off")
                aug_saturation_in=gr.Slider(minimum=0,maximum=0.5,step=0.05,value=0.0,
                                            label="Saturation jitter (±)",info="0 = off")

                gr.Markdown("**Per-frame** — random subset of frames in each window")
                aug_blur_in=gr.Slider(minimum=0,maximum=1,step=0.05,value=0.35,
                                      label="Random blur fraction",
                                      info="Fraction of frames to Gaussian-blur. 0 = off")
                aug_tdrop_in=gr.Slider(minimum=0,maximum=0.5,step=0.05,value=0.15,
                                       label="Temporal dropout fraction",
                                       info="Fraction of frames replaced by a neighbor. 0 = off")

                gr.Markdown("**Class balancing** — show selected classes more often per epoch")
                aug_mult_in=gr.Slider(minimum=1,maximum=10,step=1,value=1,
                                      label="Copies per window",
                                      info="1 = off. 3 = each window seen 3× per epoch, each pass with fresh online augmentation. Epoch time scales linearly.")
                aug_excluded_in=gr.CheckboxGroup(choices=[],value=[],
                                                 label="Do NOT multiply these classes",
                                                 info="Typically exclude majority classes like 'Other' so minority classes become relatively more frequent. Updates when you change label mapping.")

            with gr.Row():
                train_btn=gr.Button("🚀 Start training",variant="primary",size="lg")
                cancel_btn=gr.Button("⛔ Cancel training",variant="stop",size="lg")
            gr.Markdown("---")
            gr.Markdown("### ④ Validation results")
            with gr.Tabs():
                with gr.Tab("Per-epoch"):
                    val_html=gr.HTML("<p style='color:#aaa;'>Training not started</p>")
                with gr.Tab("Threshold"):
                    gr.Markdown("<p style='font-size:13px;color:#555;'>Precision (blue) and "
                                "recall (orange) across thresholds, one-vs-rest. "
                                "Dashed line = threshold with the best F1.</p>")
                    with gr.Row():
                        thr_epoch_dd=gr.Dropdown(label="Epoch",choices=[],value=None,
                                                 interactive=True,scale=2)
                        thr_steps=gr.Slider(minimum=4,maximum=20,step=1,value=10,
                                            label="Granularity (steps)",scale=3)
                    thr_refresh=gr.Button("🔄 Refresh",size="sm")
                    thr_html=gr.Image(label="Precision / Recall vs threshold",
                                      show_label=False, container=False)
                with gr.Tab("Ethogram"):
                    gr.Markdown("<p style='font-size:13px;color:#555;'>Validation videos: "
                                "<b style='color:#E8B84B;'>prediction (yellow, top)</b> vs "
                                "<b style='color:#378ADD;'>ground truth (blue, bottom)</b>. Other excluded.</p>")
                    with gr.Row():
                        eth_epoch_dd=gr.Dropdown(label="Epoch",choices=[],value=None,
                                                 interactive=True,scale=2)
                        eth_vid_dd=gr.Dropdown(label="Validation video",choices=[],
                                               value=None,interactive=True,scale=3)
                    eth_refresh=gr.Button("🔄 Refresh",size="sm")
                    eth_img=gr.Image(show_label=False, container=False)

    # ===== WIRING =====

    # Install the mapper behaviour once the page is ready
    demo.load(None, None, None, js=MAPPER_JS)
    demo.load(list_models,[repo_in],[model_dd,model_st])
    load_btn.click(load_pretrained,[repo_in,model_dd],[model_st])

    # Scan outputs: status, dist, nav, N dropdown updates, vid_dd, img, info, tl, scrubber, cursor, vid_list, summary
    scan_outputs = [scan_st, label_dist_html, nav_md, *map_dds, vid_dd,
                    frame_img, info_html, timeline_html, scrubber, cursor_state, vid_list_html, mapping_summary]

    # Class-balancing excluded-classes checkbox: keep choices in sync with current
    # training classes (new_names). Defined early so scan_d.click().then(...) can reference it.
    def _update_excluded_choices(head_mode, *dd_vals):
        data_labels = S["label_names"]
        pretrained_names = S["cfg"]["class_names"] if S["cfg"] else []
        if not data_labels:
            return gr.update(choices=[], value=[])
        vals = list(dd_vals[:len(data_labels)])
        new_names, _ = compute_label_map_from_dropdowns(head_mode, vals, data_labels, pretrained_names)
        prev = dd_vals[-1] if dd_vals else []
        prev = list(prev) if prev else []
        default_excluded = [n for n in new_names if n.lower() in ("other","others")]
        kept = [v for v in prev if v in new_names]
        value = kept if kept else default_excluded
        return gr.update(choices=list(new_names), value=value)

    # Demo button → download from HF + auto-scan (same outputs as Load folder, paths stay untouched)
    # Seed the visual mapper (JS) from Python state
    def _seed_mapper(head_mode, *dd_vals):
        return _mapper_init_js(S["label_names"], head_mode,
                               S["cfg"]["class_names"] if S["cfg"] else [],
                               list(dd_vals))
    lm_seed = gr.Textbox(visible=False, elem_id="lm_seed")
    # Apply the seed config in the browser. js= is used because gr.HTML does
    # not run inline <script> tags in Gradio 6.
    lm_seed.change(None, lm_seed, None, js="""
      (cfg) => {
        const go = () => {
          if (!window.lmInit) { setTimeout(go, 120); return; }
          try { window.lmInit(typeof cfg === "string" ? JSON.parse(cfg) : cfg); }
          catch (e) { console.error("lmInit failed:", e); }
        };
        go();
      }
    """)

    demo_btn.click(load_demo_training, [repo_in, vr_in, val_seed_in, head_mode_dd, *map_dds], scan_outputs
                   ).then(_update_excluded_choices,
                          [head_mode_dd, *map_dds, aug_excluded_in], aug_excluded_in
                   ).then(_seed_mapper, [head_mode_dd, *map_dds], lm_seed)

    # Load folder → scan user's own directories
    scan_d.click(do_scan_and_preview, [vdir_in, ldir_in, vr_in, val_seed_in, head_mode_dd, *map_dds], scan_outputs
                 ).then(_update_excluded_choices,
                        [head_mode_dd, *map_dds, aug_excluded_in], aug_excluded_in
                 ).then(_seed_mapper, [head_mode_dd, *map_dds], lm_seed)

    # Head mode change → rebuild all mapping dropdowns + timeline + summary
    map_change_outputs = [*map_dds, timeline_html, cursor_state, mapping_summary]
    head_mode_dd.change(on_head_mode_change, [head_mode_dd, *map_dds], map_change_outputs)

    # Visual mapper → dropdowns → (existing) mapping refresh chain
    lm_bridge.change(apply_mapper_bridge, [lm_bridge, head_mode_dd, *map_dds], map_dds) \
             .then(on_mapping_change, [head_mode_dd, *map_dds], map_change_outputs) \
             .then(_update_excluded_choices,
                   [head_mode_dd, *map_dds, aug_excluded_in], aug_excluded_in)

    head_mode_dd.change(_seed_mapper, [head_mode_dd, *map_dds], lm_seed)

    # Any mapping dropdown change → rebuild others + timeline + summary
    for dd in map_dds:
        dd.change(on_mapping_change, [head_mode_dd, *map_dds], map_change_outputs)

    # Keep excluded-classes checkbox in sync on mapping / head-mode changes
    head_mode_dd.change(_update_excluded_choices,
                        [head_mode_dd, *map_dds, aug_excluded_in], aug_excluded_in)
    for dd in map_dds:
        dd.change(_update_excluded_choices,
                  [head_mode_dd, *map_dds, aug_excluded_in], aug_excluded_in)

    # Val ratio or val seed change → recompute split
    vr_in.change(on_val_ratio_change,[vr_in,val_seed_in],[vid_list_html])
    val_seed_in.change(on_val_ratio_change,[vr_in,val_seed_in],[vid_list_html])

    # Video dropdown → preview with mapping
    vid_dd.change(on_vid_change,[vid_dd, head_mode_dd, *map_dds],
                  [frame_img,info_html,timeline_html,scrubber,cursor_state,vid_list_html,nav_md])

    # Scrubber
    scrubber.input(fn=None,inputs=[scrubber,cursor_state],outputs=[scrubber],js=CURSOR_JS)
    scrubber.change(on_scrub,[scrubber, head_mode_dd, *map_dds],[frame_img,info_html])

    # Nav
    prev_btn.click(lambda hm,*dd: do_nav("prev",hm,*dd),[head_mode_dd,*map_dds],
                   [frame_img,info_html,timeline_html,scrubber,cursor_state,nav_md,vid_list_html])
    next_btn.click(lambda hm,*dd: do_nav("next",hm,*dd),[head_mode_dd,*map_dds],
                   [frame_img,info_html,timeline_html,scrubber,cursor_state,nav_md,vid_list_html])

    # reveal the inputs when "separate validation folder" is ticked
    sep_val_cb.change(lambda on: gr.update(visible=on), [sep_val_cb], [sep_val_grp])

    # Training — pass head_mode + all mapping dropdowns instead of label_cb
    train_btn.click(run_training,
                    [repo_in,model_dd,vdir_in,ldir_in,odir_in,head_mode_dd,
                     ep_in,bs_in,lr_in,vr_in,val_seed_in,train_seed_in,
                     nw_in,cache_local_cb,
                     sep_val_cb,val_vdir_in,val_ldir_in,
                     aug_blur_in,aug_tdrop_in,aug_hflip_in,aug_vflip_in,
                     aug_rot_in,aug_brightness_in,aug_contrast_in,aug_saturation_in,
                     aug_mult_in,aug_excluded_in,
                     *map_dds],
                    [progress_html,val_html]) \
             .then(refresh_threshold_epochs, None, [thr_epoch_dd]) \
             .then(build_threshold_table, [thr_epoch_dd, thr_steps], [thr_html]) \
             .then(refresh_threshold_epochs, None, [eth_epoch_dd]) \
             .then(list_val_videos, [eth_epoch_dd], [eth_vid_dd]) \
             .then(build_val_ethogram, [eth_epoch_dd, eth_vid_dd], [eth_img])
    cancel_btn.click(cancel_training, [], [progress_html])

    # Threshold analysis: recompute when epoch or granularity changes
    thr_epoch_dd.change(build_threshold_table, [thr_epoch_dd, thr_steps], [thr_html])
    thr_steps.change(build_threshold_table, [thr_epoch_dd, thr_steps], [thr_html])
    thr_refresh.click(refresh_threshold_epochs, None, [thr_epoch_dd]) \
               .then(build_threshold_table, [thr_epoch_dd, thr_steps], [thr_html])

    # Val ethogram: pick epoch -> refresh video list -> draw
    eth_epoch_dd.change(list_val_videos, [eth_epoch_dd], [eth_vid_dd]) \
                .then(build_val_ethogram, [eth_epoch_dd, eth_vid_dd], [eth_img])
    eth_vid_dd.change(build_val_ethogram, [eth_epoch_dd, eth_vid_dd], [eth_img])
    eth_refresh.click(refresh_threshold_epochs, None, [eth_epoch_dd]) \
               .then(list_val_videos, [eth_epoch_dd], [eth_vid_dd]) \
               .then(build_val_ethogram, [eth_epoch_dd, eth_vid_dd], [eth_img])

demo.launch(debug=True,share=True)
