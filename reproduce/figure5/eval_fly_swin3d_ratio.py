"""
eval_fly_swin3d_ratio.py — Fly Copulation Swin3D Evaluation (frame-wise)

Usage:
    python eval_fly_swin3d_ratio.py --model_path checkpoints/fly_swin3d_ratio/model.pth
    python eval_fly_swin3d_ratio.py --model_path model.pth --save_cm cm.png --save_results results.json
"""

import os, argparse, json
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
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import ToTensor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import seaborn as sns

ALL_BEHAVIOR_NAMES = ["wing_extension","circle","copul_attempt","copulation","others"]
SELECTED_BEHAVIORS = ["wing_extension","circle","copulation","others"]
OTHERS_IDX = SELECTED_BEHAVIORS.index("others")
ORIGINAL_TO_NEW = {}
for oi, name in enumerate(ALL_BEHAVIOR_NAMES):
    ORIGINAL_TO_NEW[oi] = SELECTED_BEHAVIORS.index(name) if name in SELECTED_BEHAVIORS else OTHERS_IDX
NUM_CLASSES = len(SELECTED_BEHAVIORS)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Fly Swin3D (frame-wise)")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_video_dir", type=str, default="data/fly/videos/test")
    p.add_argument("--test_label_dir", type=str, default="data/fly/labels/test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_dropout", type=float, default=0.3)
    p.add_argument("--smooth_window_size", type=int, default=1)
    p.add_argument("--save_cm", type=str, default=None)
    p.add_argument("--save_results", type=str, default=None)
    return p.parse_args()

def filter_and_remap_labels(oh):
    return np.array([ORIGINAL_TO_NEW[l] for l in np.argmax(oh, axis=1)], dtype=np.int64)

def get_video_and_label_paths(vd, ld):
    vps, lps = [], []
    for n in sorted(os.listdir(vd)):
        if not n.lower().endswith(".mp4"): continue
        lp = os.path.join(ld, n.replace(".mp4",".csv"))
        if os.path.exists(lp): vps.append(os.path.join(vd,n)); lps.append(lp)
        else: print(f"[WARN] Label not found: {n}")
    return vps, lps

def custom_video_transform(frames, target_size=(224,224)):
    frames = [f.resize(target_size, Image.BILINEAR) for f in frames]
    frames = [ToTensor()(f) for f in frames]
    v = torch.stack(frames, dim=1)
    m = torch.tensor([0.485,0.456,0.406]).view(-1,1,1,1)
    s = torch.tensor([0.229,0.224,0.225]).view(-1,1,1,1)
    return (v - m) / s

def plot_cm(cm, names, path, title, normalize=False):
    fig, ax = plt.subplots(figsize=(10,8))
    d = cm.astype("float")/cm.sum(axis=1,keepdims=True)*100 if normalize else cm
    d = np.nan_to_num(d) if normalize else d
    sns.heatmap(d, annot=True, fmt=".1f" if normalize else "d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax, square=True)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"📊 {path}")

class WindowPredictionDataset(Dataset):
    def __init__(self, vps, lps, ws, stride, transform, skip=0):
        self.vps,self.lps,self.ws,self.stride,self.transform,self.skip = vps,lps,ws,stride,transform,skip
        self.windows, self.frame_mappings = self._gen()
    def _gen(self):
        windows, mappings = [], []
        for vp, lp in zip(self.vps, self.lps):
            df = pd.read_csv(lp)
            oh = df.iloc[:,0:len(ALL_BEHAVIOR_NAMES)].values
            vr = VideoReader(vp, ctx=cpu(0)); T = len(vr)
            if len(oh)!=T: continue
            remapped = filter_and_remap_labels(oh)
            sel = list(range(0,T,self.skip+1))
            sl_oh = np.zeros((len(sel),NUM_CLASSES))
            for i,fi in enumerate(sel): sl_oh[i, remapped[fi]] = 1.0
            f2w = [[] for _ in range(len(sel))]
            for s in range(0,len(sel)-self.ws+1,self.stride):
                wi = sel[s:s+self.ws]; windows.append((vp,wi)); widx=len(windows)-1
                for fi in range(s,s+self.ws):
                    if fi<len(f2w): f2w[fi].append(widx)
            mappings.append({"labels":sl_oh,"frame_to_windows":f2w})
        return windows, mappings
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        vp,fi = self.windows[idx]; vr = VideoReader(vp,ctx=cpu(0))
        frames = [Image.fromarray(f) for f in vr.get_batch(fi).asnumpy()]
        if len(frames)<self.ws: frames.extend([frames[-1]]*(self.ws-len(frames)))
        return self.transform(frames), idx

class MLPHead(nn.Module):
    def __init__(self,inf,nc,hd=512,do=0.3):
        super().__init__(); self.norm=nn.LayerNorm(inf); self.fc1=nn.Linear(inf,hd)
        self.relu=nn.ReLU(True); self.dropout=nn.Dropout(do); self.fc2=nn.Linear(hd,nc)
    def forward(self,x):
        x=torch.mean(x,dim=1); x=self.norm(x); return self.fc2(self.dropout(self.relu(self.fc1(x))))

class CustomSwin3D(nn.Module):
    def __init__(self,pretrained=False,T=8):
        super().__init__()
        self.model=swin3d_t(weights=Swin3D_T_Weights.DEFAULT if pretrained else None)
        self.T=T; self.model.head=nn.Identity(); self.model.avgpool=nn.Identity()
    def forward(self,x):
        x=self.model.patch_embed(x); x=self.model.pos_drop(x)
        x=self.model.features(x); x=self.model.norm(x); x=x.mean(dim=(2,3)); return self.head(x)

def evaluate_framewise(model, loader, mappings, smooth_k, device):
    model.eval(); wp = []
    with torch.no_grad():
        for v,_ in tqdm(loader, desc="Testing", leave=False):
            v=v.to(device)
            with autocast(): wp.extend(torch.softmax(model(v),dim=1).cpu().numpy())
    wp = np.array(wp)
    all_labels, all_fp = [], []
    for m in mappings:
        labels,f2w = m["labels"],m["frame_to_windows"]; F=len(labels)
        fp = np.full((F,NUM_CLASSES),1.0/NUM_CLASSES,dtype=np.float32)
        for f in range(F):
            if f2w[f]: fp[f]=np.mean(wp[f2w[f]],axis=0)
        all_fp.append(fp); all_labels.extend(np.argmax(labels,axis=1))
    def smooth(p,k):
        if k<=1: return p
        h=k//2; o=np.zeros_like(p)
        for i in range(len(p)): o[i]=np.mean(p[max(0,i-h):min(len(p),i+h+1)],axis=0)
        return o
    preds, raw = [], []
    for fp in all_fp: raw.extend(fp.tolist()); preds.extend(np.argmax(smooth(fp,smooth_k),axis=1).tolist())
    lbls=list(range(NUM_CLASSES))
    f1_pc=f1_score(all_labels,preds,average=None,labels=lbls)
    f1_m=f1_score(all_labels,preds,average="macro")
    non_others=[i for i in range(NUM_CLASSES) if i!=OTHERS_IDX]
    f1_no=float(np.mean(f1_pc[non_others]))
    cm=confusion_matrix(all_labels,preds,labels=lbls)
    oh=np.zeros((len(all_labels),NUM_CLASSES))
    for i,l in enumerate(all_labels): oh[i,l]=1.0
    raw=np.array(raw)
    ap_pc=np.array([average_precision_score(oh[:,c],raw[:,c]) if oh[:,c].sum()>0 else 0.0 for c in range(NUM_CLASSES)])
    acc=np.mean(np.array(all_labels)==np.array(preds))
    return {"f1_per_class":f1_pc,"f1_macro":f1_m,"f1_no_others":f1_no,"cm":cm,
            "ap_per_class":ap_pc,"mAP":np.mean(ap_pc),"accuracy":acc,"all_preds":preds,"all_labels":all_labels}

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.model_path): raise FileNotFoundError(args.model_path)

    print(f"\n{'='*70}")
    print("Fly Copulation — Swin3D Evaluation (frame-wise)")
    print(f"{'='*70}")
    print(f"  Model: {args.model_path}")
    print(f"  Classes: {SELECTED_BEHAVIORS}\n")

    vids,labs = get_video_and_label_paths(args.test_video_dir, args.test_label_dir)
    print(f"  Test videos: {len(vids)}")
    ds = WindowPredictionDataset(vids,labs,args.window_size,args.stride,custom_video_transform,args.skip)
    loader = DataLoader(ds,args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    print(f"  Test windows: {len(ds)}\n")

    model = CustomSwin3D(pretrained=False,T=8)
    model.head = MLPHead(768,NUM_CLASSES,args.mlp_hidden_dim,args.mlp_dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path,map_location=device))
    print("  ✅ Model loaded\n")

    m = evaluate_framewise(model,loader,ds.frame_mappings,args.smooth_window_size,device)

    print(f"{'='*70}\nRESULTS\n{'='*70}\n")
    print(f"  Accuracy:             {m['accuracy']:.4f}")
    print(f"  F1 Macro (all):       {m['f1_macro']:.4f}")
    print(f"  F1 Macro (no others): {m['f1_no_others']:.4f}")
    print(f"  mAP:                  {m['mAP']:.4f}\n")
    print("  Per-class F1:")
    for n,v in zip(SELECTED_BEHAVIORS,m["f1_per_class"]):
        print(f"    {n:>15}: {v:.4f}")
    print("\n  Per-class AP:")
    for n,v in zip(SELECTED_BEHAVIORS,m["ap_per_class"]):
        print(f"    {n:>15}: {v:.4f}")

    if args.save_cm:
        plot_cm(m["cm"],SELECTED_BEHAVIORS,args.save_cm,f"Fly Swin3D Test (F1={m['f1_macro']:.4f})")
        plot_cm(m["cm"],SELECTED_BEHAVIORS,args.save_cm.replace(".png","_norm.png"),
                f"Fly Swin3D Test Normalized",normalize=True)
    if args.save_results:
        r={"model_path":args.model_path,"accuracy":float(m["accuracy"]),
           "f1_macro":float(m["f1_macro"]),"f1_no_others":float(m["f1_no_others"]),
           "mAP":float(m["mAP"]),"f1_per_class":m["f1_per_class"].tolist(),
           "ap_per_class":m["ap_per_class"].tolist(),"confusion_matrix":m["cm"].tolist(),
           "behavior_names":SELECTED_BEHAVIORS}
        with open(args.save_results,"w") as f: json.dump(r,f,indent=2)
        print(f"\n💾 {args.save_results}")
    print("\n✅ Evaluation complete!")

if __name__=="__main__": main()
