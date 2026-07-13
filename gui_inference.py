"""
Gradio GUI for animal behavior inference.

Usage:
    python gui_inference.py
"""

import os, json, time, shutil, zipfile, tempfile
import numpy as np
import torch
import gradio as gr
import pandas as pd
from PIL import Image
from collections import Counter
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, list_repo_files

from models import build_model_from_config
from config_utils import normalize_config, find_config_for_pth
from inference import preprocess, infer_video_gen, remap_with_disabled, get_others_idx
import precrop

# ==================== 👇 修改這裡 👇 ====================
HF_REPO_ID         = "yiheng266/animal-social-models"
DEFAULT_VIDEO_DIR  = "/content/drive/My Drive/videos/"
DEFAULT_OUTPUT_DIR = "/content/drive/My Drive/results/"
DEFAULT_LOCAL_MODEL_DIRS = []
# ==================== 👆 修改以上即可 👆 ====================

VIDEO_CACHE_DIR = os.path.join(os.path.expanduser("~"), "oab_inference_cache")

def cache_video_to_local(src_path, cache_dir=VIDEO_CACHE_DIR):
    """Copy src_path to cache_dir and return the local path."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, os.path.basename(src_path))
        if os.path.exists(dst):
            try:
                if os.path.getsize(dst) == os.path.getsize(src_path):
                    return dst
            except Exception:
                pass
        shutil.copy2(src_path, dst)
        return dst
    except Exception as e:
        print(f"⚠️ Failed to cache {src_path}: {e}; falling back to original path")
        return src_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ====================== State ======================

S = {"model": None, "cfg": None, "results": {}, "cur": None, "vr": None,
     "done": [], "idx": 0, "_active_vdir": None,
     "_cursor_data": json.dumps({"T": 0, "names": [], "labels": []}),
     "model_source": None, "disabled_classes": set(),
     "_cancel_inference": False}

CLR_PALETTE    = ["#378ADD","#D85A30","#E24B4A","#7F77DD","#1D9E75","#BA7517","#888780"]
CLR_BG_PALETTE = ["rgba(55,138,221,0.9)","rgba(216,90,48,0.9)","rgba(226,75,74,0.9)",
                   "rgba(127,119,221,0.9)","rgba(29,158,117,0.9)","rgba(186,117,23,0.9)","rgba(136,135,128,0.9)"]

def get_clr(cfg, i):
    names = cfg["class_names"]
    nm = names[i] if i < len(names) else "others"
    if nm.lower() in ("others", "other"):
        return "#FFFFFF", "rgba(180,180,180,0.9)"
    idx = i % len(CLR_PALETTE)
    return CLR_PALETTE[idx], CLR_BG_PALETTE[idx]

U = gr.update()

# ====================== Model Loading ======================

def list_models(repo):
    try:
        files = list_repo_files(repo)
        pth_files = [f for f in files if f.endswith("/model.pth") or f == "model.pth"]
        if not pth_files:
            pth_files = [f for f in files if f.endswith(".pth")]
        if not pth_files:
            return gr.update(choices=[], value=None), "❌ No models found"
        model_names = [os.path.dirname(p) if "/" in p else p for p in pth_files]
        model_names = [m for m in model_names if not m.startswith("k400_")]
        return gr.update(choices=model_names, value=model_names[0] if model_names else None), f"✅ {len(model_names)} model(s)"
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ {e}"

def _post_load_updates(cfg):
    names = cfg["class_names"]
    toggle_choices = [nm for nm in names if nm.lower() not in ("others", "other")]
    S["disabled_classes"] = set()
    nf = cfg["backbone"]["num_frames"]
    info_html = (f"<div style='font-size:12px;color:#555;padding:6px 10px;background:#f7f7f7;border-radius:6px;'>"
                 f"<b>Window:</b> 16 frames &nbsp;·&nbsp; <b>Stride:</b> 4 frames &nbsp;·&nbsp; "
                 f"<b>Model frames:</b> {nf} &nbsp;·&nbsp; "
                 f"<b>Backbone:</b> {cfg['backbone']['name']}</div>")
    return (
        gr.update(choices=toggle_choices, value=toggle_choices, visible=True),
        _html_behavior_toggles(cfg),
        info_html,
    )

def _html_behavior_toggles(cfg):
    if not cfg:
        return "<p style='color:#aaa;font-size:13px;'>Load a model first</p>"
    names = cfg["class_names"]
    items = ""
    for i, nm in enumerate(names):
        if nm.lower() in ("others", "other"):
            continue
        _, bg = get_clr(cfg, i)
        items += (f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:6px;"
                  f"padding:3px 10px;border-radius:12px;background:{bg};color:white;"
                  f"font-size:12px;font-weight:600;'>{nm}</span>")
    return f"<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:center;'>{items}</div>"

def load_model_hf(repo, model_name):
    if not model_name or not repo:
        return "❌ Specify repo & model", U, "", ""
    try:
        if "/" in model_name or not model_name.endswith(".pth"):
            cfg_file = f"{model_name}/config.json"
            pth_file = f"{model_name}/model.pth"
        else:
            cfg_file = "config.json"
            pth_file = model_name
        with open(hf_hub_download(repo_id=repo, filename=cfg_file)) as f:
            raw = json.load(f)
        cfg, err = normalize_config(raw)
        if err:
            return f"❌ {err}", U, "", ""
        pth_path = hf_hub_download(repo_id=repo, filename=pth_file)
        model = build_model_from_config(cfg)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.to(device).eval()
        S.update({"model": model, "cfg": cfg, "results": {}, "done": [], "cur": None, "vr": None, "model_source": "hf"})
        toggle_upd, toggle_html, infer_info = _post_load_updates(cfg)
        return (f"✅ Loaded from HuggingFace!\n"
                f"  Model: {model_name}\n"
                f"  Backbone: {cfg['backbone']['name']} | Frames: {cfg['backbone']['num_frames']}\n"
                f"  Classes ({len(cfg['class_names'])}): {cfg['class_names']}\n"
                f"  Device: {device}",
                toggle_upd, toggle_html, infer_info)
    except Exception as e:
        return f"❌ {e}", U, "", ""

def load_model_local(local_dir, pth_name):
    if not pth_name or not local_dir:
        return "❌ Select a local model", U, "", ""
    pth_path = os.path.join(local_dir, pth_name)
    if not os.path.exists(pth_path):
        return f"❌ File not found: {pth_path}", U, "", ""
    try:
        cfg, cfg_source, err = find_config_for_pth(pth_path)
        if err:
            return f"❌ {err}\n\n💡 Place a config.json or <name>_config.json in same folder.", U, "", ""
        model = build_model_from_config(cfg)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.to(device).eval()
        S.update({"model": model, "cfg": cfg, "results": {}, "done": [], "cur": None, "vr": None, "model_source": "local"})
        toggle_upd, toggle_html, infer_info = _post_load_updates(cfg)
        size_mb = os.path.getsize(pth_path) / 1024**2
        return (f"✅ Loaded local model!\n"
                f"  File: {pth_name} ({size_mb:.1f} MB)\n"
                f"  Config: {cfg_source}\n"
                f"  Backbone: {cfg['backbone']['name']} | Frames: {cfg['backbone']['num_frames']}\n"
                f"  Classes ({len(cfg['class_names'])}): {cfg['class_names']}\n"
                f"  Device: {device}",
                toggle_upd, toggle_html, infer_info)
    except Exception as e:
        import traceback
        return f"❌ Load failed: {e}\n\n{traceback.format_exc()}", U, "", ""

def scan_local_models(local_dir):
    if not local_dir or not os.path.isdir(local_dir):
        return gr.update(choices=[], value=None), "❌ Folder not found"
    pth_files = []
    for f in sorted(os.listdir(local_dir)):
        fp = os.path.join(local_dir, f)
        if f.endswith(".pth") and os.path.isfile(fp):
            pth_files.append(f)
        elif os.path.isdir(fp):
            for sf in sorted(os.listdir(fp)):
                if sf.endswith(".pth"):
                    pth_files.append(os.path.join(f, sf))
    if not pth_files:
        return gr.update(choices=[], value=None), "❌ No .pth files found"
    return gr.update(choices=pth_files, value=pth_files[0]), f"✅ {len(pth_files)} model(s) found"

def on_toggle_change(enabled_behaviors):
    if not S["cfg"]:
        return ""
    names = S["cfg"]["class_names"]
    disabled = set()
    for i, nm in enumerate(names):
        if nm.lower() in ("others", "other"):
            continue
        if nm not in enabled_behaviors:
            disabled.add(i)
    S["disabled_classes"] = disabled
    if disabled:
        return f"⚠️ Disabled → Other: {[names[i] for i in sorted(disabled)]}"
    return "✅ All behaviors active"

DEMO_LOCAL_DIR = os.path.join(os.path.expanduser("~"), "demo_data")

def load_demo_inference(repo):
    """Download all files from HF demo/, scan videos, preview first one.
    Does NOT modify the video folder path textbox."""
    if not repo:
        return "", gr.update(choices=[], value=None), "❌ Specify repo", None, "<p style='color:#aaa;'>Select a video</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    try:
        all_files = list_repo_files(repo)
        demo_files = [f for f in all_files if f.startswith("demo/") and f != "demo/"]
        if not demo_files:
            return "", gr.update(choices=[], value=None), "❌ No files in demo/ folder", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]
        os.makedirs(DEMO_LOCAL_DIR, exist_ok=True)
        for f in demo_files:
            local = hf_hub_download(repo_id=repo, filename=f)
            fname = os.path.basename(f)
            dest = os.path.join(DEMO_LOCAL_DIR, fname)
            if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(local):
                import shutil; shutil.copy2(local, dest)
        # Scan for videos
        videos = sorted([f for f in os.listdir(DEMO_LOCAL_DIR) if f.lower().endswith((".mp4", ".avi", ".mov"))])
        if not videos:
            return "", gr.update(choices=[], value=None), "⚠️ No videos in demo/", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]
        # Store active dir in state
        S["_active_vdir"] = DEMO_LOCAL_DIR
        # Preview first video
        vf = videos[0]
        vp = os.path.join(DEMO_LOCAL_DIR, vf)
        try:
            vr = VideoReader(vp, ctx=cpu(0))
            S["_preview_vr"] = vr; S["_preview_vf"] = vf
            T = len(vr); fps = vr.get_avg_fps()
            info = (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                    f"<span style='font-size:12px;color:#666;'>F: 0/{T} | 0.00s/{T/fps:.2f}s</span></div>")
            img = vr[0].asnumpy()
        except:
            T = 0; info = ""; img = None
        return ("", gr.update(choices=videos, value=vf),
                f"✅ Demo loaded: {len(videos)} video(s)",
                img, info, gr.update(maximum=max(T - 1, 0), value=0), "", S["_cursor_data"])
    except Exception as e:
        return "", gr.update(choices=[], value=None), f"❌ {e}", None, "", gr.update(maximum=0, value=0), "", S["_cursor_data"]

def scan_videos_and_preview(vdir):
    """Load folder: scan videos and preview the first one.

    Generator. Yields (batch_prog, video_dd, scan_st, frame_img, info_html,
    scrubber, timeline, cursor) so Drive-loading progress shows up in the SAME
    two-tier progress card that batch inference uses.
    """
    empty = (gr.update(choices=[], value=None), "❌ Not found", None, "",
             gr.update(maximum=0, value=0), "", S["_cursor_data"])
    if not vdir or not os.path.isdir(vdir):
        yield "", *empty
        return
    v = sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    if not v:
        yield "", gr.update(choices=[], value=None), "❌ No videos", None, "", \
              gr.update(maximum=0, value=0), "", S["_cursor_data"]
        return
    S["_active_vdir"] = vdir

    hold = (U,) * 7
    total = len(v)
    t0 = time.perf_counter()

    # Open each video once so it's actually pulled/decoded from Drive, and
    # report progress into the shared card.
    for i, name in enumerate(v):
        yield (html_progress(i, total, name, 0, 1,
                             elapsed=time.perf_counter() - t0,
                             title="Loading", unit="files", show_rate=False,
                             done_label="✅ Loaded"),
               *hold)
        try:
            _vr = VideoReader(os.path.join(vdir, name), ctx=cpu(0))
            del _vr
        except Exception:
            pass

    # Preview first video
    vf = v[0]
    vp = os.path.join(vdir, vf)
    try:
        vr = VideoReader(vp, ctx=cpu(0))
        S["_preview_vr"] = vr; S["_preview_vf"] = vf
        T = len(vr); fps = vr.get_avg_fps()
        info = (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                f"<span style='font-size:12px;color:#666;'>F: 0/{T} | 0.00s/{T/fps:.2f}s</span></div>")
        img = vr[0].asnumpy()
    except:
        T = 0; info = ""; img = None

    yield (html_progress(total, total, vf, 1, 1,
                         elapsed=time.perf_counter() - t0,
                         title="Loading", unit="files", show_rate=False,
                         done_label="✅ Loaded"),
           gr.update(choices=v, value=vf), f"✅ {len(v)} videos",
           img, info, gr.update(maximum=max(T - 1, 0), value=0), "", S["_cursor_data"])

# ====================== Pre-crop (YOLO) ======================

def run_precrop(yolo_model_path, video_dir, crop_padding):
    """Crop all videos in video_dir with the YOLO tracker, then switch the
    active folder to the cropped output and preview the first cropped clip.

    Yields (run_crop_btn, batch_prog, *7 preview outputs). The button is
    disabled ("Cropping...") for the whole run so it can't be double-clicked,
    and re-enabled on every exit path.
    """
    hold = (U,) * 7  # video_dd, scan_st, frame_img, info_html, scrubber, timeline, cursor
    BUSY = gr.update(value="⏳ Cropping...", interactive=False)
    IDLE = gr.update(value="✂️ Run crop", interactive=True)

    def _err(m):
        return (f"<div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;"
                f"padding:10px 14px;color:#e74c3c;font-weight:600;font-size:13px;'>{m}</div>")

    if not yolo_model_path:
        yield IDLE, _err("❌ Enter the YOLO model path (.pt)"), *hold
        return
    if not video_dir or not os.path.isdir(video_dir):
        yield IDLE, _err("❌ Load a valid video folder first"), *hold
        return

    t0 = time.perf_counter()
    try:
        crop_padding = float(crop_padding)
    except Exception:
        crop_padding = 0.3

    out_dir, outputs = None, []
    try:
        for ev in precrop.crop_folder(yolo_model_path, video_dir,
                                      crop_padding=crop_padding, device=0):
            if ev["type"] == "progress":
                # outer bar = videos, inner bar = frames of the current video
                yield (BUSY,
                       html_progress(ev["vid_i"] - 1, ev["vid_n"], ev["video"],
                                     ev["frame"], ev["total"],
                                     elapsed=time.perf_counter() - t0,
                                     title="Cropping", unit="frames",
                                     show_rate=False, done_label="✅ Cropped"),
                       *hold)
            elif ev["type"] == "done":
                out_dir, outputs = ev["output_dir"], ev["outputs"]
    except Exception as e:
        yield IDLE, _err(f"❌ Crop failed: {e}"), *hold
        return

    if not outputs:
        yield IDLE, _err("❌ No videos were cropped"), *hold
        return

    n = len(outputs)
    yield (BUSY,
           html_progress(n, n, f"{n} video(s) → {out_dir}", 1, 1,
                         elapsed=time.perf_counter() - t0,
                         title="Cropping", unit="frames",
                         show_rate=False, done_label="✅ Cropped"),
           *hold)

    # Switch active folder to the cropped output and preview it. Keep the
    # button disabled through the reload, then re-enable on the last yield.
    # Use a one-step lookahead so the Loading card still streams live (draining
    # the generator into a list first would freeze it until the reload ended).
    prev = None
    for scan_out in scan_videos_and_preview(out_dir):
        if prev is not None:
            yield BUSY, *prev
        prev = scan_out
    if prev is not None:
        yield IDLE, *prev
    else:
        yield IDLE, "", *hold


# ====================== HTML Builders ======================

def html_progress(vd, vt, cur_name, wd, wt, ws=None, elapsed=None, stride=4,
                  title="Batch", unit="windows", show_rate=True, done_label=None):
    """Two-tier progress card (outer = videos, inner = current item).

    Reused by batch inference, Drive loading and preprocessing — pass a
    different ``title``/``unit`` instead of building a separate widget.
    """
    if vt == 0: return ""
    vp = (vd / vt) * 100; wp = (wd / max(wt, 1)) * 100
    vc = "#1D9E75" if vd == vt else "#D85A30"
    st = (done_label or "✅ Complete") if vd == vt else "Processing..."
    rate_str = ""
    if show_rate and elapsed and elapsed > 0.1 and wd > 0:
        wps = wd / elapsed
        fps = wd * stride / elapsed
        rate_str = f" · {wps:.1f} {unit[:3]}/s · {fps:.1f} fps"
    return f"""<div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:13px;font-weight:600;">{title} — {st}</span>
        <span style="font-size:12px;color:#888;">{vd}/{vt} videos</span></div>
      <div style="height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px;">
        <div style="width:{vp:.1f}%;height:100%;background:{vc};border-radius:4px;"></div></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;font-weight:500;">Current: {cur_name}</span>
        <span style="font-size:12px;color:#888;">{wd}/{wt} {unit}{rate_str}</span></div>
      <div style="height:6px;background:#eee;border-radius:3px;overflow:hidden;">
        <div style="width:{wp:.1f}%;height:100%;background:#1D9E75;border-radius:3px;"></div></div>
    </div>"""

def html_timeline(vf):
    r = S["results"].get(vf)
    if not r: return ""
    cfg = S["cfg"]; names = cfg["class_names"]; labels = r["frame_labels"]; T = len(labels)
    if T == 0: return ""
    segs = []; cur, cnt = labels[0], 1
    for i in range(1, T):
        if labels[i] == cur: cnt += 1
        else: segs.append((cur, cnt)); cur, cnt = labels[i], 1
    segs.append((cur, cnt))
    bar = ""
    for li, c in segs:
        clr, _ = get_clr(cfg, li); pct = (c / T) * 100
        bdr = "border-top:1px solid #ccc;border-bottom:1px solid #ccc;" if clr == "#FFFFFF" else ""
        nm = names[li] if li < len(names) else "?"
        bar += f"<div style='width:{pct:.3f}%;height:100%;background:{clr};{bdr}display:inline-block;box-sizing:border-box;' title='{nm}'></div>"
    leg = ""
    for i, nm in enumerate(names):
        clr, _ = get_clr(cfg, i)
        bdr = "border:1px solid #ccc;" if clr == "#FFFFFF" else ""
        leg += (f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:12px;'>"
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{clr};{bdr}'></span>{nm}</span>")
    nm0 = names[labels[0]] if labels[0] < len(names) else "?"
    return f"""<div style="width:100%;padding:4px 0;">
      <div style="position:relative;display:flex;height:18px;border-radius:4px;overflow:hidden;border:1px solid #ccc;">
        {bar}
        <div id="tl-cursor" style="position:absolute;top:-2px;bottom:-2px;width:2px;background:#000;left:0%;pointer-events:none;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
        <div>{leg}</div>
        <span id="tl-frame-label" style="font-size:12px;font-weight:500;color:#555;">F:0 {nm0}</span>
      </div></div>"""

def html_behavior(vf):
    r = S["results"].get(vf)
    if not r: return "<p style='color:#aaa;'>Run inference first</p>"
    cfg = S["cfg"]; names = cfg["class_names"]; T = r["total_frames"]; fps = r["fps"]
    cnt = Counter(r["frame_labels"])
    h = f"<div style='padding:4px 0;'>"
    h += f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;'><span style='font-size:14px;font-weight:600;'>Behavior statistics</span><span style='font-size:12px;color:#888;'>{vf}</span></div>"
    for i, nm in enumerate(names):
        c = cnt.get(i, 0); pct = 100 * c / max(T, 1); dur = c / max(fps, 1)
        clr, _ = get_clr(cfg, i); bar_clr = "#ddd" if clr == "#FFFFFF" else clr
        h += f"<div style='margin-bottom:8px;'><div style='display:flex;justify-content:space-between;margin-bottom:2px;'><span style='font-size:13px;font-weight:500;'>{nm}</span><span style='font-size:12px;color:#888;'>{c:,} fr · {pct:.1f}% · {dur:.1f}s</span></div><div style='height:8px;background:#f0f0f0;border-radius:4px;overflow:hidden;'><div style='width:{max(pct,0.3):.1f}%;height:100%;background:{bar_clr};border-radius:4px;'></div></div></div>"
    h += f"<div style='display:flex;gap:10px;margin-top:10px;padding-top:8px;border-top:1px solid #eee;'><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>Frames</div><div style='font-size:16px;font-weight:600;'>{T:,}</div></div><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>FPS</div><div style='font-size:16px;font-weight:600;'>{fps:.1f}</div></div><div style='flex:1;background:#f7f7f7;border-radius:6px;padding:6px;text-align:center;'><div style='font-size:11px;color:#888;'>Duration</div><div style='font-size:16px;font-weight:600;'>{T/fps:.1f}s</div></div></div></div>"
    return h

def html_export_preview(vf, fmt):
    r = S["results"].get(vf)
    if not r: return "<p style='color:#aaa;font-size:13px;'>Run inference first</p>"
    names = S["cfg"]["class_names"]; labels = r["frame_labels"]; fps = r["fps"]
    td = "padding:3px 6px;border-bottom:1px solid #eee;font-size:11px;font-family:monospace;"
    th = f"{td}font-weight:bold;color:#666;"
    if fmt == "One-hot CSV (per-frame)":
        hdr = f"<tr><td style='{th}'>frame</td>" + "".join(f"<td style='{th}'>{n[:6]}</td>" for n in names) + "</tr>"
        rows = "".join(
            f"<tr><td style='{td}'>{i}</td>" + "".join(f"<td style='{td}'>{'1' if labels[i]==ci else '0'}</td>" for ci in range(len(names))) + "</tr>"
            for i in range(min(5, len(labels)))
        ) + f"<tr><td style='{td}' colspan='{len(names)+1}'>... ({len(labels)} rows)</td></tr>"
        title = "One-hot CSV preview"
    else:
        hdr = f"<tr><td style='{th}'>time</td><td style='{th}'>media</td><td style='{th}'>subj</td><td style='{th}'>behavior</td><td style='{th}'>status</td></tr>"
        evts = []
        if labels:
            cur, st = labels[0], 0
            for i in range(1, len(labels)):
                if labels[i] != cur: evts.append((st, i, cur)); cur, st = labels[i], i
            evts.append((st, len(labels), cur))
        rows = "".join(
            f"<tr><td style='{td}'>{s/fps:.3f}</td><td style='{td}'>{vf[:12]}</td><td style='{td}'>pair</td><td style='{td}'>{names[c] if c<len(names) else '?'}</td><td style='{td}'>START</td></tr>"
            f"<tr><td style='{td}'>{e/fps:.3f}</td><td style='{td}'>{vf[:12]}</td><td style='{td}'>pair</td><td style='{td}'>{names[c] if c<len(names) else '?'}</td><td style='{td}'>STOP</td></tr>"
            for s, e, c in evts[:3]
        ) + f"<tr><td style='{td}' colspan='5'>... ({len(evts)} events)</td></tr>"
        title = "BORIS event log preview"
    return f"<div style='margin-top:4px;'><p style='font-size:12px;color:#666;font-weight:600;margin:0 0 4px;'>{title}</p><div style='overflow-x:auto;border:1px solid #eee;border-radius:4px;'><table style='border-collapse:collapse;width:100%;'>{hdr}{rows}</table></div></div>"

def update_export_preview(fmt):
    return html_export_preview(S["cur"], fmt) if S["cur"] else "<p style='color:#aaa;'>No results</p>"

# ====================== Display ======================

def preview_frame(vdir, vf, fi):
    """Get a frame from a video file directly (no inference needed)."""
    if not vf or not vdir:
        return None
    vp = os.path.join(vdir, vf)
    if not os.path.exists(vp):
        return None
    try:
        if S.get("_preview_vf") != vf or S.get("_preview_vr") is None:
            S["_preview_vr"] = VideoReader(vp, ctx=cpu(0))
            S["_preview_vf"] = vf
        vr = S["_preview_vr"]
        fi = max(0, min(int(fi), len(vr) - 1))
        return vr[fi].asnumpy()
    except:
        return None

def preview_info_html(vdir, vf, fi):
    """Frame info for preview mode (no inference labels)."""
    if not vf or not vdir:
        return "<p style='color:#aaa;'>Select a video to preview</p>"
    # If we have inference results, use those
    r = S["results"].get(vf)
    if r:
        return frame_info_html(vf, fi)
    # Otherwise just show frame / time info
    vp = os.path.join(vdir, vf)
    if not os.path.exists(vp):
        return "<p style='color:#aaa;'>Video not found</p>"
    try:
        if S.get("_preview_vf") != vf or S.get("_preview_vr") is None:
            S["_preview_vr"] = VideoReader(vp, ctx=cpu(0))
            S["_preview_vf"] = vf
        vr = S["_preview_vr"]
        T = len(vr); fps = vr.get_avg_fps(); fi = max(0, min(int(fi), T - 1))
        return (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                f"<span style='font-size:12px;color:#666;'>F: {fi}/{T} | {fi/fps:.2f}s/{T/fps:.2f}s</span></div>")
    except:
        return "<p style='color:#aaa;'>Cannot read video</p>"

def _vdir(vdir_input):
    """Return active video dir: prefer S state (set by demo/load), fallback to textbox."""
    return S.get("_active_vdir") or vdir_input

def on_video_select(vf):
    """When user selects a video from dropdown, show first frame + set scrubber."""
    vdir = S.get("_active_vdir")
    if not vf or not vdir:
        return None, "<p style='color:#aaa;'>Select a video</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    # If we have inference results for this video, show full view
    r = S["results"].get(vf)
    if r:
        S["cur"] = vf; S["vr"] = None; _update_cursor(vf)
        T = r["total_frames"]
        return (get_frame(vf, 0), frame_info_html(vf, 0),
                gr.update(maximum=max(T - 1, 0), value=0),
                html_timeline(vf), S["_cursor_data"])
    # No results yet — just preview raw video
    vp = os.path.join(vdir, vf)
    if not os.path.exists(vp):
        return None, "<p style='color:#aaa;'>Video not found</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]
    try:
        vr = VideoReader(vp, ctx=cpu(0))
        S["_preview_vr"] = vr; S["_preview_vf"] = vf
        T = len(vr); fps = vr.get_avg_fps()
        info = (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='padding:4px 12px;border-radius:6px;background:rgba(180,180,180,0.7);color:white;font-size:13px;font-weight:600;'>Preview</span>"
                f"<span style='font-size:12px;color:#666;'>F: 0/{T} | 0.00s/{T/fps:.2f}s</span></div>")
        return vr[0].asnumpy(), info, gr.update(maximum=max(T - 1, 0), value=0), "", S["_cursor_data"]
    except Exception as e:
        return None, f"<p style='color:#aaa;'>Error: {e}</p>", gr.update(maximum=0, value=0), "", S["_cursor_data"]

def get_frame(vf, fi):
    r = S["results"].get(vf)
    if not r: return None
    if S["cur"] != vf or S["vr"] is None:
        S["vr"] = VideoReader(r["video_path"], ctx=cpu(0)); S["cur"] = vf
    return S["vr"][max(0, min(fi, len(S["vr"]) - 1))].asnumpy()

def frame_info_html(vf, fi):
    r = S["results"].get(vf)
    if not r: return "<p style='color:#aaa;'>Run inference first</p>"
    cfg = S["cfg"]; names = cfg["class_names"]; T = r["total_frames"]; fps = r["fps"]
    fi = max(0, min(fi, T - 1)); li = r["frame_labels"][fi]; nm = names[li]
    conf = r["frame_confidences"][fi][li] * 100; _, bg = get_clr(cfg, li)
    return (f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='padding:4px 12px;border-radius:6px;background:{bg};color:white;font-size:13px;font-weight:600;'>{nm} ({conf:.0f}%)</span>"
            f"<span style='font-size:12px;color:#666;'>F: {fi}/{T} | {fi/fps:.2f}s/{T/fps:.2f}s</span></div>")

def nav_md():
    d = S["done"]; i = S["idx"]
    if not d: return "*No results*"
    return f"**{d[i]}** — {i+1}/{len(d)} completed"

def _update_cursor(vf):
    r = S["results"].get(vf)
    if not r: S["_cursor_data"] = json.dumps({"T": 0, "names": [], "labels": []})
    else: S["_cursor_data"] = json.dumps({"T": r["total_frames"], "names": S["cfg"]["class_names"], "labels": r["frame_labels"]})

def _full(vf, fi=0, vd=0, vt=0):
    r = S["results"].get(vf)
    if not r:
        e = ""; return e, e, None, e, e, e, "*No results*", gr.update(maximum=0, value=0), S["_cursor_data"]
    S["cur"] = vf; S["vr"] = None
    if vf in S["done"]: S["idx"] = S["done"].index(vf)
    T = r["total_frames"]; _update_cursor(vf)
    return (html_progress(vd, vt, vf, T, T), frame_info_html(vf, fi), get_frame(vf, fi),
            html_timeline(vf), html_behavior(vf),
            html_export_preview(vf, "One-hot CSV (per-frame)"),
            nav_md(), gr.update(maximum=max(T - 1, 0), value=0), S["_cursor_data"])

# ====================== Actions ======================

def cancel_inference():
    S["_cancel_inference"] = True
    return "<p style='color:#e74c3c;font-weight:600;'>⛔ Cancelling… will stop after current window.</p>"

def run_single(vf, num_workers, cache_local):
    S["_cancel_inference"] = False
    vdir = S.get("_active_vdir")
    if not S["model"]: yield "", "", None, "", "", "", "❌ Load model first", U, S["_cursor_data"]; return
    if not vf or not vdir: yield "", "", None, "", "", "", "❌ Select video", U, S["_cursor_data"]; return
    ws = S["cfg"]["backbone"]["num_frames"] if S.get("cfg") else None

    # Cache video to local if requested
    infer_vdir = vdir
    if cache_local:
        vp = os.path.join(vdir, vf)
        cached = cache_video_to_local(vp)
        if cached != vp:
            infer_vdir = os.path.dirname(cached)
            yield "<p style='font-size:13px;color:#888;'>📦 Cached to local disk</p>", U, U, U, U, U, U, U, U

    t0 = time.perf_counter()
    result = None
    for msg in infer_video_gen(infer_vdir, vf, S["model"], S["cfg"], S["disabled_classes"]):
        if S["_cancel_inference"]:
            yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Inference cancelled.</p>", U, U, U, U, U, U, U, U
            print(f"⛔ Inference cancelled: {vf}")
            return
        if isinstance(msg, dict): result = msg
        elif isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "prep":
            _, pd_, pt_ = msg
            yield html_progress(0, 1, vf, pd_, pt_,
                                elapsed=time.perf_counter()-t0,
                                title="Preprocessing", unit="steps",
                                show_rate=False), U, U, U, U, U, U, U, U
        else:
            wd, wt = msg
            yield html_progress(0, 1, vf, wd, wt, ws=ws, elapsed=time.perf_counter()-t0), U, U, U, U, U, U, U, U
    # Store result with original video path so frame preview works
    if result and cache_local:
        result["video_path"] = os.path.join(vdir, vf)
    S["results"][vf] = result
    if vf not in S["done"]: S["done"].append(vf)
    yield _full(vf, 0, 1, 1)

def run_batch(num_workers, cache_local):
    S["_cancel_inference"] = False
    vdir = S.get("_active_vdir")
    if not S["model"]: yield "", "", None, "", "", "", "❌ Load model first", U, S["_cursor_data"], ""; return
    if not vdir or not os.path.isdir(vdir): yield "", "", None, "", "", "", "❌ Load videos first", U, S["_cursor_data"], ""; return
    vids = sorted([f for f in os.listdir(vdir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    if not vids: yield "", "", None, "", "", "", "❌ No videos", U, S["_cursor_data"], ""; return
    ws = S["cfg"]["backbone"]["num_frames"] if S.get("cfg") else None
    total = len(vids); blog = []

    # Cache all videos upfront if requested
    cache_map = {}  # vf -> cached dir
    if cache_local:
        cache_t0 = time.perf_counter()
        for ci, vf in enumerate(vids):
            if S["_cancel_inference"]:
                blog.append(f"⛔ Cancelled during caching")
                yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Cancelled.</p>", U, U, U, U, U, U, U, U, "\n".join(blog)
                return
            vp = os.path.join(vdir, vf)
            cached = cache_video_to_local(vp)
            if cached != vp:
                cache_map[vf] = os.path.dirname(cached)
            mb = 0
            try: mb = os.path.getsize(cached) / (1024*1024)
            except: pass
            elapsed = time.perf_counter() - cache_t0
            yield (html_progress(ci, total, f"{vf} ({mb:.0f} MB)", ci, total,
                                 elapsed=elapsed, title="Loading from Drive",
                                 unit="files", show_rate=False,
                                 done_label="✅ Loaded"),
                   U, U, U, U, U, U, U, U, U)
        blog.append(f"📦 Cached {len(cache_map)} video(s) to local disk")

    for vi, vf in enumerate(vids):
        if S["_cancel_inference"]:
            blog.append(f"⛔ Cancelled at video {vi}/{total}")
            yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Batch cancelled.</p>", U, U, U, U, U, U, U, U, "\n".join(blog)
            print(f"⛔ Batch inference cancelled at video {vi}/{total}")
            return
        infer_vdir = cache_map.get(vf, vdir)
        t0 = time.perf_counter()
        result = None
        for msg in infer_video_gen(infer_vdir, vf, S["model"], S["cfg"], S["disabled_classes"]):
            if S["_cancel_inference"]:
                blog.append(f"⛔ Cancelled during {vf}")
                yield "<p style='color:#e74c3c;font-weight:600;'>⛔ Batch cancelled.</p>", U, U, U, U, U, U, U, U, "\n".join(blog)
                print(f"⛔ Batch inference cancelled during {vf}")
                return
            if isinstance(msg, dict): result = msg
            elif isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "prep":
                _, pd_, pt_ = msg
                yield html_progress(vi, total, vf, pd_, pt_,
                                    elapsed=time.perf_counter()-t0,
                                    title="Preprocessing", unit="steps",
                                    show_rate=False), U, U, U, U, U, U, U, U, U
            else:
                wd, wt = msg
                yield html_progress(vi, total, vf, wd, wt, ws=ws, elapsed=time.perf_counter()-t0), U, U, U, U, U, U, U, U, U
        # Store with original path for frame preview
        if result and vf in cache_map:
            result["video_path"] = os.path.join(vdir, vf)
        S["results"][vf] = result
        if vf not in S["done"]: S["done"].append(vf)
        blog.append(f"✅ {vf} ({result['total_frames']} fr)")
        S["cur"] = vf; S["vr"] = None; _update_cursor(vf)
        if vf in S["done"]: S["idx"] = S["done"].index(vf)
        T = result["total_frames"]
        yield (html_progress(vi + 1, total, vf, T, T, ws=ws, elapsed=time.perf_counter()-t0),
               frame_info_html(vf, 0), get_frame(vf, 0),
               html_timeline(vf), html_behavior(vf),
               html_export_preview(vf, "One-hot CSV (per-frame)"),
               nav_md(), gr.update(maximum=max(T - 1, 0), value=0),
               S["_cursor_data"], "\n".join(blog))

def on_scrub(fi):
    fi = int(fi)
    vdir = S.get("_active_vdir")
    vf = S["cur"]
    # If inference results exist, use them
    if vf and vf in S["results"]:
        return get_frame(vf, fi), frame_info_html(vf, fi)
    # Otherwise, preview mode — use the currently selected video
    preview_vf = S.get("_preview_vf")
    if preview_vf and vdir:
        return preview_frame(vdir, preview_vf, fi), preview_info_html(vdir, preview_vf, fi)
    return None, "<p style='color:#aaa;'>Select a video to preview</p>"

def do_nav(direction):
    d = S["done"]
    if not d: return "", "", None, "", "", "", "*No results*", gr.update(), S["_cursor_data"]
    if direction == "prev": S["idx"] = max(0, S["idx"] - 1)
    else: S["idx"] = min(len(d) - 1, S["idx"] + 1)
    return _full(d[S["idx"]], 0, len(d), len(d))

# ====================== Export ======================

def _exp_onehot(vf, od):
    if vf not in S["results"]: return "❌"
    r = S["results"][vf]; names = S["cfg"]["class_names"]; nc = len(names)
    os.makedirs(od, exist_ok=True)
    rows = [[1 if l == ci else 0 for ci in range(nc)] for l in r["frame_labels"]]
    df = pd.DataFrame(rows, columns=names); df.insert(0, "frame", range(len(rows)))
    p = os.path.join(od, vf.rsplit(".", 1)[0] + "_onehot.csv"); df.to_csv(p, index=False)
    return f"✅ {p}"

def _exp_boris(vf, od):
    if vf not in S["results"]: return "❌"
    r = S["results"][vf]; names = S["cfg"]["class_names"]; fps = r["fps"]; L = r["frame_labels"]
    os.makedirs(od, exist_ok=True); evts = []; cur, st = L[0], 0
    for i in range(1, len(L)):
        if L[i] != cur:
            evts += [{"Time": round(st/fps, 3), "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "START"},
                     {"Time": round(i/fps, 3),  "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "STOP"}]
            cur, st = L[i], i
    evts += [{"Time": round(st/fps, 3),       "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "START"},
             {"Time": round(len(L)/fps, 3),    "Media": vf, "Subject": "pair", "Behavior": names[cur], "Status": "STOP"}]
    p = os.path.join(od, vf.rsplit(".", 1)[0] + "_boris.csv"); pd.DataFrame(evts).to_csv(p, index=False)
    return f"✅ {p}"

def do_export_cur(vf, od, fmt):
    vf = S["cur"]
    if not vf: return "❌"
    return _exp_onehot(vf, od) if fmt == "One-hot CSV (per-frame)" else _exp_boris(vf, od)

def do_export_all(od, fmt):
    if not S["done"]: return "❌"
    return "\n".join(_exp_onehot(v, od) if fmt == "One-hot CSV (per-frame)" else _exp_boris(v, od) for v in S["done"])

# ====================== Ethogram ======================

def _ethogram_png(vf, od):
    """Render one ethogram PNG for a video: behavior on Y, frames on X,
    coloured bars where each behavior is active. Uses the same palette as the
    inline timeline so colours match across the app."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if vf not in S["results"] or not S["results"][vf]:
        return None, f"❌ {vf}: no result"
    r = S["results"][vf]
    names = S["cfg"]["class_names"]
    nc = len(names)
    # Guard: clamp any label that falls outside class_names (e.g. after a
    # disabled-class remap) so it can't raise IndexError mid-plot.
    L = [l if isinstance(l, (int,)) and 0 <= l < nc else nc - 1
         for l in r["frame_labels"]]
    T = len(L)
    if T == 0:
        return None, f"❌ {vf}: empty"

    os.makedirs(od, exist_ok=True)

    # Contiguous runs -> bars (far fewer artists than one per frame)
    runs = []
    cur, st = L[0], 0
    for i in range(1, T):
        if L[i] != cur:
            runs.append((cur, st, i - st)); cur, st = L[i], i
    runs.append((cur, st, T - st))

    nc = len(names)
    fig, ax = plt.subplots(figsize=(10, max(2.0, 0.42 * nc + 1.0)), dpi=150)
    ax.set_facecolor("#ebebeb")

    for cls, start, width in runs:
        ax.broken_barh([(start, width)], (nc - 1 - cls - 0.36, 0.72),
                       facecolors=CLR_PALETTE[cls % len(CLR_PALETTE)],
                       edgecolors="white", linewidth=0.4)

    ax.set_yticks(range(nc))
    ax.set_yticklabels(list(reversed(names)), fontsize=9)
    ax.set_ylim(-0.6, nc - 0.4)
    ax.set_xlim(0, T)
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("Predictions", fontsize=10, fontweight="bold")
    ax.set_title(vf, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="white", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Secondary axis in seconds
    fps = r.get("fps") or 0
    if fps:
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / fps, lambda t: t * fps))
        sec.set_xlabel("Time (s)", fontsize=9)
        sec.tick_params(labelsize=8)

    present = sorted({c for c, _, _ in runs})
    ax.legend(handles=[Patch(facecolor=CLR_PALETTE[c % len(CLR_PALETTE)], label=names[c])
                       for c in present],
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=min(len(present), 4), fontsize=8, frameon=False)

    fig.tight_layout()
    p = os.path.join(od, vf.rsplit(".", 1)[0] + "_ethogram.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, f"✅ {p}"


def do_ethogram_zip(od):
    """Make one ethogram PNG per inferred video, bundle into a zip for download.
    Returns (zip_path_for_gr.File, log). Never raises — errors go to the log so
    the UI shows a real message instead of a bare Gradio error box."""
    try:
        return _do_ethogram_zip(od)
    except ModuleNotFoundError as e:
        return None, (f"❌ Missing package: {e.name}\n"
                      f"   Install it, then restart the GUI:\n"
                      f"   !pip install matplotlib")
    except Exception as e:
        import traceback
        return None, f"❌ Ethogram failed: {type(e).__name__}: {e}\n\n{traceback.format_exc()}"


def _do_ethogram_zip(od):
    if not S.get("cfg"):
        return None, "❌ Load a model first (need class names)"

    vids = [v for v in S["done"] if S["results"].get(v)]
    if not vids:
        return None, "❌ Run inference first"

    if not od:
        return None, "❌ Set the 'Save to' folder in the Data tab"

    eth_dir = os.path.join(od, "ethograms")
    os.makedirs(eth_dir, exist_ok=True)

    pngs, log = [], []
    for v in vids:
        try:
            p, msg = _ethogram_png(v, eth_dir)
        except Exception as e:
            p, msg = None, f"❌ {v}: {type(e).__name__}: {e}"
        log.append(msg)
        if p:
            pngs.append(p)

    if not pngs:
        return None, "\n".join(log) or "❌ Nothing to plot"

    # Build the zip in the system temp dir. gr.File refuses to serve files from
    # arbitrary locations (e.g. Google Drive) — they must live in the CWD or a
    # temp dir — so the download copy goes to /tmp and we ALSO drop a copy next
    # to the PNGs on Drive for the user's own records.
    tmp_zip = os.path.join(tempfile.gettempdir(), "ethograms.zip")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=os.path.basename(p))

    log.append(f"📦 {len(pngs)} ethogram(s) → {eth_dir}")

    drive_zip = os.path.join(od, "ethograms.zip")
    try:
        shutil.copyfile(tmp_zip, drive_zip)
        log.append(f"💾 Saved a copy to {drive_zip}")
    except Exception as e:
        log.append(f"⚠️ Could not copy zip to {od}: {e}")

    return tmp_zip, "\n".join(log)


# ====================== Cursor JS ======================

CURSOR_JS = """
(fi, labels_json) => {
    let T=0, names=[], labels=[];
    try { const d=JSON.parse(labels_json); T=d.T; names=d.names; labels=d.labels; } catch(e) { return fi; }
    if (T===0) return fi;
    fi = Math.max(0, Math.min(Math.floor(fi), T-1));
    const cursor = document.getElementById('tl-cursor');
    if (cursor) cursor.style.left = ((fi/T)*100)+'%';
    const lbl = document.getElementById('tl-frame-label');
    if (lbl) { const cls=labels[fi]; lbl.textContent='F:'+fi+' '+(names[cls]||'?'); }
    return fi;
}
"""

# ====================== GUI ======================

GREEN_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
)

# ====================== Custom CSS ======================
# 全域樣式覆蓋：字體全黑、標題放大、格線加深、背景加深對比更明顯。
CUSTOM_CSS = """
/* 背景加深（配綠色主題的偏綠灰），區塊維持亮白 → 對比更明顯 */
.gradio-container { background: #cdd6cd !important; }
.gr-box, .block, .form, .gr-panel, .gr-accordion,
.gradio-container .prose { background: #ffffff !important; }

/* 全域字體轉全黑（覆蓋 Gradio 灰色 secondary / tertiary 變數） */
.gradio-container, .gradio-container * {
    color: #000000 !important;
    --color-text-primary: #000000;
    --color-text-secondary: #000000;
    --color-text-tertiary: #000000;
}

/* 邊框策略：不要對每個 .block / .form 畫完整外框（那會讓 gr.Group 內的
   元件各自變成一個框、無法黏合）。只對「實際輸入元件」和 accordion 畫框，
   格線一樣清楚，但 group 內的元件會自然黏成一整塊。 */
input, textarea, select,
.gr-input, .gr-dropdown, .gr-accordion {
    border: 1.5px solid #555555 !important;
}
/* 下拉選單本體（Gradio 把 dropdown 包在一層 div，需要單獨補框） */
.gradio-container [class*="dropdown"] > label > div,
.gradio-container [data-testid="dropdown"] {
    border: 1.5px solid #555555 !important;
    border-radius: 6px !important;
}

/* 區塊標題（Markdown 的 h1 / h2 / h3）放大一點點 */
.gradio-container h1 { font-size: 30px !important; font-weight: 700 !important; }
.gradio-container h2 { font-size: 22px !important; font-weight: 700 !important; }
.gradio-container h3 { font-size: 18px !important; font-weight: 700 !important; }

/* 元件 label 也轉黑加粗 */
label, .gr-input-label, span[data-testid="block-info"] {
    color: #000000 !important; font-weight: 600 !important;
}

/* 分頁列（HuggingFace / Local folder）給白底，讓它跟背景區隔開 */
.tab-nav, div[role="tablist"] {
    background: #ffffff !important;
    border: 1.5px solid #555555 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 2px 2px 0 2px !important;
    margin-bottom: 0 !important;
}
.tab-nav button, div[role="tablist"] button {
    background: #ffffff !important;
    color: #000000 !important;
    font-weight: 600 !important;
}
/* 分頁內容緊貼分頁列：去掉上方間隙、上緣不要圓角，讓它跟分頁列連成一體 */
.tabitem {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.tabitem > .gap:first-child,
.tabitem > *:first-child {
    margin-top: 0 !important;
}
"""

with gr.Blocks(title="Animal Behavior Inference", theme=GREEN_THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown("# Animal Social Behavior Inference\nHuggingFace & Local models — auto config detection — behavior filtering")

    cursor_state = gr.Textbox(value=S["_cursor_data"], visible=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ① Select model")
            with gr.Tabs():
                with gr.TabItem("☁️ HuggingFace"):
                    with gr.Group():
                        repo_in = gr.Textbox(value=HF_REPO_ID, label="HF Repo ID", interactive=True)
                        hf_model_dd = gr.Dropdown(label="Model", choices=[], interactive=True)
                        model_st = gr.Textbox(label="Model status", interactive=False, lines=5)
                    hf_load_btn = gr.Button("📥 Load model", variant="primary")
                with gr.TabItem("💾 Local folder"):
                    with gr.Group():
                        local_dir_in = gr.Textbox(label="Model folder path", value=DEFAULT_LOCAL_MODEL_DIRS[0] if DEFAULT_LOCAL_MODEL_DIRS else "", interactive=True)
                        local_model_dd = gr.Dropdown(label="Model (.pth)", choices=[], interactive=True)
                        model_st_local = gr.Textbox(label="Model status", interactive=False, lines=5)
                    with gr.Row():
                        local_scan_btn = gr.Button("🔍 Scan folder", variant="secondary", size="sm")
                        local_load_btn = gr.Button("📥 Load model", variant="primary")
            gr.Markdown("---")
            gr.Markdown("### ② Load video folder")
            with gr.Group():
                vdir_in = gr.Textbox(label="Video folder path", value=DEFAULT_VIDEO_DIR)
                scan_st = gr.Textbox(label="Folder status", interactive=False, lines=1)
            demo_btn = gr.Button("🎯 Load Demo", variant="secondary", size="sm")
            load_folder_btn = gr.Button("📂 Load folder", variant="secondary")

            gr.Markdown("---")
            precrop_toggle = gr.Checkbox(label="✂️ Enable pre-crop (YOLO tracker)", value=False)
            with gr.Group(visible=False) as precrop_panel:
                yolo_model_in = gr.Textbox(
                    label="YOLO model path (.pt on Drive)",
                    value="/content/drive/MyDrive/squid/model/best.pt")
                crop_pad_in = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                                        label="Crop padding")
                run_crop_btn = gr.Button("✂️ Run crop", variant="primary")

        with gr.Column(scale=2, min_width=400):
            with gr.Group():
                toggle_label_html = gr.HTML("<p style='color:#aaa;font-size:13px;'>Load a model to see behaviors</p>")
                behavior_toggles = gr.CheckboxGroup(label="Active behaviors (unchecked → merged to Other)", choices=[], value=[], interactive=True, visible=False)
                toggle_status = gr.Textbox(interactive=False, lines=1, visible=False, show_label=False)
                batch_prog = gr.HTML("")
                info_html = gr.HTML("<p style='color:#aaa;'>Load a model and run inference</p>")
                frame_img = gr.Image(label="Frame preview", type="numpy", interactive=False)
                timeline_html = gr.HTML("")
                scrubber = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame", interactive=True)
            with gr.Row():
                prev_btn = gr.Button("◀ Previous video", size="sm")
                nav_md_out = gr.Markdown("*No results yet*")
                next_btn = gr.Button("Next video ▶", size="sm")
            gr.Markdown("---")
            behavior_html = gr.HTML("<p style='color:#aaa;'>Run inference to see statistics</p>")

        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ③ Inference")
            with gr.Group():
                video_dd = gr.Dropdown(label="Select video", choices=[], interactive=True)
                batch_log_tb = gr.Textbox(label="Batch log", interactive=False, lines=6)
            with gr.Accordion("⚙️ Advanced settings", open=False):
                infer_info_html = gr.HTML("<p style='color:#aaa;font-size:12px;'>Load a model to see window/stride settings</p>")
                nw_in = gr.Slider(minimum=0, maximum=8, step=1, value=0,
                                  label="Num workers (data loading)",
                                  info="0 = main thread only. Windows: keep at 0 to avoid freezes.")
                cache_local_cb = gr.Checkbox(label="Cache videos to local disk before inference", value=False,
                                             info="Copies videos to local SSD first. Useful when reading from network/Drive.")
            batch_btn = gr.Button("📦 Batch inference (all videos)", variant="primary", size="lg")
            run_btn = gr.Button("🚀 Run inference (single)", variant="secondary")
            cancel_btn = gr.Button("⛔ Cancel inference", variant="stop")
            gr.Markdown("---")
            gr.Markdown("### ④ Export")
            with gr.Tabs():
                with gr.Tab("Data"):
                    with gr.Group():
                        exp_fmt = gr.Dropdown(label="Output format", choices=["One-hot CSV (per-frame)", "BORIS event log"], value="One-hot CSV (per-frame)", interactive=True)
                        exp_prev = gr.HTML("<p style='color:#aaa;font-size:13px;'>Run inference first</p>")
                        out_dir = gr.Textbox(label="Save to", value=DEFAULT_OUTPUT_DIR)
                    exp_cur = gr.Button("💾 Export current video", variant="primary")
                    exp_all = gr.Button("📦 Export all (batch)")
                    exp_log = gr.Textbox(label="Export log", interactive=False, lines=6)

                with gr.Tab("Ethogram"):
                    gr.Markdown(
                        "<p style='font-size:13px;color:#555;'>One ethogram PNG per inferred "
                        "video, bundled into a zip.</p>")
                    eth_btn = gr.Button("🎨 Generate ethograms + download zip", variant="primary")
                    eth_file = gr.File(label="ethograms.zip", interactive=False)
                    eth_log = gr.Textbox(label="Ethogram log", interactive=False, lines=6)

    demo.load(list_models, [repo_in], [hf_model_dd, model_st])
    hf_load_btn.click(load_model_hf, [repo_in, hf_model_dd], [model_st, behavior_toggles, toggle_label_html, infer_info_html])
    local_scan_btn.click(scan_local_models, [local_dir_in], [local_model_dd, model_st_local])
    local_load_btn.click(load_model_local, [local_dir_in, local_model_dd], [model_st_local, behavior_toggles, toggle_label_html, infer_info_html])
    behavior_toggles.change(on_toggle_change, [behavior_toggles], [toggle_status])

    # Shared outputs for demo/load folder. batch_prog is FIRST so Drive-loading
    # progress renders in the same two-tier card that batch inference uses.
    load_outputs = [batch_prog, video_dd, scan_st, frame_img, info_html, scrubber, timeline_html, cursor_state]
    demo_btn.click(load_demo_inference, [repo_in], load_outputs)
    load_folder_btn.click(scan_videos_and_preview, [vdir_in], load_outputs)

    # Pre-crop: toggle panel + run crop then swap folder to cropped output.
    # Uses load_outputs so crop progress lands in the shared progress card.
    precrop_toggle.change(lambda on: gr.update(visible=on), [precrop_toggle], [precrop_panel])
    run_crop_btn.click(
        run_precrop,
        [yolo_model_in, vdir_in, crop_pad_in],
        [run_crop_btn] + load_outputs,
    )

    # Video selection triggers preview (frame + scrubber setup)
    video_dd.change(on_video_select, [video_dd], [frame_img, info_html, scrubber, timeline_html, cursor_state])

    out9 = [batch_prog, info_html, frame_img, timeline_html, behavior_html, exp_prev, nav_md_out, scrubber, cursor_state]
    out10 = out9 + [batch_log_tb]

    run_btn.click(run_single, [video_dd, nw_in, cache_local_cb], out9)
    batch_btn.click(run_batch, [nw_in, cache_local_cb], out10)
    cancel_btn.click(cancel_inference, [], [batch_prog])
    scrubber.input(fn=None, inputs=[scrubber, cursor_state], outputs=[scrubber], js=CURSOR_JS)
    scrubber.change(on_scrub, inputs=[scrubber], outputs=[frame_img, info_html])
    prev_btn.click(lambda: do_nav("prev"), [], out9)
    next_btn.click(lambda: do_nav("next"), [], out9)
    exp_fmt.change(update_export_preview, [exp_fmt], [exp_prev])
    exp_cur.click(do_export_cur, [video_dd, out_dir, exp_fmt], [exp_log])
    exp_all.click(do_export_all, [out_dir, exp_fmt], [exp_log])
    eth_btn.click(do_ethogram_zip, [out_dir], [eth_file, eth_log])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
