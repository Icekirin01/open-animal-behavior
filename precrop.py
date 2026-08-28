"""
precrop.py — YOLO detect-and-crop preprocessing for the training / inference GUIs.

This is a faithful port of the user's proven Colab crop script (the one that
does NOT run out of RAM). The important choices are copied verbatim because
they are what keep memory flat across many long videos:

  · Read frames with cv2.VideoCapture (streaming, one frame at a time). decord's
    VideoReader caches decoded frames internally and OOMs on long clips.
  · Write with cv2.VideoWriter (mp4v). Proven stable in the user's runs.
  · Single-frame model.predict (NOT model.track / BoT-SORT) + a custom EMA
    SmoothBox for temporal stability.
  · Half precision when the GPU supports it (probed once).
  · Per-video reset_tracker(): break every predictor reference, gc.collect(),
    torch.cuda.empty_cache() + ipc_collect(). This is what stops fragmentation
    from accumulating video-to-video.
  · frame_log is dropped per video.

The GUI wrapper adds: stage pending source videos to fast local disk with four
copy workers, write each cropped clip to a local output dir (used by training)
AND mirror it to a Drive backup dir, skip already-done videos, retain the
source cache for fast reruns, and yield throttled progress dicts for the GUI.
"""

import os, glob, gc, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---- crop parameters (verbatim from the user's working script) ----
CONF_THRESHOLD = 0.15
IMG_SIZE       = 512
CROP_PADDING   = 0.3
OUTPUT_WIDTH   = 224
OUTPUT_HEIGHT  = 224
SMOOTH_ALPHA   = 0.08
MAX_SPEED      = 15
MISS_TOLERANCE = 99999

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')

# Where original videos are staged before cropping (fast local SSD, not Drive).
# Keep the same location and four-copy pipeline as the proven Colab script.
LOCAL_CACHE_DIR = "/content/_in"
STAGE_WORKERS = 4
MAX_LOCAL_DISK_FRACTION = 0.85
GUI_PROGRESS_INTERVAL_S = 1.0


# ====================== Smoothing + crop (verbatim) ======================

class SmoothBox:
    """EMA + speed clamp; when nothing is detected, reuse the last box so the
    output stays frame-aligned with the source."""
    def __init__(self, alpha=SMOOTH_ALPHA, max_speed=MAX_SPEED,
                 miss_tolerance=MISS_TOLERANCE):
        self.alpha = alpha
        self.max_speed = max_speed
        self.miss_tolerance = miss_tolerance
        self.smooth_box = None
        self.miss_count = 0

    def update(self, box):
        if box is None:
            self.miss_count += 1
            if self.miss_count > self.miss_tolerance:
                self.smooth_box = None
            return None if self.smooth_box is None else self.smooth_box.tolist()

        self.miss_count = 0
        new_box = np.array(box, dtype=np.float64)

        if self.smooth_box is None:
            self.smooth_box = new_box
            return self.smooth_box.tolist()

        diff = np.clip(new_box - self.smooth_box, -self.max_speed, self.max_speed)
        clamped = self.smooth_box + diff
        self.smooth_box = self.smooth_box * (1 - self.alpha) + clamped * self.alpha
        return self.smooth_box.tolist()


def center_crop(frame, out_w, out_h, interp):
    import cv2
    h, w = frame.shape[:2]
    s = min(h, w)
    cy, cx = h // 2, w // 2
    center = frame[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2]
    return cv2.resize(center, (out_w, out_h), interpolation=interp)


def crop_and_upscale(frame, bbox, padding, out_w, out_h, interp):
    import cv2
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * padding; x2 += bw * padding
    y1 -= bh * padding; y2 += bh * padding

    target_ratio = out_w / out_h
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w / max(crop_h, 1) < target_ratio:
        diff = crop_h * target_ratio - crop_w
        x1 -= diff / 2; x2 += diff / 2
    else:
        diff = crop_w / target_ratio - crop_h
        y1 -= diff / 2; y2 += diff / 2

    cw, ch = x2 - x1, y2 - y1
    if cw <= w_frame:
        if x1 < 0: x2 -= x1; x1 = 0
        if x2 > w_frame: x1 -= (x2 - w_frame); x2 = w_frame
    if ch <= h_frame:
        if y1 < 0: y2 -= y1; y1 = 0
        if y2 > h_frame: y1 -= (y2 - h_frame); y2 = h_frame

    x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
    x2 = min(w_frame, int(round(x2))); y2 = min(h_frame, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return center_crop(frame, out_w, out_h, interp)

    crop = frame[y1:y2, x1:x2]
    interp2 = interp if (crop.shape[1] >= out_w and crop.shape[0] >= out_h) \
        else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_w, out_h), interpolation=interp2)


def reset_tracker(model, use_cuda):
    """Reset the predictor and release CUDA cache between videos. Break every
    reference the predictor holds so GC can reclaim it (verbatim from the
    working script — this is the key to flat memory)."""
    import torch
    p = getattr(model, 'predictor', None)
    if p is not None:
        for attr in ('trackers', 'results', 'batch', 'dataset', 'vid_writer'):
            if hasattr(p, attr):
                try:
                    setattr(p, attr, None)
                except Exception:
                    pass
    model.predictor = None
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ====================== File discovery ======================

def list_videos(video_dir):
    """List video files, including ones where Google Drive appended a copy
    marker after the extension (e.g. 'D-1102-8.mp4 的副本')."""
    import re
    copy_tail = (r"(?:\s*的副本|\s*-\s*副本|\s*副本|\s*—\s*副本|\s*-?\s*[Cc]opy"
                 r"|\s*-?\s*copy\s*\d*|\s*\(\d+\))*")
    pat = re.compile(r"(" + "|".join(re.escape(e) for e in VIDEO_EXTS) + r")"
                     + copy_tail + r"$", re.IGNORECASE)
    vids = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
            if pat.search(f)]
    return sorted(set(vids))


def cache_video_to_local(src_path, cache_dir=LOCAL_CACHE_DIR):
    """Copy src_path to a fast local dir and return the local path. If a
    same-sized copy already exists, reuse it. Falls back to the source path on
    error. (Same idea as the batch-inference script's cache_video_to_local.)"""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, os.path.basename(src_path))
        if os.path.exists(dst):
            try:
                if os.path.getsize(dst) == os.path.getsize(src_path):
                    return dst
            except Exception:
                pass
        shutil.copyfile(src_path, dst)
        return dst
    except Exception as e:
        print(f"⚠️ cache failed for {src_path}: {e}; using original path")
        return src_path


def _stage_videos(video_paths, cache_dir=LOCAL_CACHE_DIR,
                  workers=STAGE_WORKERS):
    """Stage all input videos concurrently, as in the proven Colab script.

    Returns futures rather than blocking internally so ``preprocess_folder``
    can continue streaming staging progress to Gradio. Existing same-sized
    local files are reused without copying.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    futures = {
        pool.submit(cache_video_to_local, src, cache_dir): src
        for src in video_paths
    }
    return pool, futures


def _bytes_needed_for_stage(video_paths, cache_dir=LOCAL_CACHE_DIR):
    """Return bytes that still need to be copied into the local stage."""
    needed = 0
    for src in video_paths:
        try:
            dst = os.path.join(cache_dir, os.path.basename(src))
            src_size = os.path.getsize(src)
            if not (os.path.exists(dst) and os.path.getsize(dst) == src_size):
                needed += src_size
        except OSError:
            pass
    return needed


def _frame_count(path):
    """Frame count of a video via cv2, or -1 if unreadable."""
    import cv2
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return -1
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    except Exception:
        return -1


def _is_complete_crop(cropped_path, src_path):
    """A cropped clip counts as done only if it exists AND has (about) the same
    frame count as its source — cropping is frame-aligned, so a mismatch means
    the file was truncated (e.g. the session was interrupted mid-video)."""
    import os
    if not cropped_path or not os.path.exists(cropped_path):
        return False
    if os.path.getsize(cropped_path) == 0:
        return False
    out_n = _frame_count(cropped_path)
    if out_n <= 0:
        return False
    src_n = _frame_count(src_path)
    if src_n <= 0:
        # can't read source frame count — accept a non-empty readable output
        return True
    # allow a tiny tolerance for codecs that report ±1
    return abs(out_n - src_n) <= 1


def _cropped_name(vid_path):
    return f"{Path(vid_path).stem}_cropped_{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.mp4"


# ====================== Model loading ======================

def _load_model(model_path):
    """Load YOLO, decide device + precision. Returns (model, device, use_cuda,
    precision_kw). Mirrors the working script's probe."""
    from ultralytics import YOLO
    import torch

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else 'cpu'
    model = YOLO(model_path)

    precision_kw = {}
    if use_cuda:
        probe = np.zeros((64, 64, 3), dtype=np.uint8)
        for kw in ({'quantize': 'half'}, {'half': True}):
            try:
                model.predict(probe, imgsz=IMG_SIZE, device=device,
                              verbose=False, **kw)
                precision_kw = kw
                break
            except (TypeError, ValueError):
                continue
        model.predictor = None
        gc.collect(); torch.cuda.empty_cache()
    return model, device, use_cuda, precision_kw


# ====================== Orchestrated preprocess ======================

def preprocess_folder(model_path, video_dir, local_out_dir, drive_out_dir=None,
                      crop_padding=CROP_PADDING, device=None, skip_existing=True,
                      delete_source_cache=False):
    """Full preprocess for every video in ``video_dir``:

        1. stage all pending originals to fast local storage with four workers,
        2. YOLO detect-and-crop it (single-frame predict + EMA smoothing),
        3. write the cropped clip to ``local_out_dir`` (used by training) AND
           mirror it to ``drive_out_dir`` (persistent backup, if given),
        4. retain the local input cache by default so reruns can reuse it.

    Reads with cv2.VideoCapture and writes with cv2.VideoWriter, resetting the
    predictor and clearing CUDA cache per video — the memory pattern from the
    user's proven script. ``device`` is ignored (kept for signature
    compatibility); the device is chosen automatically.

    Generator yielding progress dicts + a final done dict:

        {"type": "progress", "phase": "cache"|"crop", "video": name,
         "vid_i": i, "vid_n": n, "frame": f, "total": total}
        {"type": "done", "output_dir": local_out_dir, "outputs": [...]}
    """
    import cv2
    import psutil

    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not video_dir or not os.path.isdir(video_dir):
        raise NotADirectoryError(f"Video folder not found: {video_dir}")

    os.makedirs(local_out_dir, exist_ok=True)
    if drive_out_dir:
        os.makedirs(drive_out_dir, exist_ok=True)

    # OpenCV decode/resize should use all cores (Colab sometimes defaults to 1).
    try:
        cv2.setNumThreads(max(1, psutil.cpu_count(logical=True)))
    except Exception:
        pass
    interp = cv2.INTER_AREA

    videos = list_videos(video_dir)
    n = len(videos)
    outputs = []
    model = use_cuda = precision_kw = dev = None
    pending = []
    local_map = {}

    try:
        # Resolve completed outputs before staging. This preserves the GUI's
        # robust resume behaviour while avoiding copies for already-done clips.
        for vi, src_path in enumerate(videos):
            out_name = _cropped_name(src_path)
            vid_name = Path(src_path).stem
            local_out = os.path.join(local_out_dir, out_name)
            drive_out = os.path.join(drive_out_dir, out_name) if drive_out_dir else None

            # ---- skip if a COMPLETE crop already exists ----
            # After a disconnect Colab wipes /content, so the local copy is
            # gone but the Drive mirror survives. Check both, and verify the
            # file isn't truncated (frame count must match the source).
            if skip_existing:
                done_local = _is_complete_crop(local_out, src_path)
                done_drive = (drive_out is not None
                              and _is_complete_crop(drive_out, src_path))
                if done_local or done_drive:
                    # make sure the local copy exists for training to read
                    if not done_local and done_drive:
                        try:
                            shutil.copyfile(drive_out, local_out)
                        except Exception as e:
                            print(f"⚠️ Could not restore {out_name} from Drive: {e}")
                    outputs.append(local_out)
                    yield {"type": "progress", "phase": "crop", "video": vid_name,
                           "vid_i": vi + 1, "vid_n": n, "frame": 1, "total": 1,
                           "skipped": True}
                    continue

            pending.append((vi, src_path))

        # Match the proven Colab pipeline: copy all pending Drive videos to
        # /content concurrently, then keep the GPU continuously fed.
        pending_paths = [src for _, src in pending]
        if pending_paths:
            needed = _bytes_needed_for_stage(pending_paths)
            free = shutil.disk_usage("/content").free
            if needed > free * MAX_LOCAL_DISK_FRACTION:
                raise RuntimeError(
                    f"Not enough local disk to stage videos: need "
                    f"{needed / 1e9:.1f} GB, free {free / 1e9:.1f} GB"
                )

            pool, futures = _stage_videos(pending_paths)
            staged = 0
            try:
                for future in as_completed(futures):
                    src_path = futures[future]
                    local_map[src_path] = future.result()
                    staged += 1
                    yield {"type": "progress", "phase": "cache",
                           "video": Path(src_path).stem,
                           "vid_i": staged, "vid_n": len(pending_paths),
                           "frame": staged, "total": len(pending_paths)}
            finally:
                pool.shutdown(wait=True)

        for vi, src_path in pending:
            out_name = _cropped_name(src_path)
            vid_name = Path(src_path).stem
            local_out = os.path.join(local_out_dir, out_name)
            drive_out = os.path.join(drive_out_dir, out_name) if drive_out_dir else None

            local_src = local_map.get(src_path, src_path)

            # lazy-load YOLO only when there is real work
            if model is None:
                model, dev, use_cuda, precision_kw = _load_model(model_path)

            # ---- 2. crop the local copy (cv2 read + cv2 write) ----
            cap = cv2.VideoCapture(local_src)
            if not cap.isOpened():
                print(f"⚠️ Cannot open: {local_src}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            writer = cv2.VideoWriter(local_out, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            reset_tracker(model, use_cuda)
            smoother = SmoothBox(SMOOTH_ALPHA, MAX_SPEED, MISS_TOLERANCE)
            frame_idx = 0
            last_progress = time.perf_counter()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=CONF_THRESHOLD,
                                            imgsz=IMG_SIZE, device=dev,
                                            verbose=False, **precision_kw)
                    best_box, best_conf = None, -1.0
                    for r in results:
                        b = r.boxes
                        if b is None or len(b) == 0:
                            continue
                        # One argmax and one device-to-host box transfer. The
                        # old per-box loop synchronised CUDA repeatedly.
                        j = int(b.conf.argmax().item())
                        c = float(b.conf[j].item())
                        if c > best_conf:
                            best_conf = c
                            best_box = b.xyxy[j].detach().cpu().tolist()

                    smooth_box = smoother.update(best_box)
                    if smooth_box is not None:
                        cropped = crop_and_upscale(frame, smooth_box, crop_padding,
                                                   OUTPUT_WIDTH, OUTPUT_HEIGHT, interp)
                    else:
                        cropped = center_crop(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT, interp)
                    writer.write(cropped)

                    frame_idx += 1
                    now = time.perf_counter()
                    if now - last_progress >= GUI_PROGRESS_INTERVAL_S:
                        last_progress = now
                        yield {"type": "progress", "phase": "crop", "video": vid_name,
                               "vid_i": vi + 1, "vid_n": n,
                               "frame": frame_idx,
                               "total": total_frames if total_frames > 0 else frame_idx}
            finally:
                cap.release()
                writer.release()

            outputs.append(local_out)

            # ---- 3. mirror to Drive ----
            if drive_out:
                try:
                    shutil.copyfile(local_out, drive_out)
                except Exception as e:
                    print(f"⚠️ Could not mirror {out_name} to Drive: {e}")

            # ---- 4. delete the original local copy (keep the cropped one) ----
            if delete_source_cache and local_src != src_path and os.path.exists(local_src):
                try:
                    os.remove(local_src)
                except Exception:
                    pass

            # per-video cleanup (flat memory)
            reset_tracker(model, use_cuda)

            yield {"type": "progress", "phase": "crop", "video": vid_name,
                   "vid_i": vi + 1, "vid_n": n,
                   "frame": total_frames if total_frames > 0 else frame_idx,
                   "total": total_frames if total_frames > 0 else frame_idx}
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    yield {"type": "done", "output_dir": local_out_dir, "outputs": outputs}
