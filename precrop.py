"""
precrop.py — YOLO BoT-SORT pre-crop for the inference GUI.

Adapted from the Colab "Script 3" tracker-crop workflow. Detects the target
with a YOLO model, tracks it across frames (BoT-SORT), smooths the crop box
(EMA + speed clamp), then crops + upscales each frame to a fixed size and
writes a new video per source clip.

Reading uses decord (same as the rest of the GUI); writing uses imageio /
imageio-ffmpeg, which is far more reliable on Colab than cv2.VideoWriter's
mp4v backend (that one silently produces a broken writer and can segfault the
kernel). Only CROP_PADDING and the model path are exposed in the GUI;
everything else uses the same defaults as the original script.
"""

import os, glob, gc, shutil
from pathlib import Path

import numpy as np
import cv2  # used only for resize (crop_and_upscale), not for read/write

# ---- fixed defaults (same as the original Colab script) ----
CONF_THRESHOLD = 0.5
IMG_SIZE       = 640
OUTPUT_WIDTH   = 224
OUTPUT_HEIGHT  = 224
SMOOTH_ALPHA   = 0.08
MAX_SPEED      = 15
OUTPUT_SUBDIR  = "cropped"   # created under the source video folder

# Where original videos are copied before cropping (fast local SSD, not Drive).
# Reading Drive frame-by-frame gets progressively slower; copying the whole
# file once and reading locally avoids that. Cleared per-video after cropping.
LOCAL_CACHE_DIR = "/content/oab_precrop_cache"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class SmoothBox:
    """EMA + speed limit so the crop box moves smoothly."""
    def __init__(self, alpha=SMOOTH_ALPHA, max_speed=MAX_SPEED):
        self.alpha = alpha
        self.max_speed = max_speed
        self.smooth_box = None
        self.miss_count = 0

    def update(self, box):
        if box is None:
            self.miss_count += 1
            if self.miss_count > 30:
                self.smooth_box = None
            return self.smooth_box
        self.miss_count = 0
        new_box = np.array(box, dtype=np.float64)
        if self.smooth_box is None:
            self.smooth_box = new_box
            return self.smooth_box.tolist()
        diff = np.clip(new_box - self.smooth_box, -self.max_speed, self.max_speed)
        clamped = self.smooth_box + diff
        self.smooth_box = self.smooth_box * (1 - self.alpha) + clamped * self.alpha
        return self.smooth_box.tolist()


def crop_and_upscale(frame, bbox, padding, out_w, out_h):
    """Crop bbox (with padding), keep target aspect ratio, upscale to out_w x out_h."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding
    x1 -= pad_x; y1 -= pad_y; x2 += pad_x; y2 += pad_y

    target_ratio = out_w / out_h
    crop_w, crop_h = x2 - x1, y2 - y1
    current_ratio = crop_w / max(crop_h, 1)
    if current_ratio < target_ratio:
        new_w = crop_h * target_ratio
        d = new_w - crop_w
        x1 -= d / 2; x2 += d / 2
    else:
        new_h = crop_w / target_ratio
        d = new_h - crop_h
        y1 -= d / 2; y2 += d / 2

    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w_frame, int(x2)); y2 = min(h_frame, int(y2))
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


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


def crop_folder(model_path, video_dir, crop_padding=0.3, output_dir=None, device=0):
    """
    Crop every video in ``video_dir`` and write results to ``output_dir``
    (defaults to ``video_dir/cropped``).

    ``device`` controls where YOLO runs. Default 0 = GPU. YOLO shares the GPU
    with the behavior-recognition model already loaded by the GUI, so after
    cropping finishes this function resets the tracker state and calls
    ``torch.cuda.empty_cache()`` to hand VRAM back. If you hit CUDA OOM on a
    small card, pass ``device="cpu"``.

    Reads frames with decord, writes with imageio-ffmpeg.

    This is a **generator**. It yields progress dicts while running and a
    final ``done`` dict at the end:

        {"type": "progress", "video": name, "vid_i": i, "vid_n": n,
         "frame": f, "total": total}
        {"type": "done", "output_dir": out_dir, "outputs": [paths...]}
    """
    from ultralytics import YOLO       # imported lazily so the GUI loads without it
    from decord import VideoReader, cpu
    import imageio

    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not video_dir or not os.path.isdir(video_dir):
        raise NotADirectoryError(f"Video folder not found: {video_dir}")

    if output_dir is None:
        output_dir = os.path.join(video_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    videos = list_videos(video_dir)
    outputs = []

    try:
        for vi, vid_path in enumerate(videos):
            vid_name = Path(vid_path).stem

            # ---- read with decord (same as the rest of the GUI) ----
            try:
                vr = VideoReader(vid_path, ctx=cpu(0))
            except Exception:
                continue
            total_frames = len(vr)
            fps = float(vr.get_avg_fps()) or 30.0

            out_path = os.path.join(output_dir, f"{vid_name}_cropped_{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.mp4")

            # ---- write with imageio-ffmpeg (stable on Colab) ----
            writer = imageio.get_writer(
                out_path, fps=fps, codec="libx264",
                quality=8, macro_block_size=None,  # allow 224x224 without padding
            )

            smoother = SmoothBox()
            frame_idx = 0
            try:
                for frame_idx in range(total_frames):
                    frame = vr[frame_idx].asnumpy()  # decord returns RGB

                    # YOLO expects BGR-style ndarray input; it handles RGB ndarrays
                    # fine, but tracking must run on the same colour space each call.
                    results = model.track(
                        frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE,
                        persist=True, tracker="botsort.yaml",
                        device=device, verbose=False,
                    )
                    best_box, best_conf = None, -1
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None and len(boxes) > 0:
                            for i in range(len(boxes)):
                                c = float(boxes.conf[i].cpu())
                                if c > best_conf:
                                    best_conf = c
                                    best_box = boxes.xyxy[i].cpu().numpy().tolist()

                    smooth_box = smoother.update(best_box)
                    if smooth_box is not None:
                        out_frame = crop_and_upscale(frame, smooth_box, crop_padding,
                                                     OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    else:
                        out_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                               interpolation=cv2.INTER_LANCZOS4)
                    writer.append_data(out_frame)  # RGB in, RGB out

                    if (frame_idx + 1) % 25 == 0:
                        yield {"type": "progress", "video": vid_name,
                               "vid_i": vi + 1, "vid_n": len(videos),
                               "frame": frame_idx + 1, "total": total_frames}
            finally:
                writer.close()
                del vr

            outputs.append(out_path)

            # ---- clear BoT-SORT tracker state so it doesn't carry into the
            #      next video / accumulate; also drop cached frames ----
            try:
                model.predictor.trackers[0].reset()
            except Exception:
                # fall back: drop the predictor so a fresh tracker is built next call
                model.predictor = None
            gc.collect()

            yield {"type": "progress", "video": vid_name,
                   "vid_i": vi + 1, "vid_n": len(videos),
                   "frame": total_frames, "total": total_frames}
    finally:
        # ---- release YOLO + free GPU so the behavior model gets its VRAM back ----
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    yield {"type": "done", "output_dir": output_dir, "outputs": outputs}


# ====================== Cache + orchestrated preprocess ======================

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
        shutil.copy2(src_path, dst)
        return dst
    except Exception as e:
        print(f"⚠️ cache failed for {src_path}: {e}; using original path")
        return src_path


def _cropped_name(vid_path):
    return f"{Path(vid_path).stem}_cropped_{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.mp4"


def preprocess_folder(model_path, video_dir, local_out_dir, drive_out_dir=None,
                      crop_padding=0.3, device=0, skip_existing=True,
                      delete_source_cache=True):
    """Full preprocess: for every video in ``video_dir``

        1. copy the original to fast local storage (avoids Drive slowdown),
        2. YOLO-crop it (skips if the output already exists),
        3. write the cropped clip to ``local_out_dir`` (used by training) AND
           mirror it to ``drive_out_dir`` (persistent backup, if given),
        4. delete the original local copy to save space.

    Cropped clips are kept; only the *original* local copies are deleted.

    Generator yielding the same progress dicts as ``crop_folder`` plus a final
    ``done`` dict with the local output dir:

        {"type": "progress", "phase": "cache"|"crop", "video": name,
         "vid_i": i, "vid_n": n, "frame": f, "total": total}
        {"type": "done", "output_dir": local_out_dir, "outputs": [...]}
    """
    from ultralytics import YOLO
    from decord import VideoReader, cpu
    import imageio

    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not video_dir or not os.path.isdir(video_dir):
        raise NotADirectoryError(f"Video folder not found: {video_dir}")

    os.makedirs(local_out_dir, exist_ok=True)
    if drive_out_dir:
        os.makedirs(drive_out_dir, exist_ok=True)

    videos = list_videos(video_dir)
    n = len(videos)
    outputs = []
    model = None

    try:
        for vi, src_path in enumerate(videos):
            vid_name = Path(src_path).stem
            out_name = _cropped_name(src_path)
            local_out = os.path.join(local_out_dir, out_name)
            drive_out = os.path.join(drive_out_dir, out_name) if drive_out_dir else None

            # ---- skip if already cropped (both copies present if Drive used) ----
            if skip_existing and os.path.exists(local_out) and \
               (drive_out is None or os.path.exists(drive_out)):
                outputs.append(local_out)
                yield {"type": "progress", "phase": "crop", "video": vid_name,
                       "vid_i": vi + 1, "vid_n": n, "frame": 1, "total": 1,
                       "skipped": True}
                continue

            # ---- 1. cache original to local ----
            yield {"type": "progress", "phase": "cache", "video": vid_name,
                   "vid_i": vi + 1, "vid_n": n, "frame": 0, "total": 1}
            local_src = cache_video_to_local(src_path)

            # lazy-load YOLO only when we actually have work to do
            if model is None:
                model = YOLO(model_path)

            # ---- 2. crop the local copy ----
            try:
                vr = VideoReader(local_src, ctx=cpu(0))
            except Exception as e:
                print(f"⚠️ Skipped {vid_name}: {e}")
                continue
            total_frames = len(vr)
            fps = float(vr.get_avg_fps()) or 30.0

            writer = imageio.get_writer(local_out, fps=fps, codec="libx264",
                                        quality=8, macro_block_size=None)
            smoother = SmoothBox()
            try:
                for frame_idx in range(total_frames):
                    frame = vr[frame_idx].asnumpy()
                    results = model.track(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE,
                                          persist=True, tracker="botsort.yaml",
                                          device=device, verbose=False)
                    best_box, best_conf = None, -1
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None and len(boxes) > 0:
                            for i in range(len(boxes)):
                                c = float(boxes.conf[i].cpu())
                                if c > best_conf:
                                    best_conf = c
                                    best_box = boxes.xyxy[i].cpu().numpy().tolist()
                    smooth_box = smoother.update(best_box)
                    if smooth_box is not None:
                        out_frame = crop_and_upscale(frame, smooth_box, crop_padding,
                                                     OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    else:
                        out_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                               interpolation=cv2.INTER_LANCZOS4)
                    writer.append_data(out_frame)
                    if (frame_idx + 1) % 25 == 0:
                        yield {"type": "progress", "phase": "crop", "video": vid_name,
                               "vid_i": vi + 1, "vid_n": n,
                               "frame": frame_idx + 1, "total": total_frames}
            finally:
                writer.close()
                del vr

            # reset tracker state between videos
            try:
                model.predictor.trackers[0].reset()
            except Exception:
                model.predictor = None
            gc.collect()

            # ---- 3. mirror to Drive ----
            if drive_out:
                try:
                    shutil.copy2(local_out, drive_out)
                except Exception as e:
                    print(f"⚠️ Could not mirror {out_name} to Drive: {e}")

            outputs.append(local_out)

            # ---- 4. delete the original local copy (keep the cropped one) ----
            if delete_source_cache and local_src != src_path and os.path.exists(local_src):
                try:
                    os.remove(local_src)
                except Exception:
                    pass

            yield {"type": "progress", "phase": "crop", "video": vid_name,
                   "vid_i": vi + 1, "vid_n": n,
                   "frame": total_frames, "total": total_frames}
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    yield {"type": "done", "output_dir": local_out_dir, "outputs": outputs}
