"""
precrop.py — YOLO BoT-SORT pre-crop for the inference GUI.

Adapted from the Colab "Script 3" tracker-crop workflow. Detects the target
with a YOLO model, tracks it across frames (BoT-SORT), smooths the crop box
(EMA + speed clamp), then crops + upscales each frame to a fixed size and
writes a new video per source clip.

Only CROP_PADDING and the model path are exposed in the GUI; everything else
uses the same defaults as the original script.
"""

import os, glob, time
from pathlib import Path

import cv2
import numpy as np

# ---- fixed defaults (same as the original Colab script) ----
CONF_THRESHOLD = 0.5
IMG_SIZE       = 640
OUTPUT_WIDTH   = 224
OUTPUT_HEIGHT  = 224
SMOOTH_ALPHA   = 0.08
MAX_SPEED      = 15
OUTPUT_SUBDIR  = "cropped"   # created under the source video folder

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
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        vids.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    return sorted(set(vids))


def crop_folder(model_path, video_dir, crop_padding=0.3, output_dir=None, progress=None):
    """
    Crop every video in ``video_dir`` and write results to ``output_dir``
    (defaults to ``video_dir/cropped``). ``progress`` is an optional callback
    ``progress(video_name, frame_idx, total_frames, vid_i, vid_n)`` for UI updates.

    Returns (output_dir, list_of_output_paths).
    """
    from ultralytics import YOLO  # imported lazily so the GUI loads without it

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

    for vi, vid_path in enumerate(videos):
        vid_name = Path(vid_path).stem
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = os.path.join(output_dir, f"{vid_name}_cropped_{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        smoother = SmoothBox()
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(
                frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE,
                persist=True, tracker="botsort.yaml", verbose=False,
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
            writer.write(out_frame)
            frame_idx += 1
            if progress and frame_idx % 50 == 0:
                progress(vid_name, frame_idx, total_frames, vi + 1, len(videos))

        cap.release()
        writer.release()
        outputs.append(out_path)
        if progress:
            progress(vid_name, frame_idx, total_frames, vi + 1, len(videos))

    return output_dir, outputs
