# backend/utils.py
import os
import numpy as np
import ffmpeg
import cv2
from pathlib import Path

def extract_frames_ffmpeg(video_path, num_frames=16, resize=(224,224), fps=None):
    """
    Try to use ffmpeg to extract frames quickly. Returns numpy array (T, H, W, C) in RGB.
    If ffmpeg fails, falls back to OpenCV.
    """
    video_path = str(video_path)
    try:
        probe = ffmpeg.probe(video_path)
        # duration/nb_frames fallback
        # Use ffmpeg to sample frames evenly by using -vf fps or select=not(mod(n\,k))
        out, _ = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps= (fps if fps else 1))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(resize[0], resize[1]))
            .run(capture_stdout=True, capture_stderr=True)
        )
        # Above may generate entire video frames; fallback to cv2 if mismatch.
    except Exception:
        return extract_frames_opencv(video_path, num_frames=num_frames, resize=resize)
    # Because robust ffmpeg piping is tricky cross-platform, fallback to opencv result:
    return extract_frames_opencv(video_path, num_frames=num_frames, resize=resize)

def extract_frames_opencv(video_path, num_frames=16, resize=(224,224)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # read until end
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        # choose equally spaced frame indices
        idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        frames = []
        i = 0
        next_idx_pos = 0
        success = True
        ret = True
        while ret and next_idx_pos < len(idxs):
            ret, frame = cap.read()
            if not ret:
                break
            if i == idxs[next_idx_pos]:
                if resize:
                    frame = cv2.resize(frame, resize)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                next_idx_pos += 1
            i += 1
        # if fewer frames, pad with last
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames extracted.")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    frames = np.stack(frames[:num_frames], axis=0)  # (T,H,W,C)
    return frames

def save_numpy_frames(frames, dest_path):
    np.save(dest_path, frames)
