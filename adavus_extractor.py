"""
extract_landmarks.py - standalone: dance video(s) -> one JSON per video with MediaPipe pose landmarks.

Each JSON holds, per frame:
  landmarks        33 x [x, y, z, visibility]   normalised to the frame. x and y are where the dancer
                                                 stands on screen (null if no pose found)
  world_landmarks  33 x [x, y, z]               metres, origin on the hips, so stage travel is removed
                                                 (null if no pose found)
plus per video: bone angles for the stick figure (sp, lua, lfa, rua, rfa, lt, ls, rt, rs;
0 = down, 180 = up, +90 = screen right; gaps filled and smoothed), foot heights, fps, size, detected_ratio.

Setup:
  uv add mediapipe opencv-python-headless numpy
  curl -L -o pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

The video's file name (without the extension) is used as the adavu name: Thattadavu.mp4 -> adavu "Thattadavu",
output Thattadavu.json.

Usage (the input can be a folder, a single video, or a .txt file listing one video path per line):
  uv run extract_landmarks.py /path/to/adavu_videos -o landmarks_json
  uv run extract_landmarks.py clips/Thattadavu.mp4 -o landmarks_json
  uv run extract_landmarks.py video_list.txt -o landmarks_json
  uv run extract_landmarks.py /path/to/adavu_videos -o landmarks_json --max-seconds 10   # quick trial

Assumes one dancer facing the camera, whole body in frame (their right side then appears on screen left).
Use --no-mirror if the dancer faces away or the video is mirrored.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# MediaPipe landmark indices
LSH, RSH, LEL, REL, LWR, RWR, LHP, RHP, LKN, RKN, LAN, RAN = 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
MP_LEFT = dict(ua=(LSH, LEL), fa=(LEL, LWR), t=(LHP, LKN), s=(LKN, LAN))
MP_RIGHT = dict(ua=(RSH, REL), fa=(REL, RWR), t=(RHP, RKN), s=(RKN, RAN))
BONES = ["sp", "lua", "lfa", "rua", "rfa", "lt", "ls", "rt", "rs"]


def bone_angle(p, c):
    """Bone p->c in image coords (x right, y down): 0 = down, +90 = right."""
    return float(np.degrees(np.arctan2(c[0] - p[0], c[1] - p[1])))


def frame_angles(lm, w, h, mirror, min_vis):
    pts = np.array([[l.x * w, l.y * h] for l in lm])      # scale x and y separately, otherwise angles are skewed
    vis = np.array([l.visibility or 0.0 for l in lm])
    left, right = (MP_RIGHT, MP_LEFT) if mirror else (MP_LEFT, MP_RIGHT)   # screen-left / screen-right bodies
    out = {}
    for side, pre in ((left, "l"), (right, "r")):
        for key in ("ua", "fa", "t", "s"):
            a, b = side[key]
            out[pre + key] = bone_angle(pts[a], pts[b]) if min(vis[a], vis[b]) >= min_vis else np.nan
    hip, sh = pts[[LHP, RHP]].mean(axis=0), pts[[LSH, RSH]].mean(axis=0)
    ok = min(vis[[LSH, RSH, LHP, RHP]]) >= min_vis
    dx, dy = sh - hip
    out["sp"] = float(180 - np.degrees(np.arctan2(dx, -dy))) if ok else np.nan   # 180 = upright
    torso = float(np.linalg.norm(sh - hip)) or 1.0
    la, ra = (RAN, LAN) if mirror else (LAN, RAN)
    feet = {"foot_l": (pts[la][1] - hip[1]) / torso, "foot_r": (pts[ra][1] - hip[1]) / torso}  # bigger = lower
    return out, feet


def clean(series, smooth, angle=True):
    """Fill gaps, unwrap 360-degree jumps (angles only), moving-average smooth."""
    x = np.asarray(series, float)
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    idx = np.arange(len(x))
    x = np.interp(idx, idx[good], x[good])
    if angle:
        x = np.degrees(np.unwrap(np.radians(x)))
        x -= 360 * np.round(np.median(x) / 360)
    if smooth > 1 and len(x) > smooth:
        pad = smooth // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        x = np.convolve(xp, np.ones(smooth) / smooth, mode="valid")[:len(x)]
    return x


def open_landmarker(model):
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    return vision.PoseLandmarker.create_from_options(opts)


def process_video(path, model, mirror=True, min_vis=0.5, smooth=5, max_seconds=0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lms, worlds, angs, feet_rows, times = [], [], [], [], []
    nan_a, nan_f = {b: np.nan for b in BONES}, {"foot_l": np.nan, "foot_r": np.nan}
    i = 0
    with open_landmarker(model) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / fps
            if max_seconds and t > max_seconds:
                break
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(t * 1000))
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                lms.append([[round(l.x, 4), round(l.y, 4), round(l.z, 4), round(l.visibility or 0.0, 3)] for l in lm])
                wl = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                worlds.append([[round(l.x, 4), round(l.y, 4), round(l.z, 4)] for l in wl] if wl else None)
                a, f = frame_angles(lm, w, h, mirror, min_vis)
            else:
                lms.append(None)
                worlds.append(None)
                a, f = nan_a, nan_f
            angs.append(a)
            feet_rows.append(f)
            times.append(round(t, 4))
            i += 1
    cap.release()
    if not times:
        raise RuntimeError("no frames read")
    angles = {b: clean([r[b] for r in angs], smooth) for b in BONES}
    feet = {k: clean([r[k] for r in feet_rows], smooth, angle=False) for k in ("foot_l", "foot_r")}
    return {
        "video": Path(path).name, "adavu": Path(path).stem, "label": Path(path).stem,
        "fps": fps, "width": w, "height": h, "n_frames": len(times),
        "detected_ratio": round(sum(r is not None for r in lms) / len(lms), 3), "mirrored": mirror,
        "t": times, "landmarks": lms, "world_landmarks": worlds,
        "angles": {b: [round(float(v), 1) for v in angles[b]] for b in BONES},
        "foot_height_norm": {k: [round(float(v), 3) for v in feet[k]] for k in feet},
    }


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def find_videos(src):
    """Folder (searched recursively), single video, or a .txt file with one video path per line."""
    src = Path(src).expanduser()
    if src.is_dir():
        return sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if src.suffix.lower() == ".txt":
        lines = [l.strip().strip('"') for l in src.read_text().splitlines() if l.strip()]
        return [Path(l).expanduser() if Path(l).expanduser().is_absolute() else src.parent / l for l in lines]
    return [src] if src.is_file() else []


def is_lfs_pointer(p):
    """A Git LFS pointer is a tiny text file standing in for the real video."""
    return p.stat().st_size < 2048 and p.read_bytes()[:40].startswith(b"version https://git-lfs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of videos, a single video, or a .txt listing video paths")
    ap.add_argument("-o", "--out-dir", default="landmarks_json")
    ap.add_argument("--model", default="pose_landmarker_full.task")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--min-vis", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=5, help="moving-average window in frames (1 = off)")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    videos = find_videos(a.input)
    if not videos:
        sys.exit(f"no videos found at {a.input}")
    if not Path(a.model).exists():
        sys.exit(f"model file {a.model} not found: download it with the curl command in the script header")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n, v in enumerate(videos, 1):
        out = out_dir / (v.stem + ".json")
        tag = f"[{n}/{len(videos)}] {v.name}"
        if is_lfs_pointer(v):
            print(f"{tag}: SKIPPED, Git LFS pointer, not a real video", file=sys.stderr)
            continue
        if out.exists() and not a.overwrite:
            print(f"{tag}: exists, skipping (use --overwrite)", file=sys.stderr)
            continue
        try:
            data = process_video(v, a.model, not a.no_mirror, a.min_vis, a.smooth, a.max_seconds)
        except Exception as e:
            print(f"{tag}: FAILED, {type(e).__name__}: {e}", file=sys.stderr)
            continue
        out.write_text(json.dumps(data, separators=(",", ":")))
        print(f"{tag}: {data['n_frames']} frames, pose in {data['detected_ratio']:.0%} -> {out}", file=sys.stderr)
        if data["detected_ratio"] < 0.8:
            print("    WARNING: pose lost in many frames: check framing/lighting or use the heavy model", file=sys.stderr)


if __name__ == "__main__":
    main()
