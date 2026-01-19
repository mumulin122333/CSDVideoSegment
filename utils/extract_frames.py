#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import sys

# ============================================================
# Environment
# ============================================================

os.environ.pop("LD_LIBRARY_PATH", None)
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
    print("❌ ffmpeg not found in PATH. Please install ffmpeg.", file=sys.stderr)
    sys.exit(1)

subprocess.run([ffmpeg, "-version"], stdout=subprocess.DEVNULL)


# ============================================================
# Defaults (repo-relative)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

VIDEO_ROOT = REPO_ROOT / "data" / "videos"
FRAME_ROOT = REPO_ROOT / "data" / "frames"

FPS = 1
LEFT_CROP_RATIO = 0

USE_HWACCEL = os.environ.get("HWACCEL", "").lower() == "cuda"
NUM_WORKERS = max(1, int(os.environ.get("NUM_WORKERS", min(16, cpu_count()))))

VIDEO_EXTS = {".mp4", ".mov", ".mkv"}

FRAME_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract frames from data/videos/<case_id>/*.mp4"
    )
    ap.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case IDs. Empty = all cases under data/videos/",
    )
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--left-crop", type=float, default=LEFT_CROP_RATIO)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)

    ap.add_argument(
        "--overwrite",
        choices=["auto", "force", "skip"],
        default="auto",
        help="auto=resume if frames < min; force=redo; skip=skip if any output",
    )
    ap.add_argument("--min-complete-frames", type=int, default=5)

    return ap.parse_args()

# ============================================================
# Utils
# ============================================================

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)[:180]

def list_videos(video_root: Path, target_cases: set[str]):
    cases = (
        [video_root / c for c in target_cases]
        if target_cases
        else [d for d in video_root.iterdir() if d.is_dir()]
    )

    videos = []
    for case_dir in cases:
        if not case_dir.is_dir():
            continue
        for p in case_dir.iterdir():
            if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"):
                videos.append(p)

    return videos

def build_prefix(video: Path) -> str:
    return sanitize(video.stem)

def count_existing_frames(out_dir: Path, prefix: str) -> int:
    import glob
    return len(glob.glob(str(out_dir / f"{prefix}__frame_*.png")))

def already_done(out_dir: Path, prefix: str, mode: str, min_frames: int) -> bool:
    if mode == "force":
        return False
    n = count_existing_frames(out_dir, prefix)
    if mode == "skip":
        return n > 0
    return n >= min_frames

def build_ffmpeg_cmd(video: Path, out_dir: Path, prefix: str,
                     fps: float, left_crop_ratio: float) -> list[str]:

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-y", "-threads", "1",
        "-noautorotate",
    ]

    if USE_HWACCEL:
        cmd += ["-hwaccel", "cuda"]

    cmd += [
        "-ignore_editlist", "1",
        "-i", str(video),
        "-map", "0:v:0",
        "-an",
    ]

    vf = (
        f"setsar=1,"
        f"fps={fps},"
        f"crop=iw*{1-left_crop_ratio}:ih:iw*{left_crop_ratio}:0"
    )

    cmd += [
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24",
        str(out_dir / f"{prefix}__frame_%06d.png"),
    ]

    return cmd

def extract_one(args):
    video, fps, left_crop, overwrite, min_frames = args
    try:
        case_id = video.parent.name
        out_dir = FRAME_ROOT / sanitize(case_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = build_prefix(video)

        if already_done(out_dir, prefix, overwrite, min_frames):
            return (video, True, "skip")

        cmd = build_ffmpeg_cmd(video, out_dir, prefix, fps, left_crop)
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if p.returncode == 0:
            return (video, True, "ok")
        return (video, False, p.stderr.decode("utf-8", errors="ignore")[:300])

    except Exception as e:
        return (video, False, str(e)[:300])

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    target_cases = {s.strip() for s in args.cases.split(",") if s.strip()}

    videos = list_videos(VIDEO_ROOT, target_cases)
    if not videos:
        print("❌ No videos found")
        return

    print(f"📺 Videos: {len(videos)}")
    print(f"📂 Output: {FRAME_ROOT}")
    print(f"🧵 workers={args.workers}, fps={args.fps}, crop-left={args.left_crop*100:.0f}%")
    print("⚡ CUDA" if USE_HWACCEL else "🧠 CPU")

    work_items = [
        (v, args.fps, args.left_crop, args.overwrite, args.min_complete_frames)
        for v in videos
    ]

    ok = fail = 0
    with Pool(processes=args.workers) as pool:
        for video, success, msg in tqdm(
            pool.imap_unordered(extract_one, work_items),
            total=len(work_items),
            desc="Extracting frames",
        ):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"\n[FAIL] {video}\n{msg}\n")

    print(f"\n✅ Done: ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()
