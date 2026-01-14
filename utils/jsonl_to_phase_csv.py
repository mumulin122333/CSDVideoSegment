#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any

import pandas as pd

# =========================
# Utils
# =========================

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def clean_predict(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = THINK_RE.sub("", text)
    return text.strip()

def majority(vals):
    vals = [v for v in vals if v not in ("", None)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]

# =========================
# Filename parsing
# =========================

def parse_filename(fname: str):
    """
    Expected filename example:
    03042025105638_00000000U0429638__MOVIE__M_03042025105638_00000000U0429638_1_001-1__frame_000050.png

    Returns:
        case_id
        video_id
        frame_no
    """
    base = Path(fname).name

    # frame number
    m = re.search(r"frame_(\d+)", base)
    frame_no = int(m.group(1)) if m else None

    # case_id = 前綴第一段
    case_id = base.split("__")[0]

    # video_id = MOVIE 之後那一段
    m = re.search(r"__MOVIE__(.+?)__frame_", base)
    video_id = m.group(1) if m else case_id

    return case_id, video_id, frame_no

def frame_to_sec(frame_no: int, fps: float) -> float:
    return (frame_no - 1) / fps

def hhmmss(sec: float) -> str:
    s = int(max(0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# =========================
# Step 1: JSONL → frame table
# =========================

def jsonl_to_frame_df(jsonl_path: Path) -> pd.DataFrame:
    rows: Dict[str, Dict[str, Any]] = defaultdict(dict)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            fname = obj.get("image")
            if not fname:
                continue

            prompt = obj.get("prompt", "").lower()
            pred = clean_predict(obj.get("predict", ""))

            if "procedural phase" in prompt:
                rows[fname]["phase"] = pred
            elif "how many instruments" in prompt:
                rows[fname]["num_instruments"] = pred
            elif "how many polyps" in prompt:
                rows[fname]["num_polyps"] = pred
            elif "what color" in prompt:
                rows[fname]["color"] = pred
            # elif "where is" in prompt:
            #     rows[fname]["location"] = pred
            elif "text overlay" in prompt or "non-clinical text" in prompt:
                rows[fname]["overlay_text"] = pred
            elif "appearance of the instrument sheath" in prompt:
                rows[fname]["instrument_sheath"] = pred.lower()


    data = []
    for fname, attrs in rows.items():
        case_id, video_id, frame_no = parse_filename(fname)
        if frame_no is None or "phase" not in attrs:
            continue

        data.append({
            "filename": fname,
            "case_id": case_id,
            "video_id": video_id,
            "frame_no": frame_no,
            **attrs
        })

    df = pd.DataFrame(data)
    df = df.sort_values(["case_id", "video_id", "frame_no"])
    df["frame_no"] = df["frame_no"].astype(int)
    return df

# =========================
# Step 2: phase segmentation (ONLY phase matters)
# =========================

def build_phase_segments(df: pd.DataFrame):
    segments = []

    for (case_id, video_id), g in df.groupby(["case_id", "video_id"]):
        g = g.sort_values("frame_no")

        cur = None
        for _, r in g.iterrows():
            f = r["frame_no"]
            p = r["phase"]

            if cur is None:
                cur = {
                    "case_id": case_id,
                    "video_id": video_id,
                    "phase": p,
                    "start_frame": f,
                    "end_frame": f,
                    "frames": [r],
                }
                continue

            # ⭐ 核心邏輯：只看 phase
            if p == cur["phase"]:
                cur["end_frame"] = f
                cur["frames"].append(r)
            else:
                segments.append(cur)
                cur = {
                    "case_id": case_id,
                    "video_id": video_id,
                    "phase": p,
                    "start_frame": f,
                    "end_frame": f,
                    "frames": [r],
                }

        if cur:
            segments.append(cur)

    return segments

# =========================
# Step 3: aggregate segments
# =========================

def aggregate_segments(segments, fps, min_duration):
    out = []

    for seg in segments:
        start_sec = frame_to_sec(seg["start_frame"], fps)
        end_sec   = frame_to_sec(seg["end_frame"], fps)
        duration  = end_sec - start_sec

        if duration < min_duration:
            continue

        frames = seg["frames"]

        # === 基本欄位（一定有）===
        row = {
            "case_id": seg["case_id"],
            "video_id": seg["video_id"],
            "phase": seg["phase"],
            "phase_start_time": hhmmss(start_sec),
            "phase_end_time": hhmmss(end_sec),
            "start_frame": seg["start_frame"],
            "end_frame": seg["end_frame"],
            "duration_s": round(duration, 2),
        }

        # === 動態欄位：只處理實際出現過的 ===
        # 收集這個 segment 出現過的所有 key
        appeared_keys = set()
        for f in frames:
            appeared_keys.update(f.keys())

        # 不要處理這些固定欄位
        skip_keys = {
            "filename", "case_id", "video_id", "frame_no", "phase"
        }

        dynamic_keys = appeared_keys - skip_keys

        for k in dynamic_keys:
            vals = [f.get(k) for f in frames if k in f]
            v = majority(vals)
            if v not in ("", None):
                row[k] = v

        # =========================
        # Deterministic Hemostasis rule
        # yellow + instrument + RDI overlay
        # =========================

        # normalize color
        color = row.get("color", "")
        color = color.lower().strip()

        # normalize num_instruments
        num_instr = row.get("num_instruments", 0)
        try:
            num_instr = int(num_instr)
        except (TypeError, ValueError):
            num_instr = 0

        # normalize overlay text
        overlay = row.get("overlay_text", "")
        overlay = str(overlay).upper().strip()

        is_rdi = overlay.startswith("RDI")

        # strict AND rule
        if (
            row["phase"] != "Diagnosis"
            and color == "yellow"
            and num_instr >= 1
            and is_rdi
        ):
            row["phase"] = "Hemostasis"

        # =========================
        # Deterministic Local injection rule
        # overlay none + instrument sheath present (not knife-type)
        # =========================

        overlay = row.get("overlay_text", "")
        overlay = str(overlay).lower().strip()

        sheath = row.get("instrument_sheath", "")
        sheath = str(sheath).lower().strip()

        if (
            row["phase"] not in ("Diagnosis", "Hemostasis")
            and overlay in ("none", "")
            and sheath not in (
                "no instrument visible",
                "white with blue markings",
                ""
            )
        ):
            row["phase"] = "Local injection"


        out.append(row)

    return pd.DataFrame(out)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--min-duration", type=float, default=10.0)
    args = ap.parse_args()

    df_frames = jsonl_to_frame_df(Path(args.jsonl))
    if df_frames.empty:
        print("❌ No valid frames parsed.")
        return

    segments = build_phase_segments(df_frames)
    out = aggregate_segments(segments, args.fps, args.min_duration)

    if out.empty:
        print("⚠️ No phase segments passed min_duration.")
        return

    out.sort_values(["case_id", "video_id", "start_frame"], inplace=True)
    out["segment_idx"] = out.groupby("case_id").cumcount() + 1

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"✅ Saved {len(out)} phase segments → {args.out_csv}")

if __name__ == "__main__":
    main()
