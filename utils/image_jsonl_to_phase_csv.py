#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
from utils.prompt_schema import load_prompt_schema, extract_attrs_from_prompt

import pandas as pd

# =========================
# Utils
# =========================

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def clean_predict(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = THINK_RE.sub("", text)
    return text.strip()

# =========================
# Filename parsing
# =========================

def parse_filename(fname: str):
    """
    Expected filename example:
    03042025105638_00000000U0429638__MOVIE__M_03042025105638_00000000U0429638_1_001-1__frame_000050.png

    For image processing, we expect:
    03042025105638_00000000U0429638__MOVIE__M_03042025105638_00000000U0429638_1_001-1.png

    Returns:
        case_id
        video_id
    """
    base = Path(fname).name

    # case_id = 前綴第一段
    case_id = base.split("__")[0]

    # video_id = MOVIE 之後那一段
    m = re.search(r"__MOVIE__(.+?)\.png", base)
    video_id = m.group(1) if m else case_id

    return case_id, video_id

# =========================
# Main processing
# =========================

def jsonl_to_image_df(jsonl_path: Path, schema) -> pd.DataFrame:
    rows = defaultdict(dict)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            fname = obj.get("image")
            if not fname:
                continue

            prompt = obj.get("prompt", "")
            pred = clean_predict(obj.get("predict", ""))

            attrs = extract_attrs_from_prompt(
                prompt=prompt,
                prediction=pred,
                schema=schema,
            )
            if not attrs:
                continue  # 沒問到 → 忽略

            rows[fname].update(attrs)

    data = []
    for fname, attrs in rows.items():
        case_id, video_id = parse_filename(fname)

        data.append({
            "filename": fname,
            "case_id": case_id,
            "video_id": video_id,
            **attrs
        })

    return pd.DataFrame(data)

# =========================
# Phase segmentation (for images, no aggregation needed)
# =========================

def build_image_segments(df: pd.DataFrame):
    segments = []

    for (case_id, video_id), g in df.groupby(["case_id", "video_id"]):
        g = g.sort_values("filename")

        for _, r in g.iterrows():
            # Each image becomes a single segment
            segments.append({
                "case_id": case_id,
                "video_id": video_id,
                "phase": r["phase"],
                "filename": r["filename"],
                "frames": [r]
            })

    return segments

# =========================
# Aggregate segments (for images, just output each image's data)
# =========================

def aggregate_images(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for _, row in df.iterrows():
        row = row.to_dict()

        phase = row.get("phase")
        # =========================
        # Deterministic Hemostasis rule
        # =========================
        color = str(row.get("color", "")).lower().strip()

        try:
            num_instr = int(row.get("num_instruments", 0))
        except (TypeError, ValueError):
            num_instr = 0

        overlay = str(row.get("overlay_text", "")).upper().strip()
        is_rdi = overlay.startswith("RDI")

        if phase is not None:
            color = str(row.get("color", "")).lower().strip()

            try:
                num_instr = int(row.get("num_instruments", 0))
            except (TypeError, ValueError):
                num_instr = 0

            overlay = str(row.get("overlay_text", "")).upper().strip()
            is_rdi = overlay.startswith("RDI")

            if (
                phase != "Diagnosis"
                and color == "yellow"
                and num_instr >= 1
                and is_rdi
            ):
                row["phase"] = "Hemostasis"


        # =========================
        # Deterministic Local injection rule
        # =========================
        overlay = str(row.get("overlay_text", "")).lower().strip()
        sheath  = str(row.get("instrument_sheath", "")).lower().strip()

        phase = row.get("phase")

        if phase is not None:
            overlay = str(row.get("overlay_text", "")).lower().strip()
            sheath  = str(row.get("instrument_sheath", "")).lower().strip()

            if (
                phase not in ("Diagnosis", "Hemostasis")
                and overlay in ("", "none")
                and sheath not in (
                    "",
                    "no instrument visible",
                    "white with blue markings"
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
    args = ap.parse_args()

    schema = load_prompt_schema(
        Path("/mnt/1248/open2/mumulin/endoscopy_dataset/CSDVideoSegment/configs/image_phase_questions.json")
    )

    df = jsonl_to_image_df(Path(args.jsonl), schema)
    if df.empty:
        print("❌ No valid images parsed.")
        return

    out = aggregate_images(df)

    out.sort_values(["case_id", "video_id", "filename"], inplace=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"✅ Saved {len(out)} image rows → {args.out_csv}")



if __name__ == "__main__":
    main()
