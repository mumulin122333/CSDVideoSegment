#!/usr/bin/env python3
"""Streaming video pipeline for the MedGemma phase classifier.

Frame extraction runs in a bounded producer pool.  Completed cases are pushed
to a queue while the single GPU analysis worker performs clarity filtering,
need voting, and MedGemma phase inference.  This removes the old
"extract-everything, then analyse" barrier.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PHASE_MODEL = (
    "/mnt/t24c/mumulin/LlamaFactory/saves/campaign_nested/"
    "medgemma-1.5-4b/fold0"
)
DEFAULT_CLARITY_REPO = (
    "Mumulin/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k-NTUH"
)
DEFAULT_NEED_REPOS = [
    "Mumulin/beit3_large_patch16_224.in22k_ft_in1k-NTUH",
    "Mumulin/beit3_base_patch16_224.indomain_pt-NTUH",
    "Mumulin/swinv2_large_window12_192-NTUH",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("CSD_MEDGEMMA_MODEL", DEFAULT_PHASE_MODEL),
        help="MedGemma phase checkpoint or HF model ID.",
    )
    parser.add_argument("--model-tag", default="", help="Result directory name.")
    parser.add_argument("--cases-file", type=Path, default=REPO_ROOT / "configs/cases.txt")
    parser.add_argument("--video-root", type=Path, default=REPO_ROOT / "data/videos")
    parser.add_argument("--frame-root", type=Path, default=REPO_ROOT / "data/frames")
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / ".work")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1088)
    parser.add_argument("--vlm-memory-utilization", type=float, default=0.75)
    parser.add_argument("--hwaccel", choices=("none", "cuda"), default=os.environ.get("HWACCEL", "none"))
    parser.add_argument("--overwrite", choices=("auto", "force", "skip"), default="auto")
    parser.add_argument("--min-complete-frames", type=int, default=5)
    parser.add_argument("--clarity-repo", default=DEFAULT_CLARITY_REPO)
    parser.add_argument("--need-repo", action="append", dest="need_repos")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def model_tag(model: str, requested: str) -> str:
    if requested:
        return requested
    name = Path(model.rstrip("/")).name
    if name.startswith("fold") and Path(model.rstrip("/")).parent.name:
        return f"{Path(model.rstrip('/')).parent.name}-{name}"
    return name or "medgemma-1.5-4b"


def run_command(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_frame_list(frame_dir: Path, output: Path) -> int:
    paths = sorted(
        p
        for p in frame_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(str(path.resolve()) + "\n")
    return len(paths)


def clarity_list(clarity_csv: Path, output: Path) -> int:
    count = 0
    with clarity_csv.open("r", newline="", encoding="utf-8") as handle, output.open(
        "w", encoding="utf-8"
    ) as out:
        for row in csv.DictReader(handle):
            if str(row.get("top1_label", "")).strip() == "1":
                filename = row.get("filename", "").strip()
                if filename:
                    out.write(filename + "\n")
                    count += 1
    return count


def run_frame_filters(
    case_id: str,
    args: argparse.Namespace,
    need_repos: list[str],
) -> tuple[Path, Path]:
    """Run the existing timm filters and return (need-list, work-dir)."""
    from utils import extract_frames

    safe_case = extract_frames.sanitize(case_id)
    input_dir = args.frame_root / safe_case
    work_dir = args.work_root / safe_case
    need_csv_dir = work_dir / "run_csv_need"
    work_dir.mkdir(parents=True, exist_ok=True)
    need_csv_dir.mkdir(parents=True, exist_ok=True)

    all_frames = work_dir / "all_frames.txt"
    clear_list = work_dir / "clarity_filtered.txt"
    clarity_csv = work_dir / "clarity.csv"
    need_fused = work_dir / "need_vote.csv"
    need_list = work_dir / "need_list.txt"
    n_frames = write_frame_list(input_dir, all_frames)
    if n_frames == 0:
        raise RuntimeError(f"No extracted frames found for {case_id}: {input_dir}")

    common = [
        sys.executable,
        "-m",
        "utils.run_frame_classifier",
        "--data-dir",
        str(input_dir),
        "--dataset",
        "ImageFolder",
        "--batch-size",
        "128",
        "--workers",
        "8",
        "--topk",
        "1",
        "--amp",
        "--fullname",
    ]
    run_command(
        common
        + [
            "--input-list",
            str(all_frames),
            "--model",
            args.clarity_repo,
            "--results-dir",
            str(work_dir),
            "--results-file",
            clarity_csv.name,
        ]
    )
    n_clear = clarity_list(clarity_csv, clear_list)
    if n_clear == 0:
        need_list.write_text("", encoding="utf-8")
        return need_list, work_dir

    need_runs: list[Path] = []
    for repo in need_repos:
        out_csv = need_csv_dir / f"{Path(repo).name}.csv"
        run_command(
            common
            + [
                "--input-list",
                str(clear_list),
                "--model",
                repo,
                "--results-dir",
                str(need_csv_dir),
                "--results-file",
                out_csv.name,
            ]
        )
        need_runs.append(out_csv)

    run_command(
        [
            sys.executable,
            "-m",
            "utils.fuse_frame_votes",
            str(need_fused),
            "0.9",
            "2",
            "1",
            *map(str, need_runs),
        ]
    )
    import pandas as pd

    fused = pd.read_csv(need_fused)
    if "final_label" not in fused or "filename" not in fused:
        raise RuntimeError(f"Invalid fused output: {need_fused}")
    selected = fused[fused["final_label"].astype(str).str.strip() == "0"]["filename"]
    selected.to_csv(need_list, index=False, header=False)
    return need_list, work_dir


def build_case_qa(case_id: str, need_list: Path, work_dir: Path) -> Path:
    from utils import build_vlm_dataset

    qa_json = work_dir / "phase_query.json"
    build_vlm_dataset.main(
        input_list=str(need_list),
        output_json=str(qa_json),
        question_schema=str(REPO_ROOT / "configs/video_phase_questions.json"),
    )
    return qa_json


def analyse_case(case_id: str, args: argparse.Namespace, inferencer, need_repos: list[str]):
    need_list, work_dir = run_frame_filters(case_id, args, need_repos)
    qa_json = build_case_qa(case_id, need_list, work_dir)
    with qa_json.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        print(f"⚠️ {case_id}: no frames survived filtering; no VLM request")
        return inferencer

    if inferencer is None:
        from utils.medgemma_inference import MedGemmaInferencer

        print(f"🚀 Loading MedGemma once: {args.model}", flush=True)
        inferencer = MedGemmaInferencer(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.vlm_memory_utilization,
        )

    jsonl = work_dir / f"{case_id}.jsonl"
    inferencer.infer_records(records, jsonl)
    output_csv = args.results_root / model_tag(args.model, args.model_tag) / f"{case_id}_phase_query.csv"
    run_command(
        [
            sys.executable,
            "-m",
            "utils.video_jsonl_to_phase_csv",
            "--jsonl",
            str(jsonl),
            "--out-csv",
            str(output_csv),
            "--fps",
            str(args.fps),
            "--min-duration",
            str(args.min_duration),
        ]
    )
    print(f"✅ Final: {output_csv}", flush=True)
    return inferencer


def extraction_worker(cases: list[str], args: argparse.Namespace, completed: queue.Queue) -> None:
    from utils import extract_frames

    extract_frames.VIDEO_ROOT = args.video_root
    extract_frames.FRAME_ROOT = args.frame_root
    extract_frames.USE_HWACCEL = args.hwaccel == "cuda"
    args.frame_root.mkdir(parents=True, exist_ok=True)

    remaining: dict[str, int] = {}
    case_success: dict[str, bool] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.extract_workers)) as pool:
        for case_id in cases:
            videos = extract_frames.list_videos(args.video_root, {case_id})
            remaining[case_id] = len(videos)
            case_success[case_id] = True
            if not videos:
                print(f"❌ {case_id}: no videos found under {args.video_root}", flush=True)
                completed.put((case_id, False))
                continue
            for video in videos:
                future = pool.submit(
                    extract_frames.extract_one,
                    (video, args.fps, 0.0, args.overwrite, args.min_complete_frames),
                )
                futures[future] = case_id

        for future in as_completed(futures):
            video, success, message = future.result()
            case_id = futures[future]
            if not success:
                case_success[case_id] = False
                print(f"❌ frame extraction failed: {video}: {message}", flush=True)
            remaining[case_id] -= 1
            if remaining[case_id] == 0:
                completed.put((case_id, case_success[case_id]))
    completed.put((None, True))


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases_file)
    if args.need_repos is None:
        need_repos = DEFAULT_NEED_REPOS
    else:
        need_repos = args.need_repos
    result_dir = args.results_root / model_tag(args.model, args.model_tag)
    remaining = []
    for case_id in cases:
        final_csv = result_dir / f"{case_id}_phase_query.csv"
        if final_csv.exists() and args.overwrite != "force":
            print(f"⏭️ skip {case_id}: {final_csv}")
        else:
            remaining.append(case_id)

    print(f"📄 cases={len(cases)}, remaining={len(remaining)}, model={args.model}")
    if args.dry_run:
        print(f"📦 extraction workers={args.extract_workers}, queue={args.queue_size}, hwaccel={args.hwaccel}")
        print(f"📁 results={result_dir}")
        return
    if not remaining:
        return

    completed: queue.Queue = queue.Queue(maxsize=max(1, args.queue_size))
    producer = threading.Thread(
        target=extraction_worker,
        args=(remaining, args, completed),
        name="frame-producer",
        daemon=True,
    )
    producer.start()

    inferencer = None
    while True:
        case_id, success = completed.get()
        if case_id is None:
            break
        if not success:
            print(f"⚠️ skip analysis for {case_id} because extraction failed", flush=True)
            continue
        print(f"▶️ analysing {case_id} (extraction continues in background)", flush=True)
        inferencer = analyse_case(case_id, args, inferencer, need_repos)

    producer.join()
    print("🎉 Streaming video pipeline finished", flush=True)


if __name__ == "__main__":
    main()
