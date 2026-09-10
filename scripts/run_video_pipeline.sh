#!/usr/bin/env bash
set -euo pipefail

# Streaming producer/consumer pipeline.  All options are forwarded, so this
# remains compatible with the old entry point while using MedGemma by default.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
exec python3 scripts/run_video_pipeline_parallel.py "$@"
