# syntax=docker/dockerfile:1
# vLLM's CUDA image already contains a CUDA-compatible PyTorch + vLLM stack.
# Override VLLM_IMAGE if the deployment cluster has a pinned internal image.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.15.1
FROM ${VLLM_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/CSDVideoSegment

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/CSDVideoSegment
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .

# Models, videos, frames, and results are deliberately not copied into the
# image.  Mount them at runtime so checkpoints never move between network
# zones and the image remains small/reproducible.
ENTRYPOINT ["python3", "scripts/run_video_pipeline_parallel.py"]
