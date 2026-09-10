#!/usr/bin/env python3
"""Backward-compatible entry point for one-shot VLM inference.

The implementation now uses the same MedGemma/vLLM request format as the
streaming video pipeline.  The engine still lives for the duration of this
command; callers that process multiple cases should use the streaming runner
so the engine is not recreated for every case.
"""

from __future__ import annotations

from utils.medgemma_inference import MedGemmaInferencer


def main(
    model_name_or_path: str,
    input_json: str,
    output_jsonl: str,
    lora_path: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 32,
    tensor_parallel_size: int = 1,
    batch_size: int = 16,
    max_model_len: int = 1088,
    gpu_memory_utilization: float = 0.75,
):
    if lora_path:
        raise ValueError("MedGemma CSD inference expects a merged checkpoint; LoRA is not supported here")
    inferencer = MedGemmaInferencer(
        model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    inferencer.infer_json(input_json, output_jsonl)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
