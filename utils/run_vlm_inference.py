#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor


def main(
    model_name_or_path,
    input_json,
    output_jsonl,
    lora_path=None,
    temperature=0.95,
    top_p=0.7,
    max_new_tokens=512,
    tensor_parallel_size=1,
    batch_size=16,
):
    """
    Standalone vLLM VLM batch inference script.

    - No LLaMA-Factory dependency
    - Proper batching (vLLM-friendly)
    - One tqdm only
    """

    # ------------------------------------------------------------
    # Load processor (critical for Qwen3-VL / InternVL / GLM-4V)
    # ------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------
    # Init vLLM engine
    # ------------------------------------------------------------
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=15360,
        gpu_memory_utilization=0.75,
        limit_mm_per_prompt={"image": 4},
        enable_lora=lora_path is not None,
        disable_log_stats=True,  # 關掉 throughput 統計（但 batching 才是關鍵）
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    lora_request = (
        LoRARequest("default", 1, lora_path) if lora_path is not None else None
    )

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # ------------------------------------------------------------
    # Batched inference loop
    # ------------------------------------------------------------
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for start in tqdm(
            range(0, len(data), batch_size),
            desc="vLLM VLM inference",
        ):
            batch = data[start : start + batch_size]

            requests = []
            metas = []

            # ---------- build requests ----------
            for idx, item in enumerate(batch):
                instruction = item["instruction"]
                image_path = item["images"][0]

                image = Image.open(image_path).convert("RGB")

                prompt = processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                requests.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": [image]},
                    }
                )

                metas.append(
                    {
                        "global_index": start + idx,
                        "image": os.path.basename(image_path),
                        "prompt": instruction,
                    }
                )

            # ---------- vLLM generate (ONE call per batch) ----------
            outputs = llm.generate(
                requests,
                sampling_params,
                lora_request=lora_request,
            )

            # ---------- write results ----------
            for meta, out in zip(metas, outputs):
                pred = out.outputs[0].text

                fout.write(
                    json.dumps(
                        {
                            "index": meta["global_index"],
                            "image": meta["image"],
                            "prompt": meta["prompt"],
                            "predict": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"✅ Inference done: {output_jsonl}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
