#!/usr/bin/env python3
"""Long-lived MedGemma image inference for CSDVideoSegment.

The old pipeline created a new vLLM engine for every case.  This module keeps
one processor and one engine alive and accepts a case at a time, which lets
video decoding continue while the GPU analyses the previous case.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


class MedGemmaInferencer:
    """A reusable, offline vLLM runner for one-image MedGemma requests."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tensor_parallel_size: int = 1,
        batch_size: int = 16,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_model_len: int = 1088,
        gpu_memory_utilization: float = 0.75,
        image_min_pixels: int = 784,
        image_max_pixels: int = 50176,
    ) -> None:
        # Imports are intentionally lazy so frame-only utilities can still run
        # in a CPU environment without importing vLLM.
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                min_pixels=image_min_pixels,
                max_pixels=image_max_pixels,
            )
        except TypeError:
            # Older Transformers releases do not expose these processor kwargs.
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )

        self.llm = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            disable_log_stats=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

    def _request(self, instruction: str, image: Image.Image) -> dict:
        prompt = self.processor.apply_chat_template(
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
        return {
            "prompt": prompt,
            # vLLM expects a single PIL image directly for a single-image
            # request.  A list is reserved for multiple image placeholders.
            "multi_modal_data": {"image": image},
        }

    def infer_records(self, records: list[dict], output_jsonl: str | Path) -> None:
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as fout:
            for start in tqdm(
                range(0, len(records), self.batch_size),
                desc=f"MedGemma inference ({output_path.stem})",
            ):
                batch = records[start : start + self.batch_size]
                requests = []
                metas = []
                images = []

                for offset, item in enumerate(batch):
                    image_path = item["images"][0]
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")
                    images.append(image)
                    requests.append(self._request(item["instruction"], image))
                    metas.append(
                        {
                            "index": start + offset,
                            "image": os.path.basename(image_path),
                            "prompt": item["instruction"],
                        }
                    )

                outputs = self.llm.generate(requests, self.sampling_params)
                for meta, output in zip(metas, outputs):
                    prediction = ""
                    if output.outputs:
                        prediction = output.outputs[0].text
                    fout.write(
                        json.dumps(
                            {**meta, "predict": prediction},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        print(f"✅ MedGemma inference done: {output_path}")

    def infer_json(self, input_json: str | Path, output_jsonl: str | Path) -> None:
        with Path(input_json).open("r", encoding="utf-8") as fin:
            records = json.load(fin)
        self.infer_records(records, output_jsonl)
