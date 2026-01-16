#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def load_question_schema(path: str) -> list[dict]:
    """
    Load question schema JSON.

    Expected format:
    [
      { "question": "...", "field": "phase" },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(
    input_list,
    output_json,
    question_schema,
):
    # --------------------------------------------------
    # Load question schema (canonical source)
    # --------------------------------------------------
    schema = load_question_schema(question_schema)

    # --------------------------------------------------
    # Load image list
    # --------------------------------------------------
    with open(input_list, "r", encoding="utf-8") as f:
        images = [l.strip() for l in f if l.strip()]

    data = []

    for img in tqdm(images, desc="Building flat QA"):
        img = os.path.abspath(img)

        for item in schema:
            question_text = item["question"]

            data.append(
                {
                    "instruction": question_text,
                    "input": "<image>",
                    "output": "",      # inference 時填
                    "images": [img],
                }
            )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved flat QA JSON: {output_json} ({len(data)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--question-schema", required=True)
    args = parser.parse_args()

    main(
        input_list=args.input_list,
        output_json=args.output_json,
        question_schema=args.question_schema,
    )
