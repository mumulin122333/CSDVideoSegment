#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm

from utils.load_questions import load_questions


def main(
    input_list,
    output_json,
    question_file,
):
    # --------------------------------------------------
    # Load questions
    # --------------------------------------------------
    questions = load_questions(question_file)

    # --------------------------------------------------
    # Load image list
    # --------------------------------------------------
    with open(input_list, "r", encoding="utf-8") as f:
        images = [l.strip() for l in f if l.strip()]

    data = []

    for img in tqdm(images, desc="Building QA"):
        img = os.path.abspath(img)
        for q in questions:
            data.append(
                {
                    "instruction": q,
                    "input": "<image>",
                    "output": "",
                    "images": [img],
                }
            )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved QA JSON: {output_json} ({len(data)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--question-file", required=True)
    args = parser.parse_args()

    main(
        input_list=args.input_list,
        output_json=args.output_json,
        question_file=args.question_file,
    )
