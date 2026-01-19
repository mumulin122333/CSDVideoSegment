#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt schema utilities.

This module defines:
- how to load prompt → output-field mappings from JSON
- how to extract attributes from a VLM prompt/prediction pair

Design principles:
- JSON contains FULL question text (no abbreviation)
- One prompt corresponds to exactly one output field
- If a prompt is not listed in schema, it is ignored
- Image and video pipelines share this module
"""

import json
from pathlib import Path
from typing import Dict, List


def load_prompt_schema(schema_path: Path) -> List[Dict[str, str]]:
    """
    Load prompt schema JSON.

    Expected JSON format:
    [
      {
        "question": "<FULL QUESTION TEXT>",
        "field": "<output_field_name>"
      },
      ...
    ]
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # basic validation
    for i, item in enumerate(schema):
        if "question" not in item or "field" not in item:
            raise ValueError(
                f"Invalid schema entry at index {i}: "
                f"each item must contain 'question' and 'field'"
            )

    return schema


def extract_attrs_from_prompt(
    prompt: str,
    prediction: str,
    schema: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Extract structured attributes from a single prompt/prediction pair.

    Matching rule:
    - schema[i]["question"] must appear verbatim (substring) in prompt
    - matching is case-insensitive
    - first match wins
    - if no match is found, return empty dict

    Args:
        prompt: full prompt text sent to the VLM
        prediction: model output
        schema: loaded prompt schema

    Returns:
        dict: { field_name: prediction } or {}
    """
    prompt_l = prompt.lower()

    for item in schema:
        q = item["question"].lower()
        if q in prompt_l:
            return {item["field"]: prediction}

    return {}
