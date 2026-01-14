# utils/load_timm_from_hf.py

import os
from timm import create_model
from huggingface_hub import hf_hub_download
from utils.model_registry import HF_TIMM_MODEL_MAP


def load_timm_model_from_hf(hf_repo: str):
    if hf_repo not in HF_TIMM_MODEL_MAP:
        raise KeyError(
            f"HF model not registered: {hf_repo}\n"
            f"Available models:\n" + "\n".join(HF_TIMM_MODEL_MAP.keys())
        )

    cfg = HF_TIMM_MODEL_MAP[hf_repo]

    # 1️⃣ download checkpoint
    ckpt_path = hf_hub_download(
        repo_id=hf_repo,
        filename=cfg.get("checkpoint", "model_best.pth.tar"),
    )

    # 2️⃣ build timm model
    model = create_model(
        cfg["timm_model"],
        num_classes=cfg["num_classes"],
        in_chans=cfg.get("in_chans", 3),
        pretrained=False,              # 🔥 關鍵
        checkpoint_path=ckpt_path,
    )

    return model, cfg
