# utils/hf_utils.py
from huggingface_hub import hf_hub_download
from pathlib import Path

def resolve_timm_checkpoint(repo_id: str, filename: str | None = None) -> str:
    """
    Download a timm-style checkpoint from Hugging Face and return local path.

    If filename is None, will try common names.
    """
    candidates = (
        [filename]
        if filename
        else [
            "model_best.pth.tar",
            "model.pth.tar",
            "checkpoint.pth.tar",
            "pytorch_model.bin",
        ]
    )

    last_err = None
    for fname in candidates:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
            return path
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"❌ Cannot find a valid checkpoint in {repo_id}. "
        f"Tried files: {candidates}"
    ) from last_err
