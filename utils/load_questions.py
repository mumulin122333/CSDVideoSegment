# utils/load_questions.py

from pathlib import Path


def load_questions(path: str) -> list[str]:
    """
    Load questions from a text file.
    Questions are separated by lines containing only '---'.
    """
    text = Path(path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("---")]
    return [b for b in blocks if b]
