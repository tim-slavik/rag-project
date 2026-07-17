import re


def clean_text(text: str) -> str:
    """
    Normalize whitespace and strip leading/trailing spaces.
    Keeps chunking consistent across strategies.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """
    Basic token counter using whitespace splitting.
    This keeps things simple and strategy‑agnostic.
    """
    return len(text.split())
