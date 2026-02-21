from pathlib import Path
from re import search, Match
from typing import Optional


def extract_possible_unsplash_id(filepath: Path) -> str:
    """
    Attempt to extract the 'real' Unsplash image ID.

    Unsplash IDs are typically 11-character alphanumeric strings.
    This finds the last 11-char alphanumeric sequence before '-unsplash'.

    WARNING: This is heuristic-based and may be incorrect.
    For guaranteed uniqueness, use the full filename stem instead.
    """
    filename = filepath.stem

    # Pattern: 11 alphanumeric chars (case-sensitive) before -unsplash
    match: Optional[Match[str]] = search(
        pattern=r'([a-zA-Z0-9_-]{11})-unsplash$',
        string=filename
    )

    if match:
        return match.group(1)

    # Fallback: use full stem
    return filename
