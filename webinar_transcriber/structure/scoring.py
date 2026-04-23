"""Title helpers for report structuring."""

from __future__ import annotations

from pathlib import Path

from .constants import TITLE_WORD_LIMIT


def _title_from_text(text: str, *, fallback: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return fallback

    words = cleaned.split()
    return " ".join(words[:TITLE_WORD_LIMIT]) if len(words) > TITLE_WORD_LIMIT else cleaned


def _derive_title(source_path: str) -> str:
    stem = Path(source_path).stem
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Transcription Report"
