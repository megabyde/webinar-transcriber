"""Prompt-carryover helpers for adjacent whisper inference windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .config import CARRYOVER_MAX_CHARS

if TYPE_CHECKING:
    from webinar_transcriber.models import DecodedWindow

_CARRYOVER_WHITESPACE = re.compile(r"\s+")


def build_prompt_carryover(
    decoded_window: DecodedWindow, *, max_chars: int = CARRYOVER_MAX_CHARS
) -> str | None:
    """Return a bounded prompt suffix for the next window, or `None` when confidence is weak."""
    carryover = _tail_text(decoded_window.text, max_chars=max_chars)
    return carryover or None


def _tail_text(text: str, *, max_chars: int) -> str:
    if not text:
        return ""

    cleaned = _CARRYOVER_WHITESPACE.sub(" ", text.strip())
    if not cleaned or max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[-max_chars:].lstrip()
