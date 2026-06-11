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
    """Return a bounded prompt suffix for the next window, or `None` when the text is empty."""
    cleaned = _CARRYOVER_WHITESPACE.sub(" ", decoded_window.text.strip())
    if not cleaned or max_chars <= 0:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:].lstrip()
