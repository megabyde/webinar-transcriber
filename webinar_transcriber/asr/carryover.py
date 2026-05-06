"""Prompt-carryover helpers for adjacent whisper inference windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webinar_transcriber.models import DecodedWindow

    from .config import PromptCarryoverSettings

_CARRYOVER_WHITESPACE = re.compile(r"\s+")
_SENTENCE_TERMINATORS = (
    ".!?"
    "\N{IDEOGRAPHIC FULL STOP}"
    "\N{FULLWIDTH EXCLAMATION MARK}"
    "\N{FULLWIDTH QUESTION MARK}"
    "\N{HORIZONTAL ELLIPSIS}"
)


def build_prompt_carryover(
    decoded_window: DecodedWindow, *, settings: PromptCarryoverSettings
) -> str | None:
    """Return a bounded prompt suffix for the next window, or `None` when confidence is weak."""
    if not settings.enabled:
        return None

    carryover = _tail_text(decoded_window.text, max_chars=settings.max_chars)
    return carryover or None


def _tail_text(text: str, *, max_chars: int) -> str:
    if not text:
        return ""

    cleaned = _CARRYOVER_WHITESPACE.sub(" ", text.strip())
    if not cleaned or max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    return _drop_partial_leading_sentence(cleaned[-max_chars:])


def _drop_partial_leading_sentence(text: str) -> str:
    for index, char in enumerate(text[:-1]):
        if char in _SENTENCE_TERMINATORS:
            trimmed = text[index + 1 :].strip()
            if trimmed:
                return trimmed
    return text.lstrip(f" {_SENTENCE_TERMINATORS}")
