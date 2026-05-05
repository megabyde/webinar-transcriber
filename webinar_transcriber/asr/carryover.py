"""Prompt-carryover helpers for adjacent whisper inference windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webinar_transcriber.models import DecodedWindow

    from .config import PromptCarryoverSettings

_CARRYOVER_WHITESPACE = re.compile(r"\s+")


def build_prompt_carryover(
    decoded_window: DecodedWindow, *, settings: PromptCarryoverSettings
) -> str | None:
    """Return a bounded prompt suffix for the next window, or `None` when confidence is weak."""
    if not settings.enabled:
        return None

    carryover = _tail_tokens(decoded_window.text, max_tokens=settings.max_tokens)
    return carryover or None


def _tail_tokens(text: str, *, max_tokens: int) -> str:
    if not text:
        return ""

    cleaned = _CARRYOVER_WHITESPACE.sub(" ", text.strip())
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    token_limit = min(len(tokens), max(0, max_tokens))
    tokens = tokens[-token_limit:] if token_limit else []
    return " ".join(tokens)
