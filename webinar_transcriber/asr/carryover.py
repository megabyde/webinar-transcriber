"""Prompt-carryover heuristics for adjacent whisper inference windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webinar_transcriber.models import DecodedWindow

    from .config import PromptCarryoverSettings

_CARRYOVER_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CARRYOVER_WHITESPACE = re.compile(r"\s+")
_CARRYOVER_TRAILING_NOISE = re.compile(r"[\(\[\{<\"'`]+$|[\s\-:,;]+$")


def build_prompt_carryover(
    decoded_window: DecodedWindow, *, settings: PromptCarryoverSettings
) -> str | None:
    """Return a bounded prompt suffix for the next window, or `None` when confidence is weak."""
    if _carryover_drop_reason(decoded_window, settings=settings) is not None:
        return None

    sentences = [
        stripped
        for part in _CARRYOVER_SENTENCE_SPLIT.split(decoded_window.text)
        if (stripped := part.strip())
    ]
    carryover = " ".join(sentences[-max(1, settings.max_sentences) :])
    carryover = _sanitize_prompt(carryover, max_tokens=settings.max_tokens)
    return carryover or None


def _carryover_drop_reason(
    decoded_window: DecodedWindow, *, settings: PromptCarryoverSettings
) -> str | None:
    if not settings.enabled:
        return "carryover_disabled"
    if not decoded_window.text.strip():
        return "empty_text"
    return None


def _sanitize_prompt(prompt: str | None, *, max_tokens: int) -> str:
    if not prompt:
        return ""

    cleaned = _CARRYOVER_WHITESPACE.sub(" ", prompt.strip())
    cleaned = _CARRYOVER_TRAILING_NOISE.sub("", cleaned).strip()
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    token_limit = min(len(tokens), max(0, max_tokens))
    tokens = tokens[-token_limit:] if token_limit else []
    return " ".join(tokens)
