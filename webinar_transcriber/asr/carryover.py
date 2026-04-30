"""Prompt-carryover heuristics for adjacent whisper inference windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webinar_transcriber.models import DecodedWindow

    from .config import PromptCarryoverSettings

_CARRYOVER_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CARRYOVER_WHITESPACE = re.compile(r"\s+")


def build_prompt_carryover(
    decoded_window: DecodedWindow, *, settings: PromptCarryoverSettings
) -> str | None:
    """Return a bounded prompt suffix for the next window, or `None` when confidence is weak."""
    if _should_drop_carryover(decoded_window, settings=settings):
        return None

    sentences = [
        stripped
        for part in _CARRYOVER_SENTENCE_SPLIT.split(decoded_window.text)
        if (stripped := part.strip())
    ]
    carryover = " ".join(sentences[-max(1, settings.max_sentences) :])
    carryover = _sanitize_prompt(carryover, max_tokens=settings.max_tokens)
    return carryover or None


def _should_drop_carryover(
    decoded_window: DecodedWindow, *, settings: PromptCarryoverSettings
) -> bool:
    return not settings.enabled or not decoded_window.text.strip()


def _sanitize_prompt(prompt: str | None, *, max_tokens: int) -> str:
    if not prompt:
        return ""

    cleaned = _CARRYOVER_WHITESPACE.sub(" ", prompt.strip())
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    token_limit = min(len(tokens), max(0, max_tokens))
    tokens = tokens[-token_limit:] if token_limit else []
    return " ".join(tokens)
