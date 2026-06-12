"""Prompt-carryover helpers for adjacent whisper inference windows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from webinar_transcriber.asr.config import CARRYOVER_MAX_CHARS

if TYPE_CHECKING:
    from webinar_transcriber.models import DecodedWindow

_CARRYOVER_WHITESPACE = re.compile(r"\s+")


def build_prompt_carryover(decoded_window: DecodedWindow) -> str:
    """Return a bounded prompt suffix for the next window; empty when the text is empty."""
    cleaned = _CARRYOVER_WHITESPACE.sub(" ", decoded_window.text.strip())
    if len(cleaned) <= CARRYOVER_MAX_CHARS:
        return cleaned
    return cleaned[-CARRYOVER_MAX_CHARS:].lstrip()
