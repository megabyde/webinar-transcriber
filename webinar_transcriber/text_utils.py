"""Shared text formatting helpers."""

from __future__ import annotations

import re

SENTENCE_TERMINATORS = (
    ".!?"
    "\N{IDEOGRAPHIC FULL STOP}"
    "\N{FULLWIDTH EXCLAMATION MARK}"
    "\N{FULLWIDTH QUESTION MARK}"
    "\N{HORIZONTAL ELLIPSIS}"
)

_FLEXIBLE_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


def split_paragraph_blocks(text: str) -> list[str]:
    """Split text into non-empty paragraph blocks separated by blank lines."""
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def split_llm_paragraph_blocks(text: str) -> list[str]:
    """Split LLM-emitted text into paragraphs, tolerating loose blank lines.

    Treats any run of blank/whitespace-only lines as a paragraph break and
    collapses inline runs of spaces and tabs within each paragraph.
    """
    paragraphs = []
    for block in _FLEXIBLE_BLANK_LINE_RE.split(text):
        lines = [
            _INLINE_WHITESPACE_RE.sub(" ", line.strip()) for line in block.strip().splitlines()
        ]
        paragraph = "\n".join(line for line in lines if line)
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs
