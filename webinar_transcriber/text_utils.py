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


def split_paragraph_blocks(
    text: str, *, flexible_blank_lines: bool = False, normalize_inline_whitespace: bool = False
) -> list[str]:
    """Split text into non-empty paragraph blocks."""
    blocks = re.split(r"\n\s*\n+", text) if flexible_blank_lines else text.split("\n\n")
    paragraphs = []
    for block in blocks:
        if normalize_inline_whitespace:
            lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in block.strip().splitlines()]
            paragraph = "\n".join(line for line in lines if line)
        else:
            paragraph = block.strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs
