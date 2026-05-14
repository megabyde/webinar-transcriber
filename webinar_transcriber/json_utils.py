"""Shared JSON payload helpers."""

from __future__ import annotations


def compact_speaker_fields(value: object) -> object:
    """Return a JSON payload without empty speaker fields."""
    if isinstance(value, list):
        return [compact_speaker_fields(item) for item in value]
    if not isinstance(value, dict):
        return value

    compacted: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if key == "speaker" and item is None:
            continue
        if key == "speakers" and item == []:
            continue
        compacted[key] = compact_speaker_fields(item)
    return compacted
