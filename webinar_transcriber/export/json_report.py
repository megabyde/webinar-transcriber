"""JSON export helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.models import ReportDocument


def write_json_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to JSON.

    Returns:
        Path: The written JSON artifact path.
    """
    output_path.write_text(
        json.dumps(_compact_speaker_fields(asdict(report)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _compact_speaker_fields(value: object) -> object:
    if isinstance(value, list):
        return [_compact_speaker_fields(item) for item in value]
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
        compacted[key] = _compact_speaker_fields(item)
    return compacted
