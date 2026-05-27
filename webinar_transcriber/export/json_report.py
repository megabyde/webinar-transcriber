"""JSON export helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from webinar_transcriber.models import compact_speaker_fields
from webinar_transcriber.processor.support import write_json

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.models import ReportDocument


def write_json_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to JSON.

    Returns:
        Path: The written JSON artifact path.
    """
    write_json(output_path, compact_speaker_fields(asdict(report)))
    return output_path
