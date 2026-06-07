"""JSON export helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.processor.stages import write_json

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.models import ReportDocument


def write_json_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to JSON.

    Returns:
        Path: The written JSON artifact path.
    """
    write_json(output_path, report.to_json())
    return output_path
