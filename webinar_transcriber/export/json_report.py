"""JSON export helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from webinar_transcriber.models import _compact_dict_factory
from webinar_transcriber.processor.stages import write_json

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.models import ReportDocument


def write_json_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to JSON.

    Returns:
        Path: The written JSON artifact path.
    """
    write_json(output_path, asdict(report, dict_factory=_compact_dict_factory))
    return output_path
