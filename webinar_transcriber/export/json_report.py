"""JSON export helpers."""

import json
from dataclasses import asdict
from pathlib import Path

from webinar_transcriber.models import ReportDocument


def write_json_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to JSON.

    Returns:
        Path: The written JSON artifact path.
    """
    output_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
