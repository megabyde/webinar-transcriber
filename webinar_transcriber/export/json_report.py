"""JSON export helpers."""

from pathlib import Path

from webinar_transcriber.models import ReportDocument


def write_json_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to JSON.

    Returns:
        Path: The written JSON artifact path.
    """
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output_path
