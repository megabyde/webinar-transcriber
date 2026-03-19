"""Markdown export helpers."""

from pathlib import Path

from webinar_transcriber.models import ReportDocument


def write_markdown_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to Markdown."""
    lines = [f"# {report.title}", ""]

    if report.detected_language:
        lines.extend([f"- Language: `{report.detected_language}`", ""])

    if report.summary:
        lines.extend(["## Summary", ""])
        lines.extend([f"- {item}" for item in report.summary])
        lines.append("")

    if report.action_items:
        lines.extend(["## Action Items", ""])
        lines.extend([f"- {item}" for item in report.action_items])
        lines.append("")

    lines.extend(["## Sections", ""])

    for section in report.sections:
        lines.extend(
            [
                f"### {section.title}",
                "",
                section.transcript_text,
                "",
            ]
        )

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path
