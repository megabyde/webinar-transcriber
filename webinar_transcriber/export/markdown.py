"""Markdown export helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from webinar_transcriber.export.formatting import section_timecode

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.models import ReportDocument


def write_markdown_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to Markdown.

    Returns:
        Path: The written Markdown artifact path.
    """
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
        title = section.title
        timecode = section_timecode(section.start_sec, section.end_sec)
        lines.extend([f"### {title} ({timecode})", ""])
        if section.image_path:
            lines.extend([f"![{section.title}]({section.image_path})", ""])
        if section.tldr:
            lines.extend(["**TL;DR / Cheat Sheet**", "", section.tldr, "", "**Transcript**", ""])
        lines.extend([section.transcript_text, ""])

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path
