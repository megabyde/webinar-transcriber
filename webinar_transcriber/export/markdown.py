"""Markdown export helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from webinar_transcriber.export.formatting import format_timecode, section_timecode

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

    if len(report.sections) > 1:
        lines.extend(["## Contents", ""])
        for index, section in enumerate(report.sections, start=1):
            timestamp = format_timecode(section.start_sec)
            lines.append(f"- [{timestamp}](#section-{index}) {section.title}")
        lines.append("")
    lines.extend(["## Sections", ""])

    for index, section in enumerate(report.sections, start=1):
        title = section.title
        timecode = section_timecode(section.start_sec, section.end_sec)
        if len(report.sections) > 1:
            lines.extend([f'<a name="section-{index}"></a>', ""])
        lines.extend([f"### {title} ({timecode})", ""])
        if section.image_path:
            lines.extend([
                f"![{section.title}]({_markdown_image_destination(section.image_path)})",
                "",
            ])
        if section.tldr and (tldr := section.tldr.strip()):
            lines.extend(["**AI-generated notes**", "", tldr, "", "**Transcript**", ""])
        lines.extend([section.transcript_text, ""])

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def _markdown_image_destination(path: str) -> str:
    if any(char.isspace() or char in "()[]<>" for char in path):
        return f"<{quote(path, safe='/:._-')}>"
    return path
