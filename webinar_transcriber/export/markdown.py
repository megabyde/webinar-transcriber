"""Markdown export helpers."""

from math import floor
from pathlib import Path

from webinar_transcriber.models import ReportDocument

EN_DASH = "\N{EN DASH}"


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
        title = section.title
        timecode = _section_timecode(section.start_sec, section.end_sec)
        lines.extend([f"### {title} ({timecode})", ""])
        if section.image_path:
            lines.extend([f"![{section.title}]({section.image_path})", ""])
        if section.tldr:
            lines.extend(["**TL;DR / Cheat Sheet**", "", section.tldr, "", "**Transcript**", ""])
        lines.extend([section.transcript_text, ""])

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def _section_timecode(start_sec: float, end_sec: float) -> str:
    return f"{_format_timecode(start_sec)}{EN_DASH}{_format_timecode(end_sec)}"


def _format_timecode(total_sec: float) -> str:
    clamped_sec = max(total_sec, 0.0)
    rounded_sec = floor(clamped_sec)
    hours, remainder = divmod(rounded_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
