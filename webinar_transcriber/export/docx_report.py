"""DOCX export helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from docx import Document
from docx.shared import Inches

from webinar_transcriber.export.formatting import section_timecode

if TYPE_CHECKING:
    from collections.abc import Callable

    from docx.document import Document as DocxDocument

    from webinar_transcriber.models import ReportDocument, ReportSection


def write_docx_report(
    report: ReportDocument,
    output_path: Path,
    *,
    warning_callback: Callable[[str], None] | None = None,
) -> Path:
    """Write the report to DOCX.

    Returns:
        Path: The written DOCX artifact path.
    """
    document = Document()
    document.add_heading(report.title, level=0)

    if report.detected_language:
        document.add_paragraph(f"Language: {report.detected_language}")

    if report.summary:
        document.add_heading("Summary", level=1)
        for item in report.summary:
            document.add_paragraph(item, style="List Bullet")

    if report.action_items:
        document.add_heading("Action Items", level=1)
        for item in report.action_items:
            document.add_paragraph(item, style="List Bullet")

    document.add_heading("Sections", level=1)
    for section in report.sections:
        _add_section(document, section, warning_callback=warning_callback)

    document.save(str(output_path))
    return output_path


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p for block in text.split("\n\n") if (p := block.strip())]
    return paragraphs or [text]


def _add_section(
    document: DocxDocument,
    section: ReportSection,
    *,
    warning_callback: Callable[[str], None] | None = None,
) -> None:
    title = section.title
    timecode = section_timecode(section.start_sec, section.end_sec)
    document.add_heading(f"{title} ({timecode})", level=2)
    _add_section_image(document, section.image_path, warning_callback=warning_callback)
    _add_section_tldr(document, section.tldr)
    for paragraph_text in _split_paragraphs(section.transcript_text):
        document.add_paragraph(paragraph_text)


def _add_section_image(
    document: DocxDocument,
    image_path: str | None,
    *,
    warning_callback: Callable[[str], None] | None = None,
) -> None:
    if not image_path:
        return

    resolved_path = Path(image_path)
    if not resolved_path.exists():
        if warning_callback is not None:
            warning_callback(f"Section image does not exist: {resolved_path}")
        return
    document.add_picture(str(resolved_path), width=Inches(6))


def _add_section_tldr(document: DocxDocument, tldr: str | None) -> None:
    if not tldr:
        return

    tldr_label = document.add_paragraph()
    tldr_label.add_run("TL;DR / Cheat Sheet").bold = True
    _add_text_blocks(document, tldr)
    transcript_label = document.add_paragraph()
    transcript_label.add_run("Transcript").bold = True


def _add_text_blocks(document: DocxDocument, text: str) -> None:
    for paragraph_text in _split_paragraphs(text):
        lines = [line.strip() for line in paragraph_text.splitlines() if line.strip()]
        if not lines:
            continue
        if _all_list_items(lines):
            for line in lines:
                body, style = cast("tuple[str, str]", _list_item_parts(line))
                document.add_paragraph(body, style=style)
            continue
        body, style = _list_item_parts(paragraph_text)
        if body is not None and style is not None:
            document.add_paragraph(body, style=style)
            continue
        document.add_paragraph(paragraph_text)


def _all_list_items(lines: list[str]) -> bool:
    return all(_list_item_parts(line)[0] is not None for line in lines)


def _list_item_parts(text: str) -> tuple[str | None, str | None]:
    bullet_match = re.match(r"^[-*]\s+(.*)$", text, flags=re.DOTALL)
    if bullet_match is not None:
        return bullet_match.group(1), "List Bullet"

    number_match = re.match(r"^\d+[.)]\s+(.*)$", text, flags=re.DOTALL)
    if number_match is not None:
        return number_match.group(1), "List Number"

    return None, None
