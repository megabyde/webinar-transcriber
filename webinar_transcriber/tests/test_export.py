"""Tests for report exporters."""

import json
from pathlib import Path

import pytest
from docx import Document

from webinar_transcriber.export.docx_report import write_docx_report
from webinar_transcriber.export.json_report import write_json_report
from webinar_transcriber.export.markdown import write_markdown_report
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection


def test_write_docx_report_splits_blank_line_paragraphs(tmp_path: Path) -> None:
    report = ReportDocument(
        title="Demo",
        source_file="demo.wav",
        media_type=MediaType.AUDIO,
        sections=[
            ReportSection(
                id="section-1",
                title="Section 1",
                start_sec=0.0,
                end_sec=5.0,
                transcript_text="Первый абзац.\n\nВторой абзац.",  # noqa: RUF001
            )
        ],
    )

    output_path = tmp_path / "report.docx"
    write_docx_report(report, output_path)

    document = Document(str(output_path))
    paragraph_texts = [paragraph.text for paragraph in document.paragraphs]

    assert "Первый абзац." in paragraph_texts
    assert "Второй абзац." in paragraph_texts


def test_write_docx_report_raises_clear_error_for_missing_section_image(tmp_path: Path) -> None:
    report = ReportDocument(
        title="Demo",
        source_file="demo.wav",
        media_type=MediaType.AUDIO,
        sections=[
            ReportSection(
                id="section-1",
                title="Section 1",
                start_sec=0.0,
                end_sec=5.0,
                transcript_text="Paragraph",
                image_path=str(tmp_path / "missing.png"),
            )
        ],
    )

    with pytest.raises(FileNotFoundError, match="Section image does not exist"):
        write_docx_report(report, tmp_path / "report.docx")


def test_write_markdown_report_omits_blank_image_line_for_imageless_sections(
    tmp_path: Path,
) -> None:
    report = ReportDocument(
        title="Demo",
        source_file="demo.wav",
        media_type=MediaType.AUDIO,
        detected_language="en",
        summary=["Summary point."],
        action_items=["Follow up."],
        sections=[
            ReportSection(
                id="section-1",
                title="Section 1",
                start_sec=0.0,
                end_sec=5.0,
                transcript_text="Paragraph one.",
            ),
            ReportSection(
                id="section-2",
                title="Section 2",
                start_sec=5.0,
                end_sec=10.0,
                transcript_text="Paragraph two.",
                image_path="frames/scene-2.png",
            ),
        ],
    )

    output_path = tmp_path / "report.md"
    write_markdown_report(report, output_path)

    assert output_path.read_text(encoding="utf-8") == (
        "# Demo\n\n"
        "- Language: `en`\n\n"
        "## Summary\n\n"
        "- Summary point.\n\n"
        "## Action Items\n\n"
        "- Follow up.\n\n"
        "## Sections\n\n"
        "### Section 1\n\n"
        "Paragraph one.\n\n"
        "### Section 2\n\n"
        "![Section 2](frames/scene-2.png)\n\n"
        "Paragraph two.\n"
    )


def test_write_json_report_round_trips_report_document(tmp_path: Path) -> None:
    report = ReportDocument(
        title="Demo",
        source_file="demo.wav",
        media_type=MediaType.AUDIO,
        summary=["Summary point."],
        action_items=["Follow up."],
        sections=[
            ReportSection(
                id="section-1",
                title="Section 1",
                start_sec=0.0,
                end_sec=5.0,
                transcript_text="Paragraph one.",
            )
        ],
    )

    output_path = tmp_path / "report.json"
    write_json_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    restored = ReportDocument.model_validate(payload)

    assert restored == report
