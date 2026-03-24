"""Tests for report exporters."""

from pathlib import Path

from docx import Document

from webinar_transcriber.export.docx_report import write_docx_report
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
