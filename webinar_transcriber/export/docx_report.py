"""DOCX export helpers."""

from pathlib import Path

from docx import Document
from docx.shared import Inches

from webinar_transcriber.models import ReportDocument


def write_docx_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report to DOCX."""
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
        document.add_heading(section.title, level=2)
        if section.image_path:
            image_path = Path(section.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Section image does not exist: {image_path}")
            document.add_picture(str(image_path), width=Inches(6))
        for paragraph_text in _split_paragraphs(section.transcript_text):
            document.add_paragraph(paragraph_text)

    document.save(str(output_path))
    return output_path


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p for block in text.split("\n\n") if (p := block.strip())]
    return paragraphs or [text]
