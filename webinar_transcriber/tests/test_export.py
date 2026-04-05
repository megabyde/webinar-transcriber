"""Tests for report exporters."""

import json
from pathlib import Path

from docx import Document

from webinar_transcriber.export.docx_report import write_docx_report
from webinar_transcriber.export.formatting import format_timecode
from webinar_transcriber.export.json_report import write_json_report
from webinar_transcriber.export.markdown import write_markdown_report
from webinar_transcriber.export.subtitles import write_vtt_subtitles
from webinar_transcriber.models import (
    MediaType,
    ReportDocument,
    ReportSection,
    TranscriptionResult,
    TranscriptSegment,
)

EN_DASH = "\N{EN DASH}"


def _style_name(paragraph) -> str | None:
    style = paragraph.style
    return None if style is None else style.name


def test_format_timecode_formats_short_and_long_durations() -> None:
    assert format_timecode(65.9) == "01:05"
    assert format_timecode(3661.2) == "01:01:01"


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
                tldr="Краткое резюме раздела.",
                transcript_text="Первый абзац.\n\nВторой абзац.",  # noqa: RUF001
            )
        ],
    )

    output_path = tmp_path / "report.docx"
    write_docx_report(report, output_path)

    document = Document(str(output_path))
    paragraph_texts = [paragraph.text for paragraph in document.paragraphs]

    assert f"Section 1 (00:00{EN_DASH}00:05)" in paragraph_texts
    assert "TL;DR / Cheat Sheet" in paragraph_texts
    assert "Transcript" in paragraph_texts
    assert "Краткое резюме раздела." in paragraph_texts
    assert "Первый абзац." in paragraph_texts
    assert "Второй абзац." in paragraph_texts


def test_write_docx_report_formats_cheat_sheet_lists(tmp_path: Path) -> None:
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
                tldr="- First point.\n\n1. Second point.\n2) Third point.",
                transcript_text="Transcript body.",
            )
        ],
    )

    output_path = tmp_path / "report.docx"
    write_docx_report(report, output_path)

    document = Document(str(output_path))
    paragraph_data = [(paragraph.text, _style_name(paragraph)) for paragraph in document.paragraphs]

    assert ("First point.", "List Bullet") in paragraph_data
    assert ("Second point.", "List Number") in paragraph_data
    assert ("Third point.", "List Number") in paragraph_data


def test_write_docx_report_skips_missing_section_image(tmp_path: Path, caplog) -> None:
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

    output_path = tmp_path / "report.docx"

    write_docx_report(report, output_path)

    assert output_path.exists()
    assert "Section image does not exist" in caplog.text


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
                tldr="Section recap.",
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

    expected_markdown = (
        "# Demo\n\n"
        "- Language: `en`\n\n"
        "## Summary\n\n"
        "- Summary point.\n\n"
        "## Action Items\n\n"
        "- Follow up.\n\n"
        "## Sections\n\n"
        f"### Section 1 (00:00{EN_DASH}00:05)\n\n"
        "**TL;DR / Cheat Sheet**\n\n"
        "Section recap.\n\n"
        "**Transcript**\n\n"
        "Paragraph one.\n\n"
        f"### Section 2 (00:05{EN_DASH}00:10)\n\n"
        "![Section 2](frames/scene-2.png)\n\n"
        "Paragraph two.\n"
    )

    assert output_path.read_text(encoding="utf-8") == expected_markdown


def test_write_markdown_report_uses_hour_format_for_long_sections(tmp_path: Path) -> None:
    report = ReportDocument(
        title="Demo",
        source_file="demo.wav",
        media_type=MediaType.AUDIO,
        sections=[
            ReportSection(
                id="section-1",
                title="Section 1",
                start_sec=3661.2,
                end_sec=7325.9,
                transcript_text="Paragraph one.",
            )
        ],
    )

    output_path = tmp_path / "report.md"
    write_markdown_report(report, output_path)

    markdown = output_path.read_text(encoding="utf-8")

    assert f"### Section 1 (01:01:01{EN_DASH}02:02:05)" in markdown


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


def test_write_vtt_subtitles_formats_cues(tmp_path: Path) -> None:
    transcription = TranscriptionResult(
        segments=[
            TranscriptSegment(
                id="segment-1",
                text="First subtitle.",
                start_sec=0.0,
                end_sec=1.25,
            ),
            TranscriptSegment(
                id="segment-2",
                text="Second subtitle.",
                start_sec=61.5,
                end_sec=62.75,
            ),
        ],
    )

    output_path = tmp_path / "transcript.vtt"
    write_vtt_subtitles(transcription, output_path)

    expected_vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.250\n"
        "First subtitle.\n\n"
        "00:01:01.500 --> 00:01:02.750\n"
        "Second subtitle.\n"
    )

    assert output_path.read_text(encoding="utf-8") == expected_vtt
