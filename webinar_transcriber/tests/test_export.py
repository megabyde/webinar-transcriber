"""Tests for report exporters."""

import json
from html.parser import HTMLParser
from pathlib import Path

from docx import Document
from PIL import Image

from webinar_transcriber.export.docx_report import write_docx_report
from webinar_transcriber.export.formatting import (
    format_count,
    format_duration,
    format_rtf,
    format_timecode,
    join_detail,
)
from webinar_transcriber.export.html_report import write_html_report
from webinar_transcriber.export.json_report import write_json_report
from webinar_transcriber.export.markdown import write_markdown_report
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection


def _style_name(paragraph) -> str | None:
    style = paragraph.style
    return None if style is None else style.name


class TestFormatting:
    def test_format_timecode_formats_short_and_long_durations(self) -> None:
        assert format_timecode(65.9) == "01:05"
        assert format_timecode(3661.2) == "01:01:01"

    def test_format_duration_formats_minutes_and_hours(self) -> None:
        assert format_duration(65.0) == "1m 05s"
        assert format_duration(3661.0) == "1h 01m 01s"

    def test_formats_progress_detail_parts(self) -> None:
        assert format_count(1, "region") == "1 region"
        assert format_count(2, "region") == "2 regions"
        assert format_rtf(10.0, 4.0) == "RTF 2.5x"
        assert join_detail("2 regions", None, "RTF 2.5x") == "2 regions | RTF 2.5x"


class TestDocxReport:
    def test_splits_blank_line_paragraphs(self, tmp_path: Path) -> None:
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

        assert "Section 1 (00:00\N{EN DASH}00:05)" in paragraph_texts
        assert "AI-generated notes" in paragraph_texts
        assert "Transcript" in paragraph_texts
        assert "Краткое резюме раздела." in paragraph_texts
        assert "Первый абзац." in paragraph_texts
        assert "Второй абзац." in paragraph_texts

    def test_formats_cheat_sheet_lists(self, tmp_path: Path) -> None:
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
        paragraph_data = [
            (paragraph.text, _style_name(paragraph)) for paragraph in document.paragraphs
        ]

        assert ("First point.", "List Bullet") in paragraph_data
        assert ("Second point.", "List Number") in paragraph_data
        assert ("Third point.", "List Number") in paragraph_data

    def test_formats_canonical_tldr_lines(self, tmp_path: Path) -> None:
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
                    tldr="\n\nHeading line\n- Bullet point.\n1. Numbered point\n\n\n",
                    transcript_text="Transcript body.",
                )
            ],
        )

        output_path = tmp_path / "report.docx"
        write_docx_report(report, output_path)

        document = Document(str(output_path))
        paragraph_data = [
            (paragraph.text, _style_name(paragraph)) for paragraph in document.paragraphs
        ]

        assert ("Heading line", "Normal") in paragraph_data
        assert ("Bullet point.", "List Bullet") in paragraph_data
        assert ("Numbered point", "List Number") in paragraph_data
        assert ("Transcript", "Normal") in paragraph_data

    def test_skips_missing_section_image(self, tmp_path: Path) -> None:
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

        # A missing image is skipped silently; the DOCX is still written
        assert output_path.exists()

    def test_embeds_section_image(self, tmp_path: Path) -> None:
        image_path = tmp_path / "frame.png"
        Image.new("RGB", (8, 8), color="white").save(image_path)
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.VIDEO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Section 1",
                    start_sec=0.0,
                    end_sec=5.0,
                    transcript_text="Paragraph",
                    image_path=str(image_path),
                )
            ],
        )

        output_path = tmp_path / "report.docx"
        write_docx_report(report, output_path)

        document = Document(str(output_path))
        assert len(document.inline_shapes) == 1

    def test_embeds_relative_section_image_from_report_directory(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        Image.new("RGB", (8, 8), color="white").save(frames_dir / "scene-1.png")
        report = ReportDocument(
            title="Demo",
            source_file="demo.mp4",
            media_type=MediaType.VIDEO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Section 1",
                    start_sec=0.0,
                    end_sec=5.0,
                    transcript_text="Paragraph",
                    image_path="frames/scene-1.png",
                )
            ],
        )

        output_path = tmp_path / "report.docx"
        write_docx_report(report, output_path)

        document = Document(str(output_path))
        assert len(document.inline_shapes) == 1

    def test_formats_speaker_prefixes_as_bold_runs(self, tmp_path: Path) -> None:
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
                    tldr="**S1:** Summary paragraph.",
                    transcript_text="**S1:** First paragraph.\n\n**S2:** Second paragraph.",
                )
            ],
        )

        output_path = tmp_path / "report.docx"
        write_docx_report(report, output_path)

        document = Document(str(output_path))
        speaker_paragraphs = [
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text in {"S1: First paragraph.", "S2: Second paragraph."}
        ]
        assert [paragraph.runs[0].bold for paragraph in speaker_paragraphs] == [True, True]
        assert "S1: Summary paragraph." in [paragraph.text for paragraph in document.paragraphs]


class TestHtmlReport:
    class _TagChecker(HTMLParser):
        VOID_TAGS = frozenset({"br", "img", "meta"})

        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.ids: list[str] = []
            self.urls: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            for name, value in attrs:
                if name == "id" and value:
                    self.ids.append(value)
                if name in {"src", "href"} and value:
                    self.urls.append(value)
            if tag not in self.VOID_TAGS:
                self.stack.append(tag)

        def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            # Self-closing SVG shapes open and close in one tag
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

        def handle_endtag(self, tag: str) -> None:
            assert self.stack.pop() == tag

    def _parse(self, html: str) -> _TagChecker:
        checker = self._TagChecker()
        checker.feed(html)
        checker.close()
        assert checker.stack == []
        return checker

    def test_writes_self_contained_report_with_contents(self, tmp_path: Path) -> None:
        Image.new("RGB", (2400, 1200), color="white").save(tmp_path / "scene-1.png")
        report = ReportDocument(
            title="Q&A <Live>",
            source_file="/recordings/demo.mp4",
            media_type=MediaType.VIDEO,
            detected_language="en",
            summary=["Point with **emphasis**."],
            action_items=["Follow up."],
            sections=[
                ReportSection(
                    id="section-1",
                    title="Opening",
                    start_sec=0.0,
                    end_sec=65.0,
                    tldr="- First note\n- Second note\n\n1. Step one\n2. Step two",
                    transcript_text="**S1:** Hello there.\n\n**S10:** Hi back.\n\nPlain {{title}}",
                    image_path="scene-1.png",
                ),
                ReportSection(
                    id="section-2",
                    title="Closing",
                    start_sec=65.0,
                    end_sec=3700.0,
                    transcript_text="Goodbye.",
                    image_path="frames/missing.png",
                ),
            ],
        )

        output_path = tmp_path / "report.html"
        write_html_report(report, output_path)

        html = output_path.read_text(encoding="utf-8")
        checker = self._parse(html)
        assert checker.urls
        assert not [url for url in checker.urls if "://" in url]
        assert {"section-1", "section-2", "summary", "action-items"} <= set(checker.ids)
        assert '<html lang="en">' in html
        assert "<title>Q&amp;A &lt;Live&gt;</title>" in html
        assert "<li>Point with <strong>emphasis</strong>.</li>" in html
        assert (
            "<li>1h 01m 40s</li><li>2 sections</li><li>Language: en</li><li>demo.mp4</li>" in html
        )
        assert '<span class="toc-time">01:05</span><span class="toc-title">Closing</span>' in html
        assert "<ul><li>First note</li><li>Second note</li></ul>" in html
        assert "<ol><li>Step one</li><li>Step two</li></ol>" in html
        assert '<span class="speaker spk-1">S1</span> Hello there.' in html
        assert '<span class="speaker spk-2">S10</span> Hi back.' in html
        assert "<p>Plain {{title}}</p>" in html
        assert html.count('src="data:image/webp;base64,') == 1
        assert 'width="1920" height="960"' in html

    def test_single_section_report_has_no_contents(self, tmp_path: Path) -> None:
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Only",
                    start_sec=0.0,
                    end_sec=5.0,
                    transcript_text="Line one\nline two.",
                )
            ],
        )

        output_path = tmp_path / "report.html"
        write_html_report(report, output_path)

        html = output_path.read_text(encoding="utf-8")
        self._parse(html)
        assert '<html lang="und">' in html
        assert 'class="layout no-toc"' in html
        assert 'class="toc"' not in html
        assert 'class="overview"' not in html
        assert 'class="notes"' not in html
        assert "<p>Line one<br>line two.</p>" in html


class TestMarkdownReport:
    def test_omits_blank_image_line_for_imageless_sections(self, tmp_path: Path) -> None:
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
            "## Contents\n\n"
            "- [00:00](#section-1) Section 1\n"
            "- [00:05](#section-2) Section 2\n\n"
            "## Sections\n\n"
            '<a name="section-1"></a>\n\n'
            "### Section 1 (00:00\N{EN DASH}00:05)\n\n"
            "**AI-generated notes**\n\n"
            "Section recap.\n\n"
            "**Transcript**\n\n"
            "Paragraph one.\n\n"
            '<a name="section-2"></a>\n\n'
            "### Section 2 (00:05\N{EN DASH}00:10)\n\n"
            "![Section 2](frames/scene-2.png)\n\n"
            "Paragraph two.\n"
        )

        assert output_path.read_text(encoding="utf-8") == expected_markdown

    def test_escapes_image_paths_that_need_markdown_destination_quoting(
        self, tmp_path: Path
    ) -> None:
        report = ReportDocument(
            title="Demo",
            source_file="demo.mp4",
            media_type=MediaType.VIDEO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Section 1",
                    start_sec=0.0,
                    end_sec=5.0,
                    transcript_text="Paragraph one.",
                    image_path="frames/slides (final)] 01.png",
                )
            ],
        )

        output_path = tmp_path / "report.md"
        write_markdown_report(report, output_path)

        assert "![Section 1](<frames/slides%20%28final%29%5D%2001.png>)" in output_path.read_text(
            encoding="utf-8"
        )

    def test_uses_hour_format_for_long_sections(self, tmp_path: Path) -> None:
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

        assert "### Section 1 (01:01:01\N{EN DASH}02:02:05)" in markdown

    def test_preserves_speaker_prefixes(self, tmp_path: Path) -> None:
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
                    transcript_text="**S1:** First paragraph.\n\n**S2:** Second paragraph.",
                )
            ],
        )

        output_path = tmp_path / "report.md"
        write_markdown_report(report, output_path)

        markdown = output_path.read_text(encoding="utf-8")
        assert "**S1:** First paragraph." in markdown
        assert "**S2:** Second paragraph." in markdown


class TestJsonReport:
    def test_round_trips_report_document(self, tmp_path: Path) -> None:
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

        assert payload["title"] == "Demo"
        assert payload["media_type"] == "audio"
        assert payload["summary"] == ["Summary point."]
        assert payload["action_items"] == ["Follow up."]
        assert payload["sections"] == [
            {
                "id": "section-1",
                "title": "Section 1",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "transcript_text": "Paragraph one.",
                "tldr": None,
                "image_path": None,
            }
        ]
