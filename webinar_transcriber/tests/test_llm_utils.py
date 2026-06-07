"""Tests for pure helpers in webinar_transcriber.llm.utils."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from webinar_transcriber.llm import LlmProcessingError
from webinar_transcriber.llm.utils import (
    ReportPolishResponse,
    ReportSectionUpdate,
    extract_response_metadata,
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_report_lines,
    schema_label,
    truncate_text,
    validated_section_titles,
)
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection


class TestLlmNormalization:
    def test_normalize_polished_section_text_preserves_single_line_breaks(self) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence.",
            polished_text="  First   line \nwith spacing.\n\n  Second\tparagraph.  ",
        )

        assert normalized == "First line\nwith spacing.\n\nSecond paragraph."

    def test_normalize_polished_section_text_preserves_ellipsis_for_incomplete_source(self) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence...", polished_text="Rewritten sentence..."
        )

        assert normalized == "Rewritten sentence..."

    def test_normalize_polished_section_text_keeps_original_when_polished_text_is_blank(
        self,
    ) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence.", polished_text="   "
        )

        assert normalized == "Original sentence."

    def test_normalize_polished_section_text_accepts_concise_output(self) -> None:
        normalized = normalize_polished_section_text(
            original_text="Long original text. " * 8, polished_text="Too short."
        )

        assert normalized == "Too short."

    def test_normalize_polished_section_tldr_preserves_single_line_breaks(self) -> None:
        normalized = normalize_polished_section_tldr(
            "  Bullet one \nspans whitespace.\n\n  Bullet two stays.  "
        )

        assert normalized == "Bullet one\nspans whitespace.\n\nBullet two stays."

    def test_normalize_polished_section_tldr_returns_blank_for_blank_input(self) -> None:
        assert normalize_polished_section_tldr("   ") == ""

    def test_normalize_polished_section_tldr_collapses_inline_bullet_spacing(self) -> None:
        normalized = normalize_polished_section_tldr(
            "Main points: - First point. - Second point. 1. Third point."
        )

        assert normalized == "Main points:\n- First point.\n- Second point.\n1. Third point."

    def test_normalize_report_lines_dedupes_and_limits_case_insensitively(self) -> None:
        normalized = normalize_report_lines(
            ["  First item  ", "first item", "", "Second item", "Third item"], limit=2
        )

        assert normalized == ["First item", "Second item"]

    def test_truncate_text_preserves_short_text_and_trims_long_text(self) -> None:
        assert truncate_text("Short text", 20) == "Short text"
        assert truncate_text("This sentence is too long", 11) == "This sente…"

    def test_validated_section_titles_rejects_duplicate_and_unknown_ids(self) -> None:
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Old title",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text="Transcript.",
                )
            ],
        )

        with pytest.raises(LlmProcessingError, match="unknown section ID"):
            validated_section_titles(
                report, [ReportSectionUpdate(id="section-x", title="Unexpected title")]
            )

        with pytest.raises(LlmProcessingError, match="duplicate section IDs"):
            validated_section_titles(
                report,
                [
                    ReportSectionUpdate(id="section-1", title="Title one"),
                    ReportSectionUpdate(id="section-1", title="Title two"),
                ],
            )

        with pytest.raises(LlmProcessingError, match="empty section title"):
            validated_section_titles(report, [ReportSectionUpdate(id="section-1", title="   ")])

    def test_schema_label_covers_known_and_fallback_models(self) -> None:
        class SectionTextResponse(BaseModel):
            value: str

        class OtherResponse(BaseModel):
            value: str

        assert schema_label(SectionTextResponse) == "Section polish"
        assert schema_label(ReportPolishResponse) == "Report polish"
        assert schema_label(OtherResponse) == "Structured LLM"

    def test_extract_response_metadata_skips_cycles(self) -> None:
        filter_results: dict[str, object] = {}
        filter_results["self"] = filter_results
        choice = SimpleNamespace(content_filter_results=filter_results)
        response = SimpleNamespace(choices=[choice])

        assert extract_response_metadata(response) == {}
