"""Tests for optional cloud LLM helpers."""

import json
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import BaseModel

from webinar_transcriber.llm import (
    AnthropicLLMProcessor,
    LLMConfigurationError,
    LLMProcessingError,
    OpenAILLMProcessor,
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
    build_llm_processor_from_env,
)
from webinar_transcriber.llm.flow import _BaseLLMProcessor
from webinar_transcriber.llm.utils import (
    anthropic_response_text,
    build_report_polish_payload,
    extract_json_text,
    extract_usage,
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_report_lines,
    schema_label,
    truncate_text,
    validated_section_titles,
)
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection


def _fake_openai_module(fake_client):
    return SimpleNamespace(OpenAI=lambda **_kwargs: fake_client)


def _fake_anthropic_module(fake_client):
    return SimpleNamespace(Anthropic=lambda **_kwargs: fake_client)


LLM_EXTRA_INSTALL_RE = r'uv tool install --reinstall "\.\[llm\]"'


class TestBuildLlmProcessorFromEnv:
    def test_requires_api_key_and_model(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module",
            lambda _name: object(),
        )

        with pytest.raises(LLMConfigurationError):
            build_llm_processor_from_env()

    def test_requires_llm_extra_for_openai(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module",
            lambda _name: (_ for _ in ()).throw(ImportError("missing openai")),
        )

        with pytest.raises(
            LLMConfigurationError,
            match=rf"The OpenAI provider requires the 'llm' extra\..*{LLM_EXTRA_INSTALL_RE}",
        ):
            build_llm_processor_from_env()

    def test_supports_anthropic(self, monkeypatch) -> None:
        fake_client = TestAnthropicLlmProcessor.FakeClient([])
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module",
            lambda _name: _fake_anthropic_module(fake_client),
        )
        monkeypatch.setattr(
            "webinar_transcriber.llm.anthropic_backend.importlib.import_module",
            lambda _name: _fake_anthropic_module(fake_client),
        )

        processor = build_llm_processor_from_env()

        assert isinstance(processor, AnthropicLLMProcessor)
        assert processor.provider_name == "anthropic"
        assert processor.model_name == "claude-test"

    def test_requires_llm_extra_for_anthropic(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module",
            lambda _name: (_ for _ in ()).throw(ImportError("missing anthropic")),
        )

        with pytest.raises(
            LLMConfigurationError,
            match=rf"The Anthropic provider requires the 'llm' extra\..*{LLM_EXTRA_INSTALL_RE}",
        ):
            build_llm_processor_from_env()

    def test_rejects_unknown_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "unknown")

        with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
            build_llm_processor_from_env()

    def test_defaults_to_openai(self, monkeypatch) -> None:
        fake_client = TestOpenAiLlmProcessor.FakeClient([])
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module",
            lambda _name: _fake_openai_module(fake_client),
        )
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.importlib.import_module",
            lambda _name: _fake_openai_module(fake_client),
        )

        processor = build_llm_processor_from_env()

        assert isinstance(processor, OpenAILLMProcessor)
        assert processor.provider_name == "openai"
        assert processor.model_name == "gpt-test"


class TestOpenAiLlmProcessor:
    class FakeResponse:
        """Simple stand-in for the OpenAI SDK response object."""

        def __init__(self, output_parsed, usage) -> None:
            self.output_parsed = output_parsed
            self.usage = usage

    class FakeResponsesAPI:
        """Capture parse calls and replay canned responses."""

        def __init__(self, responses) -> None:
            self.calls = []
            self._responses = list(responses)

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return self._responses.pop(0)

    class FakeClient:
        """Container exposing a fake responses API surface."""

        def __init__(self, responses) -> None:
            self.responses = TestOpenAiLlmProcessor.FakeResponsesAPI(responses)

    def test_polishes_report(self, monkeypatch) -> None:
        fake_client = self.FakeClient([
            self.FakeResponse(
                output_parsed=SectionTextResponse(
                    tldr="Short recap of the section.",
                    transcript_text="Agenda review and project status update.\n\nPlease listen.",
                ),
                usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            ),
            self.FakeResponse(
                ReportPolishResponse(
                    summary=["Improved summary."],
                    action_items=["Send the updated draft by Friday."],
                    section_updates=[
                        ReportSectionUpdate(id="section-1", title="Improved overview")
                    ],
                ),
                {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.importlib.import_module",
            lambda _name: _fake_openai_module(fake_client),
        )

        processor = OpenAILLMProcessor(api_key="test-key", model_name="gpt-test")
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            summary=["Old summary."],
            action_items=["Old action."],
            sections=[
                ReportSection(
                    id="section-1",
                    title="Old title",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text="Agenda review and project status update.",
                )
            ],
        )

        section_result = processor.polish_report_sections_with_progress(report)
        metadata_result = processor.polish_report_metadata(
            report, section_transcripts=section_result.section_transcripts
        )

        assert metadata_result.summary == ["Improved summary."]
        assert metadata_result.action_items == ["Send the updated draft by Friday."]
        assert metadata_result.section_titles == {"section-1": "Improved overview"}
        assert section_result.section_tldrs == {"section-1": "Short recap of the section."}
        assert section_result.section_transcripts == {
            "section-1": "Agenda review and project status update.\n\nPlease listen."
        }
        assert section_result.usage == {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
        assert metadata_result.usage == {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        }
        assert report.sections[0].transcript_text == "Agenda review and project status update."

    def test_build_report_polish_payload_uses_section_transcript_overrides(self) -> None:
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Agenda",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text="Original transcript text.",
                )
            ],
        )

        payload = build_report_polish_payload(
            report,
            total_char_budget=1_000,
            section_transcripts={"section-1": "Overridden transcript text."},
        )
        sections = cast("list[dict[str, object]]", payload["sections"])

        assert sections[0]["transcript_excerpt"] == "Overridden transcript text."
        assert report.sections[0].transcript_text == "Original transcript text."

    def test_rejects_unknown_report_section_id(self, monkeypatch) -> None:
        fake_client = self.FakeClient([
            self.FakeResponse(
                output_parsed=SectionTextResponse(
                    tldr="Agenda recap.", transcript_text="Agenda review and project status update."
                ),
                usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            ),
            self.FakeResponse(
                ReportPolishResponse(
                    section_updates=[ReportSectionUpdate(id="section-x", title="Unexpected title")]
                ),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.importlib.import_module",
            lambda _name: _fake_openai_module(fake_client),
        )

        processor = OpenAILLMProcessor(api_key="test-key", model_name="gpt-test")
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
                    transcript_text="Agenda review and project status update.",
                )
            ],
        )

        section_result = processor.polish_report_sections_with_progress(report)

        with pytest.raises(LLMProcessingError):
            processor.polish_report_metadata(
                report, section_transcripts=section_result.section_transcripts
            )

    def test_rejects_non_matching_section_schema(self, monkeypatch) -> None:
        fake_client = self.FakeClient([
            self.FakeResponse(
                output_parsed=object(),
                usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            )
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.importlib.import_module",
            lambda _name: _fake_openai_module(fake_client),
        )

        processor = OpenAILLMProcessor(api_key="test-key", model_name="gpt-test")
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
                    transcript_text="Agenda review and project status update.",
                )
            ],
        )

        with pytest.raises(LLMProcessingError, match="Report polish response did not match"):
            processor.polish_report_metadata(
                report,
                section_transcripts={"section-1": "Agenda review and project status update."},
            )

    def test_skips_interlude_section_text_polish(self, monkeypatch) -> None:
        progress_updates: list[int] = []
        fake_client = self.FakeClient([
            self.FakeResponse(
                output_parsed=SectionTextResponse(
                    tldr="Agenda recap.", transcript_text="Agenda review and project status update."
                ),
                usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            )
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.importlib.import_module",
            lambda _name: _fake_openai_module(fake_client),
        )

        processor = OpenAILLMProcessor(api_key="test-key", model_name="gpt-test")
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Music Interlude",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text=(
                        "Music interlude. The raw transcript is preserved in transcript.json."
                    ),
                    is_interlude=True,
                ),
                ReportSection(
                    id="section-2",
                    title="Old title",
                    start_sec=10.0,
                    end_sec=20.0,
                    transcript_text="Agenda review and project status update.",
                ),
            ],
        )

        result = processor.polish_report_sections_with_progress(
            report, progress_callback=progress_updates.append
        )

        assert len(fake_client.responses.calls) == 1
        assert result.section_tldrs == {"section-2": "Agenda recap."}
        assert result.section_transcripts["section-1"] == report.sections[0].transcript_text
        assert result.section_transcripts["section-2"] == "Agenda review and project status update."
        assert result.warnings == [
            "Skipped LLM section polish for likely music/interlude section section-1."
        ]
        assert progress_updates == [1]


class TestAnthropicLlmProcessor:
    class FakeContentBlock:
        """Simple Anthropic text content block."""

        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class FakeResponse:
        """Simple stand-in for the Anthropic SDK response object."""

        def __init__(self, text: str, usage) -> None:
            self.content = [TestAnthropicLlmProcessor.FakeContentBlock(text)]
            self.usage = usage

    class FakeMessagesAPI:
        """Capture create calls and replay canned responses."""

        def __init__(self, responses) -> None:
            self.calls = []
            self._responses = list(responses)

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return self._responses.pop(0)

    class FakeClient:
        """Container exposing a fake Anthropic messages API surface."""

        def __init__(self, responses) -> None:
            self.messages = TestAnthropicLlmProcessor.FakeMessagesAPI(responses)

    def test_polishes_report(self, monkeypatch) -> None:
        fake_client = self.FakeClient([
            self.FakeResponse(
                json.dumps({
                    "tldr": "Short recap of the section.",
                    "transcript_text": "Agenda review and project status update.\n\nPlease listen.",
                }),
                usage=type("Usage", (), {"input_tokens": 5, "output_tokens": 4})(),
            ),
            self.FakeResponse(
                json.dumps({
                    "summary": ["Improved summary."],
                    "action_items": ["Send the updated draft by Friday."],
                    "section_updates": [{"id": "section-1", "title": "Improved overview"}],
                }),
                usage=type("Usage", (), {"input_tokens": 12, "output_tokens": 8})(),
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.anthropic_backend.importlib.import_module",
            lambda _name: _fake_anthropic_module(fake_client),
        )

        processor = AnthropicLLMProcessor(api_key="test-key", model_name="claude-test")
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            summary=["Old summary."],
            action_items=["Old action."],
            sections=[
                ReportSection(
                    id="section-1",
                    title="Old title",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text="Agenda review and project status update.",
                )
            ],
        )

        section_result = processor.polish_report_sections_with_progress(report)
        metadata_result = processor.polish_report_metadata(
            report, section_transcripts=section_result.section_transcripts
        )

        assert metadata_result.summary == ["Improved summary."]
        assert metadata_result.action_items == ["Send the updated draft by Friday."]
        assert metadata_result.section_titles == {"section-1": "Improved overview"}
        assert section_result.section_tldrs == {"section-1": "Short recap of the section."}
        assert section_result.section_transcripts == {
            "section-1": "Agenda review and project status update.\n\nPlease listen."
        }
        assert section_result.usage == {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
        assert metadata_result.usage == {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        }

    def test_rejects_invalid_json_response(self, monkeypatch) -> None:
        fake_client = self.FakeClient([
            self.FakeResponse(
                "[]",
                usage=type("Usage", (), {"input_tokens": 5, "output_tokens": 4})(),
            )
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.anthropic_backend.importlib.import_module",
            lambda _name: _fake_anthropic_module(fake_client),
        )

        processor = AnthropicLLMProcessor(api_key="test-key", model_name="claude-test")
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
                    transcript_text="Agenda review and project status update.",
                )
            ],
        )

        with pytest.raises(LLMProcessingError, match="Report polish response did not match"):
            processor.polish_report_metadata(
                report,
                section_transcripts={"section-1": "Agenda review and project status update."},
            )


class TestLlmFlow:
    class StubProcessor(_BaseLLMProcessor):
        def __init__(
            self, responses: dict[str, tuple[BaseModel, dict[str, int]] | Exception]
        ) -> None:
            super().__init__(provider_name="stub", model_name="stub-model")
            self._responses = responses

        def _parse_structured_response(self, **kwargs) -> tuple[BaseModel, dict[str, int]]:
            section_id = cast("str", kwargs["user_payload"].get("id"))
            response = self._responses[section_id]
            if isinstance(response, Exception):
                raise response
            return response

    def test_returns_empty_section_result_for_report_without_sections(self) -> None:
        processor = self.StubProcessor({})
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        result = processor.polish_report_sections_with_progress(report)

        assert result.section_transcripts == {}
        assert result.section_tldrs == {}
        assert result.usage == {}
        assert result.warnings == []

    def test_keeps_interlude_tldr_during_section_polish(self) -> None:
        processor = self.StubProcessor({
            "section-2": (
                SectionTextResponse(tldr="Agenda recap.", transcript_text="Agenda review."),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        })
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Music Interlude",
                    start_sec=0.0,
                    end_sec=5.0,
                    transcript_text="Interlude placeholder.",
                    tldr="Keep me.",
                    is_interlude=True,
                ),
                ReportSection(
                    id="section-2",
                    title="Agenda",
                    start_sec=5.0,
                    end_sec=12.0,
                    transcript_text="Agenda review.",
                ),
            ],
        )

        result = processor.polish_report_sections_with_progress(report)

        assert result.section_transcripts["section-1"] == "Interlude placeholder."
        assert result.section_tldrs == {"section-1": "Keep me.", "section-2": "Agenda recap."}
        assert result.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    def test_turns_section_polish_errors_into_warnings(self) -> None:
        processor = self.StubProcessor({
            "section-1": LLMProcessingError("bad section"),
        })
        progress_updates: list[int] = []
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Agenda",
                    start_sec=0.0,
                    end_sec=12.0,
                    transcript_text="Agenda review.",
                    tldr="Existing recap.",
                )
            ],
        )

        result = processor.polish_report_sections_with_progress(
            report, progress_callback=progress_updates.append
        )

        assert result.section_transcripts == {"section-1": "Agenda review."}
        assert result.section_tldrs == {"section-1": "Existing recap."}
        assert result.usage == {}
        assert result.warnings == ["bad section"]
        assert progress_updates == [1]

    def test_warns_when_section_polish_returns_empty_transcript(self) -> None:
        processor = self.StubProcessor({
            "section-1": (
                SectionTextResponse(tldr="Recap.", transcript_text="   "),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        })
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Agenda",
                    start_sec=0.0,
                    end_sec=12.0,
                    transcript_text="Agenda review.",
                )
            ],
        )

        result = processor.polish_report_sections_with_progress(report)

        assert result.section_transcripts == {"section-1": "Agenda review."}
        assert result.section_tldrs == {"section-1": "Recap."}
        assert result.warnings == [
            "Section polish response returned an empty transcript text for section-1; "
            "kept original text."
        ]


class TestLlmNormalization:
    def test_normalize_polished_section_text_collapses_paragraph_whitespace(self) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence.",
            polished_text="  First   line \nwith spacing.\n\n  Second\tparagraph.  ",
            section_id="section-1",
        )

        assert normalized == "First line with spacing.\n\nSecond paragraph."

    def test_normalize_polished_section_text_rewrites_trailing_ellipsis_when_source_is_final(
        self,
    ) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence.",
            polished_text="Rewritten sentence...",
            section_id="section-1",
        )

        assert normalized == "Rewritten sentence."

    def test_normalize_polished_section_text_preserves_ellipsis_for_incomplete_source(self) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence...",
            polished_text="Rewritten sentence...",
            section_id="section-1",
        )

        assert normalized == "Rewritten sentence..."

    def test_normalize_polished_section_text_keeps_original_when_polished_text_is_blank(
        self,
    ) -> None:
        normalized = normalize_polished_section_text(
            original_text="Original sentence.",
            polished_text="   ",
            section_id="section-1",
        )

        assert normalized == "Original sentence."

    def test_normalize_polished_section_text_rejects_suspiciously_short_output(self) -> None:
        with pytest.raises(LLMProcessingError, match="looked truncated for section-1"):
            normalize_polished_section_text(
                original_text="Long original text. " * 8,
                polished_text="Too short.",
                section_id="section-1",
            )

    def test_normalize_polished_section_tldr_uses_same_paragraph_normalization(self) -> None:
        normalized = normalize_polished_section_tldr(
            "  Bullet one \nspans whitespace.\n\n  Bullet two stays.  "
        )

        assert normalized == "Bullet one spans whitespace.\n\nBullet two stays."

    def test_normalize_polished_section_tldr_returns_blank_for_blank_input(self) -> None:
        assert normalize_polished_section_tldr("   ") == ""

    def test_normalize_polished_section_tldr_splits_inline_bullet_items(self) -> None:
        normalized = normalize_polished_section_tldr(
            "- First point. - Second point. - Third point."
        )

        assert normalized == "- First point.\n\n- Second point.\n\n- Third point."

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

        with pytest.raises(LLMProcessingError, match="unknown section ID"):
            validated_section_titles(
                report, [ReportSectionUpdate(id="section-x", title="Unexpected title")]
            )

        with pytest.raises(LLMProcessingError, match="duplicate section IDs"):
            validated_section_titles(
                report,
                [
                    ReportSectionUpdate(id="section-1", title="Title one"),
                    ReportSectionUpdate(id="section-1", title="Title two"),
                ],
            )

        with pytest.raises(LLMProcessingError, match="empty section title"):
            validated_section_titles(report, [ReportSectionUpdate(id="section-1", title="   ")])

    def test_extract_usage_supports_dict_and_object_shapes(self) -> None:
        assert extract_usage(type("Response", (), {})()) == {}
        assert extract_usage({"usage": "ignored"}) == {}
        assert extract_usage(
            type("Response", (), {"usage": {"input_tokens": 2, "other": 1}})()
        ) == {"input_tokens": 2}

        usage_obj = type("Usage", (), {"input_tokens": 3, "output_tokens": 4})()
        response_obj = type("Response", (), {"usage": usage_obj})()
        assert extract_usage(response_obj) == {
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        }

        usage_with_total = type("Usage", (), {"input_tokens": 3, "total_tokens": 9})()
        assert extract_usage(type("Response", (), {"usage": usage_with_total})()) == {
            "input_tokens": 3,
            "total_tokens": 9,
        }

    def test_extract_json_text_supports_fenced_and_embedded_json(self) -> None:
        assert extract_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert extract_json_text('prefix {"a": 1} suffix') == '{"a": 1}'
        assert extract_json_text("  plain text  ") == "plain text"

    def test_anthropic_response_text_requires_text_content(self) -> None:
        response = type("Response", (), {"content": [type("Block", (), {"type": "image"})()]})()

        with pytest.raises(LLMProcessingError, match="did not contain text content"):
            anthropic_response_text(response)

        with pytest.raises(LLMProcessingError, match="did not contain text content"):
            anthropic_response_text(type("Response", (), {"content": "invalid"})())

    def test_schema_label_covers_known_and_fallback_models(self) -> None:
        class OtherResponse(BaseModel):
            value: str

        assert schema_label(SectionTextResponse) == "Section polish"
        assert schema_label(ReportPolishResponse) == "Report polish"
        assert schema_label(OtherResponse) == "Structured LLM"
