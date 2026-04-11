"""Tests for optional cloud LLM helpers."""

import json

import pytest

from webinar_transcriber.llm import (
    AnthropicLLMProcessor,
    LLMConfigurationError,
    LLMProcessingError,
    LLMReportPolishResult,
    OpenAILLMProcessor,
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
    build_llm_processor_from_env,
)
from webinar_transcriber.models import (
    MediaType,
    ReportDocument,
    ReportSection,
)


class TestBuildLlmProcessorFromEnv:
    def test_requires_api_key_and_model(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        with pytest.raises(LLMConfigurationError):
            build_llm_processor_from_env()

    def test_supports_anthropic(self, monkeypatch) -> None:
        fake_client = TestAnthropicLlmProcessor.FakeClient([])
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
        monkeypatch.setattr(
            "webinar_transcriber.llm.anthropic_backend.anthropic.Anthropic",
            lambda api_key: fake_client,
        )

        processor = build_llm_processor_from_env()

        assert isinstance(processor, AnthropicLLMProcessor)
        assert processor.provider_name == "anthropic"
        assert processor.model_name == "claude-test"

    def test_rejects_unknown_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "unknown")

        with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
            build_llm_processor_from_env()


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
                        ReportSectionUpdate(id="section-1", title="Improved overview"),
                    ],
                ),
                {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.openai.OpenAI",
            lambda api_key: fake_client,
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

        result = processor.polish_report(report)

        assert isinstance(result, LLMReportPolishResult)
        assert result.summary == ["Improved summary."]
        assert result.action_items == ["Send the updated draft by Friday."]
        assert result.section_titles == {"section-1": "Improved overview"}
        assert result.section_tldrs == {"section-1": "Short recap of the section."}
        assert result.section_transcripts == {
            "section-1": "Agenda review and project status update.\n\nPlease listen."
        }
        assert result.usage == {"input_tokens": 17, "output_tokens": 12, "total_tokens": 29}

    def test_rejects_unknown_report_section_id(self, monkeypatch) -> None:
        fake_client = self.FakeClient([
            self.FakeResponse(
                output_parsed=SectionTextResponse(
                    tldr="Agenda recap.",
                    transcript_text="Agenda review and project status update.",
                ),
                usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            ),
            self.FakeResponse(
                ReportPolishResponse(
                    section_updates=[
                        ReportSectionUpdate(id="section-x", title="Unexpected title"),
                    ]
                ),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.openai.OpenAI",
            lambda api_key: fake_client,
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

        with pytest.raises(LLMProcessingError):
            processor.polish_report(report)

    def test_skips_interlude_section_text_polish(self, monkeypatch) -> None:
        progress_updates: list[int] = []
        fake_client = self.FakeClient([
            self.FakeResponse(
                output_parsed=SectionTextResponse(
                    tldr="Agenda recap.",
                    transcript_text="Agenda review and project status update.",
                ),
                usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.openai_backend.openai.OpenAI",
            lambda api_key: fake_client,
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
            report,
            progress_callback=lambda advance: progress_updates.append(advance),
        )

        assert len(fake_client.responses.calls) == 1
        assert result.section_tldrs == {"section-2": "Agenda recap."}
        assert result.section_transcripts["section-1"] == report.sections[0].transcript_text
        assert result.section_transcripts["section-2"] == "Agenda review and project status update."
        assert result.warnings == [
            "Skipped LLM section polish for likely music/interlude section section-1."
        ]
        assert progress_updates == [1, 1]


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
                    "section_updates": [
                        {"id": "section-1", "title": "Improved overview"},
                    ],
                }),
                usage=type("Usage", (), {"input_tokens": 12, "output_tokens": 8})(),
            ),
        ])
        monkeypatch.setattr(
            "webinar_transcriber.llm.anthropic_backend.anthropic.Anthropic",
            lambda api_key: fake_client,
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

        result = processor.polish_report(report)

        assert isinstance(result, LLMReportPolishResult)
        assert result.summary == ["Improved summary."]
        assert result.action_items == ["Send the updated draft by Friday."]
        assert result.section_titles == {"section-1": "Improved overview"}
        assert result.section_tldrs == {"section-1": "Short recap of the section."}
        assert result.section_transcripts == {
            "section-1": "Agenda review and project status update.\n\nPlease listen."
        }
        assert result.usage == {"input_tokens": 17, "output_tokens": 12, "total_tokens": 29}
