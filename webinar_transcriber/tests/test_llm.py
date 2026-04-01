"""Tests for optional cloud LLM helpers."""

import pytest

from webinar_transcriber.llm import (
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
        self.responses = FakeResponsesAPI(responses)


def test_build_llm_processor_from_env_requires_api_key_and_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(LLMConfigurationError):
        build_llm_processor_from_env()


def test_openai_llm_processor_polishes_report(monkeypatch) -> None:
    fake_client = FakeClient([
        FakeResponse(
            output_parsed=SectionTextResponse(
                tldr="Short recap of the section.",
                transcript_text="Agenda review and project status update.\n\nPlease listen."
            ),
            usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        ),
        FakeResponse(
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
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
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


def test_openai_llm_processor_rejects_unknown_report_section_id(monkeypatch) -> None:
    fake_client = FakeClient([
        FakeResponse(
            output_parsed=SectionTextResponse(
                tldr="Agenda recap.",
                transcript_text="Agenda review and project status update."
            ),
            usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        ),
        FakeResponse(
            ReportPolishResponse(
                section_updates=[
                    ReportSectionUpdate(id="section-x", title="Unexpected title"),
                ]
            ),
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        ),
    ])
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
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


def test_openai_llm_processor_skips_interlude_section_text_polish(monkeypatch) -> None:
    progress_updates: list[int] = []
    fake_client = FakeClient([
        FakeResponse(
            output_parsed=SectionTextResponse(
                tldr="Agenda recap.",
                transcript_text="Agenda review and project status update."
            ),
            usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        ),
    ])
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
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
