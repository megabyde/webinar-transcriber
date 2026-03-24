"""Tests for optional cloud LLM helpers."""

import pytest

from webinar_transcriber.llm import (
    LLMConfigurationError,
    LLMProcessingError,
    LLMReportPolishResult,
    LLMTranscriptPolishResult,
    OpenAILLMProcessor,
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
    TranscriptPolishBatchResponse,
    TranscriptPolishItem,
    build_llm_processor_from_env,
)
from webinar_transcriber.models import (
    MediaType,
    ReportDocument,
    ReportSection,
    TranscriptionResult,
    TranscriptSegment,
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


def test_openai_llm_processor_polishes_transcript_in_batches(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[
                        TranscriptPolishItem(
                            id="segment-1",
                            text="Hello, everyone...",
                        ),
                        TranscriptPolishItem(
                            id="segment-2",
                            text="We are starting the project review.\n\nPlease pay attention.",
                        ),
                    ]
                ),
                {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            ),
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[
                        TranscriptPolishItem(
                            id="segment-3",
                            text="Please send the updated draft by Friday.",
                        )
                    ]
                ),
                {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            ),
        ]
    )
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
    )

    processor = OpenAILLMProcessor(
        api_key="test-key",
        model_name="gpt-test",
        transcript_char_budget=70,
        transcript_max_workers=2,
    )
    transcription = TranscriptionResult(
        detected_language="en",
        segments=[
            TranscriptSegment(id="segment-1", text="hello everyone", start_sec=0.0, end_sec=2.0),
            TranscriptSegment(
                id="segment-2",
                text="we are starting the project review",
                start_sec=2.0,
                end_sec=5.0,
            ),
            TranscriptSegment(
                id="segment-3",
                text="please send the updated draft by friday",
                start_sec=5.0,
                end_sec=8.0,
            ),
        ],
    )

    result = processor.polish_transcript(transcription)

    assert isinstance(result, LLMTranscriptPolishResult)
    assert [segment.text for segment in result.transcription.segments] == [
        "Hello, everyone.",
        "We are starting the project review.\n\nPlease pay attention.",
        "Please send the updated draft by Friday.",
    ]
    assert result.transcription.segments[2].start_sec == 5.0
    assert result.usage == {"input_tokens": 16, "output_tokens": 11, "total_tokens": 27}
    assert len(fake_client.responses.calls) == 2


def test_openai_llm_processor_caps_transcript_batch_size(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[
                        TranscriptPolishItem(id="segment-1", text="One."),
                        TranscriptPolishItem(id="segment-2", text="Two."),
                    ]
                ),
                {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            ),
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[TranscriptPolishItem(id="segment-3", text="Three.")]
                ),
                {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            ),
        ]
    )
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
    )

    processor = OpenAILLMProcessor(
        api_key="test-key",
        model_name="gpt-test",
        transcript_char_budget=1_000,
        transcript_max_batch_segments=2,
        transcript_max_workers=2,
    )
    transcription = TranscriptionResult(
        segments=[
            TranscriptSegment(id="segment-1", text="one", start_sec=0.0, end_sec=1.0),
            TranscriptSegment(id="segment-2", text="two", start_sec=1.0, end_sec=2.0),
            TranscriptSegment(id="segment-3", text="three", start_sec=2.0, end_sec=3.0),
        ]
    )

    result = processor.polish_transcript(transcription)

    assert [segment.text for segment in result.transcription.segments] == ["One.", "Two.", "Three."]
    assert len(fake_client.responses.calls) == 2


def test_openai_llm_processor_falls_back_for_mismatched_segment_ids(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[TranscriptPolishItem(id="segment-x", text="Changed text.")]
                ),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        ]
    )
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
    )

    processor = OpenAILLMProcessor(api_key="test-key", model_name="gpt-test")
    transcription = TranscriptionResult(
        segments=[TranscriptSegment(id="segment-1", text="hello world", start_sec=0.0, end_sec=1.0)]
    )

    result = processor.polish_transcript(transcription)

    assert result.transcription.segments[0].text == "hello world"
    assert result.warnings == ["Transcript polish response changed the segment IDs or order."]


def test_openai_llm_processor_keeps_original_text_for_empty_segment(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[TranscriptPolishItem(id="segment-1", text="   ")]
                ),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        ]
    )
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
    )

    processor = OpenAILLMProcessor(api_key="test-key", model_name="gpt-test")
    transcription = TranscriptionResult(
        segments=[TranscriptSegment(id="segment-1", text="hello world", start_sec=0.0, end_sec=1.0)]
    )

    result = processor.polish_transcript(transcription)

    assert result.transcription.segments[0].text == "hello world"
    assert result.warnings == [
        (
            "Transcript polish response returned an empty segment text "
            "for segment-1; kept original text."
        )
    ]


def test_openai_llm_processor_falls_back_for_invalid_batch(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                TranscriptPolishBatchResponse(
                    segments=[TranscriptPolishItem(id="segment-x", text="Changed text.")]
                ),
                {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        ]
    )
    monkeypatch.setattr(
        "webinar_transcriber.llm._build_openai_client", lambda _api_key: fake_client
    )

    processor = OpenAILLMProcessor(
        api_key="test-key",
        model_name="gpt-test",
        transcript_char_budget=1_000,
        transcript_max_batch_segments=10,
    )
    transcription = TranscriptionResult(
        segments=[TranscriptSegment(id="segment-1", text="hello world", start_sec=0.0, end_sec=1.0)]
    )

    result = processor.polish_transcript(transcription)

    assert result.transcription.segments[0].text == "hello world"
    assert result.warnings == ["Transcript polish response changed the segment IDs or order."]


def test_openai_llm_processor_polishes_report(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                output_parsed=SectionTextResponse(
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
        ]
    )
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
    assert result.section_transcripts == {
        "section-1": "Agenda review and project status update.\n\nPlease listen."
    }
    assert result.usage == {"input_tokens": 17, "output_tokens": 12, "total_tokens": 29}


def test_openai_llm_processor_rejects_unknown_report_section_id(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                output_parsed=SectionTextResponse(
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
        ]
    )
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
