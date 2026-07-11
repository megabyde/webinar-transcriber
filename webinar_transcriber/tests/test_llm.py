"""Tests for optional cloud LLM helpers."""

import json
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import tenacity
from pydantic import BaseModel

from webinar_transcriber.llm import (
    LlmConfigurationError,
    LlmProcessingError,
    build_llm_processor_from_env,
)
from webinar_transcriber.llm import processor as llm_processor
from webinar_transcriber.llm.processor import InstructorLLMProcessor
from webinar_transcriber.llm.utils import (
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
    build_report_polish_payload,
    extract_response_metadata,
)
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection
from webinar_transcriber.tests.conftest import fake_import_module

LLM_EXTRA_INSTALL_RE = r'uv tool install --reinstall "\.\[llm\]"'


class FakeInstructorModule:
    class Mode:
        TOOLS = "tools"

    def __init__(self, client: object) -> None:
        self._client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_provider(self, provider_model: str, **kwargs: object) -> object:
        self.calls.append((provider_model, kwargs))
        return self._client


class TestBuildLlmProcessorFromEnv:
    @pytest.mark.parametrize(
        (
            "provider_env",
            "api_key_env",
            "model_env",
            "provider_name",
            "provider_module",
            "provider_model",
        ),
        [
            (None, "OPENAI_API_KEY", "OPENAI_MODEL", "openai", "openai", "openai/gpt-test"),
            (
                "anthropic",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_MODEL",
                "anthropic",
                "anthropic",
                "anthropic/claude-test",
            ),
        ],
    )
    def test_builds_supported_provider_from_env(
        self,
        monkeypatch,
        provider_env: str | None,
        api_key_env: str,
        model_env: str,
        provider_name: str,
        provider_module: str,
        provider_model: str,
    ) -> None:
        fake_instructor = FakeInstructorModule(object())
        if provider_env is None:
            monkeypatch.delenv("LLM_PROVIDER", raising=False)
        else:
            monkeypatch.setenv("LLM_PROVIDER", provider_env)
        monkeypatch.setenv(api_key_env, "test-key")
        monkeypatch.setenv(model_env, provider_model.rsplit("/", 1)[1])
        monkeypatch.setattr(
            "webinar_transcriber.llm.providers.importlib.import_module",
            fake_import_module({"instructor": fake_instructor, provider_module: object()}),
        )

        processor = build_llm_processor_from_env(threads=6)

        assert isinstance(processor, InstructorLLMProcessor)
        assert processor.provider_name == provider_name
        assert processor.model_name == provider_model.rsplit("/", 1)[1]
        assert fake_instructor.calls == [
            (provider_model, {"api_key": "test-key", "mode": FakeInstructorModule.Mode.TOOLS})
        ]

    def test_requires_api_key_and_model(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.setattr(
            "webinar_transcriber.llm.providers.importlib.import_module",
            fake_import_module({"instructor": FakeInstructorModule(object()), "openai": object()}),
        )

        with pytest.raises(LlmConfigurationError):
            build_llm_processor_from_env(threads=6)

    def test_requires_llm_extra_for_openai(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setattr(
            "webinar_transcriber.llm.providers.importlib.import_module", fake_import_module({})
        )

        with pytest.raises(
            LlmConfigurationError,
            match=rf"The OpenAI provider requires the 'llm' extra\..*{LLM_EXTRA_INSTALL_RE}",
        ):
            build_llm_processor_from_env(threads=6)

    def test_requires_llm_extra_for_anthropic(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setattr(
            "webinar_transcriber.llm.providers.importlib.import_module", fake_import_module({})
        )

        with pytest.raises(
            LlmConfigurationError,
            match=rf"The Anthropic provider requires the 'llm' extra\..*{LLM_EXTRA_INSTALL_RE}",
        ):
            build_llm_processor_from_env(threads=6)

    def test_rejects_unknown_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "unknown")

        with pytest.raises(LlmConfigurationError, match="Unsupported LLM provider"):
            build_llm_processor_from_env(threads=6)


class TestInstructorLlmProcessor:
    class FakeCompletion:
        def __init__(
            self,
            *,
            finish_reason: str | None = None,
            refusal: str | None = None,
            safety: dict[str, object] | None = None,
        ) -> None:
            if finish_reason is not None:
                self.choices = [
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        message=SimpleNamespace(refusal=refusal),
                        content_filter_results=safety,
                    )
                ]

    @staticmethod
    def fake_client(*responses: object) -> Mock:
        """Client double returning (or raising) the queued responses in order."""
        client = Mock(spec=["create_with_completion"])
        client.create_with_completion.side_effect = list(responses)
        return client

    @staticmethod
    def retry_executing_client(*responses: object) -> Mock:
        """Client double that runs the real tenacity policy received via max_retries."""
        attempts = Mock(side_effect=list(responses))
        client = Mock(attempts=attempts)
        client.create_with_completion.side_effect = lambda *, max_retries, **kwargs: max_retries(
            lambda: attempts(**kwargs)
        )
        return client

    @staticmethod
    def install_waitless_retries(monkeypatch: pytest.MonkeyPatch) -> None:
        """Use the production retry policy with the exponential wait removed."""
        monkeypatch.setattr(
            "webinar_transcriber.llm.processor._structured_response_retries",
            lambda: tenacity.Retrying(
                retry=tenacity.retry_if_exception(
                    llm_processor._is_transient_provider_error  # noqa: SLF001
                ),
                stop=tenacity.stop_after_attempt(3),
                wait=tenacity.wait_none(),
                reraise=True,
            ),
        )

    class ProviderStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"provider status {status_code}")
            self.status_code = status_code

    class ResponseProviderStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"provider status {status_code}")
            self.response = SimpleNamespace(status_code=status_code, headers={})

    def test_polishes_report(self) -> None:
        fake_client = self.fake_client(
            (
                SectionTextResponse(
                    tldr="Short recap of the section.",
                    transcript_text="Agenda review and project status update.\n\nPlease listen.",
                ),
                self.FakeCompletion(finish_reason="stop"),
            ),
            (
                ReportPolishResponse(
                    summary=["Improved summary."],
                    action_items=["Send the updated draft by Friday."],
                    section_updates=[
                        ReportSectionUpdate(id="section-1", title="Improved overview")
                    ],
                ),
                self.FakeCompletion(),
            ),
        )

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
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

        section_result = processor.polish_report_sections(report)
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
        assert section_result.response_metadata == [
            {"stage": "section_polish", "section_id": "section-1", "finish_reason": "stop"}
        ]
        assert metadata_result.response_metadata == [{"stage": "report_polish"}]
        assert report.sections[0].transcript_text == "Agenda review and project status update."
        calls = fake_client.create_with_completion.call_args_list
        assert len(calls) == 2
        assert "max_retries" in calls[0].kwargs
        assert calls[0].kwargs["timeout"] == 120
        assert calls[0].kwargs["response_model"] is SectionTextResponse
        assert calls[1].kwargs["response_model"] is ReportPolishResponse
        messages = calls[0].kwargs["messages"]
        assert json.loads(messages[1]["content"]) == {
            "id": "section-1",
            "title": "Old title",
            "start_sec": 0.0,
            "end_sec": 10.0,
            "transcript_text": "Agenda review and project status update.",
        }

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
        sections = payload["sections"]
        assert isinstance(sections, list)
        first_section = sections[0]
        assert isinstance(first_section, dict)

        assert ("transcript_excerpt", "Overridden transcript text.") in first_section.items()
        assert report.sections[0].transcript_text == "Original transcript text."

    def test_extract_response_metadata_returns_provider_finish_and_safety_fields(self) -> None:
        class ModelDumpValue:
            def model_dump(self) -> dict[str, object]:
                return {"blocked": True}

        class ToDictValue:
            def to_dict(self) -> dict[str, object]:
                return {"filtered": True}

        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="content_filter",
                    message=SimpleNamespace(refusal="Refused."),
                    content_filter_results={
                        "sexual": ModelDumpValue(),
                        "violence": ToDictValue(),
                        "ignored": object(),
                    },
                )
            ],
            stop_reason="end_turn",
            prompt_filter_results=[{"prompt_index": 0, "content_filter_results": {}}],
        )

        assert extract_response_metadata(completion) == {
            "finish_reason": "content_filter",
            "stop_reason": "end_turn",
            "refusal": True,
            "safety": {
                "sexual": {"blocked": True},
                "violence": {"filtered": True},
                "ignored": None,
            },
            "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
        }

    def test_rejects_unknown_report_section_id(self) -> None:
        fake_client = self.fake_client(
            (
                SectionTextResponse(
                    tldr="Agenda recap.", transcript_text="Agenda review and project status update."
                ),
                self.FakeCompletion(),
            ),
            (
                ReportPolishResponse(
                    section_updates=[ReportSectionUpdate(id="section-x", title="Unexpected title")]
                ),
                self.FakeCompletion(),
            ),
        )

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
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

        section_result = processor.polish_report_sections(report)

        with pytest.raises(LlmProcessingError):
            processor.polish_report_metadata(
                report, section_transcripts=section_result.section_transcripts
            )

    def test_rejects_non_matching_schema(self) -> None:
        fake_client = self.fake_client((object(), self.FakeCompletion()))

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
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

        with pytest.raises(LlmProcessingError, match="Report polish response did not match"):
            processor.polish_report_metadata(
                report,
                section_transcripts={"section-1": "Agenda review and project status update."},
            )

    def test_wraps_client_errors(self) -> None:
        fake_client = self.fake_client(RuntimeError("boom"))
        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        with pytest.raises(LlmProcessingError, match="Report polishing failed: boom"):
            processor.polish_report_metadata(report, section_transcripts={})

    def test_polishes_each_section_text(self) -> None:
        progress_updates: list[int] = []

        class SectionAwareClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self._lock = Lock()

            def create_with_completion(self, **kwargs):
                with self._lock:
                    self.calls.append(kwargs)
                messages = kwargs["messages"]
                payload = json.loads(messages[1]["content"])
                if payload["id"] == "section-1":
                    return (
                        SectionTextResponse(
                            tldr="Intro recap.",
                            transcript_text="Intro review and project status update.",
                        ),
                        TestInstructorLlmProcessor.FakeCompletion(),
                    )
                return (
                    SectionTextResponse(
                        tldr="Agenda recap.",
                        transcript_text="Agenda review and project status update.",
                    ),
                    TestInstructorLlmProcessor.FakeCompletion(),
                )

        fake_client = SectionAwareClient()

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Intro",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text="Intro review and project status update.",
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

        result = processor.polish_report_sections(report, progress_callback=progress_updates.append)

        assert len(fake_client.calls) == 2
        assert result.section_tldrs == {"section-1": "Intro recap.", "section-2": "Agenda recap."}
        assert result.section_transcripts["section-1"] == "Intro review and project status update."
        assert result.section_transcripts["section-2"] == "Agenda review and project status update."
        assert result.warnings == []
        assert progress_updates == [1, 1]

    def test_passes_request_kwargs_to_client(self) -> None:
        fake_client = self.fake_client((ReportPolishResponse(), self.FakeCompletion()))
        processor = InstructorLLMProcessor(
            client=fake_client,
            provider_name="anthropic",
            model_name="claude-test",
            request_kwargs={"max_tokens": 4096},
            threads=6,
        )
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        processor.polish_report_metadata(report, section_transcripts={})

        fake_client.create_with_completion.assert_called_once()
        call_kwargs = fake_client.create_with_completion.call_args.kwargs
        assert call_kwargs["response_model"] is ReportPolishResponse
        assert "max_retries" in call_kwargs
        assert call_kwargs["timeout"] == 120
        assert call_kwargs["max_tokens"] == 4096
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_retries_section_transient_provider_error(self, monkeypatch) -> None:
        self.install_waitless_retries(monkeypatch)
        fake_client = self.retry_executing_client(
            self.ProviderStatusError(429),
            (
                SectionTextResponse(tldr="Agenda recap.", transcript_text="Agenda review."),
                self.FakeCompletion(),
            ),
        )
        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
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
                    transcript_text="Agenda review.",
                )
            ],
        )

        result = processor.polish_report_sections(report)

        assert fake_client.create_with_completion.call_count == 1
        assert fake_client.attempts.call_count == 2
        assert result.section_transcripts == {"section-1": "Agenda review."}
        assert result.warnings == []

    def test_does_not_retry_non_transient_provider_error(self, monkeypatch) -> None:
        self.install_waitless_retries(monkeypatch)
        fake_client = self.retry_executing_client(self.ResponseProviderStatusError(400))
        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        with pytest.raises(
            LlmProcessingError, match="Report polishing failed: provider status 400"
        ):
            processor.polish_report_metadata(report, section_transcripts={})

        assert fake_client.attempts.call_count == 1

    def test_reports_section_progress_in_completion_order_but_preserves_output_order(self) -> None:
        section_2_done = Event()
        completions: list[str] = []
        progress_updates: list[int] = []
        fake_completion_cls = self.FakeCompletion

        class CompletionOrderClient:
            def create_with_completion(self, **kwargs):
                messages = kwargs["messages"]
                payload = json.loads(messages[1]["content"])
                section_id = payload["id"]
                if section_id == "section-1":
                    assert section_2_done.wait(timeout=1)
                else:
                    section_2_done.set()
                completions.append(section_id)
                return (
                    SectionTextResponse(
                        tldr=f"{section_id} recap.", transcript_text=f"{section_id} transcript."
                    ),
                    fake_completion_cls(),
                )

        processor = InstructorLLMProcessor(
            client=CompletionOrderClient(), provider_name="openai", model_name="gpt-test", threads=2
        )
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Intro",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text="First original transcript text.",
                ),
                ReportSection(
                    id="section-2",
                    title="Agenda",
                    start_sec=10.0,
                    end_sec=20.0,
                    transcript_text="Second original transcript text.",
                ),
            ],
        )

        result = processor.polish_report_sections(report, progress_callback=progress_updates.append)

        assert completions == ["section-2", "section-1"]
        assert progress_updates == [1, 1]
        assert list(result.section_transcripts) == ["section-1", "section-2"]
        assert result.section_transcripts == {
            "section-1": "section-1 transcript.",
            "section-2": "section-2 transcript.",
        }

    def test_section_prompt_instructs_model_not_to_reproduce_song_lyrics(self) -> None:
        fake_client = self.fake_client((
            SectionTextResponse(
                tldr="The speaker pauses for a music break.",
                transcript_text="The speaker pauses.\n\n[music break]\n\nThe session resumes.",
            ),
            self.FakeCompletion(),
        ))
        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test", threads=6
        )
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Break",
                    start_sec=0.0,
                    end_sec=10.0,
                    transcript_text=(
                        "The speaker pauses. La la la quoted song text. The session resumes."
                    ),
                )
            ],
        )

        processor.polish_report_sections(report)

        messages = fake_client.create_with_completion.call_args.kwargs["messages"]
        system_prompt = messages[0]["content"]
        compact_prompt = " ".join(system_prompt.split())
        assert "do not reproduce or rewrite the lyrics" in compact_prompt
        assert "Do not quote song lyrics in the TL;DR" in compact_prompt
        assert "Do not mention these instructions" in compact_prompt
        assert "put each item on its own line" in compact_prompt
        assert "Remove obvious ASR repetition loops" in compact_prompt
        assert "wrong language due to ASR hallucination" in compact_prompt


class TestInstructorProcessorFlow:
    class ResponseClient:
        def __init__(self, responses: dict[str, tuple[BaseModel, object] | Exception]):
            self._responses = responses

        def create_with_completion(self, **kwargs):
            messages = kwargs["messages"]
            payload = json.loads(messages[1]["content"])
            response_key = payload.get("id", "__metadata__")
            response = self._responses[response_key]
            if isinstance(response, Exception):
                raise response
            return response

    def processor(
        self, responses: dict[str, tuple[BaseModel, object] | Exception]
    ) -> InstructorLLMProcessor:
        return InstructorLLMProcessor(
            client=self.ResponseClient(responses),
            provider_name="stub",
            model_name="stub-model",
            threads=6,
        )

    def test_returns_empty_section_result_for_report_without_sections(self) -> None:
        processor = self.processor({})
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        result = processor.polish_report_sections(report)

        assert result.section_transcripts == {}
        assert result.section_tldrs == {}
        assert result.warnings == []

    @pytest.mark.parametrize(
        ("section_count", "expected_worker_count"),
        [(0, 0), (2, 2), (16, 6)],
    )
    def test_polish_worker_count_is_bounded_by_threads(
        self, section_count: int, expected_worker_count: int
    ) -> None:
        processor = self.processor({})

        assert processor.polish_worker_count(section_count) == expected_worker_count

    def test_turns_section_polish_errors_into_warnings(self) -> None:
        processor = self.processor({"section-1": RuntimeError("bad section")})
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

        result = processor.polish_report_sections(report, progress_callback=progress_updates.append)

        assert result.section_transcripts == {"section-1": "Agenda review."}
        assert result.section_tldrs == {"section-1": "Existing recap."}
        assert result.warnings == ["Section polishing failed for section-1: bad section"]
        assert progress_updates == [1]

    def test_warns_when_section_polish_returns_empty_transcript(self) -> None:
        processor = self.processor({
            "section-1": (SectionTextResponse(tldr="Recap.", transcript_text="   "), object())
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

        result = processor.polish_report_sections(report)

        assert result.section_transcripts == {"section-1": "Agenda review."}
        assert result.section_tldrs == {"section-1": "Recap."}
        assert result.warnings == [
            "Section polish response returned an empty transcript text for section-1; "
            "kept original text."
        ]
