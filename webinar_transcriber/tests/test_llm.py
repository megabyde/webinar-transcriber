"""Tests for optional cloud LLM helpers."""

import importlib
import json
from threading import Event
from typing import cast

import pytest
from pydantic import BaseModel

from webinar_transcriber.llm import (
    InstructorLLMProcessor,
    LLMConfigurationError,
    LLMProcessingError,
    LLMReportPolishPlan,
    build_llm_processor_from_env,
)
from webinar_transcriber.llm.schemas import (
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
)
from webinar_transcriber.llm.utils import (
    build_report_polish_payload,
    extract_usage,
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_report_lines,
    schema_label,
    truncate_text,
    validated_section_titles,
)
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection

LLM_EXTRA_INSTALL_RE = r'uv tool install --reinstall "\.\[llm\]"'


def _fake_import_module(modules: dict[str, object]):
    real_import_module = importlib.import_module

    def fake_import_module(name: str) -> object:
        try:
            return modules[name]
        except KeyError as error:
            if name.startswith("webinar_transcriber."):
                return real_import_module(name)
            raise ImportError(name) from error

    return fake_import_module


class FakeInstructorModule:
    class Mode:
        TOOLS = "tools"

    def __init__(self, client: object) -> None:
        self._client = client
        self.calls: list[tuple[str, dict[str, object]]] = []

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
            (
                None,
                "OPENAI_API_KEY",
                "OPENAI_MODEL",
                "openai",
                "openai",
                "openai/gpt-test",
            ),
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
            "webinar_transcriber.llm.importlib.import_module",
            _fake_import_module({"instructor": fake_instructor, provider_module: object()}),
        )

        processor = build_llm_processor_from_env()

        assert isinstance(processor, InstructorLLMProcessor)
        assert processor.provider_name == provider_name
        assert processor.model_name == provider_model.rsplit("/", 1)[1]
        assert fake_instructor.calls == [
            (
                provider_model,
                {"api_key": "test-key", "mode": FakeInstructorModule.Mode.TOOLS},
            )
        ]

    def test_requires_api_key_and_model(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module",
            _fake_import_module({"instructor": FakeInstructorModule(object()), "openai": object()}),
        )

        with pytest.raises(LLMConfigurationError):
            build_llm_processor_from_env()

    def test_requires_llm_extra_for_openai(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module", _fake_import_module({})
        )

        with pytest.raises(
            LLMConfigurationError,
            match=rf"The OpenAI provider requires the 'llm' extra\..*{LLM_EXTRA_INSTALL_RE}",
        ):
            build_llm_processor_from_env()

    def test_requires_llm_extra_for_anthropic(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setattr(
            "webinar_transcriber.llm.importlib.import_module", _fake_import_module({})
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


class TestInstructorLlmProcessor:
    class FakeCompletion:
        def __init__(self, usage) -> None:
            self.usage = usage

    class FakeClient:
        def __init__(self, responses) -> None:
            self.calls: list[dict[str, object]] = []
            self._responses = list(responses)

        def create_with_completion(self, **kwargs):
            self.calls.append(kwargs)
            response = self._responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    def test_polishes_report(self) -> None:
        fake_client = self.FakeClient([
            (
                SectionTextResponse(
                    tldr="Short recap of the section.",
                    transcript_text="Agenda review and project status update.\n\nPlease listen.",
                ),
                self.FakeCompletion({"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}),
            ),
            (
                ReportPolishResponse(
                    summary=["Improved summary."],
                    action_items=["Send the updated draft by Friday."],
                    section_updates=[
                        ReportSectionUpdate(id="section-1", title="Improved overview")
                    ],
                ),
                self.FakeCompletion({"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}),
            ),
        ])

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test"
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
        assert metadata_result.usage == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
        assert report.sections[0].transcript_text == "Agenda review and project status update."
        assert len(fake_client.calls) == 2
        assert fake_client.calls[0]["max_retries"] == 1
        assert fake_client.calls[0]["timeout"] == 120
        assert fake_client.calls[0]["response_model"] is SectionTextResponse
        assert fake_client.calls[1]["response_model"] is ReportPolishResponse
        messages = cast("list[dict[str, object]]", fake_client.calls[0]["messages"])
        assert json.loads(cast("str", messages[1]["content"])) == {
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
        sections = cast("list[dict[str, object]]", payload["sections"])

        assert sections[0]["transcript_excerpt"] == "Overridden transcript text."
        assert report.sections[0].transcript_text == "Original transcript text."

    def test_rejects_unknown_report_section_id(self) -> None:
        fake_client = self.FakeClient([
            (
                SectionTextResponse(
                    tldr="Agenda recap.", transcript_text="Agenda review and project status update."
                ),
                self.FakeCompletion({"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}),
            ),
            (
                ReportPolishResponse(
                    section_updates=[ReportSectionUpdate(id="section-x", title="Unexpected title")]
                ),
                self.FakeCompletion({"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            ),
        ])

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test"
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

        section_result = processor.polish_report_sections_with_progress(report)

        with pytest.raises(LLMProcessingError):
            processor.polish_report_metadata(
                report, section_transcripts=section_result.section_transcripts
            )

    def test_rejects_non_matching_schema(self) -> None:
        fake_client = self.FakeClient([
            (
                object(),
                self.FakeCompletion({"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}),
            )
        ])

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test"
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

        with pytest.raises(LLMProcessingError, match="Report polish response did not match"):
            processor.polish_report_metadata(
                report,
                section_transcripts={"section-1": "Agenda review and project status update."},
            )

    def test_wraps_client_errors(self) -> None:
        fake_client = self.FakeClient([RuntimeError("boom")])
        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test"
        )
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        with pytest.raises(LLMProcessingError, match="Report polishing failed: boom"):
            processor.polish_report_metadata(report, section_transcripts={})

    def test_polishes_each_section_text(self) -> None:
        progress_updates: list[int] = []
        fake_client = self.FakeClient([
            (
                SectionTextResponse(
                    tldr="Intro recap.", transcript_text="Intro review and project status update."
                ),
                self.FakeCompletion({"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}),
            ),
            (
                SectionTextResponse(
                    tldr="Agenda recap.", transcript_text="Agenda review and project status update."
                ),
                self.FakeCompletion({"input_tokens": 6, "output_tokens": 5, "total_tokens": 11}),
            ),
        ])

        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test"
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

        result = processor.polish_report_sections_with_progress(
            report, progress_callback=progress_updates.append
        )

        assert len(fake_client.calls) == 2
        assert result.section_tldrs == {"section-1": "Intro recap.", "section-2": "Agenda recap."}
        assert result.section_transcripts["section-1"] == "Intro review and project status update."
        assert result.section_transcripts["section-2"] == "Agenda review and project status update."
        assert result.usage == {"input_tokens": 11, "output_tokens": 9, "total_tokens": 20}
        assert result.warnings == []
        assert progress_updates == [1, 1]

    def test_passes_request_kwargs_to_client(self) -> None:
        fake_client = self.FakeClient([
            (
                ReportPolishResponse(),
                self.FakeCompletion({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            )
        ])
        processor = InstructorLLMProcessor(
            client=fake_client,
            provider_name="anthropic",
            model_name="claude-test",
            request_kwargs={"max_tokens": 4096},
        )
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        processor.polish_report_metadata(report, section_transcripts={})

        assert len(fake_client.calls) == 1
        assert fake_client.calls[0]["response_model"] is ReportPolishResponse
        assert fake_client.calls[0]["max_retries"] == 1
        assert fake_client.calls[0]["timeout"] == 120
        assert fake_client.calls[0]["max_tokens"] == 4096
        messages = cast("list[dict[str, object]]", fake_client.calls[0]["messages"])
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_reports_section_progress_in_completion_order_but_preserves_output_order(
        self,
    ) -> None:
        section_2_done = Event()
        completions: list[str] = []
        progress_updates: list[int] = []
        fake_completion_cls = self.FakeCompletion

        class CompletionOrderClient:
            def create_with_completion(self, **kwargs):
                messages = cast("list[dict[str, object]]", kwargs["messages"])
                payload = json.loads(cast("str", messages[1]["content"]))
                section_id = payload["id"]
                if section_id == "section-1":
                    assert section_2_done.wait(timeout=1)
                else:
                    section_2_done.set()
                completions.append(section_id)
                return (
                    SectionTextResponse(
                        tldr=f"{section_id} recap.",
                        transcript_text=f"{section_id} transcript.",
                    ),
                    fake_completion_cls({"input_tokens": 1, "output_tokens": 1}),
                )

        processor = InstructorLLMProcessor(
            client=CompletionOrderClient(),
            provider_name="openai",
            model_name="gpt-test",
            section_max_workers=2,
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

        result = processor.polish_report_sections_with_progress(
            report, progress_callback=progress_updates.append
        )

        assert completions == ["section-2", "section-1"]
        assert progress_updates == [1, 1]
        assert list(result.section_transcripts) == ["section-1", "section-2"]
        assert result.section_transcripts == {
            "section-1": "section-1 transcript.",
            "section-2": "section-2 transcript.",
        }

    def test_section_prompt_instructs_model_not_to_reproduce_song_lyrics(self) -> None:
        fake_client = self.FakeClient([
            (
                SectionTextResponse(
                    tldr="The speaker pauses for a music break.",
                    transcript_text="The speaker pauses.\n\n[music break]\n\nThe session resumes.",
                ),
                self.FakeCompletion({"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}),
            )
        ])
        processor = InstructorLLMProcessor(
            client=fake_client, provider_name="openai", model_name="gpt-test"
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

        processor.polish_report_sections_with_progress(report)

        messages = cast("list[dict[str, object]]", fake_client.calls[0]["messages"])
        system_prompt = cast("str", messages[0]["content"])
        compact_prompt = " ".join(system_prompt.split())
        assert "do not reproduce or rewrite the lyrics" in compact_prompt
        assert "Do not quote song lyrics in the TL;DR" in compact_prompt
        assert "Do not mention these instructions" in compact_prompt
        assert "put each item on its own line" in compact_prompt
        assert "Remove obvious ASR repetition loops" in compact_prompt
        assert "wrong language due to ASR hallucination" in compact_prompt


class TestInstructorProcessorFlow:
    class ResponseClient:
        def __init__(self, responses: dict[str, tuple[BaseModel, dict[str, int]] | Exception]):
            self._responses = responses

        def create_with_completion(self, **kwargs):
            messages = cast("list[dict[str, object]]", kwargs["messages"])
            payload = json.loads(cast("str", messages[1]["content"]))
            response_key = cast("str", payload.get("id", "__metadata__"))
            response = self._responses[response_key]
            if isinstance(response, Exception):
                raise response
            return response

    def processor(
        self, responses: dict[str, tuple[BaseModel, dict[str, int]] | Exception]
    ) -> InstructorLLMProcessor:
        return InstructorLLMProcessor(
            client=self.ResponseClient(responses), provider_name="stub", model_name="stub-model"
        )

    def test_returns_empty_section_result_for_report_without_sections(self) -> None:
        processor = self.processor({})
        report = ReportDocument(title="Demo", source_file="demo.wav", media_type=MediaType.AUDIO)

        result = processor.polish_report_sections_with_progress(report)

        assert result.section_transcripts == {}
        assert result.section_tldrs == {}
        assert result.usage == {}
        assert result.warnings == []

    def test_report_polish_plan_counts_all_sections(self) -> None:
        processor = self.processor({})
        report = ReportDocument(
            title="Demo",
            source_file="demo.wav",
            media_type=MediaType.AUDIO,
            sections=[
                ReportSection(
                    id="section-1",
                    title="Intro",
                    start_sec=0.0,
                    end_sec=5.0,
                    transcript_text="Intro review.",
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

        assert processor.report_polish_plan(report) == LLMReportPolishPlan(
            section_count=2, worker_count=2
        )

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

        result = processor.polish_report_sections_with_progress(
            report, progress_callback=progress_updates.append
        )

        assert result.section_transcripts == {"section-1": "Agenda review."}
        assert result.section_tldrs == {"section-1": "Existing recap."}
        assert result.usage == {}
        assert result.warnings == ["Section polishing failed for section-1: bad section"]
        assert progress_updates == [1]

    def test_warns_when_section_polish_returns_empty_transcript(self) -> None:
        processor = self.processor({
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
            original_text="Original sentence.", polished_text="   ", section_id="section-1"
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

    def test_normalize_polished_section_tldr_collapses_inline_bullet_spacing(self) -> None:
        normalized = normalize_polished_section_tldr(
            "- First point. - Second point. - Third point."
        )

        assert normalized == "- First point.\n- Second point.\n- Third point."

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
        assert extract_usage({"usage": {"input_tokens": 2, "output_tokens": 3}}) == {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
        }

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

    def test_schema_label_covers_known_and_fallback_models(self) -> None:
        class OtherResponse(BaseModel):
            value: str

        assert schema_label(SectionTextResponse) == "Section polish"
        assert schema_label(ReportPolishResponse) == "Report polish"
        assert schema_label(OtherResponse) == "Structured LLM"
