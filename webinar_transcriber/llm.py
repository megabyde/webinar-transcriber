"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import importlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from webinar_transcriber.models import (
    ReportDocument,
    ReportSection,
    TranscriptionResult,
    TranscriptSegment,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


REPORT_POLISH_TOTAL_CHAR_BUDGET = 16_000
REPORT_SECTION_EXCERPT_LIMIT = 1_200
SUMMARY_ITEM_LIMIT = 5
ACTION_ITEM_LIMIT = 7
SECTION_POLISH_MAX_WORKERS = 6
TRANSCRIPT_POLISH_CHAR_BUDGET = 20_000
TRANSCRIPT_POLISH_MAX_BATCH_SEGMENTS = 120
TRANSCRIPT_POLISH_MAX_WORKERS = 6
REPORT_POLISH_SYSTEM_PROMPT = """
You are improving a structured report built from an automatic speech transcript.

Keep the original language. Do not translate. Preserve meaning, names, and terminology.
Improve clarity without adding facts or interpretation.
Return factual summary bullets, concrete action items when they were directly assigned,
strongly implied, or presented as practical recommended next steps by the speakers,
and better section titles.
Do not turn general themes, broad observations, or abstract best practices into action items.
If there are no clear action items, return an empty list.
Do not change section IDs.
""".strip()
SECTION_POLISH_SYSTEM_PROMPT = """
You are cleaning one section of an automatic speech transcript.

Keep the original language. Do not translate. Preserve names, terminology, and meaning.
Fix punctuation, capitalization, and obvious ASR mistakes.
Preserve meaning, order, and level of detail.
Apply only light rephrasing for readability.
Do not add new facts, interpretations, advice, or commentary.
Prefer normal sentence punctuation. Do not add stylistic ellipses unless the source
clearly trails off.
Split the text into natural paragraphs separated by blank lines, usually 3-6 sentences
per paragraph. Insert a paragraph break when the speaker shifts to a new subpoint or
topic. Avoid returning one giant paragraph unless the input is extremely short.
""".strip()
TRANSCRIPT_POLISH_SYSTEM_PROMPT = """
You are cleaning an automatic speech transcript.

Keep the original language. Do not translate. Preserve names, terminology, and meaning.
Fix punctuation and spelling, and apply only light rephrasing for readability.
Prefer normal sentence punctuation. Do not add stylistic ellipses unless the source
clearly trails off.
You may add paragraph breaks with blank lines when they improve readability.
Do not merge or split segments. Return the same segment IDs you received.
Do not invent facts or add content that was not said.
""".strip()


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LLMProcessingError(RuntimeError):
    """Raised when the LLM response cannot be validated or applied."""


@dataclass(frozen=True)
class LLMTranscriptPolishResult:
    """Validated result from the transcript-polishing LLM stage."""

    transcription: TranscriptionResult
    usage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMTranscriptPolishPlan:
    """Execution plan for transcript polishing progress/reporting."""

    segment_count: int
    batch_count: int
    worker_count: int


@dataclass(frozen=True)
class LLMReportPolishResult:
    """Validated result from the report-polishing LLM stage."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    section_transcripts: dict[str, str]
    usage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


class TranscriptPolishItem(BaseModel):
    """Polished text for one transcript segment."""

    id: str
    text: str


class TranscriptPolishBatchResponse(BaseModel):
    """Structured LLM response for a transcript batch."""

    segments: list[TranscriptPolishItem] = Field(default_factory=list)


class SectionTextResponse(BaseModel):
    """Structured LLM response for one polished section body."""

    transcript_text: str = ""


class ReportSectionUpdate(BaseModel):
    """Replacement content for one report section."""

    id: str
    title: str


class ReportPolishResponse(BaseModel):
    """Structured LLM response for report polishing."""

    summary: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    section_updates: list[ReportSectionUpdate] = Field(default_factory=list)


@dataclass(frozen=True)
class LLMReportPolishPlan:
    """Execution plan for report polishing progress/reporting."""

    section_count: int
    worker_count: int


class LLMProcessor(Protocol):
    """Protocol for optional transcript/report enhancement backends."""

    @property
    def provider_name(self) -> str:
        """Return the configured provider identifier."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

    def polish_transcript(self, transcription: TranscriptionResult) -> LLMTranscriptPolishResult:
        """Return a validated polished transcript."""

    def polish_transcript_with_progress(
        self,
        transcription: TranscriptionResult,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMTranscriptPolishResult:
        """Return a validated polished transcript with chunk progress updates."""

    def polish_report(self, report: ReportDocument) -> LLMReportPolishResult:
        """Return polished summary, action items, and section titles."""

    def polish_report_with_progress(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMReportPolishResult:
        """Return polished report fields with per-section progress updates."""

    def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
        """Return concurrency details for report polishing."""

    def transcript_progress_total(self, transcription: TranscriptionResult) -> int:
        """Return the number of transcript segments that will drive progress."""

    def transcript_polish_plan(
        self,
        transcription: TranscriptionResult,
    ) -> LLMTranscriptPolishPlan:
        """Return batching/concurrency details for transcript polishing."""


class OpenAILLMProcessor:
    """OpenAI-backed transcript polishing integration."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        transcript_max_batch_segments: int = TRANSCRIPT_POLISH_MAX_BATCH_SEGMENTS,
        transcript_max_workers: int = TRANSCRIPT_POLISH_MAX_WORKERS,
        section_max_workers: int = SECTION_POLISH_MAX_WORKERS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
        transcript_char_budget: int = TRANSCRIPT_POLISH_CHAR_BUDGET,
    ) -> None:
        self._model_name = model_name
        self._report_char_budget = report_char_budget
        self._transcript_char_budget = transcript_char_budget
        self._transcript_max_batch_segments = transcript_max_batch_segments
        self._transcript_max_workers = transcript_max_workers
        self._section_max_workers = section_max_workers
        self._client = _build_openai_client(api_key)

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model identifier."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Return the configured LLM provider identifier."""
        return "openai"

    def polish_transcript(self, transcription: TranscriptionResult) -> LLMTranscriptPolishResult:
        """Polish transcript segment text while preserving IDs and timing."""
        return self.polish_transcript_with_progress(transcription)

    def polish_transcript_with_progress(
        self,
        transcription: TranscriptionResult,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMTranscriptPolishResult:
        """Polish transcript segment text while preserving IDs and timing."""
        usage_totals: dict[str, int] = {}
        warnings: list[str] = []
        polished_by_id: dict[str, str] = {}
        plan = self.transcript_polish_plan(transcription)
        batches = _chunk_transcript_segments(
            transcription.segments,
            max_chars=self._transcript_char_budget,
            max_segments=self._transcript_max_batch_segments,
        )
        batch_results: dict[int, tuple[dict[str, str], dict[str, int], int]] = {}

        with ThreadPoolExecutor(max_workers=plan.worker_count) as executor:
            future_to_batch_index = {
                executor.submit(self._polish_transcript_batch, batch): batch_index
                for batch_index, batch in enumerate(batches)
            }
            for future in as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                try:
                    polished_text_by_id, usage, processed_segments, batch_warnings = future.result()
                except LLMProcessingError as error:
                    batch = batches[batch_index]
                    polished_text_by_id = {segment.id: segment.text for segment in batch}
                    usage = {}
                    processed_segments = len(batch)
                    batch_warnings = [str(error)]
                batch_results[batch_index] = (polished_text_by_id, usage, processed_segments)
                warnings.extend(batch_warnings)
                if progress_callback is not None:
                    progress_callback(processed_segments)

        for batch_index in range(len(batches)):
            polished_text_by_id, usage, _processed_segments = batch_results[batch_index]
            polished_by_id.update(polished_text_by_id)
            _merge_usage(usage_totals, usage)

        return LLMTranscriptPolishResult(
            transcription=TranscriptionResult(
                detected_language=transcription.detected_language,
                segments=[
                    TranscriptSegment(
                        id=segment.id,
                        text=polished_by_id.get(segment.id, segment.text),
                        start_sec=segment.start_sec,
                        end_sec=segment.end_sec,
                    )
                    for segment in transcription.segments
                ],
            ),
            usage=usage_totals,
            warnings=warnings,
        )

    def transcript_progress_total(self, transcription: TranscriptionResult) -> int:
        """Return the number of transcript segments that will be polished."""
        return len(transcription.segments)

    def transcript_polish_plan(
        self,
        transcription: TranscriptionResult,
    ) -> LLMTranscriptPolishPlan:
        """Return batching/concurrency details for transcript polishing."""
        batches = _chunk_transcript_segments(
            transcription.segments,
            max_chars=self._transcript_char_budget,
            max_segments=self._transcript_max_batch_segments,
        )
        return LLMTranscriptPolishPlan(
            segment_count=len(transcription.segments),
            batch_count=len(batches),
            worker_count=min(self._transcript_max_workers, max(len(batches), 1)),
        )

    def _polish_transcript_batch(
        self,
        batch: Sequence[TranscriptSegment],
    ) -> tuple[dict[str, str], dict[str, int], int, list[str]]:
        payload = {
            "segments": [{"id": segment.id, "text": segment.text} for segment in batch],
        }
        try:
            response = self._client.responses.parse(
                model=self._model_name,
                input=[
                    {"role": "system", "content": TRANSCRIPT_POLISH_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=TranscriptPolishBatchResponse,
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(f"Transcript polishing failed: {error}") from error
        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, TranscriptPolishBatchResponse):
            raise LLMProcessingError("Transcript polish response did not match the schema.")
        polished_text_by_id, warnings = _validated_polished_segment_text(batch, parsed.segments)

        return (
            polished_text_by_id,
            _extract_usage(response),
            len(batch),
            warnings,
        )

    def polish_report(self, report: ReportDocument) -> LLMReportPolishResult:
        """Polish section text, summary, action items, and section titles."""
        return self.polish_report_with_progress(report)

    def polish_report_with_progress(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMReportPolishResult:
        """Polish section text first, then polish report summary and section titles."""
        usage_totals: dict[str, int] = {}
        warnings: list[str] = []
        polished_section_texts = self._polish_section_texts(
            report,
            progress_callback=progress_callback,
            usage_totals=usage_totals,
            warnings=warnings,
        )
        polished_report = ReportDocument(
            title=report.title,
            source_file=report.source_file,
            media_type=report.media_type,
            detected_language=report.detected_language,
            summary=list(report.summary),
            action_items=list(report.action_items),
            sections=[
                ReportSection(
                    id=section.id,
                    title=section.title,
                    start_sec=section.start_sec,
                    end_sec=section.end_sec,
                    transcript_text=polished_section_texts.get(section.id, section.transcript_text),
                    bullet_points=list(section.bullet_points),
                    frame_id=section.frame_id,
                    image_path=section.image_path,
                )
                for section in report.sections
            ],
            warnings=list(report.warnings),
        )
        payload = _build_report_polish_payload(
            polished_report,
            total_char_budget=self._report_char_budget,
        )
        try:
            response = self._client.responses.parse(
                model=self._model_name,
                input=[
                    {"role": "system", "content": REPORT_POLISH_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=ReportPolishResponse,
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(f"Report polishing failed: {error}") from error

        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, ReportPolishResponse):
            raise LLMProcessingError("Report polish response did not match the schema.")

        report_usage = _extract_usage(response)
        _merge_usage(usage_totals, report_usage)

        return LLMReportPolishResult(
            summary=_normalize_report_lines(parsed.summary, limit=SUMMARY_ITEM_LIMIT),
            action_items=_normalize_report_lines(
                parsed.action_items,
                limit=ACTION_ITEM_LIMIT,
            ),
            section_titles=_validated_section_titles(report, parsed.section_updates),
            section_transcripts=polished_section_texts,
            usage=usage_totals,
            warnings=warnings,
        )

    def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
        """Return concurrency details for report polishing."""
        return LLMReportPolishPlan(
            section_count=len(report.sections),
            worker_count=min(self._section_max_workers, max(len(report.sections), 1)),
        )

    def _polish_section_texts(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None,
        usage_totals: dict[str, int],
        warnings: list[str],
    ) -> dict[str, str]:
        plan = self.report_polish_plan(report)
        if not report.sections:
            return {}

        polished_texts: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=plan.worker_count) as executor:
            future_to_section = {
                executor.submit(self._polish_section_text, section): section
                for section in report.sections
            }
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    transcript_text, usage, section_warnings = future.result()
                except LLMProcessingError as error:
                    transcript_text = section.transcript_text
                    usage = {}
                    section_warnings = [str(error)]
                polished_texts[section.id] = transcript_text
                _merge_usage(usage_totals, usage)
                warnings.extend(section_warnings)
                if progress_callback is not None:
                    progress_callback(1)

        return polished_texts

    def _polish_section_text(
        self,
        section: ReportSection,
    ) -> tuple[str, dict[str, int], list[str]]:
        payload = {
            "id": section.id,
            "title": section.title,
            "start_sec": section.start_sec,
            "end_sec": section.end_sec,
            "transcript_text": section.transcript_text,
        }
        try:
            response = self._client.responses.parse(
                model=self._model_name,
                input=[
                    {"role": "system", "content": SECTION_POLISH_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=SectionTextResponse,
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(
                f"Section polishing failed for {section.id}: {error}"
            ) from error
        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, SectionTextResponse):
            raise LLMProcessingError(
                f"Section polish response did not match the schema for {section.id}."
            )

        transcript_text = _normalize_polished_section_text(
            original_text=section.transcript_text,
            polished_text=parsed.transcript_text,
            section_id=section.id,
        )
        warnings: list[str] = []
        if transcript_text == section.transcript_text and not parsed.transcript_text.strip():
            warnings.append(
                f"Section polish response returned an empty transcript text for {section.id}; "
                "kept original text."
            )

        return transcript_text, _extract_usage(response), warnings


def build_llm_processor_from_env() -> OpenAILLMProcessor:
    """Build an OpenAI-backed LLM processor from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")

    missing_vars = [
        env_name
        for env_name, value in (
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_MODEL", model_name),
        )
        if not value
    ]
    if missing_vars:
        missing = ", ".join(missing_vars)
        raise LLMConfigurationError(f"Missing required LLM environment variables: {missing}.")

    assert api_key is not None
    assert model_name is not None
    return OpenAILLMProcessor(api_key=api_key, model_name=model_name)


def _build_openai_client(api_key: str) -> Any:
    try:
        openai_module = importlib.import_module("openai")
    except ModuleNotFoundError as error:  # pragma: no cover - dependency wiring only
        raise LLMConfigurationError(
            "LLM requested but the OpenAI SDK is not installed in this environment."
        ) from error
    return openai_module.OpenAI(api_key=api_key)


def _chunk_transcript_segments(
    segments: Sequence[TranscriptSegment],
    *,
    max_chars: int,
    max_segments: int,
) -> list[list[TranscriptSegment]]:
    chunks: list[list[TranscriptSegment]] = []
    current_chunk: list[TranscriptSegment] = []
    current_chars = 0

    for segment in segments:
        segment_chars = len(segment.text)
        would_overflow = current_chunk and (
            current_chars + segment_chars > max_chars or len(current_chunk) >= max_segments
        )
        if would_overflow:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(segment)
        current_chars += segment_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _build_report_polish_payload(
    report: ReportDocument,
    *,
    total_char_budget: int,
) -> dict[str, object]:
    section_count = max(len(report.sections), 1)
    per_section_budget = min(
        REPORT_SECTION_EXCERPT_LIMIT,
        max(200, total_char_budget // section_count),
    )

    return {
        "title": report.title,
        "source_file": report.source_file,
        "detected_language": report.detected_language,
        "current_summary": report.summary,
        "current_action_items": report.action_items,
        "sections": [
            {
                "id": section.id,
                "title": section.title,
                "start_sec": section.start_sec,
                "end_sec": section.end_sec,
                "transcript_excerpt": _truncate_text(section.transcript_text, per_section_budget),
            }
            for section in report.sections
        ],
    }


def _normalize_report_lines(lines: Sequence[str], *, limit: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        dedupe_key = cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(cleaned)
        if len(normalized) == limit:
            break

    return normalized


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def _validated_section_titles(
    report: ReportDocument,
    section_titles: Sequence[ReportSectionUpdate],
) -> dict[str, str]:
    valid_ids = {section.id for section in report.sections}
    polished_titles: dict[str, str] = {}

    for item in section_titles:
        if item.id not in valid_ids:
            raise LLMProcessingError("Report polish response returned an unknown section ID.")
        if item.id in polished_titles:
            raise LLMProcessingError("Report polish response returned duplicate section IDs.")
        cleaned_title = item.title.strip()
        if not cleaned_title:
            raise LLMProcessingError("Report polish response returned an empty section title.")
        polished_titles[item.id] = cleaned_title

    return polished_titles


def _validated_polished_segment_text(
    original_segments: Sequence[TranscriptSegment],
    polished_segments: Sequence[TranscriptPolishItem],
) -> tuple[dict[str, str], list[str]]:
    polished_by_id: dict[str, str] = {}
    warnings: list[str] = []
    expected_ids = [segment.id for segment in original_segments]
    original_text_by_id = {segment.id: segment.text for segment in original_segments}

    if len(polished_segments) != len(original_segments):
        raise LLMProcessingError(
            "Transcript polish response changed the number of segments in a batch."
        )

    for item in polished_segments:
        if item.id in polished_by_id:
            raise LLMProcessingError("Transcript polish response returned duplicate segment IDs.")
        polished_text = _normalize_polished_segment_text(
            original_text=original_text_by_id.get(item.id, ""),
            polished_text=item.text,
        )
        if not polished_text:
            polished_by_id[item.id] = original_text_by_id.get(item.id, "")
            warnings.append(
                f"Transcript polish response returned an empty segment text for {item.id}; "
                "kept original text."
            )
        else:
            polished_by_id[item.id] = polished_text

    if list(polished_by_id) != expected_ids:
        raise LLMProcessingError("Transcript polish response changed the segment IDs or order.")

    return polished_by_id, warnings


def _normalize_polished_segment_text(*, original_text: str, polished_text: str) -> str:
    cleaned = polished_text.strip()
    if not cleaned:
        return ""

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n+", cleaned)
        if paragraph.strip()
    ]
    cleaned = "\n\n".join(paragraphs)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    if not re.search(r"(?:\.{3}|…)\s*$", original_text.strip()):
        cleaned = re.sub(r"(?:\s*)(?:\.{3}|…)+\s*$", ".", cleaned)

    return cleaned


def _normalize_polished_section_text(
    *,
    original_text: str,
    polished_text: str,
    section_id: str,
) -> str:
    cleaned = _normalize_polished_segment_text(
        original_text=original_text,
        polished_text=polished_text,
    )
    if not cleaned:
        return original_text
    if len(cleaned) < 20 and len(original_text.strip()) > 100:
        raise LLMProcessingError(
            f"Section polish response looked truncated for {section_id}; kept original text."
        )
    return cleaned


def _extract_usage(response: object) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            key: value
            for key, value in usage.items()
            if isinstance(value, int) and key in {"input_tokens", "output_tokens", "total_tokens"}
        }

    extracted: dict[str, int] = {}
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        field_value = getattr(usage, field_name, None)
        if isinstance(field_value, int):
            extracted[field_name] = field_value
    return extracted


def _merge_usage(target: dict[str, int], new_usage: dict[str, int]) -> None:
    for key, value in new_usage.items():
        target[key] = target.get(key, 0) + value
