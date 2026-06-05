"""Instructor-backed optional cloud LLM integration."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeVar

import tenacity
from pydantic import BaseModel, Field

from webinar_transcriber.text_utils import split_paragraph_blocks

from . import (
    LlmProcessingError,
    LlmReportMetadataResult,
    LlmReportPolishPlan,
    LlmSectionPolishResult,
)
from .prompts import (
    ACTION_ITEM_LIMIT,
    REPORT_POLISH_SYSTEM_PROMPT,
    REPORT_POLISH_TOTAL_CHAR_BUDGET,
    REPORT_SECTION_EXCERPT_LIMIT,
    SECTION_POLISH_SYSTEM_PROMPT,
    SUMMARY_ITEM_LIMIT,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from webinar_transcriber.models import ReportDocument, ReportSection


# ---------------------------------------------------------------------------
# Pydantic wire schemas (Instructor boundary)
# ---------------------------------------------------------------------------

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


class SectionTextResponse(BaseModel):
    """Structured LLM response for one polished section body."""

    tldr: str = ""
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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_SCHEMA_LABELS = {"SectionTextResponse": "Section polish", "ReportPolishResponse": "Report polish"}


def build_report_polish_payload(
    report: ReportDocument,
    *,
    total_char_budget: int,
    section_transcripts: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the report-polish payload with a per-section excerpt budget.

    Returns:
        dict[str, object]: The structured payload sent to the metadata-polish stage.
    """
    section_count = max(len(report.sections), 1)
    per_section_budget = min(
        REPORT_SECTION_EXCERPT_LIMIT, max(200, total_char_budget // section_count)
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
                "transcript_excerpt": truncate_text(
                    (section_transcripts or {}).get(section.id, section.transcript_text),
                    per_section_budget,
                ),
            }
            for section in report.sections
        ],
    }


def normalize_report_lines(lines: Sequence[str], *, limit: int) -> list[str]:
    """Clean, dedupe, and cap a list of LLM-generated report lines.

    Returns:
        list[str]: The cleaned and deduplicated report lines.
    """
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


def truncate_text(text: str, max_chars: int) -> str:
    """Trim text to one character budget while preserving whole-word readability.

    Returns:
        str: The truncated text.
    """
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def validated_section_titles(
    report: ReportDocument, section_titles: Sequence[ReportSectionUpdate]
) -> dict[str, str]:
    """Validate returned section-title updates against the existing report.

    Returns:
        dict[str, str]: Cleaned replacement titles keyed by section ID.

    Raises:
        LlmProcessingError: If the response references invalid, duplicate, or empty titles.
    """
    valid_ids = {section.id for section in report.sections}
    polished_titles: dict[str, str] = {}

    for item in section_titles:
        if item.id not in valid_ids:
            raise LlmProcessingError("Report polish response returned an unknown section ID.")
        if item.id in polished_titles:
            raise LlmProcessingError("Report polish response returned duplicate section IDs.")
        cleaned_title = item.title.strip()
        if not cleaned_title:
            raise LlmProcessingError("Report polish response returned an empty section title.")
        polished_titles[item.id] = cleaned_title

    return polished_titles


def normalize_polished_text(*, polished_text: str) -> str:
    """Normalize multi-paragraph LLM text.

    Returns:
        str: The normalized polished text.
    """
    cleaned = polished_text.strip()
    if not cleaned:
        return ""

    cleaned = _normalize_llm_paragraphs(cleaned)
    return re.sub(r"[ \t]+\n", "\n", cleaned)


def normalize_polished_section_text(*, original_text: str, polished_text: str) -> str:
    """Normalize polished section text.

    Returns:
        str: The accepted polished section text, or the original text when empty.
    """
    cleaned = normalize_polished_text(polished_text=polished_text)
    if not cleaned:
        return original_text
    return cleaned


def normalize_polished_section_tldr(tldr: str) -> str:
    """Normalize a polished TL;DR into stable paragraph spacing.

    Returns:
        str: The normalized TL;DR text.
    """
    cleaned = tldr.strip()
    if not cleaned:
        return ""

    return _normalize_tldr_blocks(cleaned)


def _normalize_tldr_blocks(text: str) -> str:
    normalized = _normalize_llm_paragraphs(text)
    return re.sub(r"(?<=\S)[ \t]+(?=(?:[-*]|\d+[.)])\s+)", "\n", normalized)


def _normalize_llm_paragraphs(text: str) -> str:
    return "\n\n".join(
        split_paragraph_blocks(text, flexible_blank_lines=True, normalize_inline_whitespace=True)
    )


def extract_response_metadata(response: object) -> dict[str, object]:
    """Extract non-sensitive provider response metadata for diagnostics."""
    metadata: dict[str, object] = {}
    choice = _first_choice(response)
    message = getattr(choice, "message", None) if choice is not None else None

    if finish_reason := _text_attr(choice, "finish_reason"):
        metadata["finish_reason"] = finish_reason
    if stop_reason := _text_attr(response, "stop_reason"):
        metadata["stop_reason"] = stop_reason
    refusal = getattr(message, "refusal", None)
    if refusal:
        metadata["refusal"] = True

    safety = _json_safe(getattr(choice, "content_filter_results", None))
    if safety is not None:
        metadata["safety"] = safety
    prompt_filter_results = _json_safe(getattr(response, "prompt_filter_results", None))
    if prompt_filter_results is not None:
        metadata["prompt_filter_results"] = prompt_filter_results

    return metadata


def _first_choice(response: object) -> object | None:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        return choices[0]
    return None


def _text_attr(value: object | None, name: str) -> str | None:
    text = getattr(value, name, None)
    return text if isinstance(text, str) and text else None


def _json_safe(value: object) -> object | None:
    try:
        return json.loads(json.dumps(value, default=_json_default, ensure_ascii=False))
    except (TypeError, ValueError):
        return None


def _json_default(value: object) -> object:
    if callable(model_dump := getattr(value, "model_dump", None)):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if callable(to_dict := getattr(value, "to_dict", None)):
        return to_dict()
    return None


def schema_label(response_model: type[object]) -> str:
    """Return the human-facing label for one structured response schema."""
    return _SCHEMA_LABELS.get(response_model.__name__, "Structured LLM")


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolishedSection:
    """Polished section transcript and TL;DR payload."""

    id: str
    transcript: str
    tldr: str


@dataclass(frozen=True)
class SectionPolishOutputs:
    """Polished section payloads keyed by section id."""

    sections: dict[str, PolishedSection]
    response_metadata: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SectionPolishResult:
    """Result from polishing one report section."""

    section_id: str
    transcript_text: str
    tldr: str
    response_metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# InstructorLLMProcessor
# ---------------------------------------------------------------------------

LLM_RATE_LIMIT_RETRY_ATTEMPTS = 3
_DEFAULT_REQUEST_TIMEOUT_SEC = 120
_EMPTY_REQUEST_KWARGS = MappingProxyType({})


class InstructorClient(Protocol):
    """Protocol for the patched Instructor client used by this module."""

    def create_with_completion(self, **kwargs: object) -> tuple[object, object]:
        """Return the parsed response model alongside the raw completion."""


class InstructorLLMProcessor:
    """Instructor-backed transcript polishing integration."""

    def __init__(
        self,
        *,
        client: InstructorClient,
        provider_name: str,
        model_name: str,
        threads: int,
        request_kwargs: Mapping[str, object] = _EMPTY_REQUEST_KWARGS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
    ) -> None:
        """Initialize one provider-backed Instructor processor."""
        self._client = client
        self._provider_name = provider_name
        self._model_name = model_name
        self._request_kwargs = {"timeout": _DEFAULT_REQUEST_TIMEOUT_SEC, **request_kwargs}
        self._report_char_budget = report_char_budget
        self._threads = threads

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Return the configured LLM provider identifier."""
        return self._provider_name

    def polish_report_sections_with_progress(
        self, report: ReportDocument, *, progress_callback: Callable[[int], None] | None = None
    ) -> LlmSectionPolishResult:
        """Return polished section text with per-section progress updates."""
        warnings: list[str] = []
        polished_section_texts = self._polish_section_texts(
            report, progress_callback=progress_callback, warnings=warnings
        )
        section_tldrs = {
            section_id: section.tldr
            for section_id, section in polished_section_texts.sections.items()
            if section.tldr
        }
        section_transcripts = {
            section_id: section.transcript
            for section_id, section in polished_section_texts.sections.items()
        }
        return LlmSectionPolishResult(
            section_tldrs=section_tldrs,
            section_transcripts=section_transcripts,
            response_metadata=polished_section_texts.response_metadata,
            warnings=warnings,
        )

    def polish_report_metadata(
        self, report: ReportDocument, *, section_transcripts: dict[str, str]
    ) -> LlmReportMetadataResult:
        """Return polished summary, action items, and section titles."""
        payload = build_report_polish_payload(
            report,
            total_char_budget=self._report_char_budget,
            section_transcripts=section_transcripts,
        )
        parsed, response_metadata = self._create_structured_response(
            system_prompt=REPORT_POLISH_SYSTEM_PROMPT,
            user_payload=payload,
            response_model=ReportPolishResponse,
            error_prefix="Report polishing failed",
        )

        return LlmReportMetadataResult(
            summary=normalize_report_lines(parsed.summary, limit=SUMMARY_ITEM_LIMIT),
            action_items=normalize_report_lines(parsed.action_items, limit=ACTION_ITEM_LIMIT),
            section_titles=validated_section_titles(report, parsed.section_updates),
            response_metadata=[{"stage": "report_polish", **response_metadata}],
        )

    def report_polish_plan(self, report: ReportDocument) -> LlmReportPolishPlan:
        """Return concurrency details for report polishing."""
        return LlmReportPolishPlan(
            section_count=len(report.sections),
            worker_count=min(self._threads, max(len(report.sections), 1)),
        )

    def _polish_section_texts(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None,
        warnings: list[str],
    ) -> SectionPolishOutputs:
        plan = self.report_polish_plan(report)
        if not report.sections:
            return SectionPolishOutputs(sections={})

        section_results: dict[int, SectionPolishResult] = {}

        executor = ThreadPoolExecutor(max_workers=plan.worker_count)
        futures = {
            executor.submit(self._polish_section, section): index
            for index, section in enumerate(report.sections)
        }
        try:
            for future in as_completed(futures):
                section_results[futures[future]] = future.result()
                if progress_callback is not None:
                    progress_callback(1)
        except BaseException:  # pragma: no cover - cancellation/interrupt propagation
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        polished_sections: dict[str, PolishedSection] = {}
        response_metadata: list[dict[str, object]] = []
        for index in range(len(report.sections)):
            result = section_results[index]
            polished_sections[result.section_id] = PolishedSection(
                id=result.section_id, transcript=result.transcript_text, tldr=result.tldr
            )
            response_metadata.append(result.response_metadata)
            warnings.extend(result.warnings)

        return SectionPolishOutputs(sections=polished_sections, response_metadata=response_metadata)

    def _polish_section(self, section: ReportSection) -> SectionPolishResult:
        try:
            return self._polish_section_text(section)
        except LlmProcessingError as error:
            return SectionPolishResult(
                section_id=section.id,
                transcript_text=section.transcript_text,
                tldr=section.tldr or "",
                response_metadata={"stage": "section_polish", "section_id": section.id},
                warnings=[str(error)],
            )

    def _polish_section_text(self, section: ReportSection) -> SectionPolishResult:
        payload = {
            "id": section.id,
            "title": section.title,
            "start_sec": section.start_sec,
            "end_sec": section.end_sec,
            "transcript_text": section.transcript_text,
        }
        parsed, response_metadata = self._create_structured_response(
            system_prompt=SECTION_POLISH_SYSTEM_PROMPT,
            user_payload=payload,
            response_model=SectionTextResponse,
            error_prefix=f"Section polishing failed for {section.id}",
        )

        transcript_text = normalize_polished_section_text(
            original_text=section.transcript_text, polished_text=parsed.transcript_text
        )
        tldr = normalize_polished_section_tldr(parsed.tldr)
        warnings: list[str] = []
        if transcript_text == section.transcript_text and not parsed.transcript_text.strip():
            warnings.append(
                f"Section polish response returned an empty transcript text for {section.id}; "
                "kept original text."
            )

        return SectionPolishResult(
            section_id=section.id,
            transcript_text=transcript_text,
            tldr=tldr,
            response_metadata={
                "stage": "section_polish",
                "section_id": section.id,
                **response_metadata,
            },
            warnings=warnings,
        )

    def _create_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, object]]:
        try:
            parsed, completion = self._client.create_with_completion(
                response_model=response_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                max_retries=_structured_response_retries(),
                **self._request_kwargs,
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LlmProcessingError(f"{error_prefix}: {error}") from error

        if not isinstance(parsed, response_model):
            raise LlmProcessingError(
                f"{schema_label(response_model)} response did not match the schema."
            )
        return parsed, extract_response_metadata(completion)


def _structured_response_retries() -> object:  # pragma: no cover - optional llm extra only
    return tenacity.Retrying(
        retry=tenacity.retry_if_exception(_is_transient_provider_error),
        stop=tenacity.stop_after_attempt(LLM_RATE_LIMIT_RETRY_ATTEMPTS),
        wait=tenacity.wait_exponential(multiplier=1),
        reraise=True,
    )


def _is_transient_provider_error(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    if not isinstance(status_code, int):  # pragma: no cover - provider SDK shape fallback
        return False
    return status_code in {408, 429} or 500 <= status_code < 600
