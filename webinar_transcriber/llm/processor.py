"""Instructor-backed optional cloud LLM integration."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import tenacity

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
    SECTION_POLISH_SYSTEM_PROMPT,
    SUMMARY_ITEM_LIMIT,
)
from .utils import (
    ReportPolishResponse,
    ReportSectionUpdate,
    SchemaModelT,
    SectionTextResponse,
    build_report_polish_payload,
    extract_response_metadata,
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_report_lines,
    schema_label,
    validated_section_titles,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from webinar_transcriber.models import ReportDocument, ReportSection

__all__ = [
    "InstructorLLMProcessor",
    "ReportPolishResponse",
    "ReportSectionUpdate",
    "SectionTextResponse",
]

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


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

_LLM_RATE_LIMIT_RETRY_ATTEMPTS = 3
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
        results = self._polish_section_texts(report, progress_callback=progress_callback)
        return LlmSectionPolishResult(
            section_tldrs={r.section_id: r.tldr for r in results if r.tldr},
            section_transcripts={r.section_id: r.transcript_text for r in results},
            response_metadata=[r.response_metadata for r in results],
            warnings=[warning for r in results for warning in r.warnings],
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
        self, report: ReportDocument, *, progress_callback: Callable[[int], None] | None
    ) -> list[SectionPolishResult]:
        if not report.sections:
            return []

        plan = self.report_polish_plan(report)
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

        return [section_results[index] for index in range(len(report.sections))]

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
        stop=tenacity.stop_after_attempt(_LLM_RATE_LIMIT_RETRY_ATTEMPTS),
        wait=tenacity.wait_exponential(multiplier=1),
        reraise=True,
    )


def _is_transient_provider_error(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    if not isinstance(status_code, int):  # pragma: no cover - provider SDK shape fallback
        return False
    return status_code in {408, 429} or 500 <= status_code < 600
