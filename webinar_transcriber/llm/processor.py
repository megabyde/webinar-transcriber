"""Instructor-backed optional cloud LLM integration."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Protocol

from .contracts import (
    LLMProcessingError,
    LLMReportMetadataResult,
    LLMReportPolishPlan,
    LLMSectionPolishResult,
    ReportPolishResponse,
    SchemaModelT,
    SectionPolishOutputs,
    SectionTextResponse,
)
from .prompts import (
    ACTION_ITEM_LIMIT,
    REPORT_POLISH_SYSTEM_PROMPT,
    REPORT_POLISH_TOTAL_CHAR_BUDGET,
    SECTION_POLISH_MAX_WORKERS,
    SECTION_POLISH_SYSTEM_PROMPT,
    SUMMARY_ITEM_LIMIT,
)
from .utils import (
    build_report_polish_payload,
    extract_usage,
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_report_lines,
    schema_label,
    validated_section_titles,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from webinar_transcriber.models import ReportDocument, ReportSection


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
        request_kwargs: Mapping[str, object] | None = None,
        section_max_workers: int = SECTION_POLISH_MAX_WORKERS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
    ) -> None:
        """Initialize one provider-backed Instructor processor."""
        self._client = client
        self._provider_name = provider_name
        self._model_name = model_name
        self._request_kwargs = {"timeout": 120, **dict(request_kwargs or {})}
        self._report_char_budget = report_char_budget
        self._section_max_workers = section_max_workers

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
    ) -> LLMSectionPolishResult:
        """Return polished section text with per-section progress updates."""
        usage_totals: dict[str, int] = {}
        warnings: list[str] = []
        polished_section_texts = self._polish_section_texts(
            report,
            progress_callback=progress_callback,
            usage_totals=usage_totals,
            warnings=warnings,
        )
        return LLMSectionPolishResult(
            section_tldrs=polished_section_texts.tldrs,
            section_transcripts=polished_section_texts.transcripts,
            usage=usage_totals,
            warnings=warnings,
        )

    def polish_report_metadata(
        self, report: ReportDocument, *, section_transcripts: dict[str, str]
    ) -> LLMReportMetadataResult:
        """Return polished summary, action items, and section titles."""
        payload = build_report_polish_payload(
            report,
            total_char_budget=self._report_char_budget,
            section_transcripts=section_transcripts,
        )
        parsed, usage = self._create_structured_response(
            system_prompt=REPORT_POLISH_SYSTEM_PROMPT,
            user_payload=payload,
            response_model=ReportPolishResponse,
            error_prefix="Report polishing failed",
        )

        return LLMReportMetadataResult(
            summary=normalize_report_lines(parsed.summary, limit=SUMMARY_ITEM_LIMIT),
            action_items=normalize_report_lines(parsed.action_items, limit=ACTION_ITEM_LIMIT),
            section_titles=validated_section_titles(report, parsed.section_updates),
            usage=usage,
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
    ) -> SectionPolishOutputs:
        plan = self.report_polish_plan(report)
        if not report.sections:
            return SectionPolishOutputs(transcripts={}, tldrs={})

        section_results: list[tuple[str, str, str, dict[str, int], list[str]] | None] = [
            None
        ] * len(report.sections)

        with ThreadPoolExecutor(max_workers=plan.worker_count) as executor:
            futures = {
                executor.submit(self._polish_section, section): index
                for index, section in enumerate(report.sections)
            }
            for future in as_completed(futures):
                section_results[futures[future]] = future.result()
                if progress_callback is not None:
                    progress_callback(1)

        polished_transcripts: dict[str, str] = {}
        polished_tldrs: dict[str, str] = {}
        for result in section_results:
            if result is None:  # pragma: no cover - all submitted futures completed
                continue
            section_id, transcript_text, tldr, usage, section_warnings = result
            polished_transcripts[section_id] = transcript_text
            if tldr:
                polished_tldrs[section_id] = tldr
            usage_totals.update(Counter(usage_totals) + Counter(usage))
            warnings.extend(section_warnings)

        return SectionPolishOutputs(transcripts=polished_transcripts, tldrs=polished_tldrs)

    def _polish_section(
        self, section: ReportSection
    ) -> tuple[str, str, str, dict[str, int], list[str]]:
        try:
            transcript_text, tldr, usage, section_warnings = self._polish_section_text(section)
        except LLMProcessingError as error:
            transcript_text = section.transcript_text
            tldr = section.tldr or ""
            usage = {}
            section_warnings = [str(error)]
        return section.id, transcript_text, tldr, usage, section_warnings

    def _polish_section_text(
        self, section: ReportSection
    ) -> tuple[str, str, dict[str, int], list[str]]:
        payload = {
            "id": section.id,
            "title": section.title,
            "start_sec": section.start_sec,
            "end_sec": section.end_sec,
            "transcript_text": section.transcript_text,
        }
        parsed, usage = self._create_structured_response(
            system_prompt=SECTION_POLISH_SYSTEM_PROMPT,
            user_payload=payload,
            response_model=SectionTextResponse,
            error_prefix=f"Section polishing failed for {section.id}",
        )

        transcript_text = normalize_polished_section_text(
            original_text=section.transcript_text,
            polished_text=parsed.transcript_text,
            section_id=section.id,
        )
        tldr = normalize_polished_section_tldr(parsed.tldr)
        warnings: list[str] = []
        if transcript_text == section.transcript_text and not parsed.transcript_text.strip():
            warnings.append(
                f"Section polish response returned an empty transcript text for {section.id}; "
                "kept original text."
            )

        return transcript_text, tldr, usage, warnings

    def _create_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, int]]:
        try:
            parsed, completion = self._client.create_with_completion(
                response_model=response_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                max_retries=1,
                **self._request_kwargs,
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(f"{error_prefix}: {error}") from error

        if not isinstance(parsed, response_model):
            raise LLMProcessingError(
                f"{schema_label(response_model)} response did not match the schema."
            )
        return parsed, extract_usage(completion)
