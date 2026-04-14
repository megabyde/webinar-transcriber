"""Shared execution flow for provider-backed report polishing."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from .contracts import (
    LLMProcessingError,
    LLMReportMetadataResult,
    LLMReportPolishPlan,
    LLMSectionPolishResult,
    ReportPolishResponse,
    SchemaModelT,
    SectionTextResponse,
    _SectionPolishOutputs,
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
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_report_lines,
    validated_section_titles,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from webinar_transcriber.models import ReportDocument, ReportSection


class _BaseLLMProcessor:
    """Shared report-polishing flow for structured LLM backends."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        section_max_workers: int = SECTION_POLISH_MAX_WORKERS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
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
        """Polish section transcript text with per-section progress updates."""
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
        """Polish report summary, action items, and section titles."""
        polished_report = report.model_copy(deep=True)
        for section in polished_report.sections:
            section.transcript_text = section_transcripts.get(section.id, section.transcript_text)
        payload = build_report_polish_payload(
            polished_report, total_char_budget=self._report_char_budget
        )
        parsed, usage = self._parse_structured_response(
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
        polishable_sections = [section for section in report.sections if not section.is_interlude]
        return LLMReportPolishPlan(
            section_count=len(polishable_sections),
            worker_count=min(self._section_max_workers, max(len(polishable_sections), 1)),
            skipped_section_count=len(report.sections) - len(polishable_sections),
        )

    def _polish_section_texts(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None,
        usage_totals: dict[str, int],
        warnings: list[str],
    ) -> _SectionPolishOutputs:
        plan = self.report_polish_plan(report)
        if not report.sections:
            return _SectionPolishOutputs(transcripts={}, tldrs={})

        polished_transcripts: dict[str, str] = {}
        polished_tldrs: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=plan.worker_count) as executor:
            future_to_section = {
                executor.submit(self._polish_section_text, section): section
                for section in report.sections
                if not section.is_interlude
            }
            skipped_sections = [section for section in report.sections if section.is_interlude]
            for section in skipped_sections:
                polished_transcripts[section.id] = section.transcript_text
                if section.tldr:
                    polished_tldrs[section.id] = section.tldr
                warnings.append(
                    f"Skipped LLM section polish for likely music/interlude section {section.id}."
                )
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    transcript_text, tldr, usage, section_warnings = future.result()
                except LLMProcessingError as error:
                    transcript_text = section.transcript_text
                    tldr = section.tldr or ""
                    usage = {}
                    section_warnings = [str(error)]
                polished_transcripts[section.id] = transcript_text
                if tldr:
                    polished_tldrs[section.id] = tldr
                usage_totals.update(Counter(usage_totals) + Counter(usage))
                warnings.extend(section_warnings)
                if progress_callback is not None:
                    progress_callback(1)

        return _SectionPolishOutputs(transcripts=polished_transcripts, tldrs=polished_tldrs)

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
        parsed, usage = self._parse_structured_response(
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

    def _parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, int]]:
        raise NotImplementedError
