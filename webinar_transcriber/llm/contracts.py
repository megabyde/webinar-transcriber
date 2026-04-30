"""Provider-neutral contracts for optional cloud LLM integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.models import ReportDocument


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LLMProcessingError(RuntimeError):
    """Raised when the LLM response cannot be validated or applied."""


@dataclass(frozen=True)
class LLMSectionPolishResult:
    """Validated result from the section-text polishing phase."""

    section_tldrs: dict[str, str]
    section_transcripts: dict[str, str]
    usage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMReportMetadataResult:
    """Validated result from the final report metadata polishing phase."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    usage: dict[str, int]


@dataclass(frozen=True)
class SectionPolishOutputs:
    """Polished section transcript and TL;DR payloads keyed by section id."""

    transcripts: dict[str, str]
    tldrs: dict[str, str]


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

    def polish_report_sections_with_progress(
        self, report: ReportDocument, *, progress_callback: Callable[[int], None] | None = None
    ) -> LLMSectionPolishResult:
        """Return polished section text with per-section progress updates."""

    def polish_report_metadata(
        self, report: ReportDocument, *, section_transcripts: dict[str, str]
    ) -> LLMReportMetadataResult:
        """Return polished summary, action items, and section titles."""

    def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
        """Return concurrency details for report polishing."""
