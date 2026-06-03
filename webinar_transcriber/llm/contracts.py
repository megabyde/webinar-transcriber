"""Provider-neutral contracts for optional cloud LLM integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.models import ReportDocument


class LlmConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LlmProcessingError(RuntimeError):
    """Raised when the LLM response cannot be validated or applied."""


@dataclass(frozen=True)
class LlmSectionPolishResult:
    """Validated result from the section-text polishing phase."""

    section_tldrs: dict[str, str]
    section_transcripts: dict[str, str]
    response_metadata: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LlmReportMetadataResult:
    """Validated result from the final report metadata polishing phase."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    response_metadata: list[dict[str, object]] = field(default_factory=list)


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


@dataclass(frozen=True)
class LlmReportPolishPlan:
    """Execution plan for report polishing progress/reporting."""

    section_count: int
    worker_count: int


class LlmProcessor(Protocol):
    """Protocol for optional transcript/report enhancement backends."""

    @property
    def provider_name(self) -> str:
        """Return the configured provider identifier."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

    def polish_report_sections_with_progress(
        self, report: ReportDocument, *, progress_callback: Callable[[int], None] | None = None
    ) -> LlmSectionPolishResult:
        """Return polished section text with per-section progress updates."""

    def polish_report_metadata(
        self, report: ReportDocument, *, section_transcripts: dict[str, str]
    ) -> LlmReportMetadataResult:
        """Return polished summary, action items, and section titles."""

    def report_polish_plan(self, report: ReportDocument) -> LlmReportPolishPlan:
        """Return concurrency details for report polishing."""
