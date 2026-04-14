"""Shared contracts and response models for optional cloud LLM integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.models import ReportDocument

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LLMProcessingError(RuntimeError):
    """Raised when the LLM response cannot be validated or applied."""


@dataclass(frozen=True)
class LLMReportPolishResult:
    """Validated result from the report-polishing LLM stage."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    section_tldrs: dict[str, str]
    section_transcripts: dict[str, str]
    usage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


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
class _SectionPolishOutputs:
    transcripts: dict[str, str]
    tldrs: dict[str, str]


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


@dataclass(frozen=True)
class LLMReportPolishPlan:
    """Execution plan for report polishing progress/reporting."""

    section_count: int
    worker_count: int
    skipped_section_count: int = 0


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
