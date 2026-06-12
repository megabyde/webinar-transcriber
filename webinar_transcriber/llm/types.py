"""Error and result types shared across the optional LLM integration.

This module stays dependency-free so the CLI can import the error types on
installs without the `llm` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
