"""Utility helpers and wire schemas for optional cloud LLM integrations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, Field

from webinar_transcriber.llm.prompts import (
    REPORT_POLISH_TOTAL_CHAR_BUDGET,
    REPORT_SECTION_EXCERPT_LIMIT,
)
from webinar_transcriber.llm.types import LlmProcessingError
from webinar_transcriber.text_utils import split_llm_paragraph_blocks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from webinar_transcriber.models import ReportDocument


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


def build_report_polish_payload(report: ReportDocument) -> dict[str, object]:
    """Build the report-polish payload with a per-section excerpt budget.

    Returns:
        dict[str, object]: The structured payload sent to the metadata-polish stage.
    """
    section_count = max(len(report.sections), 1)
    per_section_budget = min(
        REPORT_SECTION_EXCERPT_LIMIT, max(200, REPORT_POLISH_TOTAL_CHAR_BUDGET // section_count)
    )

    return {
        "title": report.title,
        "detected_language": report.detected_language,
        "current_summary": report.summary,
        "current_action_items": report.action_items,
        "sections": [
            {
                "id": section.id,
                "title": section.title,
                "start_sec": section.start_sec,
                "end_sec": section.end_sec,
                "transcript_excerpt": truncate_text(section.transcript_text, per_section_budget),
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


def normalize_polished_section_text(*, original_text: str, polished_text: str) -> str:
    """Normalize polished section text.

    Returns:
        str: The accepted polished section text, or the original text when empty.
    """
    return _joined_llm_paragraphs(polished_text) or original_text


def normalize_polished_section_tldr(tldr: str) -> str:
    """Normalize a polished TL;DR into stable paragraph spacing.

    Returns:
        str: The normalized TL;DR text.
    """
    cleaned = _joined_llm_paragraphs(tldr)
    return re.sub(r"(?<=\S)[ \t]+(?=(?:[-*]|\d+[.)])\s+)", "\n", cleaned)


def _joined_llm_paragraphs(text: str) -> str:
    return "\n\n".join(split_llm_paragraph_blocks(text))


def extract_response_metadata(response: object) -> dict[str, object]:
    """Extract the provider's completion status for diagnostics.

    OpenAI Responses report ``status`` and ``incomplete_details``; Anthropic Messages report
    ``stop_reason``. A truncated response can still parse into the schema, so these fields are the
    only record that the output was cut short.
    """
    metadata: dict[str, object] = {}
    if status := _text_attr(response, "status"):
        metadata["status"] = status
    if incomplete_reason := _text_attr(getattr(response, "incomplete_details", None), "reason"):
        metadata["incomplete_reason"] = incomplete_reason
    if stop_reason := _text_attr(response, "stop_reason"):
        metadata["stop_reason"] = stop_reason
    return metadata


def _text_attr(value: object | None, name: str) -> str | None:
    text = getattr(value, name, None)
    return text if isinstance(text, str) and text else None
